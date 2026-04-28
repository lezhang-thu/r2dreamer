import unittest

import torch
from torch import nn

from distributions import symlog
from dreamer import Dreamer, ExpertTag, NullMemorySlot


class _FakeRSSM:
    flat_stoch = 2
    feat_size = 4  # flat_stoch + deter


class _AttnDummy:
    """Mixin exposing the memory-attention helpers without full Dreamer init."""

    _get_memory_context = Dreamer._get_memory_context
    _build_memory_tokens = Dreamer._build_memory_tokens
    _get_null_memory_token = Dreamer._get_null_memory_token
    _zero_memory_readout = Dreamer._zero_memory_readout
    _apply_memory_attention = Dreamer._apply_memory_attention
    _read_memory = Dreamer._read_memory
    _get_rl_feat = Dreamer._get_rl_feat

    def __init__(self, memory, context, deter_dim=4, act_dim=2, stale=False):
        self.memory = memory
        self._memory_context = context
        self._memory_context_stale = stale
        self.act_dim = act_dim
        self._memory_scalar_dim = 3
        self.rssm = _FakeRSSM()
        self._frozen_rssm = _FakeRSSM()
        self.refresh_calls = 0

        flat_stoch = self.rssm.flat_stoch
        rtg_emb_dim = 16
        self.rtg_proj = nn.Linear(1, rtg_emb_dim)
        self._frozen_rtg_proj = self.rtg_proj
        mem_in_dim = deter_dim + flat_stoch + act_dim + rtg_emb_dim
        self.mem_proj = nn.Sequential(
            nn.Linear(mem_in_dim, deter_dim),
            nn.SiLU(),
            nn.Linear(deter_dim, deter_dim),
        )
        self._frozen_mem_proj = self.mem_proj
        self.expert_tag = ExpertTag(deter_dim)
        self._frozen_expert_tag = self.expert_tag
        self.null_memory = NullMemorySlot(deter_dim)
        self._frozen_null_memory = self.null_memory
        self.memory_attention = nn.MultiheadAttention(deter_dim,
                                                      num_heads=1,
                                                      dropout=0.0,
                                                      batch_first=True)
        self._frozen_memory_attention = self.memory_attention

    def refresh_memory_context(self):
        self.refresh_calls += 1
        return self._memory_context


class _EncoderRecorder:

    def __init__(self):
        self.last_image = None

    def __call__(self, data):
        self.last_image = data["image"].clone() if "image" in data else None
        T = data["reward"].shape[1]
        return torch.zeros(1, T, 5)


class _ObserveRecorder:
    """Returns deterministic (post_stoch, post_deter) of the requested shape."""

    def __init__(self, T, deter_dim, stoch_groups, stoch_classes):
        self.sample_args = []
        self._T = T
        self._D = deter_dim
        self._S = stoch_groups
        self._K = stoch_classes
        self._window_size = T
        self._offset = 0

    def observe(self, embed, action, is_first, sample=True):
        del embed, action, is_first
        self.sample_args.append(sample)
        deter = (torch.arange(self._T * self._D,
                              dtype=torch.float32).reshape(1, self._T, self._D))
        stoch = torch.zeros(1, self._T, self._S, self._K)
        stoch[..., 0] = 1.0  # one-hot on class 0
        return {}, {"deter": deter, "stoch": stoch}

    def initial(self, batch_size):
        del batch_size
        return {"h_prev": torch.zeros(1, self._D)}

    def observe_with_carry(self, embed, action, is_first, carry, sample=True):
        del action, is_first
        self.sample_args.append(sample)
        T = embed.shape[1]
        start = self._offset
        end = start + T
        deter = (torch.arange(start * self._D,
                              end * self._D,
                              dtype=torch.float32).reshape(1, T, self._D))
        stoch = torch.zeros(1, T, self._S, self._K)
        stoch[..., 0] = 1.0
        self._offset = end
        carry["h_prev"] = deter[:, -1]
        return {}, {"deter": deter, "stoch": stoch}, carry


class _FixedAttention:

    def __init__(self, weights):
        self.weights = weights

    def __call__(self, query, key, value, need_weights=True):
        del value, need_weights
        weights = self.weights.to(device=query.device, dtype=query.dtype)
        attended = torch.matmul(weights, key)
        return attended, weights


class _RefreshDummy:
    _memory_episode_tensordict = Dreamer._memory_episode_tensordict
    _compute_memory_return_to_go = Dreamer._compute_memory_return_to_go
    _compute_memory_progress = Dreamer._compute_memory_progress
    _compute_memory_waypoint = Dreamer._compute_memory_waypoint
    refresh_memory_context = Dreamer.refresh_memory_context
    preprocess = Dreamer.preprocess

    def __init__(self,
                 T=3,
                 deter_dim=2,
                 act_dim=2,
                 disc=0.5,
                 rewards=(1.0, 2.0, 4.0)):
        self.device = torch.device("cpu")
        self._mem_disc = disc
        self._memory_waypoint_offset = 1
        self.memory = {
            "reward": torch.tensor(rewards, dtype=torch.float32),
            "action": torch.eye(act_dim)[torch.zeros(T, dtype=torch.long)],
            "is_first": torch.tensor([True] + [False] * (T - 1)),
        }
        self._frozen_encoder = _EncoderRecorder()
        self.rssm = _FakeRSSM()
        self._frozen_rssm = _ObserveRecorder(T=T,
                                             deter_dim=deter_dim,
                                             stoch_groups=1,
                                             stoch_classes=2)
        self._memory_context = None
        self._memory_context_stale = True


class MemoryAttentionTest(unittest.TestCase):

    def test_apply_attention_returns_zero_without_memory(self):
        dummy = _AttnDummy(memory=None, context=None)
        deter = torch.randn(2, 4)
        out = dummy._apply_memory_attention(deter)
        torch.testing.assert_close(out, torch.zeros_like(deter))
        self.assertEqual(dummy.refresh_calls, 0)

    def test_get_rl_feat_concatenates_attended_zeros_without_memory(self):
        dummy = _AttnDummy(memory=None, context=None)
        stoch = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])  # (B, S=1, K=2)
        deter = torch.tensor([[5.0, 6.0, 7.0, 8.0], [1.0, 2.0, 3.0,
                                                     4.0]])  # (B, D=4)
        rl_feat = dummy._get_rl_feat(stoch, deter)
        expected = torch.cat([
            stoch.reshape(2, 2),
            deter,
            torch.zeros_like(deter),
            torch.zeros_like(deter),
            torch.zeros(2, 2),
            torch.zeros(2, 1),
            torch.zeros(2, 1),
            torch.zeros(2, 1),
        ],
                             dim=-1)
        torch.testing.assert_close(rl_feat, expected)

    def test_build_memory_tokens_uses_all_input_channels(self):
        T, D, A = 3, 4, 2
        deter = torch.arange(T * D, dtype=torch.float32).reshape(T, D)
        stoch_flat = torch.full((T, 2), 0.5)
        action = torch.eye(A)[torch.zeros(T, dtype=torch.long)]
        rtg = torch.tensor([[3.0], [4.0], [4.0]])
        ctx = {
            "deter": deter,
            "stoch_flat": stoch_flat,
            "action": action,
            "rtg": rtg
        }

        dummy = _AttnDummy(memory={"reward": torch.zeros(T)},
                           context=ctx,
                           deter_dim=D,
                           act_dim=A)
        # Make expert_tag a known constant so we can isolate its contribution.
        with torch.no_grad():
            dummy.expert_tag.tag.fill_(0.1)

        tokens = dummy._build_memory_tokens(ctx,
                                            dtype=torch.float32,
                                            device=torch.device("cpu"),
                                            frozen=False)

        # Reference computation: same path, no tag.
        rtg_emb = dummy.rtg_proj(rtg)
        feats = torch.cat([deter, stoch_flat, action, rtg_emb], dim=-1)
        expected = dummy.mem_proj(feats) + 0.1
        torch.testing.assert_close(tokens, expected)
        self.assertEqual(tokens.shape, (T, D))

    def test_apply_attention_shapes(self):
        T, D, A = 5, 4, 2
        raw_rtg = torch.randn(T, 1)
        ctx = {
            "deter": torch.randn(T, D),
            "stoch_flat": torch.randn(T, 2),
            "action": torch.eye(A)[torch.zeros(T, dtype=torch.long)],
            "reward": torch.randn(T, 1),
            "rtg": symlog(raw_rtg),
            "raw_rtg": raw_rtg,
            "progress": torch.linspace(0.0, 1.0, T).unsqueeze(-1),
            "waypoint": torch.randn(T, D),
        }
        dummy = _AttnDummy(memory={"reward": torch.zeros(T)},
                           context=ctx,
                           deter_dim=D,
                           act_dim=A)

        # Rank-2 query.
        q2 = torch.randn(3, D)
        out2 = dummy._apply_memory_attention(q2)
        self.assertEqual(out2.shape, (3, D))

        # Rank-3 query.
        q3 = torch.randn(3, 7, D)
        out3 = dummy._apply_memory_attention(q3)
        self.assertEqual(out3.shape, (3, 7, D))

    def test_read_memory_returns_zero_valued_fields_when_null_slot_selected(
            self):
        T, D, A = 2, 4, 2
        raw_rtg = torch.tensor([[3.0], [5.0]])
        ctx = {
            "deter": torch.randn(T, D),
            "stoch_flat": torch.randn(T, 2),
            "action": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "reward": torch.tensor([[7.0], [11.0]]),
            "rtg": symlog(raw_rtg),
            "raw_rtg": raw_rtg,
            "progress": torch.tensor([[0.25], [0.75]]),
            "waypoint": torch.randn(T, D),
        }
        dummy = _AttnDummy(memory={"reward": torch.zeros(T)},
                           context=ctx,
                           deter_dim=D,
                           act_dim=A)
        dummy.memory_attention = _FixedAttention(
            torch.tensor([[[0.0, 0.0, 1.0]]], dtype=torch.float32))
        dummy._frozen_memory_attention = dummy.memory_attention

        q = torch.randn(1, D)
        readout = dummy._read_memory(q)

        torch.testing.assert_close(readout["reward"], torch.zeros(1, 1))
        torch.testing.assert_close(readout["rtg"], torch.zeros(1, 1))
        torch.testing.assert_close(readout["raw_rtg"], torch.zeros(1, 1))
        torch.testing.assert_close(readout["progress"], torch.zeros(1, 1))
        torch.testing.assert_close(readout["action"], torch.zeros(1, A))
        torch.testing.assert_close(readout["waypoint"], torch.zeros(1, D))
        self.assertEqual(readout["weights"].shape[-1], T + 1)
        torch.testing.assert_close(readout["weights"][0, -1], torch.tensor(1.0))


class RefreshMemoryContextTest(unittest.TestCase):

    def test_refresh_uses_deterministic_observe_and_stores_features(self):
        dummy = _RefreshDummy(T=3,
                              deter_dim=2,
                              act_dim=2,
                              disc=0.5,
                              rewards=(1.0, 2.0, 4.0))
        ctx = dummy.refresh_memory_context()

        self.assertEqual(dummy._frozen_rssm.sample_args, [False])
        self.assertFalse(dummy._memory_context_stale)
        for key in ("deter", "stoch_flat", "action", "reward", "rtg", "raw_rtg",
                    "progress", "waypoint"):
            self.assertIn(key, ctx)
        self.assertEqual(ctx["deter"].shape, (3, 2))
        self.assertEqual(ctx["stoch_flat"].shape, (3, 2))
        self.assertEqual(ctx["action"].shape, (3, 2))
        self.assertEqual(ctx["reward"].shape, (3, 1))
        self.assertEqual(ctx["rtg"].shape, (3, 1))
        self.assertEqual(ctx["raw_rtg"].shape, (3, 1))
        self.assertEqual(ctx["progress"].shape, (3, 1))
        self.assertEqual(ctx["waypoint"].shape, (3, 2))

    def test_refresh_builds_memory_context_in_chunks(self):
        dummy = _RefreshDummy(T=5,
                              deter_dim=2,
                              act_dim=2,
                              disc=0.5,
                              rewards=(1.0, 2.0, 3.0, 4.0, 5.0))
        dummy._frozen_rssm._window_size = 2

        ctx = dummy.refresh_memory_context()

        self.assertEqual(dummy._frozen_rssm.sample_args, [False, False, False])
        torch.testing.assert_close(ctx["deter"][:, 0],
                                   torch.tensor([0.0, 2.0, 4.0, 6.0, 8.0]))
        self.assertEqual(ctx["stoch_flat"].shape, (5, 2))
        self.assertEqual(ctx["action"].shape, (5, 2))

    def test_refresh_computes_discounted_return_to_go(self):
        # rewards = [1, 2, 4], disc = 0.5
        # G[2] = 4
        # G[1] = 2 + 0.5*4 = 4
        # G[0] = 1 + 0.5*4 = 3
        dummy = _RefreshDummy(T=3,
                              deter_dim=2,
                              act_dim=2,
                              disc=0.5,
                              rewards=(1.0, 2.0, 4.0))
        ctx = dummy.refresh_memory_context()
        expected_G = torch.tensor([[3.0], [4.0], [4.0]])
        torch.testing.assert_close(ctx["rtg"], symlog(expected_G))
        torch.testing.assert_close(ctx["raw_rtg"], expected_G)

    def test_refresh_returns_none_when_memory_absent(self):

        class _NoMemory:
            refresh_memory_context = Dreamer.refresh_memory_context

            def __init__(self):
                self.memory = None
                self._memory_context = "stale"
                self._memory_context_stale = True

        nm = _NoMemory()
        self.assertIsNone(nm.refresh_memory_context())
        self.assertIsNone(nm._memory_context)
        self.assertFalse(nm._memory_context_stale)


if __name__ == "__main__":
    unittest.main()
