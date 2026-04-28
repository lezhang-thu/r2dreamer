import unittest

import torch
from torch import nn

from dreamer import Dreamer


class _FakeRSSM:
    flat_stoch = 2
    feat_size = 4  # flat_stoch + deter


class _AttnDummy:
    """Mixin exposing the memory-attention helpers without full Dreamer init."""

    _get_memory_context = Dreamer._get_memory_context
    _refresh_memory_context_if_stale = Dreamer._refresh_memory_context_if_stale
    _zero_memory_readout = Dreamer._zero_memory_readout
    _read_memory = Dreamer._read_memory

    def __init__(self, memory, context, deter_dim=4, act_dim=2, stale=False):
        self.memory = memory
        self._memory_context = context
        self._memory_context_stale = stale
        self.act_dim = act_dim
        self.rssm = _FakeRSSM()
        self._frozen_rssm = _FakeRSSM()
        self.refresh_calls = 0

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

    def test_get_memory_context_refreshes_when_require_fresh_and_stale(self):
        context = {"reward": torch.zeros(1, 1)}
        dummy = _AttnDummy(memory={"reward": torch.zeros(1)},
                           context=context,
                           stale=True)

        out = dummy._get_memory_context(require_fresh=True)

        self.assertIs(out, context)
        self.assertEqual(dummy.refresh_calls, 1)

    def test_refresh_memory_context_if_stale_only_refreshes_when_needed(self):
        context = {"reward": torch.zeros(1, 1)}
        dummy = _AttnDummy(memory={"reward": torch.zeros(1)},
                           context=context,
                           stale=True)

        dummy._refresh_memory_context_if_stale()
        self.assertEqual(dummy.refresh_calls, 1)

        dummy._memory_context_stale = False
        dummy._refresh_memory_context_if_stale()
        self.assertEqual(dummy.refresh_calls, 1)

    def test_read_memory_returns_attention_weighted_fields(self):
        T, D, A = 2, 4, 2
        raw_rtg = torch.tensor([[3.0], [5.0]])
        ctx = {
            "deter": torch.randn(T, D),
            "raw_rtg": raw_rtg,
            "progress": torch.tensor([[0.25], [0.75]]),
        }
        dummy = _AttnDummy(memory={"reward": torch.zeros(T)},
                           context=ctx,
                           deter_dim=D,
                           act_dim=A)
        dummy.memory_attention = _FixedAttention(
            torch.tensor([[[0.25, 0.75]]], dtype=torch.float32))
        dummy._frozen_memory_attention = dummy.memory_attention

        q = torch.randn(1, D)
        readout = dummy._read_memory(q)

        torch.testing.assert_close(readout["raw_rtg"],
                                   torch.tensor([[4.5]]),
                                   atol=1e-6,
                                   rtol=0.0)
        torch.testing.assert_close(readout["progress"],
                                   torch.tensor([[0.625]]),
                                   atol=1e-6,
                                   rtol=0.0)
        self.assertEqual(readout["weights"].shape[-1], T)


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
        for key in ("deter", "raw_rtg", "progress"):
            self.assertIn(key, ctx)
        self.assertEqual(ctx["deter"].shape, (3, 2))
        self.assertEqual(ctx["raw_rtg"].shape, (3, 1))
        self.assertEqual(ctx["progress"].shape, (3, 1))

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
