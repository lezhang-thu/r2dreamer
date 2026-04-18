import unittest

import torch
from tensordict import TensorDict

from dreamer import Dreamer


class _DummyScaler:

    def scale(self, value):
        return value


class _ChunkDummy:
    _iter_start_chunks = Dreamer._iter_start_chunks


class _CalGradDummy:
    _iter_batch_chunks = Dreamer._iter_batch_chunks
    _iter_start_chunks = Dreamer._iter_start_chunks
    _cal_grad = Dreamer._cal_grad

    def __init__(self):
        self.device = torch.device("cpu")
        self.wm_accum_steps = 1
        self.ac_accum_steps = 4
        self._loss_scales = {
            "dyn": 1.0,
            "policy": 1.0,
            "value": 1.0,
        }
        self._scaler = _DummyScaler()
        self.prepare_chunk_sizes = []

    def _world_model_forward(self, data):
        seed = torch.ones((), requires_grad=True)
        losses = {"dyn": seed * 0.0}
        metrics = {}
        imag_source = {
            "post_stoch": torch.zeros(2, 5, 1, 1),
            "post_deter": torch.zeros(2, 5, 1),
            "kv_k": torch.zeros(2, 1, 7, 1),
            "kv_v": torch.zeros(2, 1, 7, 1),
            "valid_lens": torch.tensor([5, 5], dtype=torch.int64),
            "T": 5,
        }
        return losses, metrics, imag_source

    def _sample_transformer_imag_starts(self, valid_lens, T):
        del valid_lens, T
        return torch.arange(16, dtype=torch.int64).reshape(2, 8)

    def _prepare_transformer_imag_start(self, post_stoch, post_deter, kv_k,
                                        kv_v, starts):
        del post_stoch, post_deter, kv_k, kv_v
        self.prepare_chunk_sizes.append(int(starts.shape[1]))
        B, K = starts.shape
        start_stoch = torch.zeros(B * K, 1, 1)
        start_deter = torch.zeros(B * K, 1)
        imag_carry = {
            "kv_cache": torch.zeros(B * K, 1, 2, 1, 1),
            "pos": torch.zeros(B * K, dtype=torch.int32),
            "h_prev": torch.zeros(B * K, 1),
        }
        imag_mask = torch.ones(B * K, 1, 1)
        return start_stoch, start_deter, imag_carry, imag_mask

    def _actor_critic_forward(self, start_stoch, start_deter, imag_carry,
                              imag_mask):
        del start_stoch, start_deter, imag_carry, imag_mask
        policy_seed = torch.ones((), requires_grad=True)
        value_seed = torch.ones((), requires_grad=True)
        losses = {
            "policy": policy_seed * 0.0,
            "value": value_seed * 0.0,
        }
        metrics = {}
        return losses, metrics


class ImagStartChunkingTest(unittest.TestCase):

    def test_iter_start_chunks_splits_sampled_start_indices(self):
        agent = _ChunkDummy()
        starts = torch.arange(24, dtype=torch.int64).reshape(3, 8)

        chunks = list(agent._iter_start_chunks(starts, accum_steps=4))

        self.assertEqual([chunk.shape for chunk, _ in chunks], [(3, 2), (3, 2),
                                                                (3, 2), (3, 2)])
        self.assertEqual([weight for _, weight in chunks],
                         [0.25, 0.25, 0.25, 0.25])

    def test_cal_grad_builds_imag_starts_per_chunk(self):
        agent = _CalGradDummy()
        data = TensorDict(
            {
                "reward": torch.zeros(2, 1),
                "t_mask": torch.ones(2, 1),
            },
            batch_size=(2, 1),
        )

        agent._cal_grad(data)

        self.assertEqual(agent.prepare_chunk_sizes, [2, 2, 2, 2])


if __name__ == "__main__":
    unittest.main()
