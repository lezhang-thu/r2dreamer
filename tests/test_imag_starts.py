import unittest
from unittest import mock

import torch

from dreamer import Dreamer
from rssm import TransformerRSSM


class _DummyRSSM:
    build_imag_starts = TransformerRSSM.build_imag_starts

    def __init__(self, window_size=2, deter=1):
        self._window_size = window_size
        self._deter = deter


class _DummyDreamer:
    _sample_valid_imag_starts = Dreamer._sample_valid_imag_starts
    _sample_transformer_imag_starts = Dreamer._sample_transformer_imag_starts
    _prepare_transformer_imag_start = Dreamer._prepare_transformer_imag_start

    def __init__(self, imag_last, frozen_rssm):
        self.imag_last = imag_last
        self._frozen_rssm = frozen_rssm


class ImagStartTest(unittest.TestCase):

    def test_build_imag_starts_uses_episode_major_order(self):
        rssm = _DummyRSSM(window_size=2, deter=1)
        stoch_seq = torch.tensor([[[[0.0]], [[1.0]], [[2.0]], [[3.0]], [[4.0]]],
                                  [[[10.0]], [[11.0]], [[12.0]], [[13.0]],
                                   [[14.0]]]])
        deter_seq = torch.tensor([[[0.0], [1.0], [2.0], [3.0], [4.0]],
                                  [[10.0], [11.0], [12.0], [13.0], [14.0]]])
        kv_k = torch.tensor([[[[100.0], [101.0], [102.0], [103.0], [104.0],
                               [105.0], [106.0]]],
                             [[[200.0], [201.0], [202.0], [203.0], [204.0],
                               [205.0], [206.0]]]])
        kv_v = kv_k + 1000.0
        starts = torch.tensor([[0, 2], [1, 3]])

        start_stoch, start_deter, carry = rssm.build_imag_starts(
            stoch_seq, deter_seq, kv_k, kv_v, starts)

        self.assertEqual(start_stoch.shape, (4, 1, 1))
        self.assertEqual(start_deter.shape, (4, 1))
        torch.testing.assert_close(
            start_stoch[:, 0, 0],
            torch.tensor([0.0, 2.0, 11.0, 13.0]),
        )
        torch.testing.assert_close(
            start_deter[:, 0],
            torch.tensor([0.0, 2.0, 11.0, 13.0]),
        )
        torch.testing.assert_close(
            carry["pos"], torch.tensor([0, 2, 1, 3], dtype=torch.int32))
        torch.testing.assert_close(
            carry["kv_cache"][:, 0, 0, :, 0],
            torch.tensor([[100.0, 101.0], [102.0, 103.0], [201.0, 202.0],
                          [203.0, 204.0]]),
        )
        torch.testing.assert_close(
            carry["kv_cache"][:, 0, 1, :, 0],
            torch.tensor([[1100.0, 1101.0], [1102.0, 1103.0], [1201.0, 1202.0],
                          [1203.0, 1204.0]]),
        )

    def test_build_imag_starts_handles_unsorted_and_duplicate_indices(self):
        rssm = _DummyRSSM(window_size=2, deter=1)
        stoch_seq = torch.tensor([[[[0.0]], [[1.0]], [[2.0]], [[3.0]],
                                   [[4.0]]]])
        deter_seq = torch.tensor([[[0.0], [1.0], [2.0], [3.0], [4.0]]])
        kv_k = torch.tensor([[[[0.0], [0.0], [100.0], [101.0], [102.0], [103.0],
                               [104.0]]]])
        kv_v = kv_k + 1000.0
        starts = torch.tensor([[3, 1, 1, 0]])

        start_stoch, start_deter, carry = rssm.build_imag_starts(
            stoch_seq, deter_seq, kv_k, kv_v, starts)

        torch.testing.assert_close(
            start_stoch[:, 0, 0],
            torch.tensor([3.0, 1.0, 1.0, 0.0]),
        )
        torch.testing.assert_close(start_deter[:, 0],
                                   torch.tensor([3.0, 1.0, 1.0, 0.0]))
        torch.testing.assert_close(
            carry["pos"], torch.tensor([3, 1, 1, 0], dtype=torch.int32))
        torch.testing.assert_close(
            carry["kv_cache"][:, 0, 0, :, 0],
            torch.tensor([[101.0, 102.0], [0.0, 100.0], [0.0, 100.0],
                          [0.0, 0.0]]),
        )
        torch.testing.assert_close(
            carry["kv_cache"][:, 0, 1, :, 0],
            torch.tensor([[1101.0, 1102.0], [1000.0, 1100.0], [1000.0, 1100.0],
                          [1000.0, 1000.0]]),
        )

    def test_sample_valid_imag_starts_returns_contiguous_offsets_when_possible(
            self):
        rssm = _DummyRSSM(window_size=2, deter=1)
        agent = _DummyDreamer(imag_last=4, frozen_rssm=rssm)
        valid_lens = torch.tensor([5, 4], dtype=torch.int64)

        side_effects = [
            torch.tensor(1, device=valid_lens.device),
            torch.tensor(0, device=valid_lens.device),
        ]
        with mock.patch("torch.randint", side_effect=side_effects):
            starts = agent._sample_valid_imag_starts(valid_lens,
                                                     K=3,
                                                     device=valid_lens.device)

        torch.testing.assert_close(starts, torch.tensor([[1, 2, 3], [0, 1, 2]]))

    def test_sample_valid_imag_starts_returns_independent_indices_when_short(
            self):
        rssm = _DummyRSSM(window_size=2, deter=1)
        agent = _DummyDreamer(imag_last=4, frozen_rssm=rssm)
        valid_lens = torch.tensor([2, 3], dtype=torch.int64)

        side_effects = [
            torch.tensor([1, 0, 1, 1], device=valid_lens.device),
            torch.tensor([2, 0, 1, 2], device=valid_lens.device),
        ]
        with mock.patch("torch.randint", side_effect=side_effects):
            starts = agent._sample_valid_imag_starts(valid_lens,
                                                     K=4,
                                                     device=valid_lens.device)

        torch.testing.assert_close(
            starts,
            torch.tensor([[1, 0, 1, 1], [2, 0, 1, 2]], dtype=torch.int64),
        )

    def test_prepare_transformer_imag_start_accepts_arbitrary_valid_starts(
            self):
        rssm = _DummyRSSM(window_size=2, deter=1)
        agent = _DummyDreamer(imag_last=4, frozen_rssm=rssm)
        post_stoch = torch.tensor([[[[0.0]], [[1.0]], [[2.0]], [[3.0]],
                                    [[4.0]]],
                                   [[[10.0]], [[11.0]], [[12.0]], [[13.0]],
                                    [[14.0]]]])
        post_deter = torch.tensor([[[0.0], [1.0], [2.0], [3.0], [4.0]],
                                   [[10.0], [11.0], [12.0], [13.0], [14.0]]])
        kv_k = torch.tensor([[[[100.0], [101.0], [102.0], [103.0], [104.0],
                               [105.0], [106.0]]],
                             [[[200.0], [201.0], [202.0], [203.0], [204.0],
                               [205.0], [206.0]]]])
        kv_v = kv_k + 1000.0
        starts = torch.tensor([[3, 1], [0, 4]])

        start_stoch, start_deter, imag_carry, imag_mask = agent._prepare_transformer_imag_start(
            post_stoch, post_deter, kv_k, kv_v, starts)

        torch.testing.assert_close(
            start_stoch[:, 0, 0],
            torch.tensor([3.0, 1.0, 10.0, 14.0]),
        )
        torch.testing.assert_close(
            start_deter[:, 0],
            torch.tensor([3.0, 1.0, 10.0, 14.0]),
        )
        torch.testing.assert_close(
            imag_carry["pos"],
            torch.tensor([3, 1, 0, 4], dtype=torch.int32),
        )
        torch.testing.assert_close(imag_mask, torch.ones(4, 1, 1))

    def test_sample_transformer_imag_starts_uses_imag_last_cap(self):
        rssm = _DummyRSSM(window_size=2, deter=1)
        agent = _DummyDreamer(imag_last=3, frozen_rssm=rssm)
        valid_lens = torch.tensor([5, 5], dtype=torch.int64)

        side_effects = [
            torch.tensor(1, device=valid_lens.device),
            torch.tensor(2, device=valid_lens.device),
        ]
        with mock.patch("torch.randint", side_effect=side_effects):
            starts = agent._sample_transformer_imag_starts(valid_lens, T=5)

        torch.testing.assert_close(starts, torch.tensor([[1, 2, 3], [2, 3, 4]]))


if __name__ == "__main__":
    unittest.main()
