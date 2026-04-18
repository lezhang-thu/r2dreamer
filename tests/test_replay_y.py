import unittest

import numpy as np

from replay_y import ReplayY


def make_step(index, ep_len, reward=None):
    return {
        "reward": np.float32(index if reward is None else reward),
        "action": np.asarray([index], dtype=np.float32),
        "is_first": index == 0,
        "is_last": index == ep_len - 1,
        "is_terminal": index == ep_len - 1,
    }


def add_episode(replay, ep_len, reward_offset=0, worker=0):
    for index in range(ep_len):
        step = make_step(index, ep_len, reward=index + reward_offset)
        replay.add(step, worker=worker)


def make_episode(ep_len, reward_offset=0):
    episode = {}
    for index in range(ep_len):
        step = make_step(index, ep_len, reward=index + reward_offset)
        for key, value in step.items():
            episode.setdefault(key, []).append(np.asarray(value))
    return {k: np.stack(v) for k, v in episode.items()}


class ReplayYTest(unittest.TestCase):

    def test_num_segments_counts_available_windows(self):
        replay = ReplayY(length=3, capacity=4, seed=0)
        self.assertEqual(replay.num_segments(), 0)

        add_episode(replay, ep_len=5, worker=0)
        self.assertEqual(replay.num_segments(), 3)

        add_episode(replay, ep_len=2, worker=1)
        self.assertEqual(replay.num_segments(), 4)

    def test_short_episode_is_zero_padded(self):
        replay = ReplayY(length=5, capacity=4, seed=0)
        add_episode(replay, ep_len=3)

        data = replay.sample(batch=1)

        np.testing.assert_array_equal(
            data["reward"][0],
            np.asarray([0.0, 1.0, 2.0, 0.0, 0.0], dtype=np.float32),
        )
        np.testing.assert_array_equal(
            data["t_mask"][0],
            np.asarray([True, True, True, False, False]),
        )
        np.testing.assert_array_equal(
            data["is_last"][0],
            np.asarray([False, False, True, False, False]),
        )
        self.assertTrue(data["is_first"][0, 0])
        self.assertFalse(data["is_first"][0, 1:].any())

    def test_long_episode_samples_all_valid_segments(self):
        replay = ReplayY(length=3, capacity=4, seed=0)
        add_episode(replay, ep_len=5)

        seen_segments = set()
        seen_last_flags = {}
        for _ in range(128):
            data = replay.sample(batch=1)
            rewards = tuple(int(x) for x in data["reward"][0])
            seen_segments.add(rewards)
            seen_last_flags[rewards] = tuple(
                bool(x) for x in data["is_last"][0])
            np.testing.assert_array_equal(
                data["t_mask"][0],
                np.asarray([True, True, True]),
            )
            self.assertTrue(data["is_first"][0, 0])
            self.assertFalse(data["is_first"][0, 1:].any())

        self.assertEqual(seen_segments, {(0, 1, 2), (1, 2, 3), (2, 3, 4)})
        self.assertEqual(seen_last_flags[(0, 1, 2)], (False, False, False))
        self.assertEqual(seen_last_flags[(1, 2, 3)], (False, False, False))
        self.assertEqual(seen_last_flags[(2, 3, 4)], (False, False, True))

    def test_longer_episodes_get_sampled_more_often_than_short_ones(self):
        replay = ReplayY(length=3, capacity=4, seed=0)
        add_episode(replay, ep_len=5, reward_offset=100, worker=0)
        add_episode(replay, ep_len=3, reward_offset=200, worker=1)

        long_count = 0
        short_count = 0
        for _ in range(4000):
            reward0 = int(replay.sample(batch=1)["reward"][0, 0])
            if reward0 < 200:
                long_count += 1
            else:
                short_count += 1

        long_ratio = long_count / (long_count + short_count)
        self.assertGreater(long_ratio, 0.70)
        self.assertLess(long_ratio, 0.80)

    def test_num_segments_includes_memory_episode(self):
        memory = make_episode(ep_len=5, reward_offset=100)
        replay = ReplayY(length=3, capacity=4, seed=0, memory=memory)

        self.assertEqual(replay.num_segments(), 3)

        add_episode(replay, ep_len=4, reward_offset=0, worker=0)

        self.assertEqual(replay.num_segments(), 5)

    def test_sample_mixes_configured_fraction_from_memory(self):
        memory = make_episode(ep_len=5, reward_offset=100)
        replay = ReplayY(length=3,
                         capacity=4,
                         seed=0,
                         memory=memory,
                         memory_sample_frac=0.5)
        add_episode(replay, ep_len=5, reward_offset=0, worker=0)

        for _ in range(64):
            data = replay.sample(batch=8)
            first_rewards = data["reward"][:, 0]
            memory_count = int((first_rewards >= 100).sum())
            replay_count = int((first_rewards < 100).sum())
            self.assertEqual(memory_count, 4)
            self.assertEqual(replay_count, 4)

    def test_sample_reports_memory_alignment_metadata(self):
        memory = make_episode(ep_len=5, reward_offset=100)
        replay = ReplayY(length=3,
                         capacity=4,
                         seed=0,
                         memory=memory,
                         memory_sample_frac=1.0)

        data = replay.sample(batch=2)

        self.assertTrue(data["is_memory"].all())
        for index_row, mask_row in zip(data["memory_index"], data["t_mask"]):
            valid = index_row[mask_row]
            self.assertTrue((valid >= 0).all())
            np.testing.assert_array_equal(
                valid, np.arange(valid[0], valid[0] + len(valid)))

    def test_sample_falls_back_to_memory_only_when_live_buffer_empty(self):
        memory = make_episode(ep_len=5, reward_offset=100)
        replay = ReplayY(length=3,
                         capacity=4,
                         seed=0,
                         memory=memory,
                         memory_sample_frac=0.5)

        data = replay.sample(batch=6)

        self.assertTrue((data["reward"][:, 0] >= 100).all())
        self.assertTrue(data["is_first"][:, 0].all())
        self.assertFalse(data["is_first"][:, 1:].any())


if __name__ == "__main__":
    unittest.main()
