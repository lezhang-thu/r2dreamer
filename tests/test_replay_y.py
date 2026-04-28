import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from replay_y import ReplayY


def _episode(length, reward_indices=()):
    reward = np.zeros(length, dtype=np.float32)
    reward[list(reward_indices)] = 1.0
    return {
        "obs": np.arange(length, dtype=np.int64),
        "reward": reward,
        "is_first": np.arange(length) == 0,
        "is_last": np.arange(length) == length - 1,
    }


def _add_episode(replay, episode, worker=0):
    for i in range(len(episode["reward"])):
        replay.add({k: v[i] for k, v in episode.items()}, worker=worker)


def test_tracks_stable_reward_count_from_recent_completed_episodes():
    memory = _episode(8, reward_indices=(1, 3, 5))
    replay = ReplayY(
        length=1,
        capacity=3,
        memory=memory,
        reward_stability_window=3,
    )

    assert replay.stable_reward_count == 0

    _add_episode(replay, _episode(7, reward_indices=(1, 4)))
    assert replay.stable_reward_count == 0

    _add_episode(replay, _episode(7, reward_indices=(1, 4)))
    assert replay.stable_reward_count == 0

    _add_episode(replay, _episode(7, reward_indices=(1, 4)))
    assert replay.stable_reward_count == 2

    _add_episode(replay, _episode(5, reward_indices=(2,)))
    assert replay.stable_reward_count == 1

    _add_episode(replay, _episode(7, reward_indices=(1, 4)))
    assert replay.stable_reward_count == 1


def test_memory_sampling_remains_uniform_without_interval_priority():
    memory = _episode(12, reward_indices=(2, 5, 8))
    replay = ReplayY(
        length=1,
        seed=0,
        memory=memory,
        memory_sample_frac=1.0,
        reward_interval_sample_frac=0.8,
        reward_stability_window=1,
    )
    _add_episode(replay, _episode(10, reward_indices=(3,)))

    assert replay.stable_reward_count == 1

    data = replay.sample(1000)
    starts = data["obs"][:, 0]

    # Memory is expert data, so it stays uniformly sampled even when live
    # replay has a stable reward frontier.
    in_interval = (2 <= starts) & (starts <= 8)
    assert 0.50 < np.mean(in_interval) < 0.70
    assert np.any(~in_interval)


def test_stability_count_uses_recent_window_not_historical_max():
    replay = ReplayY(
        length=1,
        seed=2,
        reward_stability_window=3,
        reward_interval_sample_frac=0.8,
    )
    _add_episode(replay, _episode(10, reward_indices=(3, 7)))
    _add_episode(replay, _episode(10, reward_indices=(3,)))
    _add_episode(replay, _episode(10, reward_indices=(4,)))

    assert replay.stable_reward_count == 1

    data = replay.sample(1000)
    starts = data["obs"][:, 0]
    frontier_tail = starts >= 3

    assert np.mean(frontier_tail) > 0.85


def test_replay_sampling_biases_observed_frontier_tail():
    replay = ReplayY(
        length=1,
        seed=1,
        reward_interval_sample_frac=0.8,
        reward_stability_window=1,
    )
    _add_episode(replay, _episode(10, reward_indices=(3,)))

    assert replay.stable_reward_count == 1

    data = replay.sample(1000)
    starts = data["obs"][:, 0]

    # The live episode reached one reward but has no second reward interval
    # yet, so replay emphasizes the observed tail after the frontier reward.
    assert np.mean(starts >= 3) > 0.85
    assert np.any(starts < 3)
