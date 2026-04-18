import types
import unittest
from unittest import mock

import numpy as np

import episode_memory
from episode_memory import build_atari_memory_episode
from replay_y import ReplayY


def make_obs(fill, is_first=False, is_last=False, is_terminal=False):
    return {
        "image": np.full((2, 2, 1), fill, dtype=np.uint8),
        "is_first": np.asarray(is_first),
        "is_last": np.asarray(is_last),
        "is_terminal": np.asarray(is_terminal),
    }


def make_step(obs, reward, action):
    step = {k: np.asarray(v) for k, v in obs.items()}
    step["reward"] = np.asarray(np.float32(reward))
    step["action"] = np.asarray(action, dtype=np.float32)
    return step


class FakeEnv:

    def __init__(self, reset_obs, transitions, action_dim=4):
        self.action_space = types.SimpleNamespace(n=action_dim)
        self._reset_obs = reset_obs
        self._transitions = list(transitions)
        self.reset_seed = None
        self.step_calls = []
        self.closed = False

    def reset(self, seed=None):
        self.reset_seed = seed
        return self._reset_obs

    def step(self, action):
        if isinstance(action, np.ndarray):
            action = int(np.argmax(action))
        self.step_calls.append(int(action))
        transition = self._transitions.pop(0)
        return (
            transition["obs"],
            transition["reward"],
            transition["done"],
            {},
        )

    def close(self):
        self.closed = True


class EpisodeMemoryTest(unittest.TestCase):

    def test_make_env_forces_nonsticky_replay_and_constructor_seed_none(self):
        config = types.SimpleNamespace(
            action_repeat=4,
            size=[64, 64],
            gray=False,
            noops=7,
            lives="unused",
            sticky=True,
            actions="needed",
            time_limit=1000,
            pooling=2,
            aggregate="max",
            resize="pillow",
            autostart=False,
            clip_reward=False,
        )
        fake_env = object()
        fake_ctor = mock.Mock(return_value=fake_env)
        fake_ctor.LOCK = object()

        with mock.patch.object(episode_memory.atari, "Atari", fake_ctor):
            with mock.patch.object(episode_memory.wrappers,
                                   "OneHotAction",
                                   side_effect=lambda env: env):
                with mock.patch.object(episode_memory.wrappers,
                                       "TimeLimit",
                                       side_effect=lambda env, _: env):
                    with mock.patch.object(episode_memory.wrappers,
                                           "Dtype",
                                           side_effect=lambda env: env):
                        env = episode_memory.make_atari_memory_env(
                            config, "montezuma_revenge")

        self.assertIs(env, fake_env)
        fake_ctor.assert_called_once_with(
            name="montezuma_revenge",
            action_repeat=4,
            size=(64, 64),
            gray=False,
            noops=7,
            lives="unused",
            sticky=False,
            actions="needed",
            length=1000,
            pooling=2,
            aggregate="max",
            resize="pillow",
            autostart=False,
            clip_reward=False,
            seed=None,
        )

    def test_build_matches_replay_episode_layout(self):
        reset_obs = make_obs(fill=1, is_first=True)
        env = FakeEnv(
            reset_obs=reset_obs,
            transitions=[
                {
                    "obs": make_obs(fill=5),
                    "reward": 3.0,
                    "done": False,
                },
                {
                    "obs": make_obs(fill=9, is_last=True, is_terminal=True),
                    "reward": 7.0,
                    "done": True,
                },
            ],
        )

        memory = build_atari_memory_episode(
            actions=[2, 1],
            expected_score=10.0,
            env_ctor=lambda: env,
        )

        replay = ReplayY(length=3, capacity=1, seed=0)
        replay.add(make_step(reset_obs, 0.0, [0, 0, 1, 0]), worker=0)
        replay.add(make_step(make_obs(fill=5), 3.0, [0, 1, 0, 0]), worker=0)
        replay.add(
            make_step(
                make_obs(fill=9, is_last=True, is_terminal=True),
                7.0,
                [0, 0, 0, 0],
            ),
            worker=0,
        )
        expected = replay.eps[0]

        self.assertEqual(env.reset_seed, 0)
        self.assertEqual(env.step_calls, [2, 1])
        self.assertTrue(env.closed)
        self.assertEqual(set(memory), set(expected))
        for key in expected:
            np.testing.assert_array_equal(memory[key], expected[key])

    def test_build_rejects_score_mismatch(self):
        env = FakeEnv(
            reset_obs=make_obs(fill=1, is_first=True),
            transitions=[
                {
                    "obs": make_obs(fill=2, is_last=True, is_terminal=True),
                    "reward": 1.0,
                    "done": True,
                },
            ],
        )

        with self.assertRaisesRegex(ValueError, "Recovered score"):
            build_atari_memory_episode(
                actions=[0],
                expected_score=2.0,
                env_ctor=lambda: env,
            )

    def test_build_rejects_incomplete_episode(self):
        env = FakeEnv(
            reset_obs=make_obs(fill=1, is_first=True),
            transitions=[
                {
                    "obs": make_obs(fill=2),
                    "reward": 1.0,
                    "done": False,
                },
            ],
        )

        with self.assertRaisesRegex(ValueError,
                                    "ended before the episode terminated"):
            build_atari_memory_episode(
                actions=[0],
                expected_score=1.0,
                env_ctor=lambda: env,
            )


if __name__ == "__main__":
    unittest.main()
