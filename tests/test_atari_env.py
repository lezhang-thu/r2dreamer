import unittest
from unittest import mock

import gymnasium as gym
import numpy as np

import envs.atari as atari


class DummyLock:

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class FakeALE:

    def __init__(self, height=4, width=4, lives=3):
        self._height = height
        self._width = width
        self._lives = lives
        self._screen = np.zeros((height, width, 3), np.uint8)

    def getLegalActionSet(self):
        return list(range(18))

    def getMinimalActionSet(self):
        return [0, 1, 3, 4, 11, 12]

    def getScreenDims(self):
        return self._height, self._width

    def getScreenRGB(self, array):
        array[...] = self._screen

    def lives(self):
        return self._lives


class FakeUnwrapped:

    def __init__(self, ale, action_meanings):
        self.ale = ale
        self._action_meanings = list(action_meanings)
        self.seed_calls = []
        self.load_game_calls = 0

    def seed_game(self, seed):
        self.seed_calls.append(seed)

    def load_game(self):
        self.load_game_calls += 1

    def get_action_meanings(self):
        return list(self._action_meanings)


class FakeEnv:

    def __init__(self,
                 action_meanings,
                 transitions,
                 reset_fill=7,
                 reset_lives=3):
        self.ale = FakeALE(lives=reset_lives)
        self.unwrapped = FakeUnwrapped(self.ale, action_meanings)
        self.action_space = gym.spaces.Discrete(len(action_meanings))
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(4, 4, 3),
            dtype=np.uint8,
        )
        self._transitions = list(transitions)
        self._reset_fill = reset_fill
        self._reset_lives = reset_lives
        self.reset_calls = 0
        self.step_calls = []
        self.closed = False

    def reset(self, seed=None):
        self.reset_calls += 1
        self.ale._lives = self._reset_lives
        self.ale._screen.fill(self._reset_fill)
        return self.ale._screen.copy(), {}

    def step(self, action):
        self.step_calls.append(int(action))
        transition = self._transitions.pop(0)
        self.ale._screen.fill(transition.get("fill", 0))
        self.ale._lives = transition.get("lives", self.ale._lives)
        return (
            self.ale._screen.copy(),
            transition.get("reward", 0.0),
            transition.get("terminated", False),
            transition.get("truncated", False),
            {},
        )

    def close(self):
        self.closed = True


class AtariEnvTest(unittest.TestCase):

    def setUp(self):
        atari.Atari.LOCK = DummyLock()

    def test_gym_make_backend_preserves_reset_and_step_api(self):
        fake_env = FakeEnv(
            action_meanings=[
                "NOOP",
                "FIRE",
                "RIGHT",
                "LEFT",
                "RIGHTFIRE",
                "LEFTFIRE",
            ],
            transitions=[
                {
                    "fill": 9,
                    "reward": 0.0
                },  # autostart FIRE during reset
                {
                    "fill": 10,
                    "reward": 1.5
                },
                {
                    "fill": 20,
                    "reward": 2.5
                },
            ],
        )

        with mock.patch.object(atari.gym, "make",
                               return_value=fake_env) as make_env:
            env = atari.Atari(
                name="montezuma_revenge",
                action_repeat=2,
                size=(4, 4),
                gray=False,
                noops=0,
                sticky=False,
                actions="needed",
                pooling=2,
                autostart=True,
                seed=123,
            )

        make_env.assert_called_once_with(
            "ALE/MontezumaRevenge-v5",
            obs_type="rgb",
            frameskip=1,
            mode=None,
            difficulty=None,
            repeat_action_probability=0.0,
            full_action_space=False,
            render_mode=None,
        )
        self.assertEqual(fake_env.unwrapped.seed_calls, [123])
        self.assertEqual(fake_env.unwrapped.load_game_calls, 1)
        self.assertEqual(env.action_space.n, 6)

        obs = env.reset()
        self.assertTrue(obs["is_first"])
        self.assertFalse(obs["is_last"])
        self.assertFalse(obs["is_terminal"])
        np.testing.assert_array_equal(obs["image"],
                                      np.full((4, 4, 3), 9, np.uint8))

        step_obs, reward, done, info = env.step(2)
        self.assertEqual(fake_env.step_calls, [1, 2, 2])
        self.assertEqual(reward, 4.0)
        self.assertFalse(done)
        self.assertEqual(info, {})
        self.assertFalse(step_obs["is_terminal"])
        np.testing.assert_array_equal(step_obs["image"],
                                      np.full((4, 4, 3), 20, np.uint8))

        env.close()
        self.assertTrue(fake_env.closed)

    def test_life_loss_uses_existing_terminal_flags(self):
        fake_env = FakeEnv(
            action_meanings=[
                "NOOP",
                "FIRE",
                "RIGHT",
                "LEFT",
                "RIGHTFIRE",
                "LEFTFIRE",
            ],
            transitions=[{
                "fill": 5,
                "reward": 1.0,
                "lives": 2
            }],
            reset_lives=3,
        )

        with mock.patch.object(atari.gym, "make", return_value=fake_env):
            env = atari.Atari(
                name="pong",
                action_repeat=1,
                size=(4, 4),
                gray=False,
                lives="reset",
                sticky=False,
                actions="needed",
                pooling=2,
            )

        env.reset()
        obs, reward, done, info = env.step(2)
        self.assertEqual(fake_env.step_calls, [2])
        self.assertEqual(reward, 1.0)
        self.assertTrue(done)
        self.assertEqual(info, {})
        self.assertTrue(obs["is_last"])
        self.assertTrue(obs["is_terminal"])

    def test_gym_name_conversion_handles_repo_aliases(self):
        self.assertEqual(atari.Atari._gym_env_name("montezuma_revenge"),
                         "MontezumaRevenge")
        self.assertEqual(atari.Atari._gym_env_name("james_bond"), "Jamesbond")


if __name__ == "__main__":
    unittest.main()
