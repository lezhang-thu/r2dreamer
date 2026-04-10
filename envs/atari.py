import collections
import os

import ale_py
import gymnasium as gym
import numpy as np
from PIL import Image

gym.register_envs(ale_py)


class Atari(gym.Env):
    LOCK = None
    metadata = {}
    ACTION_MEANING = (
        "NOOP",
        "FIRE",
        "UP",
        "RIGHT",
        "LEFT",
        "DOWN",
        "UPRIGHT",
        "UPLEFT",
        "DOWNRIGHT",
        "DOWNLEFT",
        "UPFIRE",
        "RIGHTFIRE",
        "LEFTFIRE",
        "DOWNFIRE",
        "UPRIGHTFIRE",
        "UPLEFTFIRE",
        "DOWNRIGHTFIRE",
        "DOWNLEFTFIRE",
    )
    WEIGHTS = np.array([0.299, 0.587, 1 - (0.299 + 0.587)])

    @staticmethod
    def _gym_env_name(name):
        if name == "james_bond":
            name = "jamesbond"
        return "".join(part.capitalize() for part in name.split("_"))

    def __init__(
        self,
        name,
        action_repeat=4,
        size=(84, 84),
        gray=True,
        noops=0,
        lives="unused",
        sticky=True,
        actions="all",
        length=108000,
        pooling=2,
        aggregate="max",
        resize="pillow",
        autostart=False,
        clip_reward=False,
        seed=None,
    ):
        assert size[0] == size[1]
        assert lives in ("unused", "discount", "reset"), lives
        assert actions in ("all", "needed"), actions
        assert resize in ("opencv", "pillow"), resize
        assert aggregate in ("max", "mean"), aggregate
        assert pooling >= 1, pooling
        assert action_repeat >= 1, action_repeat

        if self.LOCK is None:
            import multiprocessing as mp

            mp = mp.get_context("spawn")
            self.LOCK = mp.Lock()

        self._resize_fn = resize
        if self._resize_fn == "opencv":
            import cv2

            self._cv2 = cv2

        env_name = self._gym_env_name(name)

        self._repeat = action_repeat
        self._size = size
        self._gray = gray
        self._noops = noops
        self._lives = lives
        self._length = length
        self._pooling = pooling
        self._aggregate = aggregate
        self._autostart = autostart
        self._clip_reward = clip_reward
        self._rng = np.random.default_rng(seed)

        roms_dir = os.environ.get("ALE_ROM_PATH")
        if roms_dir and "ALE_ROMS_DIR" not in os.environ:
            os.environ["ALE_ROMS_DIR"] = roms_dir

        with self.LOCK:
            self._env = gym.make(
                f"ALE/{env_name}-v5",
                obs_type="rgb",
                frameskip=1,
                mode=None,
                difficulty=None,
                repeat_action_probability=0.25 if sticky else 0.0,
                full_action_space=(actions == "all"),
                render_mode=None,
            )
            if seed is not None:
                self._env.unwrapped.seed_game(int(seed))
                self._env.unwrapped.load_game()

        self.ale = self._env.unwrapped.ale
        self.actionset = {
            "all": self.ale.getLegalActionSet,
            "needed": self.ale.getMinimalActionSet,
        }[actions]()
        self._action_meanings = tuple(self._env.unwrapped.get_action_meanings())
        assert self._action_meanings[0] == "NOOP"

        H, W = self.ale.getScreenDims()
        self._buffers = collections.deque(
            [np.zeros((H, W, 3), np.uint8) for _ in range(self._pooling)],
            maxlen=self._pooling)

        self._last_lives = None
        self._done = True
        self._step = 0
        self.reward_range = [-np.inf, np.inf]

    @property
    def observation_space(self):
        img_shape = self._size + ((1,) if self._gray else (3,))
        return gym.spaces.Dict({
            "image": gym.spaces.Box(0, 255, img_shape, np.uint8),
            "is_first": gym.spaces.Box(0, 1, (), bool),
            "is_last": gym.spaces.Box(0, 1, (), bool),
            "is_terminal": gym.spaces.Box(0, 1, (), bool),
        })

    @property
    def action_space(self):
        return gym.spaces.Discrete(len(self.actionset))

    def step(self, action):
        if isinstance(action, np.ndarray) and action.ndim >= 1:
            action = int(np.argmax(action))
        else:
            action = int(action)

        total_reward = 0.0
        dead = False
        game_over = False

        for repeat in range(self._repeat):
            _, reward, terminated, truncated, _ = self._env.step(action)
            self._step += 1
            total_reward += reward
            game_over = terminated or truncated

            if repeat >= self._repeat - self._pooling:
                self._render()

            if game_over:
                break

            current_lives = self.ale.lives()
            if self._lives != "unused" and current_lives < self._last_lives:
                dead = True
                self._last_lives = current_lives
                break
            self._last_lives = current_lives

        self._done = game_over or (self._length and self._step >= self._length)

        return self._obs(
            total_reward,
            is_last=self._done or (dead and self._lives == "reset"),
            is_terminal=dead or game_over,
        )

    def reset(self, seed=None):
        with self.LOCK:
            self._env.reset(seed=seed)

        if self._noops:
            for _ in range(self._rng.integers(self._noops + 1)):
                _, _, terminated, truncated, _ = self._env.step(0)
                if terminated or truncated:
                    with self.LOCK:
                        self._env.reset()

        if self._autostart and "FIRE" in self._action_meanings:
            fire = self._action_meanings.index("FIRE")
            _, _, terminated, truncated, _ = self._env.step(fire)
            if terminated or truncated:
                with self.LOCK:
                    self._env.reset()

        self._last_lives = self.ale.lives()
        self._render()
        # Fill the buffer with the first frame
        for _ in range(self._pooling - 1):
            self._buffers.appendleft(self._buffers[0].copy())

        self._done = False
        self._step = 0
        obs, _, _, _ = self._obs(0.0, is_first=True)
        return obs

    def _obs(self, reward, is_first=False, is_last=False, is_terminal=False):
        if self._clip_reward:
            reward = np.sign(reward)
        if self._aggregate == "max":
            image = np.amax(self._buffers, 0)
        elif self._aggregate == "mean":
            image = np.mean(self._buffers, 0).astype(np.uint8)

        if image.shape[:2] != self._size:
            if self._resize_fn == "opencv":
                image = self._cv2.resize(image,
                                         self._size,
                                         interpolation=self._cv2.INTER_AREA)
            if self._resize_fn == "pillow":
                image = Image.fromarray(image)
                image = image.resize(self._size, Image.BILINEAR)
                image = np.array(image)

        if self._gray:
            image = (image * self.WEIGHTS).sum(-1).astype(image.dtype)
            image = image[:, :, None]

        obs = {
            "image": image,
            "is_first": is_first,
            "is_last": is_last,
            "is_terminal": is_terminal,
        }
        return obs, reward, is_last, {}

    def _render(self):
        # Efficiently render by reusing buffer memory
        self._buffers.appendleft(self._buffers.pop())
        self.ale.getScreenRGB(self._buffers[0])

    def close(self):
        self._env.close()
