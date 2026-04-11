import json
import pathlib
import threading

import numpy as np

from envs import atari
from envs import wrappers


def _one_hot(index, size):
    action = np.zeros(size, dtype=np.float32)
    action[int(index)] = 1.0
    return action


def _stack_episode(episode):
    return {k: np.stack([step[k] for step in episode]) for k in episode[0]}


def _step_dict(obs, reward, action):
    step = {k: np.asarray(v) for k, v in obs.items()}
    step["reward"] = np.asarray(np.float32(reward))
    step["action"] = np.asarray(action, dtype=np.float32)
    return step


def _action_dim(space):
    if hasattr(space, "n"):
        return int(space.n)
    if hasattr(space, "shape"):
        return int(np.prod(space.shape))
    raise AttributeError("Unsupported action space without 'n' or 'shape'.")


def make_atari_memory_env(env_config=None, env_name="montezuma_revenge"):
    if atari.Atari.LOCK is None:
        # The memory episode is rebuilt in the main process only, so a
        # regular thread lock is sufficient and avoids sandbox semlock
        # restrictions from multiprocessing.Lock().
        atari.Atari.LOCK = threading.Lock()

    if env_config is None:
        return atari.Atari(name=env_name, sticky=False)

    env = atari.Atari(
        name=env_name,
        action_repeat=int(env_config.action_repeat),
        size=tuple(env_config.size),
        gray=bool(env_config.gray),
        noops=int(env_config.noops),
        lives=env_config.lives,
        # Keep sticky actions disabled for memory recovery so the action
        # sequence in ge.json deterministically reproduces the intended run.
        sticky=False,
        actions=str(env_config.actions),
        length=int(env_config.time_limit),
        pooling=int(env_config.pooling),
        aggregate=str(env_config.aggregate),
        resize=str(env_config.resize),
        autostart=bool(env_config.autostart),
        clip_reward=bool(env_config.clip_reward),
        seed=None,
    )
    env = wrappers.OneHotAction(env)
    env = wrappers.TimeLimit(
        env,
        int(env_config.time_limit) // int(env_config.action_repeat))
    env = wrappers.Dtype(env)
    return env


def build_atari_memory_episode(actions,
                               expected_score,
                               env_name="montezuma_revenge",
                               reset_seed=0,
                               env_config=None,
                               env_ctor=None):
    actions = np.asarray(actions, dtype=np.int64)
    if actions.ndim != 1 or len(actions) == 0:
        raise ValueError(
            "Memory episode requires a non-empty 1D action sequence.")

    if env_ctor is None:
        env_ctor = lambda: make_atari_memory_env(env_config, env_name)

    env = env_ctor()
    episode = []
    score = 0.0
    action_dim = _action_dim(env.action_space)
    try:
        obs = env.reset(seed=reset_seed)
        episode.append(_step_dict(obs, 0.0, _one_hot(actions[0], action_dim)))

        for index, action in enumerate(actions):
            obs, reward, done, _ = env.step(_one_hot(action, action_dim))
            score += float(reward)
            if done:
                next_action = np.zeros(action_dim, dtype=np.float32)
                if index != len(actions) - 1:
                    raise ValueError(
                        "Episode terminated before the provided action sequence ended."
                    )
            else:
                if index == len(actions) - 1:
                    raise ValueError(
                        "Action sequence ended before the episode terminated.")
                next_action = _one_hot(actions[index + 1], action_dim)
            # ReplayY stores s_t together with a_t and the reward from the
            # preceding transition. The terminal observation therefore keeps
            # the terminal reward and gets a zero action after masking.
            episode.append(_step_dict(obs, reward, next_action))
    finally:
        env.close()

    if not bool(episode[-1]["is_last"]):
        raise ValueError(
            "The provided action sequence does not reach episode end.")
    if not np.isclose(score, float(expected_score)):
        raise ValueError("Recovered score does not match ge.json: "
                         f"{score} != {float(expected_score)}")
    return _stack_episode(episode)


def load_atari_memory_episode(path,
                              env_name="montezuma_revenge",
                              reset_seed=0,
                              env_config=None,
                              env_ctor=None):
    path = pathlib.Path(path)
    with path.open() as f:
        payload = json.load(f)
    return build_atari_memory_episode(
        payload["actions"],
        payload["score"],
        env_name=env_name,
        reset_seed=reset_seed,
        env_config=env_config,
        env_ctor=env_ctor,
    )
