import atexit
import pathlib
import sys
import warnings

import hydra
import torch
from hydra.utils import to_absolute_path

import tools
from dreamer import Dreamer
from episode_memory import load_atari_memory_episode
from envs import make_envs
from replay_y import ReplayY
from trainer import OnlineTrainer

warnings.filterwarnings("ignore")
sys.path.append(str(pathlib.Path(__file__).parent))
# torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


def save_agent_state_dict(agent, path):
    torch.save(agent.state_dict(), path)
    print(f"Saved Dreamer agent state_dict to {path}.")


def load_agent_state_dict(agent, path):
    state_dict = torch.load(path, map_location=agent.device)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unsupported checkpoint format in {path}.")
    if "agent_state_dict" in state_dict or "optims_state_dict" in state_dict:
        raise TypeError(
            f"Checkpoint at {path} is a full training checkpoint; only raw "
            "Dreamer agent state_dict files are supported.")
    agent.load_state_dict(state_dict)
    agent.clone_and_freeze()
    print(f"Loaded Dreamer agent state_dict from {path}.")


def resolve_config_path(path_value):
    path = pathlib.Path(str(path_value)).expanduser()
    if path.is_absolute():
        return path
    return pathlib.Path(to_absolute_path(str(path)))


@hydra.main(version_base=None, config_path="configs", config_name="configs")
def main(config):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)

    # Mirror stdout/stderr to a file under logdir while keeping console output.
    console_f = tools.setup_console_log(logdir, filename="console.log")
    atexit.register(lambda: console_f.close())

    print("Logdir", logdir)

    logger = tools.Logger(logdir)
    # save config
    logger.log_hydra_config(config)

    if int(config.batch_length) < 1:
        raise AssertionError("config.batch_length must be >= 1 "
                             f"(got batch_length={int(config.batch_length)}).")
    if int(config.model.transformer.window_size) != int(config.batch_length):
        raise AssertionError(
            "config.model.transformer.window_size must equal config.batch_length "
            f"(got window_size={int(config.model.transformer.window_size)}, "
            f"batch_length={int(config.batch_length)}).")

    print("Create envs.")
    train_envs, eval_envs, obs_space, act_space = make_envs(config.env)

    memory_path = pathlib.Path(__file__).with_name("ge.json")
    memory = None
    if memory_path.exists():
        print(f"Load memory from {memory_path}.")
        env_task = str(config.env.task)
        memory_env_name = "montezuma_revenge"
        memory_env_config = None
        if env_task.startswith("atari_"):
            memory_env_name = env_task.split("_", 1)[1]
            memory_env_config = config.env
        memory = load_atari_memory_episode(
            memory_path,
            env_name=memory_env_name,
            env_config=memory_env_config,
        )
        print("Recovered memory episode with "
              f"{len(memory['reward'])} replay-style steps.")
    else:
        print(f"Memory file {memory_path} not found; replay.memory=None.")

    replay_buffer = ReplayY(
        length=int(config.batch_length),
        seed=config.seed,
        memory=memory,
        memory_sample_frac=float(config.buffer.memory_sample_frac),
    )

    print("Simulate agent.")
    agent = Dreamer(
        config.model,
        obs_space,
        act_space,
    ).to(config.device)
    if config.agent_load_path is not None:
        agent_load_path = resolve_config_path(config.agent_load_path)
        if not agent_load_path.exists():
            raise FileNotFoundError(
                f"Configured agent_load_path does not exist: {agent_load_path}")
        load_agent_state_dict(agent, agent_load_path)

    policy_trainer = OnlineTrainer(config.trainer, replay_buffer, logger,
                                   logdir, train_envs, eval_envs)
    interrupted_agent_path = logdir / "agent_interrupted_state_dict.pt"
    try:
        policy_trainer.begin(agent)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received during training.")
        save_agent_state_dict(agent, interrupted_agent_path)
        raise


if __name__ == "__main__":
    main()
