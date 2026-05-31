import pathlib
import sys
import warnings

import hydra
import torch
from hydra.utils import to_absolute_path

import tools
from dreamer import Dreamer
from envs import make_envs
from replay import Replay
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

    logger = tools.setup_logging(logdir, "{}.txt".format("train"))

    logger.info("Logdir %s", logdir)

    logger.info("Create envs.")
    train_envs, eval_envs, obs_space, act_space = make_envs(config.env)

    replay_buffer = Replay(
        length=int(config.batch_length),
        seed=config.seed,
    )

    logger.info("Simulate agent.")
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
                                   train_envs, eval_envs)
    interrupted_agent_path = logdir / "agent_interrupted_state_dict.pt"
    try:
        policy_trainer.begin(agent)
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received during training.")
        save_agent_state_dict(agent, interrupted_agent_path)
        raise


if __name__ == "__main__":
    main()
