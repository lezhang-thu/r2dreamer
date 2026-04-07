import atexit
import pathlib
import sys
import warnings

import hydra
import torch

import tools
from replay_y import ReplayY
from dreamer import Dreamer
from envs import make_envs
from trainer import OnlineTrainer

warnings.filterwarnings("ignore")
sys.path.append(str(pathlib.Path(__file__).parent))
# torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")


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

    replay_buffer = ReplayY(
        length=int(config.batch_length),
        seed=config.seed,
    )

    print("Create envs.")
    train_envs, eval_envs, obs_space, act_space = make_envs(config.env)

    print("Simulate agent.")
    agent = Dreamer(
        config.model,
        obs_space,
        act_space,
    ).to(config.device)

    policy_trainer = OnlineTrainer(config.trainer, replay_buffer, logger,
                                   logdir, train_envs, eval_envs)
    policy_trainer.begin(agent)

    items_to_save = {
        "agent_state_dict": agent.state_dict(),
        "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
    }
    torch.save(items_to_save, logdir / "latest.pt")


if __name__ == "__main__":
    main()
