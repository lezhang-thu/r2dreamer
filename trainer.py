import time

import numpy as np
import torch

import tools


class OnlineTrainer:

    def __init__(self, config, replay_buffer, logger, train_envs, eval_envs):
        self.replay_buffer = replay_buffer
        self.logger = logger
        self.train_envs = train_envs
        self.eval_envs = eval_envs
        self.steps = int(config.steps)
        self.eval_every = int(config.eval_every)
        self.eval_episode_num = int(config.eval_episode_num)
        self.batch_size = int(config.batch_size)
        self.batch_length = int(config.batch_length)
        self.grad_accum_steps = int(config.get("grad_accum_steps", 1))
        self._action_repeat = int(config.action_repeat)
        self.effective_batch_size = self.batch_size * self.grad_accum_steps
        self.random_action_steps = int(config.get("random_action_steps", 1e4))
        self._random_action_until_step = (self.random_action_steps *
                                          self._action_repeat)
        batch_steps = int(self.effective_batch_size * config.batch_length)
        # train_ratio is based on data steps rather than environment steps.
        self._updates_needed = tools.Every(batch_steps / config.train_ratio *
                                           self._action_repeat)
        self._should_log = tools.Every(config.update_log_every)
        self._should_eval = tools.Every(self.eval_every)
        self._last_log_step = None
        self._last_log_time = None

    def _to_log_value(self, value):
        if isinstance(value, torch.Tensor):
            value = value.detach()
            if value.numel() == 1:
                return value.item()
            return value.to(torch.float32).mean().item()
        if isinstance(value, np.ndarray):
            if value.size == 1:
                return value.item()
            return value.astype(np.float32).mean().item()
        if isinstance(value, np.generic):
            return value.item()
        return value

    def _format_metrics(self, metrics):
        parts = []
        for name, value in metrics.items():
            value = self._to_log_value(value)
            if isinstance(value, float):
                parts.append(f"{name}: {value:.6g}")
            else:
                parts.append(f"{name}: {value}")
        return ", ".join(parts)

    def _fps(self, step):
        now = time.time()
        if self._last_log_step is None:
            self._last_log_step = step
            self._last_log_time = now
            return 0.0
        duration = now - self._last_log_time
        fps = (step - self._last_log_step) / duration if duration > 0 else 0.0
        self._last_log_step = step
        self._last_log_time = now
        return fps

    def eval(self, agent, train_step):
        """Run evaluation episodes.

        Environment stepping is executed on CPU to avoid GPU<->CPU synchronizations
        in the worker processes. Observations are moved back to GPU asynchronously
        (H2D with non_blocking=True) right before policy inference.
        """
        self.logger.info("Evaluating the policy.")
        envs = self.eval_envs
        agent.eval()
        # (B,)
        done = torch.ones(envs.env_num, dtype=torch.bool, device=agent.device)
        once_done = torch.zeros(envs.env_num,
                                dtype=torch.bool,
                                device=agent.device)
        steps = torch.zeros(envs.env_num,
                            dtype=torch.int32,
                            device=agent.device)
        returns = torch.zeros(envs.env_num,
                              dtype=torch.float32,
                              device=agent.device)
        log_metrics = {}
        agent_state = agent.get_initial_state(envs.env_num)
        # (B, A)
        act = agent_state["prev_action"].clone()
        while not once_done.all():
            steps += ~done * ~once_done
            # Step environments on CPU.
            # (B, A)
            act_cpu = act.detach().to("cpu")
            # (B,)
            done_cpu = done.detach().to("cpu")
            trans_cpu, done_cpu = envs.step(act_cpu, done_cpu)
            # Move observations back to GPU asynchronously for the agent.
            # dict of (B, 1, *)
            trans = trans_cpu.to(agent.device, non_blocking=True)
            # (B,)
            done = done_cpu.to(agent.device)

            # Store transition.
            # Attach the action sent to env.step at this loop step.
            # For envs that were done, trans may be reset output, so this is
            # for logging/cache consistency rather than strict causality.
            trans["action"] = act
            # (B, A)
            act, agent_state = agent.act(trans, agent_state, eval=True)
            returns += trans["reward"][:, 0] * ~once_done
            for key, value in trans.items():
                if key.startswith("log_"):
                    if key not in log_metrics:
                        log_metrics[key] = torch.zeros_like(returns)
                    log_metrics[key] += value[:, 0] * ~once_done
            once_done |= done
        metrics = {
            "step": train_step,
            "score": returns.mean(),
            "len": steps.to(torch.float32).mean(),
        }
        for key, value in log_metrics.items():
            if key == "log_success":
                value = torch.clip(value,
                                   max=1.0)  # make sure 1.0 for success episode
            metrics[f"eval_{key[4:]}"] = value.mean()
        self.logger.info("eval: %s", self._format_metrics(metrics))
        agent.train()

    def begin(self, agent):
        """Main online training loop.

        The loop is designed to overlap CPU environment stepping and GPU model
        execution. Environments are stepped on CPU, observations are pinned,
        then transferred to GPU with non_blocking=True.
        """
        envs = self.train_envs
        step = 0
        update_count = 0
        # (B,)
        done = torch.ones(envs.env_num, dtype=torch.bool, device=agent.device)
        returns = torch.zeros(envs.env_num,
                              dtype=torch.float32,
                              device=agent.device)
        lengths = torch.zeros(envs.env_num,
                              dtype=torch.int32,
                              device=agent.device)
        train_metrics = {}
        agent_state = agent.get_initial_state(envs.env_num)
        # (B, A)
        act = agent_state["prev_action"].clone()

        while step < self.steps:
            # Evaluation
            if self._should_eval(step) and self.eval_episode_num > 0:
                self.eval(agent, step)
            # Save metrics
            if done.any():
                for i, d in enumerate(done):
                    if d and lengths[i] > 0:
                        self.logger.info(
                            "episode: %s",
                            self._format_metrics({
                                "step": step + i,
                                "score": returns[i],
                                "len": lengths[i],
                            }))
                        returns[i] = lengths[i] = 0
            step += int((
                ~done).sum()) * self._action_repeat  # step is based on env side
            lengths += ~done

            # Step environments on CPU to avoid GPU<->CPU sync in the worker processes.
            # (B, A)
            act_cpu = act.detach().to("cpu")
            # (B,)
            done_cpu = done.detach().to("cpu")
            trans_cpu, done_cpu = envs.step(act_cpu, done_cpu)

            # Move observations back to GPU asynchronously for the agent.
            # dict of (B, 1, *)
            trans = trans_cpu.to(agent.device, non_blocking=True)
            # (B,)
            done = done_cpu.to(agent.device)

            # Policy inference on GPU.
            # "agent_state" is reset by the agent based on the "is_first" flag in trans.
            # (B, A)
            use_random_action = step < self._random_action_until_step
            act, agent_state = agent.act(trans.clone(),
                                         agent_state,
                                         eval=False,
                                         random=use_random_action)

            # Store transition into Replay.
            # We pair each observation s_t with the action a_t = π(s_t) taken in response.
            # Mask actions after an episode has ended.
            trans["action"] = act * ~done.unsqueeze(-1)
            # Add each env's transition to the replay as a separate worker.
            # Scalar fields (reward, is_first, ...) have a singleton time dim
            # (B, 1) from lift_dim; squeeze it so each step stores a scalar.
            _SCALAR_KEYS = {"reward", "is_first", "is_last", "is_terminal"}
            trans_np = {
                k:
                    tools.to_np(v.squeeze(1))
                    if k in _SCALAR_KEYS else tools.to_np(v)
                for k, v in trans.items()
            }
            for i in range(envs.env_num):
                step_dict = {k: v[i] for k, v in trans_np.items()}
                self.replay_buffer.add(step_dict, worker=i)
            returns += trans["reward"][:, 0]

            if self.replay_buffer.can_sample(self.effective_batch_size):
                update_num = self._updates_needed(step)
                for _ in range(update_num):
                    _metrics = agent.update(self.replay_buffer, self.batch_size)
                    train_metrics = _metrics
                update_count += update_num
                # Log training metrics
                if self._should_log(step):
                    metrics = {"t": step, "updates": update_count}
                    for name, value in train_metrics.items():
                        metrics[name] = value
                    metrics["fps"] = self._fps(step)
                    self.logger.info("train: %s", self._format_metrics(metrics))
