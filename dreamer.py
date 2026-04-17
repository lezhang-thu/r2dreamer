import copy
from collections import OrderedDict

import torch
from tensordict import TensorDict
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR

import networks
import rssm
import tools
from distributions import symlog
from networks import Projector
from optim import LaProp, clip_grad_agc_
from tools import to_f32


class ExpertTag(nn.Module):
    """Marks retrieved memory tokens as expert context."""

    def __init__(self, dim):
        super().__init__()
        self.tag = nn.Parameter(torch.zeros(int(dim), dtype=torch.float32))

    def forward(self, x):
        return x + self.tag.to(device=x.device, dtype=x.dtype)


class Dreamer(nn.Module):

    def __init__(self, config, obs_space, act_space, memory=None):
        super().__init__()
        self.memory = memory
        self.device = torch.device(config.device)
        self.act_entropy = float(config.act_entropy)
        self.kl_free = float(config.kl_free)
        self.imag_horizon = int(config.imag_horizon)
        self.horizon = int(config.horizon)
        self.lamb = float(config.lamb)
        self.return_ema = networks.ReturnEMA(device=self.device)
        self._mem_disc = 1.0 - 1.0 / self.horizon
        self._memory_waypoint_offset = max(1, min(self.imag_horizon, 8))
        self.expert_shaping_scale = float(getattr(config, "expert_shaping_scale", 0.0))
        self.act_dim = act_space.n if hasattr(act_space, "n") else sum(act_space.shape)
        if str(config.rep_loss) != "r2dreamer":
            raise AssertionError(f"config.rep_loss must be 'r2dreamer' (got {str(config.rep_loss)!r}).")
        self.imag_last = int(getattr(config, "imag_last", 0))
        self.wm_accum_steps = max(1, int(getattr(config, "wm_accum_steps", 1)))
        self.ac_accum_steps = max(1, int(getattr(config, "ac_accum_steps", 1)))

        # World model components
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
        self.encoder = networks.MultiEncoder(config.encoder, shapes)
        self.embed_size = self.encoder.out_dim
        self.rssm = rssm.TransformerRSSM(
            config.transformer,
            self.embed_size,
            self.act_dim,
        )
        self.reward = networks.MLPHead(config.reward, self.rssm.feat_size)
        self.cont = networks.MLPHead(config.cont, self.rssm.feat_size)

        config.actor.shape = (act_space.n,) if hasattr(act_space, "n") else tuple(map(int, act_space.shape))
        self.act_discrete = False
        if hasattr(act_space, "multi_discrete"):
            config.actor.dist = config.actor.dist.multi_disc
            self.act_discrete = True
        elif hasattr(act_space, "discrete"):
            config.actor.dist = config.actor.dist.disc
            self.act_discrete = True
        else:
            config.actor.dist = config.actor.dist.cont

        # Actor-critic components
        deter_dim = int(config.transformer.deter)
        act_fn = getattr(torch.nn, config.transformer.act)
        self._memory_scalar_dim = 3  # reward, return-to-go, progress
        self._memory_rtg_embed_dim = 16
        self.rtg_proj = nn.Linear(1, self._memory_rtg_embed_dim, bias=True)
        self.mem_proj = nn.Sequential(
            nn.Linear(self.rssm.feat_size + self.act_dim + self._memory_rtg_embed_dim, deter_dim, bias=True),
            act_fn(),
            nn.Linear(deter_dim, deter_dim, bias=True),
        )
        self.expert_tag = ExpertTag(deter_dim)
        self.rl_feat_size = self.rssm.flat_stoch + 3 * deter_dim + self.act_dim + self._memory_scalar_dim
        self.actor = networks.MLPHead(config.actor, self.rl_feat_size)
        self.value = networks.MLPHead(config.critic, self.rl_feat_size)
        self.memory_attention = nn.MultiheadAttention(
            deter_dim,
            int(config.transformer.n_heads),
            dropout=0.0,
            batch_first=True,
        )
        self.memory_progress = nn.Sequential(
            nn.Linear(deter_dim, deter_dim, bias=True),
            act_fn(),
            nn.Linear(deter_dim, 1, bias=True),
        )
        self.slow_target_update = int(config.slow_target_update)
        self.slow_target_fraction = float(config.slow_target_fraction)
        self._slow_value = copy.deepcopy(self.value)
        for param in self._slow_value.parameters():
            param.requires_grad = False
        self._slow_value_updates = 0
        self._memory_context = None
        self._memory_context_stale = self.memory is not None

        self._loss_scales = dict(config.loss_scales)
        self._log_grads = bool(config.log_grads)

        modules = {
            "rssm": self.rssm,
            "actor": self.actor,
            "value": self.value,
            "rtg_proj": self.rtg_proj,
            "mem_proj": self.mem_proj,
            "expert_tag": self.expert_tag,
            "memory_attention": self.memory_attention,
            "memory_progress": self.memory_progress,
            "reward": self.reward,
            "cont": self.cont,
            "encoder": self.encoder,
        }

        # R2-Dreamer redundancy-reduction head.
        self.prj = Projector(self.rssm.feat_size, self.embed_size)
        modules.update({"projector": self.prj})
        self.barlow_lambd = float(config.r2dreamer.lambd)
        # count number of parameters in each module
        for key, module in modules.items():
            if isinstance(module, nn.Parameter):
                print(f"{module.numel():>14,}: {key}")
            else:
                print(f"{sum(p.numel() for p in module.parameters()):>14,}: {key}")
        self._named_params = OrderedDict()
        for name, module in modules.items():
            if isinstance(module, nn.Parameter):
                self._named_params[name] = module
            else:
                for param_name, param in module.named_parameters():
                    self._named_params[f"{name}.{param_name}"] = param
        print(f"Optimizer has: {sum(p.numel() for p in self._named_params.values())} parameters.")

        def _agc(params):
            clip_grad_agc_(params, float(config.agc), float(config.pmin), foreach=True)

        self._agc = _agc
        self._optimizer = LaProp(
            self._named_params.values(),
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
        )
        self._scaler = GradScaler()

        def lr_lambda(step):
            if config.warmup:
                return min(1.0, (step + 1) / config.warmup)
            return 1.0

        self._scheduler = LambdaLR(self._optimizer, lr_lambda=lr_lambda)

        self.train()
        self.clone_and_freeze()
        if config.compile:
            print("Compiling update function with torch.compile...")
            self._cal_grad = torch.compile(self._cal_grad, mode="default")

    def _update_slow_target(self):
        """Update slow-moving value target network."""
        if self._slow_value_updates % self.slow_target_update == 0:
            with torch.no_grad():
                mix = self.slow_target_fraction
                for v, s in zip(self.value.parameters(), self._slow_value.parameters()):
                    s.data.copy_(mix * v.data + (1 - mix) * s.data)
        self._slow_value_updates += 1

    def train(self, mode=True):
        super().train(mode)
        # slow_value should be always eval mode
        self._slow_value.train(False)
        return self

    def _freeze_copy(self, module):
        """Deepcopy then share .data so the clone always sees latest weights.

        NOTE: requires_grad affects whether a parameter is updated,
        not whether gradients flow through its operations.
        """
        clone = copy.deepcopy(module)
        for p_orig, p_clone in zip(module.parameters(), clone.parameters()):
            p_clone.data = p_orig.data
            p_clone.requires_grad_(False)
        return clone

    def clone_and_freeze(self):
        for name in (
            "encoder",
            "rssm",
            "reward",
            "cont",
            "actor",
            "value",
            "slow_value",
            "rtg_proj",
            "mem_proj",
            "expert_tag",
            "memory_attention",
            "memory_progress",
        ):
            setattr(
                self, f"_frozen_{name}", self._freeze_copy(getattr(self, f"_{name}" if name == "slow_value" else name))
            )

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        # Re-establish shared memory after moving the model to a new device
        self.clone_and_freeze()
        self.refresh_memory_context()
        return self

    def _memory_episode_tensordict(self):
        if self.memory is None:
            return None
        if "reward" not in self.memory:
            raise KeyError("self.memory must contain a 'reward' field.")
        T = int(len(self.memory["reward"]))
        data = {}
        for key, value in self.memory.items():
            tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
            if tensor.is_floating_point():
                tensor = tensor.to(self.device, non_blocking=True)
            else:
                tensor = tensor.to(self.device)
            data[key] = tensor.unsqueeze(0)
        if "is_first" not in data:
            data["is_first"] = torch.zeros(1, T, dtype=torch.bool, device=self.device)
        data["is_first"] = data["is_first"].to(torch.bool)
        data["is_first"][:, 0] = True
        return TensorDict(data, batch_size=(1, T))

    def _compute_memory_return_to_go(self, reward):
        reward = reward.reshape(-1).to(device=self.device, dtype=torch.float32)
        rtg = torch.zeros_like(reward)
        running = torch.zeros((), dtype=torch.float32, device=reward.device)
        disc = float(getattr(self, "_mem_disc", 1.0))
        for idx in reversed(range(int(reward.shape[0]))):
            running = reward[idx] + disc * running
            rtg[idx] = running
        return rtg.unsqueeze(-1)

    def _compute_memory_progress(self, length, device):
        length = int(length)
        if length <= 1:
            return torch.zeros(length, 1, dtype=torch.float32, device=device)
        return torch.linspace(0.0, 1.0, steps=length, dtype=torch.float32, device=device).unsqueeze(-1)

    def _compute_memory_waypoint(self, deter):
        T = int(deter.shape[0])
        offset = min(int(self._memory_waypoint_offset), max(T - 1, 0))
        index = torch.arange(T, device=deter.device) + offset
        index = torch.clamp(index, max=max(T - 1, 0))
        return deter[index]

    @torch.no_grad()
    def refresh_memory_context(self):
        if self.memory is None:
            self._memory_context = None
            self._memory_context_stale = False
            return None
        data = self.preprocess(self._memory_episode_tensordict())
        embed = self._frozen_encoder(data)
        _, feat_dict = self._frozen_rssm.observe(embed, data["action"], data["is_first"], sample=False)
        deter = feat_dict["deter"].squeeze(0).detach()
        stoch_flat = feat_dict["stoch"].squeeze(0).reshape(deter.shape[0], self.rssm.flat_stoch).detach()
        reward = data["reward"].squeeze(0).to(torch.float32).unsqueeze(-1)
        raw_rtg = self._compute_memory_return_to_go(reward).detach()
        self._memory_context = {
            "deter": deter,
            "stoch_flat": stoch_flat,
            "action": data["action"].squeeze(0).detach(),
            "reward": reward.detach(),
            "rtg": symlog(raw_rtg).detach(),
            "raw_rtg": raw_rtg,
            "progress": self._compute_memory_progress(deter.shape[0], deter.device).detach(),
            "waypoint": self._compute_memory_waypoint(deter).detach(),
        }
        self._memory_context_stale = False
        return self._memory_context

    def _get_memory_context(self, require_fresh=False):
        if self.memory is None:
            return None
        if self._memory_context is None or (require_fresh and self._memory_context_stale):
            return self.refresh_memory_context()
        return self._memory_context

    def _build_memory_tokens(self, memory_context, dtype, device, frozen=False):
        if memory_context is None:
            return None
        rtg_proj = self._frozen_rtg_proj if frozen else self.rtg_proj
        mem_proj = self._frozen_mem_proj if frozen else self.mem_proj
        expert_tag = self._frozen_expert_tag if frozen else self.expert_tag
        deter = memory_context["deter"].to(device=device, dtype=dtype)
        stoch_flat = memory_context["stoch_flat"].to(device=device, dtype=dtype)
        action = memory_context["action"].to(device=device, dtype=dtype)
        rtg = memory_context["rtg"].to(device=device, dtype=dtype)
        token_feat = torch.cat([deter, stoch_flat, action, rtg_proj(rtg)], dim=-1)
        return expert_tag(mem_proj(token_feat))

    def _zero_memory_readout(self, query):
        zeros_d = torch.zeros_like(query)
        return {
            "token": zeros_d,
            "waypoint": zeros_d,
            "action": torch.zeros(*query.shape[:-1], self.act_dim, dtype=query.dtype, device=query.device),
            "reward": torch.zeros(*query.shape[:-1], 1, dtype=query.dtype, device=query.device),
            "rtg": torch.zeros(*query.shape[:-1], 1, dtype=query.dtype, device=query.device),
            "raw_rtg": torch.zeros(*query.shape[:-1], 1, dtype=query.dtype, device=query.device),
            "progress": torch.zeros(*query.shape[:-1], 1, dtype=query.dtype, device=query.device),
            "weights": None,
        }

    def _read_memory(self, deter, frozen=False, require_fresh_memory=False):
        memory_context = self._get_memory_context(require_fresh=require_fresh_memory)
        if deter.ndim not in (2, 3):
            raise ValueError(f"Memory attention expects deter with rank 2 or 3 (got shape {tuple(deter.shape)}).")
        squeeze_time = deter.ndim == 2
        query = deter.unsqueeze(1) if squeeze_time else deter
        if memory_context is None:
            readout = self._zero_memory_readout(query)
        else:
            memory_tokens = self._build_memory_tokens(
                memory_context, dtype=query.dtype, device=query.device, frozen=frozen
            )
            memory_tokens = memory_tokens.unsqueeze(0).expand(query.shape[0], -1, -1)
            attention = self._frozen_memory_attention if frozen else self.memory_attention
            attended, weights = attention(query, memory_tokens, memory_tokens, need_weights=True)
            readout = {
                "token": attended,
                "weights": weights,
            }
            for key in ("waypoint", "action", "reward", "rtg", "raw_rtg", "progress"):
                value = memory_context[key].to(device=query.device, dtype=query.dtype)
                value = value.unsqueeze(0).expand(query.shape[0], -1, -1)
                readout[key] = torch.matmul(weights, value)
        if squeeze_time:
            for key, value in tuple(readout.items()):
                if value is not None:
                    readout[key] = value[:, 0]
        return readout

    def _apply_memory_attention(self, deter, frozen=False, require_fresh_memory=False):
        return self._read_memory(deter, frozen=frozen, require_fresh_memory=require_fresh_memory)["token"]

    def _memory_targets(self, memory_index, dtype, device, require_fresh=False):
        memory_context = self._get_memory_context(require_fresh=require_fresh)
        if memory_context is None:
            return None
        memory_index = memory_index.to(device=device, dtype=torch.long)
        gather_index = torch.clamp(memory_index, min=0)
        targets = {}
        for key in ("reward", "rtg", "raw_rtg", "progress", "waypoint"):
            value = memory_context[key].to(device=device, dtype=dtype)
            picked = value[gather_index.reshape(-1)]
            targets[key] = picked.reshape(*gather_index.shape, value.shape[-1])
        return targets

    def _get_rl_feat(self, stoch, deter, frozen=False, require_fresh_memory=False, return_aux=False):
        memory_readout = self._read_memory(
            deter,
            frozen=frozen,
            require_fresh_memory=require_fresh_memory,
        )
        stoch_flat = stoch.reshape(*stoch.shape[:-2], self.rssm.flat_stoch)
        reward_feat = symlog(memory_readout["reward"])
        rl_feat = torch.cat(
            [
                stoch_flat,
                deter,
                memory_readout["token"],
                memory_readout["waypoint"],
                memory_readout["action"],
                reward_feat,
                memory_readout["rtg"],
                memory_readout["progress"],
            ],
            dim=-1,
        )
        if return_aux:
            return rl_feat, memory_readout
        return rl_feat

    @torch.no_grad()
    def act(self, obs, state, eval=False):
        """Policy inference step."""
        torch.compiler.cudagraph_mark_step_begin()
        p_obs = self.preprocess(obs)
        # (B, E)
        embed = self._frozen_encoder(p_obs)

        # Two-phase KV-cache inference
        carry = {
            "kv_cache": state["kv_cache"],
            "pos": state["pos"],
            "h_prev": state["h_prev"],
        }
        # Trainer provides (B, 1, *) tensors; squeeze time dim
        embed_sq = embed.squeeze(1)  # (B, E)
        is_first = obs["is_first"].squeeze(1)  # (B,)
        # Phase 1: posterior from tokens
        carry, stoch, h_prev = self._frozen_rssm.get_feat_step(carry, embed_sq, is_first)
        rl_feat = self._get_rl_feat(stoch, h_prev, frozen=True, require_fresh_memory=True)
        action_dist = self._frozen_actor(rl_feat)
        action = action_dist.mode if eval else action_dist.rsample()
        # Phase 2: update KV-cache with (stoch, action)
        carry = self._frozen_rssm.update_carry(carry, stoch, action, is_first)
        return action, TensorDict(
            {
                "kv_cache": carry["kv_cache"],
                "pos": carry["pos"],
                "h_prev": carry["h_prev"],
                "prev_action": action,
            },
            batch_size=state.batch_size,
        )

    @torch.no_grad()
    def get_initial_state(self, B):
        carry = self.rssm.initial(B)
        action = torch.zeros(B, self.act_dim, dtype=torch.float32, device=self.device)
        return TensorDict(
            {
                "kv_cache": carry["kv_cache"],
                "pos": carry["pos"],
                "h_prev": carry["h_prev"],
                "prev_action": action,
            },
            batch_size=(B,),
        )

    def update(self, replay_buffer, batch_size):
        """Sample a batch from replay and perform one optimization step.

        ReplayY provides contiguous single-trajectory segments (zero-padded
        only when the sampled episode is shorter than batch_length).

        Args:
            replay_buffer: ReplayY instance.
            batch_size: Number of replay segments to sample.

        Returns:
            metrics: Dict of training metrics.
        """
        np_data = replay_buffer.sample(int(batch_size))
        # Convert numpy data to torch tensors on device.
        data = {}
        for k, v in np_data.items():
            t = torch.from_numpy(v)
            if t.is_floating_point():
                t = t.to(self.device, non_blocking=True)
            else:
                t = t.to(self.device)
            data[k] = t
        data = TensorDict(data, batch_size=data["reward"].shape[:2])

        torch.compiler.cudagraph_mark_step_begin()
        p_data = self.preprocess(data)
        self._update_slow_target()
        metrics = {}
        with autocast(device_type=self.device.type, dtype=torch.float16):
            mets = self._cal_grad(p_data)
        self._scaler.unscale_(self._optimizer)  # unscale grads in params
        if self._log_grads:
            old_params = [p.data.clone().detach() for p in self._named_params.values()]
            grads = [p.grad for p in self._named_params.values() if p.grad is not None]  # log grads before clipping
            grad_norm = tools.compute_global_norm(grads)
            grad_rms = tools.compute_rms(grads)
            mets["opt/grad_norm"] = grad_norm
            mets["opt/grad_rms"] = grad_rms
        self._agc(self._named_params.values())  # clipping
        self._scaler.step(self._optimizer)  # update params
        self._scaler.update()  # adjust scale
        self._scheduler.step()  # increment scheduler
        self._optimizer.zero_grad(set_to_none=True)  # reset grads
        self._memory_context_stale = self.memory is not None
        mets["opt/lr"] = self._scheduler.get_lr()[0]
        mets["opt/grad_scale"] = self._scaler.get_scale()
        if self._log_grads:
            updates = [(new - old) for (new, old) in zip(self._named_params.values(), old_params)]
            update_rms = tools.compute_rms(updates)
            params_rms = tools.compute_rms(self._named_params.values())
            mets["opt/param_rms"] = params_rms
            mets["opt/update_rms"] = update_rms
        metrics.update(mets)
        return metrics

    @staticmethod
    def _masked_mean(x, mask):
        """Average over valid elements only, supporting broadcastable masks."""
        mask = mask.to(dtype=x.dtype)
        if mask.shape != x.shape:
            mask = torch.broadcast_to(mask, x.shape)
        denom = mask.sum().clamp_min(1.0)
        return (x * mask).sum() / denom

    @staticmethod
    def _masked_barlow_loss(x1, x2, flat_mask, lambd, eps=1e-8):
        """Compute Barlow Twins loss using only valid rows."""
        flat_mask = flat_mask.to(dtype=x1.dtype)
        count = flat_mask.sum().clamp_min(1.0)

        x1_mean = (x1 * flat_mask).sum(0) / count
        x2_mean = (x2 * flat_mask).sum(0) / count
        x1_centered = (x1 - x1_mean) * flat_mask
        x2_centered = (x2 - x2_mean) * flat_mask

        x1_var = x1_centered.pow(2).sum(0) / count
        x2_var = x2_centered.pow(2).sum(0) / count
        x1_norm = x1_centered / torch.sqrt(x1_var + eps)
        x2_norm = x2_centered / torch.sqrt(x2_var + eps)

        c = torch.mm(x1_norm.T, x2_norm) / count
        invariance_loss = (torch.diagonal(c) - 1.0).pow(2).sum()
        off_diag_mask = ~torch.eye(c.shape[0], dtype=torch.bool, device=c.device)
        redundancy_loss = c[off_diag_mask].pow(2).sum()
        return invariance_loss + lambd * redundancy_loss

    def _iter_batch_chunks(self, data, accum_steps):
        """Yield replay batch chunks and their batch-fraction weights."""
        B = data.shape[0]
        splits = min(max(1, int(accum_steps)), int(B))
        if splits <= 1:
            yield data, 1.0
            return
        chunk = (B + splits - 1) // splits
        for start in range(0, B, chunk):
            end = min(start + chunk, B)
            yield data[start:end], float(end - start) / float(B)

    def _iter_start_chunks(self, starts, accum_steps):
        """Yield sampled imagination-start index chunks and their weights."""
        K = int(starts.shape[1])
        splits = min(max(1, int(accum_steps)), max(1, K))
        if splits <= 1:
            yield starts, 1.0
            return
        chunk = (K + splits - 1) // splits
        for start in range(0, K, chunk):
            end = min(start + chunk, K)
            yield starts[:, start:end], float(end - start) / float(K)

    def _world_model_forward(self, data):
        """World-model losses and detached cache for imagination updates."""
        # t_mask: (B, T) bool — True for real data, False for padding
        t_mask = data["t_mask"].float()  # (B, T)
        B, T = data.shape

        losses = {}
        metrics = {}

        # === World model: posterior rollout and KL losses ===
        # (B, T, E)
        embed = self.encoder(data)

        # Transformer path: posterior from tokens, transition on (stoch, a_t)
        action = data["action"]  # (B, T, A) — current action a_t
        _, feat_dict = self.rssm.observe(embed, action, data["is_first"])
        post_stoch = feat_dict["stoch"]  # (B, T, S, K)
        post_deter = feat_dict["deter"]  # (B, T, D) = h_prev
        post_logit = feat_dict["post_logit"]  # (B, T, S, K)
        prior_logit = feat_dict["prior_logit"]
        dyn_loss, rep_loss = self.rssm.kl_loss(post_logit, prior_logit, self.kl_free)
        losses["dyn"] = self._masked_mean(dyn_loss, t_mask)
        losses["rep"] = self._masked_mean(rep_loss, t_mask)

        # === Representation / auxiliary losses ===
        # (B, T, F)
        feat = self.rssm.get_feat(post_stoch, post_deter)
        flat_mask = t_mask.reshape(B * T, 1)  # (B*T, 1)
        x1 = self.prj(feat.reshape(B * T, -1))
        x2 = embed.reshape(B * T, -1).detach()
        losses["barlow"] = self._masked_barlow_loss(x1, x2, flat_mask, self.barlow_lambd)

        rew_loss = -self.reward(feat).log_prob(to_f32(data["reward"]).unsqueeze(-1))  # (B, T)
        losses["rew"] = self._masked_mean(rew_loss, t_mask)
        cont = (1.0 - to_f32(data["is_terminal"])).unsqueeze(-1)
        con_loss = -self.cont(feat).log_prob(cont)  # (B, T)
        losses["con"] = self._masked_mean(con_loss, t_mask)

        memory_mask = data["is_memory"] & data["t_mask"]
        memory_targets = self._memory_targets(
            data["memory_index"], dtype=torch.float32, device=post_deter.device, require_fresh=False
        )
        if memory_targets is not None and bool(memory_mask.any().item()):
            expert_rl_feat, memory_readout = self._get_rl_feat(
                post_stoch.detach(),
                post_deter.detach(),
                frozen=False,
                require_fresh_memory=False,
                return_aux=True,
            )
            progress_pred = torch.sigmoid(self.memory_progress(post_deter.detach()))
            progress_loss = (progress_pred - memory_targets["progress"]).pow(2)
            losses["memory_progress"] = self._masked_mean(progress_loss, memory_mask.unsqueeze(-1))

            align_log_prob = torch.log(memory_readout["weights"].clamp_min(1e-6))
            target_index = data["memory_index"].clamp(min=0).to(torch.long)
            align_loss = -align_log_prob.gather(-1, target_index.unsqueeze(-1))
            losses["memory_align"] = self._masked_mean(align_loss, memory_mask.unsqueeze(-1))

            expert_value = self.value(expert_rl_feat)
            value_prior_loss = -expert_value.log_prob(memory_targets["raw_rtg"].to(torch.float32))
            losses["memory_value"] = self._masked_mean(value_prior_loss, memory_mask)

            pred_index = memory_readout["weights"].argmax(dim=-1)
            metrics["memory_progress_mae"] = self._masked_mean(
                (progress_pred - memory_targets["progress"]).abs(), memory_mask.unsqueeze(-1)
            )
            metrics["memory_align_acc"] = self._masked_mean((pred_index == target_index).to(torch.float32), memory_mask)
            metrics["memory_rtg"] = self._masked_mean(memory_targets["raw_rtg"], memory_mask.unsqueeze(-1))

        metrics["dyn_entropy"] = torch.mean(self.rssm.get_dist(prior_logit).entropy())
        metrics["rep_entropy"] = torch.mean(self.rssm.get_dist(post_logit).entropy())

        imag_source = {
            "post_stoch": post_stoch.detach(),
            "post_deter": post_deter.detach(),
            "kv_k": feat_dict["kv_k"].detach(),
            "kv_v": feat_dict["kv_v"].detach(),
            "valid_lens": torch.clamp(t_mask.sum(dim=1).to(torch.int64), min=1),
            "T": T,
        }
        return losses, metrics, imag_source

    def _actor_critic_forward(self, start_stoch, start_deter, imag_carry, imag_mask):
        """Single actor-critic forward pass from imagination starts."""
        losses = {}
        metrics = {}

        imag_feat, imag_action, imag_stoch, imag_deter = self._imagine(
            (start_stoch, start_deter), self.imag_horizon + 1, imag_carry
        )
        imag_feat = imag_feat.detach()
        imag_action = imag_action.detach()
        imag_stoch = imag_stoch.detach()
        imag_deter = imag_deter.detach()
        imag_rl_feat = self._get_rl_feat(imag_stoch, imag_deter)

        imag_reward = self._frozen_reward(imag_feat).mode()
        imag_cont = self._frozen_cont(imag_feat).mean
        imag_value = self._frozen_value(imag_rl_feat).mode()
        imag_slow_value = self._frozen_slow_value(imag_rl_feat).mode()
        disc = 1 - 1 / self.horizon

        # Expert potential-based reward shaping: F(s,s') = γ_eff·Φ(s') - Φ(s)
        # where Φ(s) = attention-weighted raw expert return-to-go and
        # γ_eff = imag_cont * disc matches the effective discount in _lambda_return.
        if self.expert_shaping_scale > 0 and self.memory is not None:
            imag_readout = self._read_memory(imag_deter, frozen=True)
            phi = imag_readout["raw_rtg"]  # (N, T_imag, 1), raw space
            gamma_eff = imag_cont[:, 1:] * disc  # (N, T_imag-1, 1)
            shaping = gamma_eff * phi[:, 1:] - phi[:, :-1]  # (N, T_imag-1, 1)
            # Apply shaping to reward at interior steps (aligns with _lambda_return)
            imag_reward = torch.cat(
                [
                    imag_reward[:, :1],
                    imag_reward[:, 1:] + self.expert_shaping_scale * shaping,
                ],
                dim=1,
            )

        weight = torch.cumprod(imag_cont * disc, dim=1)
        last = torch.zeros_like(imag_cont)
        term = 1 - imag_cont
        ret = self._lambda_return(last, term, imag_reward, imag_value, imag_value, disc, self.lamb)  # (N, T_imag-1, 1)
        ret_offset, ret_scale = self.return_ema(ret)
        adv = (ret - imag_value[:, :-1]) / ret_scale

        policy = self.actor(imag_rl_feat)
        logpi = policy.log_prob(imag_action)[:, :-1].unsqueeze(-1)
        entropy = policy.entropy()[:, :-1].unsqueeze(-1)
        policy_loss = weight[:, :-1].detach() * -(logpi * adv.detach() + self.act_entropy * entropy)
        losses["policy"] = self._masked_mean(policy_loss, imag_mask)

        imag_value_dist = self.value(imag_rl_feat)
        tar_padded = torch.cat([ret, 0 * ret[:, -1:]], 1)
        value_loss = weight[:, :-1].detach() * (
            -imag_value_dist.log_prob(tar_padded.detach()) - imag_value_dist.log_prob(imag_slow_value.detach())
        )[:, :-1].unsqueeze(-1)
        losses["value"] = self._masked_mean(value_loss, imag_mask)

        ret_normed = (ret - ret_offset) / ret_scale
        metrics["ret"] = torch.mean(ret_normed)
        metrics["ret_005"] = self.return_ema.ema_vals[0]
        metrics["ret_095"] = self.return_ema.ema_vals[1]
        metrics["adv"] = torch.mean(adv)
        metrics["adv_std"] = torch.std(adv)
        metrics["con"] = torch.mean(imag_cont)
        metrics["rew"] = torch.mean(imag_reward)
        if self.expert_shaping_scale > 0 and self.memory is not None:
            metrics["shaping"] = torch.mean(shaping)
        metrics["val"] = torch.mean(imag_value)
        metrics["tar"] = torch.mean(ret)
        metrics["slowval"] = torch.mean(imag_slow_value)
        metrics["weight"] = torch.mean(weight)
        metrics["action_entropy"] = torch.mean(entropy)
        metrics.update(tools.tensorstats(imag_action, "action"))
        return losses, metrics

    def _cal_grad(self, data):
        """Compute gradients for one batch.

        t_mask from data masks out zero-padded positions.

        Notes
        -----
        This function computes:
        1) World model loss (dynamics + representation)
        2) Optional representation loss variants (Dreamer, R2-Dreamer, InfoNCE, DreamerPro)
        3) Imagination rollouts for actor-critic updates with start-chunk
           gradient accumulation
        """
        metrics = {}
        losses = {}
        opt_loss = torch.zeros((), dtype=torch.float32, device=self.device)
        total_valid = data["t_mask"].float().sum().clamp_min(1.0)

        def _accum(target, source, weight):
            if not isinstance(weight, torch.Tensor):
                weight = torch.tensor(weight, dtype=torch.float32, device=self.device)
            else:
                weight = weight.to(dtype=torch.float32, device=self.device)
            for name, value in source.items():
                if not isinstance(value, torch.Tensor):
                    value = torch.tensor(value, dtype=torch.float32, device=self.device)
                target[name] = target.get(
                    name,
                    torch.zeros((), dtype=torch.float32, device=self.device),
                )
                target[name] = target[name] + value.detach() * weight

        for wm_data, wm_batch_weight in self._iter_batch_chunks(data, self.wm_accum_steps):
            wm_losses, wm_metrics, imag_source = self._world_model_forward(wm_data)
            wm_valid = wm_data["t_mask"].float().sum()
            wm_loss_weight = wm_valid / total_valid
            wm_batch_weight = torch.tensor(wm_batch_weight, dtype=torch.float32, device=self.device)

            world_model_loss = sum(self._loss_scales[name] * value for name, value in wm_losses.items())
            self._scaler.scale(world_model_loss * wm_loss_weight).backward()
            opt_loss = opt_loss + world_model_loss.detach() * wm_loss_weight
            _accum(losses, wm_losses, wm_loss_weight)
            _accum(metrics, wm_metrics, wm_batch_weight)

            starts = self._sample_transformer_imag_starts(
                imag_source["valid_lens"],
                imag_source["T"],
            )
            ac_losses = {}
            ac_metrics = {}
            for start_chunk, s_weight in self._iter_start_chunks(starts, self.ac_accum_steps):
                s_weight = torch.tensor(s_weight, dtype=torch.float32, device=self.device)
                s_stoch, s_deter, s_carry, s_mask = self._prepare_transformer_imag_start(
                    imag_source["post_stoch"],
                    imag_source["post_deter"],
                    imag_source["kv_k"],
                    imag_source["kv_v"],
                    start_chunk,
                )
                chunk_losses, chunk_metrics = self._actor_critic_forward(s_stoch, s_deter, s_carry, s_mask)
                ac_total = (
                    self._loss_scales["policy"] * chunk_losses["policy"]
                    + self._loss_scales["value"] * chunk_losses["value"]
                )
                grad_weight = wm_batch_weight * s_weight
                self._scaler.scale(ac_total * grad_weight).backward()
                opt_loss = opt_loss + ac_total.detach() * grad_weight
                _accum(ac_losses, chunk_losses, s_weight)
                _accum(ac_metrics, chunk_metrics, s_weight)
            _accum(losses, ac_losses, wm_batch_weight)
            _accum(metrics, ac_metrics, wm_batch_weight)

        metrics["wm_accum_steps"] = torch.tensor(
            float(self.wm_accum_steps),
            device=self.device,
        )
        metrics["ac_accum_steps"] = torch.tensor(
            float(self.ac_accum_steps),
            device=self.device,
        )
        metrics.update({f"loss/{name}": loss.detach() for name, loss in losses.items()})
        metrics.update({"opt/loss": opt_loss.detach()})
        return metrics

    @torch.no_grad()
    def _prepare_transformer_imag_start(
        self,
        post_stoch,
        post_deter,
        kv_k,
        kv_v,
        starts,
    ):
        """Prepare transformer imagination starts from trajectory KV tensors."""
        B = post_stoch.shape[0]
        K = starts.shape[1]

        # observe() already returns KV with W prepended dummy zero slots.
        start_stoch, start_deter, imag_carry = self._frozen_rssm.build_imag_starts(
            post_stoch, post_deter, kv_k, kv_v, starts
        )
        imag_mask = torch.ones((B * K, 1, 1), device=starts.device)
        return start_stoch, start_deter, imag_carry, imag_mask

    @torch.no_grad()
    def _sample_transformer_imag_starts(self, valid_lens, T):
        """Sample all imagination start indices for one actor-critic update."""
        K = min(self.imag_last if self.imag_last > 0 else T, T)
        return self._sample_valid_imag_starts(valid_lens, K, valid_lens.device)

    @torch.no_grad()
    def _sample_valid_imag_starts(self, valid_lens, K, device):
        """Sample K valid imagination starts independently per episode."""
        offsets = torch.arange(K, device=device)
        starts = []
        for valid_len_t in valid_lens:
            valid_len = int(valid_len_t.item())
            if valid_len >= K:
                start0 = torch.randint(0, valid_len - K + 1, (), device=device)
                starts.append(start0 + offsets)
            else:
                starts.append(torch.randint(0, valid_len, (K,), device=device))
        return torch.stack(starts, dim=0)

    @torch.no_grad()
    def _imagine(self, start, imag_horizon, imag_carry=None):
        """Roll out the policy in latent space."""
        assert imag_carry is not None
        # (B, S, K), (B, D)
        feats = []
        actions = []
        stoch_seq = []
        deter_seq = []
        stoch, deter = start
        for _ in range(imag_horizon):
            # (B, F)
            feat = self._frozen_rssm.get_feat(stoch, deter)
            rl_feat = self._get_rl_feat(stoch, deter, frozen=True)
            # (B, A)
            action = self._frozen_actor(rl_feat).rsample()
            # Append feat and its corresponding sampled action at the same time step.
            feats.append(feat)
            actions.append(action)
            stoch_seq.append(stoch)
            deter_seq.append(deter)
            stoch, deter, imag_carry = self._frozen_rssm.img_step_with_carry(stoch, imag_carry, action)

        # Stack along sequence dim T_imag.
        # (B, T_imag, F), (B, T_imag, A), (B, T_imag, S, K), (B, T_imag, D)
        return (
            torch.stack(feats, dim=1),
            torch.stack(actions, dim=1),
            torch.stack(stoch_seq, dim=1),
            torch.stack(deter_seq, dim=1),
        )

    @torch.no_grad()
    def _lambda_return(self, last, term, reward, value, boot, disc, lamb):
        """
        lamb=1 means discounted Monte Carlo return.
        lamb=0 means fixed 1-step return.
        """
        assert last.shape == term.shape == reward.shape == value.shape == boot.shape
        live = (1 - to_f32(term))[:, 1:] * disc
        cont = (1 - to_f32(last))[:, 1:] * lamb
        interm = reward[:, 1:] + (1 - cont) * live * boot[:, 1:]
        out = [boot[:, -1]]
        for i in reversed(range(live.shape[1])):
            out.append(interm[:, i] + live[:, i] * cont[:, i] * out[-1])
        return torch.stack(list(reversed(out))[:-1], 1)

    @torch.no_grad()
    def preprocess(self, data):
        if "image" in data:
            data["image"] = to_f32(data["image"]) / 255.0
        return data
