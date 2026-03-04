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
from networks import Projector
from optim import LaProp, clip_grad_agc_
from tools import to_f32


class Dreamer(nn.Module):

    def __init__(self, config, obs_space, act_space):
        super().__init__()
        self.device = torch.device(config.device)
        self.act_entropy = float(config.act_entropy)
        self.kl_free = float(config.kl_free)
        self.imag_horizon = int(config.imag_horizon)
        self.horizon = int(config.horizon)
        self.lamb = float(config.lamb)
        self.return_ema = networks.ReturnEMA(device=self.device)
        self.act_dim = act_space.n if hasattr(act_space, "n") else sum(
            act_space.shape)
        self.rep_loss = str(config.rep_loss)
        self.imag_last = int(getattr(config, 'imag_last', 0))
        self.wm_accum_steps = max(1, int(getattr(config, 'wm_accum_steps', 1)))
        self.ac_accum_steps = max(1, int(getattr(config, 'ac_accum_steps', 1)))
        self.ac_repeats = max(1, int(getattr(config, 'ac_repeats', 1)))

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

        config.actor.shape = (act_space.n,) if hasattr(
            act_space, "n") else tuple(map(int, act_space.shape))
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
        self.actor = networks.MLPHead(config.actor, self.rssm.feat_size)
        self.value = networks.MLPHead(config.critic, self.rssm.feat_size)
        self.slow_target_update = int(config.slow_target_update)
        self.slow_target_fraction = float(config.slow_target_fraction)
        self._slow_value = copy.deepcopy(self.value)
        for param in self._slow_value.parameters():
            param.requires_grad = False
        self._slow_value_updates = 0

        self._loss_scales = dict(config.loss_scales)
        self._log_grads = bool(config.log_grads)

        modules = {
            "rssm": self.rssm,
            "actor": self.actor,
            "value": self.value,
            "reward": self.reward,
            "cont": self.cont,
            "encoder": self.encoder,
        }

        if self.rep_loss == "dreamer":
            self.decoder = networks.MultiDecoder(
                config.decoder,
                self.rssm._deter,
                self.rssm.flat_stoch,
                shapes,
            )
            recon = self._loss_scales.pop("recon")
            self._loss_scales.update({k: recon for k in self.decoder.all_keys})
            modules.update({"decoder": self.decoder})
        elif self.rep_loss == "r2dreamer":
            # add projector for latent to embedding
            self.prj = Projector(self.rssm.feat_size, self.embed_size)
            modules.update({"projector": self.prj})
            self.barlow_lambd = float(config.r2dreamer.lambd)
        # count number of parameters in each module
        for key, module in modules.items():
            if isinstance(module, nn.Parameter):
                print(f"{module.numel():>14,}: {key}")
            else:
                print(
                    f"{sum(p.numel() for p in module.parameters()):>14,}: {key}"
                )
        self._named_params = OrderedDict()
        for name, module in modules.items():
            if isinstance(module, nn.Parameter):
                self._named_params[name] = module
            else:
                for param_name, param in module.named_parameters():
                    self._named_params[f"{name}.{param_name}"] = param
        print(
            f"Optimizer has: {sum(p.numel() for p in self._named_params.values())} parameters."
        )

        def _agc(params):
            clip_grad_agc_(params,
                           float(config.agc),
                           float(config.pmin),
                           foreach=True)

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
                for v, s in zip(self.value.parameters(),
                                self._slow_value.parameters()):
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
        for name in ("encoder", "rssm", "reward", "cont", "actor", "value",
                     "slow_value"):
            setattr(
                self, f"_frozen_{name}",
                self._freeze_copy(
                    getattr(self,
                            f"_{name}" if name == "slow_value" else name)))

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        # Re-establish shared memory after moving the model to a new device
        self.clone_and_freeze()
        return self

    @torch.no_grad()
    def act(self, obs, state, eval=False):
        """Policy inference step."""
        torch.compiler.cudagraph_mark_step_begin()
        p_obs = self.preprocess(obs)
        # (B, E)
        embed = self._frozen_encoder(p_obs)

        # Two-phase KV-cache inference
        carry = {
            'kv_cache': state['kv_cache'],
            'pos': state['pos'],
            'h_prev': state['h_prev'],
        }
        # Trainer provides (B, 1, *) tensors; squeeze time dim
        embed_sq = embed.squeeze(1)  # (B, E)
        is_first = obs["is_first"].squeeze(1)  # (B,)
        # Phase 1: posterior from tokens
        carry, stoch, h_prev = self._frozen_rssm.get_feat_step(
            carry, embed_sq, is_first)
        feat = self._frozen_rssm.get_feat(stoch, h_prev)
        action_dist = self._frozen_actor(feat)
        action = action_dist.mode if eval else action_dist.rsample()
        # Phase 2: update KV-cache with (stoch, action)
        carry = self._frozen_rssm.update_carry(carry, stoch, action, is_first)
        return action, TensorDict(
            {
                "kv_cache": carry['kv_cache'],
                "pos": carry['pos'],
                "h_prev": carry['h_prev'],
                "prev_action": action,
            },
            batch_size=state.batch_size,
        )

    @torch.no_grad()
    def get_initial_state(self, B):
        carry = self.rssm.initial(B)
        action = torch.zeros(B,
                             self.act_dim,
                             dtype=torch.float32,
                             device=self.device)
        return TensorDict(
            {
                "kv_cache": carry['kv_cache'],
                "pos": carry['pos'],
                "h_prev": carry['h_prev'],
                "prev_action": action,
            },
            batch_size=(B,))

    @torch.no_grad()
    def get_initial_carry(self, B):
        """Return initial carry_train state for chunked replay training.

        carry_train is retained only for trainer API compatibility.
        """
        stoch = torch.zeros(B, dtype=torch.float32, device=self.device)
        deter = torch.zeros(B, dtype=torch.float32, device=self.device)
        prev_action = torch.zeros(B,
                                  self.act_dim,
                                  dtype=torch.float32,
                                  device=self.device)
        return (stoch, deter, prev_action)

    def update(self, replay_buffer, carry_train):
        """Sample a batch from replay and perform one optimization step.

        ReplayY provides complete trajectories (padded, no concatenation across
        trajectories). carry_train is kept for interface compatibility.

        Args:
            replay_buffer: ReplayY instance.
            carry_train: Placeholder tuple (unused).

        Returns:
            carry_train: Unchanged placeholder tuple.
            metrics: Dict of training metrics.
        """
        B = carry_train[0].shape[0]
        np_data = replay_buffer.sample(B)
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
            mets = self._cal_grad(p_data, carry_train)
        self._scaler.unscale_(self._optimizer)  # unscale grads in params
        if self._log_grads:
            old_params = [
                p.data.clone().detach() for p in self._named_params.values()
            ]
            grads = [
                p.grad
                for p in self._named_params.values()
                if p.grad is not None
            ]  # log grads before clipping
            grad_norm = tools.compute_global_norm(grads)
            grad_rms = tools.compute_rms(grads)
            mets["opt/grad_norm"] = grad_norm
            mets["opt/grad_rms"] = grad_rms
        self._agc(self._named_params.values())  # clipping
        self._scaler.step(self._optimizer)  # update params
        self._scaler.update()  # adjust scale
        self._scheduler.step()  # increment scheduler
        self._optimizer.zero_grad(set_to_none=True)  # reset grads
        mets["opt/lr"] = self._scheduler.get_lr()[0]
        mets["opt/grad_scale"] = self._scaler.get_scale()
        if self._log_grads:
            updates = [
                (new - old)
                for (new, old) in zip(self._named_params.values(), old_params)
            ]
            update_rms = tools.compute_rms(updates)
            params_rms = tools.compute_rms(self._named_params.values())
            mets["opt/param_rms"] = params_rms
            mets["opt/update_rms"] = update_rms
        metrics.update(mets)
        # Transformer sees complete episodes; no carry state to propagate.
        new_carry = carry_train
        return new_carry, metrics

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
            # Use native TensorDict slicing for compatibility with torch.compile.
            sub = data[start:end]
            yield sub, float(end - start) / float(B)

    def _iter_start_chunks(self, start_stoch, start_deter, imag_carry,
                           imag_mask, accum_steps):
        """Yield imagination-start chunks and their batch-fraction weights."""
        N = int(start_stoch.shape[0])
        splits = min(max(1, int(accum_steps)), max(1, N))
        if splits <= 1:
            yield start_stoch, start_deter, imag_carry, imag_mask, 1.0
            return
        chunk = (N + splits - 1) // splits
        for start in range(0, N, chunk):
            end = min(start + chunk, N)
            carry = {k: v[start:end] for k, v in imag_carry.items()}
            yield (
                start_stoch[start:end],
                start_deter[start:end],
                carry,
                imag_mask[start:end],
                float(end - start) / float(N),
            )

    def _world_model_forward(self, data):
        """World-model losses and detached cache for repeated AC updates."""
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
        post_stoch = feat_dict['stoch']  # (B, T, S, K)
        post_deter = feat_dict['deter']  # (B, T, D) = h_prev
        post_logit = feat_dict['post_logit']  # (B, T, S, K)
        prior_logit = feat_dict['prior_logit']
        dyn_loss, rep_loss = self.rssm.kl_loss(post_logit, prior_logit,
                                               self.kl_free)
        losses["dyn"] = (dyn_loss * t_mask).mean()
        losses["rep"] = (rep_loss * t_mask).mean()

        # === Representation / auxiliary losses ===
        # (B, T, F)
        feat = self.rssm.get_feat(post_stoch, post_deter)
        if self.rep_loss == "dreamer":
            recon_dists = self.decoder(post_stoch, post_deter)
            for key, dist in recon_dists.items():
                per_step = -dist.log_prob(data[key])  # (B, T, ...)
                # Reduce all dims except (B, T) then mask
                while per_step.dim() > 2:
                    per_step = per_step.sum(-1)
                losses[key] = (per_step * t_mask).mean()
        elif self.rep_loss == "r2dreamer":
            # R2-Dreamer: Barlow Twins redundancy reduction.
            # Zero out padded positions to keep static shapes for CUDAGraphs.
            flat_mask = t_mask.reshape(B * T, 1)  # (B*T, 1)
            x1 = self.prj(feat.reshape(B * T, -1)) * flat_mask
            x2 = embed.reshape(B * T, -1).detach() * flat_mask
            x1_norm = (x1 - x1.mean(0)) / (x1.std(0) + 1e-8)
            x2_norm = (x2 - x2.mean(0)) / (x2.std(0) + 1e-8)

            c = torch.mm(x1_norm.T, x2_norm) / (B * T)
            invariance_loss = (torch.diagonal(c) - 1.0).pow(2).sum()
            off_diag_mask = ~torch.eye(
                x1.shape[-1], dtype=torch.bool, device=x1.device)
            redundancy_loss = c[off_diag_mask].pow(2).sum()
            losses[
                "barlow"] = invariance_loss + self.barlow_lambd * redundancy_loss
        else:
            raise NotImplementedError

        # reward and continue — masked
        rew_loss = -self.reward(feat).log_prob(
            to_f32(data["reward"]).unsqueeze(-1))  # (B, T)
        losses["rew"] = (rew_loss * t_mask).mean()
        cont = (1.0 - to_f32(data["is_terminal"])).unsqueeze(-1)
        con_loss = -self.cont(feat).log_prob(cont)  # (B, T)
        losses["con"] = (con_loss * t_mask).mean()

        metrics["dyn_entropy"] = torch.mean(
            self.rssm.get_dist(prior_logit).entropy())
        metrics["rep_entropy"] = torch.mean(
            self.rssm.get_dist(post_logit).entropy())

        # Keep only detached tensors needed by repeated AC updates.
        imag_cache = {
            "post_stoch": post_stoch.detach(),
            "post_deter": post_deter.detach(),
            "kv_k": feat_dict["kv_k"],
            "kv_v": feat_dict["kv_v"],
            "t_mask": t_mask.detach(),
            "T": T,
        }
        return losses, metrics, imag_cache

    def _actor_critic_forward(self, start_stoch, start_deter, imag_carry,
                              imag_mask):
        """Single AC forward from a batch of imagination starts."""
        losses = {}
        metrics = {}

        imag_feat, imag_action = self._imagine(
            (start_stoch, start_deter), self.imag_horizon + 1, imag_carry)
        imag_feat, imag_action = imag_feat.detach(), imag_action.detach()

        # (N, T_imag, 1)
        imag_reward = self._frozen_reward(imag_feat).mode()
        imag_cont = self._frozen_cont(imag_feat).mean
        imag_value = self._frozen_value(imag_feat).mode()
        imag_slow_value = self._frozen_slow_value(imag_feat).mode()
        disc = 1 - 1 / self.horizon
        weight = torch.cumprod(imag_cont * disc, dim=1)
        last = torch.zeros_like(imag_cont)
        term = 1 - imag_cont
        ret = self._lambda_return(last, term, imag_reward, imag_value,
                                  imag_value, disc,
                                  self.lamb)  # (N, T_imag-1, 1)
        ret_offset, ret_scale = self.return_ema(ret)
        adv = (ret - imag_value[:, :-1]) / ret_scale

        policy = self.actor(imag_feat)
        logpi = policy.log_prob(imag_action)[:, :-1].unsqueeze(-1)
        entropy = policy.entropy()[:, :-1].unsqueeze(-1)
        policy_loss = weight[:, :-1].detach() * -(logpi * adv.detach() +
                                                  self.act_entropy * entropy)
        losses["policy"] = (policy_loss * imag_mask).mean()

        imag_value_dist = self.value(imag_feat)
        tar_padded = torch.cat([ret, 0 * ret[:, -1:]], 1)
        value_loss = (weight[:, :-1].detach() *
                      (-imag_value_dist.log_prob(tar_padded.detach()) -
                       imag_value_dist.log_prob(
                           imag_slow_value.detach()))[:, :-1].unsqueeze(-1))
        losses["value"] = (value_loss * imag_mask).mean()

        ret_normed = (ret - ret_offset) / ret_scale
        metrics["ret"] = torch.mean(ret_normed)
        metrics["ret_005"] = self.return_ema.ema_vals[0]
        metrics["ret_095"] = self.return_ema.ema_vals[1]
        metrics["adv"] = torch.mean(adv)
        metrics["adv_std"] = torch.std(adv)
        metrics["con"] = torch.mean(imag_cont)
        metrics["rew"] = torch.mean(imag_reward)
        metrics["val"] = torch.mean(imag_value)
        metrics["tar"] = torch.mean(ret)
        metrics["slowval"] = torch.mean(imag_slow_value)
        metrics["weight"] = torch.mean(weight)
        metrics["action_entropy"] = torch.mean(entropy)
        metrics.update(tools.tensorstats(imag_action, "action"))

        return losses, metrics

    def _cal_grad(self, data, _carry_train):
        """Compute gradients for one batch.

        carry_train is unused and kept only for trainer API compatibility.
        t_mask from data masks out zero-padded positions.

        Notes
        -----
        This function computes:
        1) World model loss (dynamics + representation)
        2) Optional representation loss variants (Dreamer, R2-Dreamer, InfoNCE, DreamerPro)
        3) Repeated imagination rollouts for actor-critic updates
        """
        metrics = {}
        losses = {}
        opt_loss = torch.zeros((), dtype=torch.float32, device=self.device)

        def _accum(target, source, weight):
            for name, value in source.items():
                if not isinstance(value, torch.Tensor):
                    value = torch.tensor(value,
                                         dtype=torch.float32,
                                         device=self.device)
                target[name] = target.get(
                    name,
                    torch.zeros((), dtype=torch.float32, device=self.device))
                target[name] = target[name] + value.detach() * float(weight)

        # Phase 1: world model (possibly chunked over batch) + backward.
        for wm_data, wm_weight in self._iter_batch_chunks(
                data, self.wm_accum_steps):
            wm_losses, wm_metrics, imag_cache = self._world_model_forward(
                wm_data)
            wm_total = sum(
                [self._loss_scales[k] * v for k, v in wm_losses.items()])
            self._scaler.scale(wm_total * wm_weight).backward()
            opt_loss = opt_loss + wm_total.detach() * wm_weight
            _accum(losses, wm_losses, wm_weight)
            _accum(metrics, wm_metrics, wm_weight)

            # Phase 2: repeated AC imagination from detached static WM cache.
            repeat_weight = wm_weight / float(self.ac_repeats)
            for _ in range(self.ac_repeats):
                start_stoch, start_deter, imag_carry, imag_mask = self._prepare_transformer_imag_start(
                    imag_cache["post_stoch"],
                    imag_cache["post_deter"],
                    imag_cache["kv_k"],
                    imag_cache["kv_v"],
                    imag_cache["t_mask"],
                    imag_cache["T"],
                )
                rep_losses = {}
                rep_metrics = {}
                for s_stoch, s_deter, s_carry, s_mask, s_weight in self._iter_start_chunks(
                        start_stoch, start_deter, imag_carry, imag_mask,
                        self.ac_accum_steps):
                    ac_losses, ac_metrics = self._actor_critic_forward(
                        s_stoch, s_deter, s_carry, s_mask)
                    ac_total = (
                        self._loss_scales["policy"] * ac_losses["policy"] +
                        self._loss_scales["value"] * ac_losses["value"])
                    grad_weight = wm_weight * s_weight
                    self._scaler.scale(ac_total * grad_weight).backward()
                    opt_loss = opt_loss + ac_total.detach() * grad_weight
                    _accum(rep_losses, ac_losses, s_weight)
                    _accum(rep_metrics, ac_metrics, s_weight)
                _accum(losses, rep_losses, repeat_weight)
                _accum(metrics, rep_metrics, repeat_weight)

        metrics.update({f"loss/{name}": loss for name, loss in losses.items()})
        metrics.update({"opt/loss": opt_loss})
        return metrics

    @torch.no_grad()
    def _prepare_transformer_imag_start(self, post_stoch, post_deter, kv_k,
                                        kv_v, t_mask, T):
        """Prepare transformer imagination starts from trajectory KV tensors."""
        B = post_stoch.shape[0]
        K = min(self.imag_last if self.imag_last > 0 else T, T)

        # Sample per-episode contiguous starts to increase imagination diversity.
        # start[b] is uniform in [0, eps_len[b]-K], clamped at 0.
        eps_len = t_mask.to(torch.int64).sum(dim=1)  # (B,)
        max_start = torch.clamp(eps_len - K, min=0)  # (B,)
        start = torch.floor(
            torch.rand(B, device=t_mask.device) *
            (max_start.to(torch.float32) + 1.0)).to(torch.int64)  # (B,)

        start_stoch, start_deter, imag_carry = self._frozen_rssm.build_imag_starts(
            post_stoch, post_deter, kv_k, kv_v, start, K)
        idx = start.unsqueeze(1) + torch.arange(
            K, device=t_mask.device).unsqueeze(0)  # (B, K)
        imag_mask = torch.gather(t_mask.to(torch.float32), 1,
                                 idx).reshape(B * K, 1, 1)
        return start_stoch, start_deter, imag_carry, imag_mask

    @torch.no_grad()
    def _imagine(self, start, imag_horizon, imag_carry=None):
        """Roll out the policy in latent space."""
        assert imag_carry is not None
        # (B, S, K), (B, D)
        feats = []
        actions = []
        stoch, deter = start
        for _ in range(imag_horizon):
            # (B, F)
            feat = self._frozen_rssm.get_feat(stoch, deter)
            # (B, A)
            action = self._frozen_actor(feat).rsample()
            # Append feat and its corresponding sampled action at the same time step.
            feats.append(feat)
            actions.append(action)
            stoch, deter, imag_carry = self._frozen_rssm.img_step_with_carry(
                stoch, imag_carry, action)

        # Stack along sequence dim T_imag.
        # (B, T_imag, F), (B, T_imag, A)
        return torch.stack(feats, dim=1), torch.stack(actions, dim=1)

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
