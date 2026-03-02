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
        self._use_transformer = str(getattr(config, 'dyn_type',
                                            'rssm')) == 'transformer'
        self.imag_last = int(getattr(config, 'imag_last', 0))

        # World model components
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
        self.encoder = networks.MultiEncoder(config.encoder, shapes)
        self.embed_size = self.encoder.out_dim
        if self._use_transformer:
            self.rssm = rssm.TransformerRSSM(
                config.transformer,
                self.embed_size,
                self.act_dim,
            )
        else:
            self.rssm = rssm.RSSM(
                config.rssm,
                self.embed_size,
                self.act_dim,
            )
        self.reward = networks.MLPHead(config.reward, self.rssm.feat_size)
        self.cont = networks.MLPHead(config.cont, self.rssm.feat_size)

        config.actor.shape = (act_space.n, ) if hasattr(
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
            self._cal_grad = torch.compile(self._cal_grad,
                                           mode="reduce-overhead")

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

        if self._use_transformer:
            # Two-phase KV-cache inference
            carry = {
                'kv_cache': state['kv_cache'],
                'pos': state['pos'],
                'h_prev': state['h_prev'],
            }
            # Trainer provides (B, 1, *) tensors; squeeze time dim
            embed_sq = embed.squeeze(1)  # (B, E)
            is_first = obs["is_first"].squeeze(1)  # (B,)
            # Phase 1: posterior from h_prev + tokens
            carry, stoch, h_prev = self._frozen_rssm.get_feat_step(
                carry, embed_sq, is_first)
            feat = self._frozen_rssm.get_feat(stoch, h_prev)
            action_dist = self._frozen_actor(feat)
            action = action_dist.mode if eval else action_dist.rsample()
            # Phase 2: update KV-cache with (tokens, action)
            carry = self._frozen_rssm.update_carry(carry, embed_sq, action,
                                                   is_first)
            return action, TensorDict(
                {
                    "kv_cache": carry['kv_cache'],
                    "pos": carry['pos'],
                    "h_prev": carry['h_prev'],
                    "prev_action": action,
                },
                batch_size=state.batch_size,
            )
        else:
            prev_stoch, prev_deter, prev_action = (
                state["stoch"],
                state["deter"],
                state["prev_action"],
            )
            # (B, S, K), (B, D)
            stoch, deter, _ = self._frozen_rssm.obs_step(
                prev_stoch, prev_deter, prev_action, embed, obs["is_first"])
            # (B, F)
            feat = self._frozen_rssm.get_feat(stoch, deter)
            action_dist = self._frozen_actor(feat)
            # (B, A)
            action = action_dist.mode if eval else action_dist.rsample()
            return action, TensorDict(
                {
                    "stoch": stoch,
                    "deter": deter,
                    "prev_action": action
                },
                batch_size=state.batch_size,
            )

    @torch.no_grad()
    def get_initial_state(self, B):
        if self._use_transformer:
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
                batch_size=(B, ))
        else:
            stoch, deter = self.rssm.initial(B)
            action = torch.zeros(B,
                                 self.act_dim,
                                 dtype=torch.float32,
                                 device=self.device)
            return TensorDict(
                {
                    "stoch": stoch,
                    "deter": deter,
                    "prev_action": action
                },
                batch_size=(B, ))

    @torch.no_grad()
    def get_initial_carry(self, B):
        """Return initial carry_train state for chunked replay training.

        Returns a tuple (stoch, deter, prev_action) analogous to carry_train
        in dreamerv3-jax/embodied/run/x_train.py.
        For transformer: carry_train is unused (complete episodes), but we
        return a compatible tuple for interface consistency.
        """
        stoch, deter = self.rssm.initial(B) if not self._use_transformer \
            else (torch.zeros(B, dtype=torch.float32, device=self.device),
                  torch.zeros(B, dtype=torch.float32, device=self.device))
        prev_action = torch.zeros(B,
                                  self.act_dim,
                                  dtype=torch.float32,
                                  device=self.device)
        return (stoch, deter, prev_action)

    def update(self, replay_buffer, carry_train):
        """Sample a batch from replay and perform one optimization step.

        Uses chunked replay (ReplayY) with carry_train state, analogous to
        dreamerv3-jax/embodied/run/x_train.py.

        Args:
            replay_buffer: ReplayY instance.
            carry_train: Tuple (stoch, deter, prev_action) from previous chunk.

        Returns:
            carry_train: Updated carry state for the next chunk.
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
            (stoch, deter), mets = self._cal_grad(p_data, carry_train)
        self._scaler.unscale_(self._optimizer)  # unscale grads in params
        if self._log_grads:
            old_params = [
                p.data.clone().detach() for p in self._named_params.values()
            ]
            grads = [
                p.grad for p in self._named_params.values()
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
        if self._use_transformer:
            # Transformer sees complete episodes; no carry state to propagate.
            new_carry = carry_train
        else:
            # Update carry_train with the endpoint latent state and last action.
            # clone() is needed because stoch/deter are outputs of the CUDAGraph
            # (torch.compile reduce-overhead); without cloning, the next graph
            # replay would overwrite the memory that carry_train points to.
            new_carry = (
                stoch[:, -1].detach().clone(),
                deter[:, -1].detach().clone(),
                data["action"][:, -1].detach().clone(),
            )
        return new_carry, metrics

    def _cal_grad(self, data, carry_train):
        """Compute gradients for one batch.

        Uses carry_train (stoch, deter, prev_action) from the previous chunk
        instead of storing initial latent states in the replay buffer.
        t_mask from data masks out zero-padded positions.

        Notes
        -----
        This function computes:
        1) World model loss (dynamics + representation)
        2) Optional representation loss variants (Dreamer, R2-Dreamer, InfoNCE, DreamerPro)
        3) Imagination rollouts for actor-critic updates
        4) Replay-based value learning
        """
        # t_mask: (B, T) bool — True for real data, False for padding
        t_mask = data["t_mask"].float()  # (B, T)

        losses = {}
        metrics = {}
        B, T = data.shape

        # === World model: posterior rollout and KL losses ===
        # (B, T, E)
        embed = self.encoder(data)

        if self._use_transformer:
            # Transformer path: current action, full-sequence observe
            action = data["action"]  # (B, T, A) — current action a_t
            entries, feat_dict = self.rssm.observe(embed, action,
                                                   data["is_first"])
            post_stoch = feat_dict['stoch']  # (B, T, S, K)
            post_deter = feat_dict['deter']  # (B, T, D) = h_prev
            post_logit = feat_dict['post_logit']  # (B, T, S, K)
            prior_logit = feat_dict['prior_logit']
            dyn_loss, rep_loss = self.rssm.kl_loss(post_logit, prior_logit,
                                                   self.kl_free)
            losses["dyn"] = (dyn_loss * t_mask).mean()
            losses["rep"] = (rep_loss * t_mask).mean()
            losses["align"] = (feat_dict['imag_core_loss'] * t_mask).mean()
        else:
            # GRU RSSM path: shifted prev_action, sequential observe
            carry_stoch, carry_deter, carry_prev_action = carry_train
            initial = (carry_stoch, carry_deter)
            action = torch.cat(
                [carry_prev_action.unsqueeze(1), data["action"][:, :-1]],
                dim=1)
            post_stoch, post_deter, post_logit = self.rssm.observe(
                embed, action, initial, data["is_first"])
            _, prior_logit = self.rssm.prior(post_deter)
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
        # log
        metrics["dyn_entropy"] = torch.mean(
            self.rssm.get_dist(prior_logit).entropy())
        metrics["rep_entropy"] = torch.mean(
            self.rssm.get_dist(post_logit).entropy())

        # === Imagination rollout for actor-critic ===
        if self._use_transformer:
            # Start from K contiguous positions at random offset
            K = min(self.imag_last if self.imag_last > 0 else T, T)
            s = torch.randint(0, T - K + 1, ()).item() if K < T else 0
            start = (
                post_stoch[:, s:s + K].reshape(B * K,
                                               *post_stoch.shape[2:]).detach(),
                post_deter[:, s:s + K].reshape(B * K,
                                               *post_deter.shape[2:]).detach(),
            )
            imag_feat, imag_action = self._imagine(start,
                                                   self.imag_horizon + 1)
            imag_feat, imag_action = imag_feat.detach(), imag_action.detach()
            imag_mask = t_mask[:, s:s + K].reshape(B * K, 1, 1)
        else:
            # (B*T, S, K), (B*T, D)
            start = (
                post_stoch.reshape(-1, *post_stoch.shape[2:]).detach(),
                post_deter.reshape(-1, *post_deter.shape[2:]).detach(),
            )
            imag_feat, imag_action = self._imagine(start,
                                                   self.imag_horizon + 1)
            imag_feat, imag_action = imag_feat.detach(), imag_action.detach()
            imag_mask = t_mask.reshape(B * T, 1, 1)

        # (N, T_imag, 1) where N = B*K (transformer) or B*T (rssm)
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
        # log
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

        if False:
            # === Replay-based value learning (keep gradients through world model) ===
            last, term, reward = (
                to_f32(data["is_last"]).unsqueeze(-1),
                to_f32(data["is_terminal"]).unsqueeze(-1),
                to_f32(data["reward"]).unsqueeze(-1),
            )
            feat = self.rssm.get_feat(post_stoch, post_deter)
            boot = ret[:, 0].reshape(B, T, 1)
            value = self._frozen_value(feat).mode()
            slow_value = self._frozen_slow_value(feat).mode()
            disc = 1 - 1 / self.horizon
            weight = 1.0 - last
            ret = self._lambda_return(last, term, reward, value, boot, disc,
                                      self.lamb)
            ret_padded = torch.cat([ret, 0 * ret[:, -1:]], 1)

            # Keep this attached to the world model so gradients can flow through
            value_dist = self.value(feat)
            repval_loss = (weight[:, :-1] *
                           (-value_dist.log_prob(ret_padded.detach()) -
                            value_dist.log_prob(
                                slow_value.detach()))[:, :-1].unsqueeze(-1))
            # Mask repval by t_mask[:, :-1] (repval has shape (B, T-1, 1))
            repval_mask = t_mask[:, :-1].unsqueeze(-1)  # (B, T-1, 1)
            losses["repval"] = (repval_loss * repval_mask).mean()
            # log
            metrics.update(tools.tensorstats(ret, "ret_replay"))
            metrics.update(tools.tensorstats(value, "value_replay"))
            metrics.update(tools.tensorstats(slow_value, "slow_value_replay"))

        total_loss = sum([v * self._loss_scales[k] for k, v in losses.items()])
        self._scaler.scale(total_loss).backward()

        metrics.update({f"loss/{name}": loss for name, loss in losses.items()})
        metrics.update({"opt/loss": total_loss})
        return (post_stoch, post_deter), metrics

    @torch.no_grad()
    def _imagine(self, start, imag_horizon):
        """Roll out the policy in latent space."""
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
            stoch, deter = self._frozen_rssm.img_step(stoch, deter, action)

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
