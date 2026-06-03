import copy
from collections import OrderedDict

import torch
from tensordict import TensorDict
from torch import nn
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

import networks
import rssm
import tools
from networks import Projector
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
        if str(config.rep_loss) != "r2dreamer":
            raise AssertionError("config.rep_loss must be 'r2dreamer' "
                                 f"(got {str(config.rep_loss)!r}).")

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
        self.rl_feat_size = self.rssm.feat_size
        self.actor = networks.MLPHead(config.actor, self.rl_feat_size)
        self.value = networks.MLPHead(config.critic, self.rl_feat_size)
        self.slow_target_update = int(config.slow_target_update)
        self.slow_target_fraction = float(config.slow_target_fraction)
        self._slow_value = copy.deepcopy(self.value)
        for param in self._slow_value.parameters():
            param.requires_grad = False
        self._slow_value_updates = 0
        self._train_carry = None

        self._loss_scales = dict(config.loss_scales)
        self._loss_scales.setdefault("repval", 0.3)
        self._log_grads = bool(config.log_grads)

        modules = {
            "rssm": self.rssm,
            "actor": self.actor,
            "value": self.value,
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

        self._grad_clip = float(getattr(config, "grad_clip", 1.0))
        self._optimizer = AdamW(
            self._adamw_param_groups(modules),
            lr=config.lr,
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
            print("Compiling forward loss function with torch.compile...")
            self._loss_forward = torch.compile(self._loss_forward,
                                               mode="default")

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

    @staticmethod
    def _norm_module_types():
        return (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.GroupNorm,
            nn.InstanceNorm1d,
            nn.InstanceNorm2d,
            nn.InstanceNorm3d,
            nn.LayerNorm,
            nn.RMSNorm,
            nn.SyncBatchNorm,
        )

    def _adamw_param_groups(self, modules):
        no_decay_ids = set()
        for module in modules.values():
            if isinstance(module, nn.Parameter):
                continue
            for submodule in module.modules():
                if isinstance(submodule, self._norm_module_types()):
                    no_decay_ids.update(
                        id(param)
                        for param in submodule.parameters(recurse=False))

        decay_params = []
        no_decay_params = []
        for name, param in self._named_params.items():
            if not param.requires_grad:
                continue
            if name.endswith(".bias") or id(param) in no_decay_ids:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        return [
            {
                "params": decay_params
            },
            {
                "params": no_decay_params,
                "weight_decay": 0.0
            },
        ]

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

    @staticmethod
    def _detach_carry(carry):
        return {k: v.detach() for k, v in carry.items()}

    def _ensure_train_carry(self, batch_size):
        if (self._train_carry is None or
                int(self._train_carry['pos'].shape[0]) != int(batch_size)):
            self._train_carry = self.rssm.initial_memory(int(batch_size))
        return self._train_carry

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        # Re-establish shared frozen weights after moving the model to a new device
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
        rl_feat = self._frozen_rssm.get_feat(stoch, h_prev)
        action_dist = self._frozen_actor(rl_feat)
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

    def update(self, replay_buffer, batch_size):
        """Sample a batch from replay and perform one optimization step.

        ReplayY returns Transformer-XL rows containing one real trainable
        segment. Detached memory is kept in self._train_carry.

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
        train_carry = self._ensure_train_carry(p_data.shape[0])

        self._update_slow_target()
        self._optimizer.zero_grad(set_to_none=True)
        metrics, next_train_carry = self._cal_grad(p_data, train_carry)

        self._scaler.unscale_(self._optimizer)
        if self._log_grads:
            grads = [
                p.grad
                for p in self._named_params.values()
                if p.grad is not None
            ]
            metrics["opt/grad_norm"] = tools.compute_global_norm(grads)
            metrics["opt/grad_rms"] = tools.compute_rms(grads)
        clip_grad_norm_(self._named_params.values(), self._grad_clip)
        self._scaler.step(self._optimizer)
        self._scaler.update()
        self._scheduler.step()
        self._optimizer.zero_grad(set_to_none=True)
        self._train_carry = self._detach_carry(next_train_carry)
        metrics["opt/lr"] = self._scheduler.get_last_lr()[0]
        metrics["opt/grad_scale"] = self._scaler.get_scale()
        return metrics

    @staticmethod
    def _barlow_loss(x1, x2, lambd, eps=1e-8):
        """Compute Barlow Twins loss over a dense segment batch."""
        x1_norm = (x1 - x1.mean(0)) / (x1.std(0) + eps)
        x2_norm = (x2 - x2.mean(0)) / (x2.std(0) + eps)

        c = torch.mm(x1_norm.T, x2_norm) / x1.shape[0]
        invariance_loss = (torch.diagonal(c) - 1.0).pow(2).sum()
        off_diag_mask = ~torch.eye(
            c.shape[0], dtype=torch.bool, device=c.device)
        redundancy_loss = c[off_diag_mask].pow(2).sum()
        return invariance_loss + lambd * redundancy_loss

    @staticmethod
    def _scalar_seq(x):
        x = to_f32(x)
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        return x

    def _world_model_forward(self, data, memory_carry):
        """World-model losses and detached cache for imagination updates."""
        positions = data["position"] if "position" in data.keys() else None
        B, T = data.shape

        losses = {}
        metrics = {}

        # === World model: posterior rollout and KL losses ===
        # (B, T, E)
        embed = self.encoder(data)

        # Transformer path: posterior from tokens, transition on (stoch, a_t)
        action = data["action"]  # (B, T, A) — current action a_t
        _, feat_dict = self.rssm.observe(embed,
                                         action,
                                         data["is_first"],
                                         positions=positions,
                                         memory_carry=memory_carry)
        post_stoch = feat_dict['stoch']  # (B, T, S, K)
        post_deter = feat_dict['deter']  # (B, T, D) = h_prev
        post_logit = feat_dict['post_logit']  # (B, T, S, K)
        prior_logit = feat_dict['prior_logit']
        dyn_loss, rep_loss = self.rssm.kl_loss(post_logit, prior_logit,
                                               self.kl_free)
        losses["dyn"] = dyn_loss.mean()
        losses["rep"] = rep_loss.mean()

        # === Representation / auxiliary losses ===
        # (B, T, F)
        feat = self.rssm.get_feat(post_stoch, post_deter)
        x1 = self.prj(feat.reshape(B * T, -1))
        x2 = embed.reshape(B * T, -1).detach()
        losses["barlow"] = self._barlow_loss(x1, x2, self.barlow_lambd)

        rew_loss = -self.reward(feat).log_prob(
            to_f32(data["reward"]).unsqueeze(-1))  # (B, T)
        losses["rew"] = rew_loss.mean()
        cont = (1.0 - to_f32(data["is_terminal"])).unsqueeze(-1)
        con_loss = -self.cont(feat).log_prob(cont)  # (B, T)
        losses["con"] = con_loss.mean()

        metrics["dyn_entropy"] = torch.mean(
            self.rssm.get_dist(prior_logit).entropy())
        metrics["rep_entropy"] = torch.mean(
            self.rssm.get_dist(post_logit).entropy())

        imag_source = {
            "post_stoch": post_stoch.detach(),
            "post_deter": post_deter.detach(),
            "kv_k": feat_dict["kv_k"].detach(),
            "kv_v": feat_dict["kv_v"].detach(),
            "positions": None if positions is None else positions.detach(),
            "feat": feat,
        }
        return losses, metrics, imag_source, feat_dict["next_carry"]

    def _actor_critic_forward(self, start_stoch, start_deter, imag_carry):
        """Single actor-critic forward pass from imagination starts."""
        losses = {}
        metrics = {}

        imag_feat, imag_action = self._imagine(
            (start_stoch, start_deter), self.imag_horizon + 1, imag_carry)
        imag_feat = imag_feat.detach()
        imag_action = imag_action.detach()

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
        losses["policy"] = policy_loss.mean()

        imag_value_dist = self.value(imag_feat)
        tar_padded = torch.cat([ret, 0 * ret[:, -1:]], 1)
        value_loss = (weight[:, :-1].detach() *
                      (-imag_value_dist.log_prob(tar_padded.detach()) -
                       imag_value_dist.log_prob(
                           imag_slow_value.detach()))[:, :-1].unsqueeze(-1))
        losses["value"] = value_loss.mean()

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
        return losses, metrics, ret[:, 0].detach()

    def _replay_value_forward(self, data, feat, boot):
        """Replay value loss with gradients kept through replay features."""
        B, T = data.shape
        last = self._scalar_seq(data["is_last"])
        term = self._scalar_seq(data["is_terminal"])
        reward = self._scalar_seq(data["reward"])
        boot = boot.reshape(B, T, 1)

        value = self._frozen_value(feat).mode()
        slow_value = self._frozen_slow_value(feat).mode()
        disc = 1 - 1 / self.horizon
        weight = 1.0 - last
        ret = self._lambda_return(last, term, reward, value, boot, disc,
                                  self.lamb)
        ret_padded = torch.cat([ret, 0 * ret[:, -1:]], 1)

        value_dist = self.value(feat)
        repval_loss = (
            weight[:, :-1] *
            (-value_dist.log_prob(ret_padded.detach()) -
             value_dist.log_prob(slow_value.detach()))[:, :-1].unsqueeze(-1))
        losses = {"repval": repval_loss.mean()}
        metrics = {}
        metrics.update(tools.tensorstats(ret, "ret_replay"))
        metrics.update(tools.tensorstats(value, "value_replay"))
        metrics.update(tools.tensorstats(slow_value, "slow_value_replay"))
        return losses, metrics

    def _loss_forward(self, data, train_carry):
        """Compute the joint world-model and actor-critic forward loss."""
        losses = {}
        metrics = {}

        with autocast(device_type=self.device.type, dtype=torch.float16):
            wm_losses, wm_metrics, imag_source, next_carry = self._world_model_forward(
                data, train_carry)

        s_stoch, s_deter, s_carry = self._frozen_rssm.build_imag_starts(
            imag_source["post_stoch"],
            imag_source["post_deter"],
            imag_source["kv_k"],
            imag_source["kv_v"],
            positions=imag_source["positions"],
        )
        with autocast(device_type=self.device.type, dtype=torch.float16):
            ac_losses, ac_metrics, replay_boot = self._actor_critic_forward(
                s_stoch, s_deter, s_carry)
            ac_total = (self._loss_scales["policy"] * ac_losses["policy"] +
                        self._loss_scales["value"] * ac_losses["value"])
            repval_losses, repval_metrics = self._replay_value_forward(
                data, imag_source["feat"], replay_boot)
            repval_total = (self._loss_scales["repval"] *
                            repval_losses["repval"])
        losses.update(ac_losses)
        losses.update(repval_losses)
        metrics.update(ac_metrics)
        metrics.update(repval_metrics)

        world_model_loss = sum(self._loss_scales[name] * value
                               for name, value in wm_losses.items())
        opt_loss = ac_total + repval_total + world_model_loss
        losses.update(wm_losses)
        metrics.update(wm_metrics)
        return opt_loss, losses, metrics, next_carry

    def _cal_grad(self, data, train_carry):
        """Compute gradients for one joint world-model and actor-critic update."""
        opt_loss, losses, metrics, next_carry = self._loss_forward(
            data, train_carry)
        self._scaler.scale(opt_loss).backward()

        metrics = {
            name: value.detach() if isinstance(value, torch.Tensor) else value
            for name, value in metrics.items()
        }

        metrics.update({
            f"loss/{name}": loss.detach() for name, loss in losses.items()
        })
        metrics["opt/loss"] = opt_loss.detach()
        return metrics, self._detach_carry(next_carry)

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
