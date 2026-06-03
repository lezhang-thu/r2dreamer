import torch
from torch.optim import Optimizer


class DreamerV3Optimizer(Optimizer):
    """PyTorch counterpart of the official DreamerV3 JAX optimizer chain.

    The update order matches:
    AGC -> RMS scaling -> momentum -> optional weight decay -> learning rate.
    """

    def __init__(self,
                 params,
                 lr=4e-5,
                 agc=0.3,
                 pmin=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-20,
                 weight_decay=0.0,
                 nesterov=False):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if agc < 0:
            raise ValueError(f"Invalid AGC value: {agc}")
        if pmin < 0:
            raise ValueError(f"Invalid AGC pmin value: {pmin}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0 <= betas[0] < 1:
            raise ValueError(f"Invalid beta1 value: {betas[0]}")
        if not 0 <= betas[1] < 1:
            raise ValueError(f"Invalid beta2 value: {betas[1]}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight decay value: {weight_decay}")
        defaults = dict(lr=lr,
                        agc=agc,
                        pmin=pmin,
                        betas=betas,
                        eps=eps,
                        weight_decay=weight_decay,
                        nesterov=nesterov)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            agc = group["agc"]
            pmin = group["pmin"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]

            for param in group["params"]:
                if param.grad is None:
                    continue
                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError(
                        "DreamerV3Optimizer does not support sparse gradients")

                update = grad.detach().float()
                if agc:
                    update = self._clip_by_agc(update, param, agc, pmin)

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["nu"] = torch.zeros_like(
                        param,
                        dtype=torch.float32,
                        memory_format=torch.preserve_format,
                    )
                    state["mu"] = torch.zeros_like(
                        param,
                        dtype=torch.float32,
                        memory_format=torch.preserve_format,
                    )

                state["step"] += 1
                step = state["step"]
                nu = state["nu"]
                mu = state["mu"]

                nu.mul_(beta2).addcmul_(update, update, value=1 - beta2)
                nu_hat = nu / (1 - beta2**step)
                update = update / (nu_hat.sqrt() + eps)

                mu.mul_(beta1).add_(update, alpha=1 - beta1)
                if nesterov:
                    mu_nesterov = beta1 * mu + (1 - beta1) * update
                    update = mu_nesterov / (1 - beta1**step)
                else:
                    update = mu / (1 - beta1**step)

                if weight_decay:
                    update = update.add(param.detach().float(),
                                        alpha=weight_decay)
                param.add_(update, alpha=-lr)

        return loss

    @staticmethod
    def _clip_by_agc(update, param, clip, pmin):
        update_norm = torch.linalg.vector_norm(update.float())
        param_norm = torch.linalg.vector_norm(param.detach().float())
        max_norm = clip * torch.maximum(
            param_norm,
            torch.as_tensor(pmin, device=param.device, dtype=torch.float32))
        scale = torch.minimum(
            torch.ones((), device=param.device, dtype=torch.float32),
            max_norm / torch.clamp(update_norm, min=1e-12))
        return update * scale.to(dtype=update.dtype)
