import re
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

import distributions as dists
from tools import weight_init_


class LambdaLayer(nn.Module):
    """Wrap an arbitrary callable into an ``nn.Module``."""

    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BlockLinear(nn.Module):
    """Block-wise linear layer.

    Weight layout is chosen to cooperate with PyTorch's fan-in/fan-out
    calculation used by initializers.
    """

    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 blocks: int,
                 outscale: float = 1.0):
        super().__init__()
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.blocks = int(blocks)
        self.outscale = float(outscale)

        # Store weight in a layout that works with torch's fan calculation.
        # (O/G, I/G, G)
        self.weight = nn.Parameter(
            torch.empty(self.out_ch // self.blocks, self.in_ch // self.blocks,
                        self.blocks))
        self.bias = nn.Parameter(torch.empty(self.out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (..., I)
        batch_shape = x.shape[:-1]
        # Reshape to expose block dimension.
        # (..., I) -> (..., G, I/G)
        x = x.view(*batch_shape, self.blocks, self.in_ch // self.blocks)

        # Block-wise multiplication
        # (..., G, I/G), (O/G, I/G, G) -> (..., G, O/G)
        x = torch.einsum("...gi,oig->...go", x, self.weight)
        # Merge block dimension back.
        # (..., G, O/G) -> (..., O)
        x = x.reshape(*batch_shape, self.out_ch)
        return x + self.bias


class Conv2dSamePad(nn.Conv2d):
    """A Conv2d layer that emulates TensorFlow's 'SAME' padding."""

    def _calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        i_div_s_ceil = (i + s - 1) // s
        return max((i_div_s_ceil - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ih, iw = x.size()[-2:]
        pad_h = self._calc_same_pad(ih, self.kernel_size[0], self.stride[0],
                                    self.dilation[0])
        pad_w = self._calc_same_pad(iw, self.kernel_size[1], self.stride[1],
                                    self.dilation[1])

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x,
                [
                    pad_w // 2, pad_w - pad_w // 2, pad_h // 2,
                    pad_h - pad_h // 2
                ],
            )

        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class RMSNorm2D(nn.RMSNorm):
    """RMSNorm over channel-last format applied to 4D tensors."""

    def __init__(self, ch: int, eps: float = 1e-3, dtype=None):
        super().__init__(ch, eps=eps, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply RMSNorm over the channel dimension.
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


def _maybe_norm_2d(ch: int, use_norm: bool) -> nn.Module:
    if use_norm:
        return RMSNorm2D(ch, eps=1e-04, dtype=torch.float32)
    return nn.Identity()


class ResNetBasicBlock(nn.Module):
    """Small-image ResNet basic block with stride-based downsampling."""

    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 kernel_size: int,
                 act,
                 use_norm: bool,
                 stride: int = 1):
        super().__init__()
        bias = not use_norm
        self.conv1 = Conv2dSamePad(in_ch,
                                   out_ch,
                                   kernel_size,
                                   stride=stride,
                                   bias=bias)
        self.norm1 = _maybe_norm_2d(out_ch, use_norm)
        self.act1 = act()
        self.conv2 = Conv2dSamePad(out_ch,
                                   out_ch,
                                   kernel_size,
                                   stride=1,
                                   bias=bias)
        self.norm2 = _maybe_norm_2d(out_ch, use_norm)
        if stride != 1 or in_ch != out_ch:
            skip_layers = [
                Conv2dSamePad(in_ch, out_ch, 1, stride=stride, bias=bias)
            ]
            if use_norm:
                skip_layers.append(_maybe_norm_2d(out_ch, use_norm))
            self.skip = nn.Sequential(*skip_layers)
        else:
            self.skip = nn.Identity()
        self.out_act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return self.out_act(x + residual)


class MultiEncoder(nn.Module):

    def __init__(
        self,
        config,
        shapes,
    ):
        super().__init__()
        excluded = ("is_first", "is_last", "is_terminal", "reward")
        shapes = {
            k: v
            for k, v in shapes.items()
            if k not in excluded and not k.startswith("log_")
        }
        self.cnn_shapes = {
            k: v
            for k, v in shapes.items()
            if len(v) == 3 and re.match(config.cnn_keys, k)
        }
        self.mlp_shapes = {
            k: v
            for k, v in shapes.items()
            if len(v) in (1, 2) and re.match(config.mlp_keys, k)
        }
        print("Encoder CNN shapes:", self.cnn_shapes)
        print("Encoder MLP shapes:", self.mlp_shapes)

        self.out_dim = 0
        self.selectors = []
        self.encoders = []
        if self.cnn_shapes:
            input_ch = sum([v[-1] for v in self.cnn_shapes.values()])
            input_shape = tuple(self.cnn_shapes.values())[0][:2] + (input_ch,)
            self.encoders.append(ConvEncoder(config.cnn, input_shape))
            self.selectors.append(
                lambda obs: torch.cat([obs[k] for k in self.cnn_shapes], -1))
            self.out_dim += self.encoders[-1].out_dim
        if self.mlp_shapes:
            inp_dim = sum([sum(v) for v in self.mlp_shapes.values()])
            self.encoders.append(MLP(config.mlp, inp_dim))
            self.selectors.append(
                lambda obs: torch.cat([obs[k] for k in self.mlp_shapes], -1))
            self.out_dim += self.encoders[-1].out_dim
        self.encoders = nn.ModuleList(self.encoders)

        if len(self.encoders) > 1:
            self.fuser = lambda x: torch.cat(x, dim=-1)
        elif len(self.encoders) == 1:
            self.fuser = lambda x: x[0]
        else:
            raise NotImplementedError

        self.apply(weight_init_)

    def forward(self, obs):
        """Encode a dict of observations."""
        # dict of (B, T, *)
        return self.fuser(
            [enc(sel(obs)) for enc, sel in zip(self.encoders, self.selectors)])


class ConvEncoder(nn.Module):

    def __init__(self, config, input_shape):
        super().__init__()
        act = getattr(torch.nn, config.act)
        h, w, input_ch = input_shape
        self._minres = int(config.minres)
        self.depths = tuple(
            int(config.depth) * int(mult) for mult in list(config.mults))
        self.blocks = tuple(
            int(blocks)
            for blocks in getattr(config, "blocks", [1] * len(self.depths)))
        if len(self.blocks) != len(self.depths):
            raise AssertionError(
                "ConvEncoder config.blocks must match config.mults length "
                f"(got {len(self.blocks)} blocks for {len(self.depths)} stages)."
            )
        if any(blocks < 1 for blocks in self.blocks):
            raise AssertionError("ConvEncoder config.blocks must be >= 1.")
        kernel_size = int(config.kernel_size)
        stem_stride = int(getattr(config, "stem_stride", 2))
        if stem_stride < 1:
            raise AssertionError("ConvEncoder config.stem_stride must be >= 1.")
        use_norm = bool(config.norm)
        bias = not use_norm
        self.stem = nn.Sequential(
            Conv2dSamePad(input_ch,
                          self.depths[0],
                          kernel_size,
                          stride=stem_stride,
                          bias=bias),
            _maybe_norm_2d(self.depths[0], use_norm),
            act(),
        )
        h, w = self._downsample_dims(h, w, stem_stride)
        in_dim = self.depths[0]
        stages = []
        for stage_idx, (depth, block_count) in enumerate(
                zip(self.depths, self.blocks)):
            blocks = []
            for block_idx in range(block_count):
                stride = 1 if stage_idx == 0 or block_idx > 0 else 2
                blocks.append(
                    ResNetBasicBlock(in_dim,
                                     depth,
                                     kernel_size,
                                     act,
                                     use_norm,
                                     stride=stride))
                in_dim = depth
                h, w = self._downsample_dims(h, w, stride)
            stages.append(nn.Sequential(*blocks))
        if h < self._minres or w < self._minres:
            raise AssertionError(
                "ConvEncoder output resolution fell below config.minres "
                f"(got {(h, w)}, minres={self._minres}).")

        self.out_dim = self.depths[-1] * h * w
        self.layers = nn.Sequential(*stages)

    @staticmethod
    def _downsample_dims(h: int, w: int, stride: int):
        if stride == 1:
            return h, w
        return (h + stride - 1) // stride, (w + stride - 1) // stride

    def forward(self, obs):
        """Encode image-like observations with a CNN."""
        # (B, T, H, W, C)
        obs = obs - 0.5
        # (B*T, H, W, C)
        x = obs.reshape(-1, *obs.shape[-3:])
        # (B*T, C, H, W)
        x = x.permute(0, 3, 1, 2)
        x = self.stem(x)
        # (B*T, C_feat, H_feat, W_feat)
        x = self.layers(x)
        # (B*T, C_feat*H_feat*W_feat)
        x = x.reshape(x.shape[0], -1)
        # (B, T, C_feat*H_feat*W_feat)
        return x.reshape(*obs.shape[:-3], x.shape[-1])


class MLP(nn.Module):

    def __init__(
        self,
        config,
        inp_dim,
    ):
        super().__init__()
        act = getattr(torch.nn, config.act)
        self._symlog_inputs = bool(config.symlog_inputs)
        self._device = torch.device(config.device)
        self.layers = nn.Sequential()
        for i in range(config.layers):
            self.layers.add_module(f"{config.name}_linear{i}",
                                   nn.Linear(inp_dim, config.units, bias=True))
            self.layers.add_module(
                f"{config.name}_norm{i}",
                nn.RMSNorm(config.units, eps=1e-04, dtype=torch.float32))
            self.layers.add_module(f"{config.name}_act{i}", act())
            inp_dim = config.units
        self.out_dim = config.units

    def forward(self, x):
        # (B, T, I)
        if self._symlog_inputs:
            x = dists.symlog(x)
        # (B, T, U)
        return self.layers(x)


class MLPHead(nn.Module):

    def __init__(self, config, inp_dim):
        super().__init__()
        self.mlp = MLP(config, inp_dim)
        self._dist_name = str(config.dist.name)
        self._outscale = float(config.outscale)
        self._dist = getattr(dists, str(config.dist.name))

        if self._dist_name == "bounded_normal":
            self.last = nn.Linear(self.mlp.out_dim,
                                  config.shape[0] * 2,
                                  bias=True)
            kwargs = {
                "min_std": float(config.dist.min_std),
                "max_std": float(config.dist.max_std)
            }
        elif self._dist_name == "onehot":
            self.last = nn.Linear(self.mlp.out_dim, config.shape[0], bias=True)
            kwargs = {"unimix_ratio": float(config.dist.unimix_ratio)}
        elif self._dist_name == "multi_onehot":
            self.last = nn.Linear(self.mlp.out_dim,
                                  sum(config.shape),
                                  bias=True)
            kwargs = {
                "unimix_ratio": float(config.dist.unimix_ratio),
                "shape": tuple(config.shape)
            }
        elif self._dist_name == "symexp_twohot":
            self.last = nn.Linear(self.mlp.out_dim, config.shape[0], bias=True)
            kwargs = {
                "device": torch.device(config.device),
                "bin_num": int(config.dist.bin_num)
            }
        elif self._dist_name in ("binary", "identity"):
            self.last = nn.Linear(self.mlp.out_dim, config.shape[0], bias=True)
            kwargs = {}
        else:
            raise NotImplementedError

        self._dist = partial(self._dist, **kwargs)

        self.mlp.apply(weight_init_)
        self.last.apply(weight_init_)
        # apply explicit output scaling.
        if self._outscale != 1.0:
            with torch.no_grad():
                self.last.weight.mul_(self._outscale)

    def forward(self, x):
        """Produce a distribution head."""
        # (B, T, F)
        return self._dist(self.last(self.mlp(x)))


class Projector(nn.Module):

    def __init__(self, in_ch1, in_ch2):
        super().__init__()
        self.w = nn.Linear(in_ch1, in_ch2, bias=False)
        self.apply(weight_init_)

    def forward(self, x):
        return self.w(x)


class ReturnEMA(nn.Module):
    """running mean and std"""

    def __init__(self, device, alpha=1e-2):
        super().__init__()
        self.device = device
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95], device=device)
        self.register_buffer(
            "ema_vals", torch.zeros(2, dtype=torch.float32, device=self.device))

    def __call__(self, x):
        x_quantile = torch.quantile(torch.flatten(x.detach()), self.range)
        # Using out-of-place update for torch.compile compatibility
        self.ema_vals.copy_(self.alpha * x_quantile.detach() +
                            (1 - self.alpha) * self.ema_vals)
        scale = torch.clip(self.ema_vals[1] - self.ema_vals[0], min=1.0)
        offset = self.ema_vals[0]
        return offset.detach(), scale.detach()
