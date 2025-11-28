# convnext_tiny_gn.py
# ConvNeXt-Tiny with GroupNorm(1,C) (no NHWC permutes), PyTorch
# - Depths: [3, 3, 9, 3]
# - Dims:   [96, 192, 384, 768]
# - Stem:   4x4 Conv (stride 4)  -> (B, 96, H/4, W/4)
# - Block:  DW 7x7 -> GN(1,C) -> 1x1-MLP (4x expansion) with GELU -> layer scale -> residual
# - Downsample between stages: GN(1,C) -> 2x2 Conv (stride 2)

from __future__ import annotations
from typing import List, Tuple, Optional

import torch
import torch.nn as nn


# ----------------------------
# Utilities
# ----------------------------
class DropPath(nn.Module):
    """
    Stochastic depth per sample (per residual branch).
    """
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # (B,1,1,1) for NCHW
        mask = x.new_empty(shape).bernoulli_(keep_prob).div_(keep_prob)
        return x * mask


def layer_norm_2d(num_channels: int, eps: float = 1e-6) -> nn.GroupNorm:
    """
    GroupNorm with 1 group == LayerNorm over channels for NCHW tensors.
    Avoids NHWC permutes and is widely used as a fast LN surrogate in ConvNeXt.
    """
    return nn.GroupNorm(1, num_channels, eps=eps)


# ----------------------------
# ConvNeXt Block (GN variant)
# ----------------------------
class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt-style block (NCHW friendly):
      x -> DWConv(7x7, groups=C) -> GN(1,C) -> 1x1 Conv (C->4C) -> GELU -> 1x1 Conv (4C->C)
        -> layer scale gamma -> + residual -> DropPath
    """
    def __init__(
            self,
            dim: int,
            drop_path: float = 0.0,
            layer_scale_init_value: float = 1e-6,
    ):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = layer_norm_2d(dim, eps=1e-6)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)

        # Layer scale parameter gamma (broadcast over H,W)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(1, dim, 1, 1))
            if layer_scale_init_value > 0 else None
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)   # (B,C,H,W)
        x = self.norm(x)     # GN(1,C) on NCHW
        x = self.pwconv1(x)  # 1x1
        x = self.act(x)
        x = self.pwconv2(x)  # 1x1
        if self.gamma is not None:
            x = self.gamma * x
        x = shortcut + self.drop_path(x)
        return x


# ----------------------------
# Downsample layer (between stages)
# ----------------------------
class DownsampleLayer(nn.Module):
    """
    Downsample via: GN(1,C) -> 2x2 Conv (stride 2)
    Input:  (B, C_in, H, W)
    Output: (B, C_out, H/2, W/2)
    """
    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.norm = layer_norm_2d(c_in, eps=1e-6)
        self.reduction = nn.Conv2d(c_in, c_out, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.reduction(x)
        return x


# ----------------------------
# ConvNeXt Backbone + Head (GN variant)
# ----------------------------
class ConvNeXt(nn.Module):
    """
    Generic ConvNeXt (Tiny/Small/Base/Large) backbone + classifier head.
    - stem: 4x4 Conv (stride 4) + output C=dims[0]
    - stages: 4 stages with depths and dims you pass in
    - head: global average pool -> LayerNorm -> Linear(num_classes)
    """
    def __init__(
            self,
            in_chans: int = 3,
            num_classes: int = 1000,
            depths: Tuple[int, int, int, int] = (3, 3, 9, 3),
            dims: Tuple[int, int, int, int] = (96, 192, 384, 768),
            drop_path_rate: float = 0.1,
            layer_scale_init_value: float = 1e-6,
    ):
        super().__init__()

        # Stem
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
        )
        self.downsample_layers.append(stem)

        # Downsampling layers between stages (3 of them)
        for i in range(3):
            self.downsample_layers.append(DownsampleLayer(dims[i], dims[i + 1]))

        # Stochastic depth schedule
        total_blocks = sum(depths)
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
        dp_iter = 0

        # Stages
        self.stages = nn.ModuleList()
        for stage_idx in range(4):
            blocks = []
            for _ in range(depths[stage_idx]):
                blocks.append(
                    ConvNeXtBlock(
                        dim=dims[stage_idx],
                        drop_path=dp_rates[dp_iter],
                        layer_scale_init_value=layer_scale_init_value,
                    )
                )
                dp_iter += 1
            self.stages.append(nn.Sequential(*blocks))

        # Head (GAP -> LN -> FC)
        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # acts on (B,C)
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        # Kaiming for convs; truncated normal for linears; standard LN init
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02, a=-2.0, b=2.0)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W); H,W divisible by 32 recommended
        x = self.downsample_layers[0](x)  # stem
        x = self.stages[0](x)

        x = self.downsample_layers[1](x)
        x = self.stages[1](x)

        x = self.downsample_layers[2](x)
        x = self.stages[2](x)

        x = self.downsample_layers[3](x)
        x = self.stages[3](x)

        # GAP in NCHW, then LN over channels (B,C)
        x = x.mean(dim=(2, 3))  # (B, C)
        x = self.norm(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.head(x)
        return x

    # Optional: expose multi-scale features for downstream tasks
    def forward_features_multi(
            self, x: torch.Tensor, out_indices: Optional[List[int]] = None
    ):
        feats = []
        x = self.downsample_layers[0](x); x = self.stages[0](x); feats.append(x)
        x = self.downsample_layers[1](x); x = self.stages[1](x); feats.append(x)
        x = self.downsample_layers[2](x); x = self.stages[2](x); feats.append(x)
        x = self.downsample_layers[3](x); x = self.stages[3](x); feats.append(x)
        if out_indices is None:
            # classification path
            z = x.mean(dim=(2, 3))
            z = self.norm(z)
            return z
        return [feats[i] for i in out_indices]


# ----------------------------
# Factory: ConvNeXt-Tiny (GN variant)
# ----------------------------
def convnext_tiny(
        in_chans: int = 3,
        num_classes: int = 1000,
        drop_path_rate: float = 0.1,
        layer_scale_init_value: float = 1e-6,
) -> ConvNeXt:
    """
    ConvNeXt-Tiny:
      depths = [3, 3, 9, 3]
      dims   = [96, 192, 384, 768]
    """
    return ConvNeXt(
        in_chans=in_chans,
        num_classes=num_classes,
        depths=(3, 3, 9, 3),
        dims=(96, 192, 384, 768),
        drop_path_rate=drop_path_rate,
        layer_scale_init_value=layer_scale_init_value,
    )


# ----------------------------
# Quick self-test
# ----------------------------
if __name__ == "__main__":
    model = convnext_tiny(num_classes=1000)
    model.eval()
    # Optional throughput tip:
    # model.to(memory_format=torch.channels_last)
    x = torch.randn(2, 3, 224, 224)
    # x = x.contiguous(memory_format=torch.channels_last)
    with torch.no_grad():
        y = model(x)
    print("Output:", y.shape)  # (2, 1000)
