from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def zero_module(module: nn.Module) -> nn.Module:
    for parameter in module.parameters():
        nn.init.zeros_(parameter)
    return module


def group_norm(channels: int, *, affine: bool = True) -> nn.GroupNorm:
    groups = min(32, channels)
    while channels % groups:
        groups -= 1
    return nn.GroupNorm(groups, channels, affine=affine)


def timestep_embedding(t: Tensor, dim: int, max_period: int = 10_000) -> Tensor:
    half = dim // 2
    frequencies = torch.exp(
        -math.log(max_period)
        * torch.arange(half, device=t.device, dtype=torch.float32)
        / max(half, 1)
    )
    arguments = t.float()[:, None] * frequencies[None]
    embedding = torch.cat((torch.cos(arguments), torch.sin(arguments)), dim=-1)
    if dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding.to(t.dtype)


class AdaptiveNorm(nn.Module):
    """AdaLN-style conditioning for convolutional feature maps.

    ResFlow specifies adaptive layer normalization for timestep and adapter
    conditioning, but does not publish the exact normalization or condition
    fusion formula.  This implementation uses GroupNorm by default and adds
    independently predicted timestep and auxiliary scale/shift parameters:

        (1 + gamma_t + gamma_y) * Norm(h) + beta_t + beta_y.

    This is an explicit implementation choice, not a claim about the authors'
    private implementation.
    """

    def __init__(
        self,
        channels: int,
        time_channels: int,
        *,
        auxiliary_channels: int | None = None,
        norm_type: str = "group",
    ) -> None:
        super().__init__()
        if norm_type != "group":
            raise ValueError(
                f"unsupported adaptive normalization {norm_type!r}; "
                "only the documented GroupNorm implementation is available"
            )
        self.channels = channels
        self.auxiliary_channels = auxiliary_channels
        self.norm = group_norm(channels, affine=False)
        self.time_to_scale_shift = nn.Sequential(
            nn.SiLU(), nn.Linear(time_channels, channels * 2)
        )
        self.auxiliary_to_scale_shift = (
            nn.Conv2d(auxiliary_channels, channels * 2, 1, bias=False)
            if auxiliary_channels is not None
            else None
        )

    def forward(
        self,
        x: Tensor,
        time: Tensor,
        auxiliary: Tensor | None = None,
    ) -> Tensor:
        time_scale, time_shift = self.time_to_scale_shift(time).chunk(2, dim=1)
        scale = time_scale[:, :, None, None]
        shift = time_shift[:, :, None, None]

        if auxiliary is not None:
            if self.auxiliary_to_scale_shift is None:
                raise ValueError("this AdaptiveNorm point does not accept adapter features")
            if auxiliary.shape[-2:] != x.shape[-2:]:
                raise ValueError(
                    "adapter/U-Net spatial mismatch: "
                    f"adapter={tuple(auxiliary.shape[-2:])}, U-Net={tuple(x.shape[-2:])}"
                )
            auxiliary_scale, auxiliary_shift = self.auxiliary_to_scale_shift(
                auxiliary
            ).chunk(2, dim=1)
            scale = scale + auxiliary_scale
            shift = shift + auxiliary_shift
        elif self.auxiliary_to_scale_shift is not None:
            # Omitting the zero-initialized adapter condition recovers the
            # timestep-conditioned U-Net backbone exactly.
            pass

        return (1.0 + scale) * self.norm(x) + shift


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
        dropout: float,
        *,
        auxiliary_channels: int | None = None,
        adaptive_norm_type: str = "group",
    ) -> None:
        super().__init__()
        self.in_layers = nn.Sequential(
            group_norm(in_channels), nn.SiLU(), nn.Conv2d(in_channels, out_channels, 3, padding=1)
        )
        self.adaptive_norm = AdaptiveNorm(
            out_channels,
            time_channels,
            auxiliary_channels=auxiliary_channels,
            norm_type=adaptive_norm_type,
        )
        self.out_layers = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(dropout),
            zero_module(nn.Conv2d(out_channels, out_channels, 3, padding=1)),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1)
        )

    def forward(
        self,
        x: Tensor,
        time: Tensor,
        auxiliary: Tensor | None = None,
    ) -> Tensor:
        h = self.in_layers(x)
        h = self.adaptive_norm(h, time, auxiliary)
        return self.skip(x) + self.out_layers(h)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.norm = group_norm(channels)
        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.proj = zero_module(nn.Conv1d(channels, channels, 1))

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, height, width = x.shape
        h = self.norm(x).reshape(batch, channels, height * width)
        q, k, v = self.qkv(h).chunk(3, dim=1)
        scale = channels ** -0.5
        weights = torch.einsum("bci,bcj->bij", q * scale, k)
        weights = weights.softmax(dim=-1)
        h = torch.einsum("bij,bcj->bci", weights, v)
        return x + self.proj(h).reshape(batch, channels, height, width)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class AdapterBlock(nn.Module):
    """Residual adapter block with a zero-initialized output, as in Appendix C."""

    def __init__(self, in_channels: int, out_channels: int, downsample: bool):
        super().__init__()
        stride = 2 if downsample else 1
        self.body = nn.Sequential(
            group_norm(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            group_norm(out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels and not downsample
            else nn.Conv2d(in_channels, out_channels, 1, stride=stride)
        )
        self.output = zero_module(nn.Conv2d(out_channels, out_channels, 1))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        state = self.skip(x) + self.body(x)
        return state, self.output(state)


class AuxiliaryAdapter(nn.Module):
    def __init__(self, image_channels: int, channels: Iterable[int]):
        super().__init__()
        channels = tuple(channels)
        if len(channels) != 4:
            raise ValueError(f"Figure 7 adapter requires four stages, got {len(channels)}")
        self.stem = nn.Conv2d(image_channels, channels[0], 3, padding=1)
        blocks = []
        previous = channels[0]
        for index, current in enumerate(channels):
            blocks.append(AdapterBlock(previous, current, downsample=index > 0))
            previous = current
        self.blocks = nn.ModuleList(blocks)

    def forward(self, y: Tensor) -> list[Tensor]:
        state = self.stem(y)
        outputs = []
        for block in self.blocks:
            state, output = block(state)
            outputs.append(output)
        return outputs


class ResFlowUNet(nn.Module):
    """DDPM 256 U-Net plus the paper's auxiliary adapter.

    Defaults follow Ho et al.'s official 256x256 LSUN U-Net: base width 128,
    channel multipliers (1,1,2,2,4,4), two residual blocks, attention at 16.
    """

    def __init__(
        self,
        image_channels: int = 3,
        base_channels: int = 128,
        channel_multipliers: tuple[int, ...] = (1, 1, 2, 2, 4, 4),
        num_res_blocks: int = 2,
        attention_resolutions: tuple[int, ...] = (16,),
        image_size: int = 256,
        dropout: float = 0.0,
        adaptive_norm_type: str = "group",
        timestep_scale: float = 1.0,
        predict_auxiliary_velocity: bool | None = None,
    ) -> None:
        super().__init__()
        if len(channel_multipliers) < 4:
            raise ValueError("ResFlow Figure 7 requires at least four U-Net resolution stages")
        if predict_auxiliary_velocity is False:
            raise ValueError("ResFlow must predict the complete augmented velocity [v_x, v_y]")
        self.image_channels = image_channels
        self.timestep_scale = float(timestep_scale)
        self.adapter_injection_levels = (0, 1, 2, 3)
        time_channels = base_channels * 4
        self.time_mlp = nn.Sequential(
            nn.Linear(base_channels, time_channels),
            nn.SiLU(),
            nn.Linear(time_channels, time_channels),
        )
        self.input_conv = nn.Conv2d(image_channels, base_channels, 3, padding=1)

        level_channels = [base_channels * multiplier for multiplier in channel_multipliers]
        self.adapter = AuxiliaryAdapter(image_channels, level_channels[:4])

        self.down_levels = nn.ModuleList()
        skip_channels = [base_channels]
        current = base_channels
        resolution = image_size
        for level, output_channels in enumerate(level_channels):
            blocks = nn.ModuleList()
            attentions = nn.ModuleList()
            for block_index in range(num_res_blocks):
                auxiliary_channels = (
                    level_channels[level]
                    if level in self.adapter_injection_levels
                    and block_index == num_res_blocks - 1
                    else None
                )
                blocks.append(
                    ResBlock(
                        current,
                        output_channels,
                        time_channels,
                        dropout,
                        auxiliary_channels=auxiliary_channels,
                        adaptive_norm_type=adaptive_norm_type,
                    )
                )
                current = output_channels
                attentions.append(
                    AttentionBlock(current)
                    if resolution in attention_resolutions
                    else nn.Identity()
                )
                skip_channels.append(current)
            downsample = None
            if level != len(level_channels) - 1:
                downsample = Downsample(current)
                skip_channels.append(current)
                resolution //= 2
            self.down_levels.append(nn.ModuleDict({
                "blocks": blocks,
                "attentions": attentions,
                "downsample": downsample or nn.Identity(),
            }))

        self.middle_block1 = ResBlock(
            current,
            current,
            time_channels,
            dropout,
            adaptive_norm_type=adaptive_norm_type,
        )
        self.middle_attention = AttentionBlock(current)
        self.middle_block2 = ResBlock(
            current,
            current,
            time_channels,
            dropout,
            adaptive_norm_type=adaptive_norm_type,
        )

        self.up_levels = nn.ModuleList()
        for level in reversed(range(len(level_channels))):
            output_channels = level_channels[level]
            blocks = nn.ModuleList()
            attentions = nn.ModuleList()
            level_resolution = image_size // (2**level)
            for _ in range(num_res_blocks + 1):
                skip = skip_channels.pop()
                blocks.append(
                    ResBlock(
                        current + skip,
                        output_channels,
                        time_channels,
                        dropout,
                        adaptive_norm_type=adaptive_norm_type,
                    )
                )
                current = output_channels
                attentions.append(
                    AttentionBlock(current)
                    if level_resolution in attention_resolutions
                    else nn.Identity()
                )
            upsample = Upsample(current) if level > 0 else nn.Identity()
            self.up_levels.append(nn.ModuleDict({
                "blocks": blocks,
                "attentions": attentions,
                "upsample": upsample,
            }))
        if skip_channels:
            raise RuntimeError("internal U-Net skip accounting error")

        output_channels = image_channels * 2
        self.output = nn.Sequential(
            group_norm(current),
            nn.SiLU(),
            zero_module(nn.Conv2d(current, output_channels, 3, padding=1)),
        )
        self.base_channels = base_channels

    def forward(self, x: Tensor, y: Tensor, t: Tensor) -> Tensor:
        time = self.time_mlp(
            timestep_embedding(t * self.timestep_scale, self.base_channels)
        )
        adapter_features = self.adapter(y)
        h = self.input_conv(x)
        skips = [h]
        for level, modules in enumerate(self.down_levels):
            condition = (
                adapter_features[level]
                if level in self.adapter_injection_levels
                else None
            )
            for block_index, (block, attention) in enumerate(
                zip(modules["blocks"], modules["attentions"])
            ):
                auxiliary = (
                    condition
                    if level in self.adapter_injection_levels
                    and block_index == len(modules["blocks"]) - 1
                    else None
                )
                h = attention(block(h, time, auxiliary))
                skips.append(h)
            if level != len(self.down_levels) - 1:
                h = modules["downsample"](h)
                skips.append(h)

        h = self.middle_block1(h, time)
        h = self.middle_attention(h)
        h = self.middle_block2(h, time)

        for up_index, modules in enumerate(self.up_levels):
            level = len(self.up_levels) - 1 - up_index
            for block, attention in zip(modules["blocks"], modules["attentions"]):
                h = torch.cat((h, skips.pop()), dim=1)
                h = block(h, time)
                h = attention(h)
            h = modules["upsample"](h)
        if skips:
            raise RuntimeError("not all U-Net skips were consumed")
        return self.output(h)


def build_model(config: dict) -> ResFlowUNet:
    values = dict(config)
    if "channel_multipliers" in values:
        values["channel_multipliers"] = tuple(values["channel_multipliers"])
    if "attention_resolutions" in values:
        values["attention_resolutions"] = tuple(values["attention_resolutions"])
    return ResFlowUNet(**values)
