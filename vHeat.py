# vHeat.py
# ------------------------------------------------------------
# vHeat backbone with HCO-oriented redundancy-aware optimisation
#
# Paper:
#   Redundancy-Aware Frequency and Routing Optimisation
#   for Efficient Fine-Grained Pollen Recognition
#
# ------------------------------------------------------------

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from timm.layers import DropPath, trunc_normal_


DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


# ============================================================
# Basic layers
# ============================================================

class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = F.layer_norm(
            x,
            self.normalized_shape,
            self.weight,
            self.bias,
            self.eps
        )
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class to_channels_first(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 3, 1, 2).contiguous()


class to_channels_last(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 3, 1).contiguous()


def build_norm_layer(
    dim,
    norm_layer,
    in_format="channels_last",
    out_format="channels_last",
    eps=1e-6
):
    layers = []

    if norm_layer == "BN":
        if in_format == "channels_last":
            layers.append(to_channels_first())

        layers.append(nn.BatchNorm2d(dim))

        if out_format == "channels_last":
            layers.append(to_channels_last())

    elif norm_layer == "LN":
        if in_format == "channels_first":
            layers.append(to_channels_last())

        layers.append(nn.LayerNorm(dim, eps=eps))

        if out_format == "channels_first":
            layers.append(to_channels_first())

    else:
        raise NotImplementedError(
            f"build_norm_layer does not support {norm_layer}"
        )

    return nn.Sequential(*layers)


def build_act_layer(act_layer):
    if act_layer == "ReLU":
        return nn.ReLU(inplace=True)
    elif act_layer == "SiLU":
        return nn.SiLU(inplace=True)
    elif act_layer == "GELU":
        return nn.GELU()

    raise NotImplementedError(
        f"build_act_layer does not support {act_layer}"
    )


class StemLayer(nn.Module):
    """
    Stem layer of vHeat.
    """

    def __init__(
        self,
        in_chans=3,
        out_chans=96,
        act_layer="GELU",
        norm_layer="LN"
    ):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_chans,
            out_chans // 2,
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.norm1 = build_norm_layer(
            out_chans // 2,
            norm_layer,
            "channels_first",
            "channels_first"
        )

        self.act = build_act_layer(act_layer)

        self.conv2 = nn.Conv2d(
            out_chans // 2,
            out_chans,
            kernel_size=3,
            stride=2,
            padding=1
        )
        self.norm2 = build_norm_layer(
            out_chans,
            norm_layer,
            "channels_first",
            "channels_first"
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.norm2(x)

        return x


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        channels_first=False
    ):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = (
            partial(nn.Conv2d, kernel_size=1, padding=0)
            if channels_first else nn.Linear
        )

        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.drop(x)

        return x


# ============================================================
# Heat2D / Heat Conduction Operator
# ============================================================

class Heat2D(nn.Module):
    """
    Heat2D core operator.

    This is the HCO module used in the paper. The original heat-conduction
    computation is preserved. The optional frequency mask corresponds to
    Frequency-domain Regularization (FDR) in the paper.

    Note:
        The internal variable names `apply_freq_mask` and `freq_keep_ratio`
        are kept for compatibility with the original experimental code.
    """

    def __init__(
        self,
        infer_mode=False,
        res=14,
        dim=96,
        hidden_dim=96,
        **kwargs
    ):
        super().__init__()

        self.res = res

        self.dwconv = nn.Conv2d(
            dim,
            hidden_dim,
            kernel_size=3,
            padding=1,
            groups=hidden_dim
        )

        self.hidden_dim = hidden_dim

        self.linear = nn.Linear(
            hidden_dim,
            2 * hidden_dim,
            bias=True
        )

        self.out_norm = nn.LayerNorm(hidden_dim)

        self.out_linear = nn.Linear(
            hidden_dim,
            hidden_dim,
            bias=True
        )

        self.infer_mode = infer_mode

        self.to_k = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
        )

        # Frequency-response statistics for redundancy analysis.
        self.collect_freq_stats = False
        self.freq_retention_sum = None
        self.freq_count = 0

        # Frequency-domain Regularization, FDR.
        self.apply_freq_mask = False
        self.freq_keep_ratio = 1.0

    def infer_init_heat2d(self, freq):
        weight_exp = self.get_decay_map(
            (self.res, self.res),
            device=freq.device
        )

        self.k_exp = nn.Parameter(
            torch.pow(weight_exp[:, :, None], self.to_k(freq)),
            requires_grad=False
        )

        self.infer_mode = True
        del self.to_k

    @staticmethod
    def get_cos_map(
        N=224,
        device=torch.device("cpu"),
        dtype=torch.float
    ):
        weight_x = (
            torch.linspace(
                0,
                N - 1,
                N,
                device=device,
                dtype=dtype
            ).view(1, -1) + 0.5
        ) / N

        weight_n = torch.linspace(
            0,
            N - 1,
            N,
            device=device,
            dtype=dtype
        ).view(-1, 1)

        weight = torch.cos(weight_n * weight_x * torch.pi) * math.sqrt(2 / N)
        weight[0, :] = weight[0, :] / math.sqrt(2)

        return weight

    @staticmethod
    def get_decay_map(
        resolution=(224, 224),
        device=torch.device("cpu"),
        dtype=torch.float
    ):
        resh, resw = resolution

        weight_n = torch.linspace(
            0,
            torch.pi,
            resh + 1,
            device=device,
            dtype=dtype
        )[:resh].view(-1, 1)

        weight_m = torch.linspace(
            0,
            torch.pi,
            resw + 1,
            device=device,
            dtype=dtype
        )[:resw].view(1, -1)

        weight = torch.pow(weight_n, 2) + torch.pow(weight_m, 2)
        weight = torch.exp(-weight)

        return weight

    def _collect_freq_retention(
        self,
        freq_before: torch.Tensor,
        freq_after: torch.Tensor
    ):
        """
        Collect frequency-response retention statistics.

        Args:
            freq_before: [B, Hf, Wf, C]
            freq_after:  [B, Hf, Wf, C]

        retention(h, w) =
            mean_{b,c} |A'(b,h,w,c)| / (|A(b,h,w,c)| + eps)
        """

        if not self.collect_freq_stats:
            return

        with torch.no_grad():
            mag_before = torch.abs(freq_before).float()
            mag_after = torch.abs(freq_after).float()

            retention = mag_after / (mag_before + 1e-6)
            retention = retention.mean(dim=(0, 3))

            retention = retention.detach().cpu()

            if self.freq_retention_sum is None:
                self.freq_retention_sum = retention
            else:
                self.freq_retention_sum += retention

            self.freq_count += 1

    def build_lowpass_mask(
        self,
        H,
        W,
        keep_ratio,
        device,
        dtype
    ):
        """
        Build a low-frequency retention mask for FDR.

        keep_ratio = 1.0:
            keep all frequency responses.

        keep_ratio = 0.7:
            keep the top-left 70% x 70% low-frequency region.
        """

        if isinstance(keep_ratio, torch.Tensor):
            keep_ratio = keep_ratio.item()

        keep_h = max(
            1,
            int(torch.round(torch.tensor(H) * keep_ratio).item())
        )
        keep_w = max(
            1,
            int(torch.round(torch.tensor(W) * keep_ratio).item())
        )

        mask = torch.zeros(
            H,
            W,
            1,
            device=device,
            dtype=dtype
        )
        mask[:keep_h, :keep_w, :] = 1.0

        return mask

    def forward(self, x: torch.Tensor, freq_embed=None):
        B, C, H, W = x.shape

        x = self.dwconv(x)

        x = self.linear(
            x.permute(0, 2, 3, 1).contiguous()
        )
        x, z = x.chunk(chunks=2, dim=-1)

        if (
            ((H, W) == getattr(self, "__RES__", (0, 0)))
            and (getattr(self, "__WEIGHT_COSN__", None).device == x.device)
        ):
            weight_cosn = getattr(self, "__WEIGHT_COSN__", None)
            weight_cosm = getattr(self, "__WEIGHT_COSM__", None)
            weight_exp = getattr(self, "__WEIGHT_EXP__", None)
        else:
            weight_cosn = self.get_cos_map(
                H,
                device=x.device
            ).detach_()

            weight_cosm = self.get_cos_map(
                W,
                device=x.device
            ).detach_()

            weight_exp = self.get_decay_map(
                (H, W),
                device=x.device
            ).detach_()

            setattr(self, "__RES__", (H, W))
            setattr(self, "__WEIGHT_COSN__", weight_cosn)
            setattr(self, "__WEIGHT_COSM__", weight_cosm)
            setattr(self, "__WEIGHT_EXP__", weight_exp)

        N, M = weight_cosn.shape[0], weight_cosm.shape[0]

        x = F.conv1d(
            x.contiguous().view(B, H, -1),
            weight_cosn.contiguous().view(N, H, 1)
        )

        x = F.conv1d(
            x.contiguous().view(-1, W, C),
            weight_cosm.contiguous().view(M, W, 1)
        ).contiguous().view(B, N, M, -1)

        freq_before = x

        if self.infer_mode and hasattr(self, "k_exp"):
            freq_after = torch.einsum(
                "bhwc,hwc->bhwc",
                freq_before,
                self.k_exp
            )
        else:
            if not hasattr(self, "to_k"):
                raise RuntimeError("Heat2D: 'to_k' not found")

            k = self.to_k(freq_embed)
            weight_exp = torch.pow(weight_exp[:, :, None], k)

            freq_after = torch.einsum(
                "bhwc,hwc->bhwc",
                freq_before,
                weight_exp
            )

        # Frequency-domain Regularization, FDR.
        if self.apply_freq_mask and self.freq_keep_ratio < 1.0:
            freq_mask = self.build_lowpass_mask(
                freq_after.shape[1],
                freq_after.shape[2],
                self.freq_keep_ratio,
                freq_after.device,
                freq_after.dtype,
            )
            freq_after = freq_after * freq_mask

        self._collect_freq_retention(freq_before, freq_after)

        x = freq_after

        x = F.conv1d(
            x.contiguous().view(B, N, -1),
            weight_cosn.t().contiguous().view(H, N, 1)
        )

        x = F.conv1d(
            x.contiguous().view(-1, M, C),
            weight_cosm.t().contiguous().view(W, M, 1)
        ).contiguous().view(B, H, W, -1)

        x = self.out_norm(x)
        x = x * F.silu(z)
        x = self.out_linear(x)

        x = x.permute(0, 3, 1, 2).contiguous()

        return x


# ============================================================
# HeatBlock with PHR and FDR
# ============================================================

class HeatBlock(nn.Module):
    """
    vHeat block with optional Partial Heat Routing and FDR.

    The implementation follows the original experimental code:
        - channels are split into heavy and light branches when PHR is enabled;
        - the heavy branch uses Heat2D / HCO;
        - the light branch uses a lightweight local operation;
        - FDR is injected into the Heat2D operator through freq_keep_ratio.
    """

    def __init__(
        self,
        res=14,
        infer_mode=False,
        hidden_dim=96,
        drop_path=0.0,
        norm_layer=LayerNorm2d,
        use_checkpoint=False,
        mlp_ratio=4.0,
        post_norm=True,
        layer_scale=None,

        # PHR.
        partial_heat=False,
        heat_ratio=1.0,
        light_branch_type="dwconv",

        # FDR.
        freq_pruning=False,
        freq_keep_ratio=1.0
    ):
        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.norm1 = norm_layer(hidden_dim)

        # Partial Heat Routing, PHR.
        self.partial_heat = partial_heat
        self.heat_ratio = heat_ratio
        self.light_branch_type = light_branch_type
        self.hidden_dim = hidden_dim

        # Frequency-domain Regularization, FDR.
        # The variable name freq_pruning is kept for compatibility.
        self.freq_pruning = freq_pruning
        self.freq_keep_ratio = freq_keep_ratio

        if not self.partial_heat or self.heat_ratio >= 1.0:
            self.heavy_dim = hidden_dim
            self.light_dim = 0
        else:
            self.heavy_dim = max(
                1,
                int(round(hidden_dim * heat_ratio))
            )
            self.light_dim = hidden_dim - self.heavy_dim

        self.op = Heat2D(
            res=res,
            dim=self.heavy_dim,
            hidden_dim=self.heavy_dim,
            infer_mode=infer_mode
        )

        self.op.apply_freq_mask = bool(
            self.freq_pruning and self.freq_keep_ratio < 1.0
        )
        self.op.freq_keep_ratio = float(self.freq_keep_ratio)

        if self.light_dim > 0:
            if self.light_branch_type == "identity":
                self.light_branch = nn.Identity()

            elif self.light_branch_type == "dwconv":
                self.light_branch = nn.Sequential(
                    nn.Conv2d(
                        self.light_dim,
                        self.light_dim,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        groups=self.light_dim,
                        bias=True
                    ),
                    nn.GELU()
                )

            else:
                raise ValueError(
                    f"Unsupported light_branch_type: {self.light_branch_type}"
                )
        else:
            self.light_branch = None

        self.drop_path = (
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

        self.mlp_branch = mlp_ratio > 0.0

        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)

            self.mlp = Mlp(
                in_features=hidden_dim,
                hidden_features=mlp_hidden_dim,
                act_layer=nn.GELU,
                drop=0.0,
                channels_first=True
            )

        self.post_norm = post_norm
        self.layer_scale = layer_scale is not None

        if self.layer_scale:
            self.gamma1 = nn.Parameter(
                layer_scale * torch.ones(hidden_dim),
                requires_grad=True
            )
            self.gamma2 = nn.Parameter(
                layer_scale * torch.ones(hidden_dim),
                requires_grad=True
            )

        # Channel-path redundancy analysis.
        self.collect_delta = False
        self.delta_sum = None
        self.delta_count = 0

        self.channel_mask = None
        self.apply_channel_mask = False

    def _forward(self, x: torch.Tensor, freq_embed):
        def _collect_channel_delta(
            op_in: torch.Tensor,
            op_out: torch.Tensor
        ):
            """
            Collect per-channel relative changes for redundancy analysis.

            op_in, op_out: [B, C, H, W]

            delta_c =
                ||op_out_c - op_in_c||_2 / (||op_in_c||_2 + eps)
            """

            if not self.collect_delta:
                return

            with torch.no_grad():
                diff = op_out - op_in

                diff_norm = torch.sqrt(
                    torch.sum(diff.float() * diff.float(), dim=(2, 3))
                )
                in_norm = torch.sqrt(
                    torch.sum(op_in.float() * op_in.float(), dim=(2, 3))
                )

                delta = diff_norm / (in_norm + 1e-6)
                delta = delta.sum(dim=0)

                delta = delta.detach().cpu()

                if self.delta_sum is None:
                    self.delta_sum = delta
                else:
                    self.delta_sum += delta

                self.delta_count += op_in.shape[0]

        # Partial Heat Routing.
        if self.light_dim == 0:
            heavy_in = x
            light_in = None
        else:
            heavy_in = x[:, :self.heavy_dim, :, :]
            light_in = x[:, self.heavy_dim:, :, :]

        if freq_embed is None:
            heavy_freq_embed = None
        elif self.light_dim == 0:
            heavy_freq_embed = freq_embed
        else:
            heavy_freq_embed = freq_embed[:, :, :self.heavy_dim]

        heavy_out = self.op(heavy_in, heavy_freq_embed)

        # Channel-path redundancy statistics are collected on the HCO branch.
        _collect_channel_delta(heavy_in, heavy_out)

        if self.apply_channel_mask and self.channel_mask is not None:
            mask = self.channel_mask.to(
                device=heavy_out.device,
                dtype=heavy_out.dtype
            )

            if mask.shape[0] == heavy_out.shape[1]:
                heavy_out = heavy_out * mask

        if self.light_dim > 0:
            light_out = self.light_branch(light_in)
            raw_op_out = torch.cat([heavy_out, light_out], dim=1)
        else:
            raw_op_out = heavy_out

        if not self.layer_scale:
            hco_out = raw_op_out
        else:
            hco_out = self.gamma1[:, None, None] * raw_op_out

        if self.post_norm:
            hco_out = self.norm1(hco_out)

        x = x + self.drop_path(hco_out)

        # MLP branch.
        if self.mlp_branch:
            if not self.layer_scale:
                mlp_out = self.mlp(x)
            else:
                mlp_out = self.gamma2[:, None, None] * self.mlp(x)

            if self.post_norm:
                mlp_out = self.norm2(mlp_out)

            x = x + self.drop_path(mlp_out)

        return x

    def forward(self, input: torch.Tensor, freq_embed=None):
        if self.use_checkpoint:
            return checkpoint.checkpoint(
                self._forward,
                input,
                freq_embed
            )
        else:
            return self._forward(input, freq_embed)


# ============================================================
# Redundancy analysis helper functions
# ============================================================

def enable_heat_delta_collection(model):
    """
    Enable channel-path redundancy statistics for all HeatBlock modules.
    """

    for m in model.modules():
        if isinstance(m, HeatBlock):
            m.collect_delta = True
            m.delta_sum = None
            m.delta_count = 0


def disable_heat_delta_collection(model):
    """
    Disable channel-path redundancy statistics.
    """

    for m in model.modules():
        if isinstance(m, HeatBlock):
            m.collect_delta = False


def get_heat_delta_results(model):
    """
    Return average channel delta for each HeatBlock.

    Returns:
        {
            "block_0": tensor[C],
            "block_1": tensor[C],
            ...
        }
    """

    results = {}
    block_idx = 0

    for m in model.modules():
        if isinstance(m, HeatBlock):
            if m.delta_sum is not None and m.delta_count > 0:
                results[f"block_{block_idx}"] = m.delta_sum / m.delta_count
            else:
                results[f"block_{block_idx}"] = None

            block_idx += 1

    return results


def enable_freq_stats(model):
    """
    Enable frequency-response redundancy statistics for all Heat2D modules.
    """

    for m in model.modules():
        if isinstance(m, Heat2D):
            m.collect_freq_stats = True
            m.freq_retention_sum = None
            m.freq_count = 0


def disable_freq_stats(model):
    """
    Disable frequency-response redundancy statistics.
    """

    for m in model.modules():
        if isinstance(m, Heat2D):
            m.collect_freq_stats = False


def get_freq_stats_results(model):
    """
    Return average frequency-response retention matrix for each Heat2D block.

    Returns:
        {
            "block_0": tensor[Hf, Wf],
            "block_1": tensor[Hf, Wf],
            ...
        }
    """

    results = {}
    block_idx = 0

    for m in model.modules():
        if isinstance(m, Heat2D):
            if m.freq_retention_sum is not None and m.freq_count > 0:
                results[f"block_{block_idx}"] = (
                    m.freq_retention_sum / m.freq_count
                )
            else:
                results[f"block_{block_idx}"] = None

            block_idx += 1

    return results


# ============================================================
# FDR control helper functions
# ============================================================

def disable_freq_pruning(model):
    """
    Disable FDR for all Heat2D blocks.

    The function name is kept for compatibility with the original scripts.
    """

    for m in model.modules():
        if isinstance(m, Heat2D):
            m.apply_freq_mask = False
            m.freq_keep_ratio = 1.0


def enable_freq_pruning(model, keep_ratio):
    """
    Enable a unified FDR keep ratio for all Heat2D blocks.

    The function name is kept for compatibility with the original scripts.
    """

    for m in model.modules():
        if isinstance(m, Heat2D):
            m.apply_freq_mask = True
            m.freq_keep_ratio = keep_ratio


def enable_freq_pruning_for_blocks(model, keep_ratio, target_blocks):
    """
    Enable FDR only for selected blocks.

    Example:
        target_blocks = ["block_10", "block_11"]

    The function name is kept for compatibility with the original scripts.
    """

    block_idx = 0

    for m in model.modules():
        if isinstance(m, Heat2D):
            key = f"block_{block_idx}"

            if key in target_blocks:
                m.apply_freq_mask = True
                m.freq_keep_ratio = keep_ratio
            else:
                m.apply_freq_mask = False
                m.freq_keep_ratio = 1.0

            block_idx += 1


def apply_stagewise_freq_pruning(model, stage_ratios, depths):
    """
    Apply stage-wise FDR keep ratios to Heat2D blocks.

    Args:
        model:
            vHeat model.

        stage_ratios:
            Example: [1.0, 1.0, 0.8, 0.6]

        depths:
            Example: [2, 2, 6, 2]

    The function name is kept for compatibility with the original scripts.
    """

    assert len(stage_ratios) == len(depths), (
        f"len(stage_ratios)={len(stage_ratios)} must equal "
        f"len(depths)={len(depths)}"
    )

    block_keep_ratios = []

    for s, d in enumerate(depths):
        block_keep_ratios.extend(
            [float(stage_ratios[s])] * int(d)
        )

    block_idx = 0

    for m in model.modules():
        if isinstance(m, Heat2D):
            keep_ratio = block_keep_ratios[block_idx]

            m.apply_freq_mask = keep_ratio < 1.0
            m.freq_keep_ratio = keep_ratio

            block_idx += 1

    assert block_idx == len(block_keep_ratios), (
        f"Assigned {block_idx} Heat2D blocks, "
        f"expected {len(block_keep_ratios)}"
    )


# ============================================================
# Optional channel-mask helper functions for redundancy analysis
# ============================================================

def clear_heat_channel_mask(model):
    for m in model.modules():
        if isinstance(m, HeatBlock):
            m.apply_channel_mask = False
            m.channel_mask = None


def set_heat_channel_mask(model, block_masks):
    """
    Apply channel masks to selected HeatBlock modules.

    Args:
        block_masks:
            {
                "block_4": [idx1, idx2, ...],
                "block_5": [idx1, idx2, ...],
            }

        The listed channel indices will be set to zero.
    """

    clear_heat_channel_mask(model)

    block_idx = 0

    for m in model.modules():
        if isinstance(m, HeatBlock):
            key = f"block_{block_idx}"

            if key in block_masks:
                hidden_dim = m.op.hidden_dim
                mask = torch.ones(hidden_dim, 1, 1)
                mask[block_masks[key]] = 0.0

                m.channel_mask = mask
                m.apply_channel_mask = True

            block_idx += 1


def count_masked_channels(block_masks):
    total = 0

    for _, idxs in block_masks.items():
        total += len(idxs)

    return total


# ============================================================
# vHeat backbone
# ============================================================

class vHeat(nn.Module):
    """
    vHeat backbone with HCO-oriented redundancy-aware optimisation.

    This class preserves the original experimental structure and supports:

        1. Original vHeat:
            partial_heat=False, freq_pruning=False

        2. PHR-only:
            partial_heat=True, freq_pruning=False

        3. FDR-only:
            partial_heat=False, freq_pruning=True

        4. PHR+FDR:
            partial_heat=True, freq_pruning=True

    Note:
        The internal argument `freq_pruning` corresponds to FDR in the paper.
        It is kept for compatibility with the original experimental code.
    """

    def __init__(
        self,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        depths=[2, 2, 9, 2],
        dims=[96, 192, 384, 768],
        drop_path_rate=0.2,
        patch_norm=True,
        post_norm=True,
        layer_scale=None,
        use_checkpoint=False,
        mlp_ratio=4.0,
        img_size=224,
        act_layer="GELU",
        infer_mode=False,

        # PHR.
        partial_heat=False,
        heat_ratio=1.0,
        stage_heat_ratios=None,
        block_heat_ratios=None,
        light_branch_type="dwconv",

        # FDR.
        freq_pruning=False,
        freq_keep_ratio=1.0,
        stage_freq_ratios=None,

        **kwargs
    ):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)

        self.dims = (
            dims
            if isinstance(dims, list)
            else [int(dims * 2 ** i) for i in range(self.num_layers)]
        )

        self.embed_dim = self.dims[0]
        self.infer_mode = infer_mode

        self.patch_embed = StemLayer(
            in_chans=in_chans,
            out_chans=self.embed_dim,
            act_layer=act_layer,
            norm_layer="LN"
        )

        res0 = img_size // patch_size

        self.res = [
            int(res0 // (2 ** i))
            for i in range(self.num_layers)
        ]

        dpr = [
            x.item()
            for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]

        # Resolve stage-wise PHR ratios.
        if stage_heat_ratios is None:
            stage_heat_ratios = [float(heat_ratio)] * self.num_layers
        else:
            assert len(stage_heat_ratios) == self.num_layers, (
                f"len(stage_heat_ratios)={len(stage_heat_ratios)} must "
                f"equal num_layers={self.num_layers}"
            )

        # Resolve stage-wise FDR ratios.
        if stage_freq_ratios is None:
            stage_freq_ratios = [float(freq_keep_ratio)] * self.num_layers
        else:
            assert len(stage_freq_ratios) == self.num_layers, (
                f"len(stage_freq_ratios)={len(stage_freq_ratios)} must "
                f"equal num_layers={self.num_layers}"
            )

        self.stage_heat_ratios = [
            float(x) for x in stage_heat_ratios
        ]

        self.stage_freq_ratios = [
            float(x) for x in stage_freq_ratios
        ]

        # Optional block-wise Partial Heat ratios.
        if block_heat_ratios is None:
            self.block_heat_ratios = None
        else:
            self.block_heat_ratios = {
                str(k): float(v)
                for k, v in block_heat_ratios.items()
            }

        self.freq_embed = nn.ParameterList()

        for i in range(self.num_layers):
            embed = nn.Parameter(
                torch.zeros(
                    self.res[i],
                    self.res[i],
                    self.dims[i]
                ),
                requires_grad=True
            )
            trunc_normal_(embed, std=0.02)
            self.freq_embed.append(embed)

        self.blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        global_block_idx = 0

        for i in range(self.num_layers):
            stage_blocks = []

            for d in range(depths[i]):
                block_key = f"block_{global_block_idx}"

                # Resolve current block heat ratio.
                if self.block_heat_ratios is not None:
                    cur_heat_ratio = self.block_heat_ratios.get(
                        block_key,
                        1.0
                    )
                else:
                    cur_heat_ratio = self.stage_heat_ratios[i]

                # PHR is active only when partial_heat=True and heat_ratio < 1.
                cur_partial_heat = bool(
                    partial_heat and cur_heat_ratio < 1.0
                )

                stage_blocks.append(
                    HeatBlock(
                        res=self.res[i],
                        infer_mode=infer_mode,
                        hidden_dim=self.dims[i],
                        drop_path=dpr[sum(depths[:i]) + d],
                        norm_layer=LayerNorm2d,
                        use_checkpoint=use_checkpoint,
                        mlp_ratio=mlp_ratio,

                        # Keep original experimental behavior.
                        post_norm=post_norm,
                        layer_scale=layer_scale,

                        partial_heat=cur_partial_heat,
                        heat_ratio=cur_heat_ratio,
                        light_branch_type=light_branch_type,

                        freq_pruning=freq_pruning,
                        freq_keep_ratio=self.stage_freq_ratios[i],
                    )
                )

                global_block_idx += 1

            self.blocks.append(nn.ModuleList(stage_blocks))

            if i < self.num_layers - 1:
                self.downsamples.append(
                    self._make_downsample(
                        self.dims[i],
                        self.dims[i + 1]
                    )
                )
            else:
                self.downsamples.append(nn.Identity())

        self.classifier = nn.Sequential(
            LayerNorm2d(self.dims[-1]),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(self.dims[-1], num_classes)
        )

        self.apply(self._init_weights)

    @staticmethod
    def _make_downsample(in_dim, out_dim):
        return nn.Sequential(
            nn.Conv2d(
                in_dim,
                out_dim,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),
            LayerNorm2d(out_dim)
        )

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def infer_init(self):
        for i in range(self.num_layers):
            for block in self.blocks[i]:
                if isinstance(block, HeatBlock) and hasattr(
                    block.op,
                    "infer_init_heat2d"
                ):
                    freq_i = self.freq_embed[i]

                    if freq_i.shape[-1] != block.op.hidden_dim:
                        freq_i = freq_i[:, :, :block.op.hidden_dim]

                    block.op.infer_init_heat2d(freq_i)

        del self.freq_embed
        self.infer_mode = True

    def forward_features(self, x):
        x = self.patch_embed(x)
        features = []

        for i in range(self.num_layers):
            for block in self.blocks[i]:
                if self.infer_mode:
                    x = block(x)
                else:
                    x = block(x, self.freq_embed[i])

            features.append(x)
            x = self.downsamples[i](x)

        return features

    def forward(self, x):
        x = self.forward_features(x)
        x = self.classifier(x[-1])
        return x


# ============================================================
# Builder helper
# ============================================================

def build_vheat_from_config(config, num_classes):
    """
    Build vHeat from config.

    This helper supports both the old experimental config names and the
    cleaner paper-oriented config names.

    Old-style config:
        MODEL.PARTIAL_HEAT
        MODEL.FP

    Paper-oriented config:
        MODEL.PARTIAL_HEAT
        MODEL.FP
    """

    def has_attr(obj, name):
        return hasattr(obj, name)

    def get_attr(obj, name, default=None):
        return getattr(obj, name, default)

    # Basic vHeat settings.
    depths = config.MODEL.VHEAT.DEPTHS
    dims = config.MODEL.VHEAT.DIMS
    post_norm = get_attr(config.MODEL.VHEAT, "POST_NORM", True)

    # --------------------------------------------------------
    # PHR settings
    # --------------------------------------------------------
    if has_attr(config.MODEL, "REDUNDANCY") and has_attr(config.MODEL.REDUNDANCY, "PHR"):
        phr_cfg = config.MODEL.REDUNDANCY.PHR
        partial_heat = phr_cfg.ENABLE
        stage_heat_ratios = phr_cfg.STAGE_HEAT_RATIOS
        light_branch_type = get_attr(phr_cfg, "LIGHT_BRANCH", "dwconv")

    elif has_attr(config.MODEL, "PARTIAL_HEAT"):
        phr_cfg = config.MODEL.PARTIAL_HEAT
        partial_heat = phr_cfg.ENABLE
        stage_heat_ratios = phr_cfg.STAGE_HEAT_RATIOS
        light_branch_type = get_attr(phr_cfg, "LIGHT_BRANCH", "dwconv")

    else:
        partial_heat = False
        stage_heat_ratios = [1.0 for _ in depths]
        light_branch_type = "dwconv"

    # --------------------------------------------------------
    # FDR settings
    # --------------------------------------------------------
    if has_attr(config.MODEL, "REDUNDANCY") and has_attr(config.MODEL.REDUNDANCY, "FDR"):
        fdr_cfg = config.MODEL.REDUNDANCY.FDR
        freq_pruning = fdr_cfg.ENABLE
        stage_freq_ratios = fdr_cfg.STAGE_KEEP_RATIOS
        freq_keep_ratio = get_attr(fdr_cfg, "KEEP_RATIO", 1.0)

    elif has_attr(config.MODEL, "FP"):
        fdr_cfg = config.MODEL.FP
        freq_pruning = fdr_cfg.ENABLE
        stage_freq_ratios = fdr_cfg.STAGE_RATIOS
        freq_keep_ratio = get_attr(fdr_cfg, "KEEP_RATIO", 1.0)

    else:
        freq_pruning = False
        stage_freq_ratios = [1.0 for _ in depths]
        freq_keep_ratio = 1.0

    model = vHeat(
        num_classes=num_classes,
        depths=depths,
        dims=dims,
        drop_path_rate=config.MODEL.DROP_PATH_RATE,
        img_size=config.DATA.IMG_SIZE,
        post_norm=post_norm,

        partial_heat=partial_heat,
        stage_heat_ratios=stage_heat_ratios,
        light_branch_type=light_branch_type,

        freq_pruning=freq_pruning,
        freq_keep_ratio=freq_keep_ratio,
        stage_freq_ratios=stage_freq_ratios,
    )

    return model
