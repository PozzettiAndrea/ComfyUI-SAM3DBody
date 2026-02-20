# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Consolidated model architecture for SAM 3D Body (ComfyUI-native).
# Sources: vit.py, transformer.py, swiglu_ffn.py, drop_path.py, layer_scale.py,
#          camera_embed.py, prompt_encoder.py, promptable_decoder.py,
#          keypoint_prompt_sampler.py, mhr_head.py, camera_head.py,
#          base_model.py, sam3d_body.py

import copy
import logging
import math
import os
import random
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import einops
import numpy as np
import roma
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint_utils
from torch import Tensor

import comfy.model_management
import comfy.ops
from comfy_attn import dispatch_attention
import sys
_dbg_attn_counts = {}  # track attention call counts by caller

from .utils_model import (
    aa_to_rotmat,
    cam_crop_to_full,
    compact_cont_to_model_params_body,
    compact_cont_to_model_params_hand,
    compact_model_params_to_cont_body,
    fix_wrist_euler,
    get_intrinsic_matrix,
    mhr_param_hand_mask,
    perspective_projection,
    rot6d_to_rotmat,
    rotation_angle_difference,
    to_2tuple,
    to_ntuple,
)
from .utils_data import prepare_batch
from .utils import recursive_to

log = logging.getLogger("sam3dbody")

ops = comfy.ops.manual_cast


# =============================================================================
# drop_path.py — Stochastic depth
# =============================================================================

def drop_path(
    x: torch.Tensor, drop_prob: float = 0.0, training: bool = False
) -> torch.Tensor:
    if not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    output = x.div(keep_prob) * random_tensor.floor()
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.1):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


# =============================================================================
# layer_scale.py — LayerScale
# =============================================================================

class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        layer_scale_init_value: Union[float, torch.Tensor] = 1e-5,
        inplace: bool = False,
        data_format: str = "channels_last",
    ):
        super().__init__()
        assert data_format in ("channels_last", "channels_first")
        self.inplace = inplace
        self.data_format = data_format
        self.weight = nn.Parameter(torch.ones(dim) * layer_scale_init_value)

    def forward(self, x):
        w = self.weight.to(x.dtype)
        if self.data_format == "channels_first":
            if self.inplace:
                return x.mul_(w.view(-1, 1, 1))
            else:
                return x * w.view(-1, 1, 1)
        return x.mul_(w) if self.inplace else x * w


# =============================================================================
# transformer.py — LayerNorm variants, FFN, MLP
# =============================================================================

class LayerNorm32(ops.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)


def build_norm_layer(cfg: Dict, num_features: int):
    if not isinstance(cfg, dict):
        raise TypeError("cfg must be a dict")
    if "type" not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    cfg_ = cfg.copy()
    layer_type = cfg_.pop("type")
    if layer_type != "LN":
        raise ValueError("Unsupported norm layer: ", layer_type)
    requires_grad = cfg_.pop("requires_grad", True)
    cfg_.setdefault("eps", 1e-5)
    layer = LayerNorm32(num_features, **cfg_)
    for param in layer.parameters():
        param.requires_grad = requires_grad
    return layer


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.to(x.dtype)[:, None, None] * x + self.bias.to(x.dtype)[:, None, None]
        return x


class FFN(nn.Module):
    """Implements feed-forward networks (FFNs) with identity connection."""

    def __init__(
        self,
        embed_dims=256,
        feedforward_channels=1024,
        output_dims=None,
        num_fcs=2,
        act_layer=nn.ReLU,
        ffn_drop=0.0,
        drop_path_rate=0.0,
        add_identity=True,
        layer_scale_init_value=0.0,
        dtype=None,
        device=None,
        operations=ops,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.output_dims = output_dims or embed_dims
        self.num_fcs = num_fcs

        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                nn.Sequential(
                    operations.Linear(in_channels, feedforward_channels, dtype=dtype, device=device),
                    act_layer(),
                    nn.Dropout(ffn_drop),
                )
            )
            in_channels = feedforward_channels
        layers.append(operations.Linear(in_channels, self.output_dims, dtype=dtype, device=device))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = nn.Sequential(*layers)
        self.dropout_layer = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )
        self.add_identity = add_identity

        if layer_scale_init_value > 0:
            self.gamma2 = LayerScale(embed_dims, layer_scale_init_value=layer_scale_init_value)
        else:
            self.gamma2 = nn.Identity()

    def forward(self, x, identity=None):
        out = self.layers(x)
        out = self.gamma2(out)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        return identity + self.dropout_layer(out)


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers,
                 dtype=None, device=None, operations=ops):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            operations.Linear(n, k, dtype=dtype, device=device)
            for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


# =============================================================================
# swiglu_ffn.py — SwiGLU FFN
# =============================================================================

class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        embed_dims: int,
        feedforward_channels: Optional[int] = None,
        out_dims: Optional[int] = None,
        layer_scale_init_value: float = 0.0,
        bias: bool = True,
        drop_path_rate: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        add_identity: bool = True,
        dtype=None,
        device=None,
        operations=ops,
    ) -> None:
        super().__init__()
        self.embed_dims = embed_dims
        self.out_dims = out_dims or embed_dims
        hidden_dims = feedforward_channels or embed_dims

        self.w12 = operations.Linear(self.embed_dims, 2 * hidden_dims, bias=bias, dtype=dtype, device=device)
        self.norm = norm_layer
        self.w3 = operations.Linear(hidden_dims, self.out_dims, bias=bias, dtype=dtype, device=device)

        if layer_scale_init_value > 0:
            self.gamma2 = LayerScale(
                dim=embed_dims, layer_scale_init_value=layer_scale_init_value
            )
        else:
            self.gamma2 = nn.Identity()

        self.dropout_layer = DropPath(drop_path_rate)
        self.add_identity = add_identity

    def forward(
        self, x: torch.Tensor, identity: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        hidden = self.norm(hidden)
        out = self.w3(hidden)
        out = self.gamma2(out)
        out = self.dropout_layer(out)

        if self.out_dims != self.embed_dims or not self.add_identity:
            return out

        if identity is None:
            identity = x
        return identity + out


class SwiGLUFFNFused(SwiGLUFFN):
    def __init__(
        self,
        embed_dims: int,
        feedforward_channels: Optional[int] = None,
        out_dims: Optional[int] = None,
        layer_scale_init_value: float = 0.0,
        bias: bool = True,
        dtype=None,
        device=None,
        operations=ops,
    ) -> None:
        out_dims = out_dims or embed_dims
        feedforward_channels = feedforward_channels or embed_dims
        feedforward_channels = (int(feedforward_channels * 2 / 3) + 7) // 8 * 8
        super().__init__(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            out_dims=out_dims,
            layer_scale_init_value=layer_scale_init_value,
            bias=bias,
            dtype=dtype,
            device=device,
            operations=operations,
        )


# =============================================================================
# vit.py — Vision Transformer backbone
# =============================================================================

class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        dtype=None,
        device=None,
        operations=ops,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = operations.Linear(in_features, hidden_features, dtype=dtype, device=device)
        self.act = act_layer()
        self.fc2 = operations.Linear(hidden_features, out_features, dtype=dtype, device=device)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class VitAttention(nn.Module):
    """Unified attention using comfy-attn dispatch (sage/flash/sdpa)."""

    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        attn_head_dim=None,
        dtype=None,
        device=None,
        operations=ops,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        if attn_head_dim is not None:
            head_dim = attn_head_dim
        all_head_dim = head_dim * self.num_heads
        self.qkv = operations.Linear(dim, all_head_dim * 3, bias=qkv_bias, dtype=dtype, device=device)
        self.proj = operations.Linear(all_head_dim, dim, dtype=dtype, device=device)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        k = "VitAttention"
        if k not in _dbg_attn_counts:
            _dbg_attn_counts[k] = 0
            print(f"[DEBUG] {k}.forward input x.dtype={x.dtype} x.device={x.device} shape={x.shape}", file=sys.stderr, flush=True)
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k_t, v = qkv.unbind(0)
        if _dbg_attn_counts[k] == 0:
            print(f"[DEBUG] {k} q.dtype={q.dtype} q.shape={q.shape}", file=sys.stderr, flush=True)
        _dbg_attn_counts[k] += 1

        x = dispatch_attention(q, k_t, v)
        x = x.transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_head_dim=None,
        flash_attn=False,
        attn_backend=None,
        dtype=None,
        device=None,
        operations=ops,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = VitAttention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            attn_head_dim=attn_head_dim,
            dtype=dtype,
            device=device,
            operations=operations,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            dtype=dtype,
            device=device,
            operations=operations,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, ratio=1,
                 dtype=None, device=None, operations=ops):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (
            (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (ratio**2)
        )
        self.patch_shape = (
            int(img_size[0] // patch_size[0] * ratio),
            int(img_size[1] // patch_size[1] * ratio),
        )
        self.origin_patch_shape = (
            int(img_size[0] // patch_size[0]),
            int(img_size[1] // patch_size[1]),
        )
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = operations.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=(patch_size[0] // ratio),
            padding=4 + 2 * (ratio // 2 - 1),
            dtype=dtype, device=device,
        )

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        return x, (Hp, Wp)


class PatchEmbedNoPadding(nn.Module):
    """Image to Patch Embedding (no padding)"""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, ratio=1,
                 dtype=None, device=None, operations=ops):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (
            (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (ratio**2)
        )
        self.patch_shape = (
            int(img_size[0] // patch_size[0] * ratio),
            int(img_size[1] // patch_size[1] * ratio),
        )
        self.origin_patch_shape = (
            int(img_size[0] // patch_size[0]),
            int(img_size[1] // patch_size[1]),
        )
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = operations.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size,
            stride=(patch_size[0] // ratio),
            padding=0,
            dtype=dtype, device=device,
        )

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)
        return x, (Hp, Wp)


class HybridEmbed(nn.Module):
    """CNN Feature Map Embedding"""

    def __init__(
        self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768,
        dtype=None, device=None, operations=ops,
    ):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = operations.Linear(feature_dim, embed_dim, dtype=dtype, device=device)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


def get_abs_pos(abs_pos, h, w, ori_h, ori_w, has_cls_token=True):
    """Calculate absolute positional embeddings. If needed, resize embeddings
    and remove cls_token dimension for the original embeddings."""
    cls_token = None
    B, L, C = abs_pos.shape
    if has_cls_token:
        cls_token = abs_pos[:, 0:1]
        abs_pos = abs_pos[:, 1:]

    if ori_h != h or ori_w != w:
        new_abs_pos = (
            F.interpolate(
                abs_pos.reshape(1, ori_h, ori_w, -1).permute(0, 3, 1, 2),
                size=(h, w),
                mode="bicubic",
                align_corners=False,
            )
            .permute(0, 2, 3, 1)
            .reshape(B, -1, C)
        )
    else:
        new_abs_pos = abs_pos

    if cls_token is not None:
        new_abs_pos = torch.cat([cls_token, new_abs_pos], dim=1)
    return new_abs_pos


class ViT(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=80,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        hybrid_backbone=None,
        norm_layer=None,
        use_checkpoint=False,
        frozen_stages=-1,
        ratio=1,
        last_norm=True,
        patch_padding="pad",
        freeze_attn=False,
        freeze_ffn=False,
        flash_attn=False,
        attn_backend=None,
        no_patch_padding=False,
        dtype=None,
        device=None,
        operations=ops,
    ):
        super().__init__()
        if attn_backend is None:
            attn_backend = "flash_attn" if flash_attn else "manual"
        norm_layer = norm_layer or partial(ops.LayerNorm, eps=1e-6)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = self.embed_dims = embed_dim
        self.frozen_stages = frozen_stages
        self.use_checkpoint = use_checkpoint
        self.patch_padding = patch_padding
        self.freeze_attn = freeze_attn
        self.freeze_ffn = freeze_ffn
        self.depth = depth

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans,
                embed_dim=embed_dim, dtype=dtype, device=device, operations=operations,
            )
        else:
            if no_patch_padding:
                self.patch_embed = PatchEmbedNoPadding(
                    img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                    embed_dim=embed_dim, ratio=ratio,
                    dtype=dtype, device=device, operations=operations,
                )
            else:
                self.patch_embed = PatchEmbed(
                    img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                    embed_dim=embed_dim, ratio=ratio,
                    dtype=dtype, device=device, operations=operations,
                )
        num_patches = self.patch_embed.num_patches
        self.patch_size = patch_size

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer, attn_backend=attn_backend,
                dtype=dtype, device=device, operations=operations,
            )
            for i in range(depth)
        ])

        self.last_norm = norm_layer(embed_dim) if last_norm else nn.Identity()

        if self.pos_embed is not None:
            torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = self.blocks[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

        if self.freeze_attn:
            for i in range(0, self.depth):
                m = self.blocks[i]
                m.attn.eval()
                m.norm1.eval()
                for param in m.attn.parameters():
                    param.requires_grad = False
                for param in m.norm1.parameters():
                    param.requires_grad = False

        if self.freeze_ffn:
            self.pos_embed.requires_grad = False
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            for i in range(0, self.depth):
                m = self.blocks[i]
                m.mlp.eval()
                m.norm2.eval()
                for param in m.mlp.parameters():
                    param.requires_grad = False
                for param in m.norm2.parameters():
                    param.requires_grad = False

    def init_weights(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        self.apply(_init_weights)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def forward_features(self, x, extra_embed=None):
        B, C, H, W = x.shape
        x, (Hp, Wp) = self.patch_embed(x)

        if self.pos_embed is not None:
            pos = self.pos_embed.to(x.dtype)
            x = x + pos[:, 1:] + pos[:, :1]

        if extra_embed is not None:
            x = x + extra_embed.flatten(2).transpose(1, 2).to(x)

        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint_utils.checkpoint(blk, x)
            else:
                x = blk(x)

        x = self.last_norm(x)
        xp = x.permute(0, 2, 1).reshape(B, -1, Hp, Wp).contiguous()
        return xp

    def forward(self, x, *args, **kwargs):
        x = self.forward_features(x, *args, **kwargs)
        return x

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()


def vit(cfg, dtype=None, device=None, operations=ops):
    return ViT(
        img_size=(256, 192), patch_size=16, embed_dim=1280, depth=32,
        num_heads=16, ratio=1, norm_layer=LayerNorm32, use_checkpoint=False,
        mlp_ratio=4, qkv_bias=True, drop_path_rate=0.55,
        frozen_stages=cfg.MODEL.BACKBONE.get("FROZEN_STAGES", -1),
        dtype=dtype, device=device, operations=operations,
    )


def vit_l(cfg, dtype=None, device=None, operations=ops):
    return ViT(
        img_size=(256, 192), patch_size=16, embed_dim=1024, depth=24,
        num_heads=16, ratio=1, norm_layer=LayerNorm32, use_checkpoint=False,
        mlp_ratio=4, qkv_bias=True, drop_path_rate=0.55,
        frozen_stages=cfg.MODEL.BACKBONE.get("FROZEN_STAGES", -1),
        dtype=dtype, device=device, operations=operations,
    )


def vit_b(cfg, dtype=None, device=None, operations=ops):
    return ViT(
        img_size=(256, 192), patch_size=16, embed_dim=768, depth=12,
        num_heads=12, ratio=1, norm_layer=LayerNorm32, use_checkpoint=False,
        mlp_ratio=4, qkv_bias=True, drop_path_rate=0.3,
        frozen_stages=cfg.MODEL.BACKBONE.get("FROZEN_STAGES", -1),
        dtype=dtype, device=device, operations=operations,
    )


def vit256(cfg, dtype=None, device=None, operations=ops):
    return ViT(
        img_size=(256, 256), patch_size=16, embed_dim=1280, depth=32,
        num_heads=16, ratio=1, norm_layer=LayerNorm32, use_checkpoint=False,
        mlp_ratio=4, qkv_bias=True, drop_path_rate=0.55,
        frozen_stages=cfg.MODEL.BACKBONE.get("FROZEN_STAGES", -1),
        dtype=dtype, device=device, operations=operations,
    )


def vit512_384(cfg, dtype=None, device=None, operations=ops):
    return ViT(
        img_size=(512, 384), patch_size=16, embed_dim=1280, depth=32,
        num_heads=16, ratio=1, norm_layer=LayerNorm32, use_checkpoint=False,
        mlp_ratio=4, qkv_bias=True, drop_path_rate=0.55,
        frozen_stages=cfg.MODEL.BACKBONE.get("FROZEN_STAGES", -1),
        dtype=dtype, device=device, operations=operations,
    )


# =============================================================================
# transformer.py — MultiheadAttention, CrossAttention, Encoder/Decoder layers
# =============================================================================

class MultiheadAttention(nn.Module):
    """Multi-head Attention Module."""

    def __init__(
        self,
        embed_dims,
        num_heads,
        input_dims=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path_rate=0.0,
        qkv_bias=True,
        proj_bias=True,
        v_shortcut=False,
        layer_scale_init_value=0.0,
        dtype=None,
        device=None,
        operations=ops,
    ):
        super().__init__()

        self.input_dims = input_dims or embed_dims
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.v_shortcut = v_shortcut

        self.head_dims = embed_dims // num_heads

        self.qkv = operations.Linear(self.input_dims, embed_dims * 3, bias=qkv_bias, dtype=dtype, device=device)
        self.attn_drop = attn_drop
        self.proj = operations.Linear(embed_dims, embed_dims, bias=proj_bias, dtype=dtype, device=device)
        self.proj_drop = nn.Dropout(proj_drop)

        self.out_drop = DropPath(drop_path_rate)

        if layer_scale_init_value > 0:
            layer_scale_init_value = layer_scale_init_value or 1e-5
            self.gamma1 = LayerScale(
                embed_dims, layer_scale_init_value=layer_scale_init_value
            )
        else:
            self.gamma1 = nn.Identity()

    def forward(self, x):
        B, N, _ = x.shape
        k_name = "MultiheadAttention"
        if k_name not in _dbg_attn_counts:
            _dbg_attn_counts[k_name] = 0
            print(f"[DEBUG] {k_name}.forward input x.dtype={x.dtype} x.device={x.device} shape={x.shape}", file=sys.stderr, flush=True)
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dims)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        if _dbg_attn_counts[k_name] == 0:
            print(f"[DEBUG] {k_name} q.dtype={q.dtype} q.shape={q.shape}", file=sys.stderr, flush=True)
        _dbg_attn_counts[k_name] += 1

        x = dispatch_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, self.embed_dims)

        x = self.proj(x)
        x = self.out_drop(self.gamma1(self.proj_drop(x)))

        if self.v_shortcut:
            x = v.squeeze(1) + x
        return x


class CrossAttention(nn.Module):
    """Multi-head Attention Module for both self and cross attention."""

    def __init__(
        self,
        embed_dims,
        num_heads,
        query_dims=None,
        key_dims=None,
        value_dims=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path_rate=0.0,
        qkv_bias=True,
        proj_bias=True,
        v_shortcut=False,
        layer_scale_init_value=0.0,
        dtype=None,
        device=None,
        operations=ops,
    ):
        super().__init__()

        self.query_dims = query_dims or embed_dims
        self.key_dims = key_dims or embed_dims
        self.value_dims = value_dims or embed_dims
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.v_shortcut = v_shortcut

        self.head_dims = embed_dims // num_heads

        self.q_proj = operations.Linear(self.query_dims, embed_dims, bias=qkv_bias, dtype=dtype, device=device)
        self.k_proj = operations.Linear(self.key_dims, embed_dims, bias=qkv_bias, dtype=dtype, device=device)
        self.v_proj = operations.Linear(self.value_dims, embed_dims, bias=qkv_bias, dtype=dtype, device=device)
        self.attn_drop = attn_drop
        self.proj = operations.Linear(embed_dims, self.query_dims, bias=proj_bias, dtype=dtype, device=device)
        self.proj_drop = nn.Dropout(proj_drop)

        self.out_drop = DropPath(drop_path_rate)

        if layer_scale_init_value > 0:
            layer_scale_init_value = layer_scale_init_value or 1e-5
            self.gamma1 = LayerScale(
                embed_dims, layer_scale_init_value=layer_scale_init_value
            )
        else:
            self.gamma1 = nn.Identity()

    def _separate_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        x = x.reshape(b, n, self.num_heads, self.head_dims)
        return x.transpose(1, 2)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ):
        B, N, _ = q.shape
        q = self._separate_heads(self.q_proj(q))
        k = self._separate_heads(self.k_proj(k))
        v = self._separate_heads(self.v_proj(v))

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        k_name = "CrossAttention"
        if k_name not in _dbg_attn_counts:
            _dbg_attn_counts[k_name] = 0
            print(f"[DEBUG] {k_name}.forward q.dtype={q.dtype} q.device={q.device} q.shape={q.shape}", file=sys.stderr, flush=True)
        _dbg_attn_counts[k_name] += 1

        x = dispatch_attention(q, k, v, attn_mask=attn_mask)
        x = x.transpose(1, 2).reshape(B, N, self.embed_dims)

        x = self.proj(x)
        x = self.out_drop(self.gamma1(self.proj_drop(x)))

        if self.v_shortcut:
            x = v.squeeze(1) + x
        return x


class TransformerEncoderLayer(nn.Module):
    """Implements one encoder layer in Vision Transformer."""

    def __init__(
        self,
        embed_dims,
        num_heads,
        feedforward_channels,
        layer_scale_init_value=0.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        num_fcs=2,
        qkv_bias=True,
        ffn_type="origin",
        act_layer=nn.GELU,
        norm_cfg=dict(type="LN", eps=1e-6),
        dtype=None,
        device=None,
        operations=ops,
    ):
        super().__init__()

        self.embed_dims = embed_dims

        self.ln1 = build_norm_layer(norm_cfg, self.embed_dims)

        self.attn = MultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            drop_path_rate=drop_path_rate,
            qkv_bias=qkv_bias,
            layer_scale_init_value=layer_scale_init_value,
            dtype=dtype,
            device=device,
            operations=operations,
        )

        self.ln2 = build_norm_layer(norm_cfg, self.embed_dims)

        if ffn_type == "origin":
            self.ffn = FFN(
                embed_dims=embed_dims,
                feedforward_channels=feedforward_channels,
                num_fcs=num_fcs,
                ffn_drop=drop_rate,
                drop_path_rate=drop_path_rate,
                act_layer=act_layer,
                layer_scale_init_value=layer_scale_init_value,
                dtype=dtype,
                device=device,
                operations=operations,
            )
        elif ffn_type == "swiglu_fused":
            self.ffn = SwiGLUFFNFused(
                embed_dims=embed_dims,
                feedforward_channels=feedforward_channels,
                layer_scale_init_value=layer_scale_init_value,
                dtype=dtype,
                device=device,
                operations=operations,
            )
        else:
            raise NotImplementedError

    @property
    def norm1(self):
        return self.ln1

    @property
    def norm2(self):
        return self.ln2

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = self.ffn(self.ln2(x), identity=x)
        return x


class TransformerDecoderLayer(nn.Module):
    """Implements one decoder layer in cross-attention Transformer."""

    def __init__(
        self,
        token_dims: int,
        context_dims: int,
        num_heads: int = 8,
        head_dims: int = 64,
        mlp_dims: int = 1024,
        layer_scale_init_value: float = 0.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        ffn_type: str = "origin",
        act_layer: nn.Module = nn.GELU,
        norm_cfg: Dict = dict(type="LN", eps=1e-6),
        enable_twoway: bool = False,
        repeat_pe: bool = False,
        skip_first_pe: bool = False,
        dtype=None,
        device=None,
        operations=ops,
    ):
        super().__init__()
        self.repeat_pe = repeat_pe
        self.skip_first_pe = skip_first_pe
        if self.repeat_pe:
            self.ln_pe_1 = build_norm_layer(norm_cfg, token_dims)
            self.ln_pe_2 = build_norm_layer(norm_cfg, context_dims)

        self.ln1 = build_norm_layer(norm_cfg, token_dims)

        self.self_attn = CrossAttention(
            embed_dims=num_heads * head_dims,
            num_heads=num_heads,
            query_dims=token_dims,
            key_dims=token_dims,
            value_dims=token_dims,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
            dtype=dtype,
            device=device,
            operations=operations,
        )

        self.ln2_1 = build_norm_layer(norm_cfg, token_dims)
        self.ln2_2 = build_norm_layer(norm_cfg, context_dims)

        self.cross_attn = CrossAttention(
            embed_dims=num_heads * head_dims,
            num_heads=num_heads,
            query_dims=token_dims,
            key_dims=context_dims,
            value_dims=context_dims,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
            dtype=dtype,
            device=device,
            operations=operations,
        )

        self.ln3 = build_norm_layer(norm_cfg, token_dims)

        if ffn_type == "origin":
            self.ffn = FFN(
                embed_dims=token_dims,
                feedforward_channels=mlp_dims,
                ffn_drop=drop_rate,
                drop_path_rate=drop_path_rate,
                act_layer=act_layer,
                layer_scale_init_value=layer_scale_init_value,
                dtype=dtype,
                device=device,
                operations=operations,
            )
        elif ffn_type == "swiglu_fused":
            self.ffn = SwiGLUFFNFused(
                embed_dims=token_dims,
                feedforward_channels=mlp_dims,
                layer_scale_init_value=layer_scale_init_value,
                dtype=dtype,
                device=device,
                operations=operations,
            )
        else:
            raise NotImplementedError

        self.enable_twoway = enable_twoway
        if self.enable_twoway:
            self.ln4_1 = build_norm_layer(norm_cfg, context_dims)
            self.ln4_2 = build_norm_layer(norm_cfg, token_dims)

            self.cross_attn_2 = CrossAttention(
                embed_dims=num_heads * head_dims,
                num_heads=num_heads,
                query_dims=context_dims,
                key_dims=token_dims,
                value_dims=token_dims,
                attn_drop=attn_drop_rate,
                proj_drop=drop_rate,
                drop_path_rate=drop_path_rate,
                layer_scale_init_value=layer_scale_init_value,
                dtype=dtype,
                device=device,
                operations=operations,
            )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        x_pe: Optional[torch.Tensor] = None,
        context_pe: Optional[torch.Tensor] = None,
        x_mask: Optional[torch.Tensor] = None,
    ):
        if self.repeat_pe and context_pe is not None:
            x_pe = self.ln_pe_1(x_pe)
            context_pe = self.ln_pe_2(context_pe)

        # Self attention block for tokens
        if self.repeat_pe and not self.skip_first_pe and x_pe is not None:
            q = k = self.ln1(x) + x_pe
            v = self.ln1(x)
        else:
            q = k = v = self.ln1(x)

        attn_mask = None
        if x_mask is not None:
            attn_mask = x_mask[:, :, None] @ x_mask[:, None, :]
            attn_mask.diagonal(dim1=1, dim2=2).fill_(1)
            attn_mask = attn_mask > 0
        x = x + self.self_attn(q=q, k=k, v=v, attn_mask=attn_mask)

        # Cross attention block, tokens attending to image embedding
        if self.repeat_pe and context_pe is not None:
            q = self.ln2_1(x) + x_pe
            k = self.ln2_2(context) + context_pe
            v = self.ln2_2(context)
        else:
            q = self.ln2_1(x)
            k = v = self.ln2_2(context)
        x = x + self.cross_attn(q=q, k=k, v=v)

        # MLP block
        x = self.ffn(self.ln3(x), identity=x)

        # (Optional) Cross attention block, image embeddings attending to tokens
        if self.enable_twoway:
            if self.repeat_pe and context_pe is not None:
                q = self.ln4_1(context) + context_pe
                k = self.ln4_2(x) + x_pe
                v = self.ln4_2(x)
            else:
                q = self.ln4_1(context)
                k = v = self.ln4_2(x)
            attn_mask = (
                (x_mask[:, None, :].repeat(1, context.shape[1], 1)) > 0
                if x_mask is not None
                else None
            )
            context = context + self.cross_attn_2(q=q, k=k, v=v, attn_mask=attn_mask)

        return x, context


# =============================================================================
# camera_embed.py — Camera encoder with Fourier features
# =============================================================================

class FourierPositionEncoding(nn.Module):
    def __init__(self, n, num_bands, max_resolution):
        super().__init__()
        self.num_bands = num_bands
        self.max_resolution = [max_resolution] * n

    @property
    def channels(self):
        num_dims = len(self.max_resolution)
        encoding_size = self.num_bands * num_dims
        encoding_size *= 2  # sin-cos
        encoding_size += num_dims  # concat
        return encoding_size

    def forward(self, pos):
        fourier_pos_enc = _generate_fourier_features(
            pos, num_bands=self.num_bands, max_resolution=self.max_resolution
        )
        return fourier_pos_enc


def _generate_fourier_features(pos, num_bands, max_resolution):
    b, n = pos.shape[:2]
    device = pos.device

    min_freq = 1.0
    freq_bands = torch.stack(
        [
            torch.linspace(start=min_freq, end=res / 2, steps=num_bands, device=device)
            for res in max_resolution
        ],
        dim=0,
    )

    per_pos_features = torch.stack(
        [pos[i, :, :][:, :, None] * freq_bands[None, :, :] for i in range(b)], 0
    )
    per_pos_features = per_pos_features.reshape(b, n, -1)

    per_pos_features = torch.cat(
        [torch.sin(np.pi * per_pos_features), torch.cos(np.pi * per_pos_features)],
        dim=-1,
    )

    per_pos_features = torch.cat([pos, per_pos_features], dim=-1)
    return per_pos_features


class CameraEncoder(nn.Module):
    def __init__(self, embed_dim, patch_size=14,
                 dtype=None, device=None, operations=ops):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.camera = FourierPositionEncoding(n=3, num_bands=16, max_resolution=64)

        self.conv = operations.Conv2d(embed_dim + 99, embed_dim, kernel_size=1, bias=False,
                                      dtype=dtype, device=device)
        self.norm = LayerNorm2d(embed_dim)

    def forward(self, img_embeddings, rays):
        B, D, _h, _w = img_embeddings.shape

        with torch.no_grad():
            scale = 1 / self.patch_size
            rays = F.interpolate(
                rays,
                scale_factor=(scale, scale),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            rays = rays.permute(0, 2, 3, 1).contiguous()
            rays = torch.cat([rays, torch.ones_like(rays[..., :1])], dim=-1)
            rays_embeddings = self.camera(
                pos=rays.reshape(B, -1, 3)
            )
            rays_embeddings = einops.rearrange(
                rays_embeddings, "b (h w) c -> b c h w", h=_h, w=_w
            ).contiguous()

        z = torch.concat([img_embeddings, rays_embeddings], dim=1)
        z = self.norm(self.conv(z))
        return z


# =============================================================================
# prompt_encoder.py — Positional encoding and prompt encoder
# =============================================================================

class PositionEmbeddingRandom(nn.Module):
    """Positional encoding using random spatial frequencies."""

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w
        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)

    def forward_with_coords(
        self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))


class PromptEncoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_body_joints: int,
        frozen: bool = False,
        mask_embed_type: Optional[str] = None,
        dtype=None,
        device=None,
        operations=ops,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_body_joints = num_body_joints

        # Keypoint prompts
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)
        self.point_embeddings = nn.ModuleList(
            [operations.Embedding(1, embed_dim, dtype=dtype, device=device) for _ in range(self.num_body_joints)]
        )
        self.not_a_point_embed = operations.Embedding(1, embed_dim, dtype=dtype, device=device)
        self.invalid_point_embed = operations.Embedding(1, embed_dim, dtype=dtype, device=device)

        # Mask prompt
        if mask_embed_type in ["v1"]:
            mask_in_chans = 16
            self.mask_downscaling = nn.Sequential(
                operations.Conv2d(1, mask_in_chans // 4, kernel_size=4, stride=4, dtype=dtype, device=device),
                LayerNorm2d(mask_in_chans // 4),
                nn.GELU(),
                operations.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=4, stride=4, dtype=dtype, device=device),
                LayerNorm2d(mask_in_chans),
                nn.GELU(),
                operations.Conv2d(mask_in_chans, embed_dim, kernel_size=1, dtype=dtype, device=device),
            )
        elif mask_embed_type in ["v2"]:
            mask_in_chans = 256
            self.mask_downscaling = nn.Sequential(
                operations.Conv2d(1, mask_in_chans // 64, kernel_size=2, stride=2, dtype=dtype, device=device),
                LayerNorm2d(mask_in_chans // 64),
                nn.GELU(),
                operations.Conv2d(mask_in_chans // 64, mask_in_chans // 16, kernel_size=2, stride=2, dtype=dtype, device=device),
                LayerNorm2d(mask_in_chans // 16),
                nn.GELU(),
                operations.Conv2d(mask_in_chans // 16, mask_in_chans // 4, kernel_size=2, stride=2, dtype=dtype, device=device),
                LayerNorm2d(mask_in_chans // 4),
                nn.GELU(),
                operations.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2, dtype=dtype, device=device),
                LayerNorm2d(mask_in_chans),
                nn.GELU(),
                operations.Conv2d(mask_in_chans, embed_dim, kernel_size=1, dtype=dtype, device=device),
            )
        else:
            assert mask_embed_type is None

        if mask_embed_type is not None:
            nn.init.zeros_(self.mask_downscaling[-1].weight)
            nn.init.zeros_(self.mask_downscaling[-1].bias)
            self.no_mask_embed = operations.Embedding(1, embed_dim, dtype=dtype, device=device)
            nn.init.zeros_(self.no_mask_embed.weight)

        self.frozen = frozen
        self._freeze_stages()

    def get_dense_pe(self, size: Tuple[int, int]) -> torch.Tensor:
        return self.pe_layer(size).unsqueeze(0)

    def _embed_keypoints(
        self,
        points: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        assert points.min() >= 0 and points.max() <= 1
        point_embedding = self.pe_layer._pe_encoding(points.to(torch.float))
        point_embedding[labels == -2] = 0.0
        point_embedding[labels == -2] += self.invalid_point_embed.weight
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        for i in range(self.num_body_joints):
            point_embedding[labels == i] += self.point_embeddings[i].weight
        point_mask = labels > -2
        return point_embedding, point_mask

    def _get_batch_size(
        self,
        keypoints: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor],
        masks: Optional[torch.Tensor],
    ) -> int:
        if keypoints is not None:
            return keypoints.shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

    def _get_device(self) -> torch.device:
        return self.point_embeddings[0].weight.device

    def forward(
        self,
        keypoints: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bs = self._get_batch_size(keypoints, boxes, masks)
        sparse_embeddings = torch.empty(
            (bs, 0, self.embed_dim), device=self._get_device()
        )
        sparse_masks = torch.empty((bs, 0), device=self._get_device())
        if keypoints is not None:
            coords = keypoints[:, :, :2]
            labels = keypoints[:, :, -1]
            point_embeddings, point_mask = self._embed_keypoints(coords, labels)
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
            sparse_masks = torch.cat([sparse_masks, point_mask], dim=1)
        return sparse_embeddings, sparse_masks

    def get_mask_embeddings(
        self,
        masks: Optional[torch.Tensor] = None,
        bs: int = 1,
        size: Tuple[int, int] = (16, 16),
    ) -> torch.Tensor:
        no_mask_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            bs, -1, size[0], size[1]
        )
        if masks is not None:
            mask_embeddings = self.mask_downscaling(masks)
        else:
            mask_embeddings = no_mask_embeddings
        return mask_embeddings, no_mask_embeddings

    def _freeze_stages(self):
        if self.frozen:
            for param in self.parameters():
                param.requires_grad = False


# =============================================================================
# promptable_decoder.py — Cross-attention Transformer decoder
# =============================================================================

class PromptableDecoder(nn.Module):
    """Cross-attention based Transformer decoder with prompts input."""

    def __init__(
        self,
        dims: int,
        context_dims: int,
        depth: int,
        num_heads: int = 8,
        head_dims: int = 64,
        mlp_dims: int = 1024,
        layer_scale_init_value: float = 0.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        ffn_type: str = "origin",
        act_layer: nn.Module = nn.GELU,
        norm_cfg: Dict = dict(type="LN", eps=1e-6),
        enable_twoway: bool = False,
        repeat_pe: bool = False,
        frozen: bool = False,
        do_interm_preds: bool = False,
        do_keypoint_tokens: bool = False,
        keypoint_token_update: bool = False,
        dtype=None,
        device=None,
        operations=ops,
    ):
        super().__init__()

        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(
                TransformerDecoderLayer(
                    token_dims=dims,
                    context_dims=context_dims,
                    num_heads=num_heads,
                    head_dims=head_dims,
                    mlp_dims=mlp_dims,
                    layer_scale_init_value=layer_scale_init_value,
                    drop_rate=drop_rate,
                    attn_drop_rate=attn_drop_rate,
                    drop_path_rate=drop_path_rate,
                    ffn_type=ffn_type,
                    act_layer=act_layer,
                    norm_cfg=norm_cfg,
                    enable_twoway=enable_twoway,
                    repeat_pe=repeat_pe,
                    skip_first_pe=(i == 0),
                    dtype=dtype,
                    device=device,
                    operations=operations,
                )
            )

        self.norm_final = build_norm_layer(norm_cfg, dims)
        self.do_interm_preds = do_interm_preds
        self.do_keypoint_tokens = do_keypoint_tokens
        self.keypoint_token_update = keypoint_token_update

        self.frozen = frozen
        self._freeze_stages()

    def forward(
        self,
        token_embedding: torch.Tensor,
        image_embedding: torch.Tensor,
        token_augment: Optional[torch.Tensor] = None,
        image_augment: Optional[torch.Tensor] = None,
        token_mask: Optional[torch.Tensor] = None,
        channel_first: bool = True,
        token_to_pose_output_fn=None,
        keypoint_token_update_fn=None,
        hand_embeddings=None,
        hand_augment=None,
    ):
        if channel_first:
            image_embedding = image_embedding.flatten(2).permute(0, 2, 1)
            if image_augment is not None:
                image_augment = image_augment.flatten(2).permute(0, 2, 1)
            if hand_embeddings is not None:
                hand_embeddings = hand_embeddings.flatten(2).permute(0, 2, 1)
                hand_augment = hand_augment.flatten(2).permute(0, 2, 1)
                if len(hand_augment) == 1:
                    assert len(hand_augment.shape) == 3
                    hand_augment = hand_augment.repeat(len(hand_embeddings), 1, 1)

        if self.do_interm_preds:
            assert token_to_pose_output_fn is not None
            all_pose_outputs = []

        for layer_idx, layer in enumerate(self.layers):
            if hand_embeddings is None:
                token_embedding, image_embedding = layer(
                    token_embedding,
                    image_embedding,
                    token_augment,
                    image_augment,
                    token_mask,
                )
            else:
                token_embedding, image_embedding = layer(
                    token_embedding,
                    torch.cat([image_embedding, hand_embeddings], dim=1),
                    token_augment,
                    torch.cat([image_augment, hand_augment], dim=1),
                    token_mask,
                )
                image_embedding = image_embedding[:, : image_augment.shape[1]]

            if self.do_interm_preds and layer_idx < len(self.layers) - 1:
                curr_pose_output = token_to_pose_output_fn(
                    self.norm_final(token_embedding),
                    prev_pose_output=(
                        all_pose_outputs[-1] if len(all_pose_outputs) > 0 else None
                    ),
                    layer_idx=layer_idx,
                )
                all_pose_outputs.append(curr_pose_output)

                if self.keypoint_token_update:
                    assert keypoint_token_update_fn is not None
                    token_embedding, token_augment, _, _ = keypoint_token_update_fn(
                        token_embedding, token_augment, curr_pose_output, layer_idx
                    )

        out = self.norm_final(token_embedding)

        if self.do_interm_preds:
            curr_pose_output = token_to_pose_output_fn(
                out,
                prev_pose_output=(
                    all_pose_outputs[-1] if len(all_pose_outputs) > 0 else None
                ),
                layer_idx=layer_idx,
            )
            all_pose_outputs.append(curr_pose_output)
            return out, all_pose_outputs
        else:
            return out

    def _freeze_stages(self):
        if self.frozen:
            for layer in self.layers:
                layer.eval()
            self.norm_final.eval()
            for param in self.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self._freeze_stages()


# =============================================================================
# keypoint_prompt_sampler.py — Keypoint prompt sampling
# =============================================================================

class BaseKeypointSampler(ABC):
    @abstractmethod
    def sample(
        self, gt_keypoints: torch.Tensor, pred_keypoints: torch.Tensor, is_train: bool
    ) -> torch.Tensor:
        pass

    def _get_worst_keypoint(self, distances, keypoint_list):
        cur_dist = torch.ones_like(distances) * -1
        cur_dist[keypoint_list] = distances[keypoint_list]
        keypoint_idx = int(cur_dist.argmax())
        if cur_dist[keypoint_idx] > self.distance_thresh:
            valid_keypoint = True
        else:
            valid_keypoint = False
        return keypoint_idx, valid_keypoint

    def _get_random_keypoint(self, distances, keypoint_list):
        candidates = [idx for idx in keypoint_list if distances[idx] > 0]
        if len(candidates):
            keypoint_idx = random.choice(candidates)
            valid_keypoint = True
        else:
            keypoint_idx = None
            valid_keypoint = False
        return keypoint_idx, valid_keypoint

    def _masked_distance(self, x, y, mask=None):
        distances = (x - y).pow(2).sum(dim=-1)
        if mask is not None:
            distances[mask] = -1
        return distances.T


class KeypointSamplerV1(BaseKeypointSampler):
    def __init__(
        self,
        sampler_cfg,
        prompt_keypoints: Dict,
        keybody_idx: List,
    ):
        self.prompt_keypoints = prompt_keypoints
        self._keybody_idx = keybody_idx
        self._non_keybody_idx = [
            idx for idx in self.prompt_keypoints if idx not in self._keybody_idx
        ]

        self.keybody_ratio = sampler_cfg.get("KEYBODY_RATIO", 0.8)
        self.worst_ratio = sampler_cfg.get("WORST_RATIO", 0.8)
        self.negative_ratio = sampler_cfg.get("NEGATIVE_RATIO", 0.0)
        self.dummy_ratio = sampler_cfg.get("DUMMY_RATIO", 0.1)
        self.distance_thresh = sampler_cfg.get("DISTANCE_THRESH", 0.0)

    def sample(
        self,
        gt_keypoints_2d: torch.Tensor,
        pred_keypoints_2d: torch.Tensor,
        is_train: bool = True,
        force_dummy: bool = False,
    ) -> torch.Tensor:
        mask_1 = gt_keypoints_2d[:, :, -1] < 0.5
        mask_2 = (
            (gt_keypoints_2d[:, :, :2] > 0.5) | (gt_keypoints_2d[:, :, :2] < -0.5)
        ).any(dim=-1)

        if not is_train or torch.rand(1).item() > self.negative_ratio:
            mask = mask_1 | mask_2
        else:
            mask_3 = (
                (pred_keypoints_2d[:, :, :2] > 0.5)
                | (pred_keypoints_2d[:, :, :2] < -0.5)
            ).any(dim=-1)
            mask = mask_1 | (mask_2 & mask_3)

        distances = self._masked_distance(
            pred_keypoints_2d, gt_keypoints_2d[..., :2], mask
        )

        batch_size = distances.shape[1]
        keypoints_prompt = []
        for b in range(batch_size):
            if not is_train or torch.rand(1).item() < self.worst_ratio:
                sampler = self._get_worst_keypoint
            else:
                sampler = self._get_random_keypoint

            if not is_train or torch.rand(1).item() < self.keybody_ratio:
                cur_idx = self._keybody_idx
                alt_idx = self._non_keybody_idx
            else:
                cur_idx = self._non_keybody_idx
                alt_idx = self._keybody_idx

            if not is_train or torch.rand(1).item() > self.dummy_ratio:
                keypoint_idx, valid_keypoint = sampler(distances[:, b], cur_idx)
                if not valid_keypoint:
                    keypoint_idx, valid_keypoint = self._get_worst_keypoint(
                        distances[:, b], alt_idx
                    )
            else:
                valid_keypoint = False

            if valid_keypoint:
                cur_point = gt_keypoints_2d[b, keypoint_idx].clone()
                if torch.any(cur_point[:2] > 0.5) or torch.any(cur_point[:2] < -0.5):
                    cur_point[:2] = pred_keypoints_2d[b, keypoint_idx][:2]
                    cur_point = torch.clamp(cur_point + 0.5, min=0.0, max=1.0)
                    cur_point[-1] = -1
                else:
                    cur_point = torch.clamp(cur_point + 0.5, min=0.0, max=1.0)
                    cur_point[-1] = self.prompt_keypoints[keypoint_idx]
            else:
                cur_point = torch.zeros(3).to(gt_keypoints_2d)
                cur_point[-1] = -2

            if force_dummy:
                cur_point = torch.zeros(3).to(gt_keypoints_2d)
                cur_point[-1] = -2

            keypoints_prompt.append(cur_point)

        keypoints_prompt = torch.stack(keypoints_prompt, dim=0).view(batch_size, 1, 3)
        return keypoints_prompt


def build_keypoint_sampler(sampler_cfg, prompt_keypoints, keybody_idx):
    sampler_type = sampler_cfg.get("TYPE", "v1")
    if sampler_type == "v1":
        sampler_cls = KeypointSamplerV1
    else:
        raise ValueError("Invalid sampler type: ", sampler_type)
    return sampler_cls(sampler_cfg, prompt_keypoints, keybody_idx)


# =============================================================================
# mhr_head.py — MHR head for body pose prediction
# =============================================================================

MOMENTUM_ENABLED = os.environ.get("MOMENTUM_ENABLED") is None
try:
    if MOMENTUM_ENABLED:
        from mhr.mhr import MHR
        MOMENTUM_ENABLED = True
        warnings.warn("Momentum is enabled")
    else:
        warnings.warn("Momentum is not enabled")
        raise ImportError
except Exception:
    MOMENTUM_ENABLED = False
    warnings.warn("Momentum is not enabled")


class MHRHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        mlp_depth: int = 1,
        mhr_model_path: str = "",
        extra_joint_regressor: str = "",
        ffn_zero_bias: bool = True,
        mlp_channel_div_factor: int = 8,
        enable_hand_model=False,
        dtype=None,
        device=None,
        operations=ops,
    ):
        super().__init__()

        self.num_shape_comps = 45
        self.num_scale_comps = 28
        self.num_hand_comps = 54
        self.num_face_comps = 72
        self.enable_hand_model = enable_hand_model

        self.body_cont_dim = 260
        self.npose = (
            6
            + self.body_cont_dim
            + self.num_shape_comps
            + self.num_scale_comps
            + self.num_hand_comps * 2
            + self.num_face_comps
        )

        self.proj = FFN(
            embed_dims=input_dim,
            feedforward_channels=input_dim // mlp_channel_div_factor,
            output_dims=self.npose,
            num_fcs=mlp_depth,
            ffn_drop=0.0,
            add_identity=False,
            dtype=dtype,
            device=device,
            operations=operations,
        )

        if ffn_zero_bias:
            torch.nn.init.zeros_(self.proj.layers[-2].bias)

        # MHR Parameters
        self.model_data_dir = mhr_model_path
        self.num_hand_scale_comps = self.num_scale_comps - 18
        self.num_hand_pose_comps = self.num_hand_comps

        # Buffers to be filled in by model state dict
        self.joint_rotation = nn.Parameter(torch.zeros(127, 3, 3), requires_grad=False)
        self.scale_mean = nn.Parameter(torch.zeros(68), requires_grad=False)
        self.scale_comps = nn.Parameter(torch.zeros(28, 68), requires_grad=False)
        self.faces = nn.Parameter(torch.zeros(36874, 3).long(), requires_grad=False)
        self.hand_pose_mean = nn.Parameter(torch.zeros(54), requires_grad=False)
        self.hand_pose_comps = nn.Parameter(torch.eye(54), requires_grad=False)
        self.hand_joint_idxs_left = nn.Parameter(
            torch.zeros(27).long(), requires_grad=False
        )
        self.hand_joint_idxs_right = nn.Parameter(
            torch.zeros(27).long(), requires_grad=False
        )
        self.keypoint_mapping = nn.Parameter(
            torch.zeros(308, 18439 + 127), requires_grad=False
        )
        self.right_wrist_coords = nn.Parameter(torch.zeros(3), requires_grad=False)
        self.root_coords = nn.Parameter(torch.zeros(3), requires_grad=False)
        self.local_to_world_wrist = nn.Parameter(torch.zeros(3, 3), requires_grad=False)
        self.nonhand_param_idxs = nn.Parameter(
            torch.zeros(145).long(), requires_grad=False
        )

        # Load MHR itself
        if MOMENTUM_ENABLED:
            self.mhr = MHR.from_files(
                device=comfy.model_management.get_torch_device(),
                lod=1,
            )
        else:
            self.mhr = torch.jit.load(
                mhr_model_path,
                map_location=comfy.model_management.get_torch_device(),
            )

        for param in self.mhr.parameters():
            param.requires_grad = False

    def get_zero_pose_init(self, factor=1.0):
        weights = torch.zeros(1, self.npose)
        weights[:, : 6 + self.body_cont_dim] = torch.cat(
            [
                torch.FloatTensor([1, 0, 0, 0, 1, 0]),
                compact_model_params_to_cont_body(torch.zeros(1, 133)).squeeze()
                * factor,
            ],
            dim=0,
        )
        return weights

    def replace_hands_in_pose(self, full_pose_params, hand_pose_params):
        assert full_pose_params.shape[1] == 136

        left_hand_params, right_hand_params = torch.split(
            hand_pose_params,
            [self.num_hand_pose_comps, self.num_hand_pose_comps],
            dim=1,
        )

        left_hand_params_model_params = compact_cont_to_model_params_hand(
            self.hand_pose_mean
            + torch.einsum("da,ab->db", left_hand_params, self.hand_pose_comps)
        )
        right_hand_params_model_params = compact_cont_to_model_params_hand(
            self.hand_pose_mean
            + torch.einsum("da,ab->db", right_hand_params, self.hand_pose_comps)
        )

        full_pose_params[:, self.hand_joint_idxs_left] = left_hand_params_model_params
        full_pose_params[:, self.hand_joint_idxs_right] = right_hand_params_model_params

        return full_pose_params

    def mhr_forward(
        self,
        global_trans,
        global_rot,
        body_pose_params,
        hand_pose_params,
        scale_params,
        shape_params,
        expr_params=None,
        return_keypoints=False,
        do_pcblend=True,
        return_joint_coords=False,
        return_model_params=False,
        return_joint_rotations=False,
        scale_offsets=None,
        vertex_offsets=None,
    ):
        if self.enable_hand_model:
            global_rot_ori = global_rot.clone()
            global_trans_ori = global_trans.clone()
            global_rot = roma.rotmat_to_euler(
                "xyz",
                roma.euler_to_rotmat("xyz", global_rot_ori) @ self.local_to_world_wrist,
            )
            global_trans = (
                -(
                    roma.euler_to_rotmat("xyz", global_rot)
                    @ (self.right_wrist_coords - self.root_coords)
                    + self.root_coords
                )
                + global_trans_ori
            )

        body_pose_params = body_pose_params[..., :130]

        if len(scale_params.shape) == 1:
            scale_params = scale_params[None]
        if len(shape_params.shape) == 1:
            shape_params = shape_params[None]
        scales = self.scale_mean[None, :] + scale_params @ self.scale_comps
        if scale_offsets is not None:
            scales = scales + scale_offsets

        full_pose_params = torch.cat(
            [global_trans * 10, global_rot, body_pose_params], dim=1
        )
        if hand_pose_params is not None:
            full_pose_params = self.replace_hands_in_pose(
                full_pose_params, hand_pose_params
            )
        model_params = torch.cat([full_pose_params, scales], dim=1)

        if self.enable_hand_model:
            model_params[:, self.nonhand_param_idxs] = 0

        curr_skinned_verts, curr_skel_state = self.mhr(
            shape_params, model_params, expr_params
        )
        curr_joint_coords, curr_joint_quats, _ = torch.split(
            curr_skel_state, [3, 4, 1], dim=2
        )
        curr_skinned_verts = curr_skinned_verts / 100
        curr_joint_coords = curr_joint_coords / 100
        curr_joint_rots = roma.unitquat_to_rotmat(curr_joint_quats)

        to_return = [curr_skinned_verts]
        if return_keypoints:
            model_vert_joints = torch.cat(
                [curr_skinned_verts, curr_joint_coords], dim=1
            )
            model_keypoints_pred = (
                (
                    self.keypoint_mapping
                    @ model_vert_joints.permute(1, 0, 2).flatten(1, 2)
                )
                .reshape(-1, model_vert_joints.shape[0], 3)
                .permute(1, 0, 2)
            )

            if self.enable_hand_model:
                model_keypoints_pred[:, :21] = 0
                model_keypoints_pred[:, 42:] = 0

            to_return = to_return + [model_keypoints_pred]
        if return_joint_coords:
            to_return = to_return + [curr_joint_coords]
        if return_model_params:
            to_return = to_return + [model_params]
        if return_joint_rotations:
            to_return = to_return + [curr_joint_rots]

        if isinstance(to_return, list) and len(to_return) == 1:
            return to_return[0]
        else:
            return tuple(to_return)

    def forward(
        self,
        x: torch.Tensor,
        init_estimate: Optional[torch.Tensor] = None,
        do_pcblend=True,
        slim_keypoints=False,
    ):
        batch_size = x.shape[0]
        pred = self.proj(x)
        if init_estimate is not None:
            pred = pred + init_estimate

        count = 6
        global_rot_6d = pred[:, :count]
        global_rot_rotmat = rot6d_to_rotmat(global_rot_6d)
        global_rot_euler = roma.rotmat_to_euler("ZYX", global_rot_rotmat)
        global_trans = torch.zeros_like(global_rot_euler)

        pred_pose_cont = pred[:, count : count + self.body_cont_dim]
        count += self.body_cont_dim
        pred_pose_euler = compact_cont_to_model_params_body(pred_pose_cont)
        pred_pose_euler[:, mhr_param_hand_mask] = 0
        pred_pose_euler[:, -3:] = 0

        pred_shape = pred[:, count : count + self.num_shape_comps]
        count += self.num_shape_comps
        pred_scale = pred[:, count : count + self.num_scale_comps]
        count += self.num_scale_comps
        pred_hand = pred[:, count : count + self.num_hand_comps * 2]
        count += self.num_hand_comps * 2
        pred_face = pred[:, count : count + self.num_face_comps] * 0
        count += self.num_face_comps

        output = self.mhr_forward(
            global_trans=global_trans,
            global_rot=global_rot_euler,
            body_pose_params=pred_pose_euler,
            hand_pose_params=pred_hand,
            scale_params=pred_scale,
            shape_params=pred_shape,
            expr_params=pred_face,
            do_pcblend=do_pcblend,
            return_keypoints=True,
            return_joint_coords=True,
            return_model_params=True,
            return_joint_rotations=True,
        )

        verts, j3d, jcoords, mhr_model_params, joint_global_rots = output
        j3d = j3d[:, :70]

        if verts is not None:
            verts[..., [1, 2]] *= -1
        j3d[..., [1, 2]] *= -1
        if jcoords is not None:
            jcoords[..., [1, 2]] *= -1

        output = {
            "pred_pose_raw": torch.cat(
                [global_rot_6d, pred_pose_cont], dim=1
            ),
            "pred_pose_rotmat": None,
            "global_rot": global_rot_euler,
            "body_pose": pred_pose_euler,
            "shape": pred_shape,
            "scale": pred_scale,
            "hand": pred_hand,
            "face": pred_face,
            "pred_keypoints_3d": j3d.reshape(batch_size, -1, 3),
            "pred_vertices": (
                verts.reshape(batch_size, -1, 3) if verts is not None else None
            ),
            "pred_joint_coords": (
                jcoords.reshape(batch_size, -1, 3) if jcoords is not None else None
            ),
            "faces": self.faces.cpu().numpy(),
            "joint_global_rots": joint_global_rots,
            "mhr_model_params": mhr_model_params,
        }

        return output


# =============================================================================
# camera_head.py — Perspective camera head
# =============================================================================

class PerspectiveHead(nn.Module):
    """Predict camera translation (s, tx, ty) and perform full-perspective
    2D reprojection."""

    def __init__(
        self,
        input_dim: int,
        img_size: Tuple[int, int],
        mlp_depth: int = 1,
        drop_ratio: float = 0.0,
        mlp_channel_div_factor: int = 8,
        default_scale_factor: float = 1,
        dtype=None,
        device=None,
        operations=ops,
    ):
        super().__init__()

        self.img_size = to_2tuple(img_size)
        self.ncam = 3
        self.default_scale_factor = default_scale_factor

        self.proj = FFN(
            embed_dims=input_dim,
            feedforward_channels=input_dim // mlp_channel_div_factor,
            output_dims=self.ncam,
            num_fcs=mlp_depth,
            ffn_drop=drop_ratio,
            add_identity=False,
            dtype=dtype,
            device=device,
            operations=operations,
        )

    def forward(
        self,
        x: torch.Tensor,
        init_estimate: Optional[torch.Tensor] = None,
    ):
        pred_cam = self.proj(x)
        if init_estimate is not None:
            pred_cam = pred_cam + init_estimate
        return pred_cam

    def perspective_projection(
        self,
        points_3d: torch.Tensor,
        pred_cam: torch.Tensor,
        bbox_center: torch.Tensor,
        bbox_size: torch.Tensor,
        img_size: torch.Tensor,
        cam_int: torch.Tensor,
        use_intrin_center: bool = False,
    ):
        batch_size = points_3d.shape[0]
        pred_cam = pred_cam.clone()
        pred_cam[..., [0, 2]] *= -1

        s, tx, ty = pred_cam[:, 0], pred_cam[:, 1], pred_cam[:, 2]
        bs = bbox_size * s * self.default_scale_factor + 1e-8
        focal_length = cam_int[:, 0, 0]
        tz = 2 * focal_length / bs

        if not use_intrin_center:
            cx = 2 * (bbox_center[:, 0] - (img_size[:, 0] / 2)) / bs
            cy = 2 * (bbox_center[:, 1] - (img_size[:, 1] / 2)) / bs
        else:
            cx = 2 * (bbox_center[:, 0] - (cam_int[:, 0, 2])) / bs
            cy = 2 * (bbox_center[:, 1] - (cam_int[:, 1, 2])) / bs

        pred_cam_t = torch.stack([tx + cx, ty + cy, tz], dim=-1)

        j3d_cam = points_3d + pred_cam_t.unsqueeze(1)
        j2d = perspective_projection(j3d_cam, cam_int)

        return {
            "pred_keypoints_2d": j2d.reshape(batch_size, -1, 2),
            "pred_cam_t": pred_cam_t,
            "focal_length": focal_length,
            "pred_keypoints_2d_depth": j3d_cam.reshape(batch_size, -1, 3)[:, :, 2],
        }


# =============================================================================
# base_model.py — Abstract base model (nn.Module, no Lightning)
# =============================================================================

class BaseModel(nn.Module):
    def __init__(self, cfg, dtype=None, device=None, operations=ops, **kwargs):
        super().__init__()
        self.cfg = cfg
        self._initialze_model(dtype=dtype, device=device, operations=operations, **kwargs)
        self._max_num_person = None
        self._person_valid = None

    @abstractmethod
    def _initialze_model(self, dtype=None, device=None, operations=ops, **kwargs) -> None:
        pass

    def data_preprocess(
        self,
        inputs: torch.Tensor,
        crop_width: bool = False,
        is_full: bool = False,
        crop_hand: int = 0,
    ) -> torch.Tensor:
        image_mean = self.image_mean if not is_full else self.full_image_mean
        image_std = self.image_std if not is_full else self.full_image_std

        print(f"[DEBUG] data_preprocess: inputs.dtype={inputs.dtype} inputs.device={inputs.device} "
              f"inputs.shape={inputs.shape} range=[{inputs.min().item():.3f},{inputs.max().item():.3f}] "
              f"image_mean.dtype={image_mean.dtype} image_std.dtype={image_std.dtype}", file=sys.stderr, flush=True)
        if inputs.max() > 1 and image_mean.max() <= 1.0:
            inputs = inputs / 255.0
        elif inputs.max() <= 1.0 and image_mean.max() > 1:
            inputs = inputs * 255.0
        batch_inputs = (inputs - image_mean) / image_std
        print(f"[DEBUG] data_preprocess: output.dtype={batch_inputs.dtype} range=[{batch_inputs.min().item():.3f},{batch_inputs.max().item():.3f}]", file=sys.stderr, flush=True)

        if crop_width:
            if crop_hand > 0:
                batch_inputs = batch_inputs[:, :, :, crop_hand:-crop_hand]
            elif self.cfg.MODEL.BACKBONE.TYPE in ["vit_hmr", "vit"]:
                batch_inputs = batch_inputs[:, :, :, 32:-32]
            elif self.cfg.MODEL.BACKBONE.TYPE in ["vit_hmr_512_384"]:
                batch_inputs = batch_inputs[:, :, :, 64:-64]
            else:
                raise Exception

        return batch_inputs

    def _initialize_batch(self, batch: Dict) -> None:
        if batch["img"].dim() == 5:
            self._batch_size, self._max_num_person = batch["img"].shape[:2]
            self._person_valid = self._flatten_person(batch["person_valid"]) > 0
        else:
            self._batch_size = batch["img"].shape[0]
            self._max_num_person = 0
            self._person_valid = None

    def _flatten_person(self, x: torch.Tensor) -> torch.Tensor:
        assert self._max_num_person is not None
        if self._max_num_person:
            shape = x.shape
            x = x.view(self._batch_size * self._max_num_person, *shape[2:])
        return x

    def _unflatten_person(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.shape
        if self._max_num_person:
            x = x.view(self._batch_size, self._max_num_person, *shape[1:])
        return x

    def _get_valid(self, x: torch.Tensor) -> torch.Tensor:
        assert self._max_num_person is not None
        if self._person_valid is not None:
            x = x[self._person_valid]
        return x

    def _full_to_crop(
        self, batch: Dict, pred_keypoints_2d: torch.Tensor
    ) -> torch.Tensor:
        pred_keypoints_2d_cropped = torch.cat(
            [pred_keypoints_2d, torch.ones_like(pred_keypoints_2d[:, :, [-1]])], dim=-1
        )
        affine_trans = self._flatten_person(batch["affine_trans"]).to(
            pred_keypoints_2d_cropped
        )
        img_size = self._flatten_person(batch["img_size"]).unsqueeze(1)
        pred_keypoints_2d_cropped = pred_keypoints_2d_cropped @ affine_trans.mT
        pred_keypoints_2d_cropped = pred_keypoints_2d_cropped[..., :2] / img_size - 0.5
        return pred_keypoints_2d_cropped

    def _cam_full_to_crop(
        self, batch: Dict, pred_cam_t: torch.Tensor, focal_length: torch.Tensor = None
    ) -> torch.Tensor:
        num_person = batch["img"].shape[1]
        cam_int = self._flatten_person(
            batch["cam_int"].unsqueeze(1).expand(-1, num_person, -1, -1).contiguous()
        )
        bbox_center = self._flatten_person(batch["bbox_center"])
        bbox_size = self._flatten_person(batch["bbox_scale"])[:, 0]
        img_size = self._flatten_person(batch["ori_img_size"])
        input_size = self._flatten_person(batch["img_size"])[:, 0]

        tx, ty, tz = pred_cam_t[:, 0], pred_cam_t[:, 1], pred_cam_t[:, 2]
        if focal_length is None:
            focal_length = cam_int[:, 0, 0]
        bs = 2 * focal_length / (tz + 1e-8)

        cx = 2 * (bbox_center[:, 0] - (cam_int[:, 0, 2])) / bs
        cy = 2 * (bbox_center[:, 1] - (cam_int[:, 1, 2])) / bs

        crop_cam_t = torch.stack(
            [tx - cx, ty - cy, tz * bbox_size / input_size], dim=-1
        )
        return crop_cam_t


# =============================================================================
# Factory functions
# =============================================================================

def create_backbone(name, cfg=None, dtype=None, device=None, operations=ops):
    if name in ["vit_hmr"]:
        backbone = vit(cfg, dtype=dtype, device=device, operations=operations)
    elif name in ["vit_hmr_512_384"]:
        backbone = vit512_384(cfg, dtype=dtype, device=device, operations=operations)
    elif name in ["vit_l"]:
        backbone = vit_l(cfg, dtype=dtype, device=device, operations=operations)
    elif name in ["vit_b"]:
        backbone = vit_b(cfg, dtype=dtype, device=device, operations=operations)
    elif name in [
        "dinov3_vit7b",
        "dinov3_vith16plus",
        "dinov3_vits16",
        "dinov3_vits16plus",
        "dinov3_vitb16",
        "dinov3_vitl16",
    ]:
        from .model_dinov3 import Dinov3Backbone
        backbone = Dinov3Backbone(name, cfg=cfg, dtype=dtype, device=device, operations=operations)
    else:
        raise NotImplementedError("Backbone type is not implemented")
    return backbone


def build_decoder(cfg, context_dim=None, dtype=None, device=None, operations=ops):
    if cfg.TYPE == "sam":
        return PromptableDecoder(
            dims=cfg.DIM,
            context_dims=context_dim,
            depth=cfg.DEPTH,
            num_heads=cfg.HEADS,
            head_dims=cfg.DIM_HEAD,
            mlp_dims=cfg.MLP_DIM,
            layer_scale_init_value=cfg.LAYER_SCALE_INIT,
            drop_rate=cfg.DROP_RATE,
            attn_drop_rate=cfg.ATTN_DROP_RATE,
            drop_path_rate=cfg.DROP_PATH_RATE,
            ffn_type=cfg.FFN_TYPE,
            enable_twoway=cfg.ENABLE_TWOWAY,
            repeat_pe=cfg.REPEAT_PE,
            frozen=cfg.get("FROZEN", False),
            do_interm_preds=cfg.get("DO_INTERM_PREDS", False),
            do_keypoint_tokens=cfg.get("DO_KEYPOINT_TOKENS", False),
            keypoint_token_update=cfg.get("KEYPOINT_TOKEN_UPDATE", None),
            dtype=dtype,
            device=device,
            operations=operations,
        )
    else:
        raise ValueError("Invalid decoder type: ", cfg.TYPE)


def build_head(cfg, head_type="mhr", enable_hand_model=False, default_scale_factor=1.0,
               dtype=None, device=None, operations=ops):
    if head_type == "mhr":
        return MHRHead(
            input_dim=cfg.MODEL.DECODER.DIM,
            mlp_depth=cfg.MODEL.MHR_HEAD.get("MLP_DEPTH", 1),
            mhr_model_path=cfg.MODEL.MHR_HEAD.MHR_MODEL_PATH,
            mlp_channel_div_factor=cfg.MODEL.MHR_HEAD.get("MLP_CHANNEL_DIV_FACTOR", 1),
            enable_hand_model=enable_hand_model,
            dtype=dtype,
            device=device,
            operations=operations,
        )
    elif head_type == "perspective":
        return PerspectiveHead(
            input_dim=cfg.MODEL.DECODER.DIM,
            img_size=to_2tuple(cfg.MODEL.IMAGE_SIZE),
            mlp_depth=cfg.MODEL.get("CAMERA_HEAD", dict()).get("MLP_DEPTH", 1),
            mlp_channel_div_factor=cfg.MODEL.get("CAMERA_HEAD", dict()).get(
                "MLP_CHANNEL_DIV_FACTOR", 1
            ),
            default_scale_factor=default_scale_factor,
            dtype=dtype,
            device=device,
            operations=operations,
        )
    else:
        raise ValueError("Invalid head type: ", head_type)


# =============================================================================
# SAM3DBody — Main model class
# =============================================================================

# fmt: off
PROMPT_KEYPOINTS = {  # keypoint_idx: prompt_idx
    "mhr70": {
        i: i for i in range(70)
    },  # all 70 keypoints are supported for prompting
}
KEY_BODY = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 41, 62]  # key body joints for prompting
KEY_RIGHT_HAND = list(range(21, 42))
# fmt: on


class SAM3DBody(BaseModel):
    pelvis_idx = [9, 10]  # left_hip, right_hip

    def _initialze_model(self, dtype=None, device=None, operations=ops, **kwargs):
        self.register_buffer(
            "image_mean", torch.tensor(self.cfg.MODEL.IMAGE_MEAN).view(-1, 1, 1), False
        )
        self.register_buffer(
            "image_std", torch.tensor(self.cfg.MODEL.IMAGE_STD).view(-1, 1, 1), False
        )

        # Create backbone feature extractor for human crops
        self.backbone = create_backbone(
            self.cfg.MODEL.BACKBONE.TYPE, self.cfg,
            dtype=dtype, device=device, operations=operations,
        )

        # Create header for pose estimation output
        self.head_pose = build_head(
            self.cfg, self.cfg.MODEL.PERSON_HEAD.POSE_TYPE,
            dtype=dtype, device=device, operations=operations,
        )
        self.head_pose.hand_pose_comps_ori = nn.Parameter(
            self.head_pose.hand_pose_comps.clone(), requires_grad=False
        )
        self.head_pose.hand_pose_comps.data = (
            torch.eye(54).to(self.head_pose.hand_pose_comps.data).float()
        )

        # Initialize pose token with learnable params
        self.init_pose = operations.Embedding(1, self.head_pose.npose, dtype=dtype, device=device)

        # Define header for hand pose estimation
        self.head_pose_hand = build_head(
            self.cfg, self.cfg.MODEL.PERSON_HEAD.POSE_TYPE, enable_hand_model=True,
            dtype=dtype, device=device, operations=operations,
        )
        self.head_pose_hand.hand_pose_comps_ori = nn.Parameter(
            self.head_pose_hand.hand_pose_comps.clone(), requires_grad=False
        )
        self.head_pose_hand.hand_pose_comps.data = (
            torch.eye(54).to(self.head_pose_hand.hand_pose_comps.data).float()
        )
        self.init_pose_hand = operations.Embedding(1, self.head_pose_hand.npose, dtype=dtype, device=device)

        self.head_camera = build_head(
            self.cfg, self.cfg.MODEL.PERSON_HEAD.CAMERA_TYPE,
            dtype=dtype, device=device, operations=operations,
        )
        self.init_camera = operations.Embedding(1, self.head_camera.ncam, dtype=dtype, device=device)
        nn.init.zeros_(self.init_camera.weight)

        self.head_camera_hand = build_head(
            self.cfg,
            self.cfg.MODEL.PERSON_HEAD.CAMERA_TYPE,
            default_scale_factor=self.cfg.MODEL.CAMERA_HEAD.get(
                "DEFAULT_SCALE_FACTOR_HAND", 1.0
            ),
            dtype=dtype, device=device, operations=operations,
        )
        self.init_camera_hand = operations.Embedding(1, self.head_camera_hand.ncam, dtype=dtype, device=device)
        nn.init.zeros_(self.init_camera_hand.weight)

        self.camera_type = "perspective"

        # Support conditioned information for decoder
        cond_dim = 3
        init_dim = self.head_pose.npose + self.head_camera.ncam + cond_dim
        self.init_to_token_mhr = operations.Linear(
            init_dim, self.cfg.MODEL.DECODER.DIM, dtype=dtype, device=device,
        )
        self.prev_to_token_mhr = operations.Linear(
            init_dim - cond_dim, self.cfg.MODEL.DECODER.DIM, dtype=dtype, device=device,
        )
        self.init_to_token_mhr_hand = operations.Linear(
            init_dim, self.cfg.MODEL.DECODER.DIM, dtype=dtype, device=device,
        )
        self.prev_to_token_mhr_hand = operations.Linear(
            init_dim - cond_dim, self.cfg.MODEL.DECODER.DIM, dtype=dtype, device=device,
        )

        # Create prompt encoder
        self.max_num_clicks = 0
        if self.cfg.MODEL.PROMPT_ENCODER.ENABLE:
            self.max_num_clicks = self.cfg.MODEL.PROMPT_ENCODER.MAX_NUM_CLICKS
            self.prompt_keypoints = PROMPT_KEYPOINTS[
                self.cfg.MODEL.PROMPT_ENCODER.PROMPT_KEYPOINTS
            ]

            self.prompt_encoder = PromptEncoder(
                embed_dim=self.backbone.embed_dims,
                num_body_joints=len(set(self.prompt_keypoints.values())),
                frozen=self.cfg.MODEL.PROMPT_ENCODER.get("frozen", False),
                mask_embed_type=self.cfg.MODEL.PROMPT_ENCODER.get(
                    "MASK_EMBED_TYPE", None
                ),
                dtype=dtype, device=device, operations=operations,
            )
            self.prompt_to_token = operations.Linear(
                self.backbone.embed_dims, self.cfg.MODEL.DECODER.DIM,
                dtype=dtype, device=device,
            )

            self.keypoint_prompt_sampler = build_keypoint_sampler(
                self.cfg.MODEL.PROMPT_ENCODER.get("KEYPOINT_SAMPLER", {}),
                prompt_keypoints=self.prompt_keypoints,
                keybody_idx=(
                    KEY_BODY
                    if not self.cfg.MODEL.PROMPT_ENCODER.get("SAMPLE_HAND", False)
                    else KEY_RIGHT_HAND
                ),
            )
            # To keep track of prompting history
            self.prompt_hist = np.zeros(
                (len(set(self.prompt_keypoints.values())) + 2, self.max_num_clicks),
                dtype=np.float32,
            )

            if self.cfg.MODEL.DECODER.FROZEN:
                for param in self.prompt_to_token.parameters():
                    param.requires_grad = False

        # Create promptable decoder
        self.decoder = build_decoder(
            self.cfg.MODEL.DECODER, context_dim=self.backbone.embed_dims,
            dtype=dtype, device=device, operations=operations,
        )
        # shared config for the two decoders
        self.decoder_hand = build_decoder(
            self.cfg.MODEL.DECODER, context_dim=self.backbone.embed_dims,
            dtype=dtype, device=device, operations=operations,
        )
        self.hand_pe_layer = PositionEmbeddingRandom(self.backbone.embed_dims // 2)

        # Derive backbone dtype from config (ComfyUI manages weight dtype via operations=)
        if self.cfg.TRAIN.USE_FP16:
            if self.cfg.TRAIN.get("FP16_TYPE", "float16") == "float16":
                self.backbone_dtype = torch.float16
            else:
                self.backbone_dtype = torch.bfloat16
        else:
            self.backbone_dtype = torch.float32

        self.ray_cond_emb = CameraEncoder(
            self.backbone.embed_dim,
            self.backbone.patch_size,
            dtype=dtype, device=device, operations=operations,
        )
        self.ray_cond_emb_hand = CameraEncoder(
            self.backbone.embed_dim,
            self.backbone.patch_size,
            dtype=dtype, device=device, operations=operations,
        )

        self.keypoint_embedding_idxs = list(range(70))
        self.keypoint_embedding = operations.Embedding(
            len(self.keypoint_embedding_idxs), self.cfg.MODEL.DECODER.DIM, dtype=dtype, device=device,
        )
        self.keypoint_embedding_idxs_hand = list(range(70))
        self.keypoint_embedding_hand = operations.Embedding(
            len(self.keypoint_embedding_idxs_hand), self.cfg.MODEL.DECODER.DIM, dtype=dtype, device=device,
        )

        if self.cfg.MODEL.DECODER.get("DO_HAND_DETECT_TOKENS", False):
            self.hand_box_embedding = operations.Embedding(
                2, self.cfg.MODEL.DECODER.DIM, dtype=dtype, device=device,
            )  # for two hands
            self.hand_cls_embed = operations.Linear(
                self.cfg.MODEL.DECODER.DIM, 2, dtype=dtype, device=device,
            )
            self.bbox_embed = MLP(
                self.cfg.MODEL.DECODER.DIM, self.cfg.MODEL.DECODER.DIM, 4, 3,
                dtype=dtype, device=device, operations=operations,
            )

        self.keypoint_posemb_linear = FFN(
            embed_dims=2,
            feedforward_channels=self.cfg.MODEL.DECODER.DIM,
            output_dims=self.cfg.MODEL.DECODER.DIM,
            num_fcs=2,
            add_identity=False,
            dtype=dtype, device=device, operations=operations,
        )
        self.keypoint_posemb_linear_hand = FFN(
            embed_dims=2,
            feedforward_channels=self.cfg.MODEL.DECODER.DIM,
            output_dims=self.cfg.MODEL.DECODER.DIM,
            num_fcs=2,
            add_identity=False,
            dtype=dtype, device=device, operations=operations,
        )
        self.keypoint_feat_linear = operations.Linear(
            self.backbone.embed_dims, self.cfg.MODEL.DECODER.DIM,
            dtype=dtype, device=device,
        )
        self.keypoint_feat_linear_hand = operations.Linear(
            self.backbone.embed_dims, self.cfg.MODEL.DECODER.DIM,
            dtype=dtype, device=device,
        )

        # Do all KPS
        self.keypoint3d_embedding_idxs = list(range(70))
        self.keypoint3d_embedding = operations.Embedding(
            len(self.keypoint3d_embedding_idxs), self.cfg.MODEL.DECODER.DIM, dtype=dtype, device=device,
        )

        # Assume always do full body for the hand decoder
        self.keypoint3d_embedding_idxs_hand = list(range(70))
        self.keypoint3d_embedding_hand = operations.Embedding(
            len(self.keypoint3d_embedding_idxs_hand), self.cfg.MODEL.DECODER.DIM, dtype=dtype, device=device,
        )

        self.keypoint3d_posemb_linear = FFN(
            embed_dims=3,
            feedforward_channels=self.cfg.MODEL.DECODER.DIM,
            output_dims=self.cfg.MODEL.DECODER.DIM,
            num_fcs=2,
            add_identity=False,
            dtype=dtype, device=device, operations=operations,
        )
        self.keypoint3d_posemb_linear_hand = FFN(
            embed_dims=3,
            feedforward_channels=self.cfg.MODEL.DECODER.DIM,
            output_dims=self.cfg.MODEL.DECODER.DIM,
            num_fcs=2,
            add_identity=False,
            dtype=dtype, device=device, operations=operations,
        )

    def _get_decoder_condition(self, batch: Dict) -> Optional[torch.Tensor]:
        num_person = batch["img"].shape[1]

        if self.cfg.MODEL.DECODER.CONDITION_TYPE == "cliff":
            # CLIFF-style condition info (cx/f, cy/f, b/f)
            cx, cy = torch.chunk(
                self._flatten_person(batch["bbox_center"]), chunks=2, dim=-1
            )
            img_w, img_h = torch.chunk(
                self._flatten_person(batch["ori_img_size"]), chunks=2, dim=-1
            )
            b = self._flatten_person(batch["bbox_scale"])[:, [0]]

            focal_length = self._flatten_person(
                batch["cam_int"]
                .unsqueeze(1)
                .expand(-1, num_person, -1, -1)
                .contiguous()
            )[:, 0, 0]
            if not self.cfg.MODEL.DECODER.get("USE_INTRIN_CENTER", False):
                condition_info = torch.cat(
                    [cx - img_w / 2.0, cy - img_h / 2.0, b], dim=-1
                )
            else:
                full_img_cxy = self._flatten_person(
                    batch["cam_int"]
                    .unsqueeze(1)
                    .expand(-1, num_person, -1, -1)
                    .contiguous()
                )[:, [0, 1], [2, 2]]
                condition_info = torch.cat(
                    [cx - full_img_cxy[:, [0]], cy - full_img_cxy[:, [1]], b], dim=-1
                )
            condition_info[:, :2] = condition_info[:, :2] / focal_length.unsqueeze(
                -1
            )  # [-1, 1]
            condition_info[:, 2] = condition_info[:, 2] / focal_length  # [-1, 1]
        elif self.cfg.MODEL.DECODER.CONDITION_TYPE == "none":
            return None
        else:
            raise NotImplementedError

        return condition_info.type(batch["img"].dtype)

    def forward_decoder(
        self,
        image_embeddings: torch.Tensor,
        init_estimate: Optional[torch.Tensor] = None,
        keypoints: Optional[torch.Tensor] = None,
        prev_estimate: Optional[torch.Tensor] = None,
        condition_info: Optional[torch.Tensor] = None,
        batch=None,
    ):
        batch_size = image_embeddings.shape[0]
        print(f"[DEBUG] forward_decoder: image_embeddings.dtype={image_embeddings.dtype} "
              f"device={image_embeddings.device} shape={image_embeddings.shape}", file=sys.stderr, flush=True)

        # Initial estimation for residual prediction.
        if init_estimate is None:
            init_pose = self.init_pose.weight.expand(batch_size, -1).unsqueeze(dim=1)
            if hasattr(self, "init_camera"):
                init_camera = self.init_camera.weight.expand(batch_size, -1).unsqueeze(
                    dim=1
                )

            init_estimate = (
                init_pose
                if not hasattr(self, "init_camera")
                else torch.cat([init_pose, init_camera], dim=-1)
            )

        if condition_info is not None:
            init_input = torch.cat(
                [condition_info.view(batch_size, 1, -1), init_estimate], dim=-1
            )
        else:
            init_input = init_estimate
        token_embeddings = self.init_to_token_mhr(init_input).view(
            batch_size, 1, -1
        )

        num_pose_token = token_embeddings.shape[1]
        assert num_pose_token == 1

        image_augment, token_augment, token_mask = None, None, None
        if hasattr(self, "prompt_encoder") and keypoints is not None:
            if prev_estimate is None:
                prev_estimate = init_estimate
            prev_embeddings = self.prev_to_token_mhr(prev_estimate).view(
                batch_size, 1, -1
            )

            if self.cfg.MODEL.BACKBONE.TYPE in [
                "vit_hmr",
                "vit",
                "vit_b",
                "vit_l",
            ]:
                image_augment = self.prompt_encoder.get_dense_pe((16, 16))[
                    :, :, :, 2:-2
                ]
            elif self.cfg.MODEL.BACKBONE.TYPE in [
                "vit_hmr_512_384",
            ]:
                image_augment = self.prompt_encoder.get_dense_pe((32, 32))[
                    :, :, :, 4:-4
                ]
            else:
                image_augment = self.prompt_encoder.get_dense_pe(
                    image_embeddings.shape[-2:]
                )

            image_embeddings = self.ray_cond_emb(image_embeddings, batch["ray_cond"])

            prompt_embeddings, prompt_mask = self.prompt_encoder(
                keypoints=keypoints
            )
            prompt_embeddings = self.prompt_to_token(prompt_embeddings)

            token_embeddings = torch.cat(
                [
                    token_embeddings,
                    prev_embeddings,
                    prompt_embeddings,
                ],
                dim=1,
            )

            token_augment = torch.zeros_like(token_embeddings)
            token_augment[:, [num_pose_token]] = prev_embeddings
            token_augment[:, (num_pose_token + 1) :] = prompt_embeddings
            token_mask = None

            if self.cfg.MODEL.DECODER.get("DO_HAND_DETECT_TOKENS", False):
                hand_det_emb_start_idx = token_embeddings.shape[1]
                token_embeddings = torch.cat(
                    [
                        token_embeddings,
                        self.hand_box_embedding.weight[None, :, :].repeat(
                            batch_size, 1, 1
                        ),
                    ],
                    dim=1,
                )
                token_augment = torch.cat(
                    [
                        token_augment,
                        torch.zeros_like(
                            token_embeddings[:, token_augment.shape[1] :, :]
                        ),
                    ],
                    dim=1,
                )

            assert self.cfg.MODEL.DECODER.get("DO_KEYPOINT_TOKENS", False)
            kps_emb_start_idx = token_embeddings.shape[1]
            token_embeddings = torch.cat(
                [
                    token_embeddings,
                    self.keypoint_embedding.weight[None, :, :].repeat(batch_size, 1, 1),
                ],
                dim=1,
            )
            token_augment = torch.cat(
                [
                    token_augment,
                    torch.zeros_like(token_embeddings[:, token_augment.shape[1] :, :]),
                ],
                dim=1,
            )
            if self.cfg.MODEL.DECODER.get("DO_KEYPOINT3D_TOKENS", False):
                kps3d_emb_start_idx = token_embeddings.shape[1]
                token_embeddings = torch.cat(
                    [
                        token_embeddings,
                        self.keypoint3d_embedding.weight[None, :, :].repeat(
                            batch_size, 1, 1
                        ),
                    ],
                    dim=1,
                )
                token_augment = torch.cat(
                    [
                        token_augment,
                        torch.zeros_like(
                            token_embeddings[:, token_augment.shape[1] :, :]
                        ),
                    ],
                    dim=1,
                )

        def token_to_pose_output_fn(tokens, prev_pose_output, layer_idx):
            pose_token = tokens[:, 0]

            prev_pose = init_pose.view(batch_size, -1)
            prev_camera = init_camera.view(batch_size, -1)

            pose_output = self.head_pose(pose_token, prev_pose)
            if hasattr(self, "head_camera"):
                pred_cam = self.head_camera(pose_token, prev_camera)
                pose_output["pred_cam"] = pred_cam
            pose_output = self.camera_project(pose_output, batch)

            pose_output["pred_keypoints_2d_cropped"] = self._full_to_crop(
                batch, pose_output["pred_keypoints_2d"], self.body_batch_idx
            )

            return pose_output

        kp_token_update_fn = self.keypoint_token_update_fn
        kp3d_token_update_fn = self.keypoint3d_token_update_fn

        def keypoint_token_update_fn_comb(*args):
            if kp_token_update_fn is not None:
                args = kp_token_update_fn(kps_emb_start_idx, image_embeddings, *args)
            if kp3d_token_update_fn is not None:
                args = kp3d_token_update_fn(kps3d_emb_start_idx, *args)
            return args

        pose_token, pose_output = self.decoder(
            token_embeddings,
            image_embeddings,
            token_augment,
            image_augment,
            token_mask,
            token_to_pose_output_fn=token_to_pose_output_fn,
            keypoint_token_update_fn=keypoint_token_update_fn_comb,
        )

        if self.cfg.MODEL.DECODER.get("DO_HAND_DETECT_TOKENS", False):
            return (
                pose_token[:, hand_det_emb_start_idx : hand_det_emb_start_idx + 2],
                pose_output,
            )
        else:
            return pose_token, pose_output

    def forward_decoder_hand(
        self,
        image_embeddings: torch.Tensor,
        init_estimate: Optional[torch.Tensor] = None,
        keypoints: Optional[torch.Tensor] = None,
        prev_estimate: Optional[torch.Tensor] = None,
        condition_info: Optional[torch.Tensor] = None,
        batch=None,
    ):
        batch_size = image_embeddings.shape[0]

        if init_estimate is None:
            init_pose = self.init_pose_hand.weight.expand(batch_size, -1).unsqueeze(
                dim=1
            )
            if hasattr(self, "init_camera_hand"):
                init_camera = self.init_camera_hand.weight.expand(
                    batch_size, -1
                ).unsqueeze(dim=1)

            init_estimate = (
                init_pose
                if not hasattr(self, "init_camera_hand")
                else torch.cat([init_pose, init_camera], dim=-1)
            )

        if condition_info is not None:
            init_input = torch.cat(
                [condition_info.view(batch_size, 1, -1), init_estimate], dim=-1
            )
        else:
            init_input = init_estimate
        token_embeddings = self.init_to_token_mhr_hand(init_input).view(
            batch_size, 1, -1
        )
        num_pose_token = token_embeddings.shape[1]

        image_augment, token_augment, token_mask = None, None, None
        if hasattr(self, "prompt_encoder") and keypoints is not None:
            if prev_estimate is None:
                prev_estimate = init_estimate
            prev_embeddings = self.prev_to_token_mhr_hand(prev_estimate).view(
                batch_size, 1, -1
            )

            if self.cfg.MODEL.BACKBONE.TYPE in [
                "vit_hmr",
                "vit",
                "vit_b",
                "vit_l",
            ]:
                image_augment = self.hand_pe_layer((16, 16)).unsqueeze(0)[:, :, :, 2:-2]
            elif self.cfg.MODEL.BACKBONE.TYPE in [
                "vit_hmr_512_384",
            ]:
                image_augment = self.hand_pe_layer((32, 32)).unsqueeze(0)[:, :, :, 4:-4]
            else:
                image_augment = self.hand_pe_layer(
                    image_embeddings.shape[-2:]
                ).unsqueeze(0)

            image_embeddings = self.ray_cond_emb_hand(
                image_embeddings, batch["ray_cond_hand"]
            )

            prompt_embeddings, prompt_mask = self.prompt_encoder(
                keypoints=keypoints
            )
            prompt_embeddings = self.prompt_to_token(prompt_embeddings)

            token_embeddings = torch.cat(
                [
                    token_embeddings,
                    prev_embeddings,
                    prompt_embeddings,
                ],
                dim=1,
            )

            token_augment = torch.zeros_like(token_embeddings)
            token_augment[:, [num_pose_token]] = prev_embeddings
            token_augment[:, (num_pose_token + 1) :] = prompt_embeddings
            token_mask = None

            if self.cfg.MODEL.DECODER.get("DO_HAND_DETECT_TOKENS", False):
                hand_det_emb_start_idx = token_embeddings.shape[1]
                token_embeddings = torch.cat(
                    [
                        token_embeddings,
                        self.hand_box_embedding.weight[None, :, :].repeat(
                            batch_size, 1, 1
                        ),
                    ],
                    dim=1,
                )
                token_augment = torch.cat(
                    [
                        token_augment,
                        torch.zeros_like(
                            token_embeddings[:, token_augment.shape[1] :, :]
                        ),
                    ],
                    dim=1,
                )

            assert self.cfg.MODEL.DECODER.get("DO_KEYPOINT_TOKENS", False)
            kps_emb_start_idx = token_embeddings.shape[1]
            token_embeddings = torch.cat(
                [
                    token_embeddings,
                    self.keypoint_embedding_hand.weight[None, :, :].repeat(
                        batch_size, 1, 1
                    ),
                ],
                dim=1,
            )
            token_augment = torch.cat(
                [
                    token_augment,
                    torch.zeros_like(token_embeddings[:, token_augment.shape[1] :, :]),
                ],
                dim=1,
            )

            if self.cfg.MODEL.DECODER.get("DO_KEYPOINT3D_TOKENS", False):
                kps3d_emb_start_idx = token_embeddings.shape[1]
                token_embeddings = torch.cat(
                    [
                        token_embeddings,
                        self.keypoint3d_embedding_hand.weight[None, :, :].repeat(
                            batch_size, 1, 1
                        ),
                    ],
                    dim=1,
                )
                token_augment = torch.cat(
                    [
                        token_augment,
                        torch.zeros_like(
                            token_embeddings[:, token_augment.shape[1] :, :]
                        ),
                    ],
                    dim=1,
                )

        def token_to_pose_output_fn(tokens, prev_pose_output, layer_idx):
            pose_token = tokens[:, 0]

            prev_pose = init_pose.view(batch_size, -1)
            prev_camera = init_camera.view(batch_size, -1)

            pose_output = self.head_pose_hand(pose_token, prev_pose)
            if hasattr(self, "head_camera_hand"):
                pred_cam = self.head_camera_hand(pose_token, prev_camera)
                pose_output["pred_cam"] = pred_cam
            pose_output = self.camera_project_hand(pose_output, batch)

            pose_output["pred_keypoints_2d_cropped"] = self._full_to_crop(
                batch, pose_output["pred_keypoints_2d"], self.hand_batch_idx
            )

            return pose_output

        kp_token_update_fn = self.keypoint_token_update_fn_hand
        kp3d_token_update_fn = self.keypoint3d_token_update_fn_hand

        def keypoint_token_update_fn_comb(*args):
            if kp_token_update_fn is not None:
                args = kp_token_update_fn(kps_emb_start_idx, image_embeddings, *args)
            if kp3d_token_update_fn is not None:
                args = kp3d_token_update_fn(kps3d_emb_start_idx, *args)
            return args

        pose_token, pose_output = self.decoder_hand(
            token_embeddings,
            image_embeddings,
            token_augment,
            image_augment,
            token_mask,
            token_to_pose_output_fn=token_to_pose_output_fn,
            keypoint_token_update_fn=keypoint_token_update_fn_comb,
        )

        if self.cfg.MODEL.DECODER.get("DO_HAND_DETECT_TOKENS", False):
            return (
                pose_token[:, hand_det_emb_start_idx : hand_det_emb_start_idx + 2],
                pose_output,
            )
        else:
            return pose_token, pose_output

    @torch.no_grad()
    def _get_keypoint_prompt(self, batch, pred_keypoints_2d, force_dummy=False):
        if self.camera_type == "perspective":
            pred_keypoints_2d = self._full_to_crop(batch, pred_keypoints_2d)

        gt_keypoints_2d = self._flatten_person(batch["keypoints_2d"]).clone()

        keypoint_prompt = self.keypoint_prompt_sampler.sample(
            gt_keypoints_2d,
            pred_keypoints_2d,
            is_train=self.training,
            force_dummy=force_dummy,
        )
        return keypoint_prompt

    def _get_mask_prompt(self, batch, image_embeddings):
        x_mask = self._flatten_person(batch["mask"])
        mask_embeddings, no_mask_embeddings = self.prompt_encoder.get_mask_embeddings(
            x_mask, image_embeddings.shape[0], image_embeddings.shape[2:]
        )
        if self.cfg.MODEL.BACKBONE.TYPE in [
            "vit_hmr",
            "vit",
        ]:
            mask_embeddings = mask_embeddings[:, :, :, 2:-2]
        elif self.cfg.MODEL.BACKBONE.TYPE in [
            "vit_hmr_512_384",
        ]:
            mask_embeddings = mask_embeddings[:, :, :, 4:-4]

        mask_score = self._flatten_person(batch["mask_score"]).view(-1, 1, 1, 1)
        mask_embeddings = torch.where(
            mask_score > 0,
            mask_score * mask_embeddings.to(image_embeddings),
            no_mask_embeddings.to(image_embeddings),
        )
        return mask_embeddings

    def _one_prompt_iter(self, batch, output, prev_prompt, full_output):
        image_embeddings = output["image_embeddings"]
        condition_info = output["condition_info"]

        if "mhr" in output and output["mhr"] is not None:
            pose_output = output["mhr"]
            prev_estimate = torch.cat(
                [
                    pose_output["pred_pose_raw"].detach(),
                    pose_output["shape"].detach(),
                    pose_output["scale"].detach(),
                    pose_output["hand"].detach(),
                    pose_output["face"].detach(),
                ],
                dim=1,
            ).unsqueeze(dim=1)
            if hasattr(self, "init_camera"):
                prev_estimate = torch.cat(
                    [prev_estimate, pose_output["pred_cam"].detach().unsqueeze(1)],
                    dim=-1,
                )
            prev_shape = prev_estimate.shape[1:]

            pred_keypoints_2d = output["mhr"]["pred_keypoints_2d"].detach().clone()
            kpt_shape = pred_keypoints_2d.shape[1:]

        if "mhr_hand" in output and output["mhr_hand"] is not None:
            pose_output_hand = output["mhr_hand"]
            prev_estimate_hand = torch.cat(
                [
                    pose_output_hand["pred_pose_raw"].detach(),
                    pose_output_hand["shape"].detach(),
                    pose_output_hand["scale"].detach(),
                    pose_output_hand["hand"].detach(),
                    pose_output_hand["face"].detach(),
                ],
                dim=1,
            ).unsqueeze(dim=1)
            if hasattr(self, "init_camera_hand"):
                prev_estimate_hand = torch.cat(
                    [
                        prev_estimate_hand,
                        pose_output_hand["pred_cam"].detach().unsqueeze(1),
                    ],
                    dim=-1,
                )
            prev_shape = prev_estimate_hand.shape[1:]

            pred_keypoints_2d_hand = (
                output["mhr_hand"]["pred_keypoints_2d"].detach().clone()
            )
            kpt_shape = pred_keypoints_2d_hand.shape[1:]

        all_prev_estimate = torch.zeros(
            (image_embeddings.shape[0], *prev_shape), device=image_embeddings.device
        )
        if "mhr" in output and output["mhr"] is not None:
            all_prev_estimate[self.body_batch_idx] = prev_estimate
        if "mhr_hand" in output and output["mhr_hand"] is not None:
            all_prev_estimate[self.hand_batch_idx] = prev_estimate_hand

        all_pred_keypoints_2d = torch.zeros(
            (image_embeddings.shape[0], *kpt_shape), device=image_embeddings.device
        )
        if "mhr" in output and output["mhr"] is not None:
            all_pred_keypoints_2d[self.body_batch_idx] = pred_keypoints_2d
        if "mhr_hand" in output and output["mhr_hand"] is not None:
            all_pred_keypoints_2d[self.hand_batch_idx] = pred_keypoints_2d_hand

        keypoint_prompt = self._get_keypoint_prompt(batch, all_pred_keypoints_2d)
        if len(prev_prompt):
            cur_keypoint_prompt = torch.cat(prev_prompt + [keypoint_prompt], dim=1)
        else:
            cur_keypoint_prompt = keypoint_prompt

        pose_output, pose_output_hand = None, None
        if len(self.body_batch_idx):
            tokens_output, pose_output = self.forward_decoder(
                image_embeddings[self.body_batch_idx],
                init_estimate=None,
                keypoints=cur_keypoint_prompt[self.body_batch_idx],
                prev_estimate=all_prev_estimate[self.body_batch_idx],
                condition_info=condition_info[self.body_batch_idx],
                batch=batch,
            )
            pose_output = pose_output[-1]

        output.update(
            {
                "mhr": pose_output,
                "mhr_hand": pose_output_hand,
            }
        )

        return output, keypoint_prompt

    def _full_to_crop(
        self,
        batch: Dict,
        pred_keypoints_2d: torch.Tensor,
        batch_idx: torch.Tensor = None,
    ) -> torch.Tensor:
        """Convert full-image keypoints coordinates to crop and normalize to [-0.5, 0.5]"""
        pred_keypoints_2d_cropped = torch.cat(
            [pred_keypoints_2d, torch.ones_like(pred_keypoints_2d[:, :, [-1]])], dim=-1
        )
        if batch_idx is not None:
            affine_trans = self._flatten_person(batch["affine_trans"])[batch_idx].to(
                pred_keypoints_2d_cropped
            )
            img_size = self._flatten_person(batch["img_size"])[batch_idx].unsqueeze(1)
        else:
            affine_trans = self._flatten_person(batch["affine_trans"]).to(
                pred_keypoints_2d_cropped
            )
            img_size = self._flatten_person(batch["img_size"]).unsqueeze(1)
        pred_keypoints_2d_cropped = pred_keypoints_2d_cropped @ affine_trans.mT
        pred_keypoints_2d_cropped = pred_keypoints_2d_cropped[..., :2] / img_size - 0.5

        return pred_keypoints_2d_cropped

    def camera_project(self, pose_output: Dict, batch: Dict) -> Dict:
        if hasattr(self, "head_camera"):
            head_camera = self.head_camera
            pred_cam = pose_output["pred_cam"]
        else:
            assert False

        cam_out = head_camera.perspective_projection(
            pose_output["pred_keypoints_3d"],
            pred_cam,
            self._flatten_person(batch["bbox_center"])[self.body_batch_idx],
            self._flatten_person(batch["bbox_scale"])[self.body_batch_idx, 0],
            self._flatten_person(batch["ori_img_size"])[self.body_batch_idx],
            self._flatten_person(
                batch["cam_int"]
                .unsqueeze(1)
                .expand(-1, batch["img"].shape[1], -1, -1)
                .contiguous()
            )[self.body_batch_idx],
            use_intrin_center=self.cfg.MODEL.DECODER.get("USE_INTRIN_CENTER", False),
        )

        if pose_output.get("pred_vertices", None) is not None:
            cam_out_vertices = head_camera.perspective_projection(
                pose_output["pred_vertices"],
                pred_cam,
                self._flatten_person(batch["bbox_center"])[self.body_batch_idx],
                self._flatten_person(batch["bbox_scale"])[self.body_batch_idx, 0],
                self._flatten_person(batch["ori_img_size"])[self.body_batch_idx],
                self._flatten_person(
                    batch["cam_int"]
                    .unsqueeze(1)
                    .expand(-1, batch["img"].shape[1], -1, -1)
                    .contiguous()
                )[self.body_batch_idx],
                use_intrin_center=self.cfg.MODEL.DECODER.get(
                    "USE_INTRIN_CENTER", False
                ),
            )
            pose_output["pred_keypoints_2d_verts"] = cam_out_vertices[
                "pred_keypoints_2d"
            ]

        pose_output.update(cam_out)

        return pose_output

    def camera_project_hand(self, pose_output: Dict, batch: Dict) -> Dict:
        if hasattr(self, "head_camera_hand"):
            head_camera = self.head_camera_hand
            pred_cam = pose_output["pred_cam"]
        else:
            assert False

        cam_out = head_camera.perspective_projection(
            pose_output["pred_keypoints_3d"],
            pred_cam,
            self._flatten_person(batch["bbox_center"])[self.hand_batch_idx],
            self._flatten_person(batch["bbox_scale"])[self.hand_batch_idx, 0],
            self._flatten_person(batch["ori_img_size"])[self.hand_batch_idx],
            self._flatten_person(
                batch["cam_int"]
                .unsqueeze(1)
                .expand(-1, batch["img"].shape[1], -1, -1)
                .contiguous()
            )[self.hand_batch_idx],
            use_intrin_center=self.cfg.MODEL.DECODER.get("USE_INTRIN_CENTER", False),
        )

        if pose_output.get("pred_vertices", None) is not None:
            cam_out_vertices = head_camera.perspective_projection(
                pose_output["pred_vertices"],
                pred_cam,
                self._flatten_person(batch["bbox_center"])[self.hand_batch_idx],
                self._flatten_person(batch["bbox_scale"])[self.hand_batch_idx, 0],
                self._flatten_person(batch["ori_img_size"])[self.hand_batch_idx],
                self._flatten_person(
                    batch["cam_int"]
                    .unsqueeze(1)
                    .expand(-1, batch["img"].shape[1], -1, -1)
                    .contiguous()
                )[self.hand_batch_idx],
                use_intrin_center=self.cfg.MODEL.DECODER.get(
                    "USE_INTRIN_CENTER", False
                ),
            )
            pose_output["pred_keypoints_2d_verts"] = cam_out_vertices[
                "pred_keypoints_2d"
            ]

        pose_output.update(cam_out)

        return pose_output

    def get_ray_condition(self, batch):
        B, N, _, H, W = batch["img"].shape
        meshgrid_xy = (
            torch.stack(
                torch.meshgrid(torch.arange(H), torch.arange(W), indexing="xy"), dim=2
            )[None, None, :, :, :]
            .repeat(B, N, 1, 1, 1)
            .to(self.image_mean.device)
        )
        meshgrid_xy = (
            meshgrid_xy / batch["affine_trans"][:, :, None, None, [0, 1], [0, 1]]
        )
        meshgrid_xy = (
            meshgrid_xy
            - batch["affine_trans"][:, :, None, None, [0, 1], [2, 2]]
            / batch["affine_trans"][:, :, None, None, [0, 1], [0, 1]]
        )

        meshgrid_xy = (
            meshgrid_xy - batch["cam_int"][:, None, None, None, [0, 1], [2, 2]]
        )
        meshgrid_xy = (
            meshgrid_xy / batch["cam_int"][:, None, None, None, [0, 1], [0, 1]]
        )

        return meshgrid_xy.permute(0, 1, 4, 2, 3).to(batch["img"].dtype)

    def forward_pose_branch(self, batch: Dict) -> Dict:
        """Run a forward pass for the crop-image (pose) branch."""
        batch_size, num_person = batch["img"].shape[:2]

        # Forward backbone encoder
        x = self.data_preprocess(
            self._flatten_person(batch["img"]),
            crop_width=(
                self.cfg.MODEL.BACKBONE.TYPE
                in [
                    "vit_hmr",
                    "vit",
                    "vit_b",
                    "vit_l",
                    "vit_hmr_512_384",
                ]
            ),
        )

        # Optionally get ray conditioning
        ray_cond = self.get_ray_condition(batch)
        ray_cond = self._flatten_person(ray_cond)
        if self.cfg.MODEL.BACKBONE.TYPE in [
            "vit_hmr",
            "vit",
            "vit_b",
            "vit_l",
        ]:
            ray_cond = ray_cond[:, :, :, 32:-32]
        elif self.cfg.MODEL.BACKBONE.TYPE in [
            "vit_hmr_512_384",
        ]:
            ray_cond = ray_cond[:, :, :, 64:-64]

        if len(self.body_batch_idx):
            batch["ray_cond"] = ray_cond[self.body_batch_idx].clone()
        if len(self.hand_batch_idx):
            batch["ray_cond_hand"] = ray_cond[self.hand_batch_idx].clone()
        ray_cond = None

        backbone_input = x.type(self.backbone_dtype)
        print(f"[DEBUG] forward_pose_branch: x.dtype={x.dtype} backbone_dtype={self.backbone_dtype} backbone_input.dtype={backbone_input.dtype} backbone_input.device={backbone_input.device}", file=sys.stderr, flush=True)

        image_embeddings = self.backbone(
            backbone_input, extra_embed=ray_cond
        )

        if isinstance(image_embeddings, tuple):
            image_embeddings = image_embeddings[-1]

        print(f"[DEBUG] forward_pose_branch: backbone_output.dtype={image_embeddings.dtype} -> casting to x.dtype={x.dtype}", file=sys.stderr, flush=True)
        image_embeddings = image_embeddings.type(x.dtype)

        # Mask condition if available
        if self.cfg.MODEL.PROMPT_ENCODER.get("MASK_EMBED_TYPE", None) is not None:
            if self.cfg.MODEL.PROMPT_ENCODER.get("MASK_PROMPT", "v1") == "v1":
                mask_embeddings = self._get_mask_prompt(batch, image_embeddings)
                image_embeddings = image_embeddings + mask_embeddings
            else:
                raise NotImplementedError

        # Prepare input for promptable decoder
        condition_info = self._get_decoder_condition(batch)

        # Initial estimate with a dummy prompt
        keypoints_prompt = torch.zeros((batch_size * num_person, 1, 3)).to(batch["img"])
        keypoints_prompt[:, :, -1] = -2

        # Forward promptable decoder to get updated pose tokens and regression output
        pose_output, pose_output_hand = None, None
        if len(self.body_batch_idx):
            tokens_output, pose_output = self.forward_decoder(
                image_embeddings[self.body_batch_idx],
                init_estimate=None,
                keypoints=keypoints_prompt[self.body_batch_idx],
                prev_estimate=None,
                condition_info=condition_info[self.body_batch_idx],
                batch=batch,
            )
            pose_output = pose_output[-1]
        if len(self.hand_batch_idx):
            tokens_output_hand, pose_output_hand = self.forward_decoder_hand(
                image_embeddings[self.hand_batch_idx],
                init_estimate=None,
                keypoints=keypoints_prompt[self.hand_batch_idx],
                prev_estimate=None,
                condition_info=condition_info[self.hand_batch_idx],
                batch=batch,
            )
            pose_output_hand = pose_output_hand[-1]

        output = {
            "mhr": pose_output,
            "mhr_hand": pose_output_hand,
            "condition_info": condition_info,
            "image_embeddings": image_embeddings,
        }

        if self.cfg.MODEL.DECODER.get("DO_HAND_DETECT_TOKENS", False):
            if len(self.body_batch_idx):
                output_hand_box_tokens = tokens_output
                hand_coords = self.bbox_embed(
                    output_hand_box_tokens
                ).sigmoid()
                hand_logits = self.hand_cls_embed(output_hand_box_tokens)

                output["mhr"]["hand_box"] = hand_coords
                output["mhr"]["hand_logits"] = hand_logits

            if len(self.hand_batch_idx):
                output_hand_box_tokens_hand_batch = tokens_output_hand

                hand_coords_hand_batch = self.bbox_embed(
                    output_hand_box_tokens_hand_batch
                ).sigmoid()
                hand_logits_hand_batch = self.hand_cls_embed(
                    output_hand_box_tokens_hand_batch
                )

                output["mhr_hand"]["hand_box"] = hand_coords_hand_batch
                output["mhr_hand"]["hand_logits"] = hand_logits_hand_batch

        return output

    def forward_step(
        self, batch: Dict, decoder_type: str = "body"
    ) -> Tuple[Dict, Dict]:
        batch_size, num_person = batch["img"].shape[:2]

        if decoder_type == "body":
            self.hand_batch_idx = []
            self.body_batch_idx = list(range(batch_size * num_person))
        elif decoder_type == "hand":
            self.hand_batch_idx = list(range(batch_size * num_person))
            self.body_batch_idx = []
        else:
            ValueError("Invalid decoder type: ", decoder_type)

        # Crop-image (pose) branch
        pose_output = self.forward_pose_branch(batch)

        return pose_output

    def run_inference(
        self,
        img,
        batch: Dict,
        inference_type: str = "full",
        transform_hand: Any = None,
        thresh_wrist_angle=1.4,
    ):
        height, width = img.shape[:2]
        cam_int = batch["cam_int"].clone()

        if inference_type == "body":
            pose_output = self.forward_step(batch, decoder_type="body")
            return pose_output
        elif inference_type == "hand":
            pose_output = self.forward_step(batch, decoder_type="hand")
            return pose_output
        elif not inference_type == "full":
            ValueError("Invalid inference type: ", inference_type)

        # Step 1. For full-body inference, we first inference with the body decoder.
        pose_output = self.forward_step(batch, decoder_type="body")
        left_xyxy, right_xyxy = self._get_hand_box(pose_output, batch)
        ori_local_wrist_rotmat = roma.euler_to_rotmat(
            "XZY",
            pose_output["mhr"]["body_pose"][:, [41, 43, 42, 31, 33, 32]].unflatten(
                1, (2, 3)
            ),
        )

        # Step 2. Re-run with each hand
        ## Left... Flip image & box
        flipped_img = img[:, ::-1]
        tmp = left_xyxy.copy()
        left_xyxy[:, 0] = width - tmp[:, 2] - 1
        left_xyxy[:, 2] = width - tmp[:, 0] - 1

        batch_lhand = prepare_batch(
            flipped_img, transform_hand, left_xyxy, cam_int=cam_int.clone()
        )
        batch_lhand = recursive_to(batch_lhand, self.image_mean.device)
        lhand_output = self.forward_step(batch_lhand, decoder_type="hand")

        # Unflip output
        ## Flip scale
        ### Get MHR values
        scale_r_hands_mean = self.head_pose.scale_mean[8].item()
        scale_l_hands_mean = self.head_pose.scale_mean[9].item()
        scale_r_hands_std = self.head_pose.scale_comps[8, 8].item()
        scale_l_hands_std = self.head_pose.scale_comps[9, 9].item()
        ### Apply
        lhand_output["mhr_hand"]["scale"][:, 9] = (
            (
                scale_r_hands_mean
                + scale_r_hands_std * lhand_output["mhr_hand"]["scale"][:, 8]
            )
            - scale_l_hands_mean
        ) / scale_l_hands_std
        ## Get the right hand global rotation, flip it, put it in as left.
        lhand_output["mhr_hand"]["joint_global_rots"][:, 78] = lhand_output["mhr_hand"][
            "joint_global_rots"
        ][:, 42].clone()
        lhand_output["mhr_hand"]["joint_global_rots"][:, 78, [1, 2], :] *= -1
        ### Flip hand pose
        lhand_output["mhr_hand"]["hand"][:, :54] = lhand_output["mhr_hand"]["hand"][
            :, 54:
        ]
        ### Unflip box
        batch_lhand["bbox_center"][:, :, 0] = (
            width - batch_lhand["bbox_center"][:, :, 0] - 1
        )

        ## Right...
        batch_rhand = prepare_batch(
            img, transform_hand, right_xyxy, cam_int=cam_int.clone()
        )
        batch_rhand = recursive_to(batch_rhand, self.image_mean.device)
        rhand_output = self.forward_step(batch_rhand, decoder_type="hand")

        # Step 3. replace hand pose estimation from the body decoder.
        ## CRITERIA 1: LOCAL WRIST POSE DIFFERENCE
        joint_rotations = pose_output["mhr"]["joint_global_rots"]
        ### Get lowarm
        lowarm_joint_idxs = torch.LongTensor([76, 40]).to(self.image_mean.device)
        lowarm_joint_rotations = joint_rotations[:, lowarm_joint_idxs]
        ### Get zero-wrist pose
        wrist_twist_joint_idxs = torch.LongTensor([77, 41]).to(self.image_mean.device)
        wrist_zero_rot_pose = (
            lowarm_joint_rotations
            @ self.head_pose.joint_rotation[wrist_twist_joint_idxs]
        )
        ### Get globals from left & right
        left_joint_global_rots = lhand_output["mhr_hand"]["joint_global_rots"]
        right_joint_global_rots = rhand_output["mhr_hand"]["joint_global_rots"]
        pred_global_wrist_rotmat = torch.stack(
            [
                left_joint_global_rots[:, 78],
                right_joint_global_rots[:, 42],
            ],
            dim=1,
        )
        ### Get the local poses that lead to the wrist being pred_global_wrist_rotmat
        fused_local_wrist_rotmat = torch.einsum(
            "kabc,kabd->kadc", pred_global_wrist_rotmat, wrist_zero_rot_pose
        )
        angle_difference = rotation_angle_difference(
            ori_local_wrist_rotmat, fused_local_wrist_rotmat
        )
        angle_difference_valid_mask = angle_difference < thresh_wrist_angle

        ## CRITERIA 2: hand box size
        hand_box_size_thresh = 64
        hand_box_size_valid_mask = torch.stack(
            [
                (batch_lhand["bbox_scale"].flatten(0, 1) > hand_box_size_thresh).all(
                    dim=1
                ),
                (batch_rhand["bbox_scale"].flatten(0, 1) > hand_box_size_thresh).all(
                    dim=1
                ),
            ],
            dim=1,
        )

        ## CRITERIA 3: all hand 2D KPS (including wrist) inside of box.
        hand_kps2d_thresh = 0.5
        hand_kps2d_valid_mask = torch.stack(
            [
                lhand_output["mhr_hand"]["pred_keypoints_2d_cropped"]
                .abs()
                .amax(dim=(1, 2))
                < hand_kps2d_thresh,
                rhand_output["mhr_hand"]["pred_keypoints_2d_cropped"]
                .abs()
                .amax(dim=(1, 2))
                < hand_kps2d_thresh,
            ],
            dim=1,
        )

        ## CRITERIA 4: 2D wrist distance.
        hand_wrist_kps2d_thresh = 0.25
        kps_right_wrist_idx = 41
        kps_left_wrist_idx = 62
        right_kps_full = rhand_output["mhr_hand"]["pred_keypoints_2d"][
            :, [kps_right_wrist_idx]
        ].clone()
        left_kps_full = lhand_output["mhr_hand"]["pred_keypoints_2d"][
            :, [kps_right_wrist_idx]
        ].clone()
        left_kps_full[:, :, 0] = width - left_kps_full[:, :, 0] - 1
        body_right_kps_full = pose_output["mhr"]["pred_keypoints_2d"][
            :, [kps_right_wrist_idx]
        ].clone()
        body_left_kps_full = pose_output["mhr"]["pred_keypoints_2d"][
            :, [kps_left_wrist_idx]
        ].clone()
        right_kps_dist = (right_kps_full - body_right_kps_full).flatten(0, 1).norm(
            dim=-1
        ) / batch_lhand["bbox_scale"].flatten(0, 1)[:, 0]
        left_kps_dist = (left_kps_full - body_left_kps_full).flatten(0, 1).norm(
            dim=-1
        ) / batch_rhand["bbox_scale"].flatten(0, 1)[:, 0]
        hand_wrist_kps2d_valid_mask = torch.stack(
            [
                left_kps_dist < hand_wrist_kps2d_thresh,
                right_kps_dist < hand_wrist_kps2d_thresh,
            ],
            dim=1,
        )
        ## Left-right
        hand_valid_mask = (
            angle_difference_valid_mask
            & hand_box_size_valid_mask
            & hand_kps2d_valid_mask
            & hand_wrist_kps2d_valid_mask
        )

        # Keypoint prompting with the body decoder.
        batch_size, num_person = batch["img"].shape[:2]
        self.hand_batch_idx = []
        self.body_batch_idx = list(range(batch_size * num_person))

        ## Get right & left wrist keypoints from crops; full image. Each are B x 1 x 2
        kps_right_wrist_idx = 41
        kps_left_wrist_idx = 62
        right_kps_full = rhand_output["mhr_hand"]["pred_keypoints_2d"][
            :, [kps_right_wrist_idx]
        ].clone()
        left_kps_full = lhand_output["mhr_hand"]["pred_keypoints_2d"][
            :, [kps_right_wrist_idx]
        ].clone()
        left_kps_full[:, :, 0] = width - left_kps_full[:, :, 0] - 1

        # Next, get them to crop-normalized space.
        right_kps_crop = self._full_to_crop(batch, right_kps_full)
        left_kps_crop = self._full_to_crop(batch, left_kps_full)

        # Get right & left elbow keypoints from crops; full image. Each are B x 1 x 2
        kps_right_elbow_idx = 8
        kps_left_elbow_idx = 7
        right_kps_elbow_full = pose_output["mhr"]["pred_keypoints_2d"][
            :, [kps_right_elbow_idx]
        ].clone()
        left_kps_elbow_full = pose_output["mhr"]["pred_keypoints_2d"][
            :, [kps_left_elbow_idx]
        ].clone()

        # Next, get them to crop-normalized space.
        right_kps_elbow_crop = self._full_to_crop(batch, right_kps_elbow_full)
        left_kps_elbow_crop = self._full_to_crop(batch, left_kps_elbow_full)

        # Assemble them into keypoint prompts
        keypoint_prompt = torch.cat(
            [right_kps_crop, left_kps_crop, right_kps_elbow_crop, left_kps_elbow_crop],
            dim=1,
        )
        keypoint_prompt = torch.cat(
            [keypoint_prompt, keypoint_prompt[..., [-1]]], dim=-1
        )
        keypoint_prompt[:, 0, -1] = kps_right_wrist_idx
        keypoint_prompt[:, 1, -1] = kps_left_wrist_idx
        keypoint_prompt[:, 2, -1] = kps_right_elbow_idx
        keypoint_prompt[:, 3, -1] = kps_left_elbow_idx

        if keypoint_prompt.shape[0] > 1:
            # Replace invalid keypoints to dummy prompts
            invalid_prompt = (
                (keypoint_prompt[..., 0] < -0.5)
                | (keypoint_prompt[..., 0] > 0.5)
                | (keypoint_prompt[..., 1] < -0.5)
                | (keypoint_prompt[..., 1] > 0.5)
                | (~hand_valid_mask[..., [1, 0, 1, 0]])
            ).unsqueeze(-1)
            dummy_prompt = torch.zeros((1, 1, 3)).to(keypoint_prompt)
            dummy_prompt[:, :, -1] = -2
            keypoint_prompt[:, :, :2] = torch.clamp(
                keypoint_prompt[:, :, :2] + 0.5, min=0.0, max=1.0
            )
            keypoint_prompt = torch.where(invalid_prompt, dummy_prompt, keypoint_prompt)
        else:
            # Only keep valid keypoints
            valid_keypoint = (
                torch.all(
                    (keypoint_prompt[:, :, :2] > -0.5)
                    & (keypoint_prompt[:, :, :2] < 0.5),
                    dim=2,
                )
                & hand_valid_mask[..., [1, 0, 1, 0]]
            ).squeeze()
            keypoint_prompt = keypoint_prompt[:, valid_keypoint]
            keypoint_prompt[:, :, :2] = torch.clamp(
                keypoint_prompt[:, :, :2] + 0.5, min=0.0, max=1.0
            )

        if keypoint_prompt.numel() != 0:
            pose_output, _ = self.run_keypoint_prompt(
                batch, pose_output, keypoint_prompt
            )

        ##############################################################################

        # Drop in hand pose
        left_hand_pose_params = lhand_output["mhr_hand"]["hand"][:, :54]
        right_hand_pose_params = rhand_output["mhr_hand"]["hand"][:, 54:]
        updated_hand_pose = torch.cat(
            [left_hand_pose_params, right_hand_pose_params], dim=1
        )

        # Drop in hand scales
        updated_scale = pose_output["mhr"]["scale"].clone()
        updated_scale[:, 9] = lhand_output["mhr_hand"]["scale"][:, 9]
        updated_scale[:, 8] = rhand_output["mhr_hand"]["scale"][:, 8]
        updated_scale[:, 18:] = (
            lhand_output["mhr_hand"]["scale"][:, 18:]
            + rhand_output["mhr_hand"]["scale"][:, 18:]
        ) / 2

        # Update hand shape
        updated_shape = pose_output["mhr"]["shape"].clone()
        updated_shape[:, 40:] = (
            lhand_output["mhr_hand"]["shape"][:, 40:]
            + rhand_output["mhr_hand"]["shape"][:, 40:]
        ) / 2

        ############################ Doing IK ############################

        # First, forward just FK
        joint_rotations = self.head_pose.mhr_forward(
            global_trans=pose_output["mhr"]["global_rot"] * 0,
            global_rot=pose_output["mhr"]["global_rot"],
            body_pose_params=pose_output["mhr"]["body_pose"],
            hand_pose_params=updated_hand_pose,
            scale_params=updated_scale,
            shape_params=updated_shape,
            expr_params=pose_output["mhr"]["face"],
            return_joint_rotations=True,
        )[1]

        # Get lowarm
        lowarm_joint_idxs = torch.LongTensor([76, 40]).to(self.image_mean.device)
        lowarm_joint_rotations = joint_rotations[:, lowarm_joint_idxs]

        # Get zero-wrist pose
        wrist_twist_joint_idxs = torch.LongTensor([77, 41]).to(self.image_mean.device)
        wrist_zero_rot_pose = (
            lowarm_joint_rotations
            @ self.head_pose.joint_rotation[wrist_twist_joint_idxs]
        )

        # Get globals from left & right
        left_joint_global_rots = lhand_output["mhr_hand"]["joint_global_rots"]
        right_joint_global_rots = rhand_output["mhr_hand"]["joint_global_rots"]
        pred_global_wrist_rotmat = torch.stack(
            [
                left_joint_global_rots[:, 78],
                right_joint_global_rots[:, 42],
            ],
            dim=1,
        )

        # Now we want to get the local poses that lead to the wrist being pred_global_wrist_rotmat
        fused_local_wrist_rotmat = torch.einsum(
            "kabc,kabd->kadc", pred_global_wrist_rotmat, wrist_zero_rot_pose
        )
        wrist_xzy = fix_wrist_euler(
            roma.rotmat_to_euler("XZY", fused_local_wrist_rotmat)
        )

        # Put it in.
        angle_difference = rotation_angle_difference(
            ori_local_wrist_rotmat, fused_local_wrist_rotmat
        )
        valid_angle = angle_difference < thresh_wrist_angle
        valid_angle = valid_angle & hand_valid_mask
        valid_angle = valid_angle.unsqueeze(-1)

        body_pose = pose_output["mhr"]["body_pose"][
            :, [41, 43, 42, 31, 33, 32]
        ].unflatten(1, (2, 3))
        updated_body_pose = torch.where(valid_angle, wrist_xzy, body_pose)
        pose_output["mhr"]["body_pose"][:, [41, 43, 42, 31, 33, 32]] = (
            updated_body_pose.flatten(1, 2)
        )

        hand_pose = pose_output["mhr"]["hand"].unflatten(1, (2, 54))
        pose_output["mhr"]["hand"] = torch.where(
            valid_angle, updated_hand_pose.unflatten(1, (2, 54)), hand_pose
        ).flatten(1, 2)

        hand_scale = torch.stack(
            [pose_output["mhr"]["scale"][:, 9], pose_output["mhr"]["scale"][:, 8]],
            dim=1,
        )
        updated_hand_scale = torch.stack(
            [updated_scale[:, 9], updated_scale[:, 8]], dim=1
        )
        masked_hand_scale = torch.where(
            valid_angle.squeeze(-1), updated_hand_scale, hand_scale
        )
        pose_output["mhr"]["scale"][:, 9] = masked_hand_scale[:, 0]
        pose_output["mhr"]["scale"][:, 8] = masked_hand_scale[:, 1]

        # Replace shared shape and scale
        pose_output["mhr"]["scale"][:, 18:] = torch.where(
            valid_angle.squeeze(-1).sum(dim=1, keepdim=True) > 0,
            (
                lhand_output["mhr_hand"]["scale"][:, 18:]
                * valid_angle.squeeze(-1)[:, [0]]
                + rhand_output["mhr_hand"]["scale"][:, 18:]
                * valid_angle.squeeze(-1)[:, [1]]
            )
            / (valid_angle.squeeze(-1).sum(dim=1, keepdim=True) + 1e-8),
            pose_output["mhr"]["scale"][:, 18:],
        )
        pose_output["mhr"]["shape"][:, 40:] = torch.where(
            valid_angle.squeeze(-1).sum(dim=1, keepdim=True) > 0,
            (
                lhand_output["mhr_hand"]["shape"][:, 40:]
                * valid_angle.squeeze(-1)[:, [0]]
                + rhand_output["mhr_hand"]["shape"][:, 40:]
                * valid_angle.squeeze(-1)[:, [1]]
            )
            / (valid_angle.squeeze(-1).sum(dim=1, keepdim=True) + 1e-8),
            pose_output["mhr"]["shape"][:, 40:],
        )

        ########################################################

        # Re-run forward
        with torch.no_grad():
            verts, j3d, jcoords, mhr_model_params, joint_global_rots = (
                self.head_pose.mhr_forward(
                    global_trans=pose_output["mhr"]["global_rot"] * 0,
                    global_rot=pose_output["mhr"]["global_rot"],
                    body_pose_params=pose_output["mhr"]["body_pose"],
                    hand_pose_params=pose_output["mhr"]["hand"],
                    scale_params=pose_output["mhr"]["scale"],
                    shape_params=pose_output["mhr"]["shape"],
                    expr_params=pose_output["mhr"]["face"],
                    return_keypoints=True,
                    return_joint_coords=True,
                    return_model_params=True,
                    return_joint_rotations=True,
                )
            )
            j3d = j3d[:, :70]
            verts[..., [1, 2]] *= -1
            j3d[..., [1, 2]] *= -1
            jcoords[..., [1, 2]] *= -1
            pose_output["mhr"]["pred_keypoints_3d"] = j3d
            pose_output["mhr"]["pred_vertices"] = verts
            pose_output["mhr"]["pred_joint_coords"] = jcoords
            pose_output["mhr"]["pred_pose_raw"][
                ...
            ] = 0  # pred_pose_raw is not valid anymore
            pose_output["mhr"]["mhr_model_params"] = mhr_model_params

        ########################################################
        # Project to 2D
        pred_keypoints_3d_proj = (
            pose_output["mhr"]["pred_keypoints_3d"]
            + pose_output["mhr"]["pred_cam_t"][:, None, :]
        )
        pred_keypoints_3d_proj[:, :, [0, 1]] *= pose_output["mhr"]["focal_length"][
            :, None, None
        ]
        pred_keypoints_3d_proj[:, :, [0, 1]] = (
            pred_keypoints_3d_proj[:, :, [0, 1]]
            + torch.FloatTensor([width / 2, height / 2]).to(pred_keypoints_3d_proj)[
                None, None, :
            ]
            * pred_keypoints_3d_proj[:, :, [2]]
        )
        pred_keypoints_3d_proj[:, :, :2] = (
            pred_keypoints_3d_proj[:, :, :2] / pred_keypoints_3d_proj[:, :, [2]]
        )
        pose_output["mhr"]["pred_keypoints_2d"] = pred_keypoints_3d_proj[:, :, :2]

        return pose_output, batch_lhand, batch_rhand, lhand_output, rhand_output

    def run_keypoint_prompt(self, batch, output, keypoint_prompt):
        image_embeddings = output["image_embeddings"]
        condition_info = output["condition_info"]
        pose_output = output["mhr"]  # body-only output
        # Use previous estimate as initialization
        prev_estimate = torch.cat(
            [
                pose_output["pred_pose_raw"].detach(),
                pose_output["shape"].detach(),
                pose_output["scale"].detach(),
                pose_output["hand"].detach(),
                pose_output["face"].detach(),
            ],
            dim=1,
        ).unsqueeze(dim=1)
        if hasattr(self, "init_camera"):
            prev_estimate = torch.cat(
                [prev_estimate, pose_output["pred_cam"].detach().unsqueeze(1)],
                dim=-1,
            )

        tokens_output, pose_output = self.forward_decoder(
            image_embeddings,
            init_estimate=None,
            keypoints=keypoint_prompt,
            prev_estimate=prev_estimate,
            condition_info=condition_info,
            batch=batch,
        )
        pose_output = pose_output[-1]

        output.update({"mhr": pose_output})
        return output, keypoint_prompt

    def _get_hand_box(self, pose_output, batch):
        """Get hand bbox from the hand detector"""
        pred_left_hand_box = (
            pose_output["mhr"]["hand_box"][:, 0].detach().cpu().numpy()
            * self.cfg.MODEL.IMAGE_SIZE[0]
        )
        pred_right_hand_box = (
            pose_output["mhr"]["hand_box"][:, 1].detach().cpu().numpy()
            * self.cfg.MODEL.IMAGE_SIZE[0]
        )

        # Change boxes into squares
        batch["left_center"] = pred_left_hand_box[:, :2]
        batch["left_scale"] = (
            pred_left_hand_box[:, 2:].max(axis=1, keepdims=True).repeat(2, axis=1)
        )
        batch["right_center"] = pred_right_hand_box[:, :2]
        batch["right_scale"] = (
            pred_right_hand_box[:, 2:].max(axis=1, keepdims=True).repeat(2, axis=1)
        )

        # Crop to full. batch["affine_trans"] is full-to-crop, right application
        batch["left_scale"] = (
            batch["left_scale"]
            / batch["affine_trans"][0, :, 0, 0].cpu().numpy()[:, None]
        )
        batch["right_scale"] = (
            batch["right_scale"]
            / batch["affine_trans"][0, :, 0, 0].cpu().numpy()[:, None]
        )
        batch["left_center"] = (
            batch["left_center"]
            - batch["affine_trans"][0, :, [0, 1], [2, 2]].cpu().numpy()
        ) / batch["affine_trans"][0, :, 0, 0].cpu().numpy()[:, None]
        batch["right_center"] = (
            batch["right_center"]
            - batch["affine_trans"][0, :, [0, 1], [2, 2]].cpu().numpy()
        ) / batch["affine_trans"][0, :, 0, 0].cpu().numpy()[:, None]

        left_xyxy = np.concatenate(
            [
                (
                    batch["left_center"][:, 0] - batch["left_scale"][:, 0] * 1 / 2
                ).reshape(-1, 1),
                (
                    batch["left_center"][:, 1] - batch["left_scale"][:, 1] * 1 / 2
                ).reshape(-1, 1),
                (
                    batch["left_center"][:, 0] + batch["left_scale"][:, 0] * 1 / 2
                ).reshape(-1, 1),
                (
                    batch["left_center"][:, 1] + batch["left_scale"][:, 1] * 1 / 2
                ).reshape(-1, 1),
            ],
            axis=1,
        )
        right_xyxy = np.concatenate(
            [
                (
                    batch["right_center"][:, 0] - batch["right_scale"][:, 0] * 1 / 2
                ).reshape(-1, 1),
                (
                    batch["right_center"][:, 1] - batch["right_scale"][:, 1] * 1 / 2
                ).reshape(-1, 1),
                (
                    batch["right_center"][:, 0] + batch["right_scale"][:, 0] * 1 / 2
                ).reshape(-1, 1),
                (
                    batch["right_center"][:, 1] + batch["right_scale"][:, 1] * 1 / 2
                ).reshape(-1, 1),
            ],
            axis=1,
        )

        return left_xyxy, right_xyxy

    def keypoint_token_update_fn(
        self,
        kps_emb_start_idx,
        image_embeddings,
        token_embeddings,
        token_augment,
        pose_output,
        layer_idx,
    ):
        # It's already after the last layer, we're done.
        if layer_idx == len(self.decoder.layers) - 1:
            return token_embeddings, token_augment, pose_output, layer_idx

        # Clone
        token_embeddings = token_embeddings.clone()
        token_augment = token_augment.clone()

        num_keypoints = self.keypoint_embedding.weight.shape[0]

        # Get current 2D KPS predictions
        pred_keypoints_2d_cropped = pose_output[
            "pred_keypoints_2d_cropped"
        ].clone()
        pred_keypoints_2d_depth = pose_output["pred_keypoints_2d_depth"].clone()

        pred_keypoints_2d_cropped = pred_keypoints_2d_cropped[
            :, self.keypoint_embedding_idxs
        ]
        pred_keypoints_2d_depth = pred_keypoints_2d_depth[
            :, self.keypoint_embedding_idxs
        ]

        # Get 2D KPS to be 0 ~ 1
        pred_keypoints_2d_cropped_01 = pred_keypoints_2d_cropped + 0.5

        # Get a mask of those that are 1) beyond image boundaries or 2) behind the camera
        invalid_mask = (
            (pred_keypoints_2d_cropped_01[:, :, 0] < 0)
            | (pred_keypoints_2d_cropped_01[:, :, 0] > 1)
            | (pred_keypoints_2d_cropped_01[:, :, 1] < 0)
            | (pred_keypoints_2d_cropped_01[:, :, 1] > 1)
            | (pred_keypoints_2d_depth[:, :] < 1e-5)
        )

        # Run them through the prompt encoder's pos emb function
        token_augment[:, kps_emb_start_idx : kps_emb_start_idx + num_keypoints, :] = (
            self.keypoint_posemb_linear(pred_keypoints_2d_cropped)
            * (~invalid_mask[:, :, None])
        )

        # Also update token_embeddings with the grid sampled 2D feature.
        pred_keypoints_2d_cropped_sample_points = pred_keypoints_2d_cropped * 2
        if self.cfg.MODEL.BACKBONE.TYPE in [
            "vit_hmr",
            "vit",
            "vit_b",
            "vit_l",
            "vit_hmr_512_384",
        ]:
            pred_keypoints_2d_cropped_sample_points[:, :, 0] = (
                pred_keypoints_2d_cropped_sample_points[:, :, 0] / 12 * 16
            )

        pred_keypoints_2d_cropped_feats = (
            F.grid_sample(
                image_embeddings,
                pred_keypoints_2d_cropped_sample_points[:, :, None, :],
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            .squeeze(3)
            .permute(0, 2, 1)
        )
        pred_keypoints_2d_cropped_feats = pred_keypoints_2d_cropped_feats * (
            ~invalid_mask[:, :, None]
        )
        token_embeddings = token_embeddings.clone()
        token_embeddings[
            :,
            kps_emb_start_idx : kps_emb_start_idx + num_keypoints,
            :,
        ] += self.keypoint_feat_linear(pred_keypoints_2d_cropped_feats)

        return token_embeddings, token_augment, pose_output, layer_idx

    def keypoint3d_token_update_fn(
        self,
        kps3d_emb_start_idx,
        token_embeddings,
        token_augment,
        pose_output,
        layer_idx,
    ):
        # It's already after the last layer, we're done.
        if layer_idx == len(self.decoder.layers) - 1:
            return token_embeddings, token_augment, pose_output, layer_idx

        num_keypoints3d = self.keypoint3d_embedding.weight.shape[0]

        # Get current 3D kps predictions
        pred_keypoints_3d = pose_output["pred_keypoints_3d"].clone()

        # Now, pelvis normalize
        pred_keypoints_3d = (
            pred_keypoints_3d
            - (
                pred_keypoints_3d[:, [self.pelvis_idx[0]], :]
                + pred_keypoints_3d[:, [self.pelvis_idx[1]], :]
            )
            / 2
        )

        # Get the kps we care about, _after_ pelvis norm
        pred_keypoints_3d = pred_keypoints_3d[:, self.keypoint3d_embedding_idxs]

        # Run through embedding MLP & put in
        token_augment = token_augment.clone()
        token_augment[
            :,
            kps3d_emb_start_idx : kps3d_emb_start_idx + num_keypoints3d,
            :,
        ] = self.keypoint3d_posemb_linear(pred_keypoints_3d)

        return token_embeddings, token_augment, pose_output, layer_idx

    def keypoint_token_update_fn_hand(
        self,
        kps_emb_start_idx,
        image_embeddings,
        token_embeddings,
        token_augment,
        pose_output,
        layer_idx,
    ):
        # It's already after the last layer, we're done.
        if layer_idx == len(self.decoder_hand.layers) - 1:
            return token_embeddings, token_augment, pose_output, layer_idx

        # Clone
        token_embeddings = token_embeddings.clone()
        token_augment = token_augment.clone()

        num_keypoints = self.keypoint_embedding_hand.weight.shape[0]

        # Get current 2D KPS predictions
        pred_keypoints_2d_cropped = pose_output[
            "pred_keypoints_2d_cropped"
        ].clone()
        pred_keypoints_2d_depth = pose_output["pred_keypoints_2d_depth"].clone()

        pred_keypoints_2d_cropped = pred_keypoints_2d_cropped[
            :, self.keypoint_embedding_idxs_hand
        ]
        pred_keypoints_2d_depth = pred_keypoints_2d_depth[
            :, self.keypoint_embedding_idxs_hand
        ]

        # Get 2D KPS to be 0 ~ 1
        pred_keypoints_2d_cropped_01 = pred_keypoints_2d_cropped + 0.5

        # Get a mask of those that are 1) beyond image boundaries or 2) behind the camera
        invalid_mask = (
            (pred_keypoints_2d_cropped_01[:, :, 0] < 0)
            | (pred_keypoints_2d_cropped_01[:, :, 0] > 1)
            | (pred_keypoints_2d_cropped_01[:, :, 1] < 0)
            | (pred_keypoints_2d_cropped_01[:, :, 1] > 1)
            | (pred_keypoints_2d_depth[:, :] < 1e-5)
        )

        # Run them through the prompt encoder's pos emb function
        token_augment[:, kps_emb_start_idx : kps_emb_start_idx + num_keypoints, :] = (
            self.keypoint_posemb_linear_hand(pred_keypoints_2d_cropped)
            * (~invalid_mask[:, :, None])
        )

        # Also update token_embeddings with the grid sampled 2D feature.
        pred_keypoints_2d_cropped_sample_points = pred_keypoints_2d_cropped * 2
        if self.cfg.MODEL.BACKBONE.TYPE in [
            "vit_hmr",
            "vit",
            "vit_b",
            "vit_l",
            "vit_hmr_512_384",
        ]:
            pred_keypoints_2d_cropped_sample_points[:, :, 0] = (
                pred_keypoints_2d_cropped_sample_points[:, :, 0] / 12 * 16
            )

        pred_keypoints_2d_cropped_feats = (
            F.grid_sample(
                image_embeddings,
                pred_keypoints_2d_cropped_sample_points[:, :, None, :],
                mode="bilinear",
                padding_mode="zeros",
                align_corners=False,
            )
            .squeeze(3)
            .permute(0, 2, 1)
        )
        pred_keypoints_2d_cropped_feats = pred_keypoints_2d_cropped_feats * (
            ~invalid_mask[:, :, None]
        )
        token_embeddings = token_embeddings.clone()
        token_embeddings[
            :,
            kps_emb_start_idx : kps_emb_start_idx + num_keypoints,
            :,
        ] += self.keypoint_feat_linear_hand(pred_keypoints_2d_cropped_feats)

        return token_embeddings, token_augment, pose_output, layer_idx

    def keypoint3d_token_update_fn_hand(
        self,
        kps3d_emb_start_idx,
        token_embeddings,
        token_augment,
        pose_output,
        layer_idx,
    ):
        # It's already after the last layer, we're done.
        if layer_idx == len(self.decoder_hand.layers) - 1:
            return token_embeddings, token_augment, pose_output, layer_idx

        num_keypoints3d = self.keypoint3d_embedding_hand.weight.shape[0]

        # Get current 3D kps predictions
        pred_keypoints_3d = pose_output["pred_keypoints_3d"].clone()

        # Now, pelvis normalize
        pred_keypoints_3d = (
            pred_keypoints_3d
            - (
                pred_keypoints_3d[:, [self.pelvis_idx[0]], :]
                + pred_keypoints_3d[:, [self.pelvis_idx[1]], :]
            )
            / 2
        )

        # Get the kps we care about, _after_ pelvis norm
        pred_keypoints_3d = pred_keypoints_3d[:, self.keypoint3d_embedding_idxs_hand]

        # Run through embedding MLP & put in
        token_augment = token_augment.clone()
        token_augment[
            :,
            kps3d_emb_start_idx : kps3d_emb_start_idx + num_keypoints3d,
            :,
        ] = self.keypoint3d_posemb_linear_hand(pred_keypoints_3d)

        return token_embeddings, token_augment, pose_output, layer_idx
