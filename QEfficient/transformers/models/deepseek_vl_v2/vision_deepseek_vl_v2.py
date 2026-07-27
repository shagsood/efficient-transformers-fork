# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
#
# -----------------------------------------------------------------------------

"""
Vision tower ("DeepEncoder v2") for ``deepseek-ai/DeepSeek-OCR-2``.

Two stages feed a linear projector:

1. **SAM ViT-B backbone** -- 12 blocks at width 768, windowed attention (window 14)
   with global attention at blocks [2, 5, 8, 11], decomposed relative position bias,
   then a 1x1/3x3 conv neck to 256 channels and two stride-2 convs (``net_2``,
   ``net_3``) that downsample to 896 channels: ``(B,3,1024,1024) -> (B,896,16,16)``.
2. **Qwen2-as-encoder** -- the 896-dim feature map is flattened to 256 tokens and
   concatenated with a learned query embedding of the same length. A 24-layer Qwen2
   stack then runs the pair *non-causally over the image half* and *causally over the
   query half*, and only the query half is kept: ``(B,896,16,16) -> (B,256,896)``.

The projector maps 896 -> 1280 (decoder hidden size) and a learned ``view_seperator``
row is appended, giving the 257 embeddings that replace the image placeholder tokens.

Export notes
------------
* The reference builds the non-causal attention mask by overriding
  ``Qwen2Model._update_causal_mask``, which **no longer exists** on transformers 5.x --
  the override is silently ignored there and the encoder degrades to fully causal. This
  port therefore implements the Qwen2 stack directly and builds the mask itself, so the
  intended non-causal-over-image semantics are preserved rather than silently dropped.
* ``token_type_ids`` in the reference is always ``[0]*n_query + [1]*n_query`` (it is
  constructed inside the encoder, never supplied by the caller), so the 4D mask is a
  pure function of ``n_query`` -- a compile-time constant, not data-dependent. It is
  built once and registered as a buffer, which keeps it out of the traced graph.
* Relative-position tables are exact at 1024px (window 14 -> 27 entries, global 64 ->
  127), so the reference's interpolation branch is unreachable and the gather indices
  are precomputed as buffers.

Module and parameter names mirror the reference checkpoint exactly
(``sam_model.*``, ``qwen2_model.model.model.layers.*``, ``projector.layers.*``) so the
published weights load with ``strict=True``.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN

# ---------------------------------------------------------------------------
# SAM ViT-B backbone
# ---------------------------------------------------------------------------


class QEffSamLayerNorm2d(nn.Module):
    """Channel-first LayerNorm used by the SAM conv neck."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class QEffSamPatchEmbed(nn.Module):
    """Image to patch embedding; returns channels-last ``(B, H, W, C)``."""

    def __init__(self, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).permute(0, 2, 3, 1)


class QEffSamMLPBlock(nn.Module):
    def __init__(self, embedding_dim: int, mlp_dim: int):
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))


def window_partition(x: torch.Tensor, window_size: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Partition ``(B, H, W, C)`` into non-overlapping windows, padding if needed."""
    b, h, w, c = x.shape
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    hp, wp = h + pad_h, w + pad_w
    x = x.view(b, hp // window_size, window_size, wp // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows, (hp, wp)


def window_unpartition(
    windows: torch.Tensor, window_size: int, pad_hw: Tuple[int, int], hw: Tuple[int, int]
) -> torch.Tensor:
    """Inverse of :func:`window_partition`, removing any padding."""
    hp, wp = pad_hw
    h, w = hw
    b = windows.shape[0] // (hp * wp // window_size // window_size)
    x = windows.view(b, hp // window_size, wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, hp, wp, -1)
    if hp > h or wp > w:
        x = x[:, :h, :w, :].contiguous()
    return x


def build_rel_pos_index(q_size: int, k_size: int) -> torch.Tensor:
    """Gather indices into a relative-position table for equal query/key extents.

    Mirrors the reference ``get_rel_pos`` coordinate arithmetic. The table is exact at
    1024px for both the windowed (14) and global (64) extents, so no interpolation of
    the table is required and these indices are constant.
    """
    q_coords = torch.arange(q_size)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
    return relative_coords.long()


class QEffSamAttention(nn.Module):
    """Multi-head attention with decomposed relative position bias."""

    def __init__(
        self,
        dim: int,
        num_heads: int = 12,
        qkv_bias: bool = True,
        use_rel_pos: bool = True,
        input_size: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.use_rel_pos = use_rel_pos
        if self.use_rel_pos:
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * input_size[0] - 1, self.head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * input_size[1] - 1, self.head_dim))
            self.register_buffer("rel_idx_h", build_rel_pos_index(input_size[0], input_size[0]), persistent=False)
            self.register_buffer("rel_idx_w", build_rel_pos_index(input_size[1], input_size[1]), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, h, w, _ = x.shape
        qkv = self.qkv(x).reshape(b, h * w, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, b * self.num_heads, h * w, -1).unbind(0)

        attn_bias = None
        if self.use_rel_pos:
            rh = self.rel_pos_h[self.rel_idx_h]  # (h, h, head_dim)
            rw = self.rel_pos_w[self.rel_idx_w]  # (w, w, head_dim)
            r_q = q.reshape(b * self.num_heads, h, w, self.head_dim)
            rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, rh).unsqueeze(-1)
            rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, rw).unsqueeze(-2)
            attn_bias = (rel_h + rel_w).reshape(b, self.num_heads, h * w, h * w)

        q = q.view(b, self.num_heads, h * w, -1)
        k = k.view(b, self.num_heads, h * w, -1)
        v = v.view(b, self.num_heads, h * w, -1)

        attn = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if attn_bias is not None:
            attn = attn + attn_bias
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        x = torch.matmul(attn, v)

        x = x.view(b, self.num_heads, h, w, -1).permute(0, 2, 3, 1, 4).reshape(b, h, w, -1)
        return self.proj(x)


class QEffSamBlock(nn.Module):
    """Transformer block with optional window attention."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        window_size: int = 0,
        input_size: Optional[Tuple[int, int]] = None,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = QEffSamAttention(
            dim,
            num_heads=num_heads,
            input_size=input_size if window_size == 0 else (window_size, window_size),
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = QEffSamMLPBlock(embedding_dim=dim, mlp_dim=int(dim * mlp_ratio))
        self.window_size = window_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        if self.window_size > 0:
            h, w = x.shape[1], x.shape[2]
            x, pad_hw = window_partition(x, self.window_size)
        x = self.attn(x)
        if self.window_size > 0:
            x = window_unpartition(x, self.window_size, pad_hw, (h, w))
        x = shortcut + x
        return x + self.mlp(self.norm2(x))


class QEffSamImageEncoderViT(nn.Module):
    """SAM ViT-B image encoder with the DeepEncoder-v2 downsampling neck."""

    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        out_chans: int = 256,
        window_size: int = 14,
        global_attn_indexes: Tuple[int, ...] = (2, 5, 8, 11),
    ):
        super().__init__()
        self.img_size = img_size
        grid = img_size // patch_size
        self.patch_embed = QEffSamPatchEmbed(patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, grid, grid, embed_dim))
        self.blocks = nn.ModuleList(
            [
                QEffSamBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    window_size=window_size if i not in global_attn_indexes else 0,
                    input_size=(grid, grid),
                )
                for i in range(depth)
            ]
        )
        self.neck = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans, kernel_size=1, bias=False),
            QEffSamLayerNorm2d(out_chans),
            nn.Conv2d(out_chans, out_chans, kernel_size=3, padding=1, bias=False),
            QEffSamLayerNorm2d(out_chans),
        )
        self.net_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False)
        self.net_3 = nn.Conv2d(512, 896, kernel_size=3, stride=2, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.neck(x.permute(0, 3, 1, 2))
        return self.net_3(self.net_2(x))


# ---------------------------------------------------------------------------
# Qwen2 stack run as a (partly non-causal) encoder
# ---------------------------------------------------------------------------


class QEffQwen2EncoderRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_kv_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_kv_heads * n_rep, slen, head_dim)


class QEffQwen2EncoderAttention(nn.Module):
    """Qwen2 GQA attention with an externally supplied additive 4D mask."""

    def __init__(self, hidden_size: int, num_heads: int, num_key_value_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = hidden_size // num_heads
        self.num_key_value_groups = num_heads // num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(hidden_size, num_key_value_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(num_heads * self.head_dim, hidden_size, bias=False)

    def forward(self, hidden_states, attention_mask, cos, sin):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        attn_weights = attn_weights + attention_mask
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output)


class QEffQwen2EncoderMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str = "silu"):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class QEffQwen2EncoderLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, num_key_value_heads: int, intermediate_size: int, eps: float):
        super().__init__()
        self.self_attn = QEffQwen2EncoderAttention(hidden_size, num_heads, num_key_value_heads)
        self.mlp = QEffQwen2EncoderMLP(hidden_size, intermediate_size)
        self.input_layernorm = QEffQwen2EncoderRMSNorm(hidden_size, eps=eps)
        self.post_attention_layernorm = QEffQwen2EncoderRMSNorm(hidden_size, eps=eps)

    def forward(self, hidden_states, attention_mask, cos, sin):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask, cos, sin)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        return residual + self.mlp(hidden_states)


class QEffQwen2EncoderStack(nn.Module):
    """The Qwen2 layer stack (``Qwen2Model`` minus ``embed_tokens``)."""

    def __init__(
        self,
        num_layers: int = 24,
        hidden_size: int = 896,
        num_heads: int = 14,
        num_key_value_heads: int = 2,
        intermediate_size: int = 4864,
        rms_norm_eps: float = 1e-6,
        rope_theta: float = 1000000.0,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                QEffQwen2EncoderLayer(hidden_size, num_heads, num_key_value_heads, intermediate_size, rms_norm_eps)
                for _ in range(num_layers)
            ]
        )
        self.norm = QEffQwen2EncoderRMSNorm(hidden_size, eps=rms_norm_eps)
        self.head_dim = hidden_size // num_heads
        self.rope_theta = rope_theta

    def rotary(self, seq_len: int, device, dtype):
        inv_freq = 1.0 / (
            self.rope_theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.int64).float() / self.head_dim)
        )
        t = torch.arange(seq_len, dtype=torch.int64).float()
        freqs = torch.outer(t, inv_freq.to(t.device))
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype).unsqueeze(0), emb.sin().to(dtype).unsqueeze(0)

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        cos, sin = self.rotary(inputs_embeds.shape[1], inputs_embeds.device, inputs_embeds.dtype)
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask, cos, sin)
        return self.norm(hidden_states)


class QEffQwen2DecoderAsEncoderInner(nn.Module):
    """Name-preserving wrapper (reference ``CustomQwen2Decoder``)."""

    def __init__(self, **kwargs):
        super().__init__()
        self.model = QEffQwen2EncoderStack(**kwargs)

    def forward(self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(inputs_embeds, attention_mask)


def build_image_query_mask(n_query: int, dtype: torch.dtype) -> torch.Tensor:
    """Additive 4D mask for ``[image tokens | query tokens]``.

    Reproduces the reference ``_create_custom_4d_mask`` for its only reachable input,
    ``token_type_ids = [0]*n_query + [1]*n_query``:

    * image rows attend to every image column (fully non-causal block) and to no query
      column;
    * query rows attend to every image column and causally over the query block.

    Because ``token_type_ids`` is generated inside the encoder from ``n_query`` alone,
    this is a compile-time constant rather than a data-dependent mask.
    """
    min_dtype = torch.finfo(dtype).min
    total = 2 * n_query
    mask = torch.full((total, total), min_dtype, dtype=dtype)
    mask[:n_query, :n_query] = 0.0
    mask[n_query:, :n_query] = 0.0
    causal = torch.triu(torch.full((n_query, n_query), min_dtype, dtype=dtype), diagonal=1)
    mask[n_query:, n_query:] = causal
    return mask[None, None]


class QEffQwen2Decoder2Encoder(nn.Module):
    """Learned queries + Qwen2 stack; returns only the query half."""

    def __init__(self, hidden_dimension: int = 896, **kwargs):
        super().__init__()
        self.model = QEffQwen2DecoderAsEncoderInner(hidden_size=hidden_dimension, **kwargs)
        self.query_768 = nn.Embedding(144, hidden_dimension)
        self.query_1024 = nn.Embedding(256, hidden_dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(2).transpose(1, 2)
        bs, n_query, _ = x.shape
        param_img = self.query_768.weight if n_query == 144 else self.query_1024.weight
        batch_query_imgs = param_img.unsqueeze(0).expand(bs, -1, -1)
        x_combined = torch.cat([x, batch_query_imgs], dim=1)
        attention_mask = build_image_query_mask(n_query, x_combined.dtype).to(x_combined.device)
        y = self.model(x_combined, attention_mask)
        return y[:, n_query:, :]


class QEffMlpProjector(nn.Module):
    """Linear projector from the vision width (896) to the decoder width (1280)."""

    def __init__(self, input_dim: int = 896, n_embed: int = 1280):
        super().__init__()
        self.layers = nn.Linear(input_dim, n_embed)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
