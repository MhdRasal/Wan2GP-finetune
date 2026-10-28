# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""Waypoint-1.5 DiT (Overworld, weights Apache-2.0), written for WanGP from the Waypoint-1.5 paper, the released configuration and the
checkpoint layout: a frame-causal transformer that denoises one latent frame (16x32 tokens at 512x1024) at a time.

- Conditioning: sigma -> 512 Fourier features (sqrt(2) [sin, cos] of 1000 sigma f, f = 10000^-k/255) -> FP32 SiLU MLP; two condition heads
  shared by all blocks (attention, MLP) give scale / shift / gate from SiLU(cond + the block's bias). Controller: [mouse dx, dy, 256
  buttons, scroll] -> SiLU MLP, added in every third block by x += W3 SiLU(W1 RMS(x) + W2 c).
- Blocks: RMS(x) (1 + s) + b -> attention (32 query / 16 key-value heads of 64, parameter-free RMS norm of q and k, OrthoRoPE: x and y
  pairs at pi * linspace(1, 0.8 min(h, w) / 2, 4) of the token centers in [-1, 1], time pairs at 10000^-k/8, value residual
  v = (1 - lambda) v + lambda v_first) -> x += g attention; then the controller term; then x += g MLP(RMS(x) (1 + s) + b), SiLU MLP.
- Output: SiLU(RMS(x) (1 + s) + b) with s, b from the sigma embedding -> 2x2 transposed convolution.
- Attention: the frame's own tokens and the committed frames of its window (kv_cache.WindowCache): the 15 previous frames, and in every
  fourth block the frames at multiples of 8 within the last 128."""
import math
import weakref

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmgp import cuda_graphs

from shared.attention_kit import qkv_attention
from shared.kernels import ops
from shared.utils.lora_mapping import convert_lora_keys
from shared.utils.phase_progress import check_abort

CHANNELS, HIDDEN, HEADS, KV_HEADS, HEAD_DIM, DEPTH, MLP_RATIO = 32, 2048, 32, 16, 64, 24, 4
BUTTONS, FOURIER = 256, 512
CONTROL_PERIOD, GLOBAL_PERIOD = 3, 4  # controller term in blocks 0, 3, 6...; the long window in blocks 3, 7, 11...
ROPE_NYQUIST, ROPE_THETA = 0.8, 10000.0
EPS = torch.finfo(torch.float32).eps


def is_global(index):
    return (index + 1) % GLOBAL_PERIOD == 0


class Linear2(nn.Module):
    """fc1 -> SiLU -> fc2, no biases."""

    def __init__(self, dim_in, dim_hidden, dim_out):
        super().__init__()
        self.fc1 = nn.Linear(dim_in, dim_hidden, bias=False)
        self.fc2 = nn.Linear(dim_hidden, dim_out, bias=False)

    def forward(self, x):
        return self.fc2(F.silu(self.fc1(x)))


class _UnitNorm:
    """Parameter-free per-head RMS norm in the form ops.interleaved_norm_rope_ takes (a weight of ones, exact)."""

    def __init__(self, weight):
        self.weight, self.eps = weight, EPS

    def __call__(self, x):
        return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


class ControlInput:
    """fc1_x with the frame's controller term added to its output: the first layer of a block's controller MLP for ops.mlp."""

    def __init__(self, fc1_x, term):
        self.fc1_x, self.term, self.out_features = fc1_x, term, fc1_x.out_features

    def __call__(self, x):
        return self.fc1_x(x).add_(self.term)


class Fusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1_x = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.fc1_c = nn.Linear(HIDDEN, HIDDEN, bias=False)
        self.fc2 = nn.Linear(HIDDEN, HIDDEN, bias=False)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(HIDDEN, HEADS * HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(HIDDEN, KV_HEADS * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(HIDDEN, KV_HEADS * HEAD_DIM, bias=False)
        self.out_proj = nn.Linear(HEADS * HEAD_DIM, HIDDEN, bias=False)
        self.v_lamb = nn.Parameter(torch.empty(()))


class WaypointBlock(nn.Module):
    def __init__(self, index):
        super().__init__()
        self.index = index
        self.lamb = None  # float(attn.v_lamb), set after loading (no device sync per pass)
        self.attn = Attention()
        self.dit_mlp = Linear2(HIDDEN, HIDDEN * MLP_RATIO, HIDDEN)
        self.ctrl_mlpfusion = Fusion() if index % CONTROL_PERIOD == 0 else None
        self.norm = nn.RMSNorm(HIDDEN, eps=EPS, elementwise_affine=False)

    def forward(self, x, mods, control, rope, cache, frames, first_values, commit):
        """x (tokens, 2048) updated in place. mods (6, 2048): attention scale / shift / gate, MLP scale / shift / gate. control: the frame's
        controller embedding (1, 2048). cache: the WindowCache, whose first `frames` slots of this block are attended. first_values: the
        values of block 0 (1, tokens, 16, 64), written by block 0. commit: the frame is clean and its keys / values go to the cache."""
        rows = [(0, x.shape[0], 0)]
        attn, unit, lamb = self.attn, _UnitNorm(x.new_ones(HEAD_DIM)), self.lamb

        def norm_rope(query, key, group):
            ops.interleaved_norm_rope_(query, key, rope, unit, unit)
            if commit and key is not None:
                cache.write(self.index, 0, key, group.kv)

        def values(value, group):
            heads = slice(group.kv.start, group.kv.stop)
            if self.index == 0:
                first_values[:, :, heads].copy_(value)
            else:
                value.mul_(1.0 - lamb).add_(first_values[:, :, heads], alpha=lamb)
            if commit:
                cache.write(self.index, 1, value, group.kv)

        attention = qkv_attention([ops.norm_modulate([x], self.norm, mods[1:2], mods[0:1], rows)], attn.q_proj, attn.k_proj, attn.v_proj, HEADS, HEAD_DIM,
                                  norm_rope, kv_heads=KV_HEADS, kv_prefix=cache.prefix(self.index, frames), norm_v=values)
        if commit and self.index == DEPTH - 1:  # nothing reads the last block's output of a commit
            return
        ops.gated_linear(attn.out_proj, [attention.reshape(x.shape[0], HIDDEN)], x, mods[2:3], rows)
        del attention
        fusion = self.ctrl_mlpfusion
        if fusion is not None:
            ops.mlp([self.norm(x)], ControlInput(fusion.fc1_x, fusion.fc1_c(control)[0]), fusion.fc2, "silu", residual=x)
        ops.mlp([ops.norm_modulate([x], self.norm, mods[4:5], mods[3:4], rows)], self.dit_mlp.fc1, self.dit_mlp.fc2, "silu", residual=x, gate=mods[5])


class WaypointTransformer(nn.Module):
    def __init__(self, height=16, width=32):
        """height / width: the frame's token grid (latent / 2): 16 x 32 at 512x1024, 8 x 16 at 256x512."""
        super().__init__()
        self.height, self.width = height, width
        self.denoise_step_emb = Linear2(FOURIER, HIDDEN * MLP_RATIO, HIDDEN)
        self.ctrl_emb = Linear2(BUTTONS + 3, HIDDEN * MLP_RATIO, HIDDEN)
        self.attn_cond = nn.ModuleList(nn.Linear(HIDDEN, HIDDEN, bias=False) for _ in range(3))
        self.mlp_cond = nn.ModuleList(nn.Linear(HIDDEN, HIDDEN, bias=False) for _ in range(3))
        self.cond_bias = nn.Parameter(torch.empty(DEPTH, HIDDEN))  # each block's bias of the condition heads' input
        self.blocks = nn.ModuleList(WaypointBlock(index) for index in range(DEPTH))
        self.patchify = nn.Conv2d(CHANNELS, HIDDEN, kernel_size=2, stride=2, bias=False)
        self.unpatchify = nn.ConvTranspose2d(HIDDEN, CHANNELS, kernel_size=2, stride=2)
        self.out_norm = nn.Linear(HIDDEN, 2 * HIDDEN, bias=False)
        self.norm = nn.RMSNorm(HIDDEN, eps=EPS, elementwise_affine=False)
        self._conditions = {}
        self._graph_cache = None  # the WindowCache registered as the CUDA graphs' state
        self._frame_inputs = None  # (frame, controls, rope, controller embedding): the five passes of a frame share them

    def lock_layers_dtypes(self):
        self.denoise_step_emb._lock_dtype = torch.float32  # the sigma embedding runs in FP32 (its weights stored in BF16)

    def preprocess_loras(self, model_type, state_dict):
        return convert_lora_keys({key.replace("transformer.blocks.", "blocks."): value for key, value in state_dict.items()}, dict(self.named_modules()))

    # ---- per frame and per sigma inputs

    def rope_table(self, frame, device):
        """(1, tokens, 1, 32, 2) cos / sin of the frame's tokens (row-major), FP32: 8 x pairs, 8 y pairs, 16 time pairs."""
        spatial = math.pi * torch.linspace(1.0, ROPE_NYQUIST * min(self.height, self.width) / 2, 4, device=device).repeat_interleave(2)
        temporal = (ROPE_THETA ** (-torch.arange(0, 16, 2, device=device, dtype=torch.float32) / 16)).repeat_interleave(2)
        ys, xs = torch.meshgrid(torch.arange(self.height, device=device, dtype=torch.float32), torch.arange(self.width, device=device, dtype=torch.float32), indexing="ij")
        x_angles = torch.outer(((xs.reshape(-1) + 0.5) / self.width) * 2 - 1, spatial)
        y_angles = torch.outer(((ys.reshape(-1) + 0.5) / self.height) * 2 - 1, spatial)
        t_angles = (float(frame) * temporal).expand(x_angles.shape[0], -1)
        angles = torch.cat([x_angles, y_angles, t_angles], dim=1)
        return torch.stack((angles.cos(), angles.sin()), dim=-1)[None, :, None]

    def conditions(self, sigma):
        """(blocks (24, 6, 2048), final (2, 2048)) of a noise level, computed once per sigma value (the schedule has five)."""
        key = float(sigma)
        if key not in self._conditions:
            device, dtype = self.out_norm.weight.device, self.out_norm.weight.dtype
            embed = self.denoise_step_emb
            frequencies = 10000.0 ** (-torch.arange(FOURIER // 2, device=device, dtype=torch.float32) / (FOURIER // 2 - 1))
            angles = torch.tensor(key, device=device, dtype=torch.float32) * 1000.0 * frequencies
            cond = embed(math.sqrt(2) * torch.cat([angles.sin(), angles.cos()])[None].to(embed.fc1.weight.dtype)).to(dtype)
            inputs = F.silu(cond + self.cond_bias.to(dtype))
            blocks = torch.stack([projection(inputs) for projection in (*self.attn_cond, *self.mlp_cond)], dim=1)
            final = self.out_norm(F.silu(cond)).view(2, HIDDEN)
            self._conditions[key] = (blocks, final)
        return self._conditions[key]

    def reset_conditions(self):
        self._conditions, self._frame_inputs = {}, None

    # ---- one frame

    def forward(self, latent, sigma, controls, frame, cache, commit=False):
        """latent (32, h, w) of the frame at noise level sigma -> its velocity (32, h, w); with commit (sigma 0: the clean frame), its keys /
        values go to the cache and nothing is returned. controls (259): the frame's mouse dx, dy, 256 buttons and scroll sign. frame: its
        time position."""
        blocks, final = self.conditions(sigma)
        dtype = self.out_norm.weight.dtype
        inputs = self._frame_inputs
        if inputs is None or inputs[0] != frame or inputs[1] is not controls:
            inputs = self._frame_inputs = (frame, controls, self.rope_table(frame, latent.device), self.ctrl_emb(controls.to(self.ctrl_emb.fc1.weight)[None]))
        rope, control = inputs[2:]
        x = self.patchify(latent[None].to(dtype)).flatten(2)[0].transpose(0, 1).contiguous()
        first_values = x.new_empty((1, x.shape[0], KV_HEADS, HEAD_DIM))
        if self._graph_cache is None or self._graph_cache() is not cache:  # its storage is read in place by the graphs, the cache passed as is
            cuda_graphs.register_state(self, "window cache", cache.tensors(), holders=(cache,))
            self._graph_cache = weakref.ref(cache)
        counts = cache.counts()
        # graphed when the mode is on and the local windows are full (before, each frame changes their shapes: eager rather than captures
        # used once); a commit writes the staging buffer of the cache, at fixed addresses
        with cuda_graphs.scope(enabled=cache.warm()):
            for index, block in enumerate(self.blocks):
                block(x, blocks[index], control, rope, cache, counts[index], first_values, commit)
                check_abort()
        if commit:
            return None
        x = F.silu(ops.norm_modulate([x], self.norm, final[1:2], final[0:1], [(0, x.shape[0], 0)]), inplace=True)
        return self.unpatchify(x.transpose(0, 1).reshape(1, HIDDEN, self.height, self.width))[0]
