# Copyright (c) 2026 DeepBeepMeep - part of WanGP, WanGP Community License 2.0 (see LICENSE.txt)
"""silu(group_norm(x)) of 16-bit NCHW activations, the GroupNorm + SiLU before each convolution of image VAEs, as PyTorch computes it under
autocast (GroupNorm and SiLU in FP32, one rounding to the output dtype), without its FP32 temporaries: PyTorch writes an FP32 copy of x, the
FP32 normalization and the FP32 sigmoid (two full-size FP32 tensors per chunk of groups, about 44 bytes moved per element), where these
kernels read x three times and write the output once (8 bytes per element) and allocate only the output.

Triton (compute capability 7.5 and up; BF16 read and written as bits on every GPU, so Turing and the others give the same values):
- each program sums a span of a group (contiguous in NCHW) in FP32 accumulators that carry their rounding errors (two-sum), reduced in FP64,
  and takes the largest |x|: the group's mean is as close to the exact one as FP64 rounding of exact partial sums allows (PyTorch: FP32
  Welford). x is summed times 2^-64 (exact) so that no 16-bit value can overflow the accumulators;
- a second pass sums the squares of (x - m32) * 2^-e the same way, m32 the mean rounded to FP32 and 2^e the power of two above
  max|x| + |m32| (no overflow nor loss whatever the magnitude); the variance corrects the rounding of the mean exactly
  (E[(x - m32)^2] - (m - m32)^2): no cancellation however large the mean against the spread;
- a third pass gives each program one channel's span, with scale = rstd * weight and the shift as scalars: y = (x - m32) * scale + shift
  (computed as ((x - m32) * 2^-e) * (scale * 2^e), the same values for normal numbers), then y * sigmoid(y) with the precise exponential
  and IEEE division (as PyTorch's FP32 sigmoid), rounded once to nearest even.
Activations below MIN_BYTES keep the PyTorch route: the three launches cost more there (a floor of ~0.1 ms on an RTX 4090, ~0.43 ms on
an RTX 2080 Ti under Windows), measured against the chunked route at the Flux VAE shapes. Partial sums are combined in a fixed order:
deterministic for a given GPU and configuration. The launch configuration (elements per load,
warps, pipelining stages, programs per SM) is tuned once per GPU, channels per group and dtype, on scratch output (x is only read); the
lengths, spans and channel counts are runtime values: a new resolution or tile size compiles nothing."""
import functools

import torch

from . import dispatch

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
    from .bf16_bits import bf16_bits_to_fp32, fp32_to_bf16_bits
    from .triton_tuner import LaunchTuner
except ImportError:
    triton = None

MIN_BYTES = 8 << 20  # smaller activations: PyTorch (RTX 4090: kernel x1.13 at 4 MiB, x0.79 at 8 MiB; RTX 2080 Ti: x1.0 at 4, x0.5 at 16)
MAX_SPANS = 128  # partial sums per group (one masked load combines them)
# (elements per load, warps, stages, programs per SM); the first is the untuned default
_CANDIDATES = ((2048, 8, 3, 4), (1024, 4, 3, 4), (4096, 8, 2, 4), (2048, 8, 3, 8), (1024, 4, 3, 8), (4096, 8, 2, 2), (2048, 4, 4, 4))
_TUNE_MIN_BYTES = 32 << 20  # smaller activations use the default configuration (their timings would represent launch latency)
_TUNE_MAX_BYTES = 256 << 20  # the scratch output of a tuning is at most this (a channel slice of larger activations)

if triton is not None:
    _TUNER = LaunchTuner("wan2gp_group_norm_silu", 1, "VAE")

    @triton.jit
    def _load(X, index, mask, BITS: tl.constexpr):
        x = tl.load(X + index, mask, 0)
        return bf16_bits_to_fp32(x) if BITS else x.to(tl.float32)

    @triton.jit
    def _group_total(PARTIAL, group, SPANS):
        span = tl.arange(0, 128)  # MAX_SPANS
        return tl.sum(tl.load(PARTIAL + group * SPANS + span, span < SPANS, 0.0), axis=0)

    @triton.jit
    def _group_stats(SUMS, MAXES, group, SPANS, LENGTH):
        """The group's mean (FP64), its rounding to FP32 and the power of two 2^-e above max|x| + |m32| (FP32, with e)."""
        span = tl.arange(0, 128)  # MAX_SPANS
        mean = _group_total(SUMS, group, SPANS) * 18446744073709551616.0 / LENGTH.to(tl.float64)  # sums of x * 2^-64
        mean32 = mean.to(tl.float32)
        bound = tl.max(tl.load(MAXES + group * SPANS + span, span < SPANS, 0.0), axis=0).to(tl.float64) + tl.abs(mean32).to(tl.float64)
        e = tl.minimum(tl.maximum(((bound.to(tl.int64, bitcast=True) >> 52) & 0x7FF) - 1022, -120), 120).to(tl.int32)  # bound < 2^e
        return mean, mean32, ((127 - e) << 23).to(tl.float32, bitcast=True), e

    @triton.jit(do_not_specialize=["LENGTH", "SPAN", "SPANS"])
    def _sums_kernel(X, PARTIAL, SUMS, MAXES, LENGTH, SPAN, SPANS, CENTERED: tl.constexpr, BLOCK: tl.constexpr, STAGES: tl.constexpr,
                     BITS: tl.constexpr):
        # partial sums of a span of a group: of x * 2^-64 with its largest |x| into MAXES, or with CENTERED of ((x - m32) * 2^-e)^2
        span, group = tl.program_id(0), tl.program_id(1).to(tl.int64)
        start = span.to(tl.int64) * SPAN
        end = tl.minimum(start + SPAN, LENGTH)
        base = X + group * LENGTH
        if CENTERED:
            _, mean32, unit, _ = _group_stats(SUMS, MAXES, group, SPANS, LENGTH)
            shift = mean32 * unit
        else:
            unit, shift = 5.421010862427522e-20, 0.0  # 2^-64
        hi = tl.zeros((BLOCK,), dtype=tl.float32)
        lo = tl.zeros((BLOCK,), dtype=tl.float32)
        top = tl.zeros((BLOCK,), dtype=tl.float32)
        for offset in tl.range(0, SPAN, BLOCK, num_stages=STAGES):
            index = start + offset + tl.arange(0, BLOCK)
            inside = index < end
            x = _load(base, index, inside, BITS)
            if CENTERED:
                x = tl.where(inside, x * unit - shift, 0.0)
                x = x * x
            else:
                top = tl.maximum(top, tl.abs(x))
                x = x * unit
            total = hi + x  # two-sum (launched without FP fusion): lo collects the rounding error of each addition
            back = total - hi
            lo += (hi - (total - back)) + (x - back)
            hi = total
        tl.store(PARTIAL + group * SPANS + span, tl.sum(hi.to(tl.float64) + lo.to(tl.float64), axis=0))
        if not CENTERED:
            tl.store(MAXES + group * SPANS + span, tl.max(top, axis=0))

    @triton.jit(do_not_specialize=["HW", "CHUNK", "SPANS", "LENGTH", "CHANNELS", "GROUP_CHANNELS"])
    def _apply_kernel(X, Y, SUMS, SQUARES, MAXES, W, B, HW, CHUNK, SPANS, LENGTH, CHANNELS, GROUP_CHANNELS, EPS, BLOCK: tl.constexpr,
                      STAGES: tl.constexpr, BITS: tl.constexpr, OUT_BITS: tl.constexpr):
        chunk, row = tl.program_id(0), tl.program_id(1).to(tl.int64)  # row: n * C + c
        channel = row % CHANNELS
        group = (row // CHANNELS) * (CHANNELS // GROUP_CHANNELS) + channel // GROUP_CHANNELS
        mean, mean32, unit, e = _group_stats(SUMS, MAXES, group, SPANS, LENGTH)
        power = ((1023 + e).to(tl.int64) << 52).to(tl.float64, bitcast=True)  # 2^e, exact (|e| <= 120)
        error = mean - mean32.to(tl.float64)
        variance = tl.maximum(_group_total(SQUARES, group, SPANS) / LENGTH.to(tl.float64) * power * power - error * error, 0.0)
        scale = (tl.load(W + channel).to(tl.float64) / tl.sqrt(variance + EPS) * power).to(tl.float32)  # FP64 IEEE division and root
        shift, centre = tl.load(B + channel), mean32 * unit
        start = chunk.to(tl.int64) * CHUNK
        end = tl.minimum(start + CHUNK, HW)
        base = row * HW
        for offset in tl.range(0, CHUNK, BLOCK, num_stages=STAGES):
            index = start + offset + tl.arange(0, BLOCK)
            inside = index < end
            x = _load(X + base, index, inside, BITS)
            y = (x * unit - centre) * scale + shift
            y = y * tl.div_rn(1.0, 1.0 + libdevice.exp(-y))  # PyTorch's FP32 sigmoid, then the product
            tl.store(Y + base + index, fp32_to_bf16_bits(y) if OUT_BITS else y.to(Y.dtype.element_ty), inside)


@functools.cache
def _sms(index):
    return torch.cuda.get_device_properties(index).multi_processor_count


def _launch(source, target, weight, bias, batch, channels, group_channels, hw, eps, bits, out_bits, config, sms):
    block, warps, stages, per_sm = config
    groups, length = batch * (channels // group_channels), group_channels * hw
    spans = max(1, min(MAX_SPANS, triton.cdiv(length, block), per_sm * sms // groups))
    span = triton.cdiv(triton.cdiv(length, spans), block) * block
    spans = triton.cdiv(length, span)
    sums = torch.empty((3, groups, spans), dtype=torch.float64, device=source.device)  # sums, squares, maxima (FP32 in the third)
    maxes = sums[2].view(torch.float32)
    for partial, centered in ((sums[0], False), (sums[1], True)):
        _sums_kernel[(spans, groups)](source, partial, sums[0], maxes, length, span, spans, centered, block, stages, bits, num_warps=warps,
                                      enable_fp_fusion=False)
    chunks = max(1, min(triton.cdiv(hw, block), per_sm * sms // (batch * channels)))
    chunk = triton.cdiv(triton.cdiv(hw, chunks), block) * block
    _apply_kernel[(triton.cdiv(hw, chunk), batch * channels)](source, target, sums[0], sums[1], maxes, weight, bias, hw, chunk, spans, length,
                                                              channels, group_channels, eps, block, stages, bits, out_bits, num_warps=warps)


def eligible(x, num_groups, dtype):
    """Whether the kernels run silu(group_norm(x)) to dtype: contiguous NCHW BF16 or FP16 x of MIN_BYTES or more and output, whole groups."""
    return (triton is not None and x.dim() == 4 and x.is_contiguous() and x.numel() * x.element_size() >= MIN_BYTES and x.shape[1] % num_groups == 0
            and dtype in (torch.bfloat16, torch.float16) and dispatch.cuda_inference(x, dtypes=(torch.bfloat16, torch.float16), capability_floor=(7, 5)))


def group_norm_silu(x, num_groups, weight, bias, eps, dtype):
    """silu(group_norm(x.float(), num_groups, weight, bias, eps)) rounded to dtype, x borrowed (not modified); eligible(x, ...) first."""
    batch, channels, height, width = x.shape
    hw, group_channels = height * width, channels // num_groups
    out = torch.empty(x.shape, dtype=dtype, device=x.device)
    bits, out_bits = x.dtype == torch.bfloat16, dtype == torch.bfloat16
    source, target = x.view(torch.int16) if bits else x, out.view(torch.int16) if out_bits else out
    weight, bias = weight.to(device=x.device, dtype=torch.float32), bias.to(device=x.device, dtype=torch.float32)
    sms = _sms(x.device.index)
    size = x.numel() * x.element_size()

    def scratch():  # this call's x (read only), or its first sample's first channels (contiguous), and an output of its own
        part = source if size <= _TUNE_MAX_BYTES else source[:1]
        if part.numel() * part.element_size() > _TUNE_MAX_BYTES:
            part = part[:, :max(1, _TUNE_MAX_BYTES // (part.numel() * part.element_size() // num_groups)) * group_channels]
        sample = torch.empty(part.shape, dtype=target.dtype, device=x.device)
        return lambda config: _launch(part, sample, weight, bias, part.shape[0], part.shape[1], group_channels, hw, eps, bits, out_bits, config, sms)

    config = _TUNER.config("group_norm_silu", f"{group_channels}|{x.dtype}|{dtype}", x.device, _CANDIDATES, scratch, size >= _TUNE_MIN_BYTES)
    _launch(source, target, weight, bias, batch, channels, group_channels, hw, eps, bits, out_bits, tuple(config), sms)
    return out


if triton is not None:
    from shared.kernels.triton_compilation_log import install_triton_compilation_logger  # noqa: E402

    install_triton_compilation_logger()
