# SPDX-License-Identifier: Apache-2.0
"""SlimQuant W4A8 adapter for the installed DeepGEMM MegaMoE engine.

Keep loader-owned INT4 weights resident and expand directly into shared W6
scratch. Gate/up rows and scales are interleaved in groups of eight, matching
DeepGEMM's fused SwiGLU epilogue. No standalone megamoe package is used.
"""
import math
import torch
from vllm.triton_utils import tl, triton


def validate_megamoe_int8_runtime():
    import deepgemm
    for name in ("fp8_mega_moe", "mega_moe_preprocess",
                 "get_symm_buffer_for_mega_moe"):
        if not callable(getattr(deepgemm, name, None)):
            raise RuntimeError(f"Installed deepgemm lacks MegaMoE API: {name}")


@triton.jit
def _unpack_signed_int4(P, O, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    b = tl.load(P + i // 2, i < SIZE, other=0).to(tl.int32)
    q = (b >> tl.where(i % 2 == 0, 4, 0)) & 15
    tl.store(O + i, ((q ^ 8) - 8).to(tl.int8), i < SIZE)


def unpack_signed_int4_out(packed, out, *, block=2048, num_warps=4):
    """Decode high-nibble-first two's-complement bytes, without FP quantization."""
    if packed.dtype not in (torch.int8, torch.uint8) or out.dtype != torch.int8:
        raise TypeError("packed must be byte INT4 and out must be signed INT8")
    if not packed.is_contiguous() or not out.is_contiguous():
        raise ValueError("packed and output must be contiguous")
    if packed.device != out.device or not packed.is_cuda:
        raise ValueError("packed and output must be on the same GPU")
    if out.numel() != 2 * packed.numel():
        raise ValueError("output must contain exactly two elements per packed byte")
    _unpack_signed_int4[(triton.cdiv(out.numel(), block),)](
        packed, out, out.numel(), block, num_warps=num_warps
    )
    return out


@triton.jit
def _unpack_raw_int4_w6(P, O, SIZE: tl.constexpr, N: tl.constexpr,
                         K: tl.constexpr, INTERLEAVE: tl.constexpr,
                         BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    # W6 = [E, K/64, N/16, K4, N16, K16].
    k = i % 16 + ((i // 256) % 4) * 16
    n = (i // 16) % 16 + ((i // 1024) % (N // 16)) * 16
    k += ((i // (64 * N)) % (K // 64)) * 64
    e = i // (N * K)
    if INTERLEAVE:
        # Undo gate[8],up[8] interleaving when reading raw checkpoint rows.
        n = (n // 16) * 8 + n % 8 + ((n // 8) % 2) * (N // 2)
    byte = tl.load(P + (e * N + n) * (K // 2) + k // 2,
                   i < SIZE, other=0).to(tl.int32)
    q = (byte >> tl.where(k % 2 == 0, 4, 0)) & 15
    tl.store(O + i, ((q ^ 8) - 8).to(tl.int8), i < SIZE)


def unpack_raw_int4_w6_out(packed, out, *, interleave=False):
    if packed.ndim != 3 or packed.dtype not in (torch.int8, torch.uint8):
        raise ValueError("raw INT4 must be byte [E,N,K/2]")
    e, n, kh = packed.shape
    expected = (e, kh * 2 // 64, n // 16, 4, 16, 16)
    if not e or not n or not kh or n % 16 or kh % 32:
        raise ValueError("W6 requires N%16=0 and K%64=0")
    if interleave and n % 16:
        raise ValueError("gate/up interleaving requires N%16=0")
    if out.shape != expected or out.dtype != torch.int8:
        raise ValueError("output must be signed INT8 W6 [E,K/64,N/16,4,16,16]")
    if not packed.is_contiguous() or not out.is_contiguous():
        raise ValueError("raw INT4 and output must be contiguous")
    if packed.device != out.device or not packed.is_cuda:
        raise ValueError("raw INT4 and output must be on the same GPU")
    _unpack_raw_int4_w6[(triton.cdiv(out.numel(), 2048),)](
        packed, out, out.numel(), n, kh * 2, interleave, 2048, num_warps=4)
    return out


def _pack(weight, scale, scale_multiplier, prepack=False):
    if prepack:
        raise ValueError("DeepGEMM W4A8 retains raw INT4; prepack is unsupported")
    if weight.ndim != 3 or weight.dtype not in (torch.int8, torch.uint8):
        raise ValueError("checkpoint weight must be byte [E,N,K/2]")
    e, n, kh = weight.shape
    if e <= 0 or n <= 0 or kh <= 0 or n % 16 or kh % 32:
        raise ValueError("W6 requires N%16=0 and K%64=0")
    if scale.shape not in ((e, n), (e, n, 1)) or scale.device != weight.device:
        raise ValueError("per-channel scale must be [E,N] or [E,N,1] on weight device")
    return {"raw_int4": (weight, scale.reshape(e, n).float().mul(scale_multiplier).contiguous())}


def transform_int4_weights_for_mega_moe_normal(
    l1_weights, l2_weights, *, l1_scale, l2_scale,
    scale_multiplier=16.0, prepack=False,
):
    """Retain raw signed nibbles; SlimQuant stores channel scales divided by 16."""
    if not math.isfinite(scale_multiplier) or scale_multiplier <= 0:
        raise ValueError("scale_multiplier must be finite and positive")
    if (l1_weights.ndim != 3 or l2_weights.ndim != 3
            or l1_weights.shape[0] != l2_weights.shape[0]
            or l1_weights.shape[1] != 4 * l2_weights.shape[2]
            or l2_weights.shape[1] != 2 * l1_weights.shape[2]):
        raise ValueError("expected L1=[E,2I,H/2], L2=[E,H,I/2]")
    return (_pack(l1_weights, l1_scale, scale_multiplier, prepack),
            _pack(l2_weights, l2_scale, scale_multiplier, prepack))


def int4_w4a8_mega_moe(y, l1_weights, l2_weights, sym_buffer, **kwargs):
    """Expand into stream-owned scratch and invoke installed DeepGEMM INT8."""
    import deepgemm
    if torch.cuda.is_current_stream_capturing():
        raise NotImplementedError("W4A8 scratch expansion requires eager execution")
    p1, s1 = l1_weights["raw_int4"]
    p2, s2 = l2_weights["raw_int4"]
    stream = torch.cuda.current_stream(y.device)
    key = (y.device, stream.cuda_stream, tuple(p1.shape), tuple(p2.shape))
    cache = getattr(sym_buffer, "_int4_unpack_workspaces", None)
    if cache is None:
        cache = sym_buffer._int4_unpack_workspaces = {}
    if key not in cache:
        cache[key] = tuple(
            torch.empty((p.shape[0], p.shape[2] * 2 // 64, p.shape[1] // 16, 4, 16, 16),
                        dtype=torch.int8, device=p.device) for p in (p1, p2))
    w1, w2 = cache[key]
    unpack_raw_int4_w6_out(p1, w1, interleave=True)
    unpack_raw_int4_w6_out(p2, w2)
    # Match DeepGEMM's weight interleaving without retaining another INT4 copy.
    e, n = s1.shape
    s1 = s1.reshape(e, 2, n // 16, 8).permute(0, 2, 1, 3).contiguous().reshape(e, n)
    try:
        return deepgemm.fp8_mega_moe(
            y, (w1, s1), (w2, s2), sym_buffer, is_int8=True, **kwargs)
    except RuntimeError as exc:
        if "INT8 kernels not compiled" not in str(exc):
            raise
        raise RuntimeError(
            "HCU W4A8 MegaMoE requires INT8 kernels in the installed deepgemm. "
            "Rebuild/reinstall deepgemm with DG_MEGA_BUILD_INT8=1; the current "
            "build only supports FP8 MegaMoE."
        ) from exc
