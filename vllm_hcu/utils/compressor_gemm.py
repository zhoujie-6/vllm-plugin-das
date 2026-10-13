# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""DeepSeek V4 compressor projection, preserving BF16 operands/FP32 output."""
from __future__ import annotations

import os
import torch
from vllm.logger import init_logger
from vllm_hcu.platforms.envs import custom_ops_enabled
from vllm_hcu.utils import bf16_fp32_gemm

logger = init_logger(__name__)
_CACHE_ATTR = "_hcu_compressor_nt_weight"
_BLASLT_FAILED = False


def _eligible(x: torch.Tensor, weight: torch.Tensor) -> bool:
    return (torch.version.hip is not None and x.is_cuda and weight.is_cuda
            and x.device == weight.device and x.ndim == weight.ndim == 2
            and x.dtype == weight.dtype == torch.bfloat16
            and x.is_contiguous() and weight.is_contiguous()
            and x.shape[0] > 0 and not weight.is_inference())


def _cached_nt(weight: torch.Tensor, nn_layout: bool) -> torch.Tensor:
    if not nn_layout:
        return weight
    # Parameters may be reloaded in place. Track version and storage/layout,
    # rather than retaining a stale projection after a weight update.
    key = (weight._version, weight.data_ptr(), tuple(weight.shape),
           tuple(weight.stride()), weight.dtype, weight.device, nn_layout)
    cached = getattr(weight, _CACHE_ATTR, None)
    if cached is None or cached[0] != key:
        nt = weight.T.contiguous() if nn_layout else weight
        ready = None
        if weight.is_cuda:
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(weight.device))
        cached = (key, nt, ready)
        setattr(weight, _CACHE_ATTR, cached)
    _, nt, ready = cached
    if ready is not None:
        stream = torch.cuda.current_stream(weight.device)
        stream.wait_event(ready)
        nt.record_stream(stream)
    return nt


def compressor_mm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    global _BLASLT_FAILED
    nn_layout = weight.shape[0] == x.shape[-1]
    rhs = weight if nn_layout else weight.T
    backend = os.environ.get("VLLM_HCU_COMPRESSOR_GEMM_BACKEND", "blaslt").lower()
    if backend not in ("torch", "torch_nt", "blaslt"):
        raise ValueError("VLLM_HCU_COMPRESSOR_GEMM_BACKEND must be torch, torch_nt or blaslt")
    if backend == "torch" or not custom_ops_enabled() or not _eligible(x, weight):
        return torch.mm(x, rhs, out_dtype=torch.float32)
    # Respect the existing BLASLt switch; retain original layout on disable.
    if backend == "blaslt" and (_BLASLT_FAILED or os.environ.get(
        "VLLM_USE_LINEAR_BF16_FP32_USE_BLASLT", "1").lower() in ("0", "false", "no", "off")):
        return torch.mm(x, rhs, out_dtype=torch.float32)
    nt = _cached_nt(weight, nn_layout)
    if backend == "torch_nt":
        return torch.mm(x, nt.T, out_dtype=torch.float32)
    try:
        # Explicit NT contract: bypass the generic helper's NN shape heuristic.
        out = bf16_fp32_gemm._jit_hipblaslt_bf16_fp32().linear_bf16_fp32(x, nt)
    except Exception as exc:
        _BLASLT_FAILED = True
        logger.warning_once("Compressor BLASLt failed; retaining original torch GEMM: %s", exc)
        return torch.mm(x, rhs, out_dtype=torch.float32)
    logger.info_once("DeepSeek V4 compressor uses BLASLt NT GEMM (BF16 inputs, FP32 output)")
    return out
