# SPDX-License-Identifier: Apache-2.0
"""Bounded startup-only calibration for HCU Channel-FP8 grouped GEMM."""
import inspect
from contextlib import contextmanager
from contextvars import ContextVar
from statistics import median

import torch
from vllm.logger import init_logger
from vllm_hcu.platforms import envs as henvs

_profile = ContextVar("hcu_ht_gemm_profile", default=False)
_cache = {}
_MAX_KEYS = 24
_CANDIDATES = (None, {"BLOCK_M": 64, "BLOCK_N": 256, "N_LOOP_NUM": 2},
               {"BLOCK_M": 128, "BLOCK_N": 256, "N_LOOP_NUM": 4})
logger = init_logger("vllm.hcu.ht_gemm")


@contextmanager
def ht_gemm_profile_scope():
    token = _profile.set(True)
    try:
        yield
    finally:
        _profile.reset(token)


def _supported(a, b, output, indices):
    return (a[0].device.type == "cuda" and a[0].dtype in
            (torch.float8_e4m3fn, torch.float8_e4m3fnuz)
            and output.dtype == torch.bfloat16 and b[0].ndim == 6
            and a[0].shape[0] % 256 == 0 and indices.dtype == torch.int32)


def _key(a, b, output):
    rows = a[0].shape[0]
    bucket = 1 << max(0, (rows - 1).bit_length())
    return (a[0].device, a[0].dtype, b[0].shape[0], output.shape[1],
            a[0].shape[1], a[1].shape[-1] if a[1].ndim > 1 else 1, bucket)


def _time(call):
    # Warm up/JIT outside measured trials. Only called inside model profiling.
    call()
    times = []
    for _ in range(3):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return median(times)


def _sample(indices):
    # Bound numeric-check memory; invalid/padded output rows are unspecified.
    step = max(1, indices.numel() // 128)
    rows = torch.arange(0, indices.numel(), step, device=indices.device)[:128]
    return rows[indices[rows] >= 0]


def _calibrate(fn, a, b, output, indices):
    rows = _sample(indices)
    if rows.numel() == 0:
        return None
    def call(config):
        if config is None:
            return fn(a, b, output, indices)
        return fn(a, b, output, indices, config=config)
    baseline_ms = _time(lambda: call(None))
    reference = output.index_select(0, rows)
    best, best_ms = None, baseline_ms
    for config in _CANDIDATES[1:]:
        elapsed = _time(lambda: call(config))
        check = output.index_select(0, rows)
        if torch.allclose(check, reference, rtol=0.01, atol=0.01) and elapsed < best_ms:
            best, best_ms = config, elapsed
    # Small differences are noise; retain the vendor choice unless >5% faster.
    if best_ms >= baseline_ms * 0.95:
        best = None
    logger.info("HT FP8 GEMM calibrated shape=%s, config=%s, baseline=%.3f ms, best=%.3f ms",
                _key(a, b, output), best, baseline_ms, best_ms)
    return best


def run_ht_fp8_gemm(fn, a, b, output, indices):
    enabled = henvs.optional_custom_op_enabled(henvs.VLLM_HCU_DEEPEP_HT_GEMM_TUNE)
    if not enabled or not _supported(a, b, output, indices):
        return fn(a, b, output, indices)
    key = _key(a, b, output)
    if _profile.get() and key not in _cache and len(_cache) < _MAX_KEYS:
        # Older vendor versions may lack the public config argument.
        try:
            supports_config = "config" in inspect.signature(fn).parameters
        except (TypeError, ValueError):
            supports_config = False
        if not supports_config:
            _cache[key] = None
        else:
            # Cover smaller padded-M buckets using prefix views of existing
            # scratch, never allocating another large GEMM output buffer.
            size = a[0].shape[0]
            for _ in range(4):
                av = (a[0][:size], a[1][:size])
                ov, iv = output[:size], indices[:size]
                candidate_key = _key(av, b, ov)
                if candidate_key not in _cache and len(_cache) < _MAX_KEYS:
                    _cache[candidate_key] = _calibrate(fn, av, b, ov, iv)
                size = (1 << max(0, (size - 1).bit_length())) // 2
                if size < 4096:
                    break
    config = _cache.get(key)
    # Always produce the full caller output after calibration's prefix writes.
    if config is None:
        return fn(a, b, output, indices)
    if not _profile.get():
        logger.info_once("HT FP8 GEMM using calibrated tile config outside startup profiling.")
    return fn(a, b, output, indices, config=config)
