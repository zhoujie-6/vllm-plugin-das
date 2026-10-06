# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Keep attention-cache capacity correct under HCU context parallelism."""

from __future__ import annotations

import functools
import inspect
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    already_applied,
    load_exact_module,
    require_callable,
    require_class,
)


TARGET_MODULE = "vllm.v1.kv_cache_interface"
PATCH_ID = "platform.framework_opt.pcp_kv_cache_interface"
TARGETS = (
    f"{TARGET_MODULE}.FullAttentionSpec.max_memory_usage_bytes",
    f"{TARGET_MODULE}.SlidingWindowSpec.max_memory_usage_bytes",
)
_MARKER = "_vllm_hcu_pcp_kv_cache_interface_applied"
_FULL_WRAPPER = "_vllm_hcu_pcp_full_attention_memory_wrapper"
_SLIDING_WRAPPER = "_vllm_hcu_dcp_sliding_window_memory_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    interface = load_exact_module(TARGET_MODULE, module)
    full_attention_spec = require_class(
        interface,
        "FullAttentionSpec",
        f"{TARGET_MODULE}.FullAttentionSpec",
    )
    sliding_window_spec = require_class(
        interface,
        "SlidingWindowSpec",
        f"{TARGET_MODULE}.SlidingWindowSpec",
    )
    if already_applied(
        interface,
        _MARKER,
        (
            (full_attention_spec, "max_memory_usage_bytes", _FULL_WRAPPER),
            (sliding_window_spec, "max_memory_usage_bytes", _SLIDING_WRAPPER),
        ),
    ):
        return False

    original = require_callable(
        full_attention_spec,
        "max_memory_usage_bytes",
        TARGETS[0],
    )
    original_sliding = require_callable(
        sliding_window_spec,
        "max_memory_usage_bytes",
        TARGETS[1],
    )
    signature = inspect.signature(original)
    parameters = tuple(signature.parameters.values())
    if (
        tuple(signature.parameters) != ("self", "vllm_config")
        or any(
            parameter.kind is not inspect.Parameter.POSITIONAL_OR_KEYWORD
            or parameter.default is not inspect.Parameter.empty
            for parameter in parameters
        )
    ):
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[0]} has incompatible "
            f"signature {signature}"
        )

    sliding_signature = inspect.signature(original_sliding)
    if sliding_signature != signature:
        raise PatchCompatibilityError(
            f"required HCU patch target {TARGETS[1]} has incompatible "
            f"signature {sliding_signature}"
        )

    cdiv = require_callable(interface, "cdiv", f"{TARGET_MODULE}.cdiv")

    @functools.wraps(original)
    def hcu_max_memory_usage_bytes(self, vllm_config):
        parallel_config = vllm_config.parallel_config
        if parallel_config.prefill_context_parallel_size == 1:
            return original(self, vllm_config)

        max_model_len = vllm_config.model_config.max_model_len
        dcp_world_size = parallel_config.decode_context_parallel_size
        if dcp_world_size > 1:
            max_model_len = cdiv(max_model_len, dcp_world_size)
        return cdiv(max_model_len, self.block_size) * self.page_size_bytes

    @functools.wraps(original_sliding)
    def hcu_sliding_max_memory_usage_bytes(self, vllm_config):
        if vllm_config.parallel_config.decode_context_parallel_size == 1:
            return original_sliding(self, vllm_config)

        # DeepSeek-V4 SWA and compressor-state windows remain rank-local
        # bounded caches. Allocate the complete admission window on every DCP
        # rank; ownership/local slot mapping decides which rows are populated.
        max_blocks = self.max_admission_blocks_per_request(
            max_num_batched_tokens=(
                vllm_config.scheduler_config.max_num_batched_tokens
            ),
            max_model_len=vllm_config.model_config.max_model_len,
        )
        return max_blocks * self.page_size_bytes

    setattr(hcu_max_memory_usage_bytes, _FULL_WRAPPER, True)
    setattr(hcu_sliding_max_memory_usage_bytes, _SLIDING_WRAPPER, True)
    setattr(
        full_attention_spec,
        "_vllm_hcu_original_max_memory_usage_bytes",
        original,
    )
    setattr(
        full_attention_spec,
        "max_memory_usage_bytes",
        hcu_max_memory_usage_bytes,
    )
    setattr(
        sliding_window_spec,
        "_vllm_hcu_original_max_memory_usage_bytes",
        original_sliding,
    )
    setattr(
        sliding_window_spec,
        "max_memory_usage_bytes",
        hcu_sliding_max_memory_usage_bytes,
    )
    setattr(interface, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
