# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Count PCP ranks for EPLB without changing TP/DP topology during validation."""
from __future__ import annotations
import functools
from types import ModuleType
from ..framework_opt._common import (
    load_exact_module, require_class, require_callable, PatchCompatibilityError,
)

TARGET_MODULE = "vllm.config.parallel"
PATCH_ID = "platform.core_fix.deepseek_v4.pcp_parallel"
TARGETS = (f"{TARGET_MODULE}.ParallelConfig._validate_parallel_config",)
_MARKER = "_vllm_hcu_pcp_eplb_validator"

def apply_to_module(module: ModuleType) -> bool:
    module = load_exact_module(TARGET_MODULE, module)
    cls = require_class(module, "ParallelConfig", TARGET_MODULE)
    original = require_callable(cls, "_validate_parallel_config", TARGETS[0])
    if getattr(original, _MARKER, False):
        return False
    decorators = getattr(cls, "__pydantic_decorators__", None)
    validator = getattr(decorators, "model_validators", {}).get("_validate_parallel_config")
    if validator is None or validator.func is not original:
        raise PatchCompatibilityError("PCP requires the v0.25.1 parallel validator")

    @functools.wraps(original)
    def validate(self):
        if (self.prefill_context_parallel_size > 1 and self.enable_eplb
                and self.tensor_parallel_size * self.data_parallel_size == 1):
            from copy import copy
            from vllm.platforms import current_platform

            if not current_platform.is_cuda_alike():
                raise ValueError("EPLB requires CUDA or ROCm devices.")
            if not self.enable_expert_parallel:
                raise ValueError("enable_expert_parallel must be True to use EPLB.")
            # PCP provides the additional EP ranks missing from the upstream
            # TP*DP check. Delegate every other validation/normalization to
            # the existing validator without changing the TP/DP topology.
            eplb_config = self.eplb_config
            try:
                self.enable_eplb = False
                self.eplb_config = copy(eplb_config)
                self.eplb_config.num_redundant_experts = 0
                return original(self)
            finally:
                self.enable_eplb = True
                self.eplb_config = eplb_config
        return original(self)

    setattr(validate, _MARKER, True)
    cls._validate_parallel_config = validate
    validator.func = validate
    from pydantic.dataclasses import rebuild_dataclass
    try:
        rebuild_dataclass(cls, force=True, _types_namespace=vars(module))
    except Exception:
        cls._validate_parallel_config = original
        validator.func = original
        rebuild_dataclass(cls, force=True, _types_namespace=vars(module))
        raise
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
