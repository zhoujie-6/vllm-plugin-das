# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""V4 hybrid KV groups retain full per-rank ownership under PCP."""
from __future__ import annotations
import functools
import inspect
from types import ModuleType
from ._common import load_exact_module, require_callable, PatchCompatibilityError

TARGET_MODULE = "vllm.v1.core.kv_cache_coordinator"
PATCH_ID = "platform.framework_opt.deepseek_v4.pcp_cache"
TARGETS = (f"{TARGET_MODULE}.get_kv_cache_coordinator",)
_MARKER = "_vllm_hcu_v4_pcp_cache"


def has_v4_cache(kv_cache_config):
    def is_v4(spec):
        return (getattr(spec, "model_version", None) == "deepseek_v4"
                or any(is_v4(child) for child in
                       getattr(spec, "kv_cache_specs", {}).values()))
    return any(is_v4(group.kv_cache_spec)
               for group in kv_cache_config.kv_cache_groups)


def apply_to_module(module: ModuleType) -> bool:
    module = load_exact_module(TARGET_MODULE, module)
    original = require_callable(module, "get_kv_cache_coordinator", TARGET_MODULE)
    if getattr(original, _MARKER, False):
        return False
    sig = inspect.signature(original)
    required = ("kv_cache_config", "max_model_len", "max_num_batched_tokens",
                "use_eagle", "enable_caching", "enable_kv_cache_events",
                "dcp_world_size", "pcp_world_size", "scheduler_block_size",
                "hash_block_size", "metrics_collector")
    if tuple(sig.parameters) != required:
        raise PatchCompatibilityError("DeepSeek-V4 PCP cache coordinator ABI changed")

    @functools.wraps(original)
    def hcu_coordinator(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        if (bound.arguments["pcp_world_size"] > 1
                and has_v4_cache(bound.arguments["kv_cache_config"])):
            bound.arguments["pcp_world_size"] = 1
        return original(*bound.args, **bound.kwargs)

    setattr(hcu_coordinator, _MARKER, True)
    module.get_kv_cache_coordinator = hcu_coordinator
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
