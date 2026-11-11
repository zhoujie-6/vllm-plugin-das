# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Negotiate transport defaults before upstream creates NCCL communicators."""
from __future__ import annotations

import functools
import sys
from types import ModuleType

from ._common import already_applied, load_exact_module, require_callable, require_signature_prefix

TARGET_MODULE = "vllm.distributed.parallel_state"
PATCH_ID = "platform.framework_opt.auto_network"
TARGETS = (f"{TARGET_MODULE}.init_distributed_environment",
           "vllm.v1.worker.gpu_worker.init_distributed_environment")
_MARKER = "_vllm_hcu_auto_network_applied"
_WRAPPER = "_vllm_hcu_auto_network_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    if already_applied(target, _MARKER, ((target, "init_distributed_environment", _WRAPPER),)):
        return False
    original = require_callable(target, "init_distributed_environment", TARGETS[0])
    signature = require_signature_prefix(original, TARGETS[0], (
        "world_size", "rank", "distributed_init_method", "local_rank", "backend", "timeout"))

    @functools.wraps(original)
    def initialize(*args, **kwargs):
        from vllm.config import get_current_vllm_config_or_none
        import torch.distributed as dist

        config = get_current_vllm_config_or_none()
        lifetime = None
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        a = bound.arguments
        if (config is not None and config.parallel_config.nnodes > 1
                and a["backend"] == "nccl" and not dist.is_initialized()):
            from vllm_hcu.distributed.auto_network import prepare_network

            lifetime = prepare_network(config, a["world_size"], a["rank"],
                                       a["distributed_init_method"], a["local_rank"])
        try:
            return original(*args, **kwargs)
        finally:
            # The upstream process group now owns its own reference to the same
            # multi-tenant rendezvous server. Never tear it down before this call.
            del lifetime

    setattr(initialize, _WRAPPER, True)
    target.init_distributed_environment = initialize
    # Handle an audited upstream worker imported before plugin discovery as
    # well as the normal cold-import path. Do not replace a third-party alias.
    worker = sys.modules.get("vllm.v1.worker.gpu_worker")
    if worker is not None and getattr(worker, "init_distributed_environment", None) is original:
        worker.init_distributed_environment = initialize
    setattr(target, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
