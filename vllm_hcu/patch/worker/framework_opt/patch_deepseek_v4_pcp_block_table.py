# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""MRV1 V4 PCP writes every KV slot on every rank."""
from __future__ import annotations
import functools
from types import ModuleType
from ._common import load_exact_module, require_class, require_callable

TARGET_MODULE = "vllm.v1.worker.block_table"
PATCH_ID = "worker.framework_opt.deepseek_v4.pcp_block_table"
_MARKER = "_vllm_hcu_v4_full_kv_block_table"


def apply_to_module(module: ModuleType) -> bool:
    module = load_exact_module(TARGET_MODULE, module)
    cls = require_class(module, "BlockTable", TARGET_MODULE)
    original = require_callable(cls, "__init__", TARGET_MODULE)
    if getattr(original, _MARKER, False):
        return False

    @functools.wraps(original)
    def hcu_init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        if self.pcp_world_size <= 1:
            return
        from vllm.config import get_current_vllm_config_or_none
        from vllm_hcu.deepseek_v4_runtime import is_deepseek_v4_pcp
        if is_deepseek_v4_pcp(get_current_vllm_config_or_none()):
            self.pcp_world_size = 1
            self.pcp_rank = 0

    setattr(hcu_init, _MARKER, True)
    cls.__init__ = hcu_init
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
