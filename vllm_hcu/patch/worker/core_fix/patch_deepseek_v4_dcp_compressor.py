# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Wire DCP compression into the native DeepSeek-V4 compressor class."""

import functools
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = "vllm.models.deepseek_v4.compressor"
PATCH_ID = "worker.core_fix.deepseek_v4.dcp_compressor"
_MARKER = "_vllm_hcu_dcp_compressor_applied"


def apply_to_module(module: ModuleType) -> bool:
    module = load_exact_module(TARGET_MODULE, module)
    cls = require_class(module, "DeepseekCompressor", TARGET_MODULE)
    original_init = require_callable(cls, "__init__", TARGET_MODULE)
    original_forward = require_callable(cls, "forward", TARGET_MODULE)
    if getattr(module, _MARKER, False):
        if not all(getattr(fn, _MARKER, False) for fn in (cls.__init__, cls.forward)):
            raise PatchCompatibilityError("stale DeepSeek-V4 DCP compressor patch")
        return False
    require_exact_signature(
        original_forward, f"{TARGET_MODULE}.DeepseekCompressor.forward",
        positional=("self", "kv_score", "positions", "rotary_emb"),
    )

    @functools.wraps(original_init)
    def init(self, *args, **kwargs):
        config = args[0] if args else kwargs["vllm_config"]
        original_init(self, *args, **kwargs)
        from vllm_hcu.v1.cp_layout import ContextParallelLayout
        from vllm.distributed import get_dcp_group

        self.cp_layout = ContextParallelLayout.from_config(config)
        if self.cp_layout.enabled and self.cp_layout.interleave_size != 1:
            raise NotImplementedError("DeepSeek-V4 DCP requires interleave size 1")
        self.dcp_group = get_dcp_group() if self.cp_layout.enabled else None
        if self.cp_layout.enabled:
            from vllm.logger import init_logger

            init_logger(__name__).info_once(
                "DeepSeek-V4 native compressor DCP enabled: world_size=%d rank=%d",
                self.cp_layout.world_size, self.cp_layout.rank,
            )

    @functools.wraps(original_forward)
    def forward(self, kv_score, positions, rotary_emb):
        if not self.cp_layout.enabled:
            return original_forward(self, kv_score, positions, rotary_emb)
        from vllm_hcu.model_executor.layers.deepseek_v4_dcp_compressor import (
            DeepseekV4DCPCompressor,
        )

        return DeepseekV4DCPCompressor.forward(self, kv_score, positions, rotary_emb)

    for fn in (init, forward):
        setattr(fn, _MARKER, True)
    cls.__init__ = init
    cls.forward = forward
    setattr(module, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
