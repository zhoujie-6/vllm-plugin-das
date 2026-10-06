# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Wire DCP compression into the native DeepSeek-V4 compressor class."""

import functools
import os
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
        from vllm_hcu.v1.attention.ops.deepseek_v4_ops.dcp import ContextParallelLayout
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


def audit_deepseek_v4_dcp(model, config):
    world_size = config.parallel_config.decode_context_parallel_size
    if world_size <= 1:
        return
    architectures = config.model_config.hf_config.architectures
    if "DeepseekV4ForCausalLM" not in architectures:
        return
    debug = os.environ.get("VLLM_HCU_DEEPSEEK_V4_DCP_DEBUG") == "1"
    attention_count = compressor_count = 0
    failures = []
    for name, layer in model.named_modules():
        decode = getattr(layer, "_forward_decode", None)
        prefill = getattr(layer, "_forward_prefill", None)
        if callable(decode) and callable(prefill):
            attention_count += 1
            active = all(getattr(fn, "_vllm_hcu_flashmla_sparse_decode_applied", False)
                         for fn in (decode, prefill))
            configured = getattr(layer, "dcp_world_size", None) == world_size
            if not active or not configured:
                failures.append(f"{name}: attention patched={active} dcp={getattr(layer, 'dcp_world_size', None)}")
            if debug and attention_count == 1:
                print(
                    f"[DSV4_DCP_AUDIT] file={__file__} module={name} "
                    f"class={type(layer).__module__}.{type(layer).__name__} "
                    f"decode_file={decode.__func__.__code__.co_filename} "
                    f"prefill_file={prefill.__func__.__code__.co_filename} "
                    f"heads={getattr(layer, 'n_local_heads', None)} "
                    f"dcp={getattr(layer, 'dcp_world_size', None)} patched={active}",
                    flush=True,
                )
        if type(layer).__name__ == "DeepseekCompressor":
            compressor_count += 1
            active = getattr(layer.forward, "_vllm_hcu_dcp_compressor_applied", False)
            layout = getattr(layer, "cp_layout", None)
            configured = getattr(layout, "world_size", None) == world_size
            if not active or not configured:
                failures.append(f"{name}: compressor patched={active} layout={layout}")
            if debug and compressor_count == 1:
                print(
                    f"[DSV4_DCP_AUDIT] module={name} "
                    f"class={type(layer).__module__}.{type(layer).__name__} "
                    f"forward_file={layer.forward.__func__.__code__.co_filename} "
                    f"layout={layout} patched={active}", flush=True,
                )
    if attention_count == 0:
        failures.append("No sparse attention instances found in loaded DeepSeek-V4 model")
    if failures:
        raise RuntimeError("DeepSeek-V4 DCP model wiring failed: " + "; ".join(failures[:6]))
    if debug:
        print(f"[DSV4_DCP_AUDIT] verified attention={attention_count} compressor={compressor_count}",
              flush=True)
