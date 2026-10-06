# SPDX-License-Identifier: Apache-2.0
"""Check DCP wiring on actual loaded model instances, before warmup."""

import os


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
