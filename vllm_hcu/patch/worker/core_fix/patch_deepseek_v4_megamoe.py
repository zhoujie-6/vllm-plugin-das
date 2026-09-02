# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Route AMD DeepSeek-V4 Channel-FP8 experts through standalone MegaMoE."""

from __future__ import annotations

import functools
from types import ModuleType
from types import MethodType

from ._common import load_exact_module, require_callable, require_class


TARGET_MODULE = "vllm.models.deepseek_v4.amd.model"
PATCH_ID = "worker.core_fix.deepseek_v4_amd.megamoe_fp8"
_MARKER = "_vllm_hcu_megamoe_fp8_applied"


def _requested(vllm_config) -> bool:
    return vllm_config.kernel_config.moe_backend == "deep_gemm_mega_moe"


def apply_to_module(module: ModuleType) -> bool:
    target = load_exact_module(TARGET_MODULE, module)
    moe_cls = require_class(target, "DeepseekV4MoE", f"{TARGET_MODULE}.DeepseekV4MoE")
    causal_cls = require_class(
        target, "DeepseekV4ForCausalLM", f"{TARGET_MODULE}.DeepseekV4ForCausalLM"
    )
    if getattr(moe_cls, _MARKER, False):
        return False

    original_init = require_callable(moe_cls, "__init__", f"{TARGET_MODULE}.DeepseekV4MoE.__init__")
    original_forward = require_callable(moe_cls, "forward", f"{TARGET_MODULE}.DeepseekV4MoE.forward")
    original_load = require_callable(
        causal_cls, "load_weights", f"{TARGET_MODULE}.DeepseekV4ForCausalLM.load_weights"
    )

    @functools.wraps(original_init)
    def hcu_init(self, vllm_config, prefix=""):
        if not _requested(vllm_config):
            return original_init(self, vllm_config, prefix)

        config = vllm_config.model_config.hf_config
        if getattr(config, "expert_dtype", None) != "fp8":
            raise ValueError("DCU MegaMoE requires expert_dtype='fp8'")
        if not vllm_config.parallel_config.enable_expert_parallel:
            raise ValueError("DCU MegaMoE requires --enable-expert-parallel")

        # Reuse the official AMD gate/shared-expert construction while keeping
        # its quantization selector away from the dedicated MegaMoE backend.
        requested_backend = vllm_config.kernel_config.moe_backend
        vllm_config.kernel_config.moe_backend = "triton"
        try:
            original_init(self, vllm_config, prefix)
        finally:
            vllm_config.kernel_config.moe_backend = requested_backend

        old_experts_prefix = f"{prefix}.experts"
        compilation_config = vllm_config.compilation_config
        compilation_config.static_forward_context.pop(old_experts_prefix, None)
        while old_experts_prefix in compilation_config.static_all_moe_layers:
            compilation_config.static_all_moe_layers.remove(old_experts_prefix)

        from vllm.distributed import get_ep_group
        from vllm_hcu.models.deepseek_v4_megamoe import (
            DeepseekV4MegaMoEFP8Experts,
        )

        ep_group = get_ep_group()
        if ep_group.world_size != 8:
            raise ValueError(
                f"DCU MegaMoE requires EP8, got EP size {ep_group.world_size}"
            )
        local_experts = config.n_routed_experts // ep_group.world_size
        start = ep_group.rank_in_group * local_experts
        # Preserve the FusedMoE container because AMD's expert mapping names
        # parameters below ``experts.routed_experts``.  Only replace its routed
        # expert implementation; our forward bypasses the container kernels.
        routed_experts_prefix = f"{old_experts_prefix}.routed_experts"
        self.experts.routed_experts = DeepseekV4MegaMoEFP8Experts(
            vllm_config,
            num_experts=config.n_routed_experts,
            num_local_experts=local_experts,
            experts_start_idx=start,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            prefix=routed_experts_prefix,
        )
        self.use_mega_moe = True
        # dcu_mega_v3 stages expert ids as int64.  This is independent from
        # the dtype of the optional hash-routing lookup table.
        import torch

        self.hash_indices_dtype = torch.int64

        # The fused MegaMoE output is already present on every EP rank. Shared
        # experts therefore need their normal TP reduction before addition.
        if config.n_shared_experts is not None:
            self.shared_experts = target.DeepseekV4MLP(
                hidden_size=config.hidden_size,
                intermediate_size=(
                    config.moe_intermediate_size * config.n_shared_experts
                ),
                hidden_act=config.hidden_act,
                swiglu_limit=self.swiglu_limit,
                quant_config=vllm_config.quant_config,
                reduce_results=True,
                prefix=f"{prefix}.shared_experts",
            )

    @functools.wraps(original_forward)
    def hcu_forward(self, hidden_states, input_ids=None):
        if not getattr(self, "use_mega_moe", False):
            return original_forward(self, hidden_states, input_ids)
        if self.gate.tid2eid is not None and input_ids is None:
            raise ValueError("DeepSeek V4 hash MoE routing requires input_ids")

        from vllm.model_executor.layers.fused_moe.router.fused_topk_bias_router import (
            fused_topk_bias,
        )

        original_shape = hidden_states.shape
        router_logits, _ = self.gate(hidden_states)
        topk_weights, topk_ids = fused_topk_bias(
            hidden_states=hidden_states,
            gating_output=router_logits,
            scoring_func=self.scoring_func,
            e_score_correction_bias=(
                self.gate.e_score_correction_bias.data
                if self.gate.e_score_correction_bias is not None
                else None
            ),
            topk=self.n_activated_experts,
            renormalize=self.renormalize,
            indices_type=self.hash_indices_dtype,
            input_tokens=input_ids,
            hash_indices_table=self.gate.tid2eid,
            routed_scaling_factor=self.routed_scaling_factor,
        )
        output = self.experts.routed_experts(
            hidden_states,
            topk_weights,
            topk_ids,
            activation_clamp=(
                float(self.swiglu_limit) if self.swiglu_limit is not None else None
            ),
        )
        if self.shared_experts is not None:
            output.add_(self.shared_experts(hidden_states))
        return output.view(original_shape)

    @functools.wraps(original_load)
    def hcu_load_weights(self, weights):
        # Compressed-Tensors maps checkpoint ``.scale`` tensors to the normal
        # FusedMoE ``*_weight_scale_inv`` names.  MegaMoE intentionally owns
        # channelwise ``*_weight_scale`` parameters instead, so expose narrow
        # aliases while AutoWeightsLoader builds its parameter dictionary.
        original_named_parameters = self.named_parameters

        def named_parameters_with_megamoe_scale_aliases(module, *args, **kwargs):
            for name, param in original_named_parameters(*args, **kwargs):
                yield name, param
                if (
                    ".experts.w" in name
                    and name.endswith("_weight_scale")
                ):
                    yield f"{name}_inv", param

        self.named_parameters = MethodType(
            named_parameters_with_megamoe_scale_aliases, self
        )
        try:
            loaded = original_load(self, weights)
        finally:
            del self.named_parameters
        for layer in self.model.layers:
            ffn = getattr(layer, "ffn", None)
            if getattr(ffn, "use_mega_moe", False):
                ffn.experts.routed_experts.finalize_weights()
        return loaded

    moe_cls._vllm_hcu_original_init = original_init
    moe_cls._vllm_hcu_original_forward = original_forward
    causal_cls._vllm_hcu_original_load_weights_megamoe = original_load
    moe_cls.__init__ = hcu_init
    moe_cls.forward = hcu_forward
    causal_cls.load_weights = hcu_load_weights
    setattr(moe_cls, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "apply", "apply_to_module"]
