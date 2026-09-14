# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Shard V4 MoE tokens without changing the AMD model/DSpark ABI."""
from __future__ import annotations
import functools
from types import ModuleType
import torch
from ._common import load_exact_module, require_class, require_callable, require_exact_signature, PatchCompatibilityError

TARGET_MODULE = "vllm.models.deepseek_v4.amd.model"
PATCH_ID = "worker.core_fix.deepseek_v4.pcp_model"
_MARKER = "_vllm_hcu_v4_pcp_model"


def apply_to_module(module: ModuleType) -> bool:
    module = load_exact_module(TARGET_MODULE, module)
    cls = require_class(module, "DeepseekV4MoE", TARGET_MODULE)
    if getattr(cls, _MARKER, False):
        if not getattr(cls.forward, _MARKER, False):
            raise PatchCompatibilityError("DeepSeek-V4 PCP MoE marker is stale")
        return False
    causal_cls = require_class(module, "DeepseekV4ForCausalLM", TARGET_MODULE)
    causal_init = require_callable(causal_cls, "__init__", TARGET_MODULE)
    init = require_callable(cls, "__init__", TARGET_MODULE)
    forward = require_callable(cls, "forward", TARGET_MODULE)
    require_exact_signature(forward, TARGET_MODULE + ".DeepseekV4MoE.forward",
                            positional=("self", "hidden_states", "input_ids"),
                            defaults={"input_ids": None})

    @functools.wraps(init)
    def hcu_init(self, vllm_config, prefix=""):
        init(self, vllm_config, prefix)
        from vllm_hcu.deepseek_v4_runtime import is_deepseek_v4_pcp
        self._hcu_v4_pcp = is_deepseek_v4_pcp(vllm_config)
        if self._hcu_v4_pcp:
            from vllm_hcu.model_executor.layers import deepseek_v4_pcp  # noqa: F401
            self._hcu_pcp_prefix = prefix + ".hcu_pcp_moe"
            context = vllm_config.compilation_config.static_forward_context
            if self._hcu_pcp_prefix in context:
                raise ValueError("Duplicate DeepSeek-V4 PCP MoE prefix")
            context[self._hcu_pcp_prefix] = self

    @functools.wraps(forward)
    def hcu_forward(self, hidden_states, input_ids=None):
        if not self._hcu_v4_pcp:
            return forward(self, hidden_states, input_ids)
        return torch.ops.vllm.hcu_deepseek_v4_pcp_moe(
            hidden_states, input_ids, self._hcu_pcp_prefix,
        )

    @functools.wraps(causal_init)
    def hcu_causal_init(self, *, vllm_config, prefix=""):
        causal_init(self, vllm_config=vllm_config, prefix=prefix)
        from vllm_hcu.deepseek_v4_runtime import is_deepseek_v4_pcp
        if not (is_deepseek_v4_pcp(vllm_config)
                and vllm_config.parallel_config.enable_eplb):
            return
        from types import MethodType
        from vllm.distributed import get_ep_group
        from vllm.model_executor.models.interfaces import MixtureOfExperts
        config = vllm_config.model_config.hf_config
        self.moe_layers = [layer.ffn.experts for layer in self.model.layers
                           if hasattr(layer, "ffn")]
        self.expert_weights = []
        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = getattr(config, "n_group", 1)
        self.num_logical_experts = config.n_routed_experts
        self.num_routed_experts = config.n_routed_experts
        self.num_shared_experts = config.n_shared_experts or 0
        self.num_redundant_experts = vllm_config.parallel_config.eplb_config.num_redundant_experts
        self.num_physical_experts = self.num_logical_experts + self.num_redundant_experts
        self.num_local_physical_experts = self.num_physical_experts // get_ep_group().world_size
        self.set_eplb_state = MethodType(MixtureOfExperts.set_eplb_state, self)
        self.update_physical_experts_metadata = MethodType(update_experts, self)

    def update_experts(self, num_physical_experts, num_local_physical_experts):
        if self.num_local_physical_experts != num_local_physical_experts:
            raise ValueError("PCP EPLB cannot resize the allocated local expert storage")
        self.num_physical_experts = num_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.moe_layers:
            layer.update_expert_map()

    causal_cls.__init__ = hcu_causal_init
    setattr(hcu_forward, _MARKER, True)
    cls._vllm_hcu_non_pcp_moe_forward = forward
    cls.__init__ = hcu_init
    cls.forward = hcu_forward
    setattr(cls, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))
