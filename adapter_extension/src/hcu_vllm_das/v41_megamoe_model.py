# SPDX-License-Identifier: Apache-2.0
"""Instance-local reuse of the official V4.1 MoE and public MoE factory."""
from __future__ import annotations

from vllm.config import get_current_vllm_config
from vllm.models.deepseek_v41.amd import model as native
from vllm.model_executor.layers.fused_moe.layer import FusedMoEFactory
from .v41_static_eplb import static_plan


def mega_factory(*args, **kwargs):
    cfg = get_current_vllm_config()
    plan = static_plan(cfg)
    if plan is not None:
        # Allocate physical slots through the public factory. The static
        # mapper below owns placement; the worker never starts dynamic EPLB.
        kwargs["enable_eplb"] = True
        kwargs["num_redundant_experts"] = plan.num_redundant_experts
    runner = FusedMoEFactory(*args, **kwargs)
    if plan is not None:
        runner.router.eplb_state = None
    return runner


def base_init():
    import hashlib
    import inspect
    from types import FunctionType
    fn = native.DeepseekV4MoEBase.__init__
    if hashlib.sha256(inspect.getsource(fn).encode()).hexdigest() != BASE_HASH:
        raise RuntimeError("V41 MegaMoE native base constructor changed")
    namespace = dict(fn.__globals__)
    if namespace.get("FusedMoEFactory") is not FusedMoEFactory:
        raise RuntimeError("V41 MegaMoE factory has a foreign owner")
    namespace["FusedMoEFactory"] = mega_factory
    clone = FunctionType(fn.__code__, namespace, fn.__name__,
                         fn.__defaults__, fn.__closure__)
    clone.__kwdefaults__ = fn.__kwdefaults__
    return clone


BASE_HASH = "3396ecd9e48af5d6c094d392ffb62ad4271ec94cea4c42e6a037c5f398d26610"


class MegaBase(native.DeepseekV4MoEBase):
    def __init__(self, vllm_config, prefix="", **kwargs):
        if vllm_config.kernel_config.moe_backend == "deep_gemm_mega_moe":
            base_init()(self, vllm_config, prefix=prefix, **kwargs)
        else:
            super().__init__(vllm_config, prefix=prefix, **kwargs)


class HcuV41MegaMoE(native.DeepseekV4MoE, MegaBase):
    def __init__(self, vllm_config, prefix="", use_sequence_parallel=False):
        # Native V4.1 super() resolves to MegaBase via this instance's MRO.
        # All native router/vision/shared/DSpark setup is retained.
        super().__init__(vllm_config, prefix, use_sequence_parallel)
        if vllm_config.kernel_config.moe_backend == "deep_gemm_mega_moe":
            moe = self.experts.moe_config
            self.n_logical_experts = moe.num_logical_experts
            self.n_physical_experts = moe.num_experts
            self.n_local_physical_experts = moe.num_local_experts
            self.n_redundant_experts = moe.num_experts - moe.num_logical_experts
