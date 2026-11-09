# SPDX-License-Identifier: Apache-2.0
"""Bounded native constructor namespaces for the HCU V41 PCP model.

Core has no decoder factory: its constructors resolve three hard-coded global
names. Reuse the exact reviewed code with private globals containing only those
name substitutions, preserving defaults/signatures/closures. This avoids copying
parameter lists or temporarily changing process-global Core symbols. Forward,
checkpoint loading and runner model-state methods stay native.
"""
from __future__ import annotations

import hashlib
import inspect
from types import FunctionType

from vllm.models.deepseek_v41.amd import model as native
from vllm.models.deepseek_v41.amd import vl_model as native_vl
from ..v41_pcp import HcuV41PcpAttention, validate_config
from ..v41_engram import HcuV41PcpEngram
from ..v41_target_engram import HcuV41HostEngram, use_host_target, validate_host_config

_DECODER_NATIVE = native.DeepseekV4DecoderLayer.__init__
_BACKBONE_NATIVE = native.DeepseekV4Model.__init__
_OUTER_NATIVE = native_vl.DeepseekV41ForCausalLM.__init__
_REVIEWED = (
    '919c98099d1ba3e443ea5a5c5569e6bd407caf52c40945fb792e9b4e85a13a86',
    'b804d0bd4d325397b0df3a094f47456a715b2d431cb06d2c7a06c6bc8332681b',
    '41b21ccf6dc5d66c98082650c8e046b977f0e61dce927aab890e23e508bab434',
)


def _constructor(function, expected_hash, substitutions):
    actual = hashlib.sha256(inspect.getsource(function).encode()).hexdigest()
    if actual != expected_hash:
        raise RuntimeError(f'HCU V41 PCP constructor source changed: {function.__qualname__}: {actual}')
    if not substitutions or any(name not in function.__globals__ for name in substitutions):
        raise RuntimeError('HCU V41 PCP constructor class-name anchor missing')
    namespace = dict(function.__globals__)
    namespace.update(substitutions)
    clone = FunctionType(function.__code__, namespace, function.__name__,
                         function.__defaults__, function.__closure__)
    clone.__kwdefaults__ = function.__kwdefaults__
    clone.__annotations__ = function.__annotations__
    clone.__qualname__ = function.__qualname__
    return clone


def check_native_contract():
    if (native.DeepseekV4DecoderLayer.__init__ is not _DECODER_NATIVE
            or native.DeepseekV4Model.__init__ is not _BACKBONE_NATIVE
            or native_vl.DeepseekV41ForCausalLM.__init__ is not _OUTER_NATIVE):
        raise RuntimeError('HCU V41 PCP native constructors changed after preparation')


def select_attention(cfg):
    if not validate_config(cfg):
        return native._select_dsv4_attn_cls(cfg)
    return HcuV41PcpAttention



def select_moe(vllm_config, *args, **kwargs):
    if getattr(getattr(vllm_config, "kernel_config", None), "moe_backend", None) == "deep_gemm_mega_moe":
        from ..v41_megamoe_model import HcuV41MegaMoE
        return HcuV41MegaMoE(vllm_config, *args, **kwargs)
    return native.DeepseekV4MoE(vllm_config, *args, **kwargs)


_DECODER_INIT = _constructor(_DECODER_NATIVE, _REVIEWED[0],
                             {'_select_dsv4_attn_cls': select_attention,
                              'Engram': HcuV41PcpEngram,
                              'DeepseekV4MoE': select_moe})


class HcuV41PcpDecoderLayer(native.DeepseekV4DecoderLayer):
    __init__ = _DECODER_INIT


_BACKBONE_INIT = _constructor(_BACKBONE_NATIVE, _REVIEWED[1],
                              {'DeepseekV4DecoderLayer': HcuV41PcpDecoderLayer})


class HcuV41PcpBackbone(native.DeepseekV4Model):
    def __init__(self, *, vllm_config, prefix=''):
        if validate_config(vllm_config):
            _BACKBONE_INIT(self, vllm_config=vllm_config, prefix=prefix)
        else:
            super().__init__(vllm_config=vllm_config, prefix=prefix)


class HcuV41PcpLLM(native.DeepseekV41LLMForCausalLM):
    model_cls = HcuV41PcpBackbone


_OUTER_INIT = _constructor(_OUTER_NATIVE, _REVIEWED[2],
                           {'DeepseekV41LLMForCausalLM': HcuV41PcpLLM})


_HOST_DECODER_INIT = _constructor(_DECODER_NATIVE, _REVIEWED[0],
                                   {'Engram': HcuV41HostEngram})


class HcuV41HostDecoderLayer(native.DeepseekV4DecoderLayer):
    __init__ = _HOST_DECODER_INIT


_HOST_BACKBONE_INIT = _constructor(_BACKBONE_NATIVE, _REVIEWED[1],
                                    {'DeepseekV4DecoderLayer': HcuV41HostDecoderLayer})


class HcuV41HostBackbone(native.DeepseekV4Model):
    def __init__(self, *, vllm_config, prefix=''):
        validate_host_config(vllm_config)
        _HOST_BACKBONE_INIT(self, vllm_config=vllm_config, prefix=prefix)


class HcuV41HostLLM(native.DeepseekV41LLMForCausalLM):
    model_cls = HcuV41HostBackbone


_HOST_OUTER_INIT = _constructor(_OUTER_NATIVE, _REVIEWED[2],
                                 {'DeepseekV41LLMForCausalLM': HcuV41HostLLM})


class HcuV41PcpForCausalLM(native_vl.DeepseekV41ForCausalLM):
    def __init__(self, *, vllm_config, prefix=''):
        from ..v41_pcp_activation import check_health
        check_health()
        check_native_contract()
        from ..environment import static_expert_map_path
        if static_expert_map_path() and vllm_config.kernel_config.moe_backend != "deep_gemm_mega_moe":
            raise ValueError("The V41 static expert map requires deep_gemm_mega_moe")
        if vllm_config.parallel_config.prefill_context_parallel_size > 1:
            from ..v41_pcp_activation import pcp_enabled
            if not pcp_enabled():
                raise ValueError('PCP attention requires VLLM_HCU_DSV41_PCP_ATTENTION=1')
        if validate_config(vllm_config):
            _OUTER_INIT(self, vllm_config=vllm_config, prefix=prefix)
        elif use_host_target(vllm_config):
            _HOST_OUTER_INIT(self, vllm_config=vllm_config, prefix=prefix)
        else:
            # PCP1 without the explicit host route remains fully native.
            super().__init__(vllm_config=vllm_config, prefix=prefix)
