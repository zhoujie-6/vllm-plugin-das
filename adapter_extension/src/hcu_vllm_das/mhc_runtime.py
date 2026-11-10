# SPDX-License-Identifier: Apache-2.0
"""HCU HIP mHC provider policy and staged previous-sublayer Pre semantics.

No Core globals or AITER bridge classes are modified. Vendor calls are wrapped
in Plugin-owned torch custom ops for compilation; empty and unsupported shapes
are handled before dispatch. Native Torch references remain the fallback.
"""
from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
import inspect
import logging
from types import SimpleNamespace

import torch

from .environment import _flag, custom_ops_enabled

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Policy:
    master: bool
    mhc: bool
    pre: bool
    post: bool
    tilelang: bool
    fused: bool

    @property
    def pre_enabled(self):
        return self.master and self.mhc and self.pre

    @property
    def post_enabled(self):
        return self.master and self.mhc and self.post


def v41_model():
    from vllm.config import get_current_vllm_config_or_none
    from .quantization.dsv41_channel_fp8 import _is_v41

    config = get_current_vllm_config_or_none()
    return config is not None and _is_v41(config.model_config.hf_config)


def read_policy():
    return Policy(custom_ops_enabled(), *(
        _flag(name, default=False) for name in (
            'VLLM_HCU_USE_AITER_MHC', 'VLLM_HCU_USE_AITER_MHC_PRE',
            'VLLM_HCU_USE_AITER_MHC_POST', 'VLLM_HCU_USE_AITER_TILELANG_MHC',
            'VLLM_HCU_USE_AITER_MHC_FUSED',
        )
    ))


policy = Policy(False, False, False, False, False, False)
provider = SimpleNamespace(mhc_pre=None, mhc_post=None,
                           mhc_pre_gemm_sqrsum=None, mhc_pre_big_fuse=None)
_registered = False



def _provider_signature(function, name):
    signature = inspect.signature(function)
    if tuple(signature.parameters) != ('args', 'kwargs'):
        return signature
    # AITER torch_compile_guard hides the Python signature. Its registered
    # dispatcher schema retains the actual argument names and defaults.
    packet = getattr(torch.ops.aiter, name, None)
    operation = getattr(packet, 'default', None)
    schema = getattr(operation, '_schema', None)
    if schema is None or schema.name != f'aiter::{name}':
        raise RuntimeError(f'HCU mHC provider has no verifiable schema: {name}')
    parameters = []
    for argument in schema.arguments:
        kind = (inspect.Parameter.KEYWORD_ONLY if argument.kwarg_only
                else inspect.Parameter.POSITIONAL_OR_KEYWORD)
        default = (argument.default_value if argument.has_default_value()
                   else inspect.Parameter.empty)
        parameters.append(inspect.Parameter(argument.name, kind, default=default))
    return inspect.Signature(parameters)


def load_provider():
    try:
        module = import_module('aiter.ops.mhc')
    except ImportError as error:
        logger.warning('HCU_MHC_HIP_UNAVAILABLE: native fallback: %s', error)
        return SimpleNamespace(**{name: None for name in vars(provider)})
    specs = {
        'mhc_pre': ('residual', 'fn', 'hc_scale', 'hc_base', 'rms_eps', 'hc_pre_eps',
                    'hc_sinkhorn_eps', 'hc_post_mult_value', 'sinkhorn_repeat', 'use_tf32'),
        'mhc_post': ('out', 'x', 'residual', 'post_layer_mix', 'comb_res_mix'),
        'mhc_pre_gemm_sqrsum': ('out', 'sqrsum', 'x', 'fn', 'tile_k', 'use_tf32'),
        'mhc_pre_big_fuse': ('post_mix', 'comb_mix', 'layer_input', 'gemm_out_mul',
                            'gemm_out_sqrsum', 'hc_scale', 'hc_base', 'residual',
                            'rms_eps', 'hc_pre_eps', 'hc_sinkhorn_eps',
                            'hc_post_mult_value', 'sinkhorn_repeat'),
    }
    result = {}
    for name, parameters in specs.items():
        function = getattr(module, name, None)
        if not callable(function):
            result[name] = None
            continue
        signature = _provider_signature(function, name)
        actual = tuple(signature.parameters)
        if actual[:len(parameters)] != parameters:
            raise RuntimeError(f'HCU mHC provider signature changed: {name}: {actual}')
        signature.bind(*([object()] * len(parameters)))
        result[name] = function
    return SimpleNamespace(**result)


def configure(selected):
    global policy, provider
    if selected.tilelang or selected.fused:
        raise ValueError('HCU mHC requires TILELANG=False and FUSED=False')
    policy = selected
    provider = load_provider() if selected.pre_enabled or selected.post_enabled else SimpleNamespace(
        mhc_pre=None, mhc_post=None, mhc_pre_gemm_sqrsum=None, mhc_pre_big_fuse=None)
    logger.info('HCU mHC policy: MASTER=%s MHC=%s PRE=%s POST=%s TILELANG=%s FUSED=%s',
                selected.master, selected.mhc, selected.pre, selected.post,
                selected.tilelang, selected.fused)
    logger.info('HCU mHC providers: PRE=%s POST=%s DELAYED=%s',
                provider.mhc_pre is not None, provider.mhc_post is not None,
                provider.mhc_pre_gemm_sqrsum is not None and provider.mhc_pre_big_fuse is not None)
    if selected.pre_enabled and (
        provider.mhc_pre is None or provider.mhc_pre_gemm_sqrsum is None
        or provider.mhc_pre_big_fuse is None
    ):
        logger.warning('HCU_MHC_PRE_CAPABILITY_MISSING: unavailable routes use native Torch')
    if selected.post_enabled and provider.mhc_post is None:
        logger.warning('HCU_MHC_POST_CAPABILITY_MISSING: using native Torch')


def _gpu_tensors(tensors):
    return all(t.is_cuda and t.device == tensors[0].device and t.is_contiguous()
               for t in tensors)


def pre_supported(residual, fn, scale, base, pre_mix=None, x=None):
    if residual.ndim != 3 or residual.shape[1] != 4:
        return False
    tokens, hc, hidden = residual.shape
    # The gfx938 AITER stage2 returns incorrect/non-repeatable combination
    # matrices for the V4.1 odd-token H5120 shapes. Delegate this seam to
    # the official Torch implementation; even prefill batches stay eligible.
    if tokens % 2 == 1 and hidden == 5120:
        return False
    tensors = [residual, fn, scale, base]
    if pre_mix is not None:
        if pre_mix.shape != (tokens, hc) or pre_mix.dtype != torch.float32:
            return False
        tensors.append(pre_mix)
    if x is not None:
        if x.shape != (tokens, hidden) or x.dtype != torch.bfloat16:
            return False
        tensors.append(x)
    return (tokens > 0 and hidden >= 1024 and hidden % 1024 == 0
            and residual.dtype == torch.bfloat16
            and fn.shape == (24, hidden if x is not None else 4 * hidden)
            and scale.shape == (3,) and base.shape == (24,)
            and fn.dtype == scale.dtype == base.dtype == torch.float32
            and _gpu_tensors(tensors))


def post_supported(x, residual, post, comb):
    if residual.ndim < 3:
        return False
    hidden = residual.shape[-1]
    outer = residual.shape[:-2]
    return (residual.numel() > 0 and residual.shape[-2] == 4
            and hidden >= 1024 and hidden % 1024 == 0 and hidden != 5120
            and x.shape == (*outer, hidden)
            and post.shape == (*outer, 4, 1) and comb.shape == (*outer, 4, 4)
            and residual.dtype in (torch.bfloat16, torch.float16)
            and x.dtype == residual.dtype and post.dtype == comb.dtype == torch.float32
            and _gpu_tensors([residual, x, post, comb]))


def empty_pre(residual):
    outer, hc, hidden = residual.shape[:-2], residual.shape[-2], residual.shape[-1]
    return (torch.empty((*outer, hc, 1), device=residual.device, dtype=torch.float32),
            torch.empty((*outer, hc, hc), device=residual.device, dtype=torch.float32),
            torch.empty((*outer, hidden), device=residual.device, dtype=residual.dtype),
            torch.empty((*outer, hc), device=residual.device, dtype=torch.float32))


def _pre(residual: torch.Tensor, fn: torch.Tensor, hc_scale: torch.Tensor,
         hc_base: torch.Tensor, rms_eps: float, hc_pre_eps: float,
         hc_sinkhorn_eps: float, hc_post_mult_value: float,
         sinkhorn_repeat: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return provider.mhc_pre(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                            hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, False)


def _pre_fake(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
              hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat):
    return empty_pre(residual)[:3]


def _delayed(residual: torch.Tensor, fn: torch.Tensor, hc_scale: torch.Tensor,
             hc_base: torch.Tensor, rms_eps: float, hc_pre_eps: float,
             hc_sinkhorn_eps: float, hc_post_mult_value: float, sinkhorn_repeat: int,
             pre_mix: torch.Tensor | None = None,
             x: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    tokens, hc, hidden = residual.shape
    projected = residual
    if x is not None:
        # Repetition preserves mean-square normalization of the narrow model
        # entry. Only the first HC lane contributes to the projection.
        projected = x.unsqueeze(1).expand(-1, hc, -1).contiguous()
        expanded = fn.new_zeros((24, hc * hidden))
        expanded[:, :hidden].copy_(fn)
        fn = expanded
    padded = torch.empty((1, tokens, 32), device=residual.device, dtype=torch.float32)
    gemm = padded[:, :, :24]
    sqrsum = torch.empty((1, tokens), device=residual.device, dtype=torch.float32)
    post, comb, unused_input, _ = empty_pre(residual)
    provider.mhc_pre_gemm_sqrsum(gemm, sqrsum, projected, fn, 64, False)
    provider.mhc_pre_big_fuse(post, comb, unused_input, gemm, sqrsum, hc_scale,
                             hc_base, residual, rms_eps, hc_pre_eps, hc_sinkhorn_eps,
                             hc_post_mult_value, sinkhorn_repeat)
    mixes = gemm.sum(0)[:, :hc] * torch.rsqrt(
        sqrsum.sum(0)[:, None] / (hc * hidden) + rms_eps)
    next_pre = torch.sigmoid(mixes * hc_scale[0] + hc_base[:hc]) + hc_pre_eps
    layer_input = residual[:, 0].clone() if pre_mix is None else (
        (residual.float() * pre_mix.unsqueeze(-1)).sum(1).to(residual.dtype))
    return post, comb, layer_input, next_pre


def _delayed_fake(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
                  hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat,
                  pre_mix=None, x=None):
    return empty_pre(residual)


def _post(x: torch.Tensor, residual: torch.Tensor, post_layer_mix: torch.Tensor,
          comb_res_mix: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(residual)
    provider.mhc_post(out, x, residual, post_layer_mix, comb_res_mix)
    return out


def _post_fake(x, residual, post_layer_mix, comb_res_mix):
    return torch.empty_like(residual)


def register_ops():
    global _registered
    if _registered:
        return
    from vllm.utils.torch_utils import direct_register_custom_op
    for name, function, fake in (
        ('hcu_mhc_hip_pre', _pre, _pre_fake),
        ('hcu_mhc_hip_delayed', _delayed, _delayed_fake),
        ('hcu_mhc_hip_post', _post, _post_fake),
    ):
        direct_register_custom_op(op_name=name, op_func=function, mutates_args=[],
                                  fake_impl=fake, dispatch_key='CompositeExplicitAutograd')
    _registered = True
