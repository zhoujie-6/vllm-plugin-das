# SPDX-License-Identifier: Apache-2.0
"""DSV4.1 channel INT4 experts with separate DeepEP HT and grouped GEMMs."""
import torch
from vllm.distributed import get_ep_group
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import deep_gemm_utils as core_layout
from vllm.model_executor.layers.fused_moe import modular_kernel as mk
from vllm.model_executor.layers.fused_moe.all2all_utils import maybe_make_prepare_finalize
from vllm.model_executor.layers.fused_moe.prepare_finalize.deepep_ht import DeepEPHTPrepareAndFinalize
from vllm.model_executor.layers.fused_moe.utils import _resize_cache
from vllm.model_executor.layers.quantization.utils.int8_utils import per_token_quant_int8

from .dsv41_deepgemm import ChannelDeepGemmExperts, _permute
from .dsv41_w4a8_megamoe import DSV41W4A8Method, W4A8MegaExperts, unpack_ct_words


def pack_ct_grouped_weight(words):
    from deepgemm.group_gemm_weight_pack import (
        pack_w4a8_moe_hipc_weight, pack_w4a8_weight_k_contiguous,
    )
    packed = words.view(torch.int8)
    for idx in range(words.shape[0]):
        row = slice(idx, idx + 1)
        nibbles = unpack_ct_words(words[row])
        packed[row].copy_(pack_w4a8_moe_hipc_weight(pack_w4a8_weight_k_contiguous(nibbles)))
    return packed


class W4A8GroupedExperts(ChannelDeepGemmExperts):
    _supports_quant_scheme = staticmethod(W4A8MegaExperts._supports_quant_scheme)

    def apply(self, output, hidden_states, w1, w2, topk_weights, topk_ids,
              activation, global_num_experts, expert_map, a1q_scale, a2_scale,
              workspace13, workspace2, expert_tokens_meta, apply_router_weight_on_input):
        from deepgemm import m_grouped_w4a8_gemm_nt_contiguous_hipc as grouped_gemm
        from lightop import fuse_silu_mul_clamp_quant

        if apply_router_weight_on_input or hidden_states.dtype != torch.bfloat16 or a1q_scale is not None:
            raise ValueError("DSV4.1 grouped W4A8 requires BF16 dispatch and output weighting")
        hidden = self.moe_config.hidden_dim
        intermediate = self.moe_config.intermediate_size_per_partition
        rows, _ = core_layout.compute_aligned_M_and_alignment(
            hidden_states.size(0), topk_ids.size(1), self.moe_config.num_local_experts,
            256, expert_tokens_meta)
        q, scales = per_token_quant_int8(hidden_states.contiguous())
        q_sorted = _resize_cache(workspace13.view(torch.int8), (rows, hidden))
        q_sorted, scales, ids, inverse, _ = _permute(
            q, scales, topk_ids, self.moe_config.num_local_experts,
            expert_map, expert_tokens_meta, aq_out=q_sorted, block_size=hidden)
        gate_up = _resize_cache(workspace2, (rows, 2 * intermediate))
        grouped_gemm((q_sorted, scales), (self.packed_w1, self.w1_scale), gate_up, ids)
        q2 = _resize_cache(workspace13.view(torch.int8), (rows, intermediate))
        activated = fuse_silu_mul_clamp_quant(
            gate_up, self.quant_config.gemm1_clamp_limit, output=q2)
        down = _resize_cache(workspace2, (rows, hidden))
        grouped_gemm(activated, (self.packed_w2, self.w2_scale), down, ids)
        core_layout.deepgemm_unpermute_and_reduce(
            down, topk_ids, topk_weights, inverse, expert_map, output)


class HcuDSV41W4A8DeepGemmMethod(DSV41W4A8Method):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.experts_cls = W4A8GroupedExperts

    @torch.inference_mode()
    def _process_weights(self, layer):
        if self._processed:
            raise RuntimeError("W4A8 weights already packed")
        if layer.apply_router_weight_on_input or layer.swiglu_limit != 10.0:
            raise ValueError("DSV4.1 grouped W4A8 requires output weighting and clamp=10")
        quant = self.get_fused_moe_quant_config(layer)
        prepare = maybe_make_prepare_finalize(self.moe, quant,
            routing_tables=layer._expert_routing_tables(), allow_new_interface=True)
        if not isinstance(prepare, DeepEPHTPrepareAndFinalize):
            raise ValueError("DSV4.1 grouped W4A8 requires DeepEP HT")
        for w, scale in ((layer.w13_weight_packed, layer.w13_weight_scale),
                         (layer.w2_weight_packed, layer.w2_weight_scale)):
            w.data = pack_ct_grouped_weight(w.data)
            # The native HIPC kernel multiplies signed INT4 values by 16.
            scale.data.div_(16)
        self.moe_quant_config = quant
        experts = W4A8GroupedExperts(
            self.moe, quant, layer.w13_weight_packed, layer.w2_weight_packed)
        self.moe_kernel = mk.FusedMoEKernel(prepare, experts)
        self._processed = True
        if get_ep_group().rank_in_group == 0:
            init_logger(f"vllm.{__name__}").info_once(
                "HCU DSV4.1 W4A8 grouped DeepGEMM ready: DeepEP HT, contiguous HIPC, LightOp clamp")
