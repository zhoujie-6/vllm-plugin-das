# SPDX-License-Identifier: Apache-2.0
"""Packed CT INT4 weights and token-INT8 activations for DSV4.1 MegaMoE."""
import torch

from vllm.distributed import get_ep_group
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a8_int8 import (
    CompressedTensorsW4A8Int8MoEMethod,
)
from vllm.model_executor.layers.quantization.utils.int8_utils import per_token_quant_int8
from vllm.model_executor.layers.quantization.utils.quant_utils import kInt4Static, kInt8DynamicTokenSym
from vllm.model_executor.utils import set_weight_attrs

from .dsv41_megamoe import (
    DSV41MegaMoEMethodMixin, MegaExperts, MegaKernel, MegaPrepare, _layer_buffer,
)


def w4a8_provider():
    import deepgemm.mega as mega
    from deepgemm.mega import _C
    if not getattr(_C, "w4a8_dsv41_full_row_clamp", False):
        raise RuntimeError("DeepGEMM must provide the clamped 5120/2304 W4A8 MegaMoE kernel")
    return mega


def unpack_ct_words(words):
    """CT packs the earliest unsigned offset-binary nibble in the low bits."""
    if words.dtype != torch.int32:
        raise ValueError("Expected INT32 compressed-tensors weight_packed")
    pairs = words.contiguous().view(torch.uint8)
    return (torch.stack((pairs & 15, pairs >> 4), dim=-1).flatten(-2) ^ 8)


class W4A8MegaExperts(MegaExperts):
    @staticmethod
    def _supports_quant_scheme(weight_key, activation_key):
        return (weight_key, activation_key) == (kInt4Static, kInt8DynamicTokenSym)

    def stage(self, x, ids, weights):
        # Use Core's token quantizer, including its near-zero floor and tie rounding.
        q, scale = per_token_quant_int8(x)
        n = x.shape[0]
        self.buffer.x[:n].copy_(q)
        self.buffer.x_sf[:n].copy_(scale.flatten())
        self.buffer.topk_idx[:n].copy_(ids)
        self.buffer.topk_weights[:n].copy_(weights)

    def launch(self, output):
        self.mega.fp8_mega_moe(
            output, self.l1, self.l2, self.buffer, mode="w4a8",
            activation_clamp=self.quant_config.gemm1_clamp_limit,
            w4a8_scales_predivided=True)


class DSV41W4A8Method(DSV41MegaMoEMethodMixin, CompressedTensorsW4A8Int8MoEMethod):
    def create_weights(self, layer, num_experts, hidden_size,
                       intermediate_size_per_partition, params_dtype, **extra_weight_attrs):
        for name, n, k in (("w13", 2 * intermediate_size_per_partition, hidden_size),
                           ("w2", hidden_size, intermediate_size_per_partition)):
            for suffix, shape, dtype in (
                ("weight_packed", (num_experts, n, k // 8), torch.int32),
                ("weight_scale", (num_experts, n, 1), torch.float32),
                ("weight_shape", (num_experts, 2), torch.int64),
            ):
                param = torch.nn.Parameter(torch.empty(shape, dtype=dtype), requires_grad=False)
                attrs = dict(extra_weight_attrs)
                if suffix == "weight_scale":
                    attrs["quant_method"] = "channel"
                set_weight_attrs(param, attrs)
                layer.register_parameter(f"{name}_{suffix}", param)

    def get_fused_moe_quant_config(self, layer):
        return FusedMoEQuantConfig.make(
            torch.int8, weight_dtype="int4", per_act_token_quant=True, per_out_ch_quant=True,
            w1_scale=layer.w13_weight_scale, w2_scale=layer.w2_weight_scale,
            gemm1_clamp_limit=layer.swiglu_limit)


class HcuDSV41W4A8MegaMoEMethod(DSV41W4A8Method):
    def __init__(self, weight_quant, input_quant, moe, prefix):
        super().__init__(weight_quant, input_quant, moe, prefix)
        self.experts_cls = W4A8MegaExperts
        self.mega = w4a8_provider()

    @torch.inference_mode()
    def _process_weights(self, layer):
        if self._processed:
            raise RuntimeError("MegaMoE packed weights cannot be postprocessed twice")
        if (layer.apply_router_weight_on_input or layer.expert_map_manager.num_fused_shared_experts
                or layer.swiglu_limit != 10.0):
            raise ValueError("DSV4.1 W4A8 requires separate shared experts, output routing and clamp=10")
        ep = get_ep_group()
        if ep.world_size != self.moe.ep_size:
            raise ValueError("MegaMoE EP group differs from expert placement")
        h, i = self.moe.hidden_dim, self.moe.intermediate_size_per_partition
        for meta, shape in ((layer.w13_weight_shape, (i, h)), (layer.w2_weight_shape, (h, i))):
            if not torch.equal(meta, meta.new_tensor(shape).expand_as(meta)):
                raise ValueError("Unexpected DSV4.1 checkpoint expert weight_shape")
        w1, w2 = layer.w13_weight_packed, layer.w2_weight_packed
        # Reuse checkpoint storage, expanding just one local expert at a time.
        p1, p2 = w1.data.view(torch.int8), w2.data.view(torch.int8)
        s1 = layer.w13_weight_scale.data.squeeze(-1)
        s2 = layer.w2_weight_scale.data.squeeze(-1)
        for idx in range(w1.shape[0]):
            row = slice(idx, idx + 1)
            l1, l2 = self.mega.transform_weights_for_mega_moe(
                (unpack_ct_words(w1[row]), s1[row]),
                (unpack_ct_words(w2[row]), s2[row]), mode="w4a8")
            p1[row].copy_(l1[0])
            p2[row].copy_(l2[0])
            s1[row].copy_(l1[1])
        s1.div_(16)
        s2.div_(16)
        # Keep the registered parameters used by Core's native CT W4A8 apply.
        w1.data, w2.data = p1, p2
        quant = self.get_fused_moe_quant_config(layer)
        self.moe_quant_config = quant
        with _layer_buffer(ep, self.mega, w1.device, self.moe, self.buffer_capacity, "w4a8") as buf:
            experts = W4A8MegaExperts(self.moe, quant, buf, (p1, s1), (p2, s2), self.mega)
            self.moe_kernel = MegaKernel(MegaPrepare(), experts)
            self._processed = True
        if ep.rank_in_group == 0:
            init_logger(f"vllm.{__name__}").info_once(
                "HCU DSV4.1 W4A8 MegaMoE ready: provider=%s local_experts=%d capacity=%d",
                self.mega.__file__, w1.shape[0], buf.num_max_tokens_per_rank)
