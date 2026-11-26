# SPDX-License-Identifier: Apache-2.0
"""Packed CT INT4 weights and token-INT8 activations for DSV4.1 MegaMoE."""
from functools import cache, wraps

import torch

from vllm.distributed import get_ep_group
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors_moe.compressed_tensors_moe_w4a8_int8 import (
    CompressedTensorsW4A8Int8MoEMethod,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kInt4Static, kInt8DynamicTokenSym
from vllm.model_executor.utils import set_weight_attrs

from ..dsv41.profile_worker import local_token_bound
from ..quantization.ct_w4a8_weights import PackedCTParameters, convert_ct_int4_to_hipc_bytes_
from .dsv41_megamoe import (
    DSV41MegaMoEMethodMixin, MegaExperts, MegaKernel, MegaPrepare, _layer_buffer,
)


def w4a8_provider():
    import megamoe
    from megamoe import w4a8

    if not callable(getattr(w4a8, "w4a8_mega_moe", None)):
        raise RuntimeError("The standalone MegaMoE wheel must support W4A8")
    return megamoe


def pack_ct_hipc(words):
    from deepgemm import pack_w4a8_moe_hipc_weight

    return pack_w4a8_moe_hipc_weight(convert_ct_int4_to_hipc_bytes_(words))


@cache
def _fused_stage():
    # Older standalone wheels only expose the separate LightOp quantizer.
    try:
        from megamoe._w4a8_staging import stage_inputs
    except ModuleNotFoundError as error:
        if error.name != "megamoe._w4a8_staging":
            raise
        return None
    return stage_inputs


def stage_ct_w4a8(buf, x, ids, weights):
    if x.is_contiguous() and ids.is_contiguous() and weights.is_contiguous():
        stage = _fused_stage()
        if stage is not None:
            stage(x, ids, weights, buf)
            return

    from lightop.quant import per_token_quant_int8

    rows = x.shape[0]
    quantized, scales = per_token_quant_int8(x.contiguous())
    buf.x[:rows].copy_(quantized)
    buf.x_sf[:rows].copy_(scales.reshape(-1))
    buf.topk_idx[:rows].copy_(ids)
    buf.topk_weights[:rows].copy_(weights)


class _W4A8MegaTile(MegaExperts):
    @staticmethod
    def _supports_quant_scheme(weight_key, activation_key):
        return (weight_key, activation_key) == (kInt4Static, kInt8DynamicTokenSym)

    def stage(self, x, ids, weights):
        stage_ct_w4a8(self.buffer, x, ids, weights)

    def launch(self, output):
        self.mega.w4a8.w4a8_mega_moe(
            output, self.l1, self.l2, self.buffer,
            activation_clamp=self.quant_config.gemm1_clamp_limit,
            combine_group_map=self.combine_groups,
        )


class W4A8MegaExperts(_W4A8MegaTile):
    short_tile = None

    def run(self, output, x, ids, weights):
        # Core pads PCP inputs to the largest rank. Profile/replay may carry a
        # full scheduler batch, while ordinary prefills fit in one local tile.
        # Use the common descriptor even on ranks with fewer (or zero) rows.
        extent = x.shape[0]
        desc = None
        if is_forward_context_available():
            desc = get_forward_context().batch_descriptor
            if desc is not None:
                extent = desc.num_tokens
        if extent < x.shape[0]:
            raise ValueError("PCP batch descriptor is smaller than the W4A8 input")
        if (desc is not None and self.short_tile is not None
                and extent <= self.short_tile.buffer.num_max_tokens_per_rank):
            return self.short_tile.run(output, x, ids, weights)
        cap = self.buffer.num_max_tokens_per_rank
        if extent <= cap:
            return super().run(output, x, ids, weights)
        if desc is None:
            raise ValueError("Oversize PCP batches require a common batch descriptor")
        for start in range(0, max(1, extent), cap):
            tile = slice(start, start + cap)
            super().run(output[tile], x[tile], ids[tile], weights[tile])


class _MegaMoEPackedParameters(PackedCTParameters):
    """Track successful native loads into the local physical expert slots."""

    _projections = {"w1": "gate", "w3": "up", "w2": "down"}

    def __init__(self, layer, num_experts, hidden_size, intermediate_size, weight_loader):
        super().__init__(layer, num_experts, hidden_size, intermediate_size, weight_loader)
        fields = ("weight_packed", "weight_scale")
        self._missing = {
            (slot, shard, field)
            for slot in range(num_experts) for shard in self._projections for field in fields
        }
        for stem in ("w13", "w2"):
            for field in fields:
                param = getattr(layer, f"{stem}_{field}")
                param.weight_loader = self._track_load(param.weight_loader, field)

    def _track_load(self, loader, field):
        # Static EPLB unwraps the native owner for validation, then calls this
        # wrapper for each physical replica. Preserve the bound loader signature.
        @wraps(loader)
        def load(param, loaded_weight, weight_name, shard_id, expert_id, return_success=False):
            success = loader(param, loaded_weight, weight_name, shard_id, expert_id,
                             return_success=True)
            if success:
                if loaded_weight.ndim == 3:
                    # RoutedExperts writes the full leading expert axis in this case.
                    if loaded_weight.shape[0] != self.shape[0]:
                        raise ValueError("Expected a full batch of local W4A8 expert shards")
                    for slot in range(self.shape[0]):
                        self._missing.discard((slot, shard_id, field))
                else:
                    slot = self.layer.expert_map_manager.map_global_to_local(expert_id)
                    self._missing.discard((slot, shard_id, field))
            return success if return_success else None

        return load

    def require_complete(self, prefix):
        if self._missing:
            slot, shard, field = min(self._missing)
            raise RuntimeError(
                f"Incomplete W4A8 checkpoint for {prefix}: local expert {slot}, "
                f"projection {self._projections[shard]}, missing {field} "
                f"({len(self._missing)} missing shards in this layer)"
            )


class HcuDSV41W4A8MegaMoEMethod(DSV41MegaMoEMethodMixin, CompressedTensorsW4A8Int8MoEMethod):
    supports_static_eplb = True
    supports_swa_bounded_replay = True
    supports_dspark = True

    @staticmethod
    def _buffer_capacity(config, moe):
        sched = config.scheduler_config
        return min(moe.max_num_tokens, local_token_bound(
            sched.max_num_batched_tokens, sched.max_num_seqs, moe.pcp_size))

    def __init__(self, weight_quant, input_quant, moe, prefix):
        super().__init__(weight_quant, input_quant, moe, prefix)
        self.experts_cls = W4A8MegaExperts
        self.mega = w4a8_provider()

    def create_weights(self, layer, num_experts, hidden_size,
                       intermediate_size_per_partition, params_dtype, **extra_weight_attrs):
        self._packed = _MegaMoEPackedParameters(
            layer, num_experts, hidden_size, intermediate_size_per_partition,
            extra_weight_attrs.get("weight_loader"))
        extra = {k: v for k, v in extra_weight_attrs.items()
                 if k not in ("weight_loader", "is_transposed", "quant_method")}
        for param in layer.parameters(recurse=False):
            set_weight_attrs(param, extra)

    def get_fused_moe_quant_config(self, layer):
        return FusedMoEQuantConfig.make(
            torch.int8, weight_dtype="int4", per_act_token_quant=True, per_out_ch_quant=True,
            w1_scale=layer.w13_weight_scale, w2_scale=layer.w2_weight_scale,
            gemm1_clamp_limit=layer.swiglu_limit)


    @torch.inference_mode()
    def _process_weights(self, layer):
        if self._processed:
            raise RuntimeError("MegaMoE packed weights cannot be postprocessed twice")
        if self._packed.state != "loaded":
            raise RuntimeError(
                f"Cannot pack W4A8 MegaMoE weights: expected loaded state, got {self._packed.state!r}"
            )
        self._packed.require_complete(self._prefix)
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
        self._packed.state = "packing"
        w1, w2 = layer.w13_weight_packed, layer.w2_weight_packed
        # Repack each local expert in checkpoint storage.
        p1, p2 = w1.data.view(torch.int8), w2.data.view(torch.int8)
        s1 = layer.w13_weight_scale.data.squeeze(-1)
        s2 = layer.w2_weight_scale.data.squeeze(-1)
        for idx in range(w1.shape[0]):
            row = slice(idx, idx + 1)
            p1[row].copy_(pack_ct_hipc(w1[row]))
            p2[row].copy_(pack_ct_hipc(w2[row]))
        # The standalone wheel accepts natural gate/up order and true per-channel scales.
        # Keep the registered parameters used by Core's native CT W4A8 apply.
        w1.data, w2.data = p1, p2
        quant = self.get_fused_moe_quant_config(layer)
        self.moe_quant_config = quant
        with _layer_buffer(ep, self.mega, w1.device, self.moe, self.buffer_capacity, "w4a8") as buf:
            experts = W4A8MegaExperts(self.moe, quant, buf, (p1, s1), (p2, s2), self.mega)
            placement = getattr(layer, "_hcu_static_eplb_row", None)
            experts.combine_groups = None
            if placement is not None:
                logical = self.moe.num_logical_experts
                if logical % ep.world_size or len(placement) != self.moe.num_experts:
                    raise ValueError("Static expert placement is incompatible with EP groups")
                owners = [index // (logical // ep.world_size) for index in placement]
                if any(owner != index // p1.shape[0] for index, owner in enumerate(owners)):
                    experts.combine_groups = torch.tensor(owners, dtype=torch.int32, device=p1.device)
            # The provider sizes its route scans and GEMM workspaces from
            # buffer capacity. Use a shared short tile for bounded replay.
            # Large batches and ranks without a common descriptor retain the
            # original buffer and tiling. Both buffers enter memory profiling.
            if self.buffer_capacity > 256:
                with _layer_buffer(ep, self.mega, w1.device, self.moe, 256, "w4a8") as short:
                    tile = _W4A8MegaTile(self.moe, quant, short, (p1, s1), (p2, s2), self.mega)
                    tile.combine_groups = experts.combine_groups
                    if not getattr(short, "_hcu_workspace_ready", False):
                        tile.launch(torch.empty((0, h), dtype=torch.bfloat16, device=w1.device))
                        short._hcu_workspace_ready = True
                    experts.short_tile = tile
            self.moe_kernel = MegaKernel(MegaPrepare(), experts)
            self._packed.state = "ready"
            self._processed = True
        if ep.rank_in_group == 0:
            init_logger(f"vllm.{__name__}").info_once(
                "HCU DSV4.1 W4A8 MegaMoE ready: provider=%s local_experts=%d capacity=%d",
                self.mega.__file__, w1.shape[0], buf.num_max_tokens_per_rank)
