# SPDX-License-Identifier: Apache-2.0
"""Standalone DCU MegaMoE FP8 experts for the upstream AMD DSV4 model."""

from __future__ import annotations

import os
from types import SimpleNamespace

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.distributed import get_ep_group
from vllm.forward_context import get_forward_context
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op


class DeepseekV4MegaMoEFP8Experts(nn.Module):
    """Channelwise-FP8 expert storage consumed by dcu_mega_v3 MegaMoE."""

    _symm_buffer_cache: dict[tuple[object, ...], object] = {}

    def __init__(
        self,
        vllm_config: VllmConfig,
        *,
        num_experts: int,
        num_local_experts: int,
        experts_start_idx: int,
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        prefix: str = "",
    ):
        super().__init__()
        self.prefix = prefix
        self.num_experts = num_experts
        self.num_local_experts = num_local_experts
        self.experts_start_idx = experts_start_idx
        self.experts_end_idx = experts_start_idx + num_local_experts
        self.top_k = top_k
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        # dcu_mega_v3 derives per-expert routing scratch capacity from this
        # value, not only the input buffer length.  Keep SGLang's 8192-token
        # default headroom so a skewed 4096-token route cannot overrun an
        # expert tile even though the aggregate token count is in bounds.
        configured_capacity = int(
            os.getenv("VLLM_HCU_MEGAMOE_MAX_TOKENS_PER_RANK", "8192")
        )
        self.buffer_max_num_tokens = max(self.max_num_tokens, configured_capacity)
        attrs = {"weight_loader": self.weight_loader}

        self.w13_weight = nn.Parameter(
            torch.zeros(
                num_local_experts,
                2 * intermediate_size,
                hidden_size,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        set_weight_attrs(self.w13_weight, attrs)
        self.w13_weight_scale = nn.Parameter(
            torch.zeros(num_local_experts, 2 * intermediate_size, 1),
            requires_grad=False,
        )
        set_weight_attrs(self.w13_weight_scale, attrs)
        self.w13_weight_scale.quant_method = "channel"
        self.w2_weight = nn.Parameter(
            torch.zeros(
                num_local_experts,
                hidden_size,
                intermediate_size,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        set_weight_attrs(self.w2_weight, attrs)
        self.w2_weight_scale = nn.Parameter(
            torch.zeros(num_local_experts, hidden_size, 1), requires_grad=False
        )
        set_weight_attrs(self.w2_weight_scale, attrs)
        self.w2_weight_scale.quant_method = "channel"
        self._transformed_l1_weights: object | None = None
        self._transformed_l2_weights: object | None = None
        # The retained FusedMoE container is visited by vLLM's communication
        # preparation pass.  Mark this externally-owned kernel as already
        # modularized so that pass does not try to construct a vLLM MoE kernel.
        self.quant_method = SimpleNamespace(
            supports_internal_mk=True,
            is_monolithic=False,
        )

        context = vllm_config.compilation_config.static_forward_context
        if prefix in context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        context[prefix] = self

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: bool = False,
    ) -> bool | None:
        if not self.experts_start_idx <= expert_id < self.experts_end_idx:
            return False if return_success else None
        expert_data = param.data[expert_id - self.experts_start_idx]
        if shard_id in ("w1", "w3"):
            if "w13_" not in weight_name:
                return False if return_success else None
            offset = 0 if shard_id == "w1" else self.intermediate_size
            expert_data = expert_data.narrow(0, offset, self.intermediate_size)
        elif shard_id == "w2":
            if "w2_" not in weight_name:
                return False if return_success else None
        else:
            raise ValueError(f"Unsupported expert shard id: {shard_id}")
        if expert_data.shape != loaded_weight.shape:
            raise ValueError(
                f"MegaMoE {weight_name} shape mismatch: "
                f"{tuple(expert_data.shape)} != {tuple(loaded_weight.shape)}"
            )
        expert_data.copy_(loaded_weight)
        return True if return_success else None

    def _check_runtime_supported(self) -> None:
        if not torch.cuda.is_available() or self.w13_weight.device.type != "cuda":
            raise NotImplementedError("DCU MegaMoE FP8 weights must be on CUDA")
        if not current_platform.supports_fp8():
            raise NotImplementedError("DCU MegaMoE requires FP8, not FP4 capability")
        if self.hidden_size % 128 or self.intermediate_size % 128:
            raise ValueError("MegaMoE hidden sizes must be multiples of 128")

    def finalize_weights(self) -> None:
        if self._transformed_l1_weights is not None:
            return
        self._check_runtime_supported()
        import megamoe

        self._transformed_l1_weights = {
            "unified": (
                megamoe.flatten_pack5_weight(self.w13_weight.data.contiguous()),
                self.w13_weight_scale.data.squeeze(-1).float().contiguous(),
            )
        }
        self._transformed_l2_weights = {
            "unified": (
                megamoe.flatten_pack5_weight(self.w2_weight.data.contiguous()),
                self.w2_weight_scale.data.squeeze(-1).float().contiguous(),
            )
        }
        self.w13_weight = None
        self.w13_weight_scale = None
        self.w2_weight = None
        self.w2_weight_scale = None

    def get_symm_buffer(self):
        # The standalone runtime consumes this setting while constructing its
        # symmetric buffer. Scope the safe default to the explicitly selected
        # MegaMoE path so importing this module cannot affect other backends.
        os.environ.setdefault("K3_USE_ASM_TAIL_REDUCE", "0")
        import megamoe

        group = get_ep_group().device_group
        key = (
            id(group),
            torch.accelerator.current_device_index(),
            self.num_experts,
            self.buffer_max_num_tokens,
            self.top_k,
            self.hidden_size,
            self.intermediate_size,
        )
        buffer = self._symm_buffer_cache.get(key)
        if buffer is None:
            buffer = megamoe.get_symm_buffer_for_mega_moe(
                group,
                self.num_experts,
                self.buffer_max_num_tokens,
                self.top_k,
                self.hidden_size,
                self.intermediate_size,
                use_fp8_dispatch=True,
                activation="swiglu",
            )
            self._symm_buffer_cache[key] = buffer
        return buffer

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        *,
        activation_clamp: float | None,
        fast_math: bool = True,
    ) -> torch.Tensor:
        if hidden_states.shape[0] > self.max_num_tokens:
            raise ValueError("MegaMoE token count exceeds its symmetric buffer")
        output = torch.empty_like(hidden_states, dtype=torch.bfloat16)
        torch.ops.vllm.deepseek_v4_megamoe_fp8_experts(
            hidden_states,
            topk_weights,
            topk_ids,
            output,
            self.prefix,
            activation_clamp,
            fast_math,
        )
        return output

    def _run(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        output: torch.Tensor,
        activation_clamp: float | None,
        fast_math: bool,
    ) -> None:
        import megamoe

        buffer = self.get_symm_buffer()
        num_tokens = hidden_states.shape[0]
        # Keep input preparation identical to SGLang's standalone W8A8 path.
        # The fused helper shipped by some dcu_mega_v3 wheels writes full
        # buffer views and is unsafe when eager decode reuses a 4096-token
        # symmetric buffer for a much smaller active batch.
        x_fp8, x_scale = megamoe.cast_to_fp8_channelwise(
            hidden_states if hidden_states.is_contiguous() else hidden_states.contiguous()
        )
        buffer.x[:num_tokens].copy_(x_fp8)
        buffer.x_sf[:num_tokens].copy_(x_scale)
        buffer.topk_idx[:num_tokens].copy_(topk_ids.to(buffer.topk_idx.dtype))
        buffer.topk_weights[:num_tokens].copy_(
            topk_weights.to(buffer.topk_weights.dtype)
        )
        self.finalize_weights()
        threshold = int(os.getenv("VLLM_HCU_MEGAMOE_LL_TOKEN_THRESHOLD", "496"))
        graph = torch.cuda.is_current_stream_capturing()
        graph_kwargs = {}
        if graph:
            # vLLM captures a fixed, padded token bucket on every EP rank.
            # Record the count update in the graph: Python is not run during
            # replay, and other captured buckets share this symmetric buffer.
            buffer.cuda_graph_num_tokens.fill_(num_tokens)
            # Keep the large routing scratch allocation, but specialize graph
            # launches to the current output shape, not its 8192-token capacity.
            graph_capacity = buffer.cuda_graph_max_tokens_per_rank
            buffer.cuda_graph_max_tokens_per_rank = num_tokens
            graph_kwargs["graph"] = True
        try:
            megamoe.fp8_w8a8_mega_moe(
                output,
                self._transformed_l1_weights,
                self._transformed_l2_weights,
                buffer,
                recipe=(1, 1, 32),
                activation="swiglu",
                megamoe_backend="ll" if num_tokens <= threshold else "normal",
                capacity_num_tokens=num_tokens,
                activation_clamp=activation_clamp,
                fast_math=fast_math,
                **graph_kwargs,
            )
        finally:
            if graph:
                buffer.cuda_graph_max_tokens_per_rank = graph_capacity



DeepseekV4MegaMoEFP8Experts.weight_loader.supports_moe_loading = True  # type: ignore[attr-defined]


def _op(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    clamp: float | None,
    fast_math: bool,
) -> None:
    layer = get_forward_context().no_compile_layers[layer_name]
    layer._run(hidden_states, topk_weights, topk_ids, output, clamp, fast_math)


def _fake(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    clamp: float | None,
    fast_math: bool,
) -> None:
    return None


direct_register_custom_op(
    op_name="deepseek_v4_megamoe_fp8_experts",
    op_func=_op,
    mutates_args=["output"],
    fake_impl=_fake,
)


__all__ = ["DeepseekV4MegaMoEFP8Experts"]
