# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Shared DeepSeek-V4 DCP layout, reductions and numerical diagnostics."""

import functools
import math
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from vllm.config import VllmConfig
from vllm.distributed import get_dcp_group
from vllm.triton_utils import tl, triton

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator


def dcp_softmax_reduce(
    local_max: torch.Tensor,
    local_sum: torch.Tensor,
    local_weighted_value: torch.Tensor,
    group: "GroupCoordinator",
) -> torch.Tensor:
    """Merge per-rank online-softmax statistics without losing stability."""
    valid = local_sum > 0
    local_max = torch.where(valid, local_max, torch.full_like(local_max, -torch.inf))
    gathered = group.all_gather(local_max, dim=0).reshape(
        (group.world_size,) + local_max.shape
    )
    global_max = gathered.max(dim=0).values
    scale = torch.where(valid, torch.exp(local_max - global_max), 0.0)
    payload = torch.stack((local_sum * scale, local_weighted_value * scale))
    global_sum, global_value = group.all_reduce(payload).unbind(0)
    return torch.where(global_sum > 0, global_value / global_sum, 0.0)


def check_dcp_tensor(owner, stage, tensor, allow_negative_inf=False):
    if os.environ.get("VLLM_HCU_DEEPSEEK_V4_DCP_DEBUG") != "1":
        return
    if tensor.is_cuda and torch.cuda.is_current_stream_capturing():
        return
    valid = torch.isfinite(tensor)
    if allow_negative_inf:
        valid = valid | torch.isneginf(tensor)
    if not valid.all().item():
        prefix = getattr(owner, "prefix", type(owner).__name__)
        group = getattr(owner, "dcp_group", None)
        rank = getattr(group, "rank_in_group", -1)
        raise RuntimeError(
            f"DeepSeek-V4 DCP nonfinite: layer={prefix} rank={rank} "
            f"stage={stage} shape={tuple(tensor.shape)} "
            f"nan={torch.isnan(tensor).sum().item()} "
            f"posinf={torch.isposinf(tensor).sum().item()} "
            f"neginf={torch.isneginf(tensor).sum().item()}"
        )


def lse_is_natural(lse):
    value = lse.float()
    if torch.allclose(value, torch.full_like(value, math.log(4)), atol=1e-3, rtol=1e-3):
        return True
    if torch.allclose(value, torch.full_like(value, 2.0), atol=1e-3, rtol=1e-3):
        return False
    raise RuntimeError(f"FlashMLA LSE calibration failed: expected ln(4) or log2(4), got {value.tolist()}")


@functools.cache
def calibrate_flashmla_lse(device_index):
    from vllm_hcu.v1.attention.ops.flashmla import (
        flash_mla_sparse_fwd, flash_mla_with_kvcache, get_mla_metadata,
    )

    device = torch.device("cuda", device_index)
    indices = torch.full((1, 1, 128), -1, dtype=torch.int32, device=device)
    indices[0, 0, :4] = torch.arange(4, device=device, dtype=torch.int32)
    lengths = torch.full((1,), 4, dtype=torch.int32, device=device)
    _, _, prefill_lse = flash_mla_sparse_fwd(
        q=torch.zeros(1, 64, 512, dtype=torch.bfloat16, device=device),
        kv=torch.zeros(128, 1, 512, dtype=torch.bfloat16, device=device),
        indices=indices, sm_scale=0.125, attn_sink=None, topk_length=lengths,
    )
    # UE8M0 scales occupy the tail of the page, after 64 * 576 data bytes.
    cache = torch.zeros(1, 64, 1, 584, dtype=torch.uint8, device=device)
    cache.view(-1)[64 * 576:] = 127
    _, decode_lse = flash_mla_with_kvcache(
        q=torch.zeros(1, 1, 16, 512, dtype=torch.bfloat16, device=device),
        k_cache=cache, block_table=None, head_dim_v=512,
        tile_scheduler_metadata=get_mla_metadata()[0], cache_seqlens=None,
        is_fp8_kvcache=True, indices=indices, topk_length=lengths,
        softmax_scale=0.125, attn_sink=None,
    )
    result = (lse_is_natural(prefill_lse), lse_is_natural(decode_lse))
    print(f"[DSV4_DCP_LSE] device={device_index} prefill_base={'e' if result[0] else '2'} "
          f"decode_base={'e' if result[1] else '2'}", flush=True)
    return result


@dataclass(frozen=True, slots=True)
class ContextParallelLayout:
    world_size: int = 1
    rank: int = 0
    interleave_size: int = 1

    def __post_init__(self) -> None:
        if self.world_size < 1:
            raise ValueError("world_size must be positive")
        if not 0 <= self.rank < self.world_size:
            raise ValueError("rank must be in [0, world_size)")
        if self.interleave_size < 1:
            raise ValueError("interleave_size must be positive")

    @property
    def enabled(self) -> bool:
        return self.world_size > 1

    @classmethod
    def from_config(cls, config: VllmConfig) -> "ContextParallelLayout":
        parallel = config.parallel_config
        world_size = parallel.decode_context_parallel_size
        rank = get_dcp_group().rank_in_group if world_size > 1 else 0
        return cls(world_size, rank, parallel.cp_kv_cache_interleave_size)

    def triton_kwargs(self) -> dict[str, int]:
        return {
            "DCP_WORLD_SIZE": self.world_size,
            "DCP_RANK": self.rank,
            "CP_KV_CACHE_INTERLEAVE_SIZE": self.interleave_size,
        }


@triton.jit
def cp_global_to_local_block(
    pos,
    block_size,
    DCP_WORLD_SIZE: tl.constexpr,
    DCP_RANK: tl.constexpr,
    CP_KV_CACHE_INTERLEAVE_SIZE: tl.constexpr,
):
    owner = (
        pos // CP_KV_CACHE_INTERLEAVE_SIZE
    ) % DCP_WORLD_SIZE
    local_block_offset = (
        pos // (DCP_WORLD_SIZE * CP_KV_CACHE_INTERLEAVE_SIZE)
    ) * CP_KV_CACHE_INTERLEAVE_SIZE + (
        pos % CP_KV_CACHE_INTERLEAVE_SIZE
    )
    return local_block_offset // block_size, local_block_offset % block_size, owner == DCP_RANK
