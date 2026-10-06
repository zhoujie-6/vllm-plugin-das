# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Context-parallel address translation used by DeepSeek-V4 caches."""

from dataclasses import dataclass

import torch

from vllm.config import VllmConfig
from vllm.distributed import get_dcp_group
from vllm.triton_utils import tl, triton


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

    def owns(self, global_indices: torch.Tensor) -> torch.Tensor:
        safe = global_indices.clamp_min(0)
        return (global_indices >= 0) & (
            (safe // self.interleave_size) % self.world_size == self.rank
        )

    def global_to_local(self, global_indices: torch.Tensor) -> torch.Tensor:
        safe = global_indices.clamp_min(0)
        stride = self.world_size * self.interleave_size
        base = safe // stride * self.interleave_size
        remainder = safe - base * self.world_size
        extra = torch.clamp(
            remainder - self.rank * self.interleave_size,
            0,
            self.interleave_size,
        )
        return torch.where(global_indices >= 0, base + extra, -1)

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
    block_idx = pos // block_size
    block_offset = pos % block_size
    global_block_offset = block_idx * block_size + block_offset
    owner = (
        global_block_offset // CP_KV_CACHE_INTERLEAVE_SIZE
    ) % DCP_WORLD_SIZE
    local_block_offset = (
        global_block_offset // (DCP_WORLD_SIZE * CP_KV_CACHE_INTERLEAVE_SIZE)
    ) * CP_KV_CACHE_INTERLEAVE_SIZE + (
        global_block_offset % CP_KV_CACHE_INTERLEAVE_SIZE
    )
    return local_block_offset // block_size, local_block_offset % block_size, owner == DCP_RANK
