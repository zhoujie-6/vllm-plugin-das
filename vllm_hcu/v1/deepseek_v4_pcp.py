# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Per-request 2N Full-KV PCP planning; decode tokens remain replicated."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import accumulate
from collections.abc import Sequence

import torch


@dataclass(frozen=True)
class Balanced2NCpPlan:
    cp_size: int
    cp_rank: int
    seq_len: int
    local_indices: torch.Tensor
    rerange_indices: torch.Tensor
    local_sizes: tuple[int, ...]
    local_query_lens: tuple[int, ...]
    local_query_start_loc: torch.Tensor


def build_2n_balanced_cp_plan_for_query_lens(
    query_lens: Sequence[int] | torch.Tensor,
    cp_size: int,
    cp_rank: int,
    *,
    device: torch.device | str | None = None,
) -> Balanced2NCpPlan:
    """Pair head/tail chunks per request and build the inverse gather order.

    Construct indices on the CPU from scheduler lengths, then transfer only
    the final tensors. No GPU synchronization is needed to check coverage.
    """
    if cp_size <= 0 or not 0 <= cp_rank < cp_size:
        raise ValueError(f"Invalid PCP size/rank: {cp_size}/{cp_rank}")
    if isinstance(query_lens, torch.Tensor):
        query_lens = query_lens.detach().cpu().tolist()
    lens = [int(length) for length in query_lens]
    if any(length < 0 for length in lens):
        raise ValueError(f"query_lens must be non-negative, got {lens}")
    rank_indices = [[] for _ in range(cp_size)]
    local_query_lens = []
    offset = 0
    for length in lens:
        block = (length + 2 * cp_size - 1) // (2 * cp_size)
        before = len(rank_indices[cp_rank])
        for rank, indices in enumerate(rank_indices):
            for block_id in (rank, 2 * cp_size - 1 - rank):
                start = min(block_id * block, length)
                end = min(start + block, length)
                indices.extend(range(offset + start, offset + end))
        local_query_lens.append(len(rank_indices[cp_rank]) - before)
        offset += length
    gathered = torch.tensor(
        [index for indices in rank_indices for index in indices], dtype=torch.long,
    )
    inverse = torch.empty(offset, dtype=torch.long)
    inverse[gathered] = torch.arange(offset)
    return Balanced2NCpPlan(
        cp_size=cp_size,
        cp_rank=cp_rank,
        seq_len=offset,
        local_indices=torch.tensor(rank_indices[cp_rank], dtype=torch.long, device=device),
        rerange_indices=inverse.to(device=device),
        local_sizes=tuple(map(len, rank_indices)),
        local_query_lens=tuple(local_query_lens),
        local_query_start_loc=torch.tensor(
            list(accumulate(local_query_lens, initial=0)), dtype=torch.int32, device=device,
        ),
    )


def get_local_query_chunk(
    plan: Balanced2NCpPlan, chunk_start: int, chunk_end: int, *, query_start: int,
) -> tuple[int, int, torch.Tensor]:
    """Map a request chunk to local Q slices and global combined-index rows."""
    local_start = sum(plan.local_query_lens[:chunk_start])
    local_end = local_start + sum(plan.local_query_lens[chunk_start:chunk_end])
    return local_start, local_end, plan.local_indices[local_start:local_end] - query_start


def select_tokens(tensor, plan, num_decode_tokens=0):
    if tensor is None or plan is None:
        return tensor
    indices = torch.cat((
        torch.arange(num_decode_tokens, device=tensor.device),
        plan.local_indices.to(tensor.device) + num_decode_tokens,
    ))
    return tensor.index_select(0, indices)


def gather_tokens(tensor, plan, num_decode_tokens=0, *, group=None):
    if tensor is None or plan is None:
        return tensor
    if group is None:
        from vllm.distributed import get_pcp_group
        group = get_pcp_group()
    if group.world_size != plan.cp_size or group.rank_in_group != plan.cp_rank:
        raise RuntimeError("DeepSeek-V4 PCP group does not match its token plan")
    local = tensor[num_decode_tokens:]
    if local.shape[0] != plan.local_sizes[plan.cp_rank]:
        raise ValueError("DeepSeek-V4 PCP gather received the wrong local token count")
    width = max(plan.local_sizes)
    if width == 0:
        return tensor[:num_decode_tokens].clone()
    padded = local.new_zeros((width, *local.shape[1:]))
    padded[:local.shape[0]].copy_(local)
    gathered = group.all_gather(padded, dim=0)
    packed = torch.cat([gathered[r * width:r * width + size]
                        for r, size in enumerate(plan.local_sizes)])
    restored = packed.index_select(0, plan.rerange_indices)
    return torch.cat((tensor[:num_decode_tokens], restored))


def get_forward_plan():
    from vllm.forward_context import get_forward_context
    metadata = get_forward_context().attn_metadata
    if isinstance(metadata, dict):
        for item in metadata.values():
            plan = getattr(item, "hcu_v4_pcp_plan", None)
            if plan is not None:
                return plan, item.num_decode_tokens
    return None, 0


def build_metadata_plan(config, num_decodes, num_prefills, query_start_loc_cpu, device):
    from vllm_hcu.deepseek_v4_runtime import is_deepseek_v4_pcp
    if not is_deepseek_v4_pcp(config) or not num_prefills:
        return None
    from vllm.distributed import get_pcp_group
    group = get_pcp_group()
    starts = query_start_loc_cpu[num_decodes:num_decodes + num_prefills + 1]
    return build_2n_balanced_cp_plan_for_query_lens(
        (starts[1:] - starts[:-1]).tolist(), group.world_size,
        group.rank_in_group, device=device,
    )
