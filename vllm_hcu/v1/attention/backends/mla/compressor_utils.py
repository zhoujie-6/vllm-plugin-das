# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""DCP aware compressed slot mapping for DeepSeek-V4."""

import torch

from vllm.triton_utils import tl, triton
from vllm_hcu.v1.attention.ops.deepseek_v4_ops.dcp import cp_global_to_local_block


@triton.jit
def _compressed_slot_mapping_dcp_kernel(
    compressed_slot_mapping_ptr,
    slot_mapping_ptr,
    query_start_loc_ptr,
    seq_lens_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    COMPRESS_RATIO: tl.constexpr,
    DCP_WORLD_SIZE: tl.constexpr,
    DCP_RANK: tl.constexpr,
    CP_KV_CACHE_INTERLEAVE_SIZE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    req = tl.program_id(0)
    query_start = tl.load(query_start_loc_ptr + req)
    query_end = tl.load(query_start_loc_ptr + req + 1)
    query_len = query_end - query_start
    start_pos = tl.load(seq_lens_ptr + req) - query_len
    for i in range(0, query_len, BLOCK):
        offset = i + tl.arange(0, BLOCK)
        mask = offset < query_len
        pos = start_pos + offset
        # The source mapping belongs to the uncompressed DCP layout. Its
        # -1 means this rank does not own that raw token, not that the query
        # is padding. A compression boundary can belong to a different rank
        # after compression. query_start/end already exclude padded tokens.
        valid = mask & ((pos + 1) % COMPRESS_RATIO == 0)
        compressed_pos = pos // COMPRESS_RATIO
        block_idx, block_offset, owned = cp_global_to_local_block(
            compressed_pos,
            block_size,
            DCP_WORLD_SIZE,
            DCP_RANK,
            CP_KV_CACHE_INTERLEAVE_SIZE,
        )
        valid &= owned
        block_number = tl.load(
            block_table_ptr + req * block_table_stride + block_idx,
            mask=valid,
            other=0,
        )
        slot = block_number * block_size + block_offset
        tl.store(
            compressed_slot_mapping_ptr + query_start + offset,
            tl.where(valid, slot, -1),
            mask=mask,
        )


def get_compressed_slot_mapping_dcp(
    num_tokens: int,
    slot_mapping: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    compress_ratio: int,
    out: torch.Tensor | None = None,
    *,
    dcp_world_size: int,
    dcp_rank: int,
    cp_kv_cache_interleave_size: int,
) -> torch.Tensor:
    if out is None:
        out = torch.full(
            (num_tokens,), -1, dtype=torch.int64, device=query_start_loc.device
        )
    else:
        out.fill_(-1)
    result = out[:num_tokens]
    _compressed_slot_mapping_dcp_kernel[(block_table.shape[0],)](
        result,
        slot_mapping,
        query_start_loc,
        seq_lens,
        block_table,
        block_table.stride(0),
        block_size,
        COMPRESS_RATIO=compress_ratio,
        DCP_WORLD_SIZE=dcp_world_size,
        DCP_RANK=dcp_rank,
        CP_KV_CACHE_INTERLEAVE_SIZE=cp_kv_cache_interleave_size,
        BLOCK=1024,
    )
    return result
