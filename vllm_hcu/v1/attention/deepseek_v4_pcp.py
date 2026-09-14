# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""V4 Full-KV PCP attention, adapted to the v0.25.1 AMD attention ABI.

Chunk-local row selection follows vllm-hcu 47edf3a5. Cache encoding and
sparse attention kernels remain those of the target HCU backend.
"""
from __future__ import annotations
import torch
from vllm.platforms import current_platform
from vllm.v1.worker.workspace import current_workspace_manager
from vllm.models.deepseek_v4.amd.rocm import (
    dequantize_and_gather_k_cache, combine_topk_swa_indices,
    rocm_sparse_attn_prefill,
)
from vllm_hcu.v1.deepseek_v4_pcp import get_local_query_chunk

def forward_prefill(
    self,
    q: torch.Tensor,
    positions: torch.Tensor,
    compressed_k_cache: torch.Tensor | None,
    swa_k_cache: torch.Tensor,
    output: torch.Tensor,
    attn_metadata: DeepseekV4ROCMAiterMLASparseMetadata | None,
    swa_metadata: DeepseekV4ROCMAiterSparseSWAMetadata,
) -> None:
    plan = swa_metadata.hcu_v4_pcp_plan
    swa_only = attn_metadata is None

    num_prefills = swa_metadata.num_prefills
    num_prefill_tokens = swa_metadata.num_prefill_tokens
    num_decodes = swa_metadata.num_decodes
    num_decode_tokens = swa_metadata.num_decode_tokens

    seq_lens = swa_metadata.prefill_seq_lens
    gather_lens = swa_metadata.prefill_gather_lens
    assert seq_lens is not None
    assert gather_lens is not None

    query_start_loc_cpu = swa_metadata.query_start_loc_cpu
    query_start_loc = swa_metadata.query_start_loc
    assert query_start_loc_cpu is not None
    assert query_start_loc is not None
    prefill_token_base = query_start_loc_cpu[num_decodes]

    if not swa_only:
        if self.compress_ratio == 4:
            assert self.topk_indices_buffer is not None
            topk_indices = self.topk_indices_buffer[num_decode_tokens:]
            topk_indices = topk_indices[:num_prefill_tokens]
        else:
            assert attn_metadata is not None
            topk_indices = attn_metadata.c128a_prefill_topk_indices
        assert topk_indices is not None
        top_k = topk_indices.shape[-1]
        N = (self.max_model_len + self.compress_ratio - 1) // self.compress_ratio
    else:
        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[num_decode_tokens:]
        top_k = 0
        N = 0

    M = N + self.window_size + self.max_num_batched_tokens
    num_chunks = (num_prefills + self.PREFILL_CHUNK_SIZE - 1) // (
        self.PREFILL_CHUNK_SIZE
    )

    workspace_manager = current_workspace_manager()
    kv = workspace_manager.get_simultaneous(
        ((self.PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),
    )[0]
    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * self.PREFILL_CHUNK_SIZE
        chunk_end = min(chunk_start + self.PREFILL_CHUNK_SIZE, num_prefills)
        chunk_size = chunk_end - chunk_start
        if not swa_only:
            assert attn_metadata is not None
            assert compressed_k_cache is not None
            block_table = attn_metadata.block_table[num_decodes:]
            # compressed_k_cache is OCP on every platform (Triton encoder).
            dequantize_and_gather_k_cache(
                kv[:chunk_size],
                compressed_k_cache,
                seq_lens=seq_lens[chunk_start:chunk_end] // self.compress_ratio,
                gather_lens=None,
                block_table=block_table[chunk_start:chunk_end],
                block_size=attn_metadata.block_size // self.compress_ratio,
                offset=0,
                use_fnuz=False,
            )

        swa_block_table = swa_metadata.block_table[num_decodes:]
        dequantize_and_gather_k_cache(
            kv[:chunk_size],
            swa_k_cache,
            seq_lens=seq_lens[chunk_start:chunk_end],
            gather_lens=gather_lens[chunk_start:chunk_end],
            block_table=swa_block_table[chunk_start:chunk_end],
            block_size=swa_metadata.block_size,
            offset=N,
            use_fnuz=current_platform.is_fp8_fnuz(),
        )

        query_start = (
            query_start_loc_cpu[num_decodes + chunk_start] - prefill_token_base
        )
        query_end = (
            query_start_loc_cpu[num_decodes + chunk_end] - prefill_token_base
        )

        combined_indices, combined_lens = combine_topk_swa_indices(
            topk_indices[query_start:query_end],
            query_start_loc[
                num_decodes + chunk_start : num_decodes + chunk_end + 1
            ],
            seq_lens[chunk_start:chunk_end],
            gather_lens[chunk_start:chunk_end],
            self.window_size,
            self.compress_ratio,
            top_k,
            M,
            N,
        )
        local_start, local_end, rows = get_local_query_chunk(
            plan, chunk_start, chunk_end, query_start=int(query_start),
        )
        if local_start == local_end:
            continue
        combined_indices = combined_indices.index_select(0, rows)
        combined_lens = combined_lens.index_select(0, rows)
        rocm_sparse_attn_prefill(
            q=q[local_start:local_end],
            kv=kv.view(-1, 1, q.shape[-1]),
            indices=combined_indices,
            topk_length=combined_lens,
            scale=self.scale,
            head_dim=self.head_dim,
            nope_head_dim=self.nope_head_dim,
            rope_head_dim=self.rope_head_dim,
            attn_sink=self.attn_sink,
            output=output[local_start:local_end],
        )
