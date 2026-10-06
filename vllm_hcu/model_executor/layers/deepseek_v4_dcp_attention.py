# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""DCP sparse attention forwards shared with the native ROCm DeepSeek-V4 class."""

from __future__ import annotations

import torch
import math

from vllm.v1.attention.backends.utils import get_dcp_local_seq_lens
from vllm.v1.attention.ops.common import cp_lse_ag_out_rs
from vllm.v1.attention.ops.dcp_alltoall import dcp_a2a_lse_reduce
from vllm.v1.worker.workspace import current_workspace_manager
from vllm_hcu.v1.attention.ops.deepseek_v4_ops.cache_utils import (
    combine_topk_swa_indices,
    compute_global_topk_indices_and_lens,
    dequantize_and_gather_k_cache,
)
from vllm_hcu.v1.attention.ops.flashmla import (
    flash_mla_sparse_fwd,
    flash_mla_with_kvcache,
)
from vllm_hcu.v1.attention.ops.deepseek_v4_ops.dcp import check_dcp_tensor

PREFILL_CHUNK_SIZE = 4


def _mask_empty_attention(out, lse, has_keys):
    """Empty shards contribute zero numerator and negative-infinity LSE."""
    return (
        torch.where(has_keys[:, None, None], out, 0),
        torch.where(has_keys[:, None], lse, -float("inf")),
    )


def _localize_c128a_prefill_indices(indices, world_size, rank):
    owned = (indices >= 0) & (indices.remainder(world_size) == rank)
    return torch.where(owned, indices // world_size, -1)


def _localize_c4_indices(indices, world_size, rank):
    local = _localize_c128a_prefill_indices(indices, world_size, rank)
    # Decode's topk_length counts valid entries, so owned indices must form
    # a prefix even when global score order alternates between DCP owners.
    order = torch.argsort((local < 0).to(torch.int32), dim=-1, stable=True)
    return torch.gather(local, -1, order)


def _prepare_dcp_attention(owner, out, lse, has_keys, stage):
    """Release padded kernel outputs before allocating collective buffers."""
    if lse is None:
        raise RuntimeError(f"DeepSeek-V4 DCP {stage} kernel did not return LSE")
    # Decode returns [tokens, heads, 1]; prefill returns [tokens, heads].
    if lse.dim() == 3:
        lse = lse.squeeze(-1)
    lse = lse[:, :out.shape[1]]
    if not getattr(owner, f"dcp_{stage}_lse_base_e", True):
        lse = lse * math.log(2)
    return _mask_empty_attention(out, lse, has_keys)


def _reduce_dcp_attention(owner, out, lse, stage):
    """Merge sparse shards and include the replicated sink exactly once."""
    check_dcp_tensor(owner, f"{stage}.local_output", out)
    check_dcp_tensor(owner, f"{stage}.local_lse", lse, True)
    combine = dcp_a2a_lse_reduce if owner.dcp_a2a else cp_lse_ag_out_rs
    out, lse = combine(
        out, lse, owner.dcp_group, return_lse=True, is_lse_base_on_e=True,
    )
    check_dcp_tensor(owner, f"{stage}.merged_output", out)
    check_dcp_tensor(owner, f"{stage}.merged_lse", lse, True)
    if owner.attn_sink is not None:
        sink = owner.attn_sink.reshape(-1)[:out.shape[1]].to(lse.dtype)
        sink_lse = torch.logaddexp(lse, sink.unsqueeze(0))
        weight = torch.exp(lse - sink_lse).to(out.dtype).unsqueeze(-1)
        # Preserve prefill's in-place update: long contexts already keep
        # the gathered queries and all-to-all buffers live here.
        if stage == "prefill":
            out.mul_(weight)
        else:
            out = out * weight
    return out


class DeepseekV4DCPAttention:
    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,  # Only used when compress_ratio > 1
        swa_metadata: "DeepseekSparseSWAMetadata",
        attn_metadata: FlashMLASparseMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        assert self.dcp_group is not None
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens
        check_dcp_tensor(self, "decode.query", q)

        topk_indices = None
        topk_lens = None
        if not swa_only:
            assert attn_metadata is not None
            assert swa_metadata.is_valid_token is not None
            block_size = attn_metadata.block_size // self.compress_ratio
            is_valid = swa_metadata.is_valid_token[:num_decode_tokens]
            if self.compress_ratio == 4:
                # C4A: local indices differ per layer (filled by Indexer).
                assert self.topk_indices_buffer is not None
                local_indices = _localize_c4_indices(
                    self.topk_indices_buffer[:num_decode_tokens],
                    self.dcp_world_size, self.dcp_group.rank_in_group,
                )
                global_indices, topk_lens = compute_global_topk_indices_and_lens(
                    local_indices,
                    swa_metadata.token_to_req_indices,
                    attn_metadata.block_table[:num_decodes],
                    block_size,
                    is_valid,
                )
                topk_indices = global_indices.view(num_decode_tokens, 1, -1)
            else:
                # C128A: pre-computed during metadata build.
                topk_indices = attn_metadata.c128a_global_decode_topk_indices
                topk_lens = attn_metadata.c128a_decode_topk_lens
                local_lens = get_dcp_local_seq_lens(
                    topk_lens, self.dcp_world_size,
                    self.dcp_group.rank_in_group, 1,
                )
                offsets = torch.arange(
                    topk_indices.shape[-1], device=q.device, dtype=torch.int32
                )
                local_indices = torch.where(
                    offsets[None, :] < local_lens[:, None],
                    offsets[None, :], -1,
                )
                global_indices, topk_lens = compute_global_topk_indices_and_lens(
                    local_indices, swa_metadata.token_to_req_indices,
                    attn_metadata.block_table[:num_decodes], block_size, is_valid,
                )
                topk_indices = global_indices[:, None, :]

        swa_indices = swa_metadata.decode_swa_indices
        swa_lens = swa_metadata.decode_swa_lens

        # SWA is deliberately replicated rather than DCP sharded. It must
        # nevertheless participate in the distributed softmax exactly once;
        # otherwise every DCP rank contributes the same SWA keys and changes
        # their weight relative to the sharded compressed cache and sink.
        # Rank 0 owns the replicated term for the reduction.
        if self.dcp_group.rank_in_group != 0:
            swa_lens = torch.zeros_like(swa_lens)

        # We treat queries in the same seq as different queries
        # and later we only attend by generated indices.
        # q arrives pre-padded to self.padded_heads by the outer wrapper.
        q = self.dcp_group.all_gather(q, dim=1)
        actual_heads = q.shape[1]
        if 16 < actual_heads < 64:
            q = torch.nn.functional.pad(q, (0, 0, 0, 64 - actual_heads))
        q = q.unsqueeze(1)

        # Prepare SWA cache (num_blocks, swa_block_size, 1, head_bytes)
        # Use unsqueeze to preserve strides (handles padded blocks correctly)
        swa_cache = self.swa_cache_layer.kv_cache.unsqueeze(-2)
        # Reshape KV cache to (num_blocks, block_size, 1, head_bytes)
        if kv_cache is not None:
            kv_cache = kv_cache.unsqueeze(-2)

        # One FlashMLASchedMeta per layer type, shared across all same-type
        # layers within this decode step. The first forward call per type
        # triggers the in-kernel planner (allocating tile_scheduler_metadata
        # and num_splits via PyTorch's graph-aware allocator so CUDA graph
        # capture reuses the same addresses on replay); subsequent same-type
        # layers see have_initialized=True and skip the planner.
        if self.compress_ratio <= 1:
            tile_metadata = swa_metadata.tile_sched_swaonly
        elif self.compress_ratio == 4:
            tile_metadata = swa_metadata.tile_sched_c4a
        elif self.compress_ratio == 128:
            tile_metadata = swa_metadata.tile_sched_c128a
        else:
            raise ValueError(
                f"Unsupported compress_ratio={self.compress_ratio}; "
                "expected 1, 4, or 128."
            )
        assert tile_metadata is not None, (
            "swa_metadata missing tile_sched entry for "
            f"compress_ratio={self.compress_ratio}; "
            "DeepseekSparseSWAMetadataBuilder.build_tile_scheduler did not "
            "allocate one for this layer type."
        )

        out, lse = flash_mla_with_kvcache(
            q=q,
            k_cache=swa_cache,
            block_table=None,
            head_dim_v=512,
            tile_scheduler_metadata=tile_metadata,
            cache_seqlens=None,
            is_fp8_kvcache=True,
            indices=swa_indices,
            topk_length=swa_lens,
            softmax_scale=self.scale,
            # The sink is a global softmax participant and must be applied
            # once, after rank-local LSE/output reduction.
            attn_sink=None,
            extra_k_cache=kv_cache if not swa_only else None,
            extra_indices_in_kvcache=topk_indices,
            extra_topk_length=topk_lens,
            # out=output.unsqueeze(1),
        )
        out = out.squeeze(1)[:, :actual_heads]
        has_keys = swa_lens > 0
        if topk_lens is not None:
            has_keys = has_keys | (topk_lens > 0)
        out, lse = _prepare_dcp_attention(self, out, lse, has_keys, "decode")
        out = _reduce_dcp_attention(self, out, lse, "decode")
        output.copy_(out.to(output.dtype))
        check_dcp_tensor(self, "decode.sink_output", output)

    def _forward_prefill(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,  # Only used when compress_ratio > 1
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: FlashMLASparseMetadata | None,
        swa_metadata: "DeepseekSparseSWAMetadata",
    ) -> None:
        assert self.dcp_group is not None
        swa_only = attn_metadata is None
        check_dcp_tensor(self, "prefill.query", q)

        num_prefills = swa_metadata.num_prefills
        num_prefill_tokens = swa_metadata.num_prefill_tokens
        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens

        # Use pre-computed prefill metadata.
        seq_lens = swa_metadata.prefill_seq_lens
        gather_lens = swa_metadata.prefill_gather_lens
        assert seq_lens is not None
        assert gather_lens is not None
        if self.dcp_group.rank_in_group != 0:
            gather_lens = torch.zeros_like(gather_lens)

        # Derive prefill-local token offsets from the full query_start_loc_cpu.
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
                topk_indices = _localize_c4_indices(
                    topk_indices, self.dcp_world_size, self.dcp_group.rank_in_group,
                )
            else:
                # C128A: pre-computed during metadata build.
                assert attn_metadata is not None
                topk_indices = attn_metadata.c128a_prefill_topk_indices
                topk_indices = _localize_c128a_prefill_indices(
                    topk_indices, self.dcp_world_size,
                    self.dcp_group.rank_in_group,
                )
            top_k = topk_indices.shape[-1]
            # Compressed region must fit the full compressed pool (seq_len //
            # compress_ratio), not just top_k. top_k bounds how many indices
            # the indexer selects, not the pool size it indexes into.
            N = (self.max_model_len + self.compress_ratio - 1) // self.compress_ratio
        else:
            # NOTE(woosuk): topk_indices will not be used for SWA-only layers.
            assert self.topk_indices_buffer is not None
            topk_indices = self.topk_indices_buffer[num_decode_tokens:]
            top_k = 0
            N = 0

        M = N + self.window_size + self.max_num_batched_tokens
        num_chunks = (num_prefills + PREFILL_CHUNK_SIZE - 1) // PREFILL_CHUNK_SIZE

        workspace_manager = current_workspace_manager()
        kv = workspace_manager.get_simultaneous(
            ((PREFILL_CHUNK_SIZE, M, q.shape[-1]), torch.bfloat16),
        )[0]
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * PREFILL_CHUNK_SIZE
            chunk_end = min(chunk_start + PREFILL_CHUNK_SIZE, num_prefills)
            chunk_size = chunk_end - chunk_start
            # The workspace is reused across layers and requests. FlashMLA
            # can load masked KV lanes before applying topk_length; those
            # lanes must contain finite values rather than allocator leftovers.
            kv.zero_()
            if not swa_only:
                # Gather compressed KV
                assert attn_metadata is not None
                block_table = attn_metadata.block_table[num_decodes:]
                compressed_seq_lens = (
                    seq_lens[chunk_start:chunk_end] // self.compress_ratio
                )
                compressed_seq_lens = get_dcp_local_seq_lens(
                    compressed_seq_lens,
                    self.dcp_world_size,
                    self.dcp_group.rank_in_group,
                    1,
                )
                dequantize_and_gather_k_cache(
                    kv[:chunk_size],
                    compressed_k_cache,
                    seq_lens=compressed_seq_lens,
                    gather_lens=None,
                    block_table=block_table[chunk_start:chunk_end],
                    block_size=attn_metadata.block_size // self.compress_ratio,
                    offset=0,
                )

            # Gather SWA KV
            swa_block_table = swa_metadata.block_table[num_decodes:]
            dequantize_and_gather_k_cache(
                kv[:chunk_size],
                swa_k_cache,
                seq_lens=seq_lens[chunk_start:chunk_end],
                gather_lens=gather_lens[chunk_start:chunk_end],
                block_table=swa_block_table[chunk_start:chunk_end],
                block_size=swa_metadata.block_size,
                offset=N,
            )

            # Combine the topk indices and SWA indices for gathered KV cache
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

            check_dcp_tensor(self, "prefill.gathered_kv", kv[:chunk_size])
            q_chunk = q[query_start:query_end]
            q_chunk = self.dcp_group.all_gather(q_chunk, dim=1)
            actual_heads = q_chunk.shape[1]
            if actual_heads < 64:
                # FlashMLA sparse prefill instantiates 64/128-head kernels.
                q_chunk = torch.nn.functional.pad(
                    q_chunk, (0, 0, 0, 64 - actual_heads)
                )
            output_chunk, _, local_lse = flash_mla_sparse_fwd(
                q=q_chunk,
                kv=kv.view(-1, 1, q.shape[-1]),
                indices=combined_indices.unsqueeze(1),
                sm_scale=self.scale,
                attn_sink=None,
                topk_length=combined_lens,
            )
            output_chunk = output_chunk[:, :actual_heads]
            local_lse = local_lse[:, :actual_heads] if local_lse is not None else None
            output_chunk, local_lse = _prepare_dcp_attention(
                self, output_chunk, local_lse,
                (combined_indices >= 0).any(dim=-1), "prefill",
            )
            output_chunk = _reduce_dcp_attention(self, output_chunk, local_lse, "prefill")
            output[query_start:query_end].copy_(output_chunk.to(output.dtype))
            check_dcp_tensor(self, "prefill.sink_output", output[query_start:query_end])
