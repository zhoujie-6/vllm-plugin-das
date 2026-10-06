# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""DCP compressor forwards for native DeepSeek-V4 caches."""

from __future__ import annotations

from typing import Any

import torch
from vllm.forward_context import get_forward_context
from vllm.models.deepseek_v4.common.ops.save_partial_states import save_partial_states
from vllm.triton_utils import triton
from vllm_hcu.v1.attention.ops.deepseek_v4_ops.dcp import (
    check_dcp_tensor, dcp_softmax_reduce,
)
from vllm_hcu.v1.attention.ops.deepseek_v4_ops.fused_compress_quant_cache import (
    dsv4_dcp_compressor_partial_stats_kernel,
    _fused_kv_compress_norm_rope_insert_sparse_attn,
    _fused_kv_compress_norm_rope_insert_indexer_attn,
    _fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn,
)


class DeepseekV4DCPCompressor:
    def forward(self, kv_score, positions, rotary_emb):
        metadata = get_forward_context().attn_metadata
        if not isinstance(metadata, dict):
            return
        state_metadata = metadata[self.state_cache.prefix]
        slot_mapping = state_metadata.slot_mapping
        num_actual = slot_mapping.shape[0]
        if num_actual == 0:
            return
        check_dcp_tensor(self, "compressor.input", kv_score[:num_actual])
        kv, score = kv_score.split([self.coff * self.head_dim] * 2, dim=-1)
        state_cache = self.state_cache.kv_cache
        state_width = state_cache.shape[-1] // 2
        save_partial_states(
            kv=kv, score=score, ape=self.ape, positions=positions,
            state_cache=state_cache, slot_mapping=slot_mapping,
            block_size=state_metadata.block_size, state_width=state_width,
            compress_ratio=self.compress_ratio, pdl_kwargs={},
        )
        DeepseekV4DCPCompressor._dcp_compress_and_insert(
            self, state_cache=state_cache, num_actual=num_actual,
            token_to_req_indices=state_metadata.token_to_req_indices,
            positions=positions, block_table=state_metadata.block_table,
            block_size=state_metadata.block_size, state_width=state_width,
            slot_mapping=slot_mapping, cos_sin_cache=rotary_emb.cos_sin_cache,
            kv_cache=self._static_forward_context[self.k_cache_prefix].kv_cache,
            k_cache_metadata=metadata[self.k_cache_prefix], pdl_kwargs={},
        )

    def _dcp_compress_and_insert(
        self,
        state_cache: torch.Tensor,
        num_actual: int,
        token_to_req_indices: torch.Tensor,
        positions: torch.Tensor,
        block_table: torch.Tensor,
        block_size: int,
        state_width: int,
        slot_mapping: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        kv_cache: torch.Tensor,
        k_cache_metadata: Any,
        pdl_kwargs: dict,
    ) -> None:
        partial_m = torch.empty(
            (num_actual, self.head_dim),
            dtype=torch.float32,
            device=state_cache.device,
        )
        partial_s = torch.empty_like(partial_m)
        partial_v = torch.empty_like(partial_m)

        dsv4_dcp_compressor_partial_stats_kernel[(num_actual,)](
            state_cache,
            state_cache.stride(0),
            state_cache.stride(1),
            token_to_req_indices,
            positions,
            slot_mapping,
            block_table,
            block_table.stride(0),
            block_size,
            partial_m,
            partial_s,
            partial_v,
            partial_m.stride(0),
            HEAD_SIZE=self.head_dim,
            TRITON_BLOCK_SIZE=triton.next_power_of_2(self.head_dim),
            STATE_WIDTH=state_width,
            COMPRESS_RATIO=self.compress_ratio,
            OVERLAP=self.overlap,
            num_warps=4 if self.head_dim == 512 else 1,
            **self.cp_layout.triton_kwargs(),
            **pdl_kwargs,
        )

        assert self.dcp_group is not None
        check_dcp_tensor(self, "compressor.partial_max", partial_m, True)
        check_dcp_tensor(self, "compressor.partial_sum", partial_s)
        check_dcp_tensor(self, "compressor.partial_value", partial_v)
        compressed_kv = dcp_softmax_reduce(
            partial_m,
            partial_s,
            partial_v,
            self.dcp_group,
        )
        check_dcp_tensor(self, "compressor.reduced_value", compressed_kv)

        if self.head_dim == 512:
            kernel = _fused_kv_compress_norm_rope_insert_sparse_attn
            block_table = k_cache_metadata.block_table
            block_size = k_cache_metadata.block_size // self.compress_ratio
            dcp_kwargs = self.cp_layout.triton_kwargs()
        else:
            kernel = (
                _fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn
                if self.use_fp4_cache else _fused_kv_compress_norm_rope_insert_indexer_attn
            )
            block_size = kv_cache.shape[1]
            dcp_kwargs = {}
        kernel[(num_actual,)](
            compressed_kv, compressed_kv.stride(0), 0,
            token_to_req_indices, positions, slot_mapping,
            block_table, block_table.stride(0), block_size,
            self.norm.weight, self.rms_norm_eps,
            cos_sin_cache, cos_sin_cache.stride(0),
            kv_cache, k_cache_metadata.slot_mapping, block_size,
            HEAD_SIZE=self.head_dim,
            TRITON_BLOCK_SIZE=triton.next_power_of_2(self.head_dim),
            STATE_WIDTH=state_width,
            COMPRESS_RATIO=self.compress_ratio,
            OVERLAP=self.overlap,
            ROPE_HEAD_DIM=self.rope_head_dim,
            FP8_MAX=448.0,
            QUANT_BLOCK=self._quant_block,
            TOKEN_STRIDE=self._token_stride,
            SCALE_DIM=self._scale_dim,
            KV_BLOCK_STRIDE=kv_cache.stride(0),
            PRECOMPRESSED=True,
            num_warps=4 if self.head_dim == 512 else 1,
            **dcp_kwargs,
            **pdl_kwargs,
        )
