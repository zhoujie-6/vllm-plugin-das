# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Opaque PCP calls keep graph shapes global and collective ordering explicit."""
from __future__ import annotations
import torch
from vllm.forward_context import get_forward_context
from vllm.utils.torch_utils import direct_register_custom_op
from vllm_hcu.v1.deepseek_v4_pcp import (
    gather_tokens, select_tokens, get_forward_plan,
)


def attention_forward(hidden_states: torch.Tensor, positions: torch.Tensor,
                      prefix: str) -> torch.Tensor:
    ctx = get_forward_context()
    attn = ctx.no_compile_layers[prefix]
    metadata = ctx.attn_metadata
    swa = metadata.get(attn.swa_cache_layer.prefix) if isinstance(metadata, dict) else None
    plan = getattr(swa, "hcu_v4_pcp_plan", None)
    if plan is None:
        return attn._vllm_hcu_non_pcp_forward(positions, hidden_states)
    padded_tokens = hidden_states.shape[0]
    nd = swa.num_decode_tokens
    hidden_states = hidden_states[:nd + plan.seq_len]
    positions = positions[:nd + plan.seq_len]
    local_hidden = select_tokens(hidden_states, plan, nd)
    local_positions = select_tokens(positions, plan, nd)
    qr_kv, kv_score, indexer_score, indexer_weights = (
        attn.attn_gemm_parallel_execute(local_hidden)
    )
    qr, raw_kv = qr_kv.split([attn.q_lora_rank, attn.head_dim], dim=-1)
    qr = attn.q_norm(qr)
    # These gathered tensors own their storage: a later gather must not
    # overwrite a live compressor or KV input (including on empty ranks).
    raw_kv = gather_tokens(raw_kv, plan, nd)
    kv_score = gather_tokens(kv_score, plan, nd)
    indexer_score = gather_tokens(indexer_score, plan, nd)
    if attn.indexer is not None:
        indexer = attn.indexer
        from vllm.models.deepseek_v4.attention import fused_indexer_q_rope_quant
        index_q, _ = indexer.wq_b(qr)
        index_q = index_q.view(-1, indexer.n_head, indexer.head_dim)
        q_quant, weights = fused_indexer_q_rope_quant(
            local_positions, index_q, attn.indexer_rotary_emb.cos_sin_cache,
            indexer_weights, indexer.softmax_scale, indexer.n_head**-0.5,
            use_fp4=indexer.use_fp4_kv,
        )
        # v0.25.1's indexer op has a global-row ABI. Scatter the locally
        # projected queries without communication, then select owned rows in
        # its prefill kernel. The compressor still sees the complete sequence.
        q_quant = scatter_tokens(q_quant, plan, nd)
        weights = scatter_tokens(weights, plan, nd)
        indexer.compressor(indexer_score, positions, attn.indexer_rotary_emb)
        indexer_meta = metadata[attn.indexer.k_cache.prefix]
        # The indexer reads global tensors but computes only the owned rows.
        # Its full-sized output retains -1 on rows not owned by this rank.
        previous = dict(vars(indexer_meta))
        indexer_meta.hcu_v4_pcp_plan = plan
        indexer_meta.hcu_v4_num_decode_tokens = nd
        try:
            indexer.indexer_op(hidden_states, q_quant, None, weights)
        finally:
            for key in ("hcu_v4_pcp_plan", "hcu_v4_num_decode_tokens"):
                if key in previous:
                    setattr(indexer_meta, key, previous[key])
                else:
                    delattr(indexer_meta, key)

    if attn.compressor is not None:
        attn.compressor(kv_score, positions, attn.rotary_emb)
    q = attn.wq_b(qr).view(-1, attn.n_local_heads, attn.head_dim)
    # v0.25.1 LightOp exposes the fused KVNorm+RoPE insert with one position
    # vector. Scatter local Q into global rows for this inexpensive operation,
    # then select it back. Q projections and attention remain rank-local;
    # every rank inserts the complete KV using the exact existing encoder.
    full_q = q.new_zeros((nd + plan.seq_len, *q.shape[1:]))
    rows = torch.cat((torch.arange(nd, device=q.device), plan.local_indices + nd))
    full_q.index_copy_(0, rows, q)
    full_q = attn._fused_qnorm_rope_kv_insert(full_q, raw_kv, positions, metadata)
    q = select_tokens(full_q, plan, nd)
    output = torch.empty_like(q)
    rocm_meta = metadata.get(attn.prefix)
    from vllm_hcu.v1.attention.deepseek_v4_pcp import forward_prefill
    forward_prefill(
        attn, q[nd:], local_positions[nd:],
        attn.kv_cache if attn.compress_ratio > 1 else None,
        attn.swa_cache_layer.kv_cache, output[nd:], rocm_meta, swa,
    )
    if nd:
        attn._forward_decode(
            q=q[:nd], kv_cache=attn.kv_cache if attn.compress_ratio > 1 else None,
            swa_metadata=swa, attn_metadata=rocm_meta,
            swa_only=attn.compress_ratio <= 1, output=output[:nd],
        )
    local_output = attn._o_proj(output[:, :attn.n_local_heads], local_positions)
    return pad_global_output(gather_tokens(local_output, plan, nd), padded_tokens)


def moe_forward(hidden_states: torch.Tensor, input_ids: torch.Tensor | None,
                prefix: str) -> torch.Tensor:
    layer = get_forward_context().no_compile_layers[prefix]
    plan, nd = get_forward_plan()
    if plan is None:
        return layer._vllm_hcu_non_pcp_moe_forward(hidden_states, input_ids)
    local_hidden = select_tokens(hidden_states, plan, nd)
    local_ids = select_tokens(input_ids, plan, nd)
    # Uniform width also covers TP/naive EP kernels and zero-token ranks.
    # Padding never enters the restored output or the KV cache.
    actual = local_hidden.shape[0]
    parallel = getattr(getattr(getattr(layer, "experts", None), "moe_config", None),
                       "moe_parallel_config", None)
    variable_size = bool(getattr(parallel, "use_all2all_kernels", False))
    width = actual if variable_size else nd + max(plan.local_sizes)
    if actual < width:
        local_hidden = torch.cat((local_hidden, local_hidden.new_zeros(
            (width - actual, *local_hidden.shape[1:]))))
        if local_ids is not None:
            local_ids = torch.cat((local_ids, local_ids.new_zeros(width - actual)))
    output = layer._vllm_hcu_non_pcp_moe_forward(local_hidden, local_ids)
    return pad_global_output(
        gather_tokens(output[:actual], plan, nd), hidden_states.shape[0],
    )


def scatter_tokens(tensor, plan, num_decode_tokens):
    if isinstance(tensor, tuple):
        return tuple(scatter_tokens(item, plan, num_decode_tokens) for item in tensor)
    out = tensor.new_zeros((num_decode_tokens + plan.seq_len, *tensor.shape[1:]))
    rows = torch.cat((torch.arange(num_decode_tokens, device=tensor.device),
                      plan.local_indices + num_decode_tokens))
    if tensor.is_floating_point() and tensor.element_size() == 1:
        # ROCm index_copy has no FP8 kernel. Copy the encoded bytes without
        # dequantizing or changing the indexer quantization/scales.
        out.view(torch.uint8).index_copy_(0, rows, tensor.view(torch.uint8))
    else:
        out.index_copy_(0, rows, tensor)
    return out


def pad_global_output(output, num_tokens):
    if output.shape[0] == num_tokens:
        return output
    if output.shape[0] > num_tokens:
        raise ValueError("DeepSeek-V4 PCP output exceeds the runner batch")
    return torch.cat((output, output.new_zeros(
        (num_tokens - output.shape[0], *output.shape[1:]))))


def _attention_fake(hidden_states, positions, prefix):
    return torch.empty_like(hidden_states)


def _moe_fake(hidden_states, input_ids, prefix):
    return torch.empty_like(hidden_states)


direct_register_custom_op(
    op_name="hcu_deepseek_v4_pcp_attention", op_func=attention_forward,
    mutates_args=[], fake_impl=_attention_fake,
    tags=(torch._C.Tag.cudagraph_unsafe,),
)
direct_register_custom_op(
    op_name="hcu_deepseek_v4_pcp_moe", op_func=moe_forward,
    mutates_args=[], fake_impl=_moe_fake,
    tags=(torch._C.Tag.cudagraph_unsafe,),
)
