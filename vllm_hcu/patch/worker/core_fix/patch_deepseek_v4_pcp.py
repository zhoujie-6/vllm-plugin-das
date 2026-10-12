# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Full-KV PCP for the native AMD DeepSeek V4 execution path.

Reuse MRV2's balanced head/tail layout and native compressor/indexer metadata.
Projections and sparse prefill queries are partitioned; cache writes consume
the gathered original request order. mHC, hash routing and MoE remain local.
"""

from __future__ import annotations

import functools
import importlib
from contextvars import ContextVar
from types import ModuleType

import torch

from vllm.logger import init_logger

from ._common import (
    PatchCompatibilityError, load_exact_module, require_callable, require_class,
    require_exact_signature,
)

logger = init_logger(__name__)

TARGET_MODULE = "vllm.models.deepseek_v4.amd.rocm"
PATCH_ID = "worker.core_fix.deepseek_v4.full_kv_pcp"
_MARKER = "_vllm_hcu_deepseek_v4_pcp_applied"
_ATTENTION_MANAGER = ContextVar("deepseek_v4_pcp_attention_manager", default=None)
_PREFILL_ROWS = ContextVar("deepseek_v4_pcp_prefill_rows", default=None)
_LOCAL_QUERIES = ContextVar("deepseek_v4_pcp_local_queries", default=None)
_LOCAL_MQA = ContextVar("deepseek_v4_pcp_local_mqa", default=None)


def local_queries_enabled(manager) -> bool:
    return manager is not None and _LOCAL_QUERIES.get() is manager


@functools.cache
def _pcp_insert_op():
    # LightOp's PCP entry has separate Q/KV lengths and position vectors.
    # KVNorm remains the caller's responsibility, unlike the combined op.
    try:
        import lightop
    except ImportError:
        return None
    native = getattr(lightop, "op", None)
    if native is None:
        return None
    op = getattr(native, "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_pcp",
                 None)
    return op if callable(op) else None


def _can_use_local_queries(layer, manager) -> bool:
    # Native decode consumes global metadata and replicated queries. Keep its
    # established path, including mixed prefill/decode and replicated MTP.
    from vllm_hcu.platforms import envs as henvs

    eligible = (
        henvs.optional_custom_op_enabled(
            henvs.VLLM_HCU_DEEPSEEK_V4_PCP_LOCAL_QUERIES
        )
        and hasattr(layer, "_prepare_and_attn_fn")
        and hasattr(layer, "_fused_qnorm_rope_kv_insert")
        and hasattr(layer, "forward_mqa")
        and layer.head_dim == 512
        and layer.n_local_heads == layer.padded_heads
        and layer.swa_cache_layer.kv_cache.dtype == torch.uint8
        and bool(manager.global_batch.is_prefilling_np.all())
    )
    if not eligible:
        return False
    from vllm.forward_context import get_forward_context
    metadata = get_forward_context().attn_metadata
    swa = (metadata.get(layer.swa_cache_layer.prefix)
           if isinstance(metadata, dict) else None)
    # A short prefill can be classified as decode by the attention builder
    # despite the request-level is_prefilling flag. Never pass shape-only Q
    # views to a decode kernel.
    if swa is None or getattr(swa, "num_decodes", 0) or getattr(swa, "num_decode_tokens", 0):
        return False
    if _pcp_insert_op() is None:
        logger.warning_once(
            "DeepSeek V4 PCP local queries unavailable: LightOp does not "
            "export fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert_pcp; "
            "using the full-query compatibility path."
        )
        return False
    return True


def _metadata_manager(layer):
    from vllm.forward_context import get_forward_context
    from vllm_hcu.model_executor.layers.attention.pcp import in_replicated_mtp_batch

    if in_replicated_mtp_batch():
        return None
    metadata = get_forward_context().attn_metadata
    if not isinstance(metadata, dict):
        return None
    swa = metadata.get(layer.swa_cache_layer.prefix)
    return getattr(swa, "_vllm_hcu_deepseek_v4_pcp_manager", None)


def _run_local_prefill(kernel, manager, root_q, decode_tokens, **kwargs):
    """Select cached local rows while preserving global KV coordinates."""
    q = kwargs["q"]
    start = (q.storage_offset() - root_q.storage_offset()) // root_q.stride(0)
    selected, local_rows = manager.local_query_rows(
        start + decode_tokens, start + decode_tokens + q.shape[0],
    )
    local_mqa = _LOCAL_MQA.get()
    if local_mqa is None:
        kwargs["output"].zero_()
    if selected.numel() == 0:
        return
    if local_mqa is None:
        local_q = q.index_select(0, selected).contiguous()
    else:
        local_q = local_mqa[0].index_select(0, local_rows).contiguous()
    local_output = torch.empty_like(local_q)
    full_output = kwargs["output"]
    kwargs["q"] = local_q
    for name in ("indices", "topk_length"):
        kwargs[name] = kwargs[name].index_select(0, selected).contiguous()
    kwargs["output"] = local_output
    kernel(**kwargs)
    if local_mqa is None:
        full_output.index_copy_(0, selected, local_output)
    else:
        local_mqa[1].index_copy_(0, local_rows, local_output)


def apply_to_module(module: ModuleType) -> bool:
    rocm = load_exact_module(TARGET_MODULE, module)
    cls = require_class(rocm, "DeepseekV4ROCMAiterMLAAttention", TARGET_MODULE)
    if getattr(cls, _MARKER, False):
        if not all(getattr(getattr(cls, name), _MARKER, False) for name in (
            "forward", "_run_parallel_input_projections", "_wq_b_proj",
            "_o_proj", "_forward_prefill",
        )) or not all(
            getattr(getattr(cls, name), _MARKER, False)
            for name in ("forward_mqa", "_fused_qnorm_rope_kv_insert")
            if hasattr(cls, name)
        ) or not getattr(rocm.rocm_sparse_attn_prefill, _MARKER, False):
            raise PatchCompatibilityError("stale DeepSeek V4 PCP patch")
        return False
    originals = {
        name: require_callable(cls, name, f"{TARGET_MODULE}.{name}")
        for name in (
            "forward", "_run_parallel_input_projections", "_wq_b_proj",
            "_o_proj", "_forward_prefill",
        )
    }
    for name in ("forward_mqa", "_fused_qnorm_rope_kv_insert"):
        if hasattr(cls, name):
            originals[name] = require_callable(cls, name, TARGET_MODULE)
    if "forward_mqa" in originals:
        require_exact_signature(originals["forward_mqa"], TARGET_MODULE,
                                positional=("self", "q", "kv", "positions", "output"),
                                defaults={})
    if "_fused_qnorm_rope_kv_insert" in originals:
        require_exact_signature(originals["_fused_qnorm_rope_kv_insert"], TARGET_MODULE,
                                positional=("self", "q", "kv", "positions", "attn_metadata"),
                                defaults={})
    kernel = require_callable(rocm, "rocm_sparse_attn_prefill", TARGET_MODULE)
    signatures = {
        "forward": (("self", "positions", "hidden_states", "llama_4_scaling"),
                    {"llama_4_scaling": None}),
        "_run_parallel_input_projections": (("self", "hidden_states"), {}),
        "_wq_b_proj": (("self", "qr", "qr_scale"), {"qr_scale": None}),
        "_o_proj": (("self", "o", "positions"), {}),
        "_forward_prefill": (("self", "q", "positions", "compressed_k_cache",
                              "swa_k_cache", "output", "attn_metadata", "swa_metadata"), {}),
    }
    for name, (positional, defaults) in signatures.items():
        require_exact_signature(originals[name], f"{TARGET_MODULE}.{name}",
                                positional=positional, defaults=defaults)
    require_exact_signature(
        kernel, f"{TARGET_MODULE}.rocm_sparse_attn_prefill",
        positional=("q", "kv", "indices", "topk_length", "scale", "head_dim",
                    "nope_head_dim", "rope_head_dim", "attn_sink", "output",
                    "ragged_indices", "ragged_indptr"),
        defaults={"ragged_indices": None, "ragged_indptr": None},
    )
    from vllm.models.deepseek_v4.attention import DeepseekV4Indexer
    from vllm.models.deepseek_v4.compressor import CompressorBackend
    from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWABackend

    indexer_projection = require_callable(
        DeepseekV4Indexer, "_wq_b_proj", "DeepseekV4Indexer._wq_b_proj"
    )
    require_exact_signature(indexer_projection, "DeepseekV4Indexer._wq_b_proj",
                            positional=("self", "qr", "qr_scale"),
                            defaults={"qr_scale": None})

    @functools.wraps(originals["forward"])
    def forward(self, positions, hidden_states, llama_4_scaling=None):
        manager = _metadata_manager(self)
        if manager is None:
            return originals["forward"](self, positions, hidden_states, llama_4_scaling)
        use_local_queries = _can_use_local_queries(self, manager)
        token = _ATTENTION_MANAGER.set(manager)
        local_token = _LOCAL_QUERIES.set(manager if use_local_queries else None)
        try:
            if local_queries_enabled(manager):
                logger.info_once("DeepSeek V4 PCP uses local queries and full KV.")
                # Do not restore hidden states or QR/Q merely to adapt to a
                # full-query API. Compressors still consume full KV scores.
                qr_kv, kv_score, indexer_kv_score, indexer_weights = (
                    originals["_run_parallel_input_projections"](self, hidden_states)
                )
                qr, raw_kv = qr_kv.split([self.q_lora_rank, self.head_dim], dim=-1)
                qr = self.q_norm(qr)
                # KVNorm is token-local too. Normalize only this rank's KV
                # before gathering; the PCP cache writer must not repeat it.
                raw_kv = manager.restore_hidden_states(self.kv_norm(raw_kv).contiguous())
                kv_score = (None if kv_score is None
                            else manager.restore_hidden_states(kv_score))
                indexer_kv_score = (None if indexer_kv_score is None
                                    else manager.restore_hidden_states(indexer_kv_score))
                full_tokens = manager.global_batch.num_tokens
                # Native indexer uses hidden_states only for metadata/shape;
                # all numerical projections above consumed the local tensor.
                shape_only_hidden = hidden_states[:1].expand(full_tokens, -1)
                output = hidden_states.new_zeros(
                    (hidden_states.shape[0], self.padded_heads, self.head_dim)
                )
                from vllm_hcu.model_executor.layers.attention.pcp import (
                    replicated_mtp_batch_scope,
                )
                with replicated_mtp_batch_scope():
                    self._prepare_and_attn_fn(
                        shape_only_hidden, qr, raw_kv, None, kv_score,
                        indexer_kv_score, indexer_weights,
                        manager.global_batch.positions[:full_tokens], output,
                    )
                return originals["_o_proj"](
                    self, output[:, :self.n_local_heads], positions,
                )
            from vllm_hcu.model_executor.layers.attention.pcp import (
                replicated_mtp_batch_scope,
            )

            full_hidden = manager.restore_hidden_states(hidden_states)
            # Native indexer/compressor calls already see complete KV. Reuse
            # the existing logical-PCP scope to suppress a second cache gather.
            with replicated_mtp_batch_scope():
                return originals["forward"](
                    self, manager.global_batch.positions[:full_hidden.shape[0]],
                    full_hidden, llama_4_scaling,
                )
        finally:
            _LOCAL_QUERIES.reset(local_token)
            _ATTENTION_MANAGER.reset(token)

    @functools.wraps(originals["_run_parallel_input_projections"])
    def projections(self, hidden_states):
        manager = _ATTENTION_MANAGER.get()
        if manager is None:
            return originals["_run_parallel_input_projections"](self, hidden_states)
        local = manager.select_local_tokens(hidden_states)
        results = originals["_run_parallel_input_projections"](self, local)
        return tuple(
            None if value is None else manager.restore_hidden_states(value)
            for value in results
        )

    def partition_query_projection(original):
        @functools.wraps(original)
        def query_projection(self, qr, qr_scale=None):
            manager = _ATTENTION_MANAGER.get()
            if manager is None or local_queries_enabled(manager):
                return original(self, qr, qr_scale)
            result = original(
                self, manager.select_local_tokens(qr),
                None if qr_scale is None else manager.select_local_tokens(qr_scale),
            )
            return manager.restore_hidden_states(result)
        return query_projection

    @functools.wraps(originals["_o_proj"])
    def output_projection(self, output, positions):
        manager = _ATTENTION_MANAGER.get()
        if manager is not None:
            output = manager.select_local_tokens(output)
            positions = manager.select_local_tokens(positions)
        return originals["_o_proj"](self, output, positions)

    @functools.wraps(originals["_forward_prefill"])
    def prefill(self, q, positions, compressed_k_cache, swa_k_cache,
                output, attn_metadata, swa_metadata):
        manager = _ATTENTION_MANAGER.get()
        token = None
        if manager is not None:
            token = _PREFILL_ROWS.set((manager, q, swa_metadata.num_decode_tokens))
        try:
            return originals["_forward_prefill"](
                self, q, positions, compressed_k_cache, swa_k_cache,
                output, attn_metadata, swa_metadata,
            )
        finally:
            if token is not None:
                _PREFILL_ROWS.reset(token)

    @functools.wraps(kernel)
    def sparse_prefill(q, kv, indices, topk_length, scale, head_dim,
                       nope_head_dim, rope_head_dim, attn_sink, output,
                       ragged_indices=None, ragged_indptr=None):
        selection = _PREFILL_ROWS.get()
        if selection is None:
            return kernel(q, kv, indices, topk_length, scale, head_dim,
                          nope_head_dim, rope_head_dim, attn_sink, output,
                          ragged_indices, ragged_indptr)
        manager, root_q, decode_tokens = selection
        # The native V4 prefill call provides dense combined indices. Keep
        # the installed HCU/FlashMLA provider authoritative for selected rows.
        if ragged_indices is not None or ragged_indptr is not None:
            raise ValueError("DeepSeek V4 PCP prefill requires dense combined indices")
        return _run_local_prefill(
            kernel, manager, root_q, decode_tokens, q=q, kv=kv, indices=indices,
            topk_length=topk_length, scale=scale, head_dim=head_dim,
            nope_head_dim=nope_head_dim, rope_head_dim=rope_head_dim,
            attn_sink=attn_sink, output=output,
        )

    if "_fused_qnorm_rope_kv_insert" in originals:
        @functools.wraps(originals["_fused_qnorm_rope_kv_insert"])
        def local_insert(self, q, kv, positions, attn_metadata):
            manager = _ATTENTION_MANAGER.get()
            if not local_queries_enabled(manager):
                return originals["_fused_qnorm_rope_kv_insert"](
                    self, q, kv, positions, attn_metadata,
                )
            from vllm.forward_context import get_forward_context
            metadata = get_forward_context().attn_metadata[self.swa_cache_layer.prefix]
            q = q.contiguous()
            # The PCP kernel has no KVNorm. forward() already applied the
            # learned norm on local KV before gathering (exactly once).
            normalized_kv = kv.contiguous()
            _pcp_insert_op()(
                q, normalized_kv,
                self.swa_cache_layer.kv_cache.view(
                    self.swa_cache_layer.kv_cache.shape[0], -1,
                ),
                metadata.slot_mapping[:kv.shape[0]].to(dtype=torch.int64).contiguous(),
                manager.select_local_tokens(positions).to(dtype=torch.int64),
                positions.to(dtype=torch.int64), self.rotary_emb.cos_sin_cache,
                self.eps, metadata.block_size,
            )
            return q
        setattr(local_insert, _MARKER, True)
        cls._fused_qnorm_rope_kv_insert = local_insert

    if "forward_mqa" in originals:
        @functools.wraps(originals["forward_mqa"])
        def local_mqa(self, q, kv, positions, output):
            manager = _ATTENTION_MANAGER.get()
            if not local_queries_enabled(manager):
                return originals["forward_mqa"](self, q, kv, positions, output)
            # The native prefill loop owns global chunk/KV metadata. Give it
            # tiny shape-only views with a nonzero token stride so chunk
            # offsets remain recoverable, rather than a 2-GiB global Q buffer.
            full_tokens = manager.global_batch.num_tokens
            q_proxy = q.new_empty(full_tokens).as_strided(
                (full_tokens, q.shape[1], q.shape[2]), (1, 0, 0),
            )
            out_proxy = output.new_empty(full_tokens).as_strided(
                q_proxy.shape, (1, 0, 0),
            )
            token = _LOCAL_MQA.set((q, output))
            try:
                return originals["forward_mqa"](
                    self, q_proxy, kv, positions, out_proxy,
                )
            finally:
                _LOCAL_MQA.reset(token)
        setattr(local_mqa, _MARKER, True)
        cls.forward_mqa = local_mqa

    attention_module = importlib.import_module("vllm.models.deepseek_v4.attention")
    quant = getattr(attention_module, "fused_indexer_q_rope_quant", None)
    if quant is not None and not getattr(quant, _MARKER, False):
        @functools.wraps(quant)
        def local_indexer_quant(*args, **kwargs):
            manager = _ATTENTION_MANAGER.get()
            if local_queries_enabled(manager):
                if args:
                    args = (manager.select_local_tokens(args[0]), *args[1:])
                else:
                    kwargs["positions"] = manager.select_local_tokens(kwargs["positions"])
            return quant(*args, **kwargs)
        setattr(local_indexer_quant, _MARKER, True)
        attention_module.fused_indexer_q_rope_quant = local_indexer_quant

    # These are V4-only cache backends; all of them now consume full KV on
    # each rank. Do not advertise PCP on generic MLA/SWA backends.
    for backend in (
        rocm.DeepseekV4ROCMAiterMLASparseBackend,
        CompressorBackend, DeepseekSparseSWABackend,
    ):
        backend.supports_pcp = classmethod(lambda cls: True)
    for name, wrapper in (
        ("forward", forward), ("_run_parallel_input_projections", projections),
        ("_wq_b_proj", partition_query_projection(originals["_wq_b_proj"])),
        ("_o_proj", output_projection),
        ("_forward_prefill", prefill),
    ):
        setattr(wrapper, _MARKER, True)
        setattr(cls, name, wrapper)
    indexer_wrapper = partition_query_projection(indexer_projection)
    setattr(indexer_wrapper, _MARKER, True)
    DeepseekV4Indexer._wq_b_proj = indexer_wrapper
    setattr(sparse_prefill, _MARKER, True)
    rocm.rocm_sparse_attn_prefill = sparse_prefill
    setattr(cls, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "apply", "apply_to_module"]
