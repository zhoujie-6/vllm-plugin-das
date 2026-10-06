# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""HCU DeepSeek sparse-MLA indexer runtime adapter."""

from __future__ import annotations

import functools
from types import ModuleType

from ._common import (
    PatchCompatibilityError,
    already_applied,
    load_exact_module,
    require_callable,
    require_class,
    require_exact_signature,
)

TARGET_MODULE = "vllm.v1.attention.backends.mla.indexer"
PATCH_ID = "worker.op_opt.mla.indexer_hcu"
TARGETS = (
    f"{TARGET_MODULE}.split_indexer_prefill_chunks",
    f"{TARGET_MODULE}.split_decodes_and_prefills",
    f"{TARGET_MODULE}.DeepseekV32IndexerMetadataBuilder.build",
    f"{TARGET_MODULE}.DeepseekV32IndexerMetadataBuilder.__init__",
)
_MARKER = "_vllm_hcu_mla_indexer_applied"
_WRAPPER = "_vllm_hcu_mla_indexer_wrapper"


def apply_to_module(module: ModuleType) -> bool:
    indexer = load_exact_module(TARGET_MODULE, module)
    builder_cls = require_class(indexer, "DeepseekV32IndexerMetadataBuilder", f"{TARGET_MODULE}.DeepseekV32IndexerMetadataBuilder")
    wrapped = (
        (indexer, "split_indexer_prefill_chunks", TARGETS[0], _WRAPPER),
        (indexer, "split_decodes_and_prefills", TARGETS[1], _WRAPPER),
        (builder_cls, "build", TARGETS[2], _WRAPPER),
        (builder_cls, "__init__", TARGETS[3], _WRAPPER),
    )
    if already_applied(indexer, _MARKER, wrapped):
        return False
    split_chunks = require_callable(indexer, "split_indexer_prefill_chunks", TARGETS[0])
    require_exact_signature(
        split_chunks, TARGETS[0],
        positional=("seq_lens_cpu", "query_lens_cpu", "workspace_size", "max_logits_bytes", "request_offset"),
        defaults={"request_offset": 0},
    )
    split_batch = require_callable(indexer, "split_decodes_and_prefills", TARGETS[1])
    require_exact_signature(
        split_batch, TARGETS[1],
        positional=("common_attn_metadata", "decode_threshold", "require_uniform", "treat_short_extends_as_decodes"),
        defaults={"decode_threshold": 1, "require_uniform": False,
                  "treat_short_extends_as_decodes": True},
    )
    build = require_callable(builder_cls, "build", TARGETS[2])
    require_exact_signature(
        build, TARGETS[2],
        positional=("self", "common_prefix_len", "common_attn_metadata", "fast_build"),
        defaults={"fast_build": False},
    )
    builder_init = require_callable(builder_cls, "__init__", TARGETS[3])

    @functools.wraps(builder_init)
    def hcu_builder_init(self, *args, **kwargs):
        config = kwargs.get("vllm_config")
        if config is None:
            config = next(
                (arg for arg in args if hasattr(arg, "parallel_config")), None
            )
        parallel = getattr(config, "parallel_config", None)
        dcp_world_size = int(
            getattr(parallel, "decode_context_parallel_size", 1)
        )
        if dcp_world_size <= 1:
            return builder_init(self, *args, **kwargs)
        interleave = int(getattr(parallel, "cp_kv_cache_interleave_size", 1))
        if interleave != 1:
            raise NotImplementedError(
                "DeepSeek-V4 DCP on HCU currently requires "
                "cp_kv_cache_interleave_size=1"
            )
        parallel.decode_context_parallel_size = 1
        try:
            builder_init(self, *args, **kwargs)
        finally:
            parallel.decode_context_parallel_size = dcp_world_size
        from vllm.distributed import get_dcp_group

        self.dcp_world_size = dcp_world_size
        self.dcp_rank = get_dcp_group().rank_in_group
        self.cp_kv_cache_interleave_size = interleave

    @functools.wraps(split_chunks)
    def hcu_split_chunks(seq_lens_cpu, query_lens_cpu, workspace_size,
                         max_logits_bytes, request_offset=0):
        chunks = split_chunks(seq_lens_cpu, query_lens_cpu, workspace_size,
                              max_logits_bytes, request_offset)
        return [
            (req_slice, query_slice)
            for req_slice, query_slice in chunks
            if query_slice.stop > query_slice.start
        ]

    @functools.wraps(split_batch)
    def hcu_split_batch(common_attn_metadata, decode_threshold=1,
                        require_uniform=False,
                        treat_short_extends_as_decodes=None):
        if treat_short_extends_as_decodes is None:
            treat_short_extends_as_decodes = (
                getattr(common_attn_metadata, "is_prefilling", None) is None
            )
        return split_batch(
            common_attn_metadata, decode_threshold, require_uniform,
            treat_short_extends_as_decodes,
        )

    @functools.wraps(build)
    def hcu_build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        original_mapping = getattr(indexer, "get_compressed_slot_mapping", None)
        original_localize = getattr(self, "_dcp_localize_decode_seq_lens", None)
        ratio = int(getattr(self, "compress_ratio", 1))
        dcp_world_size = int(getattr(self, "dcp_world_size", 1))
        patched_dcp = dcp_world_size > 1 and ratio > 1
        if patched_dcp:
            assert original_localize is not None
            assert original_mapping is not None
            from vllm_hcu.v1.attention.backends.mla.compressor_utils import (
                get_compressed_slot_mapping_dcp,
            )

            def dcp_mapping(
                num_tokens,
                query_start_loc,
                seq_lens,
                block_table,
                block_size,
                compress_ratio,
                out=None,
            ):
                return get_compressed_slot_mapping_dcp(
                    num_tokens,
                    common_attn_metadata.slot_mapping,
                    query_start_loc,
                    seq_lens,
                    block_table,
                    block_size,
                    compress_ratio,
                    out,
                    dcp_world_size=self.dcp_world_size,
                    dcp_rank=self.dcp_rank,
                    cp_kv_cache_interleave_size=self.cp_kv_cache_interleave_size,
                )

            def localize_compressed(seq_lens, num_decodes, is_buffer_view):
                local = original_localize(
                    seq_lens // ratio, num_decodes, is_buffer_view
                )
                return local * ratio

            indexer.get_compressed_slot_mapping = dcp_mapping
            self._dcp_localize_decode_seq_lens = localize_compressed
        try:
            result = build(self, common_prefix_len, common_attn_metadata, fast_build)
        finally:
            if patched_dcp:
                indexer.get_compressed_slot_mapping = original_mapping
            if patched_dcp:
                self._dcp_localize_decode_seq_lens = original_localize
        from vllm_hcu.model_executor.layers.attention.pcp import (
            effective_pcp_world_size,
        )

        result.num_kv_actual_tokens = getattr(
            common_attn_metadata, "num_kv_actual_tokens",
            common_attn_metadata.num_actual_tokens,
        )
        vllm_config = getattr(self, "vllm_config", None)
        if vllm_config is None:
            result.pcp_world_size = effective_pcp_world_size(
                int(getattr(common_attn_metadata, "pcp_world_size", 1))
            )
        else:
            parallel_config = getattr(vllm_config, "parallel_config", None)
            pcp_world_size = getattr(
                parallel_config,
                "prefill_context_parallel_size",
                None,
            )
            if pcp_world_size is None:
                raise PatchCompatibilityError(
                    "required vLLM 0.25.1 prefill_context_parallel_size "
                    "is missing from sparse indexer metadata builder"
                )
            result.pcp_world_size = effective_pcp_world_size(
                int(pcp_world_size)
            )
        # HCU's AITER, lightop, and torch paged-MQA paths do not consume the
        # upstream DeepGEMM schedule metadata. In particular, lightop builds
        # its schedule internally when called with ``schedule_meta=None``.
        # Precomputing it here rejects valid flattened MTP2 batch widths (for
        # example 1024 requests x 3 target tokens) before model execution.
        return result

    for function in (hcu_split_chunks, hcu_split_batch, hcu_build, hcu_builder_init):
        setattr(function, _WRAPPER, True)
    setattr(indexer, "_vllm_hcu_original_split_indexer_prefill_chunks", split_chunks)
    setattr(indexer, "_vllm_hcu_original_split_decodes_and_prefills", split_batch)
    setattr(builder_cls, "_vllm_hcu_original_build", build)
    setattr(indexer, "split_indexer_prefill_chunks", hcu_split_chunks)
    setattr(indexer, "split_decodes_and_prefills", hcu_split_batch)
    setattr(builder_cls, "build", hcu_build)
    setattr(builder_cls, "__init__", hcu_builder_init)
    setattr(indexer, _MARKER, True)
    return True


def apply(module: ModuleType | None = None) -> bool:
    return apply_to_module(load_exact_module(TARGET_MODULE, module))


__all__ = ["PATCH_ID", "TARGET_MODULE", "TARGETS", "apply", "apply_to_module"]
