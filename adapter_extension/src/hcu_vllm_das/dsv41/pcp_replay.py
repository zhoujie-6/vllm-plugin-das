# SPDX-License-Identifier: Apache-2.0
"""Distribute a long prompt's replay window as well as its initial prefill."""

import os
from dataclasses import replace

from vllm.logger import init_logger
from vllm.models.deepseek_v41.nvidia.model_state import DeepseekV41ModelState
from vllm.v1.worker.gpu.pcp_manager import PCPManager

from ..dsa_prefill_pages import active_prefill_pages


logger = init_logger('vllm.' + __name__)


def supports_balanced_replay(cfg):
    parallel = cfg.parallel_config
    return (os.environ.get('VLLM_HCU_DSV41_BALANCED_PCP_REPLAY', '1') == '1'
            and getattr(cfg.kernel_config, 'moe_backend', None) == 'deep_gemm_mega_moe'
            and cfg.model_config.enforce_eager and cfg.cache_config.swa_bounded_replay
            and parallel.prefill_context_parallel_size == 8
            and parallel.data_parallel_size == parallel.decode_context_parallel_size == 1
            and cfg.model_config.quantization == 'hcu_dsv41_ct_w4a8'
            and getattr(cfg.model_config.hf_text_config, 'sliding_window', None) == 128)


class WindowBalancedPCPManager(PCPManager):
    # Selected only for the V4.1 Flash 128-token sliding window.
    window = 128

    def _balanced(self, lengths, prefilling):
        peers = self.pcp_world_size
        pages = active_prefill_pages()
        return (pages is not None and pages.complete_prompts
                and len(lengths) > 0 and prefilling.all()
                and (lengths >= 2 * peers * self.window).all()
                and (lengths % peers == 0).all())

    def _iter_rank_chunks(self, rank, num_scheduled_tokens, is_prefilling):
        if not self._balanced(num_scheduled_tokens, is_prefilling):
            yield from super()._iter_rank_chunks(rank, num_scheduled_tokens, is_prefilling)
            return
        # Keep Core's two contiguous segments per request. Rotate the longer
        # prefix across requests to balance the amount of causal history read.
        peers = self.pcp_world_size
        tail = self.window // peers
        for request, length in enumerate(num_scheduled_tokens):
            prefix = int(length) - self.window
            width = prefix // peers
            lane = (rank + request) % peers
            yield request, lane * width, width
            yield request, prefix + (peers - 1 - lane) * tail, tail

    def partition_batch(self, input_batch, batch_desc):
        local = super().partition_batch(input_batch, batch_desc)
        if self._balanced(input_batch.num_scheduled_tokens, input_batch.is_prefilling_np):
            # Core creates a fresh local InputBatch for each real forward.
            # Both the complete and trimmed row counts now agree on all ranks.
            local._hcu_equal_replay_window = self.window
        return local


class WindowBalancedModelState(DeepseekV41ModelState):
    def _kept_input_batch(self, input_batch, slot_mappings, replay_start, kept_lens, num_padded):
        if (getattr(input_batch, '_hcu_equal_replay_window', None)
                == self.decoder_replay_layers.window):
            # Prompt logprobs keep the whole request. Its untrimmed row count
            # is also equal on all peers. No extra collective is necessary.
            num_padded = int(kept_lens.sum())
        return super()._kept_input_batch(input_batch, slot_mappings, replay_start,
                                         kept_lens, num_padded)

    def _prepare_replay_batch(self, input_batch, *args, **kwargs):
        super()._prepare_replay_batch(input_batch, *args, **kwargs)
        layers = self.decoder_replay_layers
        replay = layers.replay_batch
        if replay is None or getattr(input_batch, '_hcu_equal_replay_window', None) != layers.window:
            return
        # _kept_input_batch built the compact slots and attention metadata.
        # Publish the same extent to Core's layer input padding and MoE.
        rows = replay.rows.numel()
        logger.info_once("DSV4.1 balanced PCP replay: %d global rows, %d rows per rank",
                         replay.num_padded, rows)
        replay.num_padded = rows
        ctx = replay.forward_context
        ctx.batch_descriptor = replace(ctx.batch_descriptor, num_tokens=rows)
