# SPDX-License-Identifier: Apache-2.0
"""Per-group DCP slot mappings for DeepSeek-V4 in Model Runner V2."""

from vllm.v1.attention.backends.utils import PAD_SLOT_ID
from vllm.v1.kv_cache_interface import KVCacheSpecKind, get_kv_cache_spec_kind
from vllm.v1.worker.gpu.block_table import BlockTables, _compute_slot_mappings_kernel


def replicated_group_ids(kv_cache_config):
    return [
        i for i, group in enumerate(kv_cache_config.kv_cache_groups)
        if not getattr(
            group.kv_cache_spec, "dcp_sharded",
            get_kv_cache_spec_kind(group.kv_cache_spec) not in (
                KVCacheSpecKind.SLIDING_WINDOW, KVCacheSpecKind.SLIDING_WINDOW_MLA,
            ),
        )
    ]


class DeepseekV4DCPBlockTables(BlockTables):
    def __init__(self, original, replicated_groups):
        self.replicated_groups = tuple(replicated_groups)
        # Upstream sizes every table as sharded. Replicated state needs the
        # full sequence capacity; retain its block-size alignment as well.
        capacities = [
            table.gpu.shape[1] // original.blocks_per_kv_block[i]
            * (original.cp_size if i in self.replicated_groups else 1)
            for i, table in enumerate(original.block_tables)
        ]
        super().__init__(
            block_sizes=original.block_sizes,
            max_num_reqs=original.max_num_reqs,
            max_num_batched_tokens=original.max_num_batched_tokens,
            max_num_blocks_per_group=capacities,
            device=original.device,
            kernel_block_sizes=original.kernel_block_sizes,
            cp_size=original.cp_size,
            cp_rank=original.cp_rank,
            cp_interleave=original.cp_interleave,
        )

    def compute_slot_mappings(
        self, idx_mapping, query_start_loc, positions, num_tokens_padded,
    ):
        # Compute replicated groups first, then sharded groups separately.
        # Do not run the upstream all-group kernel: it can read past a
        # sharded table if temporarily applied with CP_SIZE=1.
        for i in range(self.num_kv_cache_groups):
            replicated = i in self.replicated_groups
            _compute_slot_mappings_kernel[(1, idx_mapping.shape[0] + 1)](
                self.max_num_batched_tokens,
                idx_mapping, query_start_loc, positions,
                self.block_table_ptrs[i:i + 1],
                self.block_table_strides[i:i + 1],
                self.block_sizes_tensor[i:i + 1],
                self.slot_mappings[i:i + 1], self.slot_mappings.stride(0),
                0 if replicated else self.cp_rank,
                CP_SIZE=1 if replicated else self.cp_size,
                CP_INTERLEAVE=self.cp_interleave,
                PAD_ID=PAD_SLOT_ID, TRITON_BLOCK_SIZE=1024,
            )
        return self.slot_mappings[:, :num_tokens_padded]
