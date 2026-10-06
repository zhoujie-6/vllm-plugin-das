# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Collective reductions for DeepSeek-V4 decode context parallelism."""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator


def dcp_softmax_reduce(
    local_max: torch.Tensor,
    local_sum: torch.Tensor,
    local_weighted_value: torch.Tensor,
    group: "GroupCoordinator",
) -> torch.Tensor:
    """Merge per-rank online-softmax statistics without losing stability."""
    valid = local_sum > 0
    local_max = torch.where(valid, local_max, torch.full_like(local_max, -torch.inf))
    gathered = group.all_gather(local_max, dim=0).reshape(
        (group.world_size,) + local_max.shape
    )
    global_max = gathered.max(dim=0).values
    scale = torch.where(valid, torch.exp(local_max - global_max), 0.0)
    payload = torch.stack((local_sum * scale, local_weighted_value * scale))
    global_sum, global_value = group.all_reduce(payload).unbind(0)
    return torch.where(global_sum > 0, global_value / global_sum, 0.0)
