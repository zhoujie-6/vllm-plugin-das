# SPDX-License-Identifier: Apache-2.0
"""Opt-in synchronous checks for small eager DCP correctness runs."""

import os

import torch


def check_dcp_tensor(owner, stage, tensor, allow_negative_inf=False):
    if os.environ.get("VLLM_HCU_DEEPSEEK_V4_DCP_DEBUG") != "1":
        return
    if tensor.is_cuda and torch.cuda.is_current_stream_capturing():
        return
    valid = torch.isfinite(tensor)
    if allow_negative_inf:
        valid = valid | torch.isneginf(tensor)
    if not valid.all().item():
        prefix = getattr(owner, "prefix", type(owner).__name__)
        group = getattr(owner, "dcp_group", None)
        rank = getattr(group, "rank_in_group", -1)
        raise RuntimeError(
            f"DeepSeek-V4 DCP nonfinite: layer={prefix} rank={rank} "
            f"stage={stage} shape={tuple(tensor.shape)} "
            f"nan={torch.isnan(tensor).sum().item()} "
            f"posinf={torch.isposinf(tensor).sum().item()} "
            f"neginf={torch.isneginf(tensor).sum().item()}"
        )
