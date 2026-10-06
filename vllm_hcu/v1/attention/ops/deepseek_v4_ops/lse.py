# SPDX-License-Identifier: Apache-2.0
"""Numerically verify the installed HCU FlashMLA LSE conventions once."""

import functools
import math

import torch


def lse_is_natural(lse):
    value = lse.float()
    if torch.allclose(value, torch.full_like(value, math.log(4)), atol=1e-3, rtol=1e-3):
        return True
    if torch.allclose(value, torch.full_like(value, 2.0), atol=1e-3, rtol=1e-3):
        return False
    raise RuntimeError(f"FlashMLA LSE calibration failed: expected ln(4) or log2(4), got {value.tolist()}")


@functools.cache
def calibrate_flashmla_lse(device_index):
    from vllm_hcu.v1.attention.ops.flashmla import (
        flash_mla_sparse_fwd, flash_mla_with_kvcache, get_mla_metadata,
    )

    device = torch.device("cuda", device_index)
    indices = torch.full((1, 1, 128), -1, dtype=torch.int32, device=device)
    indices[0, 0, :4] = torch.arange(4, device=device, dtype=torch.int32)
    lengths = torch.full((1,), 4, dtype=torch.int32, device=device)
    _, _, prefill_lse = flash_mla_sparse_fwd(
        q=torch.zeros(1, 64, 512, dtype=torch.bfloat16, device=device),
        kv=torch.zeros(128, 1, 512, dtype=torch.bfloat16, device=device),
        indices=indices, sm_scale=0.125, attn_sink=None, topk_length=lengths,
    )
    # UE8M0 scales occupy the tail of the page, after 64 * 576 data bytes.
    cache = torch.zeros(1, 64, 1, 584, dtype=torch.uint8, device=device)
    cache.view(-1)[64 * 576:] = 127
    _, decode_lse = flash_mla_with_kvcache(
        q=torch.zeros(1, 1, 16, 512, dtype=torch.bfloat16, device=device),
        k_cache=cache, block_table=None, head_dim_v=512,
        tile_scheduler_metadata=get_mla_metadata()[0], cache_seqlens=None,
        is_fp8_kvcache=True, indices=indices, topk_length=lengths,
        softmax_scale=0.125, attn_sink=None,
    )
    result = (lse_is_natural(prefill_lse), lse_is_natural(decode_lse))
    print(f"[DSV4_DCP_LSE] device={device_index} prefill_base={'e' if result[0] else '2'} "
          f"decode_base={'e' if result[1] else '2'}", flush=True)
    return result
