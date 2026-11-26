"""Check W4A8 IPC staging and the standalone expert projections on HCU."""
from types import SimpleNamespace

import pytest
import torch

from hcu_vllm_das.moe.dsv41_w4a8_megamoe import stage_ct_w4a8


@pytest.mark.skipif(not torch.cuda.is_available() or not torch.version.hip, reason="HCU required")
@pytest.mark.parametrize("contiguous", [False, True])
def test_staging_writes_only_active_rows(contiguous):
    rows, hidden, topk = 17, 5120, 6
    device = "cuda"
    x = torch.randn((rows, hidden), dtype=torch.bfloat16, device=device)
    x[:3].zero_()
    x[1, 0] = 1e-12
    x[2, :5] = x.new_tensor([127, 1.5, -1.5, 2.5, -2.5])
    ids = torch.arange(topk, device=device, dtype=torch.int32).expand(rows, -1)
    weights = torch.rand((rows, topk * 2), device=device)[:, ::2]
    if contiguous:
        ids, weights = ids.contiguous(), weights.contiguous()
    buf = SimpleNamespace(
        x=torch.full((rows + 1, hidden), -7, device=device, dtype=torch.int8),
        x_sf=torch.full((rows + 1,), -7., device=device),
        topk_idx=torch.full((rows + 1, topk), -7, device=device, dtype=torch.int64),
        topk_weights=torch.full((rows + 1, topk), -7., device=device),
    )
    from lightop.quant import per_token_quant_int8

    q, sf = per_token_quant_int8(x)
    assert q[2, :5].tolist() == [127, 2, -2, 2, -2]
    stage_ct_w4a8(buf, x, ids, weights)
    for actual, expected in ((buf.x, q), (buf.x_sf, sf.flatten()),
                             (buf.topk_idx, ids.long()), (buf.topk_weights, weights)):
        assert torch.equal(actual[:rows], expected)
        assert torch.all(actual[rows] == -7)


@pytest.mark.skipif(not torch.cuda.is_available() or not torch.version.hip, reason="HCU required")
def test_standalone_expert_projections_with_lightop():
    from hcu_vllm_das.moe.dsv41_w4a8_megamoe import pack_ct_hipc
    from lightop import fuse_silu_mul_clamp_quant_ep
    from lightop.quant import per_token_quant_int8
    from megamoe.w4a8 import grouped_gemm

    device = "cuda"
    rows, h, i = 64, 5120, 2304
    expert_ids = torch.zeros(rows, dtype=torch.int32, device=device)

    def packed_projection(n, k, first_weight, scale):
        # CT offset-binary 0x88888888 represents eight zeros. Only K=0 contributes.
        words = torch.full((1, n, k // 8), -2004318072, device=device, dtype=torch.int32)
        words[:, :, 0] += first_weight
        return pack_ct_hipc(words), scale.contiguous()

    x = torch.zeros(rows, h, device=device, dtype=torch.bfloat16)
    x[:, 0] = torch.arange(rows, device=device) % 17
    gate_scale = torch.ones(1, 2 * i, device=device)
    gate_scale[:, :i] = 2
    down_scale = torch.ones(1, h, device=device)
    down_scale[:, ::2] = 0.5
    l1 = packed_projection(2 * i, h, 1, gate_scale)
    l2 = packed_projection(h, i, -2, down_scale)

    gate_up = x.new_empty(rows, 2 * i)
    grouped_gemm(per_token_quant_int8(x), l1, gate_up, expert_ids)
    mask = torch.tensor([rows], dtype=torch.int32, device=device)
    act, sf = fuse_silu_mul_clamp_quant_ep(gate_up.unsqueeze(0), 10., mask, expect_m=rows)
    y = torch.empty_like(x)
    grouped_gemm((act[0], sf[0]), l2, y, expert_ids)

    gate = (2 * x[:, :1]).clamp(max=10)
    up = x[:, :1].clamp(-10, 10)
    expected = torch.nn.functional.silu(gate) * up
    torch.testing.assert_close(y, (-2 * expected * down_scale).bfloat16(), atol=0.02, rtol=0.01)
