# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 Hygon Information Technology Co., Ltd.

"""Exercise the native top-k boundary without importing the GPU runtime."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def helpers(native, prefill=None, decode=None):
    path = Path(__file__).resolve().parents[2] / "vllm_hcu/v1/attention/ops/rocm_aiter_mla_sparse.py"
    names = {"_decode_row_ends_from_seq_lens", "_lightop_topk_indices_decode",
             "_lightop_topk_indices_prefill", "_topk_indices_torch"}
    tree = ast.parse(path.read_text())
    module = ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)
                             and n.name in names], type_ignores=[])
    ns = {"torch": torch, "_get_lightop_attention": lambda: native}
    exec(compile(module, str(path), "exec"), ns)
    if prefill is not None:
        ns["torch"].ops.hcu_ops.sparse_mla_topk_prefill = prefill
    if decode is not None:
        ns["torch"].ops.hcu_ops.sparse_mla_topk_decode = decode
    return SimpleNamespace(**ns)


@pytest.mark.parametrize("batch,next_n", [(1, 1), (2, 2), (9, 6), (16, 6)])
def test_decode_packed_native_output_preserves_neighbor_columns(batch, next_n):
    rows, width, k = batch * next_n, 17, 5
    # Both input and destination have a larger physical row stride.
    logits = torch.arange(rows * width * 2, dtype=torch.float32).reshape(rows, -1)[:, ::2]
    backing = torch.full((rows + 1, k + 3), 991, dtype=torch.int32)
    output = backing[:rows, :k]
    lengths = torch.arange(batch, dtype=torch.int32) + 2

    def native(x, n, ends, out, m, s0, s1, topk):
        assert n == 1 and (m, topk) == (rows, k)
        assert (s0, s1) == x.stride()
        # Simulate the native ABI: raw packed writes, no output stride.
        packed = torch.as_strided(out, (rows, k), (k, 1))
        for row, length in enumerate(ends.tolist()):
            count = min(k, length)
            packed[row, :count] = torch.arange(count)
            packed[row, count:] = -1

    h = helpers(SimpleNamespace(top_k_per_row_decode=native), decode=lambda x, ends, out: native(x, 1, ends, out, rows, x.stride(0), x.stride(1), k))
    h._lightop_topk_indices_decode(logits, lengths, next_n, output, k)
    ends = (lengths[:, None] - next_n + torch.arange(next_n) + 1).flatten().clamp(0, width)
    expected = torch.where(torch.arange(k)[None, :] < ends[:, None], torch.arange(k), -1).int()
    torch.testing.assert_close(output, expected)
    assert torch.all(backing[:rows, k:] == 991)
    assert torch.all(backing[rows] == 991)


def test_decode_bounds_and_metadata_validation():
    calls = []
    def native(x, n, ends, *args):
        calls.append(ends.clone())
    h = helpers(SimpleNamespace(top_k_per_row_decode=native), decode=lambda x, ends, out: native(x, 1, ends, out))
    x = torch.zeros(4, 9)
    out = torch.empty(4, 3, dtype=torch.int32)
    h._lightop_topk_indices_decode(x, torch.tensor([[-3, 4], [9, 100]]), 2, out, 3)
    torch.testing.assert_close(calls[0], torch.tensor([0, 4, 9, 9], dtype=torch.int32))
    with pytest.raises(ValueError, match="sequence length"):
        h._lightop_topk_indices_decode(x, torch.tensor([2]), 2, out, 3)
    with pytest.raises(ValueError, match="output shape"):
        h._lightop_topk_indices_decode(x, torch.ones(4), 1, out[:2], 3)
    assert len(calls) == 1


def test_prefill_short_row_padding_and_neighbor_columns():
    def native(x, starts, ends, out):
        assert out.is_contiguous()
        out.fill_(-1)
        out[:, 0] = 0
    h = helpers(SimpleNamespace(top_k_per_row_prefill=native), prefill=native)
    backing = torch.full((9, 8), 991, dtype=torch.int32)
    h._lightop_topk_indices_prefill(torch.ones(9, 4), torch.zeros(9), torch.ones(9), backing[:, :5], 5)
    assert torch.all(backing[:, 0] == 0)
    assert torch.all(backing[:, 1:5] == -1)
    assert torch.all(backing[:, 5:] == 991)


@pytest.mark.hcu
@pytest.mark.parametrize("batch,next_n", [(2, 2), (9, 6), (16, 6)])
def test_native_decode_matches_reference(batch, next_n):
    if not torch.cuda.is_available():
        pytest.skip("requires a visible HCU GPU and LightOp")
    from lightop import attention
    import vllm_hcu.hcu_ops  # noqa: F401
    h = helpers(attention)
    rows, width, k = batch * next_n, 2048, 512
    torch.manual_seed(42)
    logits = torch.randn(rows, width * 2, device="cuda")[:, ::2]
    lengths = torch.linspace(0, width, batch, device="cuda").int()
    backing = torch.full((rows + 1, k + 8), 991, device="cuda", dtype=torch.int32)
    out = backing[:rows, :k]
    h._lightop_topk_indices_decode(logits, lengths, next_n, out, k)
    torch.cuda.synchronize()
    ends = h._decode_row_ends_from_seq_lens(lengths, next_n, rows)
    expected = h._topk_indices_torch(logits, k, row_ends=ends)
    # Top-k order is unspecified, but the selected set and sentinel count are not.
    torch.testing.assert_close(out.sort(dim=-1).values, expected.sort(dim=-1).values)
    assert torch.all(backing[:rows, k:] == 991)
    assert torch.all(backing[rows] == 991)
