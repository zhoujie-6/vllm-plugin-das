# SPDX-License-Identifier: Apache-2.0
import types
import pytest
import torch
from vllm_hcu.utils import compressor_gemm as ops


@pytest.fixture(autouse=True)
def setup(monkeypatch):
    monkeypatch.setattr(ops, "_BLASLT_FAILED", False)
    monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "1")
    monkeypatch.setenv("VLLM_USE_LINEAR_BF16_FP32_USE_BLASLT", "1")
    monkeypatch.setattr(ops, "_eligible", lambda x, w: True)
    def mm(x, w, *, out_dtype):
        assert out_dtype == torch.float32
        return x.float() @ w.float()
    monkeypatch.setattr(torch, "mm", mm)


@pytest.mark.parametrize("layout", ["nn", "nt"])
@pytest.mark.parametrize("backend", ["torch", "torch_nt", "blaslt"])
def test_outputs_and_dispatch(monkeypatch, layout, backend):
    monkeypatch.setenv("VLLM_HCU_COMPRESSOR_GEMM_BACKEND", backend)
    calls = []
    def blas(x, w):
        assert w.is_contiguous() and w.shape == (3, 5)
        calls.append(w)
        return x.float() @ w.float().T
    monkeypatch.setattr(ops.bf16_fp32_gemm, "_jit_hipblaslt_bf16_fp32", lambda: types.SimpleNamespace(linear_bf16_fp32=blas))
    x = torch.randn(7, 5, dtype=torch.bfloat16)
    nn = torch.randn(5, 3, dtype=torch.bfloat16)
    w = nn if layout == "nn" else nn.T.contiguous()
    result = ops.compressor_mm(x, w)
    torch.testing.assert_close(result, x.float() @ nn.float())
    assert result.dtype == torch.float32
    assert len(calls) == (backend == "blaslt")


def test_cache_reuse_and_reload():
    w = torch.randn(5, 3, dtype=torch.bfloat16)
    first = ops._cached_nt(w, True)
    assert ops._cached_nt(w, True) is first
    w.add_(1)
    second = ops._cached_nt(w, True)
    assert second is not first
    torch.testing.assert_close(second, w.T)


@pytest.mark.parametrize("disabled", ["master", "blaslt", "ineligible"])
def test_original_route_when_disabled(monkeypatch, disabled):
    monkeypatch.setenv("VLLM_HCU_COMPRESSOR_GEMM_BACKEND", "blaslt")
    if disabled == "master":
        monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_OPS", "0")
    elif disabled == "blaslt":
        monkeypatch.setenv("VLLM_USE_LINEAR_BF16_FP32_USE_BLASLT", "0")
    else:
        monkeypatch.setattr(ops, "_eligible", lambda x, w: False)
    monkeypatch.setattr(ops, "_cached_nt", lambda *a: pytest.fail("must preserve original route"))
    x, w = torch.randn(2, 5), torch.randn(5, 3)
    torch.testing.assert_close(ops.compressor_mm(x, w), x @ w)


def test_blaslt_failure_returns_valid_reference(monkeypatch):
    monkeypatch.setenv("VLLM_HCU_COMPRESSOR_GEMM_BACKEND", "blaslt")
    calls = []
    def fail():
        calls.append(1)
        raise RuntimeError("unsupported BLASLt shape")
    monkeypatch.setattr(ops.bf16_fp32_gemm, "_jit_hipblaslt_bf16_fp32", fail)
    x, w = torch.randn(2, 5), torch.randn(5, 3)
    for _ in range(2):
        torch.testing.assert_close(ops.compressor_mm(x, w), x @ w)
    assert calls == [1]


def test_invalid_backend(monkeypatch):
    monkeypatch.setenv("VLLM_HCU_COMPRESSOR_GEMM_BACKEND", "invalid")
    with pytest.raises(ValueError, match="BACKEND"):
        ops.compressor_mm(torch.ones(2, 5), torch.ones(5, 3))
