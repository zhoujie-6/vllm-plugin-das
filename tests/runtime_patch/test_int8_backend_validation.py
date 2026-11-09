"""Reject native GEMMs that claim success but leave invalid output."""
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch


@pytest.mark.parametrize("mode,expected", [
    ("correct", True), ("nan", False), ("wrong", False),
    ("status", False), ("exception", False),
])
def test_native_int8_validation(monkeypatch, mode, expected):
    path = Path(__file__).resolve().parents[2] / (
        "vllm_hcu/model_executor/layers/quantization/int8_runtime.py"
    )
    spec = importlib.util.spec_from_file_location("_int8_validation_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []

    def gemm(a, b, sa, sb, m, n, k, layout, dtype):
        calls.append(layout)
        if mode == "exception":
            raise RuntimeError("no native algorithm")
        out = ((a.float() * sa) @ (b.float() * sb).T).to(dtype)
        if mode == "nan":
            out.fill_(float("nan"))
        if mode == "wrong":
            out.zero_()
        return mode != "status", out

    package = ModuleType("lightop")
    package.__path__ = []
    gemm_ops = ModuleType("lightop.gemm_ops")
    gemm_ops.hipblaslt_w8a8_gemm = gemm
    monkeypatch.setitem(sys.modules, "lightop", package)
    monkeypatch.setitem(sys.modules, "lightop.gemm_ops", gemm_ops)
    for _ in range(2):
        assert module.int8_hipblaslt_is_usable(
            torch.device("cpu"), torch.bfloat16
        ) is expected
    assert calls == ["NT"]
