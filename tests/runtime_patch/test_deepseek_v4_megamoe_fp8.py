# SPDX-License-Identifier: Apache-2.0
"""Portable contracts for the DeepSeek-V4 MegaMoE FP8 path."""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm_hcu.patch.worker.core_fix import patch_deepseek_v4_megamoe


REPO = Path(__file__).resolve().parents[2]


def test_megamoe_patch_request_is_explicit() -> None:
    config = SimpleNamespace(
        kernel_config=SimpleNamespace(moe_backend="deep_gemm_mega_moe")
    )
    assert patch_deepseek_v4_megamoe._requested(config)
    config.kernel_config.moe_backend = "triton"
    assert not patch_deepseek_v4_megamoe._requested(config)


def _load_experts_class():
    source_path = REPO / "vllm_hcu/models/deepseek_v4_megamoe.py"
    source = source_path.read_text()
    tree = ast.parse(source)
    node = next(
        item
        for item in tree.body
        if isinstance(item, ast.ClassDef)
        and item.name == "DeepseekV4MegaMoEFP8Experts"
    )
    module = ast.Module(body=[node], type_ignores=[])

    def set_weight_attrs(param, attrs):
        for name, value in attrs.items():
            setattr(param, name, value)

    namespace = {
        "nn": nn,
        "os": os,
        "SimpleNamespace": SimpleNamespace,
        "torch": torch,
        "VllmConfig": object,
        "current_platform": SimpleNamespace(supports_fp8=lambda: True),
        "set_weight_attrs": set_weight_attrs,
    }
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["DeepseekV4MegaMoEFP8Experts"]


@pytest.fixture
def fp8_experts():
    cls = _load_experts_class()
    config = SimpleNamespace(
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    return cls(
        config,
        num_experts=2,
        num_local_experts=2,
        experts_start_idx=0,
        top_k=1,
        hidden_size=128,
        intermediate_size=128,
        prefix="model.layers.0.ffn.experts",
    )


def test_fp8_initialization_and_weight_loading(fp8_experts) -> None:
    assert fp8_experts.w13_weight.shape == (2, 256, 128)
    assert fp8_experts.w2_weight.shape == (2, 128, 128)
    assert fp8_experts.w13_weight.dtype == torch.float8_e4m3fn
    assert fp8_experts.w13_weight_scale.shape == (2, 256, 1)
    assert fp8_experts.w13_weight_scale.dtype == torch.float32
    assert fp8_experts.w13_weight_scale.quant_method == "channel"

    w1 = torch.full((128, 128), 2.0, dtype=torch.float8_e4m3fn)
    scale = torch.full((128, 1), 0.25, dtype=torch.float32)
    assert fp8_experts.weight_loader(
        fp8_experts.w13_weight,
        w1,
        "w13_weight",
        "w1",
        1,
        return_success=True,
    )
    assert fp8_experts.weight_loader(
        fp8_experts.w13_weight_scale,
        scale,
        "w13_weight_scale",
        "w1",
        1,
        return_success=True,
    )
    assert torch.equal(fp8_experts.w13_weight[1, :128].float(), w1.float())
    assert torch.equal(fp8_experts.w13_weight_scale[1, :128], scale)


def test_fp8_finalize_and_forward_dispatch_never_touch_fp4(
    fp8_experts, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    torch.manual_seed(7)
    fp8_experts.w13_weight.data.copy_(
        (torch.randn(fp8_experts.w13_weight.shape) * 2).to(torch.float8_e4m3fn)
    )
    fp8_experts.w2_weight.data.copy_(
        (torch.randn(fp8_experts.w2_weight.shape) * 2).to(torch.float8_e4m3fn)
    )
    fp8_experts.w13_weight_scale.data.fill_(0.01)
    fp8_experts.w2_weight_scale.data.fill_(0.01)

    class Buffer:
        x = torch.empty((8, 128), dtype=torch.float8_e4m3fn)
        x_sf = torch.empty((8, 1), dtype=torch.float32)
        topk_idx = torch.empty((8, 1), dtype=torch.int32)
        topk_weights = torch.empty((8, 1), dtype=torch.float32)

    megamoe = ModuleType("megamoe")
    megamoe.flatten_pack5_weight = lambda weight: weight

    def cast_to_fp8_channelwise(value):
        scale = value.float().abs().amax(dim=-1, keepdim=True).clamp_min(1e-6) / 448
        return (value.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn), scale

    def pre_dispatch(x, ids, weights, out_x, out_scale, out_ids, out_weights, num_tokens):
        quantized, scale = cast_to_fp8_channelwise(x)
        out_x[:num_tokens].copy_(quantized)
        out_scale[:num_tokens].copy_(scale)
        out_ids[:num_tokens].copy_(ids.to(out_ids.dtype))
        out_weights[:num_tokens].copy_(weights.to(out_weights.dtype))

    megamoe.mega_moe_pre_dispatch = pre_dispatch

    def fp8_kernel(out, l1, l2, buf, **kwargs):
        calls.append("fp8_w8a8")
        assert kwargs["recipe"] == (1, 1, 32)
        assert kwargs["activation"] == "swiglu"
        x = buf.x[: out.shape[0]].float() * buf.x_sf[: out.shape[0]]
        l1 = l1["unified"]
        l2 = l2["unified"]
        w13 = l1[0].float() * l1[1].unsqueeze(-1)
        w2 = l2[0].float() * l2[1].unsqueeze(-1)
        gate_up = x @ w13[0].T
        routed = torch.nn.functional.silu(gate_up[:, :128]) * gate_up[:, 128:]
        out.copy_((routed @ w2[0].T).to(out.dtype))

    megamoe.fp8_w8a8_mega_moe = fp8_kernel
    monkeypatch.setitem(sys.modules, megamoe.__name__, megamoe)
    monkeypatch.setattr(fp8_experts, "_check_runtime_supported", lambda: None)
    monkeypatch.setattr(fp8_experts, "get_symm_buffer", Buffer)

    fp8_experts.finalize_weights()
    assert fp8_experts._transformed_l1_weights["unified"][1].shape == (2, 256)
    assert fp8_experts._transformed_l2_weights["unified"][1].shape == (2, 128)

    hidden = torch.randn((3, 128), dtype=torch.bfloat16)
    w13 = fp8_experts._transformed_l1_weights
    w2 = fp8_experts._transformed_l2_weights
    w13 = w13["unified"]
    w2 = w2["unified"]
    w13_ref = w13[0].float() * w13[1].unsqueeze(-1)
    w2_ref = w2[0].float() * w2[1].unsqueeze(-1)
    gate_up_ref = hidden.float() @ w13_ref[0].T
    reference = (
        torch.nn.functional.silu(gate_up_ref[:, :128])
        * gate_up_ref[:, 128:]
    ) @ w2_ref[0].T
    out = torch.empty_like(hidden)
    fp8_experts._run(
        hidden,
        torch.ones((3, 1), dtype=torch.float32),
        torch.zeros((3, 1), dtype=torch.int64),
        out,
        None,
        True,
    )
    assert calls == ["fp8_w8a8"]
    error = (out.float() - reference).abs().mean()
    reference_magnitude = reference.abs().mean().clamp_min(1e-6)
    assert error / reference_magnitude < 0.05
