# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace as NS

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]


def load_file(monkeypatch, name, path):
    spec = importlib.util.spec_from_file_location(name, REPO / path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def fake_deepgemm():
    fake = ModuleType("deepgemm")
    fake.transform_weights_for_mega_moe = lambda a, b: (
        tuple(t.clone() for t in a), tuple(t.clone() for t in b))
    fake.fp8_mega_moe = lambda *args, **kwargs: None
    fake.get_symm_buffer_for_mega_moe = lambda *args, **kwargs: NS()
    def preprocess(x, ids, weights, buf, rows, is_int8=False):
        buf.x[:rows].copy_(x)
        buf.x_sf[:rows].fill_(2)
        buf.topk_idx[:rows].copy_(ids)
        buf.topk_weights[:rows].copy_(weights)
    fake.mega_moe_preprocess = preprocess
    return fake


@pytest.fixture
def runtime(monkeypatch):
    monkeypatch.setitem(sys.modules, "deepgemm", fake_deepgemm())
    class Base:
        def __init__(self, moe):
            self.moe = moe
            self.moe_kernel = self.moe_quant_config = None

    deps = {
        "vllm.config": {"get_current_vllm_config": lambda: None},
        "vllm.distributed": {"get_ep_group": lambda: NS(device_group=object())},
        "vllm.forward_context": {"get_forward_context": lambda: NS(dp_metadata=None)},
        "vllm.model_executor.layers.fused_moe.fused_moe_method_base": {"FusedMoEMethodBase": Base},
    }
    for name, symbols in deps.items():
        mod = ModuleType(name)
        mod.__dict__.update(symbols)
        monkeypatch.setitem(sys.modules, name, mod)
    return load_file(monkeypatch, "test_megamoe_runtime",
                     "vllm_hcu/model_executor/layers/fused_moe/megamoe_runtime.py")


@pytest.fixture
def int4(monkeypatch):
    import triton
    import triton.language as tl
    mod = ModuleType("vllm.triton_utils")
    mod.triton, mod.tl = triton, tl
    monkeypatch.setitem(sys.modules, mod.__name__, mod)
    return load_file(monkeypatch, "test_megamoe_int4",
                     "vllm_hcu/model_executor/layers/fused_moe/hcu_int4_megamoe.py")


@pytest.mark.parametrize("dtype", [torch.uint8, torch.int8])
def test_int4_preparation_retains_weights_and_scales(int4, dtype):
    e, h, i = 2, 256, 256
    w1 = torch.arange(e * 2 * i * h // 2).to(dtype).reshape(e, 2*i, h//2)
    w2 = torch.arange(e * h * i // 2).to(dtype).reshape(e, h, i//2)
    s1, s2 = torch.rand(e, 2*i, 1), torch.rand(e, h, 1)
    l1, l2 = int4.transform_int4_weights_for_mega_moe_normal(
        w1, w2, l1_scale=s1, l2_scale=s2, prepack=False)
    for weight, scale, prepared in ((w1, s1, l1), (w2, s2, l2)):
        values, scales = prepared["raw_int4"]
        assert values is weight  # No duplicate resident weights or shuffle.
        torch.testing.assert_close(scales, scale.squeeze(-1)*16)
        assert scales.dtype == torch.float32
    _, true_scales = int4._pack(w1, s1, 1.0, prepack=False)["raw_int4"]
    torch.testing.assert_close(true_scales, s1.squeeze(-1))


@pytest.mark.parametrize("factor", [0, -1, float("inf"), float("nan")])
def test_int4_rejects_invalid_scale(int4, factor):
    w = torch.zeros((1, 256, 128), dtype=torch.int8)
    with pytest.raises(ValueError, match="scale_multiplier"):
        int4.transform_int4_weights_for_mega_moe_normal(w, w, l1_scale=None,
                                                     l2_scale=None, scale_multiplier=factor)


def valid_config():
    config = NS(parallel_config=NS(enable_expert_parallel=True, enable_eplb=False,
                                   enable_dbo=False, tensor_parallel_size=1),
                model_config=NS(enforce_eager=True, hf_config=NS(model_type="deepseek_v4")))
    moe = NS(ep_size=8, moe_backend="deep_gemm_mega_moe", pcp_size=1, is_sequence_parallel=False,
             num_experts=256, num_local_experts=32, hidden_dim=4096,
             intermediate_size_per_partition=2048, num_fused_shared_experts=0,
             has_bias=False, is_lora_enabled=False, activation=NS(value="silu"),
             in_dtype=torch.bfloat16, skip_final_all_reduce=False,
             experts_per_token=6, max_num_tokens=16, max_capture_size=8, dp_size=1, dp_rank=0)
    layer = NS(scoring_func="sqrtsoftplus", expert_placement_strategy="linear", swiglu_limit=10.,
               expert_map_manager=NS(num_fused_shared_experts=0))
    return config, moe, layer


@pytest.mark.parametrize("field,value,match", [
    ("ep_size", 4, "EP8"),
    ("is_sequence_parallel", True, "sequence"),
    ("skip_final_all_reduce", True, "defer"),
    ("in_dtype", torch.float16, "bfloat16"),
    ("moe_backend", "triton", "deep_gemm_mega_moe"),
    ("moe_backend", "auto", "deep_gemm_mega_moe"),
])
def test_config_rejects_unsupported_modes(runtime, field, value, match):
    config, moe, layer = valid_config()
    runtime.validate_config(config, moe, layer)
    moe.moe_backend = "deep_gemm_mega_moe"
    runtime.validate_config(config, moe, layer)
    setattr(moe, field, value)
    with pytest.raises(ValueError, match=match):
        runtime.validate_config(config, moe, layer)


def test_config_rejects_graphs_and_round_robin(runtime):
    config, moe, layer = valid_config()
    config.model_config.enforce_eager = False
    with pytest.raises(ValueError, match="enforce-eager"):
        runtime.validate_config(config, moe, layer, "int8")
    config.model_config.enforce_eager = True
    layer.expert_placement_strategy = "round_robin"
    with pytest.raises(ValueError, match="linear expert"):
        runtime.validate_config(config, moe, layer)


def test_config_rejects_original_tp_before_replicated_routes_are_staged(runtime):
    config, moe, layer = valid_config()
    config.parallel_config.tensor_parallel_size = 8
    # EP has already rewritten the MoE-local TP topology; validation must use
    # the original ParallelConfig or 16 * top-6 routes would be staged 8 times.
    moe.tp_size = 1
    with pytest.raises(ValueError, match="tensor parallelism"):
        runtime.validate_config(config, moe, layer)


def test_fp8_idle_dp_rank_participates_and_clears_stale_routes(runtime, monkeypatch):
    _, moe, layer = valid_config()
    method = runtime.HcuMegaMoEMethod(moe, "fp8")
    buf = NS(topk_idx=torch.ones(16, 6, dtype=torch.int64),
             topk_weights=torch.ones(16, 6), x=torch.empty(16, 4096),
             x_sf=torch.empty(16))
    method._buffer = buf
    method._l1, method._l2 = object(), object()
    calls = []
    fake = fake_deepgemm()
    fake.fp8_mega_moe = lambda *args, **kwargs: calls.append((args, kwargs))
    fake.cast_to_fp8_channelwise = lambda x: (x, torch.ones(x.shape[0]))
    monkeypatch.setitem(sys.modules, "deepgemm", fake)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(runtime, "get_forward_context", lambda: NS(
        dp_metadata=NS(num_tokens_across_dp_cpu=torch.tensor([0, 7]))))
    x = torch.empty(0, 4096, dtype=torch.bfloat16)
    result = method.apply(layer, x, torch.empty(0, 6), torch.empty(0, 6, dtype=torch.int64))
    assert result.shape == x.shape
    assert len(calls) == 1
    assert calls[0][0][0].shape[0] == 0
    assert calls[0][1]["activation_clamp"] == 10.
    assert (buf.topk_idx == -1).all() and (buf.topk_weights == 0).all()

    monkeypatch.setattr(runtime, "get_forward_context", lambda: NS(dp_metadata=None))
    method.apply(layer, x, torch.empty(0, 6), torch.empty(0, 6, dtype=torch.int64))
    assert len(calls) == 1  # All ranks empty: no collective required.


def test_pcp_defensively_ignores_injected_dp_metadata(
    runtime, monkeypatch, context=None,
):
    _, moe, layer = valid_config()
    moe.pcp_size = 8
    method = runtime.HcuMegaMoEMethod(moe, "fp8")
    buf = NS(
        topk_idx=torch.ones(16, 6, dtype=torch.int64),
        topk_weights=torch.ones(16, 6),
        x=torch.empty(16, 4096),
        x_sf=torch.empty(16),
    )
    method._buffer, method._l1, method._l2 = buf, {}, {}
    calls = []
    fake = fake_deepgemm()
    fake.cast_to_fp8_channelwise = lambda x: (
        x.float(), torch.ones(x.shape[0], 1)
    )
    fake.fp8_mega_moe = (
        lambda *args, **kwargs: calls.append((args, kwargs))
    )
    monkeypatch.setitem(sys.modules, "deepgemm", fake)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    # Synthetic defensive input: the supported TP1/DP1/PCP8 builders do not
    # create DP metadata, even when passed an unsplit token count.
    monkeypatch.setattr(runtime, "get_forward_context", lambda: context if context is not None else NS(
        dp_metadata=NS(num_tokens_across_dp_cpu=torch.tensor([16]))
    ))
    x = torch.ones(2, 4096, dtype=torch.bfloat16)
    method.apply(
        layer, x, torch.ones(2, 6), torch.zeros(2, 6, dtype=torch.int64)
    )

    assert len(calls) == 1
    assert calls[0][0][0].shape[0] == 2
    assert "megamoe_backend" not in calls[0][1]


def test_fp8_stages_routes_and_scales(runtime, monkeypatch):
    _, moe, layer = valid_config()
    method = runtime.HcuMegaMoEMethod(moe, "fp8")
    buf = NS(topk_idx=torch.ones(16, 6, dtype=torch.int64),
             topk_weights=torch.ones(16, 6), x=torch.empty(16, 4096), x_sf=torch.empty(16))
    method._buffer, method._l1, method._l2 = buf, {}, {}
    fake = fake_deepgemm()
    fake.cast_to_fp8_channelwise = lambda x: (x.float(), torch.full((x.shape[0], 1), 2.))
    def run(y, *args, **kwargs):
        y.fill_(3)
    fake.fp8_mega_moe = run
    monkeypatch.setitem(sys.modules, "deepgemm", fake)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    x = torch.ones(2, 4096, dtype=torch.bfloat16)
    ids = torch.arange(12).reshape(2, 6)
    weights = torch.full((2, 6), .125)
    result = method.apply(layer, x, weights, ids)
    torch.testing.assert_close(buf.x[:2], x.float())
    torch.testing.assert_close(buf.topk_idx[:2], ids)
    torch.testing.assert_close(buf.topk_weights[:2], weights)
    assert (buf.x_sf[:2] == 2).all() and (result == 3).all()
    assert (buf.topk_idx[2:] == -1).all()
    with pytest.raises(ValueError, match="capacity"):
        method.apply(layer, torch.ones(17, 4096, dtype=torch.bfloat16),
                     torch.ones(17, 6), torch.ones(17, 6, dtype=torch.int64))


def test_chunk_limit_keeps_k1_strides_in_uint32(runtime):
    _, moe, _ = valid_config()
    moe.max_num_tokens = 32768
    limit = runtime.max_chunk_tokens(moe, "normal")
    rows = lambda t: runtime._k1_rows("normal", t, 8, 6, 32)
    assert rows(32768) * 4096 > 2**32 - 1  # The DSV4-Flash profile-run failure.
    assert rows(limit) * 4096 <= 2**32 - 1 < rows(limit + 1) * 4096
    assert 16384 < limit < 32768
    ll = runtime.max_chunk_tokens(moe, "ll")
    assert runtime._k1_rows("ll", ll, 8, 6, 32) * 4096 <= 2**32 - 1
    moe.max_num_tokens = 16
    assert runtime.max_chunk_tokens(moe, "normal") == 16


def test_fp8_large_batch_is_chunked_on_every_dp_rank(runtime, monkeypatch):
    _, moe, layer = valid_config()
    method = runtime.HcuMegaMoEMethod(moe, "fp8")
    method._max_chunk_tokens = 4
    buf = NS(topk_idx=torch.ones(16, 6, dtype=torch.int64),
             topk_weights=torch.ones(16, 6), x=torch.empty(16, 4096), x_sf=torch.empty(16))
    method._buffer, method._l1, method._l2 = buf, {}, {}
    calls = []
    fake = fake_deepgemm()
    fake.cast_to_fp8_channelwise = lambda x: (x.float(), torch.ones(x.shape[0], 1))
    def run(y, *args, **kwargs):
        m = y.shape[0]
        calls.append((m, buf.topk_idx[:, 0].clone()))
        y.copy_(buf.x[:m])
    fake.fp8_mega_moe = run
    monkeypatch.setitem(sys.modules, "deepgemm", fake)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(runtime, "get_forward_context", lambda: NS(
        dp_metadata=NS(num_tokens_across_dp_cpu=torch.tensor([6, 10]))))
    x = torch.arange(6, dtype=torch.bfloat16)[:, None].expand(6, 4096).contiguous()
    ids = torch.arange(6)[:, None].expand(6, 6).contiguous()
    result = method.apply(layer, x, torch.ones(6, 6), ids)
    torch.testing.assert_close(result, x)
    # Shared capacity 10 -> 4+4+2 calls; this rank has 4, 2, then 0 tokens.
    assert [m for m, _ in calls] == [4, 2, 0]
    assert calls[1][1][:3].tolist() == [4, 5, -1]  # Stale routes cleared.
    assert (calls[2][1] == -1).all()


@pytest.mark.parametrize("inference", [False, True])
def test_post_load_reuses_buffer_and_invalidates_weight_cache(runtime, monkeypatch, inference):
    _, moe, _ = valid_config()
    import torch.nn as nn
    # Tiny CPU shapes exercise lifecycle without allocating a real model.
    def layer():
        return NS(w13_weight=nn.Parameter(torch.zeros(2, 8, 4, dtype=torch.float8_e4m3fn), requires_grad=False),
                  w2_weight=nn.Parameter(torch.zeros(2, 4, 4, dtype=torch.float8_e4m3fn), requires_grad=False),
                  w13_weight_scale=nn.Parameter(torch.ones(2, 8, 1), requires_grad=False),
                  w2_weight_scale=nn.Parameter(torch.ones(2, 4, 1), requires_grad=False))
    fake = fake_deepgemm()
    calls = []
    fake.flatten_pack5_weight_asm_normal = lambda w: w.clone()
    fake.flatten_pack5_weight = lambda w: w.clone()
    fake.cast_to_fp8_channelwise = lambda x: (x, torch.ones(x.shape[0]))
    fake.fp8_mega_moe = lambda *args, **kwargs: None
    def factory(*args, **kwargs):
        calls.append(kwargs)
        return object()
    fake.get_symm_buffer_for_mega_moe = factory
    monkeypatch.setitem(sys.modules, "deepgemm", fake)
    group = object()
    ep = NS(device_group=group)
    monkeypatch.setattr(runtime, "get_ep_group", lambda: ep)
    monkeypatch.setattr(runtime.HcuMegaMoEMethod, "get_fused_moe_quant_config", lambda *args: object())
    a, b = runtime.HcuMegaMoEMethod(moe, "fp8"), runtime.HcuMegaMoEMethod(moe, "fp8")
    with torch.inference_mode(inference):
        weights = layer()
    a.process_weights_after_loading(weights)
    original = a._l1
    addresses = [t.data_ptr() for packed in (a._l1, a._l2)
                 for t in packed]
    # Real loaders write via .data, leaving Parameter version counters unchanged.
    version = None if inference else weights.w13_weight._version
    weights.w13_weight.data.copy_(torch.ones_like(weights.w13_weight))
    weights.w2_weight.data.copy_(torch.ones_like(weights.w2_weight))
    weights.w13_weight_scale.data.copy_(torch.full_like(weights.w13_weight_scale, 3))
    if not inference:
        assert weights.w13_weight._version == version
    a.process_weights_after_loading(weights)
    for packed in (a._l1, a._l2):
        assert (packed[0].float() == 1).all()
    assert (a._l1[1] == 3).all()
    assert addresses == [t.data_ptr() for packed in (a._l1, a._l2)
                         for t in packed]
    a.process_weights_after_loading(weights)
    assert a._l1 is original
    # Explicit post-load refresh also covers ordinary in-place scale updates.
    with torch.inference_mode(inference), torch.no_grad():
        weights.w13_weight_scale.fill_(2)
    a.process_weights_after_loading(weights)
    assert a._l1 is original  # Graph-safe reload preserves captured addresses.
    assert (a._l1[1] == 2).all()
    b.process_weights_after_loading(layer())
    assert a._buffer is b._buffer and len(calls) == 1


def test_opt_in_patch_preserves_default_and_is_idempotent(monkeypatch):
    from vllm_hcu.patch.worker.core_fix import patch_deepseek_v4_megamoe as patch
    calls = []
    class RoutedExperts:
        def _get_quant_method(self, prefix, quant_config, moe_config):
            calls.append((prefix, quant_config, moe_config))
            return "existing"
    module = ModuleType(patch.TARGET_MODULE)
    module.RoutedExperts = RoutedExperts
    monkeypatch.delenv("VLLM_HCU_USE_MEGAMOE", raising=False)
    assert patch.apply_to_module(module)
    assert not patch.apply_to_module(module)
    instance = RoutedExperts()
    assert instance._get_quant_method("model.layers.0.experts", None, None) == "existing"
    assert len(calls) == 1
    fake = ModuleType("vllm_hcu.model_executor.layers.fused_moe.megamoe_runtime")
    fake.select_method = lambda *args: "mega"
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    monkeypatch.setenv("VLLM_HCU_USE_MEGAMOE", "1")
    # A stale legacy variable must not override explicit backend selection.
    for backend in ("auto", "triton", "aiter", "deep_gemm"):
        assert instance._get_quant_method(
            "model.layers.0.experts", None, NS(moe_backend=backend)) == "existing"
    assert len(calls) == 5
    # Only the CLI backend opts in, even if the old variable says disabled.
    monkeypatch.setenv("VLLM_HCU_USE_MEGAMOE", "0")
    mega = NS(moe_backend="deep_gemm_mega_moe")
    assert instance._get_quant_method("model.layers.0.experts", None, mega) == "mega"
    fake.select_method = lambda *args: None
    with pytest.raises(ValueError, match="DeepSeek-V4"):
        instance._get_quant_method("model.layers.0.experts", None, mega)
    assert instance._get_quant_method(
        "model.layers.0.experts", None, NS(moe_backend="auto")) == "existing"
    RoutedExperts._get_quant_method = lambda *args: None
    with pytest.raises(RuntimeError, match="stale"):
        patch.apply_to_module(module)


def test_runner_avoids_duplicate_collectives_for_megamoe():
    # Load the real small runner methods without importing accelerator ops.
    import ast
    path = REPO / "vllm_hcu/model_executor/layers/fused_moe/moe_runner.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MoERunner")
    names = {"_fused_output_is_reduced", "do_naive_dispatch_combine",
             "_maybe_reduce_shared_expert_output", "_maybe_reduce_final_output"}
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    reduced = []
    def reduce(x):
        reduced.append(x)
        return x * 8
    namespace = {"torch": torch, "tensor_model_parallel_all_reduce": reduce}
    test_cls = ast.ClassDef(name="Runner", bases=[], keywords=[], body=methods, decorator_list=[])
    code = ast.fix_missing_locations(ast.Module(body=[test_cls], type_ignores=[]))
    exec(compile(code, str(path), "exec"), namespace)
    runner = namespace["Runner"]()
    runner._quant_method = NS(moe_kernel=None, hcu_owns_dispatch=True, hcu_output_is_reduced=True)
    runner.moe_config = NS(dp_size=2, tp_size=1, ep_size=8, is_sequence_parallel=False,
                           skip_final_all_reduce=False)
    assert not runner.do_naive_dispatch_combine
    shared = runner._maybe_reduce_shared_expert_output(torch.ones(2, 4))
    final = runner._maybe_reduce_final_output(torch.ones(2, 4)+shared, None)
    assert len(reduced) == 1 and (final == 9).all()
    runner._quant_method = NS(moe_kernel=None, supports_internal_mk=False)
    assert runner.do_naive_dispatch_combine
    assert not runner._fused_output_is_reduced
    shared = runner._maybe_reduce_shared_expert_output(torch.ones(2, 4))
    assert (shared == 1).all()
    runner._maybe_reduce_final_output(shared, None)
    assert len(reduced) == 2


def test_config_rejects_fused_shared_experts(runtime):
    config, moe, layer = valid_config()
    layer.expert_map_manager.num_fused_shared_experts = 1
    with pytest.raises(ValueError, match="fused shared"):
        runtime.validate_config(config, moe, layer)




def test_selects_slimquant_before_fallback_construction(runtime, monkeypatch):
    config, moe, layer = valid_config()
    layer.layer_name = "model.layers.60.ffn.experts"
    monkeypatch.setattr(runtime, "get_current_vllm_config", lambda: config)
    utils = ModuleType("vllm.model_executor.layers.quantization.compressed_tensors.utils")
    utils.should_ignore_layer = lambda *args, **kwargs: False
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    quant = NS(get_name=lambda: "slimquant_w4a8", ignore=[], packed_modules_mapping={})
    method = runtime.select_method(layer, quant, moe)
    assert method.quant_mode == "int8" and method.topk_indices_dtype == torch.int64
    utils.should_ignore_layer = lambda *args, **kwargs: True
    with pytest.raises(ValueError, match="ignored"):
        runtime.select_method(layer, quant, moe)


def test_selects_channel_fp8_and_rejects_mismatched_projections(runtime, monkeypatch):
    from enum import Enum
    class Strategy(Enum):
        CHANNEL = "channel"
        TOKEN = "token"
    quant_module = ModuleType("compressed_tensors.quantization")
    quant_module.QuantizationStrategy = Strategy
    monkeypatch.setitem(sys.modules, quant_module.__name__, quant_module)
    config, moe, layer = valid_config()
    layer.layer_name = "model.layers.0.ffn.experts"
    monkeypatch.setattr(runtime, "get_current_vllm_config", lambda: config)
    weight = NS(strategy=Strategy.CHANNEL, symmetric=True, dynamic=False)
    activation = NS(strategy=Strategy.TOKEN, dynamic=True, symmetric=True)
    scheme = {"weights": weight, "input_activations": activation}
    names = []
    def get_scheme(layer, name):
        names.append(name)
        return scheme
    utils = ModuleType("vllm.model_executor.layers.quantization.compressed_tensors.utils")
    utils.should_ignore_layer = lambda *args, **kwargs: False
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    quant = NS(get_name=lambda: "compressed-tensors", ignore=[], packed_modules_mapping={},
               _add_fused_moe_to_target_scheme_map=lambda: None,
               get_scheme_dict=get_scheme, _is_fp8_w8a8=lambda *args: True)
    method = runtime.select_method(layer, quant, moe)
    assert method.quant_mode == "fp8" and method.weight_quant is weight
    assert set(names) == {layer.layer_name+f".0.{p}_proj" for p in ("gate", "up", "down")}
    activation.dynamic = False
    with pytest.raises(ValueError, match="Channel-FP8"):
        runtime.select_method(layer, quant, moe)
    quant.get_scheme_dict = lambda layer, name: scheme if "gate" in name else None
    with pytest.raises(ValueError, match="matching"):
        runtime.select_method(layer, quant, moe)




def test_pcp_requires_dp1_and_eager(runtime, monkeypatch):
    config, moe, layer = valid_config()
    moe.pcp_size = 2
    runtime.validate_config(config, moe, layer)
    moe.dp_size = 2
    with pytest.raises(ValueError, match="PCP requires DP1"):
        runtime.validate_config(config, moe, layer)
    moe.dp_size = 1
    config.model_config.enforce_eager = False
    with pytest.raises(ValueError, match="PCP requires DP1"):
        runtime.validate_config(config, moe, layer)


def test_dp_requires_exact_coordinated_counts(runtime, monkeypatch):
    _, moe, layer = valid_config()
    moe.dp_size = 2
    method = runtime.HcuMegaMoEMethod(moe, "fp8")
    method._buffer = object()
    monkeypatch.setitem(sys.modules, "deepgemm", fake_deepgemm())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    x = torch.ones(3, 4096, dtype=torch.bfloat16)
    weights, ids = torch.ones(3, 6), torch.zeros(3, 6, dtype=torch.int64)
    with pytest.raises(ValueError, match="coordinated token metadata"):
        method.apply(layer, x, weights, ids)
    monkeypatch.setattr(runtime, "get_forward_context", lambda: NS(
        dp_metadata=NS(num_tokens_across_dp_cpu=torch.tensor([2, 4]))))
    with pytest.raises(ValueError, match="disagree"):
        method.apply(layer, x, weights, ids)


def test_raw_int4_transform_retains_only_canonical_weights(int4):
    w1 = torch.zeros((2, 512, 128), dtype=torch.int8)
    w2 = torch.zeros((2, 256, 128), dtype=torch.int8)
    l1, l2 = int4.transform_int4_weights_for_mega_moe_normal(
        w1, w2, l1_scale=torch.ones(2, 512, 1),
        l2_scale=torch.ones(2, 256, 1), prepack=False,
    )
    assert l1["raw_int4"][0] is w1 and l2["raw_int4"][0] is w2
    assert (l1["raw_int4"][1] == 16).all()


def test_release_buffers_destroys_once_and_drops_scratch(runtime, monkeypatch):
    events = []
    class Buffer:
        def __init__(self):
            self._int4_unpack_workspaces = {"scratch": torch.ones(4)}
        def destroy(self):
            events.append("destroy")
    buf = Buffer()
    ep = NS(_hcu_megamoe_buffers={"int8": buf})
    monkeypatch.setattr(runtime, "get_ep_group", lambda: ep)
    runtime.release_buffers()
    runtime.release_buffers()
    assert events == ["destroy"]
    assert not ep._hcu_megamoe_buffers and not buf._int4_unpack_workspaces


def test_other_models_bypass_feature_even_when_global_switch_is_on(runtime, monkeypatch):
    config, moe, layer = valid_config()
    config.model_config.hf_config.model_type = "qwen3_moe"
    monkeypatch.setattr(runtime, "get_current_vllm_config", lambda: config)
    assert runtime.select_method(layer, None, moe) is None








def test_pcp_runner_keeps_rank_local_tokens_for_megamoe():
    import ast
    path = REPO / "vllm_hcu/model_executor/layers/fused_moe/moe_runner.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MoERunner")
    names = {"_maybe_dispatch", "_maybe_combine", "do_naive_dispatch_combine",
             "_pcp_routing_inputs"}
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names]
    calls = []
    group = NS(all_gather=lambda x, **kw: calls.append("gather") or x,
               reduce_scatter=lambda x, **kw: calls.append("scatter") or x)
    from contextlib import contextmanager
    padding = torch.tensor([False, True])
    context = NS(is_padding=padding)
    namespace = {"torch": torch, "get_pcp_group": lambda: group,
                 "contextmanager": contextmanager,
                 "get_forward_context": lambda: context,
                 "is_forward_context_available": lambda: True}
    test_cls = ast.ClassDef(name="Runner", bases=[], keywords=[], body=methods, decorator_list=[])
    code = ast.fix_missing_locations(ast.Module(body=[test_cls], type_ignores=[]))
    exec(compile(code, str(path), "exec"), namespace)
    runner = namespace["Runner"]()
    runner.shared_experts = None
    runner._ep_fused_combine = False
    runner.moe_config = NS(pcp_size=2, dp_size=1, is_sequence_parallel=False,
                           moe_parallel_config=NS(use_all2all_kernels=False))
    runner._quant_method = NS(hcu_owns_dispatch=True)
    x, logits = torch.ones(2, 4), torch.ones(2, 256)
    assert runner._maybe_dispatch(x, logits) == (x, logits)
    assert runner._maybe_combine(None, x) is x
    ids = torch.tensor([17, 23])
    for _ in range(2):
        with runner._pcp_routing_inputs(ids, 2) as routing_ids:
            assert routing_ids is ids
            assert context.is_padding is padding
        assert context.is_padding is padding
    assert calls == []
    runner._quant_method = NS(supports_internal_mk=False)
    runner._maybe_dispatch(x, logits)
    runner._maybe_combine(None, x)
    with runner._pcp_routing_inputs(ids, 1) as routing_ids:
        torch.testing.assert_close(routing_ids, ids[:1])
        torch.testing.assert_close(context.is_padding, padding[:1])
    assert context.is_padding is padding
    assert calls == ["gather", "gather", "scatter", "gather", "gather"]


def test_offload_sleep_and_nonstandard_activation_rejected(runtime):
    config, moe, layer = valid_config()
    config.offload_config = NS(uva=NS(cpu_offload_gb=1), prefetch=NS(offload_group_size=0))
    with pytest.raises(ValueError, match="offloading"):
        runtime.validate_config(config, moe, layer)
    config.offload_config.uva.cpu_offload_gb = 0
    config.model_config.enable_sleep_mode = True
    with pytest.raises(ValueError, match="sleep mode"):
        runtime.validate_config(config, moe, layer)
    config.model_config.enable_sleep_mode = False
    layer.swiglu_alpha = 2.
    with pytest.raises(ValueError, match="parameterized"):
        runtime.validate_config(config, moe, layer)






@pytest.mark.parametrize("owns_dispatch,expected", [(True, False), (False, True)])
def test_ep_fused_combine_excludes_backend_owned_dispatch(owns_dispatch, expected):
    import ast
    path = REPO / "vllm_hcu/model_executor/layers/fused_moe/moe_runner.py"
    tree = ast.parse(path.read_text())
    assignment = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Attribute) and target.attr == "_ep_fused_combine"
            for target in node.targets)
    )
    runner = NS(_quant_method=NS(hcu_owns_dispatch=owns_dispatch))
    namespace = dict(
        self=runner, _EP_FUSED_COMBINE=True, shared_experts=object(),
        moe_config=NS(dp_size=8, pcp_size=1, is_sequence_parallel=False,
                      skip_final_all_reduce=False),
        routed_scaling_factor=1., routed_input_transform=None,
        routed_output_transform=None,
    )
    exec(compile(ast.fix_missing_locations(ast.Module(body=[assignment], type_ignores=[])),
                 str(path), "exec"), namespace)
    assert runner._ep_fused_combine is expected






def test_signed_int4_triton_unpack_on_hcu(int4):
    try:
        torch.cuda.init()
    except RuntimeError as exc:
        pytest.skip(f"requires an HCU to execute the unpack kernel: {exc}")
    packed = torch.arange(256).to(torch.uint8).to("cuda")
    out = torch.empty(512, dtype=torch.int8, device="cuda")
    int4.unpack_signed_int4_out(packed, out)
    raw = torch.arange(256, dtype=torch.int16)
    reference = ((torch.stack((raw >> 4, raw & 15), -1) ^ 8)-8).to(torch.int8)
    torch.testing.assert_close(out.cpu(), reference.flatten())


@pytest.mark.parametrize("name", ["compressed-tensors", "slimquant_compressed_tensors_marlin"])
def test_fp8_ignored_experts_rejected_before_scheme_lookup(runtime, monkeypatch, name):
    config, moe, layer = valid_config()
    layer.layer_name = "model.layers.0.ffn.experts"
    monkeypatch.setattr(runtime, "get_current_vllm_config", lambda: config)
    ignore = ["model.layers.*.ffn.experts"]
    mapping = {"gate_up_proj": ["gate_proj", "up_proj"]}
    utils = ModuleType("vllm.model_executor.layers.quantization.compressed_tensors.utils")
    def should_ignore(layer_name, **kwargs):
        assert layer_name == layer.layer_name
        assert kwargs == dict(ignore=ignore, fused_mapping=mapping, use_fnmatch=True)
        return True
    utils.should_ignore_layer = should_ignore
    monkeypatch.setitem(sys.modules, utils.__name__, utils)
    def unexpected(*args):
        pytest.fail("Ignored experts must be rejected before FP8 scheme lookup/construction")
    quant = NS(get_name=lambda: name, ignore=ignore, packed_modules_mapping=mapping,
               _add_fused_moe_to_target_scheme_map=unexpected, get_scheme_dict=unexpected)
    monkeypatch.setattr(runtime, "HcuMegaMoEMethod", unexpected)
    with pytest.raises(ValueError, match="ignored/unquantized"):
        runtime.select_method(layer, quant, moe)


def test_int4_native_layerwise_reload_releases_temporary_weights(runtime, int4, monkeypatch):
    import gc
    import logging
    import weakref
    import importlib

    # Execute the installed native reload helpers, stubbing only platform imports.
    reload_dir = next((Path(p) / "vllm/model_executor/model_loader/reload"
                       for p in sys.path
                       if (Path(p) / "vllm/model_executor/model_loader/reload/layerwise.py").is_file()), None)
    if reload_dir is None:
        pytest.skip("requires native vLLM reload helpers")
    package = ModuleType("test_native_megamoe_reload")
    package.__path__ = [str(reload_dir)]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    monkeypatch.setattr(sys.modules["vllm.config"], "ModelConfig", object, raising=False)
    def weight_loader(param, loaded_weight):
        param.data.copy_(loaded_weight)
    deps = {
        "vllm.logger": {"init_logger": logging.getLogger},
        "vllm.model_executor.layers.attention": {"is_deferred_attention_layer": lambda layer: False},
        "vllm.model_executor.layers.quantization.base_config": {"QuantizeMethodBase": runtime.FusedMoEMethodBase},
        "vllm.model_executor.model_loader.weight_utils": {"default_weight_loader": weight_loader},
    }
    for name, symbols in deps.items():
        mod = ModuleType(name)
        mod.__dict__.update(symbols)
        monkeypatch.setitem(sys.modules, name, mod)
    native = importlib.import_module(package.__name__ + ".layerwise")
    monkeypatch.setattr(runtime, "__package__", package.__name__)
    monkeypatch.setitem(sys.modules, package.__name__ + ".hcu_int4_megamoe", int4)
    fake = fake_deepgemm()
    fake.get_symm_buffer_for_mega_moe = lambda *args, **kwargs: NS(quant_mode="int8")
    monkeypatch.setitem(sys.modules, "deepgemm", fake)
    monkeypatch.setattr(int4, "validate_megamoe_int8_runtime", lambda: None)
    monkeypatch.setattr(runtime, "get_ep_group", lambda: NS(device_group=object()))
    monkeypatch.setattr(runtime.HcuMegaMoEMethod, "get_fused_moe_quant_config", lambda *args: None)
    _, moe, _ = valid_config()
    method = runtime.HcuMegaMoEMethod(moe, "int8")
    layer = torch.nn.Module()
    for name, shape in (("w13_weight", (2, 512, 128)), ("w2_weight", (2, 256, 128)),
                        ("w13_weight_scale", (2, 512, 1)), ("w2_weight_scale", (2, 256, 1))):
        param = torch.nn.Parameter(torch.ones(shape, dtype=torch.int8 if "scale" not in name else torch.float32),
                                   requires_grad=False)
        param.weight_loader = weight_loader
        layer.register_parameter(name, param)
    del param
    layer.quant_method = method
    native.record_metadata_for_reloading(layer)
    method.process_weights_after_loading(layer)
    canonical = dict(layer.named_parameters())
    temporary = []
    transform = int4.transform_int4_weights_for_mega_moe_normal
    def track(w1, w2, **kwargs):
        temporary.extend((weakref.ref(w1), weakref.ref(w2)))
        return transform(w1, w2, **kwargs)
    monkeypatch.setattr(int4, "transform_int4_weights_for_mega_moe_normal", track)
    for value in (2, 3):
        native.initialize_layerwise_reload(layer)
        for name in canonical:
            target = getattr(layer, name)
            target.weight_loader(target, torch.full_like(canonical[name], value))
        del target
        native.finalize_layerwise_reload(layer, None)
        gc.collect()
        assert all(ref() is None for ref in temporary)
        for name, param in canonical.items():
            assert getattr(layer, name) is param
            assert (param == value).all()
        assert method._l1 is None and method._l2 is None
        assert all((scale == value * 16).all() for scale in method._int4_scales)

    # Forward must resolve the restored canonical weights, not the temporaries.
    layer.swiglu_limit = None
    method._buffer.topk_idx = torch.zeros(16, 6, dtype=torch.int64)
    method._buffer.topk_weights = torch.zeros(16, 6)
    method._buffer.x = torch.empty(16, 4096)
    method._buffer.x_sf = torch.empty(16)
    fake.mega_moe_preprocess = lambda *args, **kwargs: None
    seen = []
    def run(y, l1, l2, *args, **kwargs):
        assert l1["raw_int4"][0] is canonical["w13_weight"]
        assert l2["raw_int4"][0] is canonical["w2_weight"]
        seen.append(True)
    monkeypatch.setattr(int4, "int4_w4a8_mega_moe", run)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(runtime, "get_forward_context", lambda: NS(dp_metadata=None))
    method.apply(layer, torch.ones(1, 4096, dtype=torch.bfloat16),
                 torch.ones(1, 6), torch.zeros(1, 6, dtype=torch.int64))
    assert seen == [True]




@pytest.mark.parametrize("dp_size,backend,auto,rejected", [
    (2, "deepep_low_latency", False, True),
    (1, "deepep_low_latency", False, False),
    (2, "deepep_low_latency", True, False),
    (2, "deepep_high_throughput", False, False),
])
def test_config_rejects_dp_backend_without_token_metadata(runtime, dp_size, backend, auto, rejected):
    config, moe, layer = valid_config()
    moe.dp_size = dp_size
    config.parallel_config.all2all_backend = backend
    config.parallel_config._vllm_hcu_deepep_auto = auto
    if rejected:
        with pytest.raises(ValueError, match="deepep_low_latency"):
            runtime.validate_config(config, moe, layer)
    else:
        runtime.validate_config(config, moe, layer)




def _native_context_symbols():
    """Execute installed source definitions without GPU/platform import effects.

    Keep the original builder, DPMetadata.make and ParallelConfig property;
    only context storage and platform hooks are replaced below.
    """
    import ast
    from contextlib import contextmanager
    from dataclasses import dataclass

    native = next((Path(p) / "vllm/forward_context.py" for p in sys.path
                   if (Path(p) / "vllm/forward_context.py").is_file()), None)
    if native is None:
        pytest.skip("requires installed native vLLM context source")
    tree = ast.parse(native.read_text())
    definitions = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                   and n.name in {"DPMetadata", "set_forward_context"}]
    assert len(definitions) == 2
    module = ModuleType("test_native_context_source")
    module.__dict__.update(torch=torch, contextmanager=contextmanager,
                           dataclass=dataclass, CUDAGraphMode=NS(NONE=0))
    # Postpone annotations just as an isolated source loader would.
    code = ast.Module(body=[ast.ImportFrom(module="__future__",
                      names=[ast.alias(name="annotations")], level=0), *definitions],
                      type_ignores=[])
    sys.modules[module.__name__] = module
    try:
        exec(compile(ast.fix_missing_locations(code), str(native), "exec"), module.__dict__)
    finally:
        sys.modules.pop(module.__name__, None)
    parallel = ast.parse((native.parent / "config/parallel.py").read_text())
    cls = next(n for n in parallel.body if isinstance(n, ast.ClassDef)
               and n.name == "ParallelConfig")
    prop = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                and n.name == "use_sequence_parallel_moe")
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[prop], type_ignores=[])),
                 str(native.parent / "config/parallel.py"), "exec"), namespace)
    return module, namespace["use_sequence_parallel_moe"]


@pytest.mark.parametrize("builder", ["native", "hcu"])
@pytest.mark.parametrize("dp_size,global_tokens", [(1, None), (1, [16]), (2, [2, 7])])
def test_pcp_metadata_from_actual_context_builders(runtime, monkeypatch, builder,
                                                   dp_size, global_tokens):
    from contextlib import contextmanager

    module, sp_property = _native_context_symbols()
    parallel = NS(tensor_parallel_size=1, data_parallel_size=dp_size,
                  data_parallel_rank=0, prefill_context_parallel_size=8,
                  enable_expert_parallel=True, is_moe_model=True,
                  all2all_backend="allgather_reducescatter")
    parallel.use_sequence_parallel_moe = sp_property.fget(parallel)
    assert parallel.use_sequence_parallel_moe is False
    config = NS(parallel_config=parallel)
    contexts = []

    @contextmanager
    def override(context):
        contexts.append(context)
        try:
            yield
        finally:
            contexts.pop()

    module.track_batchsize = False
    module.current_platform = NS(set_additional_forward_context=lambda **kw: {})
    module.create_forward_context = lambda *args, **kw: NS(dp_metadata=args[2])
    module.override_forward_context = override
    tokens = None if global_tokens is None else torch.tensor(global_tokens, dtype=torch.int32)
    if builder == "hcu":
        hcu = load_file(monkeypatch, "test_hcu_context_source",
                        "vllm_hcu/forward_context_runtime.py")
        scope = hcu.set_forward_context(module, None, config, num_tokens=2,
                                       num_tokens_across_dp=tokens)
    else:
        scope = module.set_forward_context(None, config, num_tokens=2,
                                          num_tokens_across_dp=tokens)
    with scope:
        context = contexts[-1]
        if dp_size == 1:
            assert context.dp_metadata is None
            test_pcp_defensively_ignores_injected_dp_metadata(runtime, monkeypatch, context)
        else:
            # Control for the actual creation branch; DP2/PCP8 is not a
            # supported MegaMoE configuration and must not be sent to apply.
            assert context.dp_metadata.num_tokens_across_dp_cpu.tolist() == [2, 7]
    assert contexts == []


@pytest.mark.parametrize("interleave", [False, True])
@pytest.mark.parametrize("e,n,k", [(1,32,64),(2,256,128),(2,512,256)])
@pytest.mark.parametrize("dtype", [torch.uint8, torch.int8])
def test_raw_w6_kernel_matches_independent_layout(int4, interleave, e, n, k, dtype):
    from triton.runtime.interpreter import InterpretedFunction
    raw = torch.arange(e*n*k//2).to(torch.uint8).view(dtype).reshape(e,n,k//2)
    out = torch.empty((e,k//64,n//16,4,16,16), dtype=torch.int8)
    kernel = InterpretedFunction(int4._unpack_raw_int4_w6.fn)
    kernel[(int4.triton.cdiv(out.numel(),2048),)](
        raw,out,out.numel(),n,k,interleave,2048,num_warps=4)
    byte = raw.to(torch.int16)
    ref = (((torch.stack(((byte >> 4)&15, byte&15),-1)^8)-8)
           .to(torch.int8).reshape(e,n,k))
    if interleave:
        ref = torch.stack((ref[:,:n//2].reshape(e,n//16,8,k),
                           ref[:,n//2:].reshape(e,n//16,8,k)),2).reshape(e,n,k)
    ref = ref.reshape(e,n//16,16,k//64,4,16).permute(0,3,1,4,2,5).contiguous()
    torch.testing.assert_close(out,ref)


@pytest.mark.parametrize("capturing", [False,True])
def test_deepgemm_graph_uses_padded_output_and_clears_routes(runtime,monkeypatch,capturing):
    _,moe,layer=valid_config()
    method=runtime.HcuMegaMoEMethod(moe,"fp8",graph_tokens=8)
    buf=NS(topk_idx=torch.ones(16,6,dtype=torch.int64),topk_weights=torch.ones(16,6),
           x=torch.empty(16,4096),x_sf=torch.empty(16))
    method._buffer,method._l1,method._l2=buf,(),()
    fake=sys.modules["deepgemm"]
    calls=[]
    def run(y,*args,**kwargs):
        calls.append((y.shape,kwargs,buf.topk_idx.clone()))
        y.fill_(4)
    fake.fp8_mega_moe=run
    monkeypatch.setattr(torch.cuda,"is_current_stream_capturing",lambda:capturing)
    monkeypatch.setattr(runtime,"get_forward_context",lambda:NS(
        dp_metadata=None,cudagraph_runtime_mode=NS(name="FULL")))
    for n in (6,2):
        x=torch.ones(n,4096,dtype=torch.bfloat16)
        y=method.apply(layer,x,torch.ones(n,6),torch.zeros(n,6,dtype=torch.int64))
        assert y.shape==x.shape and (y==4).all()
        shape,kwargs,ids=calls[-1]
        assert shape==(n,4096)
        assert kwargs==dict(activation_clamp=10.,fast_math=True)
        assert (ids[n:]==-1).all()


def test_deepgemm_graph_warmup_cached_and_does_not_need_old_token_buffer(runtime,monkeypatch):
    _,moe,layer=valid_config()
    method=runtime.HcuMegaMoEMethod(moe,"fp8",graph_tokens=8)
    buf=NS(topk_idx=torch.ones(16,6,dtype=torch.int64),topk_weights=torch.ones(16,6),
           x=torch.empty(16,4096),x_sf=torch.empty(16))
    method._buffer,method._l1,method._l2=buf,(),()
    calls=[]
    sys.modules["deepgemm"].fp8_mega_moe=lambda *args,**kwargs:calls.append(args)
    method._warmup_graph(layer)
    method._warmup_graph(layer)
    assert len(calls)==1 and calls[0][0].shape==(8,4096)
    assert (buf.topk_idx==-1).all()
    method._max_chunk_tokens=4
    with pytest.raises(ValueError,match="stride capacity"):
        method._warmup_graph(layer)


@pytest.mark.parametrize("missing",["deepgemm","deepgemm_native_dependency"])
def test_deepgemm_import_failure_is_actionable_and_preserves_inner_error(runtime,monkeypatch,missing):
    import builtins
    original=builtins.__import__
    def importing(name,*args,**kwargs):
        if name=="megamoe":
            pytest.fail("must never import standalone megamoe")
        if name=="deepgemm":
            raise ModuleNotFoundError(name=missing)
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,"__import__",importing)
    _,moe,_=valid_config()
    method=runtime.HcuMegaMoEMethod(moe,"fp8")
    if missing=="deepgemm":
        with pytest.raises(RuntimeError,match="installed deepgemm"):
            method.process_weights_after_loading(None)
    else:
        with pytest.raises(ModuleNotFoundError) as exc:
            method.process_weights_after_loading(None)
        assert exc.value.name==missing


def test_deepgemm_missing_mega_api_fails_before_weight_allocation(runtime,monkeypatch):
    _,moe,_=valid_config()
    del sys.modules["deepgemm"].mega_moe_preprocess
    with pytest.raises(RuntimeError,match="mega_moe_preprocess"):
        runtime.HcuMegaMoEMethod(moe,"fp8").process_weights_after_loading(None)


def test_w4a8_uses_deepgemm_int8_and_reuses_stream_scratch(int4,monkeypatch):
    fake=fake_deepgemm()
    calls=[]
    fake.fp8_mega_moe=lambda *a,**kw:calls.append((a,kw))
    monkeypatch.setitem(sys.modules,"deepgemm",fake)
    monkeypatch.setattr(torch.cuda,"is_current_stream_capturing",lambda:False)
    stream=NS(cuda_stream=1)
    monkeypatch.setattr(torch.cuda,"current_stream",lambda *a:stream)
    unpacked=[]
    def unpack(p,out,**kw):
        unpacked.append(kw.get("interleave",False))
        out.fill_(1)
    monkeypatch.setattr(int4,"unpack_raw_int4_w6_out",unpack)
    p1,p2=torch.zeros(2,512,128,dtype=torch.int8),torch.zeros(2,256,128,dtype=torch.int8)
    s1,s2=torch.arange(1024.).reshape(2,512),torch.ones(2,256)
    l1,l2={"raw_int4":(p1,s1)},{"raw_int4":(p2,s2)}
    y,buf=torch.empty(1,256,dtype=torch.bfloat16),NS()
    for _ in range(2):
        int4.int4_w4a8_mega_moe(y,l1,l2,buf,activation_clamp=10.)
    assert len(buf._int4_unpack_workspaces)==1
    assert calls[0][0][1][0] is calls[1][0][1][0]
    assert unpacked==[True,False,True,False]
    assert calls[0][1]==dict(is_int8=True,activation_clamp=10.)
    expected=torch.stack((s1[:,:256].reshape(2,32,8),s1[:,256:].reshape(2,32,8)),2).reshape(2,512)
    torch.testing.assert_close(calls[0][0][1][1],expected)
    def unavailable(*a,**kw):
        raise RuntimeError("MegaMoE INT8 kernels not compiled (DG_MEGA_BUILD_INT8=0)")
    fake.fp8_mega_moe=unavailable
    with pytest.raises(RuntimeError,match="DG_MEGA_BUILD_INT8=1") as exc:
        int4.int4_w4a8_mega_moe(y,l1,l2,buf)
    assert "INT8 kernels not compiled" in str(exc.value.__cause__)
    fake.fp8_mega_moe=lambda *a,**kw:None
    stream.cuda_stream=2
    int4.int4_w4a8_mega_moe(y,l1,l2,buf)
    assert len(buf._int4_unpack_workspaces)==2
    monkeypatch.setattr(torch.cuda,"is_current_stream_capturing",lambda:True)
    with pytest.raises(NotImplementedError,match="eager"):
        int4.int4_w4a8_mega_moe(y,l1,l2,buf)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires HCU")
@pytest.mark.parametrize("interleave",[False,True])
def test_raw_w6_gpu_matches_installed_deepgemm(int4,interleave):
    import deepgemm
    from deepgemm.m_group_gemm import pack_int8_weight_w6_ll
    # A few hundred KiB, no IPC allocation or model weights.
    e,n,k=2,256,128
    raw=torch.arange(e*n*k//2,device="cuda").to(torch.uint8).reshape(e,n,k//2)
    b=raw.to(torch.int16)
    unpacked=(((torch.stack(((b>>4)&15,b&15),-1)^8)-8).to(torch.int8).reshape(e,n,k))
    if interleave:
        from deepgemm.mega.mega_moe import interleave_gate_up_weights
        unpacked=interleave_gate_up_weights(unpacked)
    reference=pack_int8_weight_w6_ll(unpacked.contiguous().clone())
    out=torch.empty_like(reference)
    int4.unpack_raw_int4_w6_out(raw,out,interleave=interleave)
    torch.cuda.synchronize()
    torch.testing.assert_close(out,reference)
    assert "/site-packages/deepgemm/" in deepgemm.__file__


def test_runtime_has_no_standalone_megamoe_import():
    import ast
    for name in ("megamoe_runtime.py","hcu_int4_megamoe.py"):
        path=REPO/"vllm_hcu/model_executor/layers/fused_moe"/name
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node,ast.Import):
                assert all(n.name.split(".")[0]!="megamoe" for n in node.names)
            if isinstance(node,ast.ImportFrom):
                assert (node.module or "").split(".")[0]!="megamoe"


@pytest.mark.parametrize("model_type", ["deepseek_v4", "deepseek_v4_dspark"])
def test_pcp_dspark_megamoe_accepts_decode_graphs(runtime, model_type):
    config, moe, layer = valid_config()
    config.model_config.hf_config.model_type = model_type
    config.model_config.enforce_eager = False
    config.speculative_config = NS(method="dspark")
    moe.pcp_size = 8
    runtime.validate_config(config, moe, layer, "fp8")
    with pytest.raises(ValueError, match="enforce-eager"):
        runtime.validate_config(config, moe, layer, "int8")
    moe.dp_size = 2
    with pytest.raises(ValueError, match="DP1"):
        runtime.validate_config(config, moe, layer, "fp8")
