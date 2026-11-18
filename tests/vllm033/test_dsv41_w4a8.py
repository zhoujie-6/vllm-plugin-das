"""CPU contracts for DSV4.1 PCP + channel W4A8."""
from types import SimpleNamespace as NS

import pytest
import torch

from hcu_vllm_das.moe import dsv41_w4a8_megamoe as w4
from hcu_vllm_das.quantization import dsv41_channel_w4a8 as cfg


def test_ct_nibbles_across_int32_sign_bit():
    word = torch.tensor([0x76543210, -0x1234568], dtype=torch.int32)
    assert w4.unpack_ct_words(word).tolist() == [
        8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7]


@pytest.mark.parametrize("rows", [0, 1, 5])
def test_uneven_local_batch_launches_clamped_w4a8(monkeypatch, rows):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    q = torch.arange(rows * 8, dtype=torch.int8).view(rows, 8)
    sf = torch.arange(1, rows + 1, dtype=torch.float).view(rows, 1)
    def quantize(x):
        assert x.shape == (rows, 8) and rows
        return q, sf
    monkeypatch.setattr(w4, "per_token_quant_int8", quantize)
    x = torch.zeros(rows, 8, dtype=torch.bfloat16)
    ids = torch.arange(rows * 2).reshape(rows, 2) % 4
    weights = torch.full((rows, 2), .25)
    calls = []
    ex = object.__new__(w4.W4A8MegaExperts)
    ex.moe_config = NS(hidden_dim=8, experts_per_token=2)
    ex.quant_config = NS(gemm1_clamp_limit=10.)
    ex.l1 = ex.l2 = None
    ex.buffer = NS(x=torch.zeros(6, 8, dtype=torch.int8), x_sf=torch.zeros(6),
                   num_max_tokens_per_rank=6, topk_idx=torch.full((6, 2), -1),
                   topk_weights=torch.zeros(6, 2))
    ex.mega = NS(fp8_mega_moe=lambda *a, **kw: calls.append((a, kw)))
    out = torch.empty_like(x)
    ex.run(out, x, ids, weights)
    assert len(calls) == 1
    args, kw = calls[0]
    assert args[0] is out and args[3] is ex.buffer
    assert kw == dict(mode="w4a8", activation_clamp=10., w4a8_scales_predivided=True)
    assert torch.equal(ex.buffer.x[:rows], q)
    assert torch.equal(ex.buffer.x_sf[:rows], sf.flatten())
    assert torch.equal(ex.buffer.topk_idx[:rows], ids)
    assert torch.equal(ex.buffer.topk_weights[:rows], weights)
    assert torch.all(ex.buffer.topk_idx[rows:] == -1)


def test_schema_requires_symmetric_token_and_channel():
    w = dict(type="int", num_bits=4, symmetric=True, strategy="channel", group_size=-1)
    a = dict(type="int", num_bits=8, symmetric=True, strategy="token", dynamic=True)
    s = dict(format="pack-quantized", weights=w, input_activations=a)
    assert cfg._channel_w4a8(s)
    assert not cfg._channel_w4a8(dict(s, weights=dict(w, strategy="group")))
    assert not cfg._channel_w4a8(dict(s, input_activations=dict(a, symmetric=False)))
    assert not cfg._channel_w4a8(dict(s, format="int-quantized"))



@pytest.mark.parametrize("backend", ["deep_gemm_mega_moe", "deep_gemm"])
def test_registered_ct_config_routes_pcp8_w4a8(monkeypatch, backend):
    from vllm.model_executor.layers.fused_moe import RoutedExperts
    from hcu_vllm_das import ct_w4a8_activation
    from hcu_vllm_das.quantization import ct_w4a8
    from hcu_vllm_das.moe import dsv41_w4a8_deepgemm as grouped

    monkeypatch.setattr(ct_w4a8_activation, "check_health", lambda: None)
    config = object.__new__(ct_w4a8.HcuDSV41CTW4A8Config)
    config.target_scheme_map = {}
    weights = dict(type="int", num_bits=4, symmetric=True, strategy="channel")
    activations = dict(type="int", num_bits=8, symmetric=True, strategy="token", dynamic=True)
    scheme = dict(weights=weights, input_activations=activations, format="pack-quantized")
    monkeypatch.setattr(config, "get_scheme_dict", lambda *args: scheme)
    layer = object.__new__(RoutedExperts)
    torch.nn.Module.__init__(layer)
    layer.moe_config = NS(moe_backend=backend, pcp_size=8, ep_size=8)
    layer.ckpt_gate_proj_name = "gate_proj"
    layer.ckpt_up_proj_name = "up_proj"
    layer.ckpt_down_proj_name = "down_proj"
    calls = []
    expected = object()
    def construct(*args):
        calls.append(args)
        return expected
    owner, name = ((w4, "HcuDSV41W4A8MegaMoEMethod") if backend == "deep_gemm_mega_moe"
                   else (grouped, "HcuDSV41W4A8DeepGemmMethod"))
    monkeypatch.setattr(owner, name, construct)
    assert config.get_quant_method(layer, "model.layers.0.mlp.experts") is expected
    assert calls == [(weights, activations, layer.moe_config, "model.layers.0.mlp.experts")]
    assert cfg.HcuDSV41ChannelW4A8Config is ct_w4a8.HcuDSV41CTW4A8Config


@pytest.mark.parametrize("backend", ["deep_gemm_mega_moe", "deep_gemm"])
def test_w4a8_keeps_local_pcp_profile(backend):
    from hcu_vllm_das.dsv41.worker import uses_pcp_channel_fp8
    config = NS(parallel_config=NS(
        tensor_parallel_size=1, data_parallel_size=1, pipeline_parallel_size=1,
        prefill_context_parallel_size=8, decode_context_parallel_size=1,
        enable_expert_parallel=True),
        model_config=NS(quantization="hcu_dsv41_ct_w4a8",
                        hf_config=NS(model_type="deepseek_v41")),
        kernel_config=NS(moe_backend=backend))
    assert uses_pcp_channel_fp8(config)
    config.parallel_config.prefill_context_parallel_size = 1
    assert not uses_pcp_channel_fp8(config)
