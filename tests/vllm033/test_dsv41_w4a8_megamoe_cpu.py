"""CPU contracts for the isolated PCP8 W4A8 MegaMoE provider."""
from types import SimpleNamespace as NS
import pytest
import torch
from hcu_vllm_das.moe import dsv41_w4a8_megamoe as w4

def test_ct_nibbles_across_int32_sign_bit(monkeypatch):
    import deepgemm
    monkeypatch.setattr(deepgemm, "pack_w4a8_moe_hipc_weight", lambda x: x)
    words = torch.tensor([0x76543210, -0x1234568], dtype=torch.int32)
    assert w4.pack_ct_hipc(words.view(1, 1, 2)).view(torch.uint8).flatten().tolist() == [
        0x89, 0xab, 0xcd, 0xef, 0x01, 0x23, 0x45, 0x67]


@pytest.mark.parametrize("rows", [0, 1, 5])
def test_uneven_local_batch_launches_clamped_w4a8(monkeypatch, rows):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    q = torch.arange(rows * 8, dtype=torch.int8).view(rows, 8)
    sf = torch.arange(1, rows + 1, dtype=torch.float).view(rows, 1)
    def stage(buf, x, ids, weights):
        assert x.shape == (rows, 8) and rows
        buf.x[:rows].copy_(q)
        buf.x_sf[:rows].copy_(sf.flatten())
        buf.topk_idx[:rows].copy_(ids)
        buf.topk_weights[:rows].copy_(weights)
    monkeypatch.setattr(w4, "stage_ct_w4a8", stage)
    x = torch.zeros(rows, 8, dtype=torch.bfloat16)
    ids = torch.arange(rows * 2).reshape(rows, 2) % 4
    weights = torch.full((rows, 2), .25)
    calls = []
    ex = object.__new__(w4.W4A8MegaExperts)
    ex.moe_config = NS(hidden_dim=8, experts_per_token=2)
    ex.quant_config = NS(gemm1_clamp_limit=10.)
    ex.l1 = ex.l2 = (object(), object())
    ex.buffer = NS(x=torch.zeros(6, 8, dtype=torch.int8), x_sf=torch.zeros(6),
                   num_max_tokens_per_rank=6, topk_idx=torch.full((6, 2), -1),
                   topk_weights=torch.zeros(6, 2), buffer=object(), local_buffer=object(),
                   handle=NS(buffer_ptrs=[1], signal_ptrs=[2]),
                   group=NS(rank=lambda: 0), num_experts=4, num_topk=2)
    ex.combine_groups = object()
    ex.mega = NS(w4a8=NS(w4a8_mega_moe=lambda *a, **kw: calls.append((a, kw))))
    out = torch.empty_like(x)
    ex.run(out, x, ids, weights)
    assert len(calls) == 1
    args, kw = calls[0]
    assert args[0].shape == out.shape and args[0].data_ptr() == out.data_ptr()
    assert args[1:] == (ex.l1, ex.l2, ex.buffer)
    assert kw == {"activation_clamp": 10., "combine_group_map": ex.combine_groups}
    assert torch.equal(ex.buffer.x[:rows], q)
    assert torch.equal(ex.buffer.x_sf[:rows], sf.flatten())
    assert torch.equal(ex.buffer.topk_idx[:rows], ids)
    assert torch.equal(ex.buffer.topk_weights[:rows], weights)
    assert torch.all(ex.buffer.topk_idx[rows:] == -1)


@pytest.mark.parametrize("available", [False, True])
def test_provider_requires_standalone_w4a8(monkeypatch, available):
    import sys
    from types import ModuleType
    package = ModuleType("megamoe")
    package.w4a8 = NS(w4a8_mega_moe=(lambda *args: None) if available else None)
    monkeypatch.setitem(sys.modules, "megamoe", package)
    if available:
        assert w4.w4a8_provider() is package
    else:
        with pytest.raises(RuntimeError, match="standalone"):
            w4.w4a8_provider()


def test_registered_ct_config_routes_w4a8_megamoe(monkeypatch):
    from vllm.model_executor.layers.fused_moe import RoutedExperts
    from hcu_vllm_das import ct_w4a8_activation
    from hcu_vllm_das.quantization import ct_w4a8
    monkeypatch.setattr(ct_w4a8_activation, "check_health", lambda: None)
    monkeypatch.setattr(ct_w4a8, "get_current_vllm_config",
                        lambda: NS(kernel_config=NS(moe_backend="deep_gemm_mega_moe")))
    config = object.__new__(ct_w4a8.HcuDSV41CTW4A8Config)
    config.target_scheme_map = {}
    weights = dict(type="int", num_bits=4, symmetric=True, strategy="channel")
    activations = dict(type="int", num_bits=8, symmetric=True, strategy="token", dynamic=True)
    scheme = dict(weights=weights, input_activations=activations, format="pack-quantized")
    monkeypatch.setattr(config, "get_scheme_dict", lambda *args: scheme)
    layer = object.__new__(RoutedExperts)
    torch.nn.Module.__init__(layer)
    layer.moe_config = NS(moe_backend="deep_gemm_mega_moe", pcp_size=8, ep_size=8)
    layer.ckpt_gate_proj_name, layer.ckpt_up_proj_name, layer.ckpt_down_proj_name = "gate_proj", "up_proj", "down_proj"
    calls = []
    expected = object()
    def construct(*args):
        calls.append(args)
        return expected
    monkeypatch.setattr(w4, "HcuDSV41W4A8MegaMoEMethod", construct)
    assert config.get_quant_method(layer, "model.layers.0.mlp.experts") is expected
    assert calls == [(weights, activations, layer.moe_config, "model.layers.0.mlp.experts")]
    scheme["output_activations"] = dict(num_bits=8)
    with pytest.raises(ValueError, match="plain channel"):
        config.get_quant_method(layer, "model.layers.0.mlp.experts")


def test_w4a8_static_eplb_requires_explicit_admission():
    from test_dsv41_megamoe_cpu import mega_config
    from hcu_vllm_das.moe.dsv41_megamoe import validate_config
    assert w4.HcuDSV41W4A8MegaMoEMethod.supports_static_eplb
    assert w4.HcuDSV41W4A8MegaMoEMethod.supports_swa_bounded_replay
    config, moe = mega_config()
    validate_config(config, moe)
    config.speculative_config = NS(method="dspark")
    with pytest.raises(ValueError, match="speculative"):
        validate_config(config, moe)
    config.speculative_config = None
    moe.num_experts = 416
    with pytest.raises(ValueError, match="expert counts"):
        validate_config(config, moe)


def test_w4a8_failed_reload_blocks_forward():
    method = object.__new__(w4.HcuDSV41W4A8MegaMoEMethod)
    method._processed = True
    method._failure = None
    with pytest.raises(RuntimeError, match="postprocessed twice"):
        method.process_weights_after_loading(None)
    with pytest.raises(RuntimeError, match="post-load failed"):
        method.apply(None, None, None, None, None, None)


@pytest.mark.parametrize("state", ["loading", "packing", "failed", "ready"])
def test_w4a8_requires_loaded_state_before_packing(state):
    method = object.__new__(w4.HcuDSV41W4A8MegaMoEMethod)
    method._processed = False
    method._failure = None
    method._packed = NS(state=state)
    with pytest.raises(RuntimeError, match="expected loaded"):
        method.process_weights_after_loading(None)
    assert method._packed.state == state
    assert not method._processed


def test_actual_provider_has_standalone_w4a8_entry():
    assert w4.w4a8_provider().__name__ == "megamoe"
    assert not torch.cuda.is_initialized()

@pytest.mark.parametrize("rows,expected", [(0, [0, 0, 0]), (6, [4, 2, 0]), (10, [4, 4, 2])])
def test_replay_tiles_follow_common_pcp_extent(monkeypatch, rows, expected):
    # Simulate different local lengths for the same scheduler step.
    monkeypatch.setattr(w4, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(w4, "get_forward_context", lambda: NS(batch_descriptor=NS(num_tokens=10)))
    seen = []
    def launch(self, out, x, ids, weights):
        seen.append(x.shape[0])
        out.copy_(x)
        assert ids[:, 0].tolist() == x[:, 0].tolist()
        assert weights[:, 0].tolist() == x[:, 0].tolist()
    monkeypatch.setattr(w4.MegaExperts, "run", launch)
    ex = object.__new__(w4.W4A8MegaExperts)
    ex.buffer = NS(num_max_tokens_per_rank=4)
    x = torch.arange(rows).reshape(-1, 1)
    out = torch.empty_like(x)
    ex.run(out, x, x, x)
    assert seen == expected
    assert torch.equal(out, x)


@pytest.mark.parametrize("shard,projection", [("w1", "gate"), ("w3", "up"), ("w2", "down")])
@pytest.mark.parametrize("missing", ["weight_packed", "weight_scale"])
def test_incomplete_expert_checkpoint(monkeypatch, shard, projection, missing):
    layer, method = _load_checkpoint(monkeypatch, omit=(2, shard, missing))
    assert layer.w13_weight_shape.tolist() == [[256, 512]] * 3
    assert layer.w2_weight_shape.tolist() == [[512, 256]] * 3
    before = [p.clone() for p in layer.parameters()]
    with pytest.raises(RuntimeError, match=rf"layers.7.*local expert 2, projection {projection}, missing {missing}"):
        method.process_weights_after_loading(layer)
    assert method._packed.state == "loaded" and not method._processed
    assert method.moe_kernel is None
    for original, param in zip(before, layer.parameters()):
        assert torch.equal(original.view(torch.uint8), param.view(torch.uint8))


def _load_checkpoint(monkeypatch, *, omit=None, static=False):
    from types import MethodType
    from contextlib import nullcontext
    from hcu_vllm_das import static_eplb_runtime as eplb
    from test_ct_w4a8_static_eplb import make_experts

    layer, _ = make_experts(rank=1)  # Physical IDs 3, 4, 5 map to local slots 0, 1, 2.
    method = object.__new__(w4.HcuDSV41W4A8MegaMoEMethod)
    method._prefix, method._processed, method._failure = "layers.7", False, None
    method.moe_kernel, method.buffer_capacity = None, 1
    layer.quant_method = method
    layer.apply_router_weight_on_input, layer.swiglu_limit = False, 10.0
    method.moe = NS(hidden_dim=512, intermediate_size_per_partition=256, ep_size=2,
                    num_logical_experts=4, num_experts=6)
    method.create_weights(layer, 3, 512, 256, torch.bfloat16, weight_loader=layer.weight_loader)
    if static:
        # Logical expert 1 has two replicas on this rank (physical 3 and 5).
        layer._hcu_static_eplb_row = [0, 2, 3, 1, 0, 1]
        for name, param in layer.named_parameters():
            original = eplb._loader_function(param.weight_loader, layer, name)
            param.weight_loader = MethodType(eplb._static_loader(original), layer)

    for expert in range(4 if static else 6):
        for shard, stem, n, k in [("w1", "w13", 256, 512), ("w3", "w13", 256, 512),
                                  ("w2", "w2", 512, 256)]:
            value = (expert + int(shard[1])) % 8
            for field, data in [
                ("weight_shape", torch.tensor([n, k])),
                ("weight_packed", torch.full((n, k // 8), value * 0x11111111, dtype=torch.int32)),
                ("weight_scale", torch.full((n, 1), float(value))),
            ]:
                name = f"{stem}_{field}"
                param = getattr(layer, name)
                if omit == (2, shard, field):
                    if not static and expert == 5:
                        continue
                    if static and expert == 1:
                        # The fan-out still reports success through physical ID 3.
                        mapping = layer.expert_map_manager.map_global_to_local
                        with monkeypatch.context() as patch:
                            patch.setattr(layer.expert_map_manager, "map_global_to_local",
                                          lambda physical: -1 if physical == 5 else mapping(physical))
                            assert param.weight_loader(param, data, name, shard, expert, True)
                        continue
                param.weight_loader(param, data, name, shard_id=shard, expert_id=expert)

    import deepgemm
    monkeypatch.setattr(deepgemm, "pack_w4a8_moe_hipc_weight", lambda data: data)
    method.mega = NS()
    monkeypatch.setattr(w4, "get_ep_group", lambda: NS(world_size=2, rank_in_group=1))
    monkeypatch.setattr(w4, "_layer_buffer", lambda *args: nullcontext(None))
    monkeypatch.setattr(w4, "W4A8MegaExperts", lambda moe, quant, buf, l1, l2, mega: NS(l1=l1, l2=l2))
    monkeypatch.setattr(w4, "MegaKernel", lambda prepare, experts: experts)
    return layer, method


def test_static_fanout_requires_every_replica(monkeypatch):
    layer, method = _load_checkpoint(monkeypatch, static=True, omit=(2, "w1", "weight_scale"))
    with pytest.raises(RuntimeError, match="local expert 2, projection gate, missing weight_scale"):
        method.process_weights_after_loading(layer)
    assert method.moe_kernel is None


@pytest.mark.parametrize("static,experts", [(False, [3, 4, 5]), (True, [1, 0, 1])])
def test_complete_checkpoint_packs_local_experts(monkeypatch, static, experts):
    layer, method = _load_checkpoint(monkeypatch, static=static)
    method.process_weights_after_loading(layer)
    assert method._packed.state == "ready" and method._processed
    (w13, s13), (w2, s2) = method.moe_kernel.l1, method.moe_kernel.l2
    if static:
        assert method.moe_kernel.combine_groups.tolist() == [0, 1, 1, 0, 0, 0]
    else:
        assert method.moe_kernel.combine_groups is None
    for slot, expert in enumerate(experts):
        for weights, scales, offset in [(w13[slot, :256], s13[slot, :256], 1),
                                        (w13[slot, 256:], s13[slot, 256:], 3),
                                        (w2[slot], s2[slot], 2)]:
            value = (expert + offset) % 8
            assert torch.all(weights.view(torch.uint8) == (value ^ 8) * 0x11)
            assert torch.all(scales == value)


def test_full_local_tensor_load_fills_all_expert_slots(monkeypatch):
    layer, method = _load_checkpoint(monkeypatch, omit=(2, "w2", "weight_scale"))
    scales = layer.w2_weight_scale
    rows = torch.arange(1, 4, dtype=scales.dtype).view(3, 1, 1).expand_as(scales)
    assert scales.weight_loader(scales, rows, "w2_weight_scale", "w2", 3, True)
    method.process_weights_after_loading(layer)
    _, packed_scales = method.moe_kernel.l2
    torch.testing.assert_close(packed_scales, rows.squeeze(-1))


@pytest.mark.parametrize("extent,local,short", [(128, 128, True), (128, 0, True),
                                                (257, 8, False), (None, 8, False)])
def test_short_w4a8_tile_uses_common_extent(monkeypatch, extent, local, short):
    monkeypatch.setattr(w4, "is_forward_context_available", lambda: extent is not None)
    monkeypatch.setattr(w4, "get_forward_context", lambda: NS(batch_descriptor=NS(num_tokens=extent)))
    selected = []
    monkeypatch.setattr(w4.MegaExperts, "run", lambda self, *args: selected.append(self))
    ex = object.__new__(w4.W4A8MegaExperts)
    ex.buffer = NS(num_max_tokens_per_rank=4352)
    ex.short_tile = object.__new__(w4._W4A8MegaTile)
    ex.short_tile.buffer = NS(num_max_tokens_per_rank=256)
    x = torch.empty((local, 8))
    ex.run(x, x, x, x)
    assert selected == [ex.short_tile if short else ex]
