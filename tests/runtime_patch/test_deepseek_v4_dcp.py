# SPDX-License-Identifier: Apache-2.0

import importlib.util
import ast
import functools
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch
import pytest


def _load_source(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parents[2] / relative_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _identity_jit(function=None, **_kwargs):
    return function if function is not None else (lambda fn: fn)


_stub_names = ("vllm", "vllm.config", "vllm.distributed", "vllm.triton_utils")
_saved_modules = {name: sys.modules.get(name) for name in _stub_names}
try:
    vllm_config = ModuleType("vllm.config")
    vllm_config.VllmConfig = object
    vllm_distributed = ModuleType("vllm.distributed")
    vllm_distributed.get_dcp_group = lambda: None
    vllm_triton = ModuleType("vllm.triton_utils")
    vllm_triton.triton = SimpleNamespace(jit=_identity_jit)
    vllm_triton.tl = SimpleNamespace(constexpr=object())
    sys.modules["vllm"] = ModuleType("vllm")
    sys.modules["vllm.config"] = vllm_config
    sys.modules["vllm.distributed"] = vllm_distributed
    sys.modules["vllm.triton_utils"] = vllm_triton

    ContextParallelLayout = _load_source(
        "_test_hcu_cp_layout", "vllm_hcu/v1/cp_layout.py"
    ).ContextParallelLayout
    dcp_softmax_reduce = _load_source(
        "_test_hcu_dsv4_dcp", "vllm_hcu/v1/attention/ops/deepseek_v4_ops/dcp.py"
    ).dcp_softmax_reduce
finally:
    for _name, _module in _saved_modules.items():
        if _module is None:
            sys.modules.pop(_name, None)
        else:
            sys.modules[_name] = _module


def test_context_parallel_layout_interleaved_ownership_and_lengths():
    layouts = [ContextParallelLayout(2, rank, 2) for rank in range(2)]
    positions = torch.arange(12)
    assert layouts[0].owns(positions).tolist() == [
        True, True, False, False, True, True, False, False, True, True, False, False
    ]
    assert layouts[1].owns(positions).logical_not().equal(layouts[0].owns(positions))
    assert layouts[0].global_to_local(torch.tensor([0, 1, 2, 3, 4, 8])).tolist() == [
        0, 1, 2, 2, 2, 4
    ]
    assert layouts[1].global_to_local(torch.tensor([0, 1, 2, 3, 4, 8])).tolist() == [
        0, 0, 0, 1, 2, 4
    ]


def test_wrapped_sliding_caches_keep_replicated_slot_mappings():
    source = Path(__file__).parents[2] / "vllm_hcu/v1/hcu_model_runner.py"
    tree = ast.parse(source.read_text())
    method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                  and n.name == "may_reinitialize_input_batch")
    kinds = SimpleNamespace(MAMBA="mamba", SLIDING_WINDOW="swa",
                            SLIDING_WINDOW_MLA="swa_mla")
    # DeepSeek-V4 groups use UniformTypeKVCacheSpecs, whose outer class is
    # not a SlidingWindowSpec. The registry resolves its inner cache kind.
    specs = [SimpleNamespace(block_size=256, inner_kind=kind)
             for kind in ("swa", "swa_mla", "mla")]
    tables = [SimpleNamespace(dcp_world_size=2, dcp_rank=1) for _ in specs]
    runner = SimpleNamespace(
        max_model_len=16384, max_encoder_len=0,
        vllm_config=SimpleNamespace(parallel_config=SimpleNamespace(
            decode_context_parallel_size=2, prefill_context_parallel_size=1)),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
        _init_block_sizes=[256]*3, _init_kernel_block_sizes=[256]*3,
        input_batch=SimpleNamespace(block_table=SimpleNamespace(block_tables=tables)),
    )
    namespace = dict(KVCacheConfig=object, EncoderOnlyAttentionSpec=type("Encoder", (), {}),
                     MambaSpec=type("Mamba", (), {}), KVCacheSpecKind=kinds,
                     get_kv_cache_spec_kind=lambda spec: spec.inner_kind,
                     get_dcp_group=lambda: SimpleNamespace(rank_in_group=1),
                     cdiv=lambda a,b: (a+b-1)//b)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    config = SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=s) for s in specs])
    namespace["may_reinitialize_input_batch"](runner, config, [256]*3)
    assert [(t.dcp_world_size, t.dcp_rank) for t in tables] == [(1,0),(1,0),(2,1)]


class _FakeGroup:
    world_size = 2

    def __init__(self, other_max, other_sum, other_value):
        self.other_max = other_max
        self.other_sum = other_sum
        self.other_value = other_value

    def all_gather(self, value, dim=0):
        assert dim == 0
        return torch.cat((value, self.other_max), dim=0)

    def all_reduce(self, payload):
        global_max = torch.maximum(
            self.other_max, payload.new_tensor([[3.0, 1.0]])
        )
        scale = torch.exp(self.other_max - global_max)
        other = torch.stack((self.other_sum * scale, self.other_value * scale))
        return payload + other


def test_dcp_softmax_reduce_matches_full_softmax():
    values0 = torch.tensor([[2.0, 8.0], [5.0, 1.0]])
    scores0 = torch.tensor([[3.0, 0.0], [1.0, 1.0]])
    values1 = torch.tensor([[4.0, 6.0], [7.0, 3.0]])
    scores1 = torch.tensor([[2.0, -1.0], [0.0, -2.0]])

    max0 = scores0.max(0, keepdim=True).values
    weight0 = torch.exp(scores0 - max0)
    sum0 = weight0.sum(0, keepdim=True)
    weighted0 = (weight0 * values0).sum(0, keepdim=True)
    max1 = scores1.max(0, keepdim=True).values
    weight1 = torch.exp(scores1 - max1)
    sum1 = weight1.sum(0, keepdim=True)
    weighted1 = (weight1 * values1).sum(0, keepdim=True)

    actual = dcp_softmax_reduce(
        max0, sum0, weighted0, _FakeGroup(max1, sum1, weighted1)
    )
    expected = (
        torch.softmax(torch.cat((scores0, scores1), dim=0), dim=0)
        * torch.cat((values0, values1), dim=0)
    ).sum(0, keepdim=True)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("world_size,backend", [(1, "ag_rs"), (2, "a2a"), (2, "ag_rs")])
@pytest.mark.parametrize("keyword_config", [False, True])
def test_rocm_dcp_constructor_captures_config_without_forward_context(
    monkeypatch, world_size, backend, keyword_config
):
    source = Path(__file__).parents[2] / (
        "vllm_hcu/patch/worker/core_fix/patch_deepseek_v4_rocm_flashmla_sparse.py"
    )
    tree = ast.parse(source.read_text())
    helper = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == "_initialize_dcp_state")
    wrapper = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                   and n.name == "attention_init")
    group = object()
    distributed = ModuleType("vllm.distributed")
    calls = []

    def get_group():
        calls.append("group")
        return group

    distributed.get_dcp_group = get_group
    monkeypatch.setitem(sys.modules, "vllm.distributed", distributed)
    calibration = ModuleType("vllm_hcu.v1.attention.ops.deepseek_v4_ops.lse")
    calibration.calibrate_flashmla_lse = lambda device: (True, True)
    monkeypatch.setitem(sys.modules, calibration.__name__, calibration)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    def original_init(self, vllm_config, prefix):
        self.prefix = prefix
        self.n_local_heads = 8

    native_patch = ModuleType("_test_rocm_constructor.patch_deepseek_v4_dcp_compressor")
    native_patch.apply = lambda: None
    monkeypatch.setitem(sys.modules, native_patch.__name__, native_patch)
    logger = ModuleType("vllm.logger")
    logger.init_logger = lambda name: SimpleNamespace(info_once=lambda *args: None)
    monkeypatch.setitem(sys.modules, logger.__name__, logger)
    namespace = {"functools": functools, "torch": torch, "original_attention_init": original_init,
                 "__package__": "_test_rocm_constructor"}
    exec(compile(ast.Module(body=[helper, wrapper], type_ignores=[]),
                 str(source), "exec"), namespace)
    config = SimpleNamespace(parallel_config=SimpleNamespace(
        decode_context_parallel_size=world_size, dcp_comm_backend=backend
    ))
    attention = SimpleNamespace()
    if keyword_config:
        namespace["attention_init"](attention, vllm_config=config, prefix="layer")
    else:
        namespace["attention_init"](attention, config, "layer")
    assert attention.prefix == "layer"
    assert attention.dcp_world_size == world_size
    assert attention.dcp_group is (group if world_size > 1 else None)
    assert attention.dcp_a2a == (backend == "a2a")
    assert calls == (["group"] if world_size > 1 else [])
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in (
            "forward_decode", "forward_prefill"
        ):
            assert "get_current_vllm_config" not in ast.unparse(node)
            assert "prepare_dcp_state" not in ast.unparse(node)


def _load_dcp_function(name):
    source = Path(__file__).parents[2] / (
        "vllm_hcu/model_executor/layers/deepseek_v4_dcp_attention.py"
    )
    tree = ast.parse(source.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == name)
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"),
         namespace)
    return namespace[name]


def test_empty_attention_shard_has_zero_output_and_negative_infinity_lse():
    mask_empty = _load_dcp_function("_mask_empty_attention")
    out = torch.tensor([[[float("nan"), float("inf")]], [[2.0, 3.0]]])
    lse = torch.tensor([[float("nan")], [4.0]])
    actual_out, actual_lse = mask_empty(out, lse, torch.tensor([False, True]))
    torch.testing.assert_close(actual_out, torch.tensor([[[0.0, 0.0]], [[2.0, 3.0]]]))
    assert actual_lse[0].isneginf().all()
    assert actual_lse[1].item() == 4.0


def test_c128a_prefill_indices_partition_compressed_pool_without_duplicates():
    localize = _load_dcp_function("_localize_c128a_prefill_indices")
    indices = torch.tensor([[0, 1, 2, 3, 4, -1], [0, -1, -1, -1, -1, -1]])
    expected = [
        [[0, -1, 1, -1, 2, -1], [0, -1, -1, -1, -1, -1]],
        [[-1, 0, -1, 1, -1, -1], [-1, -1, -1, -1, -1, -1]],
    ]
    for rank in range(2):
        assert localize(indices, 2, rank).tolist() == expected[rank]


def test_global_c4_topk_is_localized_and_compacted_before_decode():
    localize = _load_dcp_function("_localize_c4_indices")
    localize.__globals__["_localize_c128a_prefill_indices"] = _load_dcp_function(
        "_localize_c128a_prefill_indices"
    )
    ids = torch.tensor([[1, 0, -1, -1], [4, 3, 1, 0]])
    assert localize(ids, 2, 0).tolist() == [[0,-1,-1,-1],[2,0,-1,-1]]
    assert localize(ids, 2, 1).tolist() == [[0,-1,-1,-1],[1,0,-1,-1]]


def test_compressed_slot_ownership_is_independent_of_raw_slot_ownership():
    class Pointer:
        def __init__(self, data, index=0):
            self.data, self.index = data, index

        def __add__(self, index):
            return Pointer(self.data, self.index + index)

    def load(ptr, mask=None, other=0):
        if mask is None:
            return ptr.data[ptr.index]
        ids = torch.as_tensor(ptr.index)
        result = torch.full(ids.shape, other, dtype=ptr.data.dtype)
        result[mask] = ptr.data[ids[mask]]
        return result

    def store(ptr, value, mask):
        ptr.data[ptr.index[mask]] = value[mask]

    source = Path(__file__).parents[2] / "vllm_hcu/v1/attention/backends/mla/compressor_utils.py"
    fn = next(n for n in ast.parse(source.read_text()).body
              if isinstance(n, ast.FunctionDef) and n.name == "_compressed_slot_mapping_dcp_kernel")
    fn.decorator_list = []
    namespace = {"tl": SimpleNamespace(constexpr=object(), program_id=lambda axis: 0,
                                      load=load, store=store, arange=torch.arange, where=torch.where),
                 "cp_global_to_local_block": lambda pos, block, world, rank, interleave:
                 (pos//world//block, pos//world%block, pos%world==rank)}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), namespace)
    for rank, owned_position in [(0,3),(1,7)]:
        out = torch.full((8,), -1, dtype=torch.int64)
        namespace[fn.name](Pointer(out), Pointer(torch.full((8,), -1)),
                           Pointer(torch.tensor([0,8])), Pointer(torch.tensor([8])),
                           Pointer(torch.tensor([5])), 1, 64, 4, 2, rank, 1, 8)
        expected = torch.full_like(out,-1)
        expected[owned_position] = 5*64
        assert out.equal(expected)


@pytest.mark.parametrize("gather_len", [0, 4])
def test_prefill_combine_omits_absent_swa_and_preserves_invalid_topk(gather_len):
    # Evaluate the Triton index arithmetic on CPU, including a nonzero chunk
    # row so accidentally adding the row offset to -1 cannot go unnoticed.
    class Pointer:
        def __init__(self, data, index=0):
            self.data, self.index = data, index

        def __add__(self, index):
            return Pointer(self.data, self.index + index)

    def load(ptr, mask=None, other=0):
        if mask is None:
            return ptr.data[ptr.index]
        indices = torch.as_tensor(ptr.index)
        values = torch.full_like(indices, other)
        values[mask] = ptr.data[indices[mask]]
        return values

    def store(ptr, values, mask=None):
        if mask is None:
            ptr.data[ptr.index] = values
        else:
            indices = torch.as_tensor(ptr.index)
            ptr.data[indices[mask]] = torch.as_tensor(
                values, dtype=ptr.data.dtype
            ).expand_as(indices)[mask]

    source = Path(__file__).parents[2] / (
        "vllm_hcu/v1/attention/ops/deepseek_v4_ops/cache_utils.py"
    )
    tree = ast.parse(source.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == "_combine_topk_swa_indices_kernel")
    fn.decorator_list = []
    tl = SimpleNamespace(
        constexpr=object(), program_id=lambda axis: 1 if axis == 0 else 0,
        num_programs=lambda axis: 1, arange=torch.arange,
        load=load, store=store, where=torch.where,
        minimum=lambda x, y: torch.minimum(torch.as_tensor(x), torch.as_tensor(y)),
    )
    namespace = {"tl": tl}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), namespace)
    combined = torch.full((12,), -1, dtype=torch.int32)
    lengths = torch.zeros(2, dtype=torch.int32)
    namespace[fn.name](
        Pointer(combined), 6, Pointer(lengths),
        Pointer(torch.tensor([0, -1, 0, -1])), 2,
        Pointer(torch.tensor([0, 1, 2])), Pointer(torch.tensor([8, 8])),
        Pointer(torch.tensor([gather_len, gather_len])),
        32, 4, 2, 4, 4, 2,
    )
    assert combined[6:8].tolist() == [32, -1]
    assert lengths[1].item() == 2 + gather_len
    if gather_len == 0:
        assert combined[8:].eq(-1).all()
    else:
        assert combined[8:].tolist() == [36, 37, 38, 39]


@pytest.mark.parametrize("local_heads", [2, 16])
def test_dcp_decode_normalizes_flashmla_lse_and_empty_shard_before_collective(local_heads):
    source = Path(__file__).parents[2] / (
        "vllm_hcu/model_executor/layers/deepseek_v4_dcp_attention.py"
    )
    tree = ast.parse(source.read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
              and n.name == "_forward_decode")
    fn.returns = None
    for arg in fn.args.args:
        arg.annotation = None

    def flash(**kwargs):
        q = kwargs["q"]
        assert q.shape[2] == (64 if local_heads == 16 else 4)
        assert kwargs["attn_sink"] is None
        assert kwargs["topk_length"].eq(0).all()
        return torch.full_like(q, float("nan")), torch.full(
            (q.shape[0], q.shape[2], 1), float("nan")
        )

    def combine(out, lse, group, **kwargs):
        assert out.shape == (2, local_heads * 2, 512)
        assert lse.shape == (2, local_heads * 2)
        assert out.eq(0).all()
        assert lse.isneginf().all()
        # Simulate the nonempty peer's output and global attention LSE.
        return torch.ones(2, local_heads, 512), torch.full((2, local_heads), torch.log(torch.tensor(2.0)))

    namespace = {
        "torch": torch,
        "current_platform": SimpleNamespace(is_rocm=lambda: True),
        "henvs": SimpleNamespace(VLLM_HCU_DEEPSEEK_V4_ROCM_DECODE_FALLBACK=False),
        "flash_mla_with_kvcache": flash,
        "dcp_a2a_lse_reduce": combine,
        "_mask_empty_attention": _load_dcp_function("_mask_empty_attention"),
        "check_dcp_tensor": lambda *args: None,
    }
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), namespace)
    attention = SimpleNamespace(
        compress_ratio=1, dcp_world_size=2, dcp_a2a=True,
        dcp_group=SimpleNamespace(rank_in_group=1,
                                  all_gather=lambda q, dim: torch.cat((q, q), dim=dim)),
        swa_cache_layer=SimpleNamespace(kv_cache=torch.zeros(1, 4, 584, dtype=torch.uint8)),
        attn_sink=torch.zeros(local_heads), scale=0.125,
    )
    metadata = SimpleNamespace(
        num_decodes=2, num_decode_tokens=2,
        decode_swa_indices=torch.zeros(2, 1, 4, dtype=torch.int32),
        decode_swa_lens=torch.full((2,), 4), tile_sched_swaonly=object(),
    )
    q = torch.zeros(2, local_heads, 512)
    output = torch.empty_like(q)
    namespace[fn.name](attention, q, None, metadata, None, True, output)
    torch.testing.assert_close(output, torch.full_like(q, 2.0 / 3.0))


@pytest.mark.parametrize("world_size", [1, 2])
def test_native_compressor_patch_routes_both_cache_writers(monkeypatch, world_size):
    package = ModuleType("_test_dcp_compressor_patch")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, package.__name__, package)
    _load_source(package.__name__ + "._common",
                 "vllm_hcu/patch/worker/core_fix/_common.py")
    patch = _load_source(package.__name__ + ".adapter",
                        "vllm_hcu/patch/worker/core_fix/patch_deepseek_v4_dcp_compressor.py")
    calls = []
    group = object()
    layout_module = ModuleType("vllm_hcu.v1.cp_layout")
    layout_module.ContextParallelLayout = SimpleNamespace(
        from_config=lambda config: SimpleNamespace(
            enabled=world_size > 1, interleave_size=1, world_size=world_size, rank=0
        )
    )
    monkeypatch.setitem(sys.modules, layout_module.__name__, layout_module)
    distributed = ModuleType("vllm.distributed")
    distributed.get_dcp_group = lambda: group
    monkeypatch.setitem(sys.modules, distributed.__name__, distributed)
    logger_module = ModuleType("vllm.logger")
    logger_module.init_logger = lambda name: SimpleNamespace(info_once=lambda *args: None)
    monkeypatch.setitem(sys.modules, logger_module.__name__, logger_module)
    implementation = ModuleType("vllm_hcu.model_executor.layers.deepseek_v4_dcp_compressor")
    implementation.DeepseekV4DCPCompressor = SimpleNamespace(
        forward=lambda self, *args: calls.append(("dcp", self.head_dim, args))
    )
    monkeypatch.setitem(sys.modules, implementation.__name__, implementation)

    class Compressor:
        def __init__(self, vllm_config, head_dim):
            self.head_dim = head_dim

        def forward(self, kv_score, positions, rotary_emb):
            calls.append(("original", self.head_dim, (kv_score, positions, rotary_emb)))

    module = ModuleType(patch.TARGET_MODULE)
    module.DeepseekCompressor = Compressor
    assert patch.apply_to_module(module)
    assert not patch.apply_to_module(module)
    for head_dim in (128, 512):
        compressor = Compressor(vllm_config=object(), head_dim=head_dim)
        assert compressor.dcp_group is (group if world_size > 1 else None)
        compressor.forward("score", "positions", "rope")
    assert calls == [
        ("dcp" if world_size > 1 else "original", head_dim, ("score", "positions", "rope"))
        for head_dim in (128, 512)
    ]


@pytest.mark.parametrize("padding", [False, True])
def test_compressor_partial_stats_read_replicated_ring_slots(padding):
    class Pointer:
        def __init__(self, data, index=0):
            self.data, self.index = data, index

        def __add__(self, index):
            return Pointer(self.data, self.index + index)

        def __getitem__(self, key):
            return Pointer(self.data, self.index[key])

    def load(ptr, mask=None, other=0):
        if mask is None:
            return ptr.data[ptr.index]
        indices, mask = torch.broadcast_tensors(torch.as_tensor(ptr.index), mask)
        result = torch.full(indices.shape, other, dtype=ptr.data.dtype)
        result[mask] = ptr.data[indices[mask]]
        return result

    def store(ptr, value, mask=None):
        if mask is None:
            ptr.data[ptr.index] = value
        else:
            indices = torch.as_tensor(ptr.index)
            ptr.data[indices[mask]] = torch.as_tensor(
                value, dtype=ptr.data.dtype
            ).expand_as(indices)[mask]

    source = Path(__file__).parents[2] / (
        "vllm_hcu/v1/attention/ops/deepseek_v4_ops/fused_compress_quant_cache.py"
    )
    tree = ast.parse(source.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == "dsv4_dcp_compressor_partial_stats_kernel")
    fn.decorator_list = []
    tl = SimpleNamespace(
        constexpr=object(), program_id=lambda axis: 0, arange=torch.arange,
        load=load, store=store, where=torch.where,
        int32=torch.int32, int64=torch.int64,
        max=lambda x, axis: x.max(dim=axis).values,
        sum=lambda x, axis: x.sum(dim=axis), exp=torch.exp,
    )
    namespace = {"tl": tl}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(source), "exec"), namespace)
    state = torch.arange(3 * 4 * 8, dtype=torch.float32).reshape(3, 4, 8) / 17
    table = torch.tensor([2, 0])
    stats = []
    for rank in range(2):
        m, s, v = (torch.empty(2) for _ in range(3))
        namespace[fn.name](
            Pointer(state.flatten()), 32, 8, Pointer(torch.tensor([0])),
            Pointer(torch.tensor([7])), Pointer(torch.tensor([-1 if padding else 0])),
            Pointer(table), 2, 4, Pointer(m), Pointer(s), Pointer(v), 2,
            2, 2, 4, 4, True, 2, rank, 1,
        )
        stats.append((m, s, v))
    if padding:
        for m, s, v in stats:
            assert m.isneginf().all() and s.eq(0).all() and v.eq(0).all()
        return
    values, scores = [], []
    for pos in range(8):
        row = state[table[pos // 4], pos % 4]
        offset = 0 if pos < 4 else 2
        values.append(row[offset:offset + 2])
        scores.append(row[4 + offset:6 + offset])
    expected = (torch.stack(scores).softmax(dim=0) * torch.stack(values)).sum(dim=0)
    global_max = torch.stack([x[0] for x in stats]).max(dim=0).values
    sums = [s * torch.exp(m - global_max) for m, s, v in stats]
    weighted = [v * torch.exp(m - global_max) for m, s, v in stats]
    torch.testing.assert_close(sum(weighted) / sum(sums), expected)


@pytest.mark.parametrize("patched,world_size", [(False, 2), (True, 1), (True, 2)])
def test_loaded_model_audit_rejects_missing_or_misconfigured_dcp(patched, world_size):
    audit = _load_source("_test_dcp_model_audit", "vllm_hcu/v1/deepseek_v4_dcp_audit.py")

    class Attention:
        dcp_world_size = world_size

        def _forward_decode(self):
            pass

        def _forward_prefill(self):
            pass

    if patched:
        for fn in (Attention._forward_decode, Attention._forward_prefill):
            fn._vllm_hcu_flashmla_sparse_decode_applied = True
    model = SimpleNamespace(named_modules=lambda: [("model.layers.0.attn", Attention())])
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(decode_context_parallel_size=2),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(
            architectures=["DeepseekV4ForCausalLM"]
        )),
    )
    if patched and world_size == 2:
        audit.audit_deepseek_v4_dcp(model, config)
    else:
        with pytest.raises(RuntimeError, match="DCP model wiring failed"):
            audit.audit_deepseek_v4_dcp(model, config)


def test_flashmla_lse_calibration_distinguishes_bases_and_rejects_bad_statistics():
    module = _load_source("_test_lse_convention", "vllm_hcu/v1/attention/ops/deepseek_v4_ops/lse.py")
    assert module.lse_is_natural(torch.full((2, 8), torch.log(torch.tensor(4.0))))
    assert not module.lse_is_natural(torch.full((2, 8), 2.0))
    with pytest.raises(RuntimeError, match="LSE calibration failed"):
        module.lse_is_natural(torch.zeros(2, 8))


@pytest.mark.parametrize("world_size", [1, 2])
@pytest.mark.parametrize("drafts,parallel", [(0, False), (5, False), (5, True)])
def test_speculative_dcp_mla_split_matches_swa(world_size, drafts, parallel):
    source = Path(__file__).parents[2] / (
        "vllm_hcu/patch/worker/core_fix/patch_deepseek_v4_rocm_flashmla_sparse.py"
    )
    helper = next(n for n in ast.parse(source.read_text()).body
                  if isinstance(n, ast.FunctionDef)
                  and n.name == "_initialize_dcp_decode_threshold")
    namespace = {}
    exec(compile(ast.Module(body=[helper], type_ignores=[]), str(source), "exec"), namespace)
    # Exercise the real generic initializer: its DCP reset caused the crash.
    base = Path("/usr/local/lib/python3.10/dist-packages/vllm/v1/attention/backend.py")
    init = next(n for n in ast.walk(ast.parse(base.read_text()))
                if isinstance(n, ast.FunctionDef) and n.name == "_init_reorder_batch_threshold")
    exec(compile(ast.Module(body=[init], type_ignores=[]), str(base), "exec"), namespace)
    spec = SimpleNamespace(num_speculative_tokens=drafts, parallel_drafting=parallel)
    builder = SimpleNamespace(vllm_config=SimpleNamespace(
        speculative_config=spec,
        parallel_config=SimpleNamespace(decode_context_parallel_size=world_size),
    ))
    builder._init_reorder_batch_threshold = functools.partial(
        namespace["_init_reorder_batch_threshold"], builder,
    )
    builder._init_reorder_batch_threshold(1, supports_spec_as_decode=True)
    namespace["_initialize_dcp_decode_threshold"](builder)
    assert builder.reorder_batch_threshold == 1 + drafts * (2 if parallel else 1)


@pytest.mark.parametrize("rank", [0, 1])
def test_v2_dcp_slot_mappings_replicate_state_and_shard_compressed_cache(rank):
    source = Path(__file__).parents[2] / "vllm_hcu/v1/deepseek_v4_dcp_block_tables.py"
    cls = next(n for n in ast.parse(source.read_text()).body if isinstance(n, ast.ClassDef))
    calls = []

    class Kernel:
        def __getitem__(self, grid):
            def run(max_tokens, idx, starts, positions, tables, strides, sizes,
                    slots, slot_stride, cp_rank, **kw):
                assert grid == (1, 2)
                cp_size = kw["CP_SIZE"]
                calls.append(cp_size)
                size = int(sizes[0])
                for token, pos in enumerate(positions.tolist()):
                    block = int(tables[0][0, pos // (size * cp_size)])
                    local = pos // cp_size % size
                    slots[0, token] = (
                        block * size + local if pos % cp_size == cp_rank else -1
                    )
                slots[0, len(positions):].fill_(-1)
            return run

    namespace = dict(BlockTables=object, _compute_slot_mappings_kernel=Kernel(), PAD_SLOT_ID=-1)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"), namespace)
    tables = object.__new__(namespace["DeepseekV4DCPBlockTables"])
    tables.replicated_groups = (0,)
    tables.num_kv_cache_groups = 2
    tables.max_num_batched_tokens = 20
    tables.cp_size, tables.cp_rank, tables.cp_interleave = 2, rank, 1
    tables.block_table_ptrs = [torch.tensor([[10, 11, 12, 13]]), torch.tensor([[20, 21]])]
    tables.block_table_strides = [4, 2]
    tables.block_sizes_tensor = torch.tensor([4, 4])
    tables.slot_mappings = torch.zeros(2, 20, dtype=torch.int64)
    result = tables.compute_slot_mappings(
        torch.tensor([0]), torch.tensor([0, 16]), torch.arange(16), 18,
    )
    assert calls == [1, 2]
    assert result[0].tolist() == list(range(40, 56)) + [-1, -1]
    assert result[1].tolist() == [
        80 + pos // 2 if pos % 2 == rank else -1 for pos in range(16)
    ] + [-1, -1]
