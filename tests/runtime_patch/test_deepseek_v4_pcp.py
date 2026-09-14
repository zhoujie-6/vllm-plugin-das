# SPDX-License-Identifier: Apache-2.0
"""Full-KV PCP contracts, including the uneven/empty-shard collective path."""
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from types import SimpleNamespace, ModuleType

import pytest
import torch

from vllm_hcu.v1.deepseek_v4_pcp import (
    build_2n_balanced_cp_plan_for_query_lens as build_plan,
    get_local_query_chunk, select_tokens, gather_tokens,
)


@pytest.mark.parametrize('width', [1, 2, 4, 8])
@pytest.mark.parametrize('lens', [[], [0], [1], [3, 1, 17], [127, 128, 129], [9, 0, 2]])
def test_plans_cover_each_request_exactly_once(width, lens):
    plans = [build_plan(lens, width, rank) for rank in range(width)]
    owned = torch.cat([p.local_indices for p in plans])
    assert torch.equal(owned.sort().values, torch.arange(sum(lens)))
    for p in plans:
        assert tuple(p.local_query_start_loc.diff().tolist()) == p.local_query_lens
        offset = 0
        for req, size in enumerate(lens):
            a, b, rows = get_local_query_chunk(p, req, req + 1, query_start=offset)
            assert b - a == p.local_query_lens[req]
            assert bool(((rows >= 0) & (rows < size)).all())
            offset += size


class Collective:
    """Synchronous CPU collectives; a missing empty-rank call times out."""
    def __init__(self, width):
        self.width = width
        self.barrier = Barrier(width, timeout=20)
        self.values = [None] * width

    def group(self, rank):
        def all_gather(tensor, dim=0):
            self.values[rank] = tensor.clone()
            self.barrier.wait()
            result = torch.cat(self.values, dim=dim)
            self.barrier.wait()
            return result
        return SimpleNamespace(world_size=self.width, rank_in_group=rank,
                               all_gather=all_gather)


@pytest.mark.parametrize('width,lens,nd', [(2, [33, 3], 2), (4, [1], 0),
                                         (8, [3, 1, 19], 3), (4, [0], 2)])
def test_gather_restores_mixed_batch_without_aliasing(width, lens, nd):
    full = torch.arange((nd + sum(lens)) * 3).reshape(-1, 3).float()
    collective = Collective(width)
    def run(rank):
        plan = build_plan(lens, width, rank)
        local = select_tokens(full, plan, nd)
        a = gather_tokens(local, plan, nd, group=collective.group(rank))
        b = gather_tokens(local + 17, plan, nd, group=collective.group(rank))
        torch.testing.assert_close(a, full)
        torch.testing.assert_close(b, full + 17)
        if a.numel():
            assert a.data_ptr() != b.data_ptr()
    with ThreadPoolExecutor(width) as pool:
        list(pool.map(run, range(width)))


def test_v4_config_preserves_source_scope_and_other_architectures():
    from tests.runtime_patch.test_glm52_pcp_config import _make_pcp_config
    from vllm_hcu.patch.platform.core_fix.patch_vllm_config import _validate_hcu_pcp_scope
    for use_v2 in (False, True):
        config = _make_pcp_config(architecture='DeepseekV4ForCausalLM', use_v2=use_v2,
                                  dp=2, pp=2, enforce_eager=False)
        config.parallel_config.cp_kv_cache_interleave_size = 1
        assert _validate_hcu_pcp_scope(config)
        config.parallel_config.decode_context_parallel_size = 2
        with pytest.raises(ValueError, match='DCP'):
            _validate_hcu_pcp_scope(config)
        config.parallel_config.decode_context_parallel_size = 1
        config.parallel_config.cp_kv_cache_interleave_size = 2
        with pytest.raises(ValueError, match='interleave'):
            _validate_hcu_pcp_scope(config)
        config.parallel_config.cp_kv_cache_interleave_size = 1
        config.kv_transfer_config = object()
        with pytest.raises(ValueError, match='KV transfer'):
            _validate_hcu_pcp_scope(config)
    glm = _make_pcp_config(use_v2=False)
    with pytest.raises(ValueError, match='Model Runner V2'):
        _validate_hcu_pcp_scope(glm)


@pytest.mark.parametrize('v4,pcp,expected', [(True, 4, 1), (True, 1, 1), (False, 4, 4)])
def test_only_v4_cache_factory_uses_unitary_ownership(v4, pcp, expected):
    from vllm_hcu.patch.platform.framework_opt import patch_deepseek_v4_pcp_cache as patch
    module = ModuleType(patch.TARGET_MODULE)
    def get_kv_cache_coordinator(kv_cache_config, max_model_len, max_num_batched_tokens,
                                 use_eagle, enable_caching, enable_kv_cache_events,
                                 dcp_world_size, pcp_world_size, scheduler_block_size,
                                 hash_block_size, metrics_collector=None):
        return pcp_world_size, enable_caching, hash_block_size
    module.get_kv_cache_coordinator = get_kv_cache_coordinator
    assert patch.apply_to_module(module)
    assert not patch.apply_to_module(module)
    cache = SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=
        SimpleNamespace(model_version='deepseek_v4' if v4 else None))])
    for prefix in (False, True):
        assert module.get_kv_cache_coordinator(cache, 1000, 512, False, prefix,
                    False, 1, pcp, 256, 64) == (expected, prefix, 64)


def test_packed_v4_cache_is_recognized():
    from vllm_hcu.patch.platform.framework_opt.patch_deepseek_v4_pcp_cache import has_v4_cache
    spec = SimpleNamespace(kv_cache_specs={'a': SimpleNamespace(model_version='deepseek_v4')})
    assert has_v4_cache(SimpleNamespace(kv_cache_groups=[SimpleNamespace(kv_cache_spec=spec)]))


@pytest.mark.parametrize('width,lens,nd', [(2, [17, 3], 2), (4, [1], 0)])
@pytest.mark.parametrize('compressed', [False, True])
def test_attention_shards_compute_and_preserves_full_kv(monkeypatch, width, lens, nd, compressed):
    import sys
    from threading import local
    from vllm_hcu.model_executor.layers import deepseek_v4_pcp as runtime
    tls = local()
    from vllm.models.deepseek_v4 import attention as attention_module
    def quantize(pos, q, rope, weights, *args, **kwargs):
        return q, weights
    monkeypatch.setattr(attention_module, 'fused_indexer_q_rope_quant', quantize)
    monkeypatch.setattr(runtime, 'get_forward_context', lambda: tls.context)
    collective = Collective(width)
    monkeypatch.setattr(runtime, 'gather_tokens',
        lambda x, p, n: gather_tokens(x, p, n, group=collective.group(tls.rank)))
    # The backend is separately tested below; this checks projection/cache
    # layouts and mixed decode/prefill restoration without accelerator kernels.
    backend = ModuleType('vllm_hcu.v1.attention.deepseek_v4_pcp')
    def prefill(attn, q, pos, compressed_cache, swa_cache, out, metadata, swa):
        out.copy_(q * 2 + pos[:, None, None])
    backend.forward_prefill = prefill
    monkeypatch.setitem(sys.modules, backend.__name__, backend)
    n = nd + sum(lens)
    hidden = torch.arange(n * 2).float().reshape(n, 2)
    positions = torch.arange(n) + 7
    # Include a runner padding tail; it must be excluded from KV writes.
    padded_hidden = torch.cat((hidden, torch.zeros(3, 2)))
    padded_positions = torch.cat((positions, torch.zeros(3, dtype=positions.dtype)))
    expected = ((hidden + positions[:, None]) * 2 + positions[:, None])
    expected = torch.cat((expected, torch.zeros(3, 2)))

    def run(rank):
        tls.rank = rank
        plan = build_plan(lens, width, rank)
        swa = SimpleNamespace(hcu_v4_pcp_plan=plan, num_decode_tokens=nd)
        cache = SimpleNamespace(prefix='swa', kv_cache=torch.empty(n, 2))
        attn = SimpleNamespace(swa_cache_layer=cache, q_lora_rank=2, head_dim=2,
                               n_local_heads=1, prefix='attn', compress_ratio=4 if compressed else 1,
                               kv_cache=torch.empty(1), rotary_emb=None, indexer=None)
        sizes = []
        def project(x):
            sizes.append(len(x))
            return (torch.cat((x, x + 100), -1),
                    x + 200 if compressed else None,
                    x + 300 if compressed else None,
                    x[:, :1] + 400 if compressed else None)
        attn.attn_gemm_parallel_execute = project
        attn.q_norm = lambda x: x
        attn.wq_b = lambda x: x
        def insert(q, kv, pos, metadata):
            torch.testing.assert_close(kv, hidden + 100)
            torch.testing.assert_close(pos, positions)
            cache.kv_cache.copy_(kv)
            return q + pos[:, None, None]
        attn._fused_qnorm_rope_kv_insert = insert
        def compressor(score, pos, rope):
            torch.testing.assert_close(score, hidden + 200)
        attn.compressor = compressor if compressed else None
        index_meta = SimpleNamespace()
        if compressed:
            def index_compressor(score, pos, rope):
                torch.testing.assert_close(score, hidden + 300)
            def index_op(h, q, k, weights):
                torch.testing.assert_close(select_tokens(q.flatten(1), plan, nd),
                                           select_tokens(hidden, plan, nd))
                torch.testing.assert_close(select_tokens(weights, plan, nd),
                                           select_tokens(hidden[:, :1] + 400, plan, nd))
                assert index_meta.hcu_v4_pcp_plan is plan
            attn.indexer = SimpleNamespace(
                wq_b=lambda q: (q, None), n_head=1, head_dim=2, softmax_scale=1.,
                use_fp4_kv=False, k_cache=SimpleNamespace(prefix='index'),
                compressor=index_compressor, indexer_op=index_op,
            )
            attn.indexer_rotary_emb = SimpleNamespace(cos_sin_cache=None)
        attn._forward_decode = lambda **kw: kw['output'].copy_(
            kw['q'] * 2 + positions[:nd, None, None])
        attn._o_proj = lambda out, pos: out.flatten(1)
        tls.context = SimpleNamespace(no_compile_layers={'attn': attn},
                                       attn_metadata={'attn': object(), 'swa': swa, 'index': index_meta})
        result = runtime.attention_forward(padded_hidden, padded_positions, 'attn')
        torch.testing.assert_close(result, expected)
        assert sizes == [nd + plan.local_sizes[rank]]
        assert not hasattr(index_meta, 'hcu_v4_pcp_plan')
    with ThreadPoolExecutor(width) as pool:
        list(pool.map(run, range(width)))


def test_pcp_eplb_counts_pcp_ranks_and_preserves_validation():
    from vllm.config import parallel
    from vllm_hcu.patch.platform.core_fix import patch_deepseek_v4_pcp_parallel as patch
    patch.apply_to_module(parallel)
    assert not patch.apply_to_module(parallel)
    config = parallel.ParallelConfig(
        tensor_parallel_size=1, data_parallel_size=1,
        prefill_context_parallel_size=2, enable_expert_parallel=True,
        enable_eplb=True, distributed_executor_backend='mp',
    )
    assert config.tensor_parallel_size == config.data_parallel_size == 1
    assert config.prefill_context_parallel_size == 2
    with pytest.raises(ValueError, match='enable_expert_parallel'):
        parallel.ParallelConfig(prefill_context_parallel_size=2, enable_eplb=True,
                                enable_expert_parallel=False)
    with pytest.raises(ValueError, match='EPLB requires'):
        parallel.ParallelConfig(prefill_context_parallel_size=1, enable_eplb=True,
                                enable_expert_parallel=True)


def test_pcp_custom_ops_have_global_fake_shapes():
    from torch._subclasses.fake_tensor import FakeTensorMode
    from vllm_hcu.model_executor.layers import deepseek_v4_pcp  # noqa: F401
    with FakeTensorMode():
        hidden = torch.empty(13, 64)
        ids = torch.empty(13, dtype=torch.long)
        output = torch.ops.vllm.hcu_deepseek_v4_pcp_attention(hidden, ids, 'attn')
        moe_output = torch.ops.vllm.hcu_deepseek_v4_pcp_moe(hidden, ids, 'moe')
        assert output.shape == moe_output.shape == hidden.shape


def test_v4_pcp_runner_does_not_use_glm_virtual_rows(pcp_runner_module):
    module, events = pcp_runner_module
    config = SimpleNamespace(model_config=SimpleNamespace(architectures=['DeepseekV4ForCausalLM']),
                             parallel_config=SimpleNamespace(prefill_context_parallel_size=2))
    runner = module.HcuGPUModelRunnerV2(config, torch.device('cpu'))
    runner.initialize_kv_cache(SimpleNamespace(kv_cache_groups=[object(), object()]))
    assert runner.pcp_manager is None
    assert events.count('super.initialize_kv_cache') == 1


# Reuse the runner ABI fixture, not a duplicate approximation of that ABI.
from tests.runtime_patch.test_glm52_pcp_runner import pcp_runner_module


@pytest.mark.parametrize('v4', [False, True])
def test_pcp_graph_splitting_is_scoped_to_v4(v4):
    from tests.runtime_patch.test_platform_hcu_config import _make_compilation_module
    from vllm_hcu.patch.platform.core_fix import patch_compilation_config as patch
    module = _make_compilation_module()
    patch.apply_to_module(module)
    config = module.CompilationConfig()
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(architectures=['DeepseekV4ForCausalLM' if v4 else 'GlmMoeDsaForCausalLM']),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=2),
        additional_config={}, compilation_config=config,
    )
    patch.bind_hcu_config(vllm_config)
    config.set_splitting_ops_for_v1('allgather_reducescatter')
    config.set_splitting_ops_for_v1('allgather_reducescatter')
    for op in ('vllm::hcu_deepseek_v4_pcp_attention', 'vllm::hcu_deepseek_v4_pcp_moe'):
        assert config.splitting_ops.count(op) == int(v4)


@pytest.mark.parametrize('ratio', [1, 4, 128])
def test_backend_selects_chunk_rows_in_global_query_coordinates(monkeypatch, ratio):
    from vllm_hcu.v1.attention import deepseek_v4_pcp as backend
    lens, nd = [1, 3, 7], 2
    n = sum(lens)
    starts = torch.tensor([0, nd, nd + 1, nd + 4, nd + n], dtype=torch.int32)
    monkeypatch.setattr(backend, 'current_workspace_manager', lambda: SimpleNamespace(
        get_simultaneous=lambda spec: [torch.empty(spec[0], dtype=spec[1])]))
    monkeypatch.setattr(backend, 'dequantize_and_gather_k_cache', lambda *a, **kw: None)
    def combine(topk, qs, *args):
        return (torch.arange(int(qs[0]) - nd, int(qs[-1]) - nd).view(-1, 1),
                torch.ones(int(qs[-1] - qs[0]), dtype=torch.int32))
    monkeypatch.setattr(backend, 'combine_topk_swa_indices', combine)
    monkeypatch.setattr(backend, 'rocm_sparse_attn_prefill',
        lambda **kw: kw['output'].copy_(kw['q'] * 2 + kw['indices'][:, :, None]))
    full_q = torch.arange(n * 2).reshape(n, 1, 2).float()
    for rank in range(4):
        plan = build_plan(lens, 4, rank)
        swa = SimpleNamespace(hcu_v4_pcp_plan=plan, num_prefills=3,
            num_prefill_tokens=n, num_decodes=1, num_decode_tokens=nd,
            prefill_seq_lens=torch.tensor([1, 7, 9]), prefill_gather_lens=torch.tensor(lens),
            query_start_loc_cpu=starts, query_start_loc=starts,
            block_table=torch.zeros(4, 1, dtype=torch.int32), block_size=64)
        attn = SimpleNamespace(compress_ratio=ratio, topk_indices_buffer=torch.zeros(nd+n, 2),
            max_model_len=16, max_num_batched_tokens=16, window_size=4,
            PREFILL_CHUNK_SIZE=2, scale=1., head_dim=2, nope_head_dim=2,
            rope_head_dim=0, attn_sink=None)
        meta = None if ratio == 1 else SimpleNamespace(
            block_table=torch.zeros(4, 1, dtype=torch.int32), block_size=256,
            c128a_prefill_topk_indices=torch.zeros(n, 2))
        q = select_tokens(full_q, plan)
        output = torch.empty_like(q)
        backend.forward_prefill(attn, q, plan.local_indices, torch.empty(1),
                                torch.empty(1), output, meta, swa)
        expected = select_tokens(full_q * 2 + torch.arange(n)[:, None, None], plan)
        torch.testing.assert_close(output, expected)


def _gloo_worker(rank, width, rendezvous):
    import torch.distributed as dist
    from datetime import timedelta
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous,
                            rank=rank, world_size=width, timeout=timedelta(seconds=45))
    try:
        def all_gather(tensor, dim=0):
            outputs = [torch.empty_like(tensor) for _ in range(width)]
            dist.all_gather(outputs, tensor)
            return torch.cat(outputs, dim=dim)
        group = SimpleNamespace(rank_in_group=rank, world_size=width, all_gather=all_gather)
        for lens, nd in [([1], 0), ([3, 17, 1], 2), ([0, 2], 1)]:
            plan = build_plan(lens, width, rank)
            global_tensor = torch.arange((sum(lens) + nd) * 5).reshape(-1, 5).float()
            local = select_tokens(global_tensor, plan, nd)
            actual = gather_tokens(local, plan, nd, group=group)
            torch.testing.assert_close(actual, global_tensor)
    finally:
        dist.destroy_process_group()


def test_real_gloo_collectives_restore_uneven_shards(tmp_path):
    torch.multiprocessing.spawn(_gloo_worker, args=(2, str(tmp_path / 'store')),
                                nprocs=2, join=True)


def test_v4_ep_does_not_repeat_pcp_dispatch():
    import ast
    from pathlib import Path
    source = Path('vllm_hcu/model_executor/layers/fused_moe/moe_runner.py').read_text()
    cls = next(node for node in ast.parse(source).body
               if isinstance(node, ast.ClassDef) and node.name == 'MoERunner')
    functions = [node for node in cls.body if isinstance(node, ast.FunctionDef)
                 and node.name in ('_maybe_dispatch', '_maybe_combine')]
    namespace = {'torch': torch,
                 'get_pcp_group': lambda: pytest.fail('duplicate PCP collective')}
    exec(compile(ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[])),
                 '<MoERunner PCP contract>', 'exec'), namespace)
    runner = SimpleNamespace(
        do_naive_dispatch_combine=False, _hcu_v4_pcp=True, _hcu_v4_pcp_ep=True,
        moe_config=SimpleNamespace(pcp_size=4,
            moe_parallel_config=SimpleNamespace(use_all2all_kernels=False)),
        shared_experts=None,
    )
    hidden, logits = torch.randn(3, 4), torch.randn(3, 8)
    out, routed = namespace['_maybe_dispatch'](runner, hidden, logits)
    assert out is hidden and routed is logits
    assert namespace['_maybe_combine'](runner, None, hidden) is hidden



def test_pcp_eplb_model_interface_is_instance_scoped(monkeypatch):
    from vllm_hcu.patch.worker.core_fix import patch_deepseek_v4_pcp_model as patch
    import vllm.distributed
    monkeypatch.setattr(vllm.distributed, 'get_ep_group',
                        lambda: SimpleNamespace(world_size=2))
    events = []
    expert = SimpleNamespace(
        get_expert_weights=lambda: ('weights',),
        set_eplb_state=lambda **kwargs: events.append(kwargs),
        update_expert_map=lambda: events.append('update'),
    )
    class DeepseekV4MoE:
        def __init__(self, vllm_config, prefix=''):
            self.experts = expert
        def forward(self, hidden_states, input_ids=None):
            return hidden_states
    class DeepseekV4ForCausalLM:
        def __init__(self, *, vllm_config, prefix=''):
            self.model = SimpleNamespace(layers=[SimpleNamespace(ffn=SimpleNamespace(experts=expert))])
    module = ModuleType(patch.TARGET_MODULE)
    module.DeepseekV4MoE = DeepseekV4MoE
    module.DeepseekV4ForCausalLM = DeepseekV4ForCausalLM
    assert patch.apply_to_module(module)
    assert not patch.apply_to_module(module)
    config = SimpleNamespace(
        model_config=SimpleNamespace(architectures=['DeepseekV4ForCausalLM'],
            hf_config=SimpleNamespace(n_routed_experts=4, n_shared_experts=1)),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=2, enable_eplb=True,
            eplb_config=SimpleNamespace(num_redundant_experts=2)),
    )
    model = module.DeepseekV4ForCausalLM(vllm_config=config)
    assert model.num_moe_layers == 1
    assert model.num_physical_experts == 6 and model.num_local_physical_experts == 3
    model.set_eplb_state(torch.zeros(1, 6), torch.zeros(1, 4), torch.ones(1, 4))
    assert model.expert_weights == [('weights',)]
    assert events[0]['moe_layer_idx'] == 0
    model.update_physical_experts_metadata(6, 3)
    assert events[-1] == 'update'
    config.parallel_config.prefill_context_parallel_size = 1
    ordinary = module.DeepseekV4ForCausalLM(vllm_config=config)
    assert not hasattr(ordinary, 'moe_layers')


@pytest.mark.parametrize("v4,pcp,ep,dp,sp,expected", [
    (True, 8, True, 1, False, True),
    (False, 8, True, 1, False, False),
    (True, 1, True, 1, False, False),
    (True, 8, False, 1, False, False),
    (False, 1, True, 8, False, True),
    (False, 1, True, 1, True, True),
])
def test_v4_pcp_enables_deepep_without_changing_other_dispatch(
    v4, pcp, ep, dp, sp, expected,
):
    from vllm_hcu.model_executor.layers.fused_moe.config_runtime import (
        use_all2all_kernels,
    )
    config = SimpleNamespace(_hcu_v4_pcp=v4, pcp_size=pcp, use_ep=ep,
                             dp_size=dp, is_sequence_parallel=sp)
    assert use_all2all_kernels(config) is expected


@pytest.mark.parametrize("arch,expected", [(None, 8),
    ("DeepseekV4ForCausalLM", 1), ("Glm4MoeForCausalLM", 8)])
def test_block_table_warmup_without_model_context(monkeypatch, arch, expected):
    import vllm.config
    from vllm_hcu.patch.worker.framework_opt import patch_deepseek_v4_pcp_block_table as patch
    config = None if arch is None else SimpleNamespace(
        model_config=SimpleNamespace(hf_config=SimpleNamespace(architectures=[arch])),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=8))
    monkeypatch.setattr(vllm.config, "get_current_vllm_config_or_none", lambda: config)
    class BlockTable:
        def __init__(self):
            self.pcp_world_size = 8
            self.pcp_rank = 3
    module = ModuleType(patch.TARGET_MODULE)
    module.BlockTable = BlockTable
    assert patch.apply_to_module(module)
    assert not patch.apply_to_module(module)
    table = module.BlockTable()
    assert table.pcp_world_size == expected
    assert table.pcp_rank == (0 if expected == 1 else 3)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_scatter_fp8_preserves_encoded_bytes(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU required for ROCm FP8 regression")
    from vllm_hcu.model_executor.layers.deepseek_v4_pcp import scatter_tokens
    plan = build_plan([17, 1], 8, 2, device=device)
    nd = 1
    local = torch.arange((nd + plan.local_sizes[2])*4, device=device).reshape(-1, 4).to(torch.float8_e4m3fn)
    output = scatter_tokens(local, plan, nd)
    expected = torch.zeros(nd + plan.seq_len, 4, dtype=torch.uint8, device=device)
    rows = torch.cat((torch.arange(nd, device=device), plan.local_indices + nd))
    expected.index_copy_(0, rows, local.view(torch.uint8))
    assert output.dtype == local.dtype
    torch.testing.assert_close(output.view(torch.uint8), expected)


@pytest.mark.parametrize('invalid_dcp', [False, True])
def test_pcp_validator_reuses_normalization_and_restores_eplb(invalid_dcp):
    from vllm.config import parallel
    from vllm_hcu.patch.platform.core_fix import patch_deepseek_v4_pcp_parallel
    patch_deepseek_v4_pcp_parallel.apply_to_module(parallel)
    config = parallel.ParallelConfig(
        prefill_context_parallel_size=2, enable_expert_parallel=True,
        enable_eplb=True, distributed_executor_backend='mp',
    )
    eplb = config.eplb_config
    eplb.num_redundant_experts = 2
    config.all2all_backend = 'naive'
    config.decode_context_parallel_size = 2 if invalid_dcp else 1
    if invalid_dcp:
        with pytest.raises(ValueError, match='divisible'):
            config._validate_parallel_config()
    else:
        assert config._validate_parallel_config() is config
    assert config.all2all_backend == 'allgather_reducescatter'
    assert config.enable_eplb and config.eplb_config is eplb
    assert eplb.num_redundant_experts == 2
    assert config.tensor_parallel_size == config.data_parallel_size == 1
