# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for the installed V41 MegaMoE plugin (no device claims)."""
import json
from types import SimpleNamespace as NS

import pytest
import torch
from hcu_vllm_das import v41_static_eplb as static
from hcu_vllm_das.moe.v41_megamoe import (
    MegaExperts, MegaPrepare, MegaKernel, validate_config, provider,
)
from hcu_vllm_das.v41_megamoe_model import base_init, HcuV41MegaMoE, MegaBase
from vllm.models.deepseek_v41.amd import model as native


def artifact(tmp_path, rows=None):
    rows = rows or [[1, 0, 1, 2, 3, 0, 2, 3]]
    data = dict(version=2, format="vllm_offline_eplb_physical_to_logical_by_model",
                model_maps={"DeepseekV41ForCausalLM":dict(
                    record_kind="proposal", model_name="/model",
                    model_class="DeepseekV41ForCausalLM",num_moe_layers=len(rows),
                    num_logical_experts=4,num_physical_experts=8,
                    num_redundant_experts=4,physical_to_logical_map=rows)})
    path=tmp_path/"map.json";path.write_text(json.dumps(data));return path


def config():
    return NS(parallel_config=NS(enable_eplb=False),model_config=NS(
        model="/model",hf_config=NS(num_hidden_layers=1,n_routed_experts=4)))


def test_existing_v2_plan_and_byte_change(tmp_path, monkeypatch):
    path=artifact(tmp_path);monkeypatch.setenv("VLLM_HCU_DSV41_STATIC_EXPERT_MAP",str(path))
    cfg=config();plan=static.static_plan(cfg)
    assert plan.layer_map(0)==(1,0,1,2,3,0,2,3)
    path.write_text(path.read_text().replace("[1, 0, 1, 2, 3, 0, 2, 3]","[0, 1, 1, 2, 3, 0, 2, 3]"))
    with pytest.raises(ValueError,match="changed"):static.static_plan(cfg)


def test_replica_routes_and_padding(tmp_path,monkeypatch):
    path=artifact(tmp_path);monkeypatch.setenv("VLLM_HCU_DSV41_STATIC_EXPERT_MAP",str(path))
    plan=static.static_plan(config());table=static.routing_table(plan,"model.layers.0.ffn.experts","cpu")
    import vllm.distributed
    monkeypatch.setattr(vllm.distributed,"get_ep_group",lambda:NS(rank_in_group=3))
    logical=torch.tensor([[0,1,2],[3,-1,0]])
    physical=static.map_routes(logical,table)
    row=plan.physical_to_logical_map[0]
    assert torch.equal(row[physical.clamp_min(0)][logical>=0],logical[logical>=0])
    assert physical[1,1]==-1
    assert torch.equal(logical,torch.tensor([[0,1,2],[3,-1,0]]))


@pytest.mark.parametrize("row",[[0,1,2,2,0,1,2,2],[True,1,2,3,0,1,2,3],[0,1,2,4,0,1,2,3]])
def test_invalid_placement_rejected(tmp_path,monkeypatch,row):
    path=artifact(tmp_path,[row]);monkeypatch.setenv("VLLM_HCU_DSV41_STATIC_EXPERT_MAP",str(path))
    with pytest.raises(ValueError):static.static_plan(config())


def test_loader_fans_out_to_physical_replicas(tmp_path,monkeypatch):
    path=artifact(tmp_path);monkeypatch.setenv("VLLM_HCU_DSV41_STATIC_EXPERT_MAP",str(path))
    plan=static.static_plan(config())
    class Layer:
        def weight_loader(self,param,loaded_weight,weight_name,shard_id,expert_id,return_success=False):
            param.data[expert_id].copy_(loaded_weight)
            return True if return_success else None
    layer=Layer();layer.moe_config=NS(num_experts=8,num_logical_experts=4)
    layer.expert_map_manager=NS(num_fused_shared_experts=0)
    for name in ("w13_weight","w2_weight","w13_weight_scale","w2_weight_scale"):
        setattr(layer,name,torch.nn.Parameter(torch.zeros(8,2),requires_grad=False))
    static.bind_weight_loaders(layer,plan,"model.layers.0.ffn.experts")
    for logical in range(4):
        p=layer.w13_weight
        assert p.weight_loader(p,torch.full((2,),float(logical+1)),"w13_weight","w1",logical,True)
    for physical,logical in enumerate(plan.layer_map(0)):
        assert torch.equal(layer.w13_weight[physical],torch.full((2,),float(logical+1)))


def test_native_constructor_and_mro():
    clone=base_init()
    assert clone.__code__ is native.DeepseekV4MoEBase.__init__.__code__
    assert clone.__globals__["FusedMoEFactory"] is not native.DeepseekV4MoEBase.__init__.__globals__["FusedMoEFactory"]
    mro=HcuV41MegaMoE.__mro__
    assert mro.index(native.DeepseekV4MoE)<mro.index(MegaBase)<mro.index(native.DeepseekV4MoEBase)


@pytest.mark.parametrize("rows",[0,1,7])
def test_collective_empty_rank_and_no_weight_rescaling(monkeypatch,rows):
    monkeypatch.setattr(torch.cuda,"is_current_stream_capturing",lambda:False)
    calls=[]
    class Vendor:
        def mega_moe_preprocess(self,x,ids,weights,buf,n):
            calls.append(("pre",n,ids.clone(),weights.clone()))
        def fp8_mega_moe(self,out,l1,l2,buf,**kw):
            calls.append(("launch",out.shape[0],kw));out.fill_(3)
    expert=object.__new__(MegaExperts)
    expert.moe_config=NS(hidden_dim=8,experts_per_token=2)
    expert.quant_config=NS(gemm1_clamp_limit=7.0)
    expert.buffer=NS(num_max_tokens_per_rank=16)
    expert.mega=Vendor();expert.l1=expert.l2=None
    x=torch.zeros(rows,8,dtype=torch.bfloat16)
    ids=torch.zeros(rows,2,dtype=torch.int64);weights=torch.full((rows,2),0.25)
    out=torch.empty_like(x);expert.run(out,x,ids,weights)
    assert calls[-1][0:2]==("launch",rows)
    assert calls[-1][2]["activation_clamp"]==7.0
    assert len(calls)==(2 if rows else 1)
    if rows:assert torch.equal(calls[0][3],weights)
    assert torch.equal(out,torch.full_like(out,3))


def test_prepare_owns_combine_and_rejects_input_weighting():
    prepare=MegaPrepare();x=torch.ones(2,4);ids=torch.zeros(2,1,dtype=torch.int64);w=torch.ones(2,1)
    assert prepare.output_is_reduced() and not prepare.supports_async()
    assert prepare.prepare(x,w,ids,8,None,False,None,True)[0] is x
    with pytest.raises(ValueError,match="after GEMM2"):
        prepare.prepare(x,w,ids,8,None,True,None,True)


def test_native_provider_signature():
    assert provider().__name__=="deepgemm.mega"

def test_static_factory_allocates_redundant_without_dynamic_worker(tmp_path,monkeypatch):
    from hcu_vllm_das import v41_megamoe_model as model
    path=artifact(tmp_path);monkeypatch.setenv("VLLM_HCU_DSV41_STATIC_EXPERT_MAP",str(path))
    monkeypatch.setattr(model,"get_current_vllm_config",config)
    calls=[]
    def factory(*args,**kwargs):
        calls.append(kwargs)
        return NS(router=NS(eplb_state=object()))
    monkeypatch.setattr(model,"FusedMoEFactory",factory)
    result=model.mega_factory(num_experts=4)
    assert calls==[dict(num_experts=4,enable_eplb=True,num_redundant_experts=4)]
    assert result.router.eplb_state is None


def test_dynamic_eplb_with_static_plan_rejected(tmp_path,monkeypatch):
    path=artifact(tmp_path);monkeypatch.setenv("VLLM_HCU_DSV41_STATIC_EXPERT_MAP",str(path))
    cfg=config();cfg.parallel_config.enable_eplb=True
    with pytest.raises(ValueError,match="dynamic"):static.static_plan(cfg)


def test_static_prefix_rejects_uncovered_draft(tmp_path,monkeypatch):
    path=artifact(tmp_path);monkeypatch.setenv("VLLM_HCU_DSV41_STATIC_EXPERT_MAP",str(path))
    plan=static.static_plan(config())
    with pytest.raises(ValueError,match="target layer"):static.routing_table(plan,"model.layers.1.ffn.experts","cpu")

@pytest.mark.parametrize("write_outputs",[False,True])
def test_startup_probe_rejects_nonexecuting_vendor(write_outputs):
    from hcu_vllm_das.moe.v41_megamoe import preflight_staging
    class Vendor:
        def mega_moe_preprocess(self,x,ids,weights,buf,rows):
            if write_outputs:
                buf.topk_idx.copy_(ids);buf.topk_weights.copy_(weights)
                buf.x.fill_(448);buf.x_sf.fill_(1/448)
    vendor=Vendor()
    if write_outputs:
        preflight_staging(vendor,"cpu")
    else:
        with pytest.raises(RuntimeError,match="startup execution probe"):
            preflight_staging(vendor,"cpu")


def test_rejected_reload_latches_forward_failure():
    from hcu_vllm_das.moe.v41_megamoe import HcuV41MegaMoEMethod
    method=object.__new__(HcuV41MegaMoEMethod)
    method._processed=True;method._failure=None
    with pytest.raises(RuntimeError,match="postprocessed twice"):
        method.process_weights_after_loading(None)
    with pytest.raises(RuntimeError,match="postprocessed twice"):
        method.apply(None,None,None,None,None,None)


def test_staging_routing_dtype_ignores_bf16_default():
    from hcu_vllm_das.moe.v41_megamoe import preflight_staging
    class Vendor:
        def mega_moe_preprocess(self,x,ids,weights,buf,rows):
            assert weights.dtype == buf.topk_weights.dtype == torch.float32
            assert buf.x_sf.dtype == torch.float32
            buf.topk_idx.copy_(ids);buf.topk_weights.copy_(weights)
            buf.x.fill_(448);buf.x_sf.fill_(1/448)
    previous=torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        preflight_staging(Vendor(),"cpu")
    finally:
        torch.set_default_dtype(previous)


def test_explicit_static_plan_is_not_silently_ignored(monkeypatch):
    from hcu_vllm_das import v41_pcp_activation as activation
    from vllm.model_executor.models import ModelRegistry
    from vllm import envs
    monkeypatch.setattr(activation,"_state",None)
    monkeypatch.setattr(activation,"_failure",None)
    monkeypatch.setattr(activation,"_is_hcu",lambda:True)
    monkeypatch.setattr(ModelRegistry,"models",dict(ModelRegistry.models))
    monkeypatch.setattr(envs,"environment_variables",dict(envs.environment_variables))
    monkeypatch.delenv("VLLM_HCU_DSV41_PCP_ATTENTION",raising=False)
    monkeypatch.delenv("VLLM_HCU_DSV41_ENGRAM_HOST",raising=False)
    monkeypatch.setenv("VLLM_HCU_DSV41_STATIC_EXPERT_MAP","/placement.json")
    activation.activate_extension(NS(vllm_version="0.33.0"))
    assert activation._key(ModelRegistry.models[activation._ARCH]) == activation._PLUGIN
    assert activation.policy_factor() == (False,False,"/placement.json")


def test_redundant_experts_align_native_ht_ring():
    from hcu_vllm_das.moe.v41_megamoe import make_buffer
    allocated=[]
    class Buffer:
        def __init__(self,cap):
            self.num_max_tokens_per_rank=cap
            self.l1_acts=NS(shape=(61440,5120))
            self.l2_acts=NS(shape=(48*cap+49*192,2304))
            self.destroyed=False
        def destroy(self):self.destroyed=True
    class Vendor:
        def get_symm_buffer_for_mega_moe(self,group,experts,cap,*args,**kwargs):
            buf=Buffer(cap);allocated.append(buf);return buf
    buf=make_buffer(Vendor(),NS(size=lambda:8),392,2048,6,5120,2304)
    assert buf.num_max_tokens_per_rank==2052
    assert buf.l2_acts.shape[0]%128==0
    assert allocated[0].destroyed and not buf.destroyed


def test_incompatible_native_pool_is_destroyed_before_rejection():
    from hcu_vllm_das.moe.v41_megamoe import make_buffer
    destroyed=[]
    class Vendor:
        def get_symm_buffer_for_mega_moe(self,*args,**kwargs):
            return NS(l1_acts=NS(shape=(61441,5120)),l2_acts=NS(shape=(128,2304)),
                      destroy=lambda:destroyed.append(True))
    with pytest.raises(RuntimeError,match="pool capacity"):
        make_buffer(Vendor(),NS(size=lambda:8),392,2048,6,5120,2304)
    assert destroyed==[True]
