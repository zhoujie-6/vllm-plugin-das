# SPDX-License-Identifier: Apache-2.0
"""Run explicitly with torchrun --standalone --nproc_per_node=8.

Default: FP8 eager, idle-rank and graph replay checks.
Add --int8 to check the W4A8 adapter; requires DG_MEGA_BUILD_INT8=1.
Allocates under 2 GiB per rank and does not load a model.
"""


def main():
    import os,sys,importlib.util
    from types import ModuleType,SimpleNamespace as NS
    import torch
    import torch.distributed as dist
    import deepgemm
    rank=int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("gloo")
    group=dist.group.WORLD
    E,H,I,K,C=256,4096,2048,6,2
    buf=deepgemm.get_symm_buffer_for_mega_moe(group,E,C,K,H,I)
    from pathlib import Path
    root=str(Path(__file__).resolve().parents[2])
    import triton,triton.language as tl
    mod=ModuleType("vllm.triton_utils");mod.triton=triton;mod.tl=tl
    sys.modules[mod.__name__]=mod
    spec=importlib.util.spec_from_file_location("smoke_int4",root+"/vllm_hcu/model_executor/layers/fused_moe/hcu_int4_megamoe.py")
    int4=importlib.util.module_from_spec(spec);spec.loader.exec_module(int4)
    print(f"rank={rank} DEEPGEMM={deepgemm.__file__}",flush=True)
    for mode in (["fp8","int8"] if "--int8" in sys.argv else ["fp8"]):
     if mode=="fp8":
      w1=torch.full((32,2*I,H),1/512,dtype=torch.float8_e4m3fn,device="cuda")
      w2=torch.full((32,H,I),1/512,dtype=torch.float8_e4m3fn,device="cuda")
      s1=torch.ones((32,2*I),device="cuda");s2=torch.ones((32,H),device="cuda")
      l1,l2=deepgemm.transform_weights_for_mega_moe((w1,s1),(w2,s2))
      del w1,w2
     else:
      w1=torch.full((32,2*I,H//2),0x11,dtype=torch.uint8,device="cuda")
      w2=torch.full((32,H,I//2),0x11,dtype=torch.uint8,device="cuda")
      l1,l2=int4.transform_int4_weights_for_mega_moe_normal(
       w1,w2,l1_scale=torch.full((32,2*I),1/8192,device="cuda"),
       l2_scale=torch.full((32,H),1/8192,device="cuda"),prepack=False)
     for idle in [False,True]:
      n=0 if idle and rank==0 else C
      x=torch.full((n,H),1/32,dtype=torch.bfloat16,device="cuda")
      ids=(torch.arange(n*K,device="cuda").reshape(n,K)+rank*32).remainder(E).long()
      ws=torch.full((n,K),1/K,device="cuda")
      buf.topk_idx.fill_(-1);buf.topk_weights.zero_()
      if n: deepgemm.mega_moe_preprocess(x,ids,ws,buf,n,is_int8=mode=="int8")
      y=torch.empty_like(x)
      if mode=="fp8":deepgemm.fp8_mega_moe(y,l1,l2,buf,activation_clamp=10.)
      else:int4.int4_w4a8_mega_moe(y,l1,l2,buf,activation_clamp=10.)
      torch.cuda.synchronize()
      expected=4*torch.nn.functional.silu(torch.tensor(.25))*.25
      if n:torch.testing.assert_close(y.float(),torch.full_like(y.float(),expected.item()),rtol=.04,atol=.004)
      print(f"PASS rank={rank} mode={mode} idle={idle} n={n} output={y.flatten()[:1].tolist()}",flush=True)
     if mode=="fp8":
      # Capture after eager warmup. Padded invalid routes represent an idle rank.
      x=torch.full((C,H),1/32,dtype=torch.bfloat16,device="cuda")
      ids=(torch.arange(C*K,device="cuda").reshape(C,K)+rank*32).remainder(E).long()
      ws=torch.full((C,K),1/K,device="cuda")
      y=torch.empty_like(x)
      graph=torch.cuda.CUDAGraph()
      with torch.cuda.graph(graph):
       buf.topk_idx.fill_(-1);buf.topk_weights.zero_()
       deepgemm.mega_moe_preprocess(x,ids,ws,buf,C)
       deepgemm.fp8_mega_moe(y,l1,l2,buf,activation_clamp=10.)
      graph.replay();torch.cuda.synchronize()
      torch.testing.assert_close(y.float(),torch.full_like(y.float(),expected.item()),rtol=.04,atol=.004)
      if rank==0:ids.fill_(-1);ws.zero_()
      graph.replay();torch.cuda.synchronize()
      if rank!=0:torch.testing.assert_close(y.float(),torch.full_like(y.float(),expected.item()),rtol=.04,atol=.004)
      print(f"PASS rank={rank} mode=fp8 graph replay active/idle",flush=True)
      del graph
     del l1,l2
     if mode=="int8":del w1,w2
     torch.cuda.empty_cache()
    dist.barrier()
    buf.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
