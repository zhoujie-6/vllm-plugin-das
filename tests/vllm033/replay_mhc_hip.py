#!/usr/bin/env python3
"""CPU public CustomOp integration; AITER/device and RMSNorm are explicit doubles.
Written before production implementation. Uses native Torch math as reference.
"""
from __future__ import annotations
import argparse
from contextlib import ExitStack
import hashlib
import importlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import patch
import torch
from hcu_adapter.extensions import ExtensionContext, ExtensionCommitError
from vllm.model_executor import custom_op
from vllm.model_executor.layers import mhc
reference = importlib.import_module('vllm.model_executor.kernels.mhc.torch')

RESULTS=[]
CALLS=[]
CONTEXT=ExtensionContext('0.1.0','0.33.0')
CLASSES=(mhc.MHCPreOp,mhc.MHCPreDelayedOp,mhc.MHCPostOp,mhc.MHCFusedPostPreOp)
try:
    activation=importlib.import_module('hcu_vllm_das.mhc_activation')
    runtime=importlib.import_module('hcu_vllm_das.mhc_runtime')
except ModuleNotFoundError as error:
    if error.name not in ('hcu_vllm_das.mhc_activation','hcu_vllm_das.mhc_runtime'): raise
    activation=runtime=None


def case(name,fn):
    try: fn(); RESULTS.append({'case':name,'status':'passed'})
    except Exception as error: RESULTS.append({'case':name,'status':'failed','error':f'{type(error).__name__}: {error}'})


def require():
    assert activation is not None,'mHC production compatibility module is missing'


def expect_error(fn,kind=Exception):
    try: fn()
    except kind: return
    raise AssertionError('required error did not occur')


def norm(x,w,eps):
    if w is None:return x
    return (x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+eps)*w.float()).to(x.dtype)


def mixes(projected,sq,scale,base,width,eps,pre_eps,sink_eps,post_mult,repeat):
    m=projected*torch.rsqrt(sq[:,None]/width+eps)
    pre=torch.sigmoid(m[:,:4]*scale[0]+base[:4])+pre_eps
    post=(torch.sigmoid(m[:,4:8]*scale[1]+base[4:8])*post_mult).unsqueeze(-1)
    comb=torch.softmax(m[:,8:].view(-1,4,4)*scale[2]+base[8:].view(1,4,4),-1)+sink_eps
    comb=comb/(comb.sum(-2,keepdim=True)+sink_eps)
    for _ in range(repeat-1):
        comb=comb/(comb.sum(-1,keepdim=True)+sink_eps)
        comb=comb/(comb.sum(-2,keepdim=True)+sink_eps)
    return post,comb,pre


def provider_pre(residual,fn,hc_scale,hc_base,rms_eps=1e-6,hc_pre_eps=1e-6,hc_sinkhorn_eps=1e-6,hc_post_mult_value=1.,sinkhorn_repeat=20,use_tf32=False):
    CALLS.append('pre'); assert use_tf32 is False
    return reference.mhc_pre_torch(residual,fn,hc_scale,hc_base,rms_eps,hc_pre_eps,hc_sinkhorn_eps,hc_post_mult_value,sinkhorn_repeat)


def provider_post(out,x,residual,post_layer_mix,comb_res_mix):
    CALLS.append('post');out.copy_(reference.mhc_post_torch(x,residual,post_layer_mix,comb_res_mix))


def provider_project(out,sqrsum,x,fn,tile_k=128,use_tf32=False):
    CALLS.append('project');assert out.shape[0]==1 and tile_k==64 and not use_tf32
    flat=x.flatten(1).float();out[0].copy_(flat@fn.t());sqrsum[0].copy_(flat.square().sum(-1))


def provider_big_fuse(post_mix,comb_mix,layer_input,gemm_out_mul,gemm_out_sqrsum,hc_scale,hc_base,residual,rms_eps=1e-6,hc_pre_eps=1e-6,hc_sinkhorn_eps=1e-6,hc_post_mult_value=1.,sinkhorn_repeat=20):
    CALLS.append('big_fuse')
    post,comb,pre=mixes(gemm_out_mul.sum(0),gemm_out_sqrsum.sum(0),hc_scale,hc_base,
        residual.shape[1]*residual.shape[2],rms_eps,hc_pre_eps,hc_sinkhorn_eps,hc_post_mult_value,sinkhorn_repeat)
    post_mix.copy_(post);comb_mix.copy_(comb)
    layer_input.copy_((residual.float()*pre.unsqueeze(-1)).sum(1).to(residual.dtype))


PROVIDER=SimpleNamespace(mhc_pre=provider_pre,mhc_post=provider_post,
    mhc_pre_gemm_sqrsum=provider_project,mhc_pre_big_fuse=provider_big_fuse)


def tensors(seed,hidden=1024,tokens=2,narrow=False):
    g=torch.Generator().manual_seed(seed)
    residual=(torch.randn(tokens,4,hidden,generator=g)*.05).to(torch.bfloat16)
    fn=torch.randn(24,hidden if narrow else 4*hidden,generator=g)*.01
    scale=torch.tensor([.1,.2,.3]);base=torch.randn(24,generator=g)*.01
    pre=torch.rand(tokens,4,generator=g)
    x=(torch.randn(tokens,hidden,generator=g)*.05).to(torch.bfloat16)
    post=torch.rand(tokens,4,1,generator=g);comb=torch.rand(tokens,4,4,generator=g)
    return residual,fn,scale,base,pre,x,post,comb


def compare(left,right):
    if isinstance(left,torch.Tensor):left,right=(left,),(right,)
    assert len(left)==len(right)
    for a,b in zip(left,right):
        assert a.shape==b.shape and a.dtype==b.dtype and a.device==b.device
        torch.testing.assert_close(a,b,atol=.002 if a.dtype==torch.bfloat16 else 2e-5,rtol=.01 if a.dtype==torch.bfloat16 else 2e-5)


def configure(stack,enabled):
    for k,v in {'VLLM_HCU_USE_CUSTOM_OPS':'1','VLLM_HCU_USE_AITER_MHC':'1' if enabled else '0',
                'VLLM_HCU_USE_AITER_MHC_PRE':'1','VLLM_HCU_USE_AITER_MHC_POST':'1',
                'VLLM_HCU_USE_AITER_TILELANG_MHC':'0','VLLM_HCU_USE_AITER_MHC_FUSED':'0'}.items():
        stack.enter_context(patch.dict(os.environ,{k:v}))
    stack.enter_context(patch.object(custom_op,'op_registry_oot',dict(custom_op.op_registry_oot)))
    stack.enter_context(patch.object(activation,'_state',None))
    stack.enter_context(patch.object(activation,'_failure',None))
    stack.enter_context(patch.object(activation,'_is_hcu',return_value=True))
    stack.enter_context(patch.object(runtime,'v41_model',return_value=True))
    stack.enter_context(patch.object(runtime,'_gpu_tensors',side_effect=lambda tensors:all(t.device==tensors[0].device and t.is_contiguous() for t in tensors)))
    stack.enter_context(patch.object(runtime,'load_provider',return_value=PROVIDER))
    stack.enter_context(patch.object(mhc,'_apply_mhc_norm',side_effect=norm))
    cfg=SimpleNamespace(enabled_custom_ops=set(),disabled_custom_ops=set())
    stack.enter_context(patch.object(custom_op,'get_cached_compilation_config',return_value=cfg))
    stack.enter_context(patch.object(custom_op,'current_platform',SimpleNamespace(is_rocm=lambda:True)))
    stack.enter_context(patch.object(mhc,'HAS_TILELANG_MHC',True))
    stack.enter_context(patch.object(mhc,'HAS_AITER_MHC',True))
    activation.activate_extension(CONTEXT)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);args=parser.parse_args()
    torch.set_num_threads(1)
    originals=tuple(c.forward_hip for c in CLASSES)
    def core_reference_native():
        require()
        with ExitStack() as stack:
            configure(stack,False)
            pre,delayed,post,fused=[c() for c in CLASSES]
            assert all(type(o).__module__=='hcu_vllm_das.mhc_ops' for o in (pre,delayed,post,fused))
            for seed in range(200):
                r,fn,s,b,pm,x,p,c=tensors(seed,hidden=32,tokens=seed%3+1)
                eps=(1e-6,1e-6,1e-6,2.,3)
                w=torch.ones(32);neps=1e-5
                expected=reference.mhc_pre_torch(r,fn,s,b,*eps)
                compare(pre(r,fn,s,b,*eps,norm_weight=w,norm_eps=neps),(*expected[:2],norm(expected[2],w,neps)))
                rn=reference.mhc_post_torch(x,r,p,c)
                expected=reference.mhc_pre_delayed_torch(rn,fn,s,b,*eps,pre_mix=pm)
                compare(delayed(r,fn,s,b,*eps,pre_mix=pm,norm_weight=w,norm_eps=neps,sublayer_out=x,post_layer_mix=p,comb_res_mix=c),
                    (rn,*expected[:2],norm(expected[2],w,neps),expected[3]))
                expected=reference.mhc_pre_torch(rn,fn,s,b,*eps)
                compare(fused(x,r,p,c,fn,s,b,*eps,norm_weight=w,norm_eps=neps),(rn,*expected[:2],norm(expected[2],w,neps)))
                compare(post(x,r,p,c),rn)
            assert not CALLS
    case('200_seed_actual_OOT_native_fallback_pre_delayed_post_fused',core_reference_native)

    def provider_reference():
        require()
        with ExitStack() as stack:
            configure(stack,True)
            pre,delayed,post,fused=[c() for c in CLASSES]
            for seed in range(200):
                narrow=seed%2==0
                r,fn,s,b,pm,x,p,c=tensors(seed,tokens=seed%2+1,narrow=narrow)
                eps=(1e-6,1e-6,1e-6,2.,3)
                rn=reference.mhc_post_torch(x,r,p,c)
                expected=reference.mhc_pre_delayed_torch(rn,fn,s,b,*eps,pre_mix=pm,x=x if narrow else None)
                compare(delayed(r,fn,s,b,*eps,pre_mix=pm,x=x if narrow else None,sublayer_out=x,post_layer_mix=p,comb_res_mix=c),
                    (rn,*expected))
                if not narrow:
                    compare(pre(r,fn,s,b,*eps),reference.mhc_pre_torch(r,fn,s,b,*eps))
            assert all(name in CALLS for name in ('pre','post','project','big_fuse'))
    case('200_seed_HIP_provider_ABI_and_previous_mix_narrow_projection',provider_reference)

    def matrix():
        require()
        with ExitStack() as stack:
            configure(stack,True)
            pre,delayed,post,fused=[c() for c in CLASSES]
            for hidden in (1024,4096,5120,7168):
                r,fn,s,b,pm,x,p,c=tensors(4,hidden=hidden,tokens=1)
                CALLS.clear();pre(r,fn,s,b,1e-6,1e-6,1e-6,2.,3)
                assert ('pre' in CALLS)==(hidden!=5120)
                CALLS.clear();post(x,r,p,c)
                assert ('post' in CALLS)==(hidden!=5120)
            r,fn,s,b,pm,x,p,c=tensors(4,tokens=0)
            CALLS.clear();eps=(1e-6,1e-6,1e-6,2.,3)
            assert pre(r,fn,s,b,*eps)[2].shape==(0,1024)
            assert delayed(r,fn,s,b,*eps,pre_mix=pm)[4].shape==(0,4)
            assert post(x,r,p,c).shape==r.shape
            assert fused(x,r,p,c,fn,s,b,*eps)[3].shape==(0,1024)
            assert not CALLS
            r,fn,s,b,pm,x,p,c=tensors(4,tokens=2)
            CALLS.clear();post(x,r.transpose(0,1).contiguous().transpose(0,1),p,c)
            assert 'post' not in CALLS
    case('width5120_asymmetry_empty_and_noncontiguous_routing',matrix)

    def registration():
        require()
        with ExitStack() as stack:
            configure(stack,False)
            before=dict(custom_op.op_registry_oot);activation.activate_extension(CONTEXT);activation.check_health()
            assert custom_op.op_registry_oot==before
            key=CLASSES[0].__name__;custom_op.op_registry_oot[key]=object
            expect_error(activation.check_health,ExtensionCommitError)
            expect_error(lambda:activation.activate_extension(CONTEXT),ExtensionCommitError)
        assert originals==tuple(c.forward_hip for c in CLASSES)
    case('idempotence_lost_owner_sticky_and_native_symbol_identity',registration)

    def missing_and_failure():
        require()
        with ExitStack() as stack:
            configure(stack,True)
            r,fn,s,b,pm,x,p,c=tensors(7)
            obj=mhc.MHCPostOp()
            with patch.object(runtime,'provider',SimpleNamespace(mhc_pre=None,mhc_post=None,mhc_pre_gemm_sqrsum=None,mhc_pre_big_fuse=None)):
                compare(obj(x,r,p,c),reference.mhc_post_torch(x,r,p,c))
            error=RuntimeError('provider wrote then failed')
            with patch.object(runtime.provider,'mhc_post',side_effect=error):
                try:obj(x,r,p,c)
                except RuntimeError as caught:assert caught is error
                else:raise AssertionError('provider error swallowed')
    case('optional_absence_native_and_execution_failure_propagation',missing_and_failure)

    def activation_edges():
        require()
        with ExitStack() as stack:
            configure(stack,False)
            with patch.dict(os.environ,{'VLLM_HCU_USE_AITER_MHC_PRE':'0'}):
                expect_error(activation.check_health,ExtensionCommitError)
        with ExitStack() as stack:
            stack.enter_context(patch.object(activation,'_state',None));stack.enter_context(patch.object(activation,'_failure',None))
            stack.enter_context(patch.object(activation,'_is_hcu',return_value=False))
            before=dict(custom_op.op_registry_oot);activation.activate_extension(CONTEXT)
            assert custom_op.op_registry_oot==before
        with ExitStack() as stack:
            stack.enter_context(patch.object(activation,'_state',None));stack.enter_context(patch.object(activation,'_failure',None))
            stack.enter_context(patch.object(activation,'_is_hcu',return_value=True))
            stack.enter_context(patch.dict(os.environ,{'VLLM_HCU_USE_AITER_TILELANG_MHC':'1'}))
            expect_error(lambda:activation.activate_extension(CONTEXT),ValueError)
    case('policy_drift_nonHCU_and_TileLang_request_rejection',activation_edges)
    def foreign_and_partial():
        require()
        with ExitStack() as stack:
            configure(stack,False)
            stack.enter_context(patch.object(activation,'_state',None));stack.enter_context(patch.object(activation,'_failure',None))
            custom_op.op_registry_oot['MHCPreOp']=object
            before=dict(custom_op.op_registry_oot)
            expect_error(lambda:activation.activate_extension(CONTEXT),RuntimeError)
            assert custom_op.op_registry_oot==before
        with ExitStack() as stack:
            configure(stack,False)
            stack.enter_context(patch.object(activation,'_state',None));stack.enter_context(patch.object(activation,'_failure',None))
            custom_op.op_registry_oot.clear()
            register=custom_op.CustomOp.register_oot.__func__
            def broken(cls,_decorated_op_cls=None,name=None):
                decorator=register(cls,_decorated_op_cls,name)
                def publish(value):
                    result=decorator(value)
                    raise RuntimeError('published then interrupted')
                return publish
            with patch.object(custom_op.CustomOp,'register_oot',classmethod(broken)):
                expect_error(lambda:activation.activate_extension(CONTEXT),ExtensionCommitError)
                expect_error(lambda:activation.activate_extension(CONTEXT),ExtensionCommitError)
    case('foreign_owner_and_partial_publication',foreign_and_partial)
    assert not torch.cuda.is_initialized()
    root=Path(__file__).resolve().parents[2]
    sha=lambda p:subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip()
    sources={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'adapter_extension/src/hcu_vllm_das').glob('mhc*.py')}
    report={'level':'CPU real CustomOp dispatch and Torch reference; AITER,device,norm doubled','plugin':sha(root),
        'core':sha(root.parent/'vllm'),'adapter':sha(root.parent/'adapter-das'),'installed_vllm_metadata':metadata.version('vllm'),
        'cuda_initialized':False,'source_sha256':sources,'cases':RESULTS,
        'passed':sum(r['status']=='passed' for r in RESULTS),'total':len(RESULTS)}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    return 0 if report['passed']==report['total'] else 1

if __name__=='__main__':raise SystemExit(main())
