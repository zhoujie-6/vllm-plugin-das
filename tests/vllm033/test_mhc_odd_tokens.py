from contextlib import ExitStack
import torch
from replay_mhc_hip import CALLS, CLASSES, configure, tensors, compare, reference, require


def test_odd_token_5120_reference_route():
    require()
    with ExitStack() as stack:
        configure(stack,True)
        pre,delayed,post,fused=[c() for c in CLASSES]
        eps=(1e-6,1e-6,1e-6,2.,20)
        for tokens in (1,2,3,4,7,8,31,32,1):
            r,fn,s,b,pm,x,p,c=tensors(417+tokens,hidden=5120,tokens=tokens)
            original=[v.clone() for v in (r,fn,s,b,pm,x,p,c)]
            CALLS.clear()
            actual=pre(r,fn,s,b,*eps)
            compare(actual,reference.mhc_pre_torch(r,fn,s,b,*eps))
            assert ('pre' in CALLS)==(tokens%2==0),CALLS
            CALLS.clear()
            actual=delayed(r,fn,s,b,*eps,pre_mix=pm,
                sublayer_out=x,post_layer_mix=p,comb_res_mix=c)
            residual=reference.mhc_post_torch(x,r,p,c)
            expected=reference.mhc_pre_delayed_torch(residual,fn,s,b,*eps,pre_mix=pm)
            compare(actual,(residual,*expected))
            assert ('project' in CALLS)==(tokens%2==0),CALLS
            assert ('big_fuse' in CALLS)==(tokens%2==0),CALLS
            assert all(torch.equal(a,b) for a,b in zip((r,fn,s,b,pm,x,p,c),original))
