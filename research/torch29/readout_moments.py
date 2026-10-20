"""Fused evaluation of PR29's six-product cubic readout expansion.

All six moments share the same vocabulary tiles, exponentials and direction
loads. The first product uses TF32x3; the other five use BF16, as in PR29.
This is an execution experiment; the polynomial formula is prior work.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _cycle(BASE,C,Y,STEP,SQ,DOTS,DS,OUT,N:tl.constexpr,V:tl.constexpr,P:tl.constexpr,W:tl.constexpr,
           BM:tl.constexpr,BP:tl.constexpr,BV:tl.constexpr,STABLE:tl.constexpr):
    mi=tl.program_id(0)*BM+tl.arange(0,BM)
    pj=tl.program_id(1)*BP+tl.arange(0,BP)
    sq=tl.load(SQ+mi,mi<N,1.)
    step=tl.load(STEP+mi,mi<N,1.)
    inv0=tl.rsqrt(sq/W+1.1920928955078125e-7)
    a0=tl.full((BM,BP),0,tl.float32);a1=tl.full((BM,BP),0,tl.float32)
    a2=tl.full((BM,BP),0,tl.float32);a3=tl.full((BM,BP),0,tl.float32)
    a4=tl.full((BM,BP),0,tl.float32);a5=tl.full((BM,BP),0,tl.float32)
    total=tl.full((BM,),0,tl.float32);r1=tl.full((BM,),0,tl.float32)
    r2=tl.full((BM,),0,tl.float32);r3=tl.full((BM,),0,tl.float32)
    for v in range(tl.cdiv(V,BV)):
        vi=v*BV+tl.arange(0,BV)
        base=tl.load(BASE+mi[:,None]*V+vi[None,:],(mi[:,None]<N)&(vi[None,:]<V),0.)
        c=tl.load(C+pj[None,:]*V+vi[:,None],(pj[None,:]<P)&(vi[:,None]<V),0.)
        z=base*inv0[:,None]
        tanh=2./(1.+tl.exp(2.*(-z/15.)))-1.
        f=tl.exp(15.*tanh)
        f=tl.where(vi[None,:]<V,f,0.)
        g1=1.-tanh*tanh
        d1=-2.*tanh*g1/15.
        d2=-2.*g1*(1.-3.*tanh*tanh)/225.
        g2=g1*g1+d1;g3=g1*g1*g1+3.*g1*d1+d2
        fz=f*z;fzz=fz*z
        a0+=tl.dot(f*g1,c,input_precision='tf32x3')
        c1=c.to(tl.bfloat16);c2=(c*c).to(tl.bfloat16);c3=(c*c*c).to(tl.bfloat16)
        a1+=tl.dot((fz*g2).to(tl.bfloat16),c1)
        a2+=tl.dot((fzz*g3).to(tl.bfloat16),c1)
        a3+=tl.dot((f*g2).to(tl.bfloat16),c2)
        a4+=tl.dot((fz*g3).to(tl.bfloat16),c2)
        a5+=tl.dot((f*g3).to(tl.bfloat16),c3)
        total+=tl.sum(f,1);r1+=tl.sum(fz*g1,1);r2+=tl.sum(fzz*g2,1);r3+=tl.sum(fzz*z*g3,1)
    dots=tl.load(DOTS+mi[:,None]*P+pj[None,:],(mi[:,None]<N)&(pj[None,:]<P),0.)
    ds=tl.load(DS+pj,pj<P,0.)
    inv=tl.rsqrt((sq[:,None]+2.*step[:,None]*dots+step[:,None]*step[:,None]*ds[None,:])/W+1.1920928955078125e-7)
    delta=inv/inv0[:,None]-1.;sigma=step[:,None]*inv
    ratio=(sigma*a0+delta*sigma*a1+.5*delta*delta*sigma*a2+.5*sigma*sigma*a3
           +.5*delta*sigma*sigma*a4+sigma*sigma*sigma/6.*a5
           +delta*r1[:,None]+.5*delta*delta*r2[:,None]+delta*delta*delta/6.*r3[:,None])
    y=tl.load(Y+mi,mi<N,0)
    by=tl.load(BASE+mi*V+y,mi<N,0.)
    cy=tl.load(C+pj[None,:]*V+y[:,None],(mi[:,None]<N)&(pj[None,:]<P),0.)
    # libdevice supplies log1p/tanh; their cancellation behaviour matters here.
    clean_t=tl.extra.cuda.libdevice.tanh(by*inv0/15.)
    if STABLE:
        change=(delta*(by*inv0)[:,None]+sigma*cy)/15.
        tm=tl.extra.cuda.libdevice.tanh(change)
        dy=15.*tm*(1.-clean_t[:,None]*clean_t[:,None])/(1.+clean_t[:,None]*tm)
        result=(tl.extra.cuda.libdevice.log1p(ratio/total[:,None])-dy)/step[:,None]
    else:
        clean=tl.log(total)-15.*clean_t
        moved=tl.log(total)[:,None]+tl.extra.cuda.libdevice.log1p(ratio/total[:,None])-15.*tl.extra.cuda.libdevice.tanh((by[:,None]+step[:,None]*cy)*inv/15.)
        result=(moved-clean[:,None])/step[:,None]
    tl.store(OUT+mi[:,None]*P+pj[None,:],result,(mi[:,None]<N)&(pj[None,:]<P))


def cycle(base,c,y,steps,sq,dots,ds,width,stable=True,bm=16,bp=32,bv=128):
    n,v=base.shape;p=c.shape[0]
    out=torch.empty((n,p),device=base.device)
    _cycle[(triton.cdiv(n,bm),triton.cdiv(p,bp))](base,c,y,steps,sq,dots,ds,out,n,v,p,width,bm,bp,bv,stable,num_warps=4)
    return out
