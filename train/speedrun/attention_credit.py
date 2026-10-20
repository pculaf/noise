"""Tiled implementation of PR26's existing local attention-credit algebra.

Standard online-softmax/FlashAttention tiling is prior work. Unlike the JAX PR25
coordinate-probe kernels, this preserves the actual AC/Gq/Gk/Gv calculation in
the accepted PyTorch fast_qkv_layer_kernel. No autograd/backward API is called.
"""
import math
import torch
import triton
import triton.language as tl


@triton.jit
def _dot(a,b,PRECISION:tl.constexpr):
    if PRECISION=='tf32rn':
        a=tl.inline_asm_elementwise('cvt.rn.tf32.f32 $0, $1;',constraints='=f,f',args=[a],dtype=tl.float32,is_pure=True,pack=1)
        b=tl.inline_asm_elementwise('cvt.rn.tf32.f32 $0, $1;',constraints='=f,f',args=[b],dtype=tl.float32,is_pure=True,pack=1)
        return tl.dot(a,b,input_precision='tf32')
    else:
        return tl.dot(a,b,input_precision=PRECISION)


@torch.compile(fullgraph=True,dynamic=False)
def prepare(q,k,cos,sin):
    d=q.shape[-1]//2
    q=torch.cat([q[...,:d]*cos+q[...,d:]*sin,-q[...,:d]*sin+q[...,d:]*cos],-1)
    k=torch.cat([k[...,:d]*cos+k[...,d:]*sin,-k[...,:d]*sin+k[...,d:]*cos],-1)
    eps=torch.finfo(torch.float32).eps
    return (q*torch.rsqrt(q.square().mean(-1,keepdim=True)+eps)).contiguous(), \
           (k*torch.rsqrt(k.square().mean(-1,keepdim=True)+eps)).contiguous()


@triton.jit
def _query(Q,K,V,H,GQ,LSE,Mean,T:tl.constexpr,D:tl.constexpr,WINDOW:tl.constexpr,
           BQ:tl.constexpr,BK:tl.constexpr,PRECISION:tl.constexpr):
    head,tile=tl.program_id(0),tl.program_id(1)
    qs=tile*BQ+tl.arange(0,BQ)
    ds=tl.arange(0,D)
    q=tl.load(Q+head*T*D+qs[:,None]*D+ds[None,:],qs[:,None]<T,0)
    hub=tl.load(H+head*T*D+qs[:,None]*D+ds[None,:],qs[:,None]<T,0)
    maximum=tl.full((BQ,),float('-inf'),tl.float32)
    denom=tl.zeros((BQ,),tl.float32)
    mean=tl.zeros((BQ,),tl.float32)
    first_moment=tl.zeros((BQ,D),tl.float32)
    mixed_moment=tl.zeros((BQ,D),tl.float32)
    first=tl.maximum(0,(tile*BQ-WINDOW)//BK)
    last=tl.minimum(tl.cdiv(T,BK),tl.cdiv((tile+1)*BQ,BK))
    for j in range(first,last):
        ks=j*BK+tl.arange(0,BK)
        key=tl.load(K+head*T*D+ks[:,None]*D+ds[None,:],ks[:,None]<T,0)
        value=tl.load(V+head*T*D+ks[:,None]*D+ds[None,:],ks[:,None]<T,0)
        logits=_dot(q,tl.trans(key),PRECISION)*(D**-.5)
        allowed=(qs[:,None]>=ks[None,:]) & (qs[:,None]-ks[None,:]<=WINDOW) & (ks[None,:]<T)
        logits=tl.where(allowed,logits,float('-inf'))
        newmax=tl.maximum(maximum,tl.max(logits,1))
        rescale=tl.exp(maximum-newmax)
        prob=tl.exp(logits-newmax[:,None])
        hv=_dot(hub,tl.trans(value),PRECISION)
        denom=denom*rescale+tl.sum(prob,1)
        mean=mean*rescale+tl.sum(prob*hv,1)
        first_moment=first_moment*rescale[:,None]+_dot(prob,key,PRECISION)
        mixed_moment=mixed_moment*rescale[:,None]+_dot(prob*hv,key,PRECISION)
        maximum=newmax
    expected=mean/denom
    gq=(mixed_moment/denom[:,None]-expected[:,None]*(first_moment/denom[:,None]))*(D**-.5)
    tl.store(GQ+head*T*D+qs[:,None]*D+ds[None,:],gq,qs[:,None]<T)
    tl.store(LSE+head*T+qs,maximum+tl.log(denom),qs<T)
    tl.store(Mean+head*T+qs,expected,qs<T)


@triton.jit
def _key_value(Q,K,V,H,LSE,Mean,GK,GV,T:tl.constexpr,D:tl.constexpr,WINDOW:tl.constexpr,
               BQ:tl.constexpr,BK:tl.constexpr,PRECISION:tl.constexpr):
    head,tile=tl.program_id(0),tl.program_id(1)
    ks=tile*BK+tl.arange(0,BK);ds=tl.arange(0,D)
    key=tl.load(K+head*T*D+ks[:,None]*D+ds[None,:],ks[:,None]<T,0)
    value=tl.load(V+head*T*D+ks[:,None]*D+ds[None,:],ks[:,None]<T,0)
    gk=tl.zeros((BK,D),tl.float32);gv=tl.zeros((BK,D),tl.float32)
    first=(tile*BK)//BQ
    last=tl.minimum(tl.cdiv(T,BQ),tl.cdiv((tile+1)*BK+WINDOW,BQ))
    for j in range(first,last):
        qs=j*BQ+tl.arange(0,BQ)
        q=tl.load(Q+head*T*D+qs[:,None]*D+ds[None,:],qs[:,None]<T,0)
        hub=tl.load(H+head*T*D+qs[:,None]*D+ds[None,:],qs[:,None]<T,0)
        lse=tl.load(LSE+head*T+qs,qs<T,0)
        mean=tl.load(Mean+head*T+qs,qs<T,0)
        logits=_dot(q,tl.trans(key),PRECISION)*(D**-.5)
        allowed=(qs[:,None]>=ks[None,:]) & (qs[:,None]-ks[None,:]<=WINDOW) & (qs[:,None]<T) & (ks[None,:]<T)
        prob=tl.where(allowed,tl.exp(logits-lse[:,None]),0.)
        hv=_dot(hub,tl.trans(value),PRECISION)
        ac=prob*(hv-mean[:,None])
        gk+=_dot(tl.trans(ac),q,PRECISION)*(D**-.5)
        gv+=_dot(tl.trans(prob),hub,PRECISION)
    tl.store(GK+head*T*D+ks[:,None]*D+ds[None,:],gk,ks[:,None]<T)
    tl.store(GV+head*T*D+ks[:,None]*D+ds[None,:],gv,ks[:,None]<T)


@torch.compile(fullgraph=True,dynamic=False)
def finish(q_raw,k_raw,gq,gk,gv,cos,sin,sigma):
    hd=q_raw.shape[-1];d=hd//2;eps=torch.finfo(torch.float32).eps
    def moved(raw,g):
        u=torch.cat([g[...,:d]*cos-g[...,d:]*sin,g[...,:d]*sin+g[...,d:]*cos],-1)
        sq=raw.square().sum(-1,keepdim=True)
        step=sigma*(sq/hd).sqrt().clamp(min=1e-6)
        inv0=torch.rsqrt(sq/hd+eps)
        inv=torch.rsqrt((sq+2*step*raw+step.square())/hd+eps)
        return u*inv+(raw*u).sum(-1,keepdim=True)*((inv-inv0)/step)
    return moved(q_raw,gq),moved(k_raw,gk),gv


def local_credit(q_raw,k_raw,values,hub,cos,sin,allowed,key_sigma,*,window,
                 precision='tf32x3',block=32):
    h,t,d=q_raw.shape
    if d!=64 or t%block:raise ValueError('First implementation requires head dim 64 and complete tiles')
    q,k=prepare(q_raw,k_raw,cos,sin)
    v,hub=values.contiguous(),hub.contiguous()
    gq,gk,gv=torch.empty_like(q),torch.empty_like(q),torch.empty_like(q)
    lse=torch.empty((h,t),device=q.device);mean=torch.empty_like(lse)
    _query[(h,triton.cdiv(t,block))](q,k,v,hub,gq,lse,mean,t,d,window,block,block,precision,num_warps=4)
    _key_value[(h,triton.cdiv(t,block))](q,k,v,hub,lse,mean,gk,gv,t,d,window,block,block,precision,num_warps=4)
    return finish(q_raw,k_raw,gq,gk,gv,cos,sin,key_sigma)
