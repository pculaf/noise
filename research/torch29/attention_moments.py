"""Triton port of PR #29's degree-two attention moments; see PROVENANCE.md.

Unlike our older attention-credit kernel, this evaluates finite raw-coordinate
moves through both QK normalization and the attention softmax. No autograd.
"""
import torch
import triton
import triton.language as tl
from moments import rope32, closed_coefficients


@triton.jit
def _moments(Q, K, V, H, COEF, LSE, MEAN, OUT, T:tl.constexpr, D:tl.constexpr,
             WINDOW:tl.constexpr, QUERY:tl.constexpr, J:tl.constexpr, PAIRED:tl.constexpr, EARLY:tl.constexpr,
             BM:tl.constexpr, BN:tl.constexpr):
    head, tile, chunk = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    pos = tile * BM + tl.arange(0, BM)
    dims = tl.arange(0, D)
    feat = tl.arange(0, 2 * J)
    direction = chunk * J + feat % J
    column = direction % (D // 2) + (feat // J) * (D // 2)
    row = head * T * D + pos[:, None] * D + dims[None, :]
    moving = tl.load(Q + row) if QUERY else tl.load(K + row)
    own = tl.load(H + row) if QUERY else tl.load(V + row)
    # Four products: W[X,Y], W[X²,Y²], W[XY,XY], (W L)[X,Y].
    p0=tl.full((BM,2*J),0,tl.float32);p1=tl.full((BM,2*J),0,tl.float32)
    p2=tl.full((BM,2*J),0,tl.float32);p3=tl.full((BM,2*J),0,tl.float32)
    s1=tl.full((BM,),0,tl.float32);s2=tl.full((BM,),0,tl.float32)
    if QUERY:
        z0=tl.full((BM,2*J),0,tl.float32);z1=tl.full((BM,2*J),0,tl.float32)
        z2=tl.full((BM,2*J),0,tl.float32);z3=tl.full((BM,2*J),0,tl.float32)
        t1=tl.full((BM,),0,tl.float32);t2=tl.full((BM,),0,tl.float32)
        lse=tl.load(LSE+head*T+pos);mean=tl.load(MEAN+head*T+pos)
        first=tl.maximum(0,(tile*BM-WINDOW)//BN)
        last=tl.cdiv((tile+1)*BM,BN)
    else:
        first=tile*BM//BN
        last=tl.minimum(T//BN,tl.cdiv((tile+1)*BM+WINDOW,BN))
    if EARLY:
        cp=head*3*T*D+pos[:,None]*D+direction[None,:]
        aa=tl.load(COEF+cp);uu=tl.load(COEF+cp+T*D);ww=tl.load(COEF+cp+2*T*D)
        pp=tl.where(feat[None,:]<J,uu,ww);rr=tl.where(feat[None,:]<J,ww,uu)
        cp=cp+D//2
        ab=tl.load(COEF+cp);ub=tl.load(COEF+cp+T*D);wb=tl.load(COEF+cp+2*T*D)
        pb=tl.where(feat[None,:]<J,ub,wb);rb=tl.where(feat[None,:]<J,wb,ub)
        en0=tl.full((BM,2*J),0,tl.float32);en1=tl.full((BM,2*J),0,tl.float32)
        if QUERY:
            ed0=tl.full((BM,2*J),0,tl.float32);ed1=tl.full((BM,2*J),0,tl.float32)
    for n in range(first,last):
        other_pos=n*BN+tl.arange(0,BN)
        ptr=head*T*D+other_pos[:,None]*D+dims[None,:]
        other=tl.load(K+ptr) if QUERY else tl.load(Q+ptr)
        other_credit=tl.load(V+ptr) if QUERY else tl.load(H+ptr)
        logits=tl.dot(moving.to(tl.bfloat16),tl.trans(other.to(tl.bfloat16)))*(D**-.5)
        hv=tl.dot(own.to(tl.bfloat16),tl.trans(other_credit.to(tl.bfloat16)))
        if QUERY:
            lag=pos[:,None]-other_pos[None,:]
            prob=tl.exp(logits-lse[:,None]);c=hv-mean[:,None]
        else:
            lag=other_pos[None,:]-pos[:,None]
            lse=tl.load(LSE+head*T+other_pos);mean=tl.load(MEAN+head*T+other_pos)
            prob=tl.exp(logits-lse[None,:]);c=hv-mean[None,:]
        prob=tl.where((lag>=0)&(lag<=WINDOW),prob,0.)
        fptr=head*T*D+other_pos[:,None]*D+column[None,:]
        # Source features use fp32 normalized columns before casting monomials.
        # Q/K stored fp32 here; dot operands above are explicitly cast below.
        x=tl.load(K+fptr).to(tl.float32) if QUERY else tl.load(Q+fptr).to(tl.float32)
        paired=column+(tl.where(feat<J,D//2,-D//2))
        yptr=head*T*D+other_pos[:,None]*D+paired[None,:]
        y=tl.load(K+yptr).to(tl.float32) if QUERY else tl.load(Q+yptr).to(tl.float32)
        x=x*(D**-.5);y=y*(D**-.5)
        f0=x.to(tl.bfloat16);f1=(x*x).to(tl.bfloat16);f2=(x*y).to(tl.bfloat16)
        w1=prob*c
        w2=w1 if QUERY else (.5*prob-prob*prob)*c
        if EARLY:
            fct=0.5 if QUERY else 1.
            cross=1. if QUERY else 2.
            dp=tl.dot(w1.to(tl.bfloat16),f0)
            en0+=dp*pp;en1+=dp*pb
            dp=tl.dot(w2.to(tl.bfloat16),f1)
            en0+=fct*dp*pp*pp;en1+=fct*dp*pb*pb
            dp=tl.dot(w2.to(tl.bfloat16),f2)
            en0+=fct*dp*pp*rr;en1+=fct*dp*pb*rb
            dp=tl.dot((w2*logits).to(tl.bfloat16),f0)
            en0+=cross*dp*aa*pp;en1+=cross*dp*ab*pb
            ds1=tl.sum(w1*logits,1)[:,None];ds2=tl.sum(w2*logits*logits,1)[:,None]
            en0+=.5*ds1*aa+.5*fct*ds2*aa*aa
            en1+=.5*ds1*ab+.5*fct*ds2*ab*ab
            if QUERY:
                dp=tl.dot(prob.to(tl.bfloat16),f0)
                ed0+=dp*pp;ed1+=dp*pb
                dp=tl.dot(prob.to(tl.bfloat16),f1)
                ed0+=.5*dp*pp*pp;ed1+=.5*dp*pb*pb
                dp=tl.dot(prob.to(tl.bfloat16),f2)
                ed0+=.5*dp*pp*rr;ed1+=.5*dp*pb*rb
                dp=tl.dot((prob*logits).to(tl.bfloat16),f0)
                ed0+=dp*aa*pp;ed1+=dp*ab*pb
                ds1=tl.sum(prob*logits,1)[:,None];ds2=tl.sum(prob*logits*logits,1)[:,None]
                ed0+=.5*ds1*aa+.25*ds2*aa*aa
                ed1+=.5*ds1*ab+.25*ds2*ab*ab
        else:
            p0+=tl.dot(w1.to(tl.bfloat16),f0)
            p1+=tl.dot(w2.to(tl.bfloat16),f1)
            p2+=tl.dot(w2.to(tl.bfloat16),f2)
            p3+=tl.dot((w2*logits).to(tl.bfloat16),f0)
            s1+=tl.sum(w1*logits,1);s2+=tl.sum(w2*logits*logits,1)
            if QUERY:
                z0+=tl.dot(prob.to(tl.bfloat16),f0)
                z1+=tl.dot(prob.to(tl.bfloat16),f1)
                z2+=tl.dot(prob.to(tl.bfloat16),f2)
                z3+=tl.dot((prob*logits).to(tl.bfloat16),f0)
                t1+=tl.sum(prob*logits,1);t2+=tl.sum(prob*logits*logits,1)
    for side in tl.static_range(2 if PAIRED else 1):
        cp=head*3*T*D+pos[:,None]*D+(direction[None,:]+side*(D//2))
        a=tl.load(COEF+cp)
        u=tl.load(COEF+cp+T*D);w=tl.load(COEF+cp+2*T*D)
        p=tl.where(feat[None,:]<J,u,w);r=tl.where(feat[None,:]<J,w,u)
        if QUERY:
            numerator=p0*p+.5*p1*p*p+.5*p2*p*r+p3*a*p+.5*s1[:,None]*a+.25*s2[:,None]*a*a
            denominator=z0*p+.5*z1*p*p+.5*z2*p*r+z3*a*p+.5*t1[:,None]*a+.25*t2[:,None]*a*a
            if EARLY:
                numerator=en0 if side==0 else en1
                denominator=ed0 if side==0 else ed1
            tl.store(OUT+((head*2)*T+pos[:,None])*D*2+(chunk+side*(D//2//J))*(2*J)+feat[None,:],numerator)
            tl.store(OUT+((head*2+1)*T+pos[:,None])*D*2+(chunk+side*(D//2//J))*(2*J)+feat[None,:],denominator)
        else:
            numerator=p0*p+p1*p*p+p2*p*r+2*p3*a*p+.5*s1[:,None]*a+.5*s2[:,None]*a*a
            if EARLY:numerator=en0 if side==0 else en1
            tl.store(OUT+(head*T+pos[:,None])*D*2+(chunk+side*(D//2//J))*(2*J)+feat[None,:],numerator)



@triton.jit
def _value(Q,K,H,LSE,OUT,T:tl.constexpr,D:tl.constexpr,WINDOW:tl.constexpr,B:tl.constexpr):
    head,tile=tl.program_id(0),tl.program_id(1)
    ks=tile*B+tl.arange(0,B);ds=tl.arange(0,D)
    k=tl.load(K+head*T*D+ks[:,None]*D+ds[None,:]).to(tl.bfloat16)
    total=tl.full((B,D),0,tl.float32)
    for n in range(tile,tl.minimum(T//B,tl.cdiv((tile+1)*B+WINDOW,B))):
        qs=n*B+tl.arange(0,B)
        q=tl.load(Q+head*T*D+qs[:,None]*D+ds[None,:]).to(tl.bfloat16)
        h=tl.load(H+head*T*D+qs[:,None]*D+ds[None,:]).to(tl.bfloat16)
        lse=tl.load(LSE+head*T+qs)
        lag=qs[None,:]-ks[:,None]
        a=tl.exp(tl.dot(k,tl.trans(q))*(D**-.5)-lse[None,:])
        a=tl.where((lag>=0)&(lag<=WINDOW),a,0.).to(tl.bfloat16)
        total+=tl.dot(a,h)
    tl.store(OUT+head*T*D+ks[:,None]*D+ds[None,:],total)


def score(q,k,v,hub,coef,lse,mean,window,query,chunk=16,block=32,paired=False,early=False):
    bh,t,d=q.shape
    if early and not paired:raise ValueError('Early coefficient application requires paired reuse')
    if d % chunk or t % block:raise ValueError('Complete tiles required')
    out=torch.empty((bh,2,t,2*d) if query else (bh,t,2*d),device=q.device)
    _moments[(bh,t//block,d//chunk//(2 if paired else 1))](q,k,v,hub,coef,lse,mean,out,t,d,window,query,chunk,paired,early,block,block,num_warps=4)
    z=out.view(*out.shape[:-1],d//chunk,2,chunk).sum(-2).flatten(-2)
    return z[:,0]/(1+z[:,1]) if query else z


def local_credit(qraw,kraw,v,hub,cos,sin,window,chunk=16,block=32,lse=None,out=None,paired=False,early=False):
    q,k=rope32(qraw,cos,sin).contiguous(),rope32(kraw,cos,sin).contiguous()
    v,hub=v.bfloat16().contiguous(),hub.bfloat16().contiguous()
    if lse is None:
        # Reference path until the forward cache supplies these. Production must
        # use cached statistics or a tiled stats kernel, never this dense path.
        t,d=q.shape[-2:]
        lag=torch.arange(t,device=q.device)[:,None]-torch.arange(t,device=q.device)[None,:]
        logits=q.bfloat16().float()@k.bfloat16().float().transpose(-1,-2)/d**.5
        logits=logits.masked_fill((lag<0)|(lag>window),-torch.inf)
        lse=logits.logsumexp(-1)
        out=logits.softmax(-1)@v.float()
    mean=(hub.float()*out.float()).sum(-1)
    cq,sq=closed_coefficients(qraw,cos,sin);ck,sk=closed_coefficients(kraw,cos,sin)
    gq=score(q,k,v,hub,cq,lse,mean,window,True,chunk,block,paired,early)/sq
    gk=score(q,k,v,hub,ck,lse,mean,window,False,chunk,block,paired,early)/sk
    gv=torch.empty_like(q)
    bh,t,d=q.shape
    _value[(bh,t//block)](q,k,hub,lse,gv,t,d,window,block,num_warps=4)
    return gq,gk,gv
