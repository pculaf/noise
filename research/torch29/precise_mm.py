"""FP32-input tensor-core product for PR29's explicitly precise first moment."""
import torch
import triton
import triton.language as tl


@triton.jit
def _mm(A,B,C,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
        BM:tl.constexpr=32,BN:tl.constexpr=64,BK:tl.constexpr=32):
    m=tl.program_id(0)*BM+tl.arange(0,BM)
    n=tl.program_id(1)*BN+tl.arange(0,BN)
    kk=tl.arange(0,BK)
    acc=tl.full((BM,BN),0,tl.float32)
    for offset in range(tl.cdiv(K,BK)):
        k=offset*BK+kk
        a=tl.load(A+m[:,None]*K+k[None,:],(m[:,None]<M)&(k[None,:]<K),0.)
        b=tl.load(B+n[None,:]*K+k[:,None],(n[None,:]<N)&(k[:,None]<K),0.)
        acc+=tl.dot(a,b,input_precision='tf32x3')
    tl.store(C+m[:,None]*N+n[None,:],acc,(m[:,None]<M)&(n[None,:]<N))


def mm_transposed(a,b):
    a,b=a.contiguous(),b.contiguous()
    m,k=a.shape;n=b.shape[0]
    out=torch.empty((m,n),device=a.device)
    _mm[(triton.cdiv(m,32),triton.cdiv(n,64))](a,b,out,m,n,k,32,64,32)
    return out
