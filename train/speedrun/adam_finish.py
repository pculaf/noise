"""Graph-safe addcdiv with a device step-size scalar and fp32 accumulation.

Keep the baseline Adam denominator rounding, instead of capturable Adam's extra
division by step_size in the bf16 denominator. Moment updates remain native foreach.
"""
import triton
import triton.language as tl


@triton.jit
def _finish(P,M,Den,Control,Offset:tl.constexpr,N:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    p=tl.load(P+i,i<N,0).to(tl.float32)
    m=tl.load(M+i,i<N,0).to(tl.float32)
    den=tl.load(Den+i,i<N,1).to(tl.float32)
    step=tl.load(Control+Offset)
    tl.store(P+i,p+(step*m)/den,i<N)


def addcdiv(p,m,den,control,offset):
    _finish[(triton.cdiv(p.numel(),512),)](p,m,den,control,offset,p.numel(),512,enable_fp_fusion=False)
