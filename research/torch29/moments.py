"""PyTorch port of Robert T. Lange's Noise PR #29 moment formulas (MIT).

Reference: RobertTLange/noise@866615d3a0332219b3d62ef8eaadc9c9d905d2fe,
train/speedrun/jax/train.py and kernels/exact_moments.py. This is attributed
prior work, not a new estimator. Plain tensor references are diagnostic only.
"""
import torch
from precise_mm import mm_transposed

EPS = torch.finfo(torch.float32).eps


def bfmm(a, b):
    return torch.mm(a.bfloat16(), b.bfloat16(), out_dtype=torch.float32)


def cycle_derivatives(base, c, y, steps, sq, dots, dir_sq, width, stable=False):
    inv = torch.rsqrt((sq + 2 * steps * dots + steps.square() * dir_sq) / width + EPS)
    inv0 = torch.rsqrt(sq / width + EPS)
    delta, sigma = inv / inv0 - 1, steps * inv
    z = base * inv0
    t = torch.tanh(z / 15)
    f = torch.exp(15 * t)
    g1 = 1 - t.square()
    d1 = -2 * t * g1 / 15
    d2 = -2 * g1 * (1 - 3 * t.square()) / 225
    g2 = g1.square() + d1
    g3 = g1.pow(3) + 3 * g1 * d1 + d2
    fz, fzz = f * z, f * z.square()
    total = f.sum(-1, keepdim=True)
    # Preserve the precise first product even when the surrounding trainer
    # enables TF32 for its other FP32 GEMMs.
    ratio = (sigma * mm_transposed(f * g1, c)
             + delta * sigma * bfmm(fz * g2, c.T)
             + .5 * delta.square() * sigma * bfmm(fzz * g3, c.T)
             + .5 * sigma.square() * bfmm(f * g2, c.square().T)
             + .5 * delta * sigma.square() * bfmm(fz * g3, c.square().T)
             + sigma.pow(3) / 6 * bfmm(f * g3, c.pow(3).T)
             + delta * (fz * g1).sum(-1, keepdim=True)
             + .5 * delta.square() * (fzz * g2).sum(-1, keepdim=True)
             + delta.pow(3) / 6 * (fzz * z * g3).sum(-1, keepdim=True))
    base_y = base.gather(1, y[:, None])
    if stable:
        # Algebraic difference of the two tanh values, avoiding subtraction of
        # complete CE losses. This is a separate candidate, not part of PR29.
        clean_t = torch.tanh(base_y * inv0 / 15)
        move = (delta * base_y * inv0 + sigma * c.T[y]) / 15
        tmove = torch.tanh(move)
        target_change = 15 * tmove * (1 - clean_t.square()) / (1 + clean_t * tmove)
        return (torch.log1p(ratio / total) - target_change) / steps
    clean = total.log() - 15 * torch.tanh(base_y * inv0 / 15)
    moved = total.log() + torch.log1p(ratio / total) - 15 * torch.tanh((base_y + steps * c.T[y]) * inv / 15)
    return (moved - clean) / steps


def rope32(raw, cos, sin):
    a, b = raw.chunk(2, -1)
    rotated = torch.cat((a * cos + b * sin, -a * sin + b * cos), -1)
    return rotated * torch.rsqrt(rotated.square().mean(-1, keepdim=True) + 1e-6)


def closed_coefficients(raw, cos, sin, sigma=.003):
    d = raw.shape[-1]
    sq = raw.square().sum(-1, keepdim=True)
    steps = sigma * (sq / d).sqrt().clamp_min(1e-6)
    change = (2 * steps * raw + steps.square()) / d / (sq / d + 1e-6)
    a = torch.expm1(-.5 * torch.log1p(change))
    b = steps * torch.rsqrt(sq / d + 1e-6) * (1 + a)
    u, w = torch.cat((cos, sin), -1), torch.cat((-sin, cos), -1)
    return torch.stack((a, b * u, b * w), 1), steps


def attention_reference(qraw, kraw, v, hub, cos, sin, window, degree=None):
    """Independent explicit coordinate moves; small diagnostic shapes only.

    degree=None evaluates the finite change exactly. degree=2 explicitly
    expands each logit change before summing (no moment factorization).
    """
    q, k = rope32(qraw, cos, sin), rope32(kraw, cos, sin)
    t, d = q.shape[-2:]
    lag = torch.arange(t, device=q.device)[:, None] - torch.arange(t, device=q.device)[None, :]
    mask = (lag >= 0) & (lag <= window)
    logits = q @ k.transpose(-1, -2) / d**.5
    a = logits.masked_fill(~mask, -torch.inf).softmax(-1)
    c = hub @ v.transpose(-1, -2) - (hub * (a @ v)).sum(-1, keepdim=True)
    outs = []
    for query, raw, other in ((True, qraw, k), (False, kraw, q)):
        _, steps = closed_coefficients(raw, cos, sin)
        result = []
        for j in range(d):
            moved = raw.clone()
            moved[..., j] += steps[..., 0]
            delta = rope32(moved, cos, sin) - (q if query else k)
            dl = (delta @ other.transpose(-1, -2) if query else other @ delta.transpose(-1, -2)) / d**.5
            if degree is None:
                ch = a * torch.expm1(dl)
                score = (ch * c).sum(-1) / (1 + ch.sum(-1)) if query else (ch / (1 + ch) * c).sum(-2)
            elif degree == 2:
                ch = a * (dl + .5 * dl.square())
                score = (ch * c).sum(-1) / (1 + ch.sum(-1)) if query else ((a * dl + (.5 * a - a.square()) * dl.square()) * c).sum(-2)
            else:
                raise ValueError(degree)
            result.append(score)
        outs.append(torch.stack(result, -1) / steps)
    return *outs, a.transpose(-1, -2) @ hub
