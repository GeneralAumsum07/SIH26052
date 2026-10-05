"""Fused FP32 GRU for training (low-delay plan Task 4b, `perf.numerics.gru_kernel: fused`).

The Mini's GRUs (512 sequences x hidden 24 at B32) are latency-bound on the GPU: cuDNN launches small kernels for
every time step of the backward pass. This module splits the GRU into
  - the input projections of all time steps as ONE GEMM (x @ W_ih^T + b_ih), strict FP32;
  - the recurrence as a persistent Triton kernel: each program keeps its sequences' hidden state on-chip across all
    time steps, computing W_hh h + b_hh and the gates in strict IEEE FP32 (tl.dot input_precision "ieee", no TF32);
    forward stores the gates it needs, backward runs the reverse recursion in the same kernel style;
  - the weight gradients as GEMMs over all time steps after the backward recursion.
Its parameters remain the nn.GRU tensors (or their per-step products under over-parameterization), so checkpoints,
fe_load, the step graph and the export are unchanged: use_fused_gru(model) only reroutes each Block's GRU forward.

`gru_reference` is the same decomposition in plain torch (explicit backward): it is what the Triton kernels compute
and runs on any device; tests hold it and the kernels to nn.GRU in FP32 with TF32 off (outputs 1e-6, gradients 1e-5
relative). Adopt the fused kernel only after its parity test passes on the training GPU (Gate A); otherwise every r8
run uses cuDNN.
"""
from __future__ import annotations

import contextlib

import torch
import torch.nn as nn
from types import MethodType

try:
    import triton
    import triton.language as tl
except ImportError:   # CPU boxes and torch builds without triton: the kernels are unavailable, the reference is not
    triton = None
    tl = None


def _weights(rnn: nn.GRU):
    return rnn.weight_ih_l0, rnn.weight_hh_l0, rnn.bias_ih_l0, rnn.bias_hh_l0


# ---- reference decomposition (plain torch, explicit backward) --------------------------------------------------
class _GRURef(torch.autograd.Function):
    """GRU (gate order r, z, n; b_hn inside r * (...), as PyTorch) with the fused kernels' decomposition."""

    @staticmethod
    def forward(ctx, x, w_ih, w_hh, b_ih, b_hh):
        b, t, _ = x.shape
        hdim = w_hh.shape[1]
        gi = torch.addmm(b_ih, x.reshape(b * t, -1), w_ih.t()).reshape(b, t, 3 * hdim)   # one GEMM
        h = x.new_zeros(b, hdim)
        out = x.new_empty(b, t, hdim)
        gates = x.new_empty(b, t, 4 * hdim)          # r, z, n, (W_hn h + b_hn) per step, for backward
        for s in range(t):
            gh = torch.addmm(b_hh, h, w_hh.t())
            r = torch.sigmoid(gi[:, s, :hdim] + gh[:, :hdim])
            z = torch.sigmoid(gi[:, s, hdim:2 * hdim] + gh[:, hdim:2 * hdim])
            ghn = gh[:, 2 * hdim:]
            n = torch.tanh(gi[:, s, 2 * hdim:] + r * ghn)
            h = n + z * (h - n)
            out[:, s] = h
            gates[:, s] = torch.cat([r, z, n, ghn], 1)
        ctx.save_for_backward(x, w_ih, w_hh, out, gates)
        return out

    @staticmethod
    def backward(ctx, dout):
        x, w_ih, w_hh, out, gates = ctx.saved_tensors
        b, t, hdim = out.shape
        dgi = torch.empty(b, t, 3 * hdim, dtype=out.dtype, device=out.device)
        dgh = torch.empty_like(dgi)
        dh = out.new_zeros(b, hdim)
        for s in range(t - 1, -1, -1):
            r, z, n, ghn = gates[:, s].split(hdim, 1)
            h_prev = out[:, s - 1] if s > 0 else out.new_zeros(b, hdim)
            dh = dh + dout[:, s]
            dn = dh * (1 - z)
            dz = dh * (h_prev - n)
            dan = dn * (1 - n * n)               # d(pre-tanh)
            dr = dan * ghn
            dar = dr * r * (1 - r)
            daz = dz * z * (1 - z)
            dgi[:, s] = torch.cat([dar, daz, dan], 1)
            dghs = torch.cat([dar, daz, dan * r], 1)
            dgh[:, s] = dghs
            dh = dh * z + dghs @ w_hh
        h_prev_all = torch.cat([out.new_zeros(b, 1, hdim), out[:, :-1]], 1)
        dw_hh = dgh.reshape(b * t, -1).t() @ h_prev_all.reshape(b * t, -1)
        db_hh = dgh.sum((0, 1))
        dw_ih = dgi.reshape(b * t, -1).t() @ x.reshape(b * t, -1)
        db_ih = dgi.sum((0, 1))
        dx = (dgi.reshape(b * t, -1) @ w_ih).reshape(x.shape)
        return dx, dw_ih, dw_hh, db_ih, db_hh


def gru_reference(rnn: nn.GRU, x: torch.Tensor) -> torch.Tensor:
    """(B, T, C) batch-first, zero initial state -> (B, T, H): the fused decomposition in plain torch."""
    return _GRURef.apply(x, *_weights(rnn))


# ---- Triton kernels ----------------------------------------------------------------------------------------
if triton is not None:
    @triton.jit
    def _gru_fwd_kernel(GI, WHH, BHH, OUT, GATES, B, T, H: tl.constexpr, HP: tl.constexpr, BLOCK_B: tl.constexpr):
        pid = tl.program_id(0)
        rows = pid * BLOCK_B + tl.arange(0, BLOCK_B)
        rmask = rows < B
        cols = tl.arange(0, HP)
        cmask = cols < H
        m2 = rmask[:, None] & cmask[None, :]
        wmask = cmask[:, None] & cmask[None, :]
        # W_hh^T blocks (HP x HP each), resident for the whole sequence
        wr = tl.load(WHH + (cols[None, :] + 0 * H) * H + cols[:, None], mask=wmask, other=0.0)
        wz = tl.load(WHH + (cols[None, :] + 1 * H) * H + cols[:, None], mask=wmask, other=0.0)
        wn = tl.load(WHH + (cols[None, :] + 2 * H) * H + cols[:, None], mask=wmask, other=0.0)
        br = tl.load(BHH + 0 * H + cols, mask=cmask, other=0.0)
        bz = tl.load(BHH + 1 * H + cols, mask=cmask, other=0.0)
        bn = tl.load(BHH + 2 * H + cols, mask=cmask, other=0.0)
        h = tl.zeros((BLOCK_B, HP), dtype=tl.float32)
        for s in range(0, T):
            base = (rows[:, None] * T + s) * (3 * H) + cols[None, :]
            gir = tl.load(GI + base, mask=m2, other=0.0)
            giz = tl.load(GI + base + H, mask=m2, other=0.0)
            gin = tl.load(GI + base + 2 * H, mask=m2, other=0.0)
            ghr = tl.dot(h, wr, input_precision="ieee") + br[None, :]
            ghz = tl.dot(h, wz, input_precision="ieee") + bz[None, :]
            ghn = tl.dot(h, wn, input_precision="ieee") + bn[None, :]
            r = tl.sigmoid(gir + ghr)
            z = tl.sigmoid(giz + ghz)
            a = gin + r * ghn
            n = 2.0 * tl.sigmoid(2.0 * a) - 1.0          # tanh
            h = n + z * (h - n)
            h = tl.where(m2, h, 0.0)
            tl.store(OUT + (rows[:, None] * T + s) * H + cols[None, :], h, mask=m2)
            gb = (rows[:, None] * T + s) * (4 * H) + cols[None, :]
            tl.store(GATES + gb, r, mask=m2)
            tl.store(GATES + gb + H, z, mask=m2)
            tl.store(GATES + gb + 2 * H, n, mask=m2)
            tl.store(GATES + gb + 3 * H, ghn, mask=m2)

    @triton.jit
    def _gru_bwd_kernel(DOUT, OUT, GATES, WHH, DGI, DGH, B, T, H: tl.constexpr, HP: tl.constexpr,
                        BLOCK_B: tl.constexpr):
        pid = tl.program_id(0)
        rows = pid * BLOCK_B + tl.arange(0, BLOCK_B)
        rmask = rows < B
        cols = tl.arange(0, HP)
        cmask = cols < H
        m2 = rmask[:, None] & cmask[None, :]
        wmask = cmask[:, None] & cmask[None, :]
        # W_hh blocks as (out -> in) so d h_prev = dgh_blk @ W_blk
        wr = tl.load(WHH + (cols[:, None] + 0 * H) * H + cols[None, :], mask=wmask, other=0.0)
        wz = tl.load(WHH + (cols[:, None] + 1 * H) * H + cols[None, :], mask=wmask, other=0.0)
        wn = tl.load(WHH + (cols[:, None] + 2 * H) * H + cols[None, :], mask=wmask, other=0.0)
        dh = tl.zeros((BLOCK_B, HP), dtype=tl.float32)
        for k in range(0, T):
            s = T - 1 - k
            gb = (rows[:, None] * T + s) * (4 * H) + cols[None, :]
            r = tl.load(GATES + gb, mask=m2, other=0.0)
            z = tl.load(GATES + gb + H, mask=m2, other=0.0)
            n = tl.load(GATES + gb + 2 * H, mask=m2, other=0.0)
            ghn = tl.load(GATES + gb + 3 * H, mask=m2, other=0.0)
            hp = tl.load(OUT + (rows[:, None] * T + s - 1) * H + cols[None, :], mask=m2 & (s > 0), other=0.0)
            dh = dh + tl.load(DOUT + (rows[:, None] * T + s) * H + cols[None, :], mask=m2, other=0.0)
            dn = dh * (1.0 - z)
            dz = dh * (hp - n)
            dan = dn * (1.0 - n * n)
            dar = dan * ghn * r * (1.0 - r)
            daz = dz * z * (1.0 - z)
            db = (rows[:, None] * T + s) * (3 * H) + cols[None, :]
            tl.store(DGI + db, dar, mask=m2)
            tl.store(DGI + db + H, daz, mask=m2)
            tl.store(DGI + db + 2 * H, dan, mask=m2)
            dhn = dan * r
            tl.store(DGH + db, dar, mask=m2)
            tl.store(DGH + db + H, daz, mask=m2)
            tl.store(DGH + db + 2 * H, dhn, mask=m2)
            dh = dh * z + tl.dot(dar, wr, input_precision="ieee") + tl.dot(daz, wz, input_precision="ieee") \
                + tl.dot(dhn, wn, input_precision="ieee")
            dh = tl.where(m2, dh, 0.0)


def _launch_cfg(b, hdim):
    hp = max(16, triton.next_power_of_2(hdim))
    # bwd at HP>=64 with Triton's default 3 load stages needs 110,592 B smem > sm_120's 101,376; 2 stages time the same
    bwd_stages = 3 if hp <= 32 else 2
    return hp, 16, (triton.cdiv(b, 16),), bwd_stages


@contextlib.contextmanager
def _no_tf32():
    # Under torch.compile TF32 is already off globally (runtime.tune_backends tf32=False); Dynamo cannot read the flag
    if torch.compiler.is_compiling():
        yield
        return
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev


class _GRUFused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_ih, w_hh, b_ih, b_hh):
        b, t, _ = x.shape
        hdim = w_hh.shape[1]
        x, w_ih, w_hh, b_ih, b_hh = (v.contiguous().float() for v in (x, w_ih, w_hh, b_ih, b_hh))
        with _no_tf32():
            gi = torch.addmm(b_ih, x.reshape(b * t, -1), w_ih.t()).reshape(b, t, 3 * hdim).contiguous()
        out = torch.empty(b, t, hdim, device=x.device, dtype=torch.float32)
        gates = torch.empty(b, t, 4 * hdim, device=x.device, dtype=torch.float32)
        hp, bb, grid, _ = _launch_cfg(b, hdim)
        _gru_fwd_kernel[grid](gi, w_hh, b_hh, out, gates, b, t, H=hdim, HP=hp, BLOCK_B=bb)
        ctx.save_for_backward(x, w_ih, w_hh, out, gates)
        return out

    @staticmethod
    def backward(ctx, dout):
        x, w_ih, w_hh, out, gates = ctx.saved_tensors
        b, t, hdim = out.shape
        dgi = torch.empty(b, t, 3 * hdim, device=out.device, dtype=torch.float32)
        dgh = torch.empty_like(dgi)
        hp, bb, grid, stages = _launch_cfg(b, hdim)
        _gru_bwd_kernel[grid](dout.contiguous().float(), out, gates, w_hh, dgi, dgh, b, t, H=hdim, HP=hp, BLOCK_B=bb,
                              num_stages=stages)
        with _no_tf32():
            h_prev = torch.cat([out.new_zeros(b, 1, hdim), out[:, :-1]], 1).reshape(b * t, hdim)
            dw_hh = dgh.reshape(b * t, -1).t() @ h_prev
            dw_ih = dgi.reshape(b * t, -1).t() @ x.reshape(b * t, -1)
            dx = (dgi.reshape(b * t, -1) @ w_ih).reshape(x.shape)
        return dx, dw_ih, dw_hh, dgi.sum((0, 1)), dgh.sum((0, 1))


def fused_available(device) -> bool:
    return triton is not None and torch.device(device).type == "cuda"


# Opaque to torch.compile: traced, the Function's matmuls drifted ~7e-4 from eager; eager-inside is bit-identical
@torch.compiler.disable
def gru_fused(rnn: nn.GRU, x: torch.Tensor) -> torch.Tensor:
    if not fused_available(x.device):
        raise RuntimeError("the fused GRU kernel needs CUDA and triton; set perf.numerics.gru_kernel: cudnn")
    return _GRUFused.apply(x, *_weights(rnn))


def _routed_forward(rnn, x, hx=None):
    if hx is not None:
        raise ValueError("the fused GRU starts from a zero state")
    fn = {"fused": gru_fused, "reference": gru_reference}[rnn._vaani_gru_impl]
    return fn(rnn, x), None


def use_fused_gru(model, impl: str = "fused"):
    """Route every VaaniFE Block's training GRU through the fused kernel (impl "fused") or its plain-torch reference
    ("reference"). Parameters, state dicts and the step graph are untouched."""
    if impl not in ("fused", "reference"):
        raise ValueError("GRU implementation must be fused or reference")
    for blk in model.blocks:
        rnn = blk.rnn
        if rnn.num_layers != 1 or rnn.bidirectional or not rnn.batch_first:
            raise ValueError("the fused GRU supports one batch-first unidirectional layer")

        # Closures survive deepcopy and make EMA read live weights. Bound methods
        # rebind to the copied module, preserving the shadow's own parameters.
        rnn._vaani_gru_impl = impl
        rnn.forward = MethodType(_routed_forward, rnn)
    return model
