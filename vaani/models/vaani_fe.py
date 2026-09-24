"""VaaniFE: dual-mic RNNFormer family (time GRU + frequency self-attention), plan 11.3.

Derived from FastEnhancer (arXiv 2509.21867): per-frame frequency convolutions (time kernel 1),
K blocks of [one-step time GRU over F tokens -> multi-head self-attention across the F tokens of
the same frame], and a complex mask on the power-law-compressed primary spectrum. The only
recurrent state is each block's GRU hidden state (K, F, C2), plus a Slice+Concat frame cache
when the low-band deep filter ablation is on.

Tiers (C1/C2/F/K/L): mini 32/24/16/2/1 is the only one trained (r8); mid, large and large_plus are
untrained projections. C1 = encoder/decoder conv width, C2 = block width, F = frequency tokens,
K = blocks, L = extra encoder convs (and matching decoder stages).

Tensor layouts (B batch, T frames, 257 = rfft bins of the 512/256 STFT in vaani/dsp/stft.py):
  forward(spec6, feats=None, ref_avail=None) -> (B,257,T,2)       offline, training
    spec6      (B,257,T,n) raw STFT RI, channels [p_re,p_im,r_re,r_im,nhat_re,nhat_im]; n >= n_raw,
               extra trailing channels are ignored, so train.prepare_batch's spec6 plugs in as-is
    feats      (B,T,18) DSP features: accepted for signature compatibility, unused
    ref_avail  (B,T) reference validity in {0,1} from the capture path; None = all valid
  step(spec, valid, state) -> (spec_out, state_out)                streaming, export
    spec       (B,257,1,n) one raw frame, same channel order
    valid      (B,1) this frame's validity (ignored when inputs == "p")
    state      (B,S) flat float32: K blocks of F*C2 hidden (block-major, token-major), then
               (df_taps-1) cached compressed primary low-band frames of 2*DF_BINS (oldest first)
    spec_out   (B,257,1,2) enhanced raw STFT frame (decompressed)

Inputs option (ablation 2); every plane is 256 bins, compressed RI = X*|X|^(0.3-1):
  p        P RI                                  n_in 2 (mono; no validity plane)
  pr       P RI, R RI*v, v plane                 n_in 5 (default)
  pr_nhat  + NLMS output RI*v                    n_in 7
  pr_pld   + PLD*v, PLD = (|Pc|^2-|Rc|^2)/(|Pc|^2+|Rc|^2+eps)  n_in 6
PLD is a bounded, scale-free primary/reference power-level difference on the compressed
spectrum, after PLDNet's PLD guidance (our bounded variant, not PLDNet's exact formula).
Reference-derived planes are multiplied by validity inside the model, so validity 0 means the
reference is ignored even if the backend passes stale samples (the trained mono fallback).

Bin 256 (Nyquist): the net sees bins 0..255; bin 256 is masked with bin 255's mask (edge copy).
Mask (ablation 4): unbounded complex mask (default) or bounded, |M| = tanh(|m|) with m's phase.
df_taps D > 0 (ablation 4): bins 0..DF_BINS-1 are replaced by a D-tap complex filter over the
current and D-1 past compressed primary frames. The refiner is dropped (ablation 5 not built).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

HOPS_PER_S = 62.5
N_BINS = 257
NET_BINS = 256        # bins 0..255 enter the network; bin 256 reuses bin 255's mask
COARSE = 64           # pre-conv stride 4: 256 -> 64 coarse bins
DF_BINS = 64          # low band for the deep-filter ablation (0-2 kHz)
ALPHA = 0.3           # power-law compression exponent
EPS = 1e-12           # keeps pow finite at |X| = 0; FP32 only (underflows in FP16)
INPUTS = {"p": 2, "pr": 5, "pr_nhat": 7, "pr_pld": 6}   # -> n_in planes
RAW = {"p": 2, "pr": 4, "pr_nhat": 6, "pr_pld": 4}      # -> raw STFT channels read from spec6

TIERS = {
    "mini": dict(c1=32, c2=24, f=16, k=2, l=1),
    "mid": dict(c1=48, c2=40, f=32, k=3, l=2),
    "large": dict(c1=80, c2=64, f=48, k=4, l=2),
    "large_plus": dict(c1=96, c2=72, f=48, k=4, l=3),
}
TIER_STATUS = {"mini": "to be trained in r8", "mid": "projection, untrained",
               "large": "projection, untrained", "large_plus": "projection, untrained"}


def gru_cell(x, h, rnn: nn.GRU):
    """One nn.GRU step as Gemm ops: gate order r,z,n; b_hn inside r*(...) as PyTorch does."""
    gi = F.linear(x, rnn.weight_ih_l0, rnn.bias_ih_l0)
    gh = F.linear(h, rnn.weight_hh_l0, rnn.bias_hh_l0)
    ir, iz, i_n = gi.chunk(3, -1)
    hr, hz, h_n = gh.chunk(3, -1)
    r = torch.sigmoid(ir + hr)
    z = torch.sigmoid(iz + hz)
    n = torch.tanh(i_n + r * h_n)
    return n + z * (h - n)  # == (1-z)*n + z*h with one fewer op


class FreqAttn(nn.Module):
    """MHSA across the F tokens of ONE frame: no time context, so no KV cache."""

    def __init__(self, c, heads):
        super().__init__()
        if c % heads:
            raise ValueError(f"c2={c} not divisible by heads={heads}")
        self.h = heads
        self.qkv = nn.Linear(c, 3 * c, bias=False)
        self.last_macs = 0

    def forward(self, x):  # (N, F, C)
        n, f, c = x.shape
        d = c // self.h
        q, k, v = self.qkv(x).reshape(n, f, 3, self.h, d).permute(2, 0, 3, 1, 4)
        # explicit softmax attention: exports as MatMul/Softmax, no fused-attention op
        a = torch.softmax(q @ k.transpose(-1, -2) * d ** -0.5, dim=-1)
        self.last_macs = 2 * n * f * f * c  # QK^T + AV, invisible to Linear hooks
        return (a @ v).transpose(1, 2).reshape(n, f, c)


class Block(nn.Module):
    """Time GRU (per token, one step per frame) then frequency attention, both residual."""

    def __init__(self, c, heads):
        super().__init__()
        self.rnn = nn.GRU(c, c, batch_first=True)  # offline path; step() reuses its weights as Gemm cells
        self.rnn_fc = nn.Linear(c, c)
        self.mix = FreqAttn(c, heads)
        self.mix_fc = nn.Linear(c, c)

    def freq(self, x):  # (N, F, C) per-frame part
        return x + self.mix_fc(self.mix(x))


def _cbr(cin, cout, k, norm, **kw):
    """Conv (+BN, folded at export) + ReLU."""
    layers = [nn.Conv1d(cin, cout, k, **kw)]
    if norm == "bn":
        layers.append(nn.BatchNorm1d(cout))
    return nn.Sequential(*layers, nn.ReLU())


class VaaniFE(nn.Module):
    def __init__(self, c1=32, c2=24, f=16, k=2, l=1, heads=4, inputs="pr", mask="unbounded",
                 df_taps=0, norm="bn"):
        super().__init__()
        if inputs not in INPUTS:
            raise ValueError(f"inputs must be one of {sorted(INPUTS)}, got {inputs!r}")
        if mask not in ("unbounded", "bounded"):
            raise ValueError(f"mask must be 'unbounded' or 'bounded', got {mask!r}")
        if df_taps not in (0, 2, 3):
            raise ValueError(f"df_taps must be 0, 2 or 3, got {df_taps!r}")
        if norm not in ("bn", "none"):
            raise ValueError(f"norm must be 'bn' or 'none', got {norm!r}")
        self.cfg = dict(c1=c1, c2=c2, f=f, k=k, l=l, heads=heads, inputs=inputs, mask=mask,
                        df_taps=df_taps, norm=norm)
        self.c1, self.c2, self.f, self.k, self.l = c1, c2, f, k, l
        self.inputs, self.mask_kind, self.df_taps = inputs, mask, df_taps
        self.n_in, self.n_raw = INPUTS[inputs], RAW[inputs]
        self.uses_ref = inputs != "p"
        self.pre = _cbr(self.n_in, c1, 8, norm, stride=4, padding=2)            # 256 -> 64 bins
        self.enc = nn.ModuleList(_cbr(c1, c1, 3, norm, padding=1) for _ in range(l))
        self.rf_pre_lin, self.rf_pre_conv = nn.Linear(COARSE, f, bias=False), nn.Conv1d(c1, c2, 1)
        self.pe = nn.Parameter(torch.zeros(f, c2))
        self.blocks = nn.ModuleList(Block(c2, heads) for _ in range(k))
        self.rf_post_lin, self.rf_post_conv = nn.Linear(f, COARSE, bias=False), nn.Conv1d(c2, c1, 1)
        self.dec = nn.ModuleList(nn.Sequential(_cbr(2 * c1, c1, 1, norm), _cbr(c1, c1, 3, norm, padding=1))
                                 for _ in range(l))
        self.post = _cbr(2 * c1, c1, 1, norm)
        self.up = nn.ConvTranspose1d(c1, 2, 8, stride=4, padding=2)            # 64 -> 256 bins, complex mask
        if df_taps:  # DF taps from the lowest 16 coarse bins -> 64 fine bins
            self.df = nn.ConvTranspose1d(c1, 2 * df_taps, 8, stride=4, padding=2)

    # ---- sizes -------------------------------------------------------------------------------
    @property
    def hidden_size(self):
        return self.k * self.f * self.c2

    @property
    def df_cache_size(self):
        return max(self.df_taps - 1, 0) * 2 * DF_BINS

    @property
    def state_size(self):
        return self.hidden_size + self.df_cache_size

    def init_state(self, batch=1, device=None):
        return torch.zeros(batch, self.state_size, device=device)

    # ---- shared per-frame parts --------------------------------------------------------------
    def _planes(self, x, v):
        """x (N, n_raw, 257) raw RI, v (N,1) -> (N, n_in, 256) network planes, (N,2,257) compressed P."""
        n = x.shape[0]
        re, im = x[:, 0::2], x[:, 1::2]                                         # (N, n_raw/2, 257)
        s = (re * re + im * im + EPS) ** ((ALPHA - 1) / 2)
        xc = torch.stack([re * s, im * s], 2).reshape(n, self.n_raw, N_BINS)   # compressed RI, same order
        pc = xc[:, :2]
        if not self.uses_ref:
            return pc[..., :NET_BINS], pc
        vb = v.reshape(n, 1, 1)
        planes = [pc, xc[:, 2:] * vb]                                          # ref (and n_hat) gated by validity
        if self.inputs == "pr_pld":
            p2 = xc[:, 0:1] ** 2 + xc[:, 1:2] ** 2
            r2 = xc[:, 2:3] ** 2 + xc[:, 3:4] ** 2
            planes.append((p2 - r2) / (p2 + r2 + EPS) * vb)
        planes.append(torch.zeros(1, 1, N_BINS, dtype=x.dtype, device=x.device) + vb)
        return torch.cat(planes, 1)[..., :NET_BINS], pc

    def _encode(self, planes):
        x = self.pre(planes)
        skips = [x]
        for c in self.enc:
            x = c(x)
            skips.append(x)
        tok = self.rf_pre_conv(self.rf_pre_lin(x)).transpose(1, 2) + self.pe   # (N, F, C2)
        return tok, skips

    def _decode(self, tok, skips, pc, pc_hist=None):
        """tok (N,F,C2) -> compressed-domain output (N,2,257); pc_hist (N,D,2,DF_BINS) newest first."""
        x = self.rf_post_conv(self.rf_post_lin(tok.transpose(1, 2)))           # (N, C1, 64)
        for d in self.dec:
            x = d(torch.cat([x, skips.pop()], 1))
        x = self.post(torch.cat([x, skips.pop()], 1))
        m = self.up(x)                                                         # (N, 2, 256)
        m = torch.cat([m, m[..., -1:]], -1)                                    # bin 256 := bin 255's mask
        if self.mask_kind == "bounded":
            mag = torch.sqrt(m[:, :1] ** 2 + m[:, 1:] ** 2 + EPS)
            m = m * (torch.tanh(mag) / mag)
        mr, mi, pr_, pi_ = m[:, 0], m[:, 1], pc[:, 0], pc[:, 1]
        y = torch.stack([pr_ * mr - pi_ * mi, pr_ * mi + pi_ * mr], 1)
        if self.df_taps:
            w = self.df(x[..., :DF_BINS // 4]).reshape(-1, self.df_taps, 2, DF_BINS)
            wr, wi, hr, hi = w[:, :, 0], w[:, :, 1], pc_hist[:, :, 0], pc_hist[:, :, 1]
            low = torch.stack([(hr * wr - hi * wi).sum(1), (hr * wi + hi * wr).sum(1)], 1)
            y = torch.cat([low, y[..., DF_BINS:]], -1)
        return y

    @staticmethod
    def _decompress(y):
        """Compressed RI -> raw RI: Y * |Y|^(1/alpha - 1)."""
        g = (y[:, :1] ** 2 + y[:, 1:] ** 2 + EPS) ** ((1 / ALPHA - 1) / 2)
        return y * g

    # ---- offline -----------------------------------------------------------------------------
    def forward(self, spec6, feats=None, ref_avail=None):
        b, nb, t, _ = spec6.shape
        n = b * t
        x = spec6[..., :self.n_raw].permute(0, 2, 3, 1).reshape(n, self.n_raw, nb)
        v = spec6.new_ones(n, 1) if ref_avail is None else ref_avail.to(spec6.dtype).reshape(n, 1)
        planes, pc = self._planes(x, v)
        tok, skips = self._encode(planes)
        for blk in self.blocks:
            seq = tok.reshape(b, t, self.f, self.c2).transpose(1, 2).reshape(b * self.f, t, self.c2)
            y = blk.rnn(seq)[0].reshape(b, self.f, t, self.c2).transpose(1, 2).reshape(n, self.f, self.c2)
            tok = blk.freq(tok + blk.rnn_fc(y))
        hist = None
        if self.df_taps:
            low = pc[..., :DF_BINS].reshape(b, t, 2, DF_BINS)
            hist = torch.stack([F.pad(low, (0, 0, 0, 0, d, 0))[:, :t] for d in range(self.df_taps)], 2)
            hist = hist.reshape(n, self.df_taps, 2, DF_BINS)
        out = self._decompress(self._decode(tok, skips, pc, hist))
        return out.reshape(b, t, 2, nb).permute(0, 3, 1, 2)

    # ---- streaming ---------------------------------------------------------------------------
    def step(self, spec, valid, state):
        b, nb = spec.shape[0], spec.shape[1]
        x = spec[..., 0, :self.n_raw].transpose(1, 2)                           # (B, n_raw, 257)
        v = spec.new_ones(b, 1) if valid is None else valid.reshape(b, 1)
        planes, pc = self._planes(x, v)
        tok, skips = self._encode(planes)
        fc, new = self.f * self.c2, []
        for i, blk in enumerate(self.blocks):
            h = state[:, i * fc:(i + 1) * fc].reshape(b * self.f, self.c2)
            h = gru_cell(tok.reshape(b * self.f, self.c2), h, blk.rnn)
            new.append(h.reshape(b, fc))
            tok = blk.freq(tok + blk.rnn_fc(h.reshape(b, self.f, self.c2)))
        hist = None
        if self.df_taps:
            cur = pc[..., :DF_BINS].reshape(b, 2 * DF_BINS)
            cache = state[:, self.hidden_size:]                                 # oldest first
            frames = [cur] + [cache[:, j * 2 * DF_BINS:(j + 1) * 2 * DF_BINS] for j in range(self.df_taps - 2, -1, -1)]
            hist = torch.stack(frames, 1).reshape(b, self.df_taps, 2, DF_BINS)
            new.append(torch.cat([cache[:, 2 * DF_BINS:], cur], 1))             # Slice+Concat, no ScatterND
        out = self._decompress(self._decode(tok, skips, pc, hist))
        return out.reshape(b, 2, nb, 1).permute(0, 2, 3, 1), torch.cat(new, 1)


class StepGraph(nn.Module):
    """Export wrapper: step() with named positional I/O (valid dropped for inputs='p')."""

    def __init__(self, model: VaaniFE):
        super().__init__()
        self.m = model

    def forward(self, spec, *rest):
        valid, state = rest if self.m.uses_ref else (None, rest[0])
        return self.m.step(spec, valid, state)

    def io_names(self):
        ins = ["spec", "valid", "state"] if self.m.uses_ref else ["spec", "state"]
        return ins, ["spec_out", "state_out"]

    def example_inputs(self, seed=0):
        g = torch.Generator().manual_seed(seed)
        spec = torch.randn(1, N_BINS, 1, self.m.n_raw, generator=g) * 0.1
        state = self.m.init_state(1)
        return (spec, torch.ones(1, 1), state) if self.m.uses_ref else (spec, state)


# ---- helpers ---------------------------------------------------------------------------------
def build(tier="mini", **overrides):
    return VaaniFE(**{**TIERS[tier], **overrides})


def from_arch(cfg: dict) -> VaaniFE:
    """configs/arch/*.yaml 'model_cfg' (or a bare dict of VaaniFE kwargs, optionally with 'tier')."""
    cfg = dict(cfg.get("model_cfg", cfg))
    tier = cfg.pop("tier", None)
    return VaaniFE(**{**(TIERS[tier] if tier else {}), **cfg})


def load_arch(path) -> VaaniFE:
    import yaml
    with open(path, encoding="utf-8") as fh:
        return from_arch(yaml.safe_load(fh))


def param_count(model, inference=True):
    """Total entries. inference=True counts the BN-folded form (BN adds no entries once folded into
    a biased conv); training form includes BN affine weights and running stats."""
    n = sum(p.numel() for p in model.parameters())
    if inference:
        n -= sum(p.numel() for m in model.modules() if isinstance(m, nn.BatchNorm1d) for p in m.parameters())
    else:
        n += sum(bf.numel() for m in model.modules() if isinstance(m, nn.BatchNorm1d)
                 for bf in (m.running_mean, m.running_var))
    return n


def count_macs(model: VaaniFE):
    """Per-hop matrix MACs: vaani.export.layer_macs rule (dense Conv/ConvTranspose/Linear/GRU,
    padding included, no bias/norm/elementwise/DSP) plus attention QK^T and AV products."""
    total, handles = [0], []

    def hook(m, args, out):
        if isinstance(m, nn.Conv1d):
            n = out.numel() * (m.in_channels // m.groups) * math.prod(m.kernel_size)
        elif isinstance(m, nn.ConvTranspose1d):
            n = args[0].numel() * (m.out_channels // m.groups) * math.prod(m.kernel_size)
        elif isinstance(m, nn.Linear):
            n = out.numel() * m.in_features
        elif isinstance(m, nn.GRU):
            n = (args[0].numel() // m.input_size) * sum(p.numel() for k, p in m.named_parameters() if k.startswith("weight_"))
        elif isinstance(m, FreqAttn):
            n = m.last_macs
        else:
            return
        total[0] += int(n)

    for m in model.modules():
        handles.append(m.register_forward_hook(hook))
    was = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(torch.zeros(1, N_BINS, 1, model.n_raw))
    finally:
        for h in handles:
            h.remove()
        model.train(was)
    return total[0]


def summary(model: VaaniFE):
    macs = count_macs(model)
    return {"cfg": dict(model.cfg), "params": param_count(model), "params_training_form": param_count(model, False),
            "mac_per_hop": macs, "mmac_per_s": round(macs * HOPS_PER_S / 1e6, 3),
            "state_floats": model.state_size, "state_bytes": model.state_size * 4}
