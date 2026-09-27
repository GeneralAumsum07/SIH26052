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
    spec       (B,n_raw,257) one raw frame, channels-first in the same channel order
    valid      (B,1) this frame's validity (ignored when inputs == "p")
    state      (B,S) flat float32: K blocks of F*C2 GRU hidden (block-major, token-major), then
               (df_taps-1) cached compressed primary low-band frames of 2*DF_BINS (oldest first)
    spec_out   (B,2,257) enhanced raw STFT frame [re, im] (decompressed)
  Channels-first step I/O keeps the exported graph free of input/output transposes.

Inputs option (ablation 2); every plane is 257 bins, compressed RI = X*|X|^(0.3-1):
  p        P RI                                  n_in 2 (mono; no validity plane)
  pr       P RI, R RI*v, v plane                 n_in 5 (default)
  pr_nhat  + NLMS output RI*v                    n_in 7
  pr_pld   + PLD*v, PLD = (|Pc|^2-|Rc|^2)/(|Pc|^2+|Rc|^2+eps)  n_in 6
PLD is a bounded, scale-free primary/reference power-level difference on the compressed
spectrum, after PLDNet's PLD guidance (our bounded variant, not PLDNet's exact formula).
Reference-derived planes are multiplied by validity inside the model, so validity 0 means the
reference is ignored even if the backend passes stale samples (the trained mono fallback).

Bin 256 (Nyquist) is part of the network, not dropped: the stride-4 pre-conv's last window
(bins 250..257, right-padded) covers it, and the transposed conv's output_padding=1 emits its
mask as a learned 257th output. Parameters and MACs equal the 256-bin prototype's.
Mask (ablation 4): unbounded complex mask (default) or bounded, |M| = tanh(|m|) with m's phase.
df_taps D > 0 (ablation 4): bins 0..DF_BINS-1 are replaced by a D-tap complex filter over the
current and D-1 past compressed primary frames. The refiner is dropped (ablation 5 not built).
Pair sums, channel swaps and mask broadcasts use fixed 0/+-1 1x1 convs (exact in FP32) instead
of Slice/Concat, to keep G2's layout-op share down; they carry no parameters and no counted MACs.

Low-delay options (plan "VaaniFE low-latency r8", Section 3.4; every default is the legacy Mini, so legacy configs and
checkpoints load unchanged):
  audio_contract  vaani.audio_contract ID the network is trained for (None = C0); summary() takes hops/s from it
  freq_windows    "p18" | "p32": the Mini-P tiling. Bins 0..255 are partitioned into windows of 4..64 bins; each
                  resolution has its own strided input conv (kernel = stride = width, no padding) and transposed
                  output conv, so the mask keeps 31.25 Hz resolution. Bin 256's mask is a learned complex constant.
                  The validity plane is replaced by v x a learned C1-vector (valid_bias), equal to convolving a
                  constant plane on unpadded windows at C1 MACs. Only inputs "p" and "pr".
  df_bins, df_lags deep-filter band (bins 0..df_bins-1) and tap lags in frames (default 64 and 0..df_taps-1).
                  Mini-P18: 96 bins, lags (0, 3, 5); Mini-P32: 144 bins, lags (0, 2, 4). The cache holds max(lags)
                  past compressed low-band frames, oldest first.
  gru_init        "tc_matched": shift each update-gate bias so the initial retention is z^(H/256) (Tallec & Ollivier);
                  deterministic, consumes no random numbers, skipped at H = 256. "default": PyTorch's.
  fp32_islands    run the compression/decompression power sums and the GRU recurrence in true FP32 (no autocast, no
                  TF32), restoring the global backend flags afterwards.
  overparam       training-time over-parameterization: RepVGG branches on the 3-tap convs, ExpandNets factors (inner
                  width 4x the output) on every other linear map. fold() returns the plain network, exactly.
"""
from __future__ import annotations

import contextlib
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrize

from vaani.audio_contract import get_audio_contract

HOPS_PER_S = 62.5
N_BINS = 257
COARSE = 64           # pre-conv stride 4: 257 -> 64 coarse bins
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
# Mini-P tilings (Section 3.4): (window width in bins, count), bins 0..255 in order at 31.25 Hz per bin
FREQ_WINDOWS = {
    "p18": ((4, 8), (8, 4), (16, 2), (32, 3), (64, 1)),
    "p32": ((4, 16), (8, 10), (16, 5), (32, 1)),
}
# the low-delay recipe per tiling (Section 3.4): deep-filter band and lag-matched taps
MINI_P = {
    "p18": dict(freq_windows="p18", valid_bias=True, df_bins=96, df_lags=(0, 3, 5)),
    "p32": dict(freq_windows="p32", valid_bias=True, df_bins=144, df_lags=(0, 2, 4)),
}
LEGACY_HOP = 256

TIER_STATUS = {"mini": "to be trained in r8", "mid": "projection, untrained",
               "large": "projection, untrained", "large_plus": "projection, untrained"}


def gru_cell(x, h, rnn: nn.GRU):
    """One nn.GRU step as per-gate Gemms: gate order r,z,n; b_hn inside r*(...) as PyTorch does.
    Weight slices are initializers, so they fold away and no Split node is exported."""
    c = rnn.hidden_size
    wi, wh, bi, bh = rnn.weight_ih_l0, rnn.weight_hh_l0, rnn.bias_ih_l0, rnn.bias_hh_l0
    g = [(F.linear(x, wi[j * c:(j + 1) * c], bi[j * c:(j + 1) * c]),
          F.linear(h, wh[j * c:(j + 1) * c], bh[j * c:(j + 1) * c])) for j in range(3)]
    r = torch.sigmoid(g[0][0] + g[0][1])
    z = torch.sigmoid(g[1][0] + g[1][1])
    n = torch.tanh(g[2][0] + r * g[2][1])
    return n + z * (h - n)  # == (1-z)*n + z*h with one fewer op


@contextlib.contextmanager
def fp32_island(enabled=True, device_type="cuda"):
    """True FP32 inside: autocast off, cuDNN and matmul TF32 off; the global flags are restored afterwards
    (runtime.tune_backends turns TF32 on globally, so the island switches it off itself)."""
    if not enabled:
        yield
        return
    prev = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        with torch.autocast(device_type, enabled=False):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = prev


def _at_least_fp32(x):
    return x.float() if x.dtype in (torch.float16, torch.bfloat16) else x


def tc_matched_bias(b_z: torch.Tensor, hop: int) -> torch.Tensor:
    """Combined update-gate bias b -> logit(sigmoid(b) ** (hop/256)): the initial retention z' = z^(H/256)
    keeps the initial GRU time constant in seconds when the hop shrinks from 256 to H."""
    z = torch.sigmoid(b_z.double()) ** (hop / LEGACY_HOP)
    return torch.logit(z).to(b_z.dtype)


def apply_gru_init(blocks, hop: int):
    """Shift b_iz so b_iz + b_hz = tc_matched_bias(b_iz + b_hz). No random numbers; no-op at H = 256."""
    if hop == LEGACY_HOP:
        return
    with torch.no_grad():
        for blk in blocks:
            c = blk.rnn.hidden_size
            bi, bh = blk.rnn.bias_ih_l0[c:2 * c], blk.rnn.bias_hh_l0[c:2 * c]
            b = bi + bh
            bi.add_(tc_matched_bias(b, hop) - b)


class FreqAttn(nn.Module):
    """MHSA across the F tokens of ONE frame: no time context, so no KV cache."""

    def __init__(self, c, heads):
        super().__init__()
        if c % heads:
            raise ValueError(f"c2={c} not divisible by heads={heads}")
        self.h = heads
        self.qkv = nn.Linear(c, 3 * c, bias=False)
        self.last_macs = 0

    def forward(self, x, n, f):  # x (n*f, C) or (n, f, C)
        c = x.shape[-1]
        d = c // self.h
        q, k, v = self.qkv(x).reshape(n, f, 3 * self.h, d).transpose(1, 2).split(self.h, 1)
        # explicit softmax attention: exports as MatMul/Softmax, no fused-attention op
        a = torch.softmax(q @ k.transpose(-1, -2) * d ** -0.5, dim=-1)
        self.last_macs = 2 * n * f * f * c  # QK^T + AV, invisible to Linear hooks
        return (a @ v).transpose(1, 2).reshape(x.shape)


class Block(nn.Module):
    """Time GRU (per token, one step per frame) then frequency attention, both residual."""

    def __init__(self, c, heads):
        super().__init__()
        self.rnn = nn.GRU(c, c, batch_first=True)  # offline path; step() reuses its weights as Gemm cells
        self.rnn_fc = nn.Linear(c, c)
        self.mix = FreqAttn(c, heads)
        self.mix_fc = nn.Linear(c, c)

    def freq(self, x, n, f):  # per-frame part
        return x + self.mix_fc(self.mix(x, n, f))


def _cbr(cin, cout, k, norm, **kw):
    """Conv (+BN, folded at export) + ReLU."""
    layers = [nn.Conv1d(cin, cout, k, **kw)]
    if norm == "bn":
        layers.append(nn.BatchNorm1d(cout))
    return nn.Sequential(*layers, nn.ReLU())


def _fixed(rows):
    return torch.tensor(rows, dtype=torch.float32)[..., None]  # (out, in, 1) conv weight


class VaaniFE(nn.Module):
    def __init__(self, c1=32, c2=24, f=16, k=2, l=1, heads=4, inputs="pr", mask="unbounded",
                 df_taps=0, norm="bn", audio_contract=None, freq_windows=None, valid_bias=None, df_bins=None,
                 df_lags=None, gru_init="default", fp32_islands=False, overparam=False):
        super().__init__()
        if inputs not in INPUTS:
            raise ValueError(f"inputs must be one of {sorted(INPUTS)}, got {inputs!r}")
        if mask not in ("unbounded", "bounded"):
            raise ValueError(f"mask must be 'unbounded' or 'bounded', got {mask!r}")
        if df_lags is not None:
            df_lags = tuple(int(x) for x in df_lags)
            if not df_lags or df_lags[0] != 0 or any(b <= a for a, b in zip(df_lags, df_lags[1:])):
                raise ValueError(f"df_lags must start at 0 and increase strictly, got {df_lags!r}")
            if df_taps not in (0, len(df_lags)):
                raise ValueError("df_taps must equal len(df_lags) when both are given")
            df_taps = len(df_lags)
        elif df_taps not in (0, 2, 3):
            raise ValueError(f"df_taps must be 0, 2 or 3, got {df_taps!r}")
        if norm not in ("bn", "none"):
            raise ValueError(f"norm must be 'bn' or 'none', got {norm!r}")
        if gru_init not in ("default", "tc_matched"):
            raise ValueError(f"gru_init must be 'default' or 'tc_matched', got {gru_init!r}")
        if freq_windows is not None and freq_windows not in FREQ_WINDOWS:
            raise ValueError(f"freq_windows must be one of {sorted(FREQ_WINDOWS)} or absent, got {freq_windows!r}")
        if freq_windows is not None and inputs not in ("p", "pr"):
            raise ValueError("the Mini-P tiling supports inputs 'p' and 'pr' only")
        if freq_windows is None and valid_bias:
            raise ValueError("valid_bias belongs to the Mini-P tiling (freq_windows)")
        if overparam and norm != "bn":
            raise ValueError("overparam needs norm 'bn' (RepVGG branches carry their own BatchNorm)")
        self.contract = get_audio_contract(audio_contract)
        df_bins = DF_BINS if df_bins is None else int(df_bins)
        if valid_bias is None:
            valid_bias = freq_windows is not None and inputs != "p"
        self.cfg = dict(c1=c1, c2=c2, f=f, k=k, l=l, heads=heads, inputs=inputs, mask=mask,
                        df_taps=df_taps, norm=norm)
        # new keys enter cfg only when they differ from the legacy defaults, so legacy cfgs are unchanged
        for key, val, dflt in (("audio_contract", audio_contract, None), ("freq_windows", freq_windows, None),
                               ("valid_bias", bool(valid_bias), freq_windows is not None and inputs != "p"),
                               ("df_bins", df_bins, DF_BINS),
                               ("df_lags", list(df_lags) if df_lags is not None else None, None),
                               ("gru_init", gru_init, "default"), ("fp32_islands", bool(fp32_islands), False),
                               ("overparam", bool(overparam), False)):
            if val != dflt or (key == "valid_bias" and freq_windows is not None):
                self.cfg[key] = val
        self.c1, self.c2, self.f, self.k, self.l = c1, c2, f, k, l
        self.inputs, self.mask_kind, self.df_taps = inputs, mask, df_taps
        self.df_bins = df_bins
        self.df_lags = df_lags if df_lags is not None else tuple(range(df_taps))
        self.df_max_lag = max(self.df_lags) if df_taps else 0
        self.freq_windows, self.gru_init = freq_windows, gru_init
        self.fp32_islands, self.overparam = bool(fp32_islands), bool(overparam)
        self.n_in, self.n_raw = INPUTS[inputs], RAW[inputs]
        self.uses_ref = inputs != "p"
        if freq_windows is None:
            if df_taps and (df_bins % 4 or not 0 < df_bins <= 256):
                raise ValueError("df_bins must be a multiple of 4 in (0, 256] on the native tiling")
            self.pre = _cbr(self.n_in, c1, 8, norm, stride=4, padding=2)            # 257 -> 64 bins
            n_pos = COARSE
        else:
            self.n_in -= int(self.uses_ref)   # no validity plane: v x valid_vec instead
            self.res, b0, p0 = [], 0, 0     # (width, bin start, bin end, pos start, pos end) per resolution
            for w, cnt in FREQ_WINDOWS[freq_windows]:
                self.res.append((w, b0, b0 + w * cnt, p0, p0 + cnt)); b0 += w * cnt; p0 += cnt
            assert b0 == N_BINS - 1
            n_pos = p0
            self.inp = nn.ModuleList(nn.Conv1d(self.n_in, c1, w, stride=w, bias=norm != "bn") for w, *_ in self.res)
            # v x a learned C1-vector as a 1x1 conv on the (N,1,1) validity: C1 MACs, C1 entries, no Unsqueeze
            self.valid_vec = nn.Conv1d(1, c1, 1, bias=False) if valid_bias else None
            self.pre_norm = nn.BatchNorm1d(c1) if norm == "bn" else nn.Identity()
            if df_taps:
                edges = {r[2] for r in self.res}
                if df_bins not in edges:
                    raise ValueError(f"df_bins must end on a window boundary of {freq_windows}: {sorted(edges)}")
                self.df_res = [r for r in self.res if r[2] <= df_bins]
        self.n_pos = n_pos
        self.enc = nn.ModuleList(_cbr(c1, c1, 3, norm, padding=1) for _ in range(l))
        self.rf_pre_lin, self.rf_pre_conv = nn.Linear(n_pos, f, bias=False), nn.Conv1d(c1, c2, 1)
        self.pe = nn.Parameter(torch.zeros(f, c2))
        self.blocks = nn.ModuleList(Block(c2, heads) for _ in range(k))
        self.rf_post_lin, self.rf_post_conv = nn.Linear(f, n_pos, bias=False), nn.Conv1d(c2, c1, 1)
        self.dec = nn.ModuleList(nn.Sequential(_cbr(2 * c1, c1, 1, norm), _cbr(c1, c1, 3, norm, padding=1))
                                 for _ in range(l))
        self.post = _cbr(2 * c1, c1, 1, norm)
        if freq_windows is None:
            # 64 -> 257 bins (output_padding emits the Nyquist bin), complex mask
            self.up = nn.ConvTranspose1d(c1, 2, 8, stride=4, padding=2, output_padding=1)
            if df_taps:  # DF taps from the lowest df_bins/4 coarse bins -> df_bins fine bins
                self.df = nn.ConvTranspose1d(c1, 2 * df_taps, 8, stride=4, padding=2)
        else:
            self.outs = nn.ModuleList(nn.ConvTranspose1d(c1, 2, w, stride=w) for w, *_ in self.res)
            self.nyq = nn.Parameter(torch.tensor([1.0, 0.0]))   # bin 256: learned complex constant mask
            if df_taps:
                self.df = nn.ModuleList(nn.ConvTranspose1d(c1, 2 * df_taps, w, stride=w) for w, *_ in self.df_res)
        # fixed exact 0/+-1 maps (non-persistent: not weights, not in checkpoints)
        self.register_buffer("w_pair", _fixed([[1 if i // 2 == o // 2 else 0 for i in range(self.n_raw)]
                                              for o in range(self.n_raw)]), persistent=False)
        self.register_buffer("w_pair2", _fixed([[1, 1], [1, 1]]), persistent=False)
        self.register_buffer("w_rr", _fixed([[1, 0], [1, 0]]), persistent=False)
        self.register_buffer("w_ii", _fixed([[0, 1], [0, 1]]), persistent=False)
        self.register_buffer("w_j", _fixed([[0, -1], [1, 0]]), persistent=False)
        self.register_buffer("g_p", torch.tensor([1.0, 1.0] + [0.0] * (self.n_raw - 2))[None, :, None], persistent=False)
        self.register_buffer("g_r", 1 - self.g_p, persistent=False)
        if inputs == "pr_pld":
            self.register_buffer("w_pld", _fixed([[1, 1, -1, -1], [1, 1, 1, 1]]), persistent=False)
        self.df_fixed = bool(df_taps) and (df_lags is not None or freq_windows is not None)
        if self.df_fixed:   # the lag-matched DF as fixed 0/+-1 1x1 convs: no Gather/stack/Slice in the step graph
            t2 = 2 * df_taps
            self.register_buffer("w_df_swap", _fixed([[1 if i == (o ^ 1) else 0 for i in range(t2)] for o in range(t2)]),
                                 persistent=False)
            self.register_buffer("w_df_re", _fixed([[(1 if i % 2 == 0 else -1) for i in range(t2)], [0] * t2]),
                                 persistent=False)
            self.register_buffer("w_df_im", _fixed([[0] * t2, [1] * t2]), persistent=False)
            ml = self.df_max_lag
            sel = [(ml - d) * 2 + c for d in self.df_lags[1:] for c in (0, 1)]   # cache channel of each lagged plane
            self.register_buffer("w_df_sel", _fixed([[1 if i == k else 0 for i in range(2 * ml)] for k in sel]),
                                 persistent=False)
        if freq_windows is not None:
            self.register_buffer("w_sel_p", _fixed([[1 if i == o else 0 for i in range(self.n_raw)] for o in range(2)]),
                                 persistent=False)
            self._in_split = [b1 - b0 for _, b0, b1, _, _ in self.res] + [1]          # + the Nyquist bin
            self._pos_split = [p1 - p0 for _, _, _, p0, p1 in self.res]
        if gru_init == "tc_matched":
            apply_gru_init(self.blocks, self.contract.hop)
        if self.overparam:
            _overparameterize(self)

    # ---- sizes -------------------------------------------------------------------------------
    @property
    def hidden_size(self):
        return self.k * self.f * self.c2

    @property
    def df_cache_size(self):
        return self.df_max_lag * 2 * self.df_bins

    @property
    def state_size(self):
        return self.hidden_size + self.df_cache_size

    def init_state(self, batch=1, device=None):
        return torch.zeros(batch, self.state_size, device=device)

    # ---- shared per-frame parts --------------------------------------------------------------
    def _planes(self, x, v):
        """x (N, n_raw, 257) raw RI, v (N,1) -> (N, n_in, 257) network planes, (N,2,257) compressed P."""
        with fp32_island(self.fp32_islands, x.device.type):
            if self.fp32_islands:
                x = _at_least_fp32(x)
            s = (F.conv1d(x * x, self.w_pair) + EPS) ** ((ALPHA - 1) / 2)       # |X|^(alpha-1) per channel
            xc = x * s
        pc = F.conv1d(xc, self.w_sel_p) if self.freq_windows is not None else xc[:, :2]
        if not self.uses_ref:
            return xc, pc
        vb = v.reshape(-1, 1, 1)
        planes = [xc * (self.g_p + self.g_r * vb)]                         # reference (and n_hat) gated by validity
        if self.inputs == "pr_pld":
            nd = F.conv1d(xc * xc, self.w_pld)                             # [|Pc|^2-|Rc|^2, |Pc|^2+|Rc|^2]
            planes.append(nd[:, :1] / (nd[:, 1:] + EPS) * vb)
        if self.freq_windows is None:
            planes.append(torch.zeros(1, 1, N_BINS, dtype=x.dtype, device=x.device) + vb)
        return (planes[0] if len(planes) == 1 else torch.cat(planes, 1)), pc

    def _encode(self, planes, v=None):
        if self.freq_windows is None:
            x = self.pre(planes)
        else:
            parts = torch.split(planes, self._in_split, -1)                     # one Split, not a Slice per window
            x = torch.cat([conv(parts[k]) for k, conv in enumerate(self.inp)], -1)
            if self.valid_vec is not None:
                x = x + self.valid_vec(v.reshape(-1, 1, 1))   # == a convolved constant validity plane, C1 MACs
            x = F.relu(self.pre_norm(x))
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
        if self.freq_windows is None:
            m = self.up(x)                                                     # (N, 2, 257)
        else:
            xs = torch.split(x, self._pos_split, -1)
            m = torch.cat([up(xs[k]) for k, up in enumerate(self.outs)]
                          + [x.new_zeros(x.shape[0], 2, 1) + self.nyq.reshape(1, 2, 1)], -1)
        if self.mask_kind == "bounded":
            mag = torch.sqrt(F.conv1d(m * m, self.w_pair2) + EPS)
            m = m * (torch.tanh(mag) / mag)
        y = F.conv1d(m, self.w_rr) * pc + F.conv1d(m, self.w_ii) * F.conv1d(pc, self.w_j)  # complex M*Pc
        if self.df_taps and self.df_fixed:
            # pc_hist (N, 2T, db): planes [h0r, h0i, h1r, h1i, ...] per tap lag; w the same layout
            db = self.df_bins
            if self.freq_windows is None:
                w = self.df(x[..., :db // 4])
            else:
                w = torch.cat([df(xs[k]) for k, df in enumerate(self.df)], -1)
            prod = pc_hist * w                                                 # [hr wr, hi wi, ...]
            cross = pc_hist * F.conv1d(w, self.w_df_swap)                      # [hr wi, hi wr, ...]
            low = F.conv1d(prod, self.w_df_re) + F.conv1d(cross, self.w_df_im)
            y = torch.cat([low, torch.split(y, [db, N_BINS - db], -1)[1]], -1)
        elif self.df_taps:
            w = self.df(x[..., :DF_BINS // 4]).reshape(-1, self.df_taps, 2, DF_BINS)
            wr, wi, hr, hi = w[:, :, 0], w[:, :, 1], pc_hist[:, :, 0], pc_hist[:, :, 1]
            low = torch.stack([(hr * wr - hi * wi).sum(1), (hr * wi + hi * wr).sum(1)], 1)
            y = torch.cat([low, y[..., DF_BINS:]], -1)
        return y

    def _decompress(self, y):
        """Compressed RI -> raw RI: Y * |Y|^(1/alpha - 1)."""
        with fp32_island(self.fp32_islands, y.device.type):
            if self.fp32_islands:
                y = _at_least_fp32(y)
            return y * (F.conv1d(y * y, self.w_pair2) + EPS) ** ((1 / ALPHA - 1) / 2)

    def _gru_seq(self, blk, seq):
        if not self.fp32_islands:
            return blk.rnn(seq)[0]
        with fp32_island(True, seq.device.type):
            return blk.rnn(_at_least_fp32(seq))[0]

    # ---- offline -----------------------------------------------------------------------------
    def forward(self, spec6, feats=None, ref_avail=None):
        if self.overparam and self.training:
            with parametrize.cached():   # each factor product is formed once per step, not once per access
                return self._forward(spec6, ref_avail)
        return self._forward(spec6, ref_avail)

    def _forward(self, spec6, ref_avail=None):
        b, nb, t, _ = spec6.shape
        n = b * t
        x = spec6[..., :self.n_raw].permute(0, 2, 3, 1).reshape(n, self.n_raw, nb)
        v = spec6.new_ones(n, 1) if ref_avail is None else ref_avail.to(spec6.dtype).reshape(n, 1)
        planes, pc = self._planes(x, v)
        tok, skips = self._encode(planes, v)
        for blk in self.blocks:
            seq = tok.reshape(b, t, self.f, self.c2).transpose(1, 2).reshape(b * self.f, t, self.c2)
            y = self._gru_seq(blk, seq).reshape(b, self.f, t, self.c2).transpose(1, 2).reshape(n, self.f, self.c2)
            tok = blk.freq(tok + blk.rnn_fc(y.to(tok.dtype) if self.fp32_islands else y), n, self.f)
        hist = None
        if self.df_taps:
            db = self.df_bins
            low = pc[..., :db].reshape(b, t, 2, db)
            if self.df_fixed:
                hist = torch.cat([F.pad(low, (0, 0, 0, 0, d, 0))[:, :t] for d in self.df_lags], 2).reshape(n, 2 * self.df_taps, db)
            else:
                hist = torch.stack([F.pad(low, (0, 0, 0, 0, d, 0))[:, :t] for d in self.df_lags], 2)
                hist = hist.reshape(n, self.df_taps, 2, db)
        out = self._decompress(self._decode(tok, skips, pc, hist))
        return out.reshape(b, t, 2, nb).permute(0, 3, 1, 2)

    # ---- streaming ---------------------------------------------------------------------------
    def step(self, spec, valid, state):
        b = spec.shape[0]
        v = spec.new_ones(b, 1) if valid is None else valid
        planes, pc = self._planes(spec if spec.shape[1] == self.n_raw else spec[:, :self.n_raw], v)
        tok, skips = self._encode(planes, v)
        tok = tok.reshape(b * self.f, self.c2)                                  # 2-D tokens: every Linear is a Gemm
        fc, new = self.f * self.c2, []
        for i, blk in enumerate(self.blocks):
            h = gru_cell(tok, state[:, i * fc:(i + 1) * fc].reshape(b * self.f, self.c2), blk.rnn)
            new.append(h.reshape(b, fc))
            tok = blk.freq(tok + blk.rnn_fc(h), b, self.f)
        hist = None
        if self.df_taps and self.df_fixed:
            db, ml = self.df_bins, self.df_max_lag
            cur = torch.split(pc, [db, N_BINS - db], -1)[0]                    # (b, 2, db)
            cache = state[:, self.hidden_size:].reshape(b, 2 * ml, db)         # oldest first: lags ml .. 1
            hist = torch.cat([cur, F.conv1d(cache, self.w_df_sel)], 1)          # (b, 2T, db), lags in df_lags order
            new.append(torch.cat([cache[:, 2:], cur], 1).reshape(b, 2 * ml * db))
        elif self.df_taps:
            cur = pc[..., :DF_BINS].reshape(b, 2 * DF_BINS)
            cache = state[:, self.hidden_size:]                                 # oldest first
            frames = [cur] + [cache[:, j * 2 * DF_BINS:(j + 1) * 2 * DF_BINS] for j in range(self.df_taps - 2, -1, -1)]
            hist = torch.stack(frames, 1).reshape(b, self.df_taps, 2, DF_BINS)
            new.append(torch.cat([cache[:, 2 * DF_BINS:], cur], 1))             # Slice+Concat, no ScatterND
        out = self._decompress(self._decode(tok.reshape(b, self.f, self.c2), skips, pc, hist))
        return out, torch.cat(new, 1)


    # ---- training-time over-parameterization ------------------------------------------------------
    def fold(self) -> "VaaniFE":
        """The plain network: every ExpandNets product and RepVGG branch folded exactly into the deployed layers.
        Its entries and MACs are exactly the plain model's; the result loads into VaaniFE(**cfg without overparam)."""
        if not self.overparam:
            return copy.deepcopy(self)
        cfg = {k: v for k, v in self.cfg.items() if k != "overparam"}
        with torch.random.fork_rng(devices=[]):       # building the shell must not move the caller's RNG
            plain = VaaniFE(**cfg)
        dev = next(self.parameters()).device
        plain.to(dev).train(self.training)
        reps = {n for n, m in self.named_modules() if isinstance(m, RepConv3)}
        with torch.no_grad():
            for name, pm in plain.named_modules():
                if name in reps:
                    pm.load_state_dict(self.get_submodule(name).folded().state_dict())
                    continue
                if any(name.startswith(r + ".") for r in reps):
                    continue                              # inside a folded RepVGG block: already loaded
                src = self.get_submodule(name) if name else self
                for pn, t in list(pm.named_parameters(recurse=False)) + list(pm.named_buffers(recurse=False)):
                    t.copy_(getattr(src, pn).detach())   # a parametrized weight reads as its factor product
        return plain


class RepConv3(nn.Module):
    """RepVGG training block for a 3-tap Conv + BN + ReLU: (3-tap + BN) + (1-tap + BN) + (identity BN), then ReLU."""

    def __init__(self, c):
        super().__init__()
        self.c3, self.b3 = nn.Conv1d(c, c, 3, padding=1, bias=False), nn.BatchNorm1d(c)
        self.c1, self.b1 = nn.Conv1d(c, c, 1, bias=False), nn.BatchNorm1d(c)
        self.bid = nn.BatchNorm1d(c)

    def forward(self, x):
        return F.relu(self.b3(self.c3(x)) + self.b1(self.c1(x)) + self.bid(x))

    @staticmethod
    def _fuse(kernel, bn):
        std = (bn.running_var + bn.eps).sqrt()
        return kernel * (bn.weight / std)[:, None, None], bn.bias - bn.running_mean * bn.weight / std

    @torch.no_grad()
    def folded(self):
        c = self.c3.out_channels
        k3, b3 = self._fuse(self.c3.weight.double(), _bn64(self.b3))
        k1, b1 = self._fuse(F.pad(self.c1.weight.double(), (1, 1)), _bn64(self.b1))
        eye = torch.zeros(c, c, 3, dtype=torch.float64)
        eye[torch.arange(c), torch.arange(c), 1] = 1.0
        ki, bi = self._fuse(eye, _bn64(self.bid))
        out = _cbr(c, c, 3, "bn", padding=1)
        out[0].weight.copy_((k3 + k1 + ki).float())
        out[0].bias.copy_((b3 + b1 + bi).float())
        bn = out[1]   # identity in eval: (x - 0) / sqrt(var + eps) * 1 + 0 with var + eps == 1
        bn.running_mean.zero_(); bn.running_var.fill_(1.0 - bn.eps); bn.weight.fill_(1.0); bn.bias.zero_()
        return out.to(self.c3.weight.device)


def _bn64(bn):
    class _B:
        pass
    b = _B()
    b.running_var, b.running_mean = bn.running_var.double(), bn.running_mean.double()
    b.weight, b.bias, b.eps = bn.weight.double(), bn.bias.double(), bn.eps
    return b


class ExpandFactor(nn.Module):
    """ExpandNets factorisation of a weight: W.reshape(rows, cols) = A (rows x inner) @ B (inner x cols), inner = 4 x
    the layer's output width. Factors take the standard (Linear) initialisation, as in ExpandNets."""

    def __init__(self, shape, rows, inner):
        super().__init__()
        self.shape = tuple(shape)
        cols = math.prod(shape) // rows
        la, lb = nn.Linear(inner, rows, bias=False), nn.Linear(cols, inner, bias=False)
        self.a, self.b = nn.Parameter(la.weight.detach().clone()), nn.Parameter(lb.weight.detach().clone())

    def forward(self, _w):
        return (self.a @ self.b).reshape(self.shape)


def _expand(mod, name, rows, out_width):
    w = getattr(mod, name)
    parametrize.register_parametrization(mod, name, ExpandFactor(w.shape, rows, 4 * out_width), unsafe=True)


def _overparameterize(m: VaaniFE):
    """RepVGG on the 3-tap convs (encoder, decoder); ExpandNets factors on the other linear maps."""
    for i, e in enumerate(m.enc):
        m.enc[i] = RepConv3(m.c1)
    for d in m.dec:
        d[1] = RepConv3(m.c1)
    convs = [m.rf_pre_conv, m.rf_post_conv, m.post[0]] + [d[0][0] for d in m.dec]
    if m.freq_windows is not None:
        convs += list(m.inp)
    else:
        convs.append(m.pre[0])
    for c in convs:                               # Conv1d weight (out, in, k)
        _expand(c, "weight", c.out_channels, c.out_channels)
    tconvs = list(m.outs) if m.freq_windows is not None else [m.up]
    if m.df_taps:
        tconvs += list(m.df) if m.freq_windows is not None else [m.df]
    for c in tconvs:                              # ConvTranspose1d weight (in, out, k)
        _expand(c, "weight", c.in_channels, c.out_channels)
    for lin in (m.rf_pre_lin, m.rf_post_lin):
        _expand(lin, "weight", lin.out_features, lin.out_features)
    for blk in m.blocks:
        for lin in (blk.rnn_fc, blk.mix_fc, blk.mix.qkv):
            _expand(lin, "weight", lin.out_features, lin.out_features)
        for name in ("weight_ih_l0", "weight_hh_l0"):
            _expand(blk.rnn, name, 3 * blk.rnn.hidden_size, 3 * blk.rnn.hidden_size)


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
        spec = torch.randn(1, self.m.n_raw, N_BINS, generator=g) * 0.1
        state = self.m.init_state(1)
        return (spec, torch.ones(1, 1), state) if self.m.uses_ref else (spec, state)


def frame_to_step(spec_frame):
    """(B,257,1,n) offline-layout frame -> (B,n,257) step layout."""
    return spec_frame[:, :, 0].transpose(1, 2)


def step_to_frame(out):
    """(B,2,257) step output -> (B,257,1,2) offline layout."""
    return out.transpose(1, 2)[:, :, None]

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


def profile_of(model: VaaniFE) -> str | None:
    """Network name of a model: the tier when its sizes are a named tier's, with the Mini-P tiling ("mini_p18") or
    the deep-filter band of a plain tiling ("mini_df96", Arm R); None for an unnamed size."""
    c = model.cfg
    tier = next((t for t, d in TIERS.items() if all(c.get(k) == v for k, v in d.items())), None)
    if tier is None:
        return None
    fw = c.get("freq_windows")
    if fw is not None:
        return f"{tier}_{fw}" if isinstance(fw, str) else f"{tier}_fw{len(fw)}"
    if c.get("df_bins") is not None:
        return f"{tier}_df{c['df_bins']}"
    return tier


def summary(model: VaaniFE):
    """Budget figures of the deployed (folded) network at its contract's hop rate."""
    if model.overparam:
        model = model.fold()
    macs = count_macs(model)
    hps = model.contract.hops_per_s
    return {"cfg": dict(model.cfg), "params": param_count(model), "params_training_form": param_count(model, False),
            "audio_contract": model.contract.audio_contract_id, "hops_per_s": hps,
            "mac_per_hop": macs, "mmac_per_s": round(macs * hps / 1e6, 3),
            "state_floats": model.state_size, "state_bytes": model.state_size * 4,
            "gru_state_bytes": model.hidden_size * 4, "df_cache_bytes": model.df_cache_size * 4}
