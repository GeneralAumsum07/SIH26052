"""VaaniNet: GTCRN widened to three inputs (primary, reference, NLMS estimate; 27->16
first conv), mask on primary only. Encoder/StreamEncoder are re-declared here (vendored
files untouched). Architecture flags (`model_cfg` in the experiment config; defaults = r1/r2):
  film      zero-init FiLM shift from the 18 DSP features (r1/r2; measured inert, -0.003 STOI)
  coh       causal magnitude-squared-coherence map primary<->reference as a 10th input channel
  df_order  deep-filter taps across past frames: out[t] = sum_k M_k[t] * X[t-k]; 1 = plain CRM"""
import torch
import torch.nn as nn

from vaani.models.gtcrn import ERB, SFE, ConvBlock, GTConvBlock, DPGRNN, Decoder, Mask
from vaani.models import gtcrn_stream as gs

MAX_PARAMS = 60_000
N_FEAT = 18
N_SIG = 3  # primary, reference, n_hat
N_PRIM = 9  # primary's slice of the first-conv input channels (SFE keeps signal order)
COH_ALPHA = 0.9  # per-frame EMA for the coherence spectra: ~150 ms at a 16 ms hop
DF_CACHE_MAX = 4  # df_cache is always exported at this depth so the ONNX signature is fixed


def _validate_architecture(channels, up=.02, down=.3):
    # Grouped bidirectional GRUs split channels twice. Multiples of four preserve
    # the encoder/decoder and recurrent residual dimensions at every width.
    if not isinstance(channels, int) or channels < 4 or channels % 4:
        raise ValueError("channels must be a positive multiple of four (8/16/32 for the frontier)")
    if not 0 < up <= down <= 1:
        raise ValueError("noise-floor rates must satisfy 0 < up <= down <= 1")


def _noise_step(spec_t, state, up=.02, down=.3):
    """Power-domain recursive floor; state[:,1] is an explicit initialized flag.

    Seed from the first observed frame, not zero: otherwise the ratio is enormous
    at stream start and a long quiet lead-in is implicitly assumed. This is an
    asymmetric floor heuristic, not an unbiased noise PSD estimator.
    """
    power = spec_t[..., :2].float().square().sum(-1)
    floor = state[:, 0]
    rate = torch.where(power > floor, up, down)
    floor = torch.where(state[:, 1] > 0, floor + rate * (power - floor), power)
    features = torch.stack([torch.log1p(floor), torch.log1p((power / (floor + 1e-8)).clamp(max=1e4))], 1)
    return features, torch.stack([floor, torch.ones_like(floor)], 1)


def noise_floor_features(spec6, state=None, up=.02, down=.3):
    """Causal two-channel (B,2,T,257) map; same update used by the streaming twin."""
    if state is None:
        state = spec6.new_zeros(spec6.shape[0], 2, spec6.shape[1], dtype=torch.float32)
    frames = []
    for t in range(spec6.shape[2]):
        feat, state = _noise_step(spec6[:, :, t], state, up, down)
        frames.append(feat)
    return torch.stack(frames, 2), state


def _decoder(channels, stream=False):
    # Retain the original module/key layout so default checkpoints load exactly.
    decoder = gs.StreamDecoder() if stream else Decoder()
    if channels == 16:
        return decoder
    block = gs.StreamGTConvBlock if stream else GTConvBlock
    conv = gs.ConvBlock if stream else ConvBlock
    decoder.de_convs = nn.ModuleList([
        block(channels, channels, (3, 3), (1, 1), (0 if stream else 2*d, 1), (d, 1), use_deconv=True)
        for d in (5, 2, 1)
    ] + [conv(channels, channels, (1, 5), (1, 2), (0, 2), groups=2, use_deconv=True),
         conv(channels, 2, (1, 5), (1, 2), (0, 2), use_deconv=True, is_last=True)])
    return decoder


def _sig_feats(spec6):
    """(B,257,T,6) -> (B,9,T,257): per signal (mag, re, im)."""
    outs = []
    for i in range(N_SIG):
        re = spec6[..., 2 * i].permute(0, 2, 1); im = spec6[..., 2 * i + 1].permute(0, 2, 1)
        outs += [torch.sqrt(re ** 2 + im ** 2 + 1e-12), re, im]
    return torch.stack(outs, dim=1)


def _coh_step(spec_t, state):
    """One EMA step of the cross-spectra. spec_t (B,257,6), state (B,4,257) = Pxx, Pyy, Re Pxy, Im Pxy.
    Returns MSC (B,257) in [0,1] and the new state."""
    xr, xi, yr, yi = spec_t[..., 0], spec_t[..., 1], spec_t[..., 2], spec_t[..., 3]
    inst = torch.stack([xr * xr + xi * xi, yr * yr + yi * yi, xr * yr + xi * yi, xi * yr - xr * yi], dim=1)
    state = COH_ALPHA * state + (1 - COH_ALPHA) * inst
    msc = (state[:, 2] ** 2 + state[:, 3] ** 2) / (state[:, 0] * state[:, 1] + 1e-10)
    return msc.clamp(0.0, 1.0), state


def coherence_map(spec6, state=None):
    """(B,257,T,6) -> (B,1,T,257) causal MSC; the loop is per frame so batch and stream agree exactly."""
    B, F, T, _ = spec6.shape
    if state is None:
        state = spec6.new_zeros(B, 4, F)
    outs = []
    for t in range(T):
        m, state = _coh_step(spec6[:, :, t], state); outs.append(m)
    return torch.stack(outs, dim=1)[:, None], state


def _cat_coh(sig, coh):
    return sig if coh is None else torch.cat([sig, coh], dim=1)


FEAT_SCALE = [1 / 10.0, 1.0, 1 / 5.0, 1.0, 1.0, 1.0, *([1.0] * 8), 1 / 10.0, 1.0, 1.0, 1.0]


class _Encoder(nn.Module):
    def __init__(self, blocks, film=True, channels=16):
        super().__init__()
        self.en_convs = nn.ModuleList(blocks)
        self.film = None
        if film:
            self.film = nn.Linear(N_FEAT, channels)
            nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)  # inert at step 0
        # dB-valued features reach +/-40 while the rest are 0..1; unscaled they wreck the pretrained
        # encoder within the first epoch. Raw features stay the deploy contract, scaling lives here.
        self.register_buffer("feat_scale", torch.tensor(FEAT_SCALE), persistent=False)

    def _cond(self, x, feats):
        # x: (B,16,T,F). feats: (B,T,18) -> shift (B,16,T,1)
        if self.film is None:
            return x
        f = torch.clamp(feats * self.feat_scale, -3.0, 3.0)
        return x + self.film(f).permute(0, 2, 1)[..., None]


def _n_in(coh, noise_floor=False):
    return (N_SIG * 3 + int(coh) + 2 * int(noise_floor)) * 3


class Encoder(_Encoder):
    def __init__(self, film=True, coh=False, channels=16, noise_floor=False):
        super().__init__([
            ConvBlock(_n_in(coh, noise_floor), channels, (1, 5), stride=(1, 2), padding=(0, 2)),
            ConvBlock(channels, channels, (1, 5), stride=(1, 2), padding=(0, 2), groups=2),
            GTConvBlock(channels, channels, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(1, 1)),
            GTConvBlock(channels, channels, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(2, 1)),
            GTConvBlock(channels, channels, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(5, 1)),
        ], film, channels)

    def forward(self, x, feats):
        en_outs = []
        for i, blk in enumerate(self.en_convs):
            x = blk(x)
            if i == 0:
                x = self._cond(x, feats)
            en_outs.append(x)
        return x, en_outs


class DeepFilterHead(nn.Module):
    """Taps 1..K-1 of a per-bin complex FIR over past frames, from the decoder's last hidden map.
    Plain tanh, no BatchNorm: zero weights give exactly zero taps, so a CRM warm start is preserved
    at step 0 (a BN on an all-zero channel would divide its gradient by sqrt(eps))."""
    def __init__(self, order, channels=16):
        super().__init__()
        self.order = order
        self.conv = nn.ConvTranspose2d(channels, 2 * (order - 1), (1, 5), stride=(1, 2), padding=(0, 2))
        nn.init.zeros_(self.conv.weight); nn.init.zeros_(self.conv.bias)

    def forward(self, h):
        return torch.tanh(self.conv(h))  # (B,2(K-1),T,129)


def _apply_taps(mask_fn, taps, past):
    """taps (B,2(K-1),T,F) applied to past (B,K-1,2,T,F) (index k-1 = X[t-k]); returns (B,2,T,F)."""
    out = 0
    for k in range(past.shape[1]):
        out = out + mask_fn(taps[:, 2 * k:2 * k + 2], past[:, k])
    return out


class VaaniNet(nn.Module):
    """forward(spec6 (B,257,T,6), feats (B,T,18)) -> enhanced primary (B,257,T,2)."""

    def __init__(self, df_order: int = 1, film: bool = True, coh: bool = False,
                 channels: int = 16, noise_floor: bool = False, noise_floor_up=.02, noise_floor_down=.3):
        super().__init__()
        assert 1 <= df_order <= DF_CACHE_MAX, df_order
        _validate_architecture(channels, noise_floor_up, noise_floor_down)
        self.channels, self.use_noise_floor = channels, noise_floor
        self.noise_floor_up, self.noise_floor_down = noise_floor_up, noise_floor_down
        self.df_order, self.use_coh = df_order, coh
        self.erb = ERB(65, 64); self.sfe = SFE(3, 1)
        self.encoder = Encoder(film, coh, channels, noise_floor)
        self.dpgrnn1 = DPGRNN(channels, 33, channels); self.dpgrnn2 = DPGRNN(channels, 33, channels)
        self.decoder = _decoder(channels); self.mask = Mask()
        self.df = DeepFilterHead(df_order, channels) if df_order > 1 else None

    def _decode(self, feat, en_outs):
        """Decoder.forward, but also hands back the last block's input for the deep-filter head."""
        d = self.decoder.de_convs; n = len(d)
        x = feat
        for i in range(n - 1):
            x = d[i](x + en_outs[n - 1 - i])
        h = x + en_outs[0]
        return d[-1](h), h

    def forward(self, spec6, feats):
        prim = spec6[..., :2]
        coh = coherence_map(spec6)[0] if self.use_coh else None
        feat = _cat_coh(_sig_feats(spec6), coh)
        if self.use_noise_floor:
            floor, _ = noise_floor_features(spec6, up=self.noise_floor_up, down=self.noise_floor_down)
            feat = torch.cat([feat, floor], 1)
        feat = self.erb.bm(feat)
        feat = self.sfe(feat)                                    # (B,27|30,T,129)
        feat, en_outs = self.encoder(feat, feats)
        feat = self.dpgrnn1(feat); feat = self.dpgrnn2(feat)
        m, h = self._decode(feat, en_outs)
        p = prim.permute(0, 3, 2, 1)                             # (B,2,T,257)
        out = self.mask(self.erb.bs(m), p)
        if self.df is not None:
            taps = self.erb.bs(self.df(h))
            # X[t-k] by zero-padding the front of the time axis: same as an all-zero df_cache at stream start
            past = torch.stack([nn.functional.pad(p, (0, 0, k, 0))[:, :, :p.shape[2]] for k in range(1, self.df_order)], dim=1)
            out = out + _apply_taps(self.mask, taps, past)
        return out.permute(0, 3, 2, 1)

    def load_state_dict(self, sd, strict=True):
        # checkpoints saved before feat_scale became non-persistent carry the constant
        return super().load_state_dict({k: v for k, v in sd.items() if k != "encoder.feat_scale"}, strict)

    def warm_start(self, sd):
        """Load a checkpoint of a possibly narrower architecture: first-conv input slices it lacks stay zero,
        FiLM weights are dropped when film is off, deep-filter taps keep their zero init."""
        own = self.state_dict(); first = "encoder.en_convs.0.conv.weight"
        if sd[first].shape[0] != self.channels:
            raise ValueError("Width changes require scratch initialization; warm_start only widens input features")
        source_extra = sd[first].shape[1] // 3 - 9
        source_coh, source_floor = bool(source_extra % 2), source_extra >= 2
        if source_extra not in (0, 1, 2, 3) or (source_coh and not self.use_coh) or (source_floor and not self.use_noise_floor):
            raise ValueError("Warm start cannot remove existing spectral features")
        for k, v in sd.items():
            if k == "encoder.feat_scale" or (k.startswith("encoder.film") and self.encoder.film is None):
                continue
            if k not in own:
                raise KeyError(f"checkpoint key {k!r} has no matching VaaniNet parameter")
            if own[k].shape == v.shape:
                own[k] = v
            elif k == first and own[k].shape[1] > v.shape[1]:
                new = torch.zeros_like(own[k]); new[:, :27] = v[:, :27]
                if source_coh:
                    new[:, 27:30] = v[:, 27:30]
                if source_floor:
                    start = 27 + 3 * int(self.use_coh)
                    new[:, start:start+6] = v[:, 27+3*int(source_coh):]
                own[k] = new
            else:
                raise KeyError(f"shape mismatch for {k!r}: {tuple(v.shape)} -> {tuple(own[k].shape)}")
        self.load_state_dict(own)
        return self

    @classmethod
    def from_pretrained_gtcrn(cls, ckpt_path, **model_cfg):
        """Copy every GTCRN weight; first conv gets primary slice, rest stays zero."""
        v = cls(**model_cfg)
        if v.channels != 16:
            raise ValueError("The pretrained GTCRN has channels=16; other widths require scratch initialization")
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
        own = v.state_dict()
        first = "encoder.en_convs.0.conv.weight"
        # Every GTCRN key must map 1:1 (only the first conv differs in shape); no silent skips.
        for k, w in sd.items():
            if k == first:
                new = torch.zeros_like(own[k]); new[:, :N_PRIM] = w   # primary slice
                own[k] = new
            elif k in own and own[k].shape == w.shape:
                own[k] = w
            else:
                raise KeyError(f"checkpoint key {k!r} has no matching VaaniNet parameter")
        missing = {k for k in set(own) - set(sd) if not (k.startswith("encoder.film") or k.startswith("df."))}
        if missing:
            raise KeyError(f"checkpoint lacks {sorted(missing)}")
        v.load_state_dict(own)
        return v


class StreamEncoder(_Encoder):
    def __init__(self, film=True, coh=False, channels=16, noise_floor=False):
        super().__init__([
            gs.ConvBlock(_n_in(coh, noise_floor), channels, (1, 5), stride=(1, 2), padding=(0, 2)),
            gs.ConvBlock(channels, channels, (1, 5), stride=(1, 2), padding=(0, 2), groups=2),
            gs.StreamGTConvBlock(channels, channels, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(1, 1)),
            gs.StreamGTConvBlock(channels, channels, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(2, 1)),
            gs.StreamGTConvBlock(channels, channels, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(5, 1)),
        ], film, channels)

    def forward(self, x, feats, conv_cache, tra_cache):
        en_outs = []
        x = self._cond(self.en_convs[0](x), feats); en_outs.append(x)
        x = self.en_convs[1](x); en_outs.append(x)
        x, conv_cache[:, :, :2, :], tra_cache[0] = self.en_convs[2](x, conv_cache[:, :, :2, :], tra_cache[0]); en_outs.append(x)
        x, conv_cache[:, :, 2:6, :], tra_cache[1] = self.en_convs[3](x, conv_cache[:, :, 2:6, :], tra_cache[1]); en_outs.append(x)
        x, conv_cache[:, :, 6:16, :], tra_cache[2] = self.en_convs[4](x, conv_cache[:, :, 6:16, :], tra_cache[2]); en_outs.append(x)
        return x, en_outs, conv_cache, tra_cache


class StreamVaaniNet(nn.Module):
    """Frame-by-frame twin; load weights via convert_to_stream(stream, batch). Caches as StreamGTCRN plus
    df_cache (1,257,DF_CACHE_MAX-1,2): past primary spectra, newest first, and coh_cache (1,4,257)."""

    def __init__(self, df_order: int = 1, film: bool = True, coh: bool = False,
                 channels: int = 16, noise_floor: bool = False, noise_floor_up=.02, noise_floor_down=.3):
        super().__init__()
        assert 1 <= df_order <= DF_CACHE_MAX, df_order
        _validate_architecture(channels, noise_floor_up, noise_floor_down)
        self.channels, self.use_noise_floor = channels, noise_floor
        self.noise_floor_up, self.noise_floor_down = noise_floor_up, noise_floor_down
        self.df_order, self.use_coh = df_order, coh
        self.erb = gs.ERB(65, 64); self.sfe = gs.SFE(3, 1)
        self.encoder = StreamEncoder(film, coh, channels, noise_floor)
        self.dpgrnn1 = gs.DPGRNN(channels, 33, channels); self.dpgrnn2 = gs.DPGRNN(channels, 33, channels)
        self.decoder = _decoder(channels, stream=True); self.mask = gs.Mask()
        self.df = DeepFilterHead(df_order, channels) if df_order > 1 else None

    def _decode(self, x, en_outs, conv_cache, tra_cache):
        d = self.decoder.de_convs
        x, conv_cache[:, :, 6:16, :], tra_cache[0] = d[0](x + en_outs[4], conv_cache[:, :, 6:16, :], tra_cache[0])
        x, conv_cache[:, :, 2:6, :], tra_cache[1] = d[1](x + en_outs[3], conv_cache[:, :, 2:6, :], tra_cache[1])
        x, conv_cache[:, :, :2, :], tra_cache[2] = d[2](x + en_outs[2], conv_cache[:, :, :2, :], tra_cache[2])
        x = d[3](x + en_outs[1])
        h = x + en_outs[0]
        return d[4](h), h, conv_cache, tra_cache

    def forward(self, spec6, feats, conv_cache, tra_cache, inter_cache, df_cache, coh_cache, noise_cache=None):
        prim = spec6[..., :2]
        if self.use_coh:
            msc, coh_cache = _coh_step(spec6[:, :, 0], coh_cache); coh = msc[:, None, None, :]
        else:
            coh = None
        feat = _cat_coh(_sig_feats(spec6), coh)
        if self.use_noise_floor:
            if noise_cache is None:
                raise ValueError("noise_floor model requires its per-stream noise_cache")
            floor, noise_cache = _noise_step(spec6[:, :, 0], noise_cache, self.noise_floor_up, self.noise_floor_down)
            feat = torch.cat([feat, floor[:, :, None]], 1)
        feat = self.sfe(self.erb.bm(feat))
        feat, en_outs, conv_cache[0], tra_cache[0] = self.encoder(feat, feats, conv_cache[0], tra_cache[0])
        feat, inter_cache[0] = self.dpgrnn1(feat, inter_cache[0])
        feat, inter_cache[1] = self.dpgrnn2(feat, inter_cache[1])
        m_feat, h, conv_cache[1], tra_cache[1] = self._decode(feat, en_outs, conv_cache[1], tra_cache[1])
        p = prim.permute(0, 3, 2, 1)
        out = self.mask(self.erb.bs(m_feat), p)
        if self.df is not None:
            taps = self.erb.bs(self.df(h))
            past = df_cache[:, :, :self.df_order - 1].permute(0, 2, 3, 1)[:, :, :, None, :]  # (B,K-1,2,1,257)
            out = out + _apply_taps(self.mask, taps, past)
        if self.encoder.film is None:
            out = out + 0.0 * feats[:, 0, 0, None, None, None]  # keeps `feats` in the traced graph: the ONNX signature must not depend on model_cfg
        # shift the newest primary frame in regardless of df_order so the cache contract does not depend on it
        df_cache = torch.cat([prim, df_cache[:, :, :-1]], dim=2)
        outputs = (out.permute(0, 3, 2, 1), conv_cache, tra_cache, inter_cache, df_cache, coh_cache)
        return (*outputs, noise_cache) if self.use_noise_floor else outputs


def init_caches(device="cpu", channels=16, noise_floor=False):
    _validate_architecture(channels)
    conv_cache = torch.zeros(2, 1, channels, 16, 33, device=device)
    tra_cache = torch.zeros(2, 3, 1, 1, channels, device=device)
    inter_cache = torch.zeros(2, 1, 33, channels, device=device)
    df_cache = torch.zeros(1, 257, DF_CACHE_MAX - 1, 2, device=device)
    coh_cache = torch.zeros(1, 4, 257, device=device)
    caches = (conv_cache, tra_cache, inter_cache, df_cache, coh_cache)
    return (*caches, torch.zeros(1, 2, 257, device=device)) if noise_floor else caches
