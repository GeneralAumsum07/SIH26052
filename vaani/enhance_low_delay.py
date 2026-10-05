"""The shared VaaniFE enhancement runner and the re-synthesis loss (low-delay plan, Sections 3.6 and 4 / Task 3).

Every VaaniFE route (checkpoint scoring, the frozen and composite validation screens, vaani/eval.py, fault and
physical evaluation) dispatches here on the model's audio contract:
  C0          the legacy route: front_end (limiter + ref_policy), the centered 512/256 STFT, the model with its
              frame validity (pipeline.frame_avail), stft.istft
  low delay   enhance_low_delay: LowDelayFrontend, the asymmetric analysis, the model with per-contract frame
              validity, the explicit synthesis. It never calls the legacy transform on model output and never runs
              the legacy NLMS pipeline (tests/test_low_delay_eval.py); inputs pr_nhat get n_hat from the frontend's
              decoupled-cadence NLMS (vaani.dsp.decoupled_nlms, owner decision D5).
Both routes then go through the same metric code on identical clips.

Resampler in the loop (Section 4): with `resampler`, enhance_low_delay takes 48 kHz input, runs the pair's decimator
before the model and its interpolator after it, and returns 48 kHz output (not delay-compensated: the delay is
physical). Quality measured that way is scored against `filter_reference(clean)`, the clean reference through the
same pair, so the score measures the model and not the filter's phase. The registered offline metrics stay at 16 kHz
without a resampler.

Loss (Section 3.6): ResynthesisFELoss synthesizes the low-delay model output, re-analyzes it and the clean target with
the legacy 512/256 stft.stft, and calls the unchanged FELoss (r8 weights and kappa) on those spectra, with the
waveforms passed so FELoss does not invert spectra it already knows. The consistency term is zero on these spectra up
to rounding, so its weight is 0; its value is still logged. NativeFELoss is the Stage-2 native-domain ablation.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from vaani.audio_contract import AudioContract, contract_of, get_audio_contract
from vaani.dsp import low_delay_stft as ld
from vaani.dsp import pipeline, stft
from vaani.dsp.low_delay_frontend import LowDelayFrontend

SR = 16000
WARMUP_SAMPLES = 7680        # 480 ms past-context warm-up prefix (Stage-2 item 4): divisible by 96, 128 and 256


# ---- model inputs --------------------------------------------------------------------------------
def ld_model_inputs(mix: torch.Tensor, avail: torch.Tensor | None, contract, n_raw: int = 4):
    """(B, 2, N) processed mix ((B, 3, N) with n_hat for inputs pr_nhat), (B, N) per-sample availability ->
    (spec (B,257,T,n_raw), frame validity (B,T))."""
    c = get_audio_contract(contract)
    if n_raw not in (4, 6) or mix.shape[1] != n_raw // 2:
        raise ValueError(f"the low-delay route takes inputs 'pr' (2 channels) or 'pr_nhat' (3 channels with n_hat); "
                         f"got {mix.shape[1]} channels for {n_raw} raw planes")
    spec = torch.cat([ld.analyze(mix[:, i], c) for i in range(mix.shape[1])], dim=-1)
    valid = None if avail is None else ld.frame_validity(avail, c, spec.shape[2]).to(spec.dtype)
    return spec, valid


def _dsp_needs_nlms(cfg) -> bool:
    return (cfg.get("model_cfg") or {}).get("inputs") == "pr_nhat"


# ---- runners -----------------------------------------------------------------------------------
@torch.no_grad()
def enhance_low_delay(mix, available, model, contract, resampler=None, dsp_cfg=None, device=None,
                      preprocessed=False, controller_on=True):
    """(2, N) mix (16 kHz; 48 kHz with `resampler`) + availability ((N,) bool or None) -> (y (N,), diagnostics).
    preprocessed=True: `mix` is already the frontend output (validation screens cache it; (3, N) with n_hat for a
    model with inputs pr_nhat). controller_on gates the decoupled NLMS's adaptation (pr_nhat only).
    Diagnostics carry absolute input sample positions: for frame j, `computable_at` = (j+1)H - 1 and
    `release` = [(j+1)H - L, (j+2)H - L)."""
    c = get_audio_contract(contract)
    if c.is_legacy:
        raise ValueError("enhance_low_delay: C0 uses the legacy route (enhance_fe dispatches)")
    device = torch.device(device or "cpu")
    x = np.asarray(mix, np.float32)
    n_in = x.shape[1]
    av = np.ones(n_in, bool) if available is None else np.asarray(available, bool)
    nhat = getattr(model, "n_raw", 4) == 6
    if resampler is not None:
        if n_in % 3:
            raise ValueError("48 kHz input length must be a multiple of 3")
        x = resampler.decimator(2)(x)
        av = av.reshape(-1, 3).all(1)
    n = x.shape[1]
    if n == 0:
        return np.zeros(n_in, np.float32), {"frames": 0}
    if not preprocessed:
        x, av = LowDelayFrontend(c, dsp_cfg, nhat=nhat, controller_on=controller_on).process_offline(x, av)
    was = model.training
    model.eval()
    xt = torch.from_numpy(np.ascontiguousarray(x))[None].to(device)
    spec, valid = ld_model_inputs(xt, torch.from_numpy(av)[None].to(device), c, 6 if nhat else 4)
    out = model(spec, None, valid).float()
    model.train(was)
    y, _ = ld.synthesize(out, [n], c)
    y = y[0].cpu().numpy().astype(np.float32)
    if resampler is not None:
        y = resampler.interpolator(1)(y[None])[0]
    t = spec.shape[2]
    j = np.arange(t)
    diag = {"frames": t, "contract": c.audio_contract_id, "validity": valid[0].cpu().numpy(),
            "computable_at": (j + 1) * c.hop - 1,
            "release_start": (j + 1) * c.hop - c.support, "release_end": (j + 2) * c.hop - c.support,
            "resampler": None if resampler is None else resampler.id}
    return y, diag


def filter_reference(clean, resampler):
    """The clean reference through the same resampler pair (48 kHz in, 48 kHz out), for scoring with the resampler
    in the loop."""
    x = np.atleast_2d(np.asarray(clean, np.float32))
    y = resampler.interpolator(x.shape[0])(resampler.decimator(x.shape[0])(x))
    return y[0] if np.ndim(clean) == 1 else y


@torch.no_grad()
def enhance_c0(mix, available, model, cfg, device=None):
    """The legacy VaaniFE route, validity included: front_end (or pipeline.run for pr_nhat), 512/256 STFT, model."""
    from vaani.data.dataset import front_end
    device = torch.device(device or "cpu")
    mix = np.asarray(mix, np.float32)
    n = mix.shape[1]
    dsp = cfg.get("dsp")
    if _dsp_needs_nlms(cfg):
        pol = (dsp or {}).get("ref_policy") is not None
        r = pipeline.run(mix, controller_on=cfg["controller_on"], dsp_cfg=dsp, ref_avail=available if pol else None)
        m2, nh = r["mix"], r["n_hat"]
        fa = r.get("ref_avail")
        if fa is None:
            fa = pipeline.frame_avail(np.ones(n, bool) if available is None else available, n // stft.HOP + 1)
    else:
        m2, fa = front_end(mix, dsp, available)
        nh = None
    x = torch.from_numpy(m2)[None].to(device)
    chans = [stft.stft(x[:, 0]), stft.stft(x[:, 1])]
    if nh is not None:
        chans.append(stft.stft(torch.from_numpy(nh)[None].to(device)))
    was = model.training
    model.eval()
    out = model(torch.cat(chans, -1), None, torch.from_numpy(fa)[None].to(device)).float()
    model.train(was)
    return stft.istft(out, length=n)[0].cpu().numpy()


@torch.no_grad()
def forward_fe_batch(model, cfg, mixes, avails=None, n_hats=None, device=None):
    """Frontend-processed clips of one length -> enhanced waveforms, batched (validation screens). mixes (B, 2, N),
    avails (B, N) per-sample availability or None, n_hats (B, N) for inputs pr_nhat (C0: pipeline.run's; low delay:
    the frontend's decoupled NLMS). Eval mode: BatchNorm uses its running statistics, so every item equals its
    batch-1 forward."""
    device = torch.device(device or "cpu")
    c = contract_of(cfg.get("model_cfg"))
    x = torch.as_tensor(np.ascontiguousarray(mixes, np.float32), device=device)
    b, _, n = x.shape
    av = None if avails is None else torch.as_tensor(np.asarray(avails), device=device).float()
    was = model.training
    model.eval()
    if c.is_legacy:
        chans = [stft.stft(x[:, 0]), stft.stft(x[:, 1])]
        if n_hats is not None:
            chans.append(stft.stft(torch.as_tensor(np.asarray(n_hats, np.float32), device=device)))
        spec = torch.cat(chans, -1)
        fa = None if av is None else ld.frame_validity(av, c, spec.shape[2])
        y = stft.istft(model(spec, None, fa).float(), length=n)
    else:
        if n_hats is not None:
            x = torch.cat([x, torch.as_tensor(np.asarray(n_hats, np.float32), device=device)[:, None]], 1)
        spec, valid = ld_model_inputs(x, av, c, getattr(model, "n_raw", 4))
        y, _ = ld.synthesize(model(spec, None, valid).float(), [n] * b, c)
    model.train(was)
    return y.cpu().numpy()


def enhance_fe(mix, available, model, cfg, device=None, resampler=None):
    """Contract dispatch for a VaaniFE model trained under `cfg` -> (N,) enhanced waveform."""
    c = contract_of(cfg.get("model_cfg"))
    if c.is_legacy:
        if resampler is not None:
            raise ValueError("resampler-in-the-loop scoring is defined for the low-delay route")
        return enhance_c0(mix, available, model, cfg, device)
    return enhance_low_delay(mix, available, model, c, resampler, cfg.get("dsp"), device,
                             controller_on=cfg.get("controller_on", True))[0]


def load_fe_checkpoint(path, device="cpu"):
    """(model in eval mode, config) of a VaaniFE checkpoint; over-parameterized weights are folded."""
    from vaani.train import build_model
    ck = torch.load(path, map_location="cpu", weights_only=True)
    cfg = ck["config"]
    if cfg["model"] != "vaani_fe":
        raise ValueError(f"{path} is a {cfg['model']!r} checkpoint, not vaani_fe")
    m = build_model("vaani_fe", model_cfg=cfg.get("model_cfg"))
    m.load_state_dict(ck["model"])
    if getattr(m, "overparam", False):
        m = m.fold()
    return m.to(device).eval(), cfg


def make_fe_system(path, device="cpu", resampler=None):
    """enhance(mix, avail) -> (N,) for a VaaniFE checkpoint, validity always passed."""
    m, cfg = load_fe_checkpoint(path, device)

    def enhance(mix, avail=None):
        return enhance_fe(mix, avail, m, cfg, device, resampler)
    enhance.contract = contract_of(cfg.get("model_cfg")).audio_contract_id
    return enhance


# ---- losses -------------------------------------------------------------------------------------
def _pad256(y: torch.Tensor) -> torch.Tensor:
    r = (-y.shape[-1]) % stft.HOP
    return torch.nn.functional.pad(y, (0, r)) if r else y


class ResynthesisFELoss(nn.Module):
    """FELoss on the synthesized waveform and its 512/256 spectra (Section 3.6). forward(pred (B,257,T,2) in the
    contract's domain, clean (B,N) waveform, frame_weight ignored (inert for loss: fe), is_clean, lengths=None,
    scored=None). scored=(a, b): only samples [a, b) enter the loss (the warm-up arm scores [7680, 71680)).
    Lengths other than multiples of 256 are zero-padded identically for prediction and target."""

    def __init__(self, fe_loss, contract):
        super().__init__()
        self.fe = fe_loss
        self.contract = get_audio_contract(contract)
        if self.contract.is_legacy:
            raise ValueError("ResynthesisFELoss is for low-delay contracts; C0 keeps its native FELoss")
        self.fe.w["consistency"] = 0.0     # zero up to rounding on re-analyzed spectra; still computed and logged

    # the trainer reads these like FELoss's
    @property
    def w(self):
        return self.fe.w

    @property
    def w_snr(self):
        return self.fe.w_snr

    @property
    def last_snr_clamp_fraction(self):
        return self.fe.last_snr_clamp_fraction

    @property
    def last_terms(self):
        return self.fe.last_terms

    @property
    def last_pesq_items(self):
        return self.fe.last_pesq_items

    def synthesize(self, pred, clean, lengths=None, scored=None):
        n = clean.shape[-1]
        lengths = [n] * clean.shape[0] if lengths is None else lengths
        y, mask = ld.synthesize(pred, lengths, self.contract)
        clean = clean[:, :y.shape[-1]] * mask.to(clean.dtype)
        if scored is not None:
            a, b = scored
            y, clean = y[:, a:b], clean[:, a:b]
        return _pad256(y), _pad256(clean)

    def forward(self, pred, clean, frame_weight=None, is_clean=None, noisy=None, lengths=None, scored=None):
        y, yc = self.synthesize(pred, clean, lengths, scored)
        return self.fe(stft.stft(y), stft.stft(yc), None, is_clean, y_pred=y, y_true=yc)


class NativeFELoss(ResynthesisFELoss):
    """Stage-2 ablation (Section 3.8 item 5): spectral terms and consistency on the low-delay spectra, weighted by the
    Section 3.1 boundary weights normalized by their sum; waveform terms on the synthesized waveform."""

    def __init__(self, fe_loss, contract, w_consistency=0.3):
        super().__init__(fe_loss, contract)
        self.fe.w["consistency"] = w_consistency
        self._w = {}   # (n, dtype, device) -> normalized weights; filled in eager warm-up, reused inside CUDA-graph capture

    def _weights(self, n, like):
        # a host->device copy inside capture is illegal (the 2026-09-30 box failure), so build the tensor once
        k = (n, like.dtype, like.device)
        if k not in self._w:
            w = torch.as_tensor(ld.boundary_weights(n, self.contract), dtype=like.dtype, device=like.device)
            self._w[k] = w / w.mean()
        return self._w[k]

    def forward(self, pred, clean, frame_weight=None, is_clean=None, noisy=None, lengths=None, scored=None):
        if scored is not None:
            raise ValueError("the native-domain ablation is not combined with the warm-up prefix")
        c = self.contract
        n = clean.shape[-1]
        lengths = [n] * clean.shape[0] if lengths is None else lengths
        y, mask = ld.synthesize(pred, lengths, c)
        yc = clean[:, :y.shape[-1]] * mask.to(clean.dtype)
        target = ld.analyze(yc, c)
        w = self._weights(n, pred)[None].expand(pred.shape[0], -1)     # mean(x * w) == sum(x w) / sum(w) per bin
        return self.fe(pred, target, w, is_clean, y_pred=y, y_true=yc, reanalyze=lambda s: ld.analyze(s, c))


def build_fe_loss(loss_cfg: dict, model_cfg: dict | None, loss_domain: str = "resynthesis"):
    """The r8 FE loss for a model_cfg: C0 -> FELoss; a low-delay contract -> ResynthesisFELoss (default) or
    NativeFELoss (loss_domain "native", the Stage-2 ablation)."""
    from vaani import losses
    base = losses.build_loss("fe", loss_cfg)
    c = contract_of(model_cfg)
    if c.is_legacy:
        return base
    if loss_domain == "native":
        return NativeFELoss(base, c, loss_cfg.get("w_consistency", 0.3))
    if loss_domain != "resynthesis":
        raise ValueError(f"loss_domain must be resynthesis or native, got {loss_domain!r}")
    return ResynthesisFELoss(base, c)


def term_grad_norms(loss_fn, params) -> dict:
    """Gradient norm of each weighted loss term w.r.t. `params` (logging only; needs loss_fn.fe.keep_live_terms)."""
    fe = getattr(loss_fn, "fe", loss_fn)
    live = getattr(fe, "live_terms", None) or {}
    params = [p for p in params if p.requires_grad]
    out = {}
    for k, t in live.items():
        wk = fe.w["mag"] if k in ("mag", "over") else fe.w.get(k, 0.0)
        if not wk or not t.requires_grad:
            out[k] = 0.0
            continue
        g = torch.autograd.grad(wk * t, params, retain_graph=True, allow_unused=True)
        out[k] = float(torch.sqrt(sum((x.detach().float() ** 2).sum() for x in g if x is not None)))
    return out
