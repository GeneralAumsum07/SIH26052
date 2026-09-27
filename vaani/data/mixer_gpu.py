"""GPU renderer for mixer v2 (low-delay plan Task 4b, `perf.numerics.render: gpu`).

The CPU mixer (vaani.data.mixer.mix_v2) stays the reference and the fallback; this module splits its work so the CPU
keeps what defines the data and the GPU does the arithmetic:
  CPU  mix_v2_recipe: every random draw of mix_v2, in its order, and every audio read. It calls the CPU mixer's own
       draw helpers (mix_v2_ref_draw, fit_xfade, wind_pair, the RIR bank's sample) and draws the diffuse-field, point
       arrival and self-noise arrays exactly where mix_v2 draws them, so the random stream, and with it the paired-seed
       design, is unchanged. The output is a recipe: arrays and scalars, no rendering.
  GPU  render_recipe / render_batch: the recipe rendered in float64: RIR convolutions, the diffuse field (an STFT
       matched to SciPy's window and padding), the head transfer, the A-weighted and speech-active levels, the 60 Hz
       high-pass as a batched biquad, saturation and rails; plus the item's metadata.
The limiter and the reference ramp run afterwards on the CPU with the compiled kernel (one limiter kernel for every
path), as do the reference faults of DynamicMixDataset._finish.

Parity (tests/test_mixer_gpu.py): at least 1,000 recipes rendered here against mix_v2 on the same seeds, within 1e-5
of full scale with the same discrete metadata; then G1 is re-run on GPU renders. The renderer is adopted only if both
pass, and then for every arm, C0 included. Validation and test sets are pre-rendered and do not change.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from vaani.data import calib
from vaani.data.mixer import (C_SOUND, SILENT_DB, SR, fit_xfade, mix_v2_ref_draw, v2_params, wind_pair)

F64 = torch.float64


def _f32(x):
    """Round to float32 where the CPU mixer stores float32 (arithmetic between those points stays float64), so the
    renderer carries the reference's storage rounding: at a saturating level (e.g. 157 x full scale before the rails)
    one float32 step is already 1.5e-5."""
    return x.float().double()


# ---- recipe (CPU): the draws of mix_v2, in order ------------------------------------------------------------
def mix_v2_recipe(rng, speech, noises, impulse, impulse_onsets_s, bank, cfg, scene=None) -> dict:
    """The random draws and audio of mix_v2(rng, ...) with norm_gain None, without rendering. After this call `rng` is
    in the state mix_v2 would leave it in."""
    from vaani.data import scenes as _scenes
    p = v2_params(cfg)
    n = len(speech); xf = int(p["xfade_s"] * SR)
    if scene is None:
        scene = _scenes.sample_scene(rng, crop_s=n / SR)
    meta = {"mix_version": 2, "scene": scene["name"], "clean_bucket": False, "clipped": False, "ref_dropout": False,
            "impulse_peak_db": None, "impulse_onsets_s": [], "norm_gain": 1.0, "effort": scene["effort"],
            "speech_spl_db": float(scene["speech_spl"])}
    meta["lombard"] = bool(p["lombard"] and scene.get("lombard"))
    if scene.get("speech_lombard"):
        meta["speech_lombard"] = True
    rec = {"n": n, "p": p, "meta": meta, "speech": np.asarray(speech, np.float32),
           "tilt": meta["lombard"] and not scene.get("speech_lombard"), "speech_spl": float(scene["speech_spl"]),
           "scene_name": scene["name"]}
    mode, gain, delay, shadow = mix_v2_ref_draw(rng, p)
    meta["ref_mode"] = mode
    rec["ref"] = (mode, gain, delay, shadow)
    if p["path"] is not None:
        use_room = p["path"] == "room" and bank is not None
    else:
        use_room = bank is not None and scene["rir"] in ("room", "armoured") and rng.random() < cfg.p_room
    meta["path"] = "room" if use_room else "param"
    r = None
    if use_room:
        try:
            r = bank.sample(rng, armoured=scene["rir"] == "armoured")
        except TypeError:
            r = bank.sample(rng)
        rec["rir_speech"] = np.asarray(r["speech"])
    rec["use_room"] = use_room
    srcs = scene.get("sources") or []
    rng_rho = float(rng.uniform(*p["stereo_corr"]))
    comp, items = [], []
    for i, nz in enumerate(noises):
        spec = srcs[i] if i < len(srcs) else dict(role="bed", spl=srcs[0]["spl"] if srcs else 60.0, weighting="A")
        nz = fit_xfade(np.asarray(nz, np.float32), n, rng, xf)
        role = spec["role"]; ild = None
        if nz.ndim == 2 and role != "bed":
            nz = np.ascontiguousarray(nz[:, 0])
        it = {"role": role, "spl": float(spec["spl"]), "weighting": spec.get("weighting", "A"), "x": nz}
        if nz.ndim == 2:
            it["kind"] = "measured"
        elif mode == "stereo":
            it.update(kind="diffuse", gamma=rng_rho, model="spherical", W=rng.standard_normal(n))
        elif role == "bed":
            model = "cylindrical" if rng.random() < p["p_cylindrical"] else "spherical"
            it.update(kind="diffuse", gamma=None, model=model, W=rng.standard_normal(n))
        else:
            if role == "near":
                ild = float(rng.uniform(*p["near_ild_db"])) * (1 if rng.random() < p["near_pos_share"] else -1)
            else:
                ild = float(rng.uniform(-p["point_ild_db"], p["point_ild_db"]))
            if use_room:
                it.update(kind="room", rir=np.asarray(r["noise"][i % len(r["noise"])]), ild=ild)
            else:
                it.update(kind="point", ild=ild, tau=p["mic_spacing_m"] * np.cos(rng.uniform(0, np.pi)) / C_SOUND)
        items.append(it)
        comp.append({"role": role, "spl_db": float(spec["spl"]), "ild_db": ild, "tag": spec.get("tag")})
    meta["noise_sources"] = comp
    rec["noises"] = items
    wind_mps = float(scene.get("wind_mps") or 0.0)
    rec["wind"] = None
    if wind_mps > 0:
        w2, wm = wind_pair(rng, n, wind_mps, float(rng.uniform(*p["wind_ref_db"])), p["wind_ref_mps"],
                           p["windscreen_db"], p["wind_frame_s"])
        rec["wind"] = w2; meta.update(wm)
    meta["wind_mps"] = wind_mps
    if rng.random() < cfg.p_clean:
        meta["clean_bucket"] = True
    rec["impulse"] = None
    if impulse is not None and not meta["clean_bucket"]:
        ev = scene.get("event") or {}
        pk_spl = float(ev["peak_spl"]) if "peak_spl" in ev else float(scene["speech_spl"] + rng.uniform(*cfg.impulse_peak_db))
        start = int(rng.integers(0, max(1, n - len(impulse))))
        seg = np.asarray(impulse[: n - start], np.float32)
        imp = {"seg": seg, "start": start, "pk_spl": pk_spl}
        if use_room:
            imp.update(kind="room", rir=np.asarray(r["noise"][-1]))
        else:
            ild_i = float(rng.uniform(-p["far_ild_db"], p["far_ild_db"]))
            imp.update(kind="point", ild=ild_i, tau=p["mic_spacing_m"] * np.cos(rng.uniform(0, np.pi)) / C_SOUND)
        rec["impulse"] = imp
        meta["impulse_peak_db"] = pk_spl
        meta["impulse_onsets_s"] = [start / SR + o for o in impulse_onsets_s]
    rec["gains"] = rng.uniform(-p["mic_gain_db"], p["mic_gain_db"], 2) if p["front_end"] else np.zeros(2)
    rec["self_noise"] = None
    if p["front_end"]:
        if p["fe_hpf_order"] == "pre":
            if p["self_noise_db"] is not None:
                rec["self_noise"] = rng.standard_normal((2, n))
        elif p["fe_hpf_order"] == "post":
            rec["self_noise"] = rng.standard_normal((2, n))
        else:
            raise ValueError(f"fe_hpf_order must be pre or post, got {p['fe_hpf_order']!r}")
    return rec


# ---- torch float64 primitives -------------------------------------------------------------------------------
def _next_fast_len(n):
    from scipy.fft import next_fast_len
    return next_fast_len(n)


def fftconv_head(x, h, n):
    """fftconvolve(x, h)[:n] for 1-D x and h."""
    m = x.shape[-1] + h.shape[-1] - 1
    nf = _next_fast_len(m)
    return torch.fft.irfft(torch.fft.rfft(x, nf) * torch.fft.rfft(h, nf), nf)[..., :n]


def head_transfer_t(x, gain_db, delay_s, shadow_db=0.0, a=0.0875):
    al = 10 ** (shadow_db / 20); w0 = C_SOUND / a
    n = x.shape[-1]; m = _next_fast_len(n + 256)
    f = torch.fft.rfftfreq(m, 1 / SR, dtype=F64, device=x.device)
    w = 2 * math.pi * f
    H = 10 ** (gain_db / 20) * torch.exp(-1j * w * delay_s) * (1 + 1j * al * w / (2 * w0)) / (1 + 1j * w / (2 * w0))
    return torch.fft.irfft(torch.fft.rfft(x, m) * H, m)[..., :n]


def rms_db_t(x):
    return 10 * torch.log10((x * x).mean() + 1e-30)


def active_rms_db_t(x, frame=320, thresh_db=-30.0):
    f = x[: x.shape[-1] // frame * frame].reshape(-1, frame)
    e = (f * f).mean(1) + 1e-30
    keep = e > e.max() * 10 ** (thresh_db / 10)
    return 10 * torch.log10(e[keep].mean() if bool(keep.any()) else e.mean())


def a_weighted_rms_db_t(x):
    n = x.shape[-1]
    X = torch.fft.rfft(x)
    f = torch.fft.rfftfreq(n, 1 / SR, dtype=F64, device=x.device)
    f2 = torch.clamp(f, min=1e-3) ** 2
    ra = (12194.0 ** 2 * f2 ** 2) / ((f2 + 20.6 ** 2) * torch.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2)) * (f2 + 12194.0 ** 2))
    w = 10 ** ((20 * torch.log10(ra) + 2.0) / 20)
    p = ((X * w).abs() ** 2).sum() * 2 / n ** 2
    return 10 * torch.log10(p + 1e-30)


def speech_active_power_t(x, frame=320, thresh_db=-30.0):
    f = x[: x.shape[-1] // frame * frame].reshape(-1, frame)
    e = (f * f).mean(1) + 1e-12
    keep = e > e.max() * 10 ** (thresh_db / 10)
    return e[keep].mean() if bool(keep.any()) else e.mean()


def lombard_tilt_t(x, alpha_db=calib.LOMBARD_ALPHA_DB):
    n = x.shape[-1]
    X = torch.fft.rfft(x)
    f = torch.fft.rfftfreq(n, 1 / SR, dtype=F64, device=x.device)
    s = (f / 1000.0) ** 4 / (1 + (f / 1000.0) ** 4)
    y = torch.fft.irfft(X * 10 ** (alpha_db * s / 20), n)
    return y * torch.sqrt((x * x).mean() / ((y * y).mean() + 1e-30))


_HANN: dict = {}


def _hann(nfft, device):
    k = (nfft, str(device))
    if k not in _HANN:
        _HANN[k] = torch.hann_window(nfft, periodic=True, dtype=F64, device=device)
    return _HANN[k]


def _scipy_stft(x, nfft, hop):
    """scipy.signal.stft(x, nperseg=nfft, noverlap=nfft-hop, boundary='zeros', padded=True) up to its constant
    1/sum(win) scaling (the diffuse bed is invariant to it): (F, frames) complex."""
    w = _hann(nfft, x.device)
    xp = torch.nn.functional.pad(x, (nfft // 2, nfft // 2))
    nadd = (-(xp.shape[-1] - nfft) % hop) % nfft
    xp = torch.nn.functional.pad(xp, (0, nadd))
    fr = xp.unfold(-1, nfft, hop) * w
    return torch.fft.rfft(fr, dim=-1).T


def _scipy_istft(X, nfft, hop, n):
    """scipy.signal.istft(X, nperseg=nfft, noverlap=nfft-hop, boundary=True)[1][:n], same scaling convention."""
    w = _hann(nfft, X.device)
    fr = torch.fft.irfft(X.T, n=nfft, dim=-1) * w                       # (frames, nfft)
    t = fr.shape[0]
    total = nfft + (t - 1) * hop
    y = torch.nn.functional.fold(fr.T[None], (1, total), (1, nfft), stride=(1, hop)).reshape(total)
    env = torch.nn.functional.fold((w * w)[:, None].expand(nfft, t)[None], (1, total), (1, nfft),
                                   stride=(1, hop)).reshape(total)
    y = torch.where(env > 1e-10, y / torch.where(env > 1e-10, env, torch.ones_like(env)), y)
    y = y[nfft // 2: total - nfft // 2]
    return y[:n]


def diffuse_pair_t(x, W_noise, gamma=None, d=0.12, model="spherical", nfft=512, hop=128):
    n = x.shape[-1]
    X1 = _scipy_stft(x, nfft, hop)
    f = torch.fft.rfftfreq(nfft, 1.0, dtype=F64, device=x.device) * SR
    if gamma is None:
        z = 2 * math.pi * f * d / C_SOUND
        if model == "spherical":
            g = torch.where(z == 0, torch.ones_like(z), torch.sin(z) / torch.where(z == 0, torch.ones_like(z), z))
        else:
            g = torch.special.bessel_j0(z)
    else:
        g = torch.full_like(f, float(gamma))
    p2 = (X1.abs() ** 2)[None, None]
    sm = torch.nn.functional.avg_pool2d(torch.nn.functional.pad(p2, (1, 1, 1, 1), mode="replicate"), 3, 1)[0, 0]
    env = torch.sqrt(torch.clamp(sm, min=0.0))
    W = _scipy_stft(W_noise, nfft, hop)
    W = W / torch.sqrt((W.abs() ** 2).mean() + 1e-30)
    X2 = g[:, None] * X1 + torch.sqrt(torch.clamp(1 - g ** 2, min=0.0))[:, None] * env * W
    return torch.stack([x, _scipy_istft(X2, nfft, hop, n)])


def _conv2_t(x, h2, n):
    return torch.stack([fftconv_head(x, h2[m], n) for m in range(2)])


def _level_t(x, weighting):
    return {"active": active_rms_db_t, "rms": rms_db_t, "A": a_weighted_rms_db_t}[weighting](x)


def _to_spl_t(pair, spl, weighting):
    lvl = _level_t(pair[0], weighting)
    if float(lvl) < SILENT_DB:
        return torch.zeros_like(pair)
    return pair * 10 ** ((spl - calib.SPL_TO_FLOAT_RMS_DB - lvl) / 20)


_HPF: dict = {}


def hpf_t(x, hz=calib.HPF_HZ):
    """calib.hpf (2nd-order Butterworth high-pass, sosfilt) as a batched biquad on the device."""
    from torchaudio.functional import lfilter
    k = (float(hz), str(x.device))
    if k not in _HPF:
        from scipy.signal import butter
        sos = butter(2, hz, "highpass", fs=SR, output="sos")[0]
        _HPF[k] = (torch.as_tensor(sos[:3], dtype=F64, device=x.device), torch.as_tensor(sos[3:], dtype=F64, device=x.device))
    b, a = _HPF[k]
    shp = x.shape
    return lfilter(x.reshape(-1, shp[-1]), a, b, clamp=False).reshape(shp)


def _softsat_t(x, knee):
    a = x.abs()
    return torch.where(a > knee, torch.sign(x) * (knee + (1 - knee) * torch.tanh((a - knee) / (1 - knee))), x)


def _level_flags_t(x2, fs_off):
    a = x2.abs()
    n = x2.shape[-1] // calib.AOP_FRAME * calib.AOP_FRAME
    fr = (x2[:, :n].reshape(x2.shape[0], -1, calib.AOP_FRAME) ** 2).mean(-1) if n else torch.zeros(1, 1, dtype=F64)
    pk = float(a.max()) if a.numel() else 0.0
    return {"past_knee": bool(pk > calib.soft_knee()),
            "past_aop": bool((10 * torch.log10(fr + 1e-30) + calib.SPL_TO_FLOAT_RMS_DB > calib.AOP_DB_SPL).any()),
            "past_rails": bool(pk >= 1.0),
            "peak_db_spl": float(20 * np.log10(pk + 1e-12) + calib.SPL_TO_FLOAT_RMS_DB + fs_off)}


# ---- render -------------------------------------------------------------------------------------------------
def render_recipe(rec: dict, device="cpu"):
    """One recipe -> (mix (2, n) float32, clean (n,) float32, meta), as mix_v2 returns them."""
    dev = torch.device(device)
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=F64, device=dev)
    p, n, meta = rec["p"], rec["n"], dict(rec["meta"])
    meta["noise_sources"] = [dict(c) for c in rec["meta"]["noise_sources"]]
    speech = t(rec["speech"])
    if rec["tilt"]:
        speech = lombard_tilt_t(speech)
    mode, gain, delay, shadow = rec["ref"]
    if rec["use_room"]:
        rs = t(rec["rir_speech"])
        h_s = rs / (rs[0].abs().max() + 1e-9)
        s_p = _f32(fftconv_head(speech, h_s[0], n))
        s_r_room = _f32(fftconv_head(speech, h_s[1], n))
    else:
        s_p, s_r_room = speech.clone(), None
    lvl_s = active_rms_db_t(s_p)
    g0 = 0.0 if float(lvl_s) < SILENT_DB else 10 ** ((rec["speech_spl"] - calib.SPL_TO_FLOAT_RMS_DB - lvl_s) / 20)
    s_p = _f32(s_p * g0)
    if s_r_room is not None:
        s_r_room = _f32(s_r_room * g0)
    d_mic = p["mic_spacing_m"]
    if mode in ("physical", "low_ild"):
        if s_r_room is not None:
            s_r = _f32(head_transfer_t(s_r_room, 0.0, 0.0, shadow, p["head_radius_m"]))
        else:
            s_r = _f32(head_transfer_t(s_p, 0.0, delay, shadow, p["head_radius_m"]))
        s_r = _f32(s_r * _f32(10 ** ((gain - (rms_db_t(s_r) - rms_db_t(s_p))) / 20)))
    elif mode == "stereo":
        s_r = _f32(head_transfer_t(s_p, gain, delay))
    else:
        s_r = s_p.clone()
    s2 = torch.stack([s_p, s_r])
    meta["ref_speech_gain_db"] = float(rms_db_t(s_r) - rms_db_t(s_p))

    noise2 = torch.zeros(2, n, dtype=F64, device=dev)
    for it in rec["noises"]:
        x = t(it["x"])
        k = it["kind"]
        if k == "measured":
            pair = x.T
        elif k == "diffuse":
            pair = _f32(diffuse_pair_t(x, t(it["W"]), gamma=it["gamma"], d=d_mic, model=it["model"],
                                       nfft=p["stft_n"], hop=p["stft_hop"]))
        elif k == "room":
            h = t(it["rir"]); h = h / (h[0].abs().max() + 1e-9)
            pair = _f32(_conv2_t(x, h, n))
            pair = torch.stack([pair[0], _f32(pair[1] * _f32(torch.tensor(10 ** (-it["ild"] / 20))))])
        else:
            pair = torch.stack([x, _f32(head_transfer_t(x, -it["ild"], it["tau"]))])
        noise2 = _f32(noise2 + _f32(_to_spl_t(pair, it["spl"], it["weighting"])))
    if rec["wind"] is not None:
        noise2 = _f32(noise2 + t(rec["wind"]))
    if meta["clean_bucket"]:
        noise2 = torch.zeros_like(noise2)
    acoustic = _f32(s2 + noise2)
    imp = rec["impulse"]
    if imp is not None:
        seg = t(imp["seg"])
        if imp["kind"] == "room":
            h = t(imp["rir"]); h = h / (h[0].abs().max() + 1e-9)
            ip = _f32(_conv2_t(seg, h, seg.shape[-1]))
        else:
            ip = torch.stack([seg, _f32(head_transfer_t(seg, -imp["ild"], imp["tau"]))])
        ip = _f32(ip * _f32(10 ** ((imp["pk_spl"] - calib.SPL_TO_FLOAT_RMS_DB) / 20) / (ip[0].abs().max() + 1e-12)))
        imp2 = torch.zeros(2, n, dtype=F64, device=dev)
        imp2[:, imp["start"]:imp["start"] + seg.shape[-1]] = ip
        acoustic = _f32(s2 + noise2 + imp2)
    fs_off = float(p["mic_fs_spl_db"]) - 120.0
    if fs_off:
        acoustic = acoustic * 10 ** (-fs_off / 20); s_p = s_p * 10 ** (-fs_off / 20)
    gains = np.asarray(rec["gains"], float)
    gl = t(10 ** (gains / 20))
    if p["front_end"]:
        hz = float(p["fe_hpf_hz"])
        gl = _f32(gl)
        lin = _f32(hpf_t(_f32(acoustic * gl[:, None]), hz))
        clean = _f32(hpf_t(_f32(s_p * gl[0])[None], hz)[0])
        if p["fe_hpf_order"] == "pre":
            sat_in = lin
            out, fe = _nonlinear_t(lin, p["fe_curve"])
            if rec["self_noise"] is not None:
                out = _f32(out + _f32(t(rec["self_noise"]) * 10 ** (p["self_noise_db"] / 20)))
        else:
            sat_in = _f32(acoustic * gl[:, None])
            out, fe = _nonlinear_t(sat_in, p["fe_curve"])
            out = _f32(_f32(hpf_t(out, hz)) + _f32(t(rec["self_noise"]) * 10 ** (p["self_noise_db"] / 20)))
    else:
        lin = acoustic; clean = s_p.clone(); sat_in = lin
        out, fe = lin.clone(), {"clip_frac": 0.0, "saturated": False}
    if mode == "mono":
        out = torch.stack([out[0], out[0]])
    meta["mic_gain_db"] = [float(g) for g in gains]
    meta["clip_frac"] = fe["clip_frac"]; meta["clipped"] = fe["clip_frac"] > 0; meta["overloaded"] = fe["saturated"]
    meta.update(_level_flags_t(sat_in, fs_off))
    ps = speech_active_power_t(clean)
    pre = lin[0] - clean
    meta["snr_db"] = np.inf if meta["clean_bucket"] else float(10 * torch.log10(ps / ((pre ** 2).mean() + 1e-20)))
    resid = out[0] - clean
    meta["snr_achieved_db"] = float(10 * torch.log10(speech_active_power_t(clean) / ((resid ** 2).mean() + 1e-20)))
    meta["noise_class"] = "clean" if meta["clean_bucket"] else rec["scene_name"]
    return out.float().cpu().numpy(), clean.float().cpu().numpy(), meta


def _nonlinear_t(y, curve):
    knee = calib.soft_knee()
    a = y.abs()
    fe = {"clip_frac": float((a >= 1.0).double().mean()), "saturated": False}
    mx = float(a.max()) if a.numel() else 0.0
    if curve != "knee105":
        fe["saturated"] = mx > knee
        xs = calib.tanh_scale()
        y = torch.clamp(_f32(xs * torch.tanh(y / xs)), -1.0, 1.0)
    elif mx > knee:
        y = torch.clamp(_f32(_softsat_t(y, knee)), -1.0, 1.0); fe["saturated"] = True
    return y, fe


def render_batch(recipes, device="cuda"):
    """A batch of recipes rendered on `device` -> [(mix, clean, meta)] on the host."""
    return [render_recipe(r, device) for r in recipes]


# ---- data path ------------------------------------------------------------------------------------------------
class RecipeDataset(torch.utils.data.Dataset):
    """DynamicMixDataset's items as recipes (for DataLoader workers): (epoch, i, recipe)."""

    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return self.ds.recipe(idx)


def collate_recipes(batch):
    return batch


def render_and_finish(ds, recipes, device):
    """[(epoch, i, recipe)] -> the collated batch __getitem__ + collate would give (GPU render, CPU finish)."""
    from vaani.data.dataset import collate
    items = []
    for (epoch, i, rec), (mix, clean, meta) in zip(recipes, render_batch([r for _, _, r in recipes], device)):
        items.append(ds.finish(epoch, i, mix, clean, meta))
    return collate(items)
