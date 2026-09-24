"""Synthetic impulsive noise for training: fast onsets, varied decay, no corpus dependence."""
import numpy as np

KINDS = ("burst", "click_train", "gated_noise")  # the eval-set draw; frozen (hashes). r3 adds "blast" via MixConfig.impulse_kinds
EXTRA_KINDS = ("blast",)  # Friedlander blast wave (vaani/data/blast.py): the only kind with a gunshot-like crest


def _decay(rng, sr, tau_s):
    # Exponential-envelope colored noise; 1-pole lowpass gives a "thud"
    n = int(sr * min(2.0, tau_s * 6))
    t = np.arange(n) / sr
    x = rng.standard_normal(n)
    a = rng.uniform(0.6, 0.95)
    for i in range(1, n):
        x[i] += a * x[i - 1]
    return (x * np.exp(-t / tau_s)).astype(np.float32)


def generate(rng: np.random.Generator, sr: int = 16000, kind: str | None = None, blast_kind: str | None = None,
             physics: str = "v1", **scene):
    """blast_kind pins the blast sub-kind ("small_arms" | "artillery"); None draws it, as training does.
    physics="v2" (blast only) uses blast.blast_v2, or blast.burst for blast_kind "burst"; scene passes their parameters.
    v2 meta carries peak_spl_db at the mic for the caller to calibrate to dBFS; the waveform is still peak-normalised."""
    kind = kind or rng.choice(KINDS)
    if physics == "v2":
        if kind != "blast":
            raise ValueError("physics='v2' applies to kind='blast' only")
        return _generate_v2(rng, sr, blast_kind, scene)
    if physics != "v1" or scene:
        raise ValueError(f"physics={physics!r} with {sorted(scene)}: only v2 takes scene parameters")
    if kind == "blast":
        from vaani.data import blast as _blast
        x, m = _blast.blast(rng, sr, kind=blast_kind)
        pre = int(rng.uniform(0.05, 0.3) * sr)  # same silent lead-in as burst so the onset is inside the clip
        x = np.concatenate([np.zeros(pre, np.float32), x]); onsets = [pre / sr]
    elif kind == "burst":
        x = _decay(rng, sr, rng.uniform(0.02, 0.25))
        pre = int(rng.uniform(0.05, 0.3) * sr)
        x = np.concatenate([np.zeros(pre, np.float32), x])
        onsets = [pre / sr]
    elif kind == "click_train":
        n_clicks = int(rng.integers(3, 12))
        gap = rng.uniform(0.04, 0.15)
        pieces, onsets, pos = [], [], 0.0
        for _ in range(n_clicks):
            c = _decay(rng, sr, rng.uniform(0.003, 0.02))
            g = np.zeros(int(gap * sr), np.float32)
            onsets.append(pos); pos += (len(c) + len(g)) / sr
            pieces += [c, g]
        x = np.concatenate(pieces)
    else:  # gated_noise: wideband noise switched on/off abruptly
        dur = rng.uniform(0.3, 1.5)
        x = rng.standard_normal(int(dur * sr)).astype(np.float32)
        on, off = int(0.1 * sr), int(rng.uniform(0.3, 0.9) * dur * sr)
        x[:on] = 0; x[off:] = 0
        onsets = [on / sr]
    n = int(np.clip(len(x), 0.2 * sr, 2.0 * sr))
    x = np.pad(x, (0, max(0, n - len(x))))[:n]
    x = x / (np.abs(x).max() + 1e-9)
    meta = {"kind": str(kind), "onsets_s": [float(o) for o in onsets]}
    if kind == "blast":
        meta.update(blast_kind=m["kind"], distance=m["distance"])
    return x.astype(np.float32), meta


V2_MAX_S = 3.5   # a 30-round burst at 650 rpm lasts 2.6 s; the mixer crops to the clip


def _generate_v2(rng, sr, blast_kind, scene):
    from vaani.data import blast as _blast
    if blast_kind == "burst":
        x, m = _blast.burst(rng, sr, **scene)
    else:
        x, m = _blast.blast_v2(rng, sr, kind=blast_kind, **scene)
    pre = int(rng.uniform(0.05, 0.3) * sr)   # same silent lead-in as v1 so the onset is inside the clip
    x = np.concatenate([np.zeros(pre), x])[:int(V2_MAX_S * sr)]
    x = (x / (np.abs(x).max() + 1e-30)).astype(np.float32)
    keep = ("physics", "distance_m", "peak_spl_db", "source_spl_1m", "charge_kg", "n_rounds", "rpm", "ballistic")
    meta = {"kind": "blast", "blast_kind": m["kind"], "distance": None,
            "onsets_s": [pre / sr + o for o in m["onsets_s"] if (pre / sr + o) * sr < len(x)],
            **{k: m[k] for k in keep if k in m}}
    return x, meta


def detect_onsets(x: np.ndarray, sr: int = 16000, frame_ms: float = 5.0, rise_db: float = 12.0,
                  floor_ms: float = 100.0, min_gap_ms: float = 30.0) -> list[float]:
    """Onset times for a recorded impulse clip, which carries no generator metadata.
    An onset is a frame that jumps rise_db above the median energy of the preceding floor_ms;
    the loudest sample is returned when nothing stands out so callers never get an empty list."""
    fl = max(1, int(sr * frame_ms / 1000)); nf = len(x) // fl
    if nf == 0:
        return [0.0]
    e = 10 * np.log10((x[: nf * fl].reshape(nf, fl) ** 2).mean(axis=1) + 1e-12)
    look = max(1, int(floor_ms / frame_ms)); gap = max(1, int(min_gap_ms / frame_ms))
    # assume silence before the clip so a click on the very first frame still counts as an onset
    e = np.concatenate([np.full(look, e.min()), e])
    onsets, last = [], -gap
    for k in range(look, len(e)):
        floor = np.median(e[k - look:k])
        # crossing, not level: a decaying tail that stays above the floor is one event, not many
        if e[k] - floor >= rise_db and e[k - 1] - floor < rise_db and k - last >= gap:
            onsets.append((k - look) * fl / sr); last = k
    return onsets or [float(int(np.argmax(np.abs(x))) / sr)]
