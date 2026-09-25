"""Pure training controls: test budgets/resume without launching a trainer."""
import math
import hashlib
from pathlib import Path


def verify_checkpoint_hash(path, expected=None):
    if expected is not None:
        if path is None or hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError("checkpoint hash differs from the declared training recipe")


def make_schedule_config(epochs, steps_per_epoch, warmup_steps=500, max_steps=None):
    total = max_steps if max_steps is not None else epochs * steps_per_epoch
    if total < 1 or warmup_steps < 0 or steps_per_epoch < 1:
        raise ValueError("schedule needs positive steps and nonnegative warmup")
    return dict(kind="cosine", total_steps=int(total), warmup_steps=int(warmup_steps))


def cosine_lr_multiplier(step, warmup_steps, total_steps):
    """Legacy warmup x full-budget cosine, reaching exactly zero at the budget."""
    if total_steps <= 0 or warmup_steps < 0 or step < 0:
        raise ValueError("invalid schedule step/budget")
    warmup = min(1., (step + 1) / warmup_steps) if warmup_steps else 1.
    return float(warmup * .5 * (1 + math.cos(math.pi * min(step, total_steps) / total_steps)))


def validate_resume_schedule(saved, requested):
    if saved != requested:
        raise RuntimeError("Cannot resume a changed cosine schedule; use a new run name and init_from for a declared restart")


def should_stop_for_patience(history, patience=None, min_delta=0.):
    """Stop after patience consecutive finite-validation failures to improve.

    This is a resource-control rule, not evidence that the model is converged.
    Replay the whole saved history to preserve behavior across a resume.
    """
    if patience is None:
        return False
    if not isinstance(patience, int) or patience < 1 or min_delta < 0:
        raise ValueError("patience must be positive and min_delta nonnegative")
    best, stale = -math.inf, 0
    for row in history:
        value = row["val_stoi"]
        if math.isfinite(value) and value > best + min_delta:
            best, stale = value, 0
        else:
            stale += 1
    return stale >= patience


# --- r8 checkpoint selection (plan 11.2 / 11.6); the final metric is Rachit's call (plan 11.9 Q7) ---
COMPOSITE_TARGETS = dict(snr=15.0, stoi=0.85, pesq=2.5)   # README:21 PS targets


def _mean(xs):
    xs = [x for x in xs if x is not None and math.isfinite(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def composite_summary(rows, targets=None, ild_max_loss=0.06, margin_db=1.0, base_d_snr=None, include_clean=False):
    """Per-(clip, condition) rows -> the composite val metric.
    rows: dicts with cond, clean_item, snr_in, snr_out, stoi, pesq, speech_loss (None where not scored).
    pass_rate: share of nominal (present) clips meeting all three targets; clean clips excluded unless include_clean.
    Hard filters (G3): mean speech loss <= ild_max_loss at every ILD step; on mono-duplicated and web-stereo inputs
    mean dSNR >= base_d_snr - margin_db (base = gtcrn_pretrained on the same clips; skipped when base is None)."""
    t = {**COMPOSITE_TARGETS, **(targets or {})}
    pres = [r for r in rows if r["cond"] == "present" and (include_clean or not r["clean_item"])]
    ok = [r["snr_out"] > t["snr"] and r["stoi"] > t["stoi"] and r["pesq"] > t["pesq"] for r in pres]
    ild = {}
    for r in rows:
        if r["cond"].startswith("ild_"):
            ild.setdefault(r["cond"], []).append(r["speech_loss"])
    ild = {k: _mean(v) for k, v in ild.items()}
    d = {c: _mean([r["snr_out"] - r["snr_in"] for r in rows if r["cond"] == c and not r["clean_item"]])
         for c in ("mono", "web_stereo")}
    ild_max = max(ild.values()) if ild else float("nan")
    filt = {"ild": bool(ild) and ild_max <= ild_max_loss}
    for c in ("mono", "web_stereo"):
        filt[c] = True if base_d_snr is None else bool(d[c] >= base_d_snr - margin_db)
    return dict(pass_rate=sum(ok) / len(ok) if ok else 0.0, n_clips=len(pres),
                stoi=_mean([r["stoi"] for r in pres]), snr_out=_mean([r["snr_out"] for r in pres]),
                pesq=_mean([r["pesq"] for r in pres]), ild_loss=ild, ild_loss_max=ild_max,
                d_snr_mono=d["mono"], d_snr_web=d["web_stereo"], base_d_snr=base_d_snr,
                filters=filt, passes=all(filt.values()))


def composite_key(s):
    """Filter-passing checkpoints beat failing ones; then pass rate; then nominal STOI (a total order, so best.pt
    always exists even before any checkpoint clears the filters)."""
    stoi = s["stoi"] if math.isfinite(s["stoi"]) else -1.0
    return (int(s["passes"]), float(s["pass_rate"]), float(stoi))


def ema_decay(decay, step, warmup=True):
    """EMA decay with the usual (1+s)/(10+s) ramp, so early shadows are not dominated by the random init."""
    return min(decay, (1 + step) / (10 + step)) if warmup else decay
