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
