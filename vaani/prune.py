"""Global magnitude pruning of a trained checkpoint, as a measured answer rather than a claim.

Clause 14c of SIH26052 names pruning. The expected finding here is a considered "no", and the
point of this module is to make that "no" a number instead of a silence: prune at a sweep of
sparsities, evaluate each on the frozen split, and report where quality falls off.

Why the prior is negative, stated before the measurement so the result cannot be retrofitted:

- The cascade has 52,747 parameters, of which only about 22,000 are learned weights in prunable
  layer types (see `inventory`). Pruning normally pays on over-parameterised models; this one was
  already designed against a 16 ms/hop budget.
- The layers are narrow -- 16 channels, GRU hidden 16. Unstructured sparsity at these widths does
  not become dense-matrix speedup without a sparse kernel that the ONNX Runtime CPU path does not
  provide, so an unstructured win is a file-size win at best.
- Structured pruning, which *would* shrink the dense matrices, has almost nothing to remove: a
  16-channel layer has 16 channels to choose from, and the ONNX cache shapes in `deploy/CONTRACT.md`
  are keyed to those widths, so any structured change is a re-export and a contract change.

Two tensor groups are deliberately excluded from every sparsity level:

`erb_fc` / `ierb_fc` are the ERB band-split analysis and synthesis matrices. They are stored as
parameters with `requires_grad=False` (`vaani/models/gtcrn.py`), but they are a fixed
signal-processing transform, not learned capacity -- and they are 24,576 of the 46,528 weight
elements in prunable module types. Pruning them would corrupt the band split *and* flatter the
sparsity figure by letting half the budget fall on weights that were never trained. `requires_grad`
cannot be the discriminator here, because refiner training freezes the whole first stage.

    uv run python -m vaani.prune results_r2/runs/vaani_tier46_refiner/best.pt \\
        --sparsity 0.1 0.2 0.3 0.4 0.5 --out-dir runs/prune --report-json results_r2/optim/prune_sparsity.json
"""
import copy
import json
from pathlib import Path

import torch
import torch.nn.utils.prune as tprune

from vaani.models import cascade
from vaani.models.vaani_net import VaaniNet
from vaani.train import build_model


# Fixed transforms, never learned capacity. See the module docstring.
EXCLUDE_SUBSTRINGS = ("erb_fc", "ierb_fc")
PRUNABLE_TYPES = (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Linear, torch.nn.GRU)


def load_checkpoint(ckpt_path):
    """The checkpoint's batch module, alongside the raw checkpoint dict so it can be rewritten."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    cfg = ck["config"]
    if cfg["model"] == cascade.MODEL_NAME:
        m = cascade.FrozenCascade.from_config(cfg)
    elif cfg["model"] == "vaani":
        m = VaaniNet(**cfg.get("model_cfg", {}))
    else:
        m = build_model(cfg["model"], model_cfg=cfg.get("model_cfg"))
    m.load_state_dict(ck["model"])
    return m.eval(), ck


def prunable_tensors(model, scope="all"):
    """(module, param_name, qualified_name) for every weight magnitude pruning may touch.

    scope selects a stage of a cascade: "first" (the frozen 50 K first stage), "refiner" (the
    2.5 K residual head), or "all". Biases and normalisation parameters are never included --
    zeroing a bias shifts the whole output rather than removing a connection.
    """
    out = []
    for name, mod in model.named_modules():
        if not isinstance(mod, PRUNABLE_TYPES):
            continue
        if scope == "first" and not name.startswith("first."):
            continue
        if scope == "refiner" and not name.startswith("refiner."):
            continue
        if any(s in name for s in EXCLUDE_SUBSTRINGS):
            continue
        for pname, p in mod.named_parameters(recurse=False):
            if pname.startswith("weight") and p.dim() >= 2:  # GRU contributes weight_ih_l0/weight_hh_l0 and reverse
                out.append((mod, pname, f"{name}.{pname}"))
    return out


def inventory(model, scope="all"):
    """What the sparsity budget is actually computed over, so the denominator is never implicit."""
    tensors = prunable_tensors(model, scope)
    return {"prunable_tensors": len(tensors),
            "prunable_weight_elements": int(sum(getattr(m, n).numel() for m, n, _ in tensors)),
            "total_parameters": int(sum(p.numel() for p in model.parameters())),
            "excluded_substrings": list(EXCLUDE_SUBSTRINGS),
            "tensors": [q for _, _, q in tensors]}


def _refresh_rnn(model):
    """Rebuild GRU flat-weight views after pruning rebinds the underlying Parameters.

    `prune.remove` assigns a fresh Parameter onto the module, but an RNN caches `_flat_weights`
    as a list of references captured at construction. Without this the module keeps running on
    the pre-prune tensors and the sweep silently measures nothing.
    """
    for mod in model.modules():
        if isinstance(mod, torch.nn.RNNBase):
            mod._init_flat_weights()


def global_magnitude_prune(model, sparsity, scope="all"):
    """Zero the globally smallest-magnitude `sparsity` fraction of prunable weights, in place.

    Global rather than per-layer: a per-layer budget forces the same fraction out of a 32-element
    GRU gate matrix as out of the refiner's 2,304-element 3x3 convolution, which is a statement
    about layer shapes rather than about which weights matter.
    """
    if not 0.0 <= sparsity < 1.0:
        raise ValueError(f"sparsity must be in [0, 1), got {sparsity}")
    tensors = prunable_tensors(model, scope)
    if not tensors:
        raise ValueError(f"no prunable tensors for scope {scope!r}")
    # Weights that were already exactly zero must not be counted as pruning's work: a freshly
    # initialised model has some, and crediting them would overstate what the sweep removed.
    pre_zero = {q: (getattr(m, n) == 0) for m, n, q in tensors}
    if sparsity > 0:
        tprune.global_unstructured([(m, n) for m, n, _ in tensors], tprune.L1Unstructured, amount=sparsity)
        for m, n, _ in tensors:
            tprune.remove(m, n)   # bake the mask into the weight: the checkpoint must be a plain state_dict
        _refresh_rnn(model)
    per_tensor, zeros, new, pre, total = {}, 0, 0, 0, 0
    for m, n, q in tensors:
        w = getattr(m, n)
        z = int((w == 0).sum())
        nz = int(((w == 0) & ~pre_zero[q]).sum())
        per_tensor[q] = {"elements": w.numel(), "zeros": z, "newly_zeroed": nz, "sparsity": z / w.numel()}
        zeros += z; new += nz; pre += z - nz; total += w.numel()
    # achieved_sparsity describes the weights as they now are, which is what a deployed graph sees;
    # newly_zeroed_weights is what this call did. They differ whenever the model already had zeros.
    return {"requested_sparsity": sparsity, "scope": scope,
            "achieved_sparsity": zeros / total, "zeroed_weights": zeros,
            "newly_zeroed_weights": new, "pre_existing_zeros": pre, "prunable_weight_elements": total,
            # Against every parameter, including biases, norms and the excluded ERB matrices: this is
            # the number that matters for a size claim, and it is always lower than the headline.
            "sparsity_over_all_parameters": zeros / sum(p.numel() for p in model.parameters()),
            "per_tensor": per_tensor}


def write_pruned(ckpt_path, out_path, sparsity, scope="all"):
    """A pruned checkpoint that loads exactly like the original -- same config, plain state_dict."""
    model, ck = load_checkpoint(ckpt_path)
    stats = global_magnitude_prune(model, sparsity, scope)
    out = copy.copy(ck)
    out["model"] = model.state_dict()
    out["pruning"] = {k: v for k, v in stats.items() if k != "per_tensor"}
    out["pruning"]["source_checkpoint"] = Path(ckpt_path).as_posix()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)
    stats["checkpoint"] = out_path.as_posix()
    return stats


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("ckpt")
    ap.add_argument("--sparsity", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.4, 0.5])
    ap.add_argument("--scope", default="all", choices=["all", "first", "refiner"])
    ap.add_argument("--out-dir", default="runs/prune")
    ap.add_argument("--report-json")
    a = ap.parse_args()
    model, _ = load_checkpoint(a.ckpt)
    report = {"source_checkpoint": Path(a.ckpt).as_posix(), "scope": a.scope,
              "inventory": inventory(model, a.scope), "levels": []}
    for s in a.sparsity:
        name = f"p{int(round(s * 100)):02d}"
        stats = write_pruned(a.ckpt, Path(a.out_dir) / name / "best.pt", s, a.scope)
        report["levels"].append({k: v for k, v in stats.items() if k != "per_tensor"})
        print(f"{name}: achieved {stats['achieved_sparsity']:.4f} over {stats['prunable_weight_elements']} "
              f"prunable weights ({stats['sparsity_over_all_parameters']:.4f} of all parameters) -> {stats['checkpoint']}")
    if a.report_json:
        Path(a.report_json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report_json).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
