"""Compare enhancement systems with reference-informed oracle diagnostics.

The ERB-projected row is deliberately a *diagnostic*, not an architecture ceiling:
the trained decoder, its temporal state, and any later model revision can change what
is reachable. The unrestricted complex mask is the useful reconstruction sanity
check; it is allowed to reproduce the reference exactly.

By default this scores every item in ``data/eval_r2/val``. Test data is protected
behind ``--allow-test`` so a convenient diagnostic command cannot silently tune on it.
"""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from vaani import metrics
from vaani.data.dataset import RenderedDataset
from vaani.dsp import stft as S
from vaani.eval import enhance_fn
from vaani.models.gtcrn import ERB
from vaani.losses import absolute_snr


METRIC_NAMES = ("snr_db", "stoi", "pesq_wb", "loss_snr_db", "clamp_at_20", "clamp_at_30")


def to_c(x):
    """Use the project's STFT so every oracle has the model's reconstruction limits."""
    a = S.stft(torch.from_numpy(np.asarray(x, np.float32))[None])[0].numpy()
    return a[..., 0] + 1j * a[..., 1]


def from_c(c, length):
    t = torch.from_numpy(np.stack([c.real, c.imag], -1).astype(np.float32))[None]
    return S.istft(t, length=length)[0].numpy()


def safe_complex_ratio(numerator, denominator, eps=1e-10):
    """Finite complex division, including silent STFT bins, without biasing others."""
    power = np.abs(denominator) ** 2
    return numerator * np.conj(denominator) / np.maximum(power, eps)


def erb_projector():
    """Return the linear ERB-mask subspace projection used only for comparison."""
    erb = ERB(65, 64).eval()
    basis = erb.ierb_fc.weight.detach().numpy()
    return basis @ np.linalg.pinv(basis)


def erb_project_mask(mask, projector):
    """Project high bins only; low bins bypass the ERB encoder in the real model."""
    projected = np.asarray(mask).copy()
    if projected.shape[0] > 65:
        projected[65:] = projector @ projected[65:]
    return projected


def oracle_spectra(mix_spec, clean_spec, system_spec=None, projector=None):
    """Build reference-informed spectra. This pure function is tested on zeros.

    ``oracle_iam`` preserves noisy phase; ``oracle_phase_with_system_magnitude``
    holds the supplied system's amplitude fixed while replacing its phase only.
    """
    noise_spec = mix_spec - clean_spec
    irm = np.sqrt(np.abs(clean_spec) ** 2 /
                  (np.abs(clean_spec) ** 2 + np.abs(noise_spec) ** 2 + 1e-10))
    iam = np.abs(clean_spec) / np.maximum(np.abs(mix_spec), 1e-10)
    ideal_complex = safe_complex_ratio(clean_spec, mix_spec)
    out = {
        "raw": mix_spec,
        "oracle_irm": irm * mix_spec,
        "oracle_iam": iam * mix_spec,
        "oracle_complex_unrestricted": ideal_complex * mix_spec,
    }
    if projector is not None:
        out["erb_projected_complex_mask_diagnostic"] = erb_project_mask(ideal_complex, projector) * mix_spec
    if system_spec is not None:
        # Spectrum magnitude avoids dividing by a silent mixture bin; when non-silent
        # it is equivalent to applying the system-mask magnitude to the mixture.
        out["system"] = system_spec
        out["oracle_phase_with_system_magnitude"] = np.abs(system_spec) * np.exp(1j * np.angle(clean_spec))
    return out


def finite_metrics(clean, estimate):
    """A failed perceptual metric is represented as NaN per item, never a failed run."""
    estimate = np.asarray(estimate, np.float32)[:len(clean)]
    values = []
    for fn in (metrics.snr_db, metrics.stoi, metrics.pesq_wb):
        try:
            values.append(fn(clean, estimate))
        except (ValueError, RuntimeError):
            values.append(float("nan"))
    # Use the training loss's epsilon convention instead of assuming evaluation
    # SNR is numerically identical near perfect reconstruction. Aggregated binary
    # columns are the binding fraction, not an assertion about model quality.
    loss_snr = float(absolute_snr(torch.from_numpy(estimate)[None],
                                torch.from_numpy(np.asarray(clean, np.float32))[None])[0])
    values.extend([loss_snr, float(loss_snr >= 20), float(loss_snr >= 30)])
    return {name: float(value) if np.isfinite(value) else float("nan")
            for name, value in zip(METRIC_NAMES, values, strict=True)}


def aggregate_rows(rows):
    """Count validity per metric: PESQ can reject a clip while SNR remains valid."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["bucket"], row["variant"])].append(row)
    result = []
    for (bucket, variant), group in sorted(grouped.items()):
        aggregate = {"bucket": bucket, "variant": variant, "n_items": len(group)}
        for name in METRIC_NAMES:
            valid = np.asarray([r.get(name, float("nan")) for r in group], float)
            valid = valid[np.isfinite(valid)]
            aggregate[f"n_valid_{name}"] = int(len(valid))
            aggregate[f"mean_{name}"] = float(valid.mean()) if len(valid) else None
        result.append(aggregate)
    return result


def dataset_identity(dataset, root, split):
    """Hash names and metadata, cheaply pinning rendered inputs without rereading WAVs."""
    digest = hashlib.sha256()
    for item in dataset.items:
        digest.update(str(item.relative_to(root)).replace("\\", "/").encode())
        digest.update(item.with_name(item.name.replace(".mix.wav", ".json")).read_bytes())
    return {"eval_root": str(root.resolve()), "split": split, "item_count": len(dataset),
            "manifest_sha256": digest.hexdigest()}


def system_identity(spec):
    """Record a checkpoint/config hash when its path is explicit in the system spec."""
    identity = {"spec": spec}
    prefix, sep, raw_path = spec.partition(":")
    if sep and prefix in {"ckpt", "cascade", "post"}:
        path = Path(raw_path)
        if path.is_file():
            identity.update(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return identity


def write_items(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["id", "bucket", "variant", *METRIC_NAMES]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        writer.writerows(rows)


def write_aggregate(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        fields = ["bucket", "variant", "n_items", *[f"n_valid_{n}" for n in METRIC_NAMES],
                  *[f"mean_{n}" for n in METRIC_NAMES]]
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
            writer.writerows(payload["aggregates"])
    else:
        path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", default="data/eval_r2")
    ap.add_argument("--split", default="val", choices=("val", "test"))
    ap.add_argument("--allow-test", action="store_true", help="required when --split test is intentional")
    ap.add_argument("--system", help="optional baseline or ckpt:/cascade:/post: system to compare")
    ap.add_argument("--per-bucket", type=int, help="explicit diagnostic subset; default evaluates every item")
    ap.add_argument("--items-out", default="results/ceiling_analysis_items.csv")
    ap.add_argument("--aggregate-out", default="results/ceiling_analysis_aggregate.json")
    a = ap.parse_args()
    if a.split == "test" and not a.allow_test:
        ap.error("--split test requires --allow-test")
    if a.per_bucket is not None and a.per_bucket < 1:
        ap.error("--per-bucket must be positive")

    root = Path(a.eval_root)
    dataset = RenderedDataset(root / a.split)
    if not len(dataset):
        raise SystemExit(f"no rendered items under {root / a.split}")
    fn = enhance_fn(a.system, device="cpu") if a.system else None
    projector = erb_projector()
    seen, rows = defaultdict(int), []
    for index in range(len(dataset)):
        item = dataset[index]; meta = item["meta"]; bucket = meta.get("bucket", "?")
        if a.per_bucket is not None and seen[bucket] >= a.per_bucket:
            continue
        seen[bucket] += 1
        mix, clean = item["mix"].numpy(), item["clean"].numpy(); length = len(clean)
        mix_spec, clean_spec = to_c(mix[0]), to_c(clean)
        system_spec = to_c(np.asarray(fn(mix), np.float32)[:length]) if fn else None
        for variant, spectrum in oracle_spectra(mix_spec, clean_spec, system_spec, projector).items():
            row = {"id": meta.get("id"), "bucket": bucket, "variant": variant}
            row.update(finite_metrics(clean, from_c(spectrum, length)))
            rows.append(row)

    aggregates = aggregate_rows(rows)
    payload = {"generated_utc": datetime.now(UTC).isoformat(), "input": dataset_identity(dataset, root, a.split),
               "system": system_identity(a.system) if a.system else None, "per_bucket": a.per_bucket,
               "aggregates": aggregates}
    write_items(Path(a.items_out), rows); write_aggregate(Path(a.aggregate_out), payload)
    print(json.dumps({"items_out": a.items_out, "aggregate_out": a.aggregate_out,
                      "input": payload["input"], "system": payload["system"], "aggregates": aggregates}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
