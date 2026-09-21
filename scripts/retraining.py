"""Generate/check retraining recipes; only the explicit launch subcommand trains."""
import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path

import torch
import yaml

from vaani.experiments import preflight, recipes


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    subs = ap.add_subparsers(dest="command", required=True)
    gen = subs.add_parser("generate")
    gen.add_argument("--base", default="configs/exp/vaani_full_r4_ctl.yaml")
    gen.add_argument("--anchor", default="runs/vaani_full_r4_ctl/best.pt")
    gen.add_argument("--out", default="configs/retraining")
    for command in ("preflight", "launch"):
        parser = subs.add_parser(command)
        parser.add_argument("configs", nargs="+")
        if command == "preflight":
            parser.add_argument("--out", default=None)
            parser.add_argument("--skip-file-checks", action="store_true", help="architecture-only; still needs source checkpoints")
    a = ap.parse_args(argv)
    if a.command == "generate":
        root = Path(a.out); root.mkdir(parents=True, exist_ok=True)
        configs = recipes(yaml.safe_load(Path(a.base).read_text()), a.anchor)
        for name, cfg in configs.items():
            p = root / f"{name}.yaml"; text = yaml.safe_dump(cfg, sort_keys=False)
            if p.exists() and p.read_text() != text:
                raise FileExistsError(f"Refusing to overwrite a changed recipe: {p}")
            p.write_text(text, encoding="utf-8")
        print(f"Prepared {len(configs)} recipes; no training launched")
        return 0
    paths = []
    for pattern in a.configs:
        matches = sorted(glob.glob(pattern))
        if not matches:
            ap.error(f"No config matches {pattern!r}")
        paths.extend(matches)
    if len(paths) != len(set(paths)):
        ap.error("Duplicate configuration paths")
    torch.set_num_threads(1)
    reports, configs = [], []
    for p in paths:
        cfg = yaml.safe_load(Path(p).read_text()); configs.append(cfg)
        try:
            result = preflight(cfg, check_files=not getattr(a, "skip_file_checks", False))
            reports.append(dict(config=p, status="ready", **result))
        except (ValueError, RuntimeError, KeyError, OSError, TypeError) as error:
            reports.append(dict(config=p, status="blocked", error=str(error), training_launched=False))
    text = json.dumps(reports, indent=2)
    print(text)
    if getattr(a, "out", None):
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text + "\n", encoding="utf-8")
    if any(r["status"] != "ready" for r in reports):
        return 1
    if a.command == "launch":
        # Entire batch passes preflight before the first subprocess can start.
        # Run serially by default; concurrency/memory fit is a measured host decision.
        import os
        env = dict(os.environ)
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
            env[key] = "1"
        for p, cfg in zip(paths, configs):
            # Initialize the anchor's RIR bank once before spawning loader workers
            # to avoid concurrent unpacking of the bank's shared .npy files.
            from vaani.data.rirs import RirBank
            if cfg["model"] == "vaani_cascade":
                data = torch.load(cfg["base_checkpoint"], map_location="cpu", weights_only=True)["config"]["data"]
            else:
                data = cfg["data"]
            if data.get("bank"):
                RirBank(data["bank"])
            module = "vaani.train_refiner" if cfg["model"] == "vaani_cascade" else "vaani.train"
            subprocess.run([sys.executable, "-m", module, p], env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
