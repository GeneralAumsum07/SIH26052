"""Tier 4.6 evaluation protocol (plan 2026-09-20-tier46-second-stage §2).

  freeze  content-hash every file of the frozen test + val splits, validate the WAVs, pin the anchor checkpoint
          (copied to runs/tier46_anchor/best.pt, never overwritten by a different file) and write anchor.json
  check   re-verify a split against anchor.json (bytes, not just presence)
  gate    run the paired kill gate (vaani.tier46_gate) and exit 1 when the candidate fails

The existing verify_eval_set.py hashes the metas only; a half-copied or re-rendered WAV passes it. This one does not.
"""
import argparse, hashlib, json, os, shutil, sys
from pathlib import Path

import numpy as np, soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vaani import tier46_gate  # noqa: E402

SR = 16000


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def _split_manifest(root: Path):
    """Hash every file; validate rate/channels/length/finiteness; twins are required wherever the meta says an impulse."""
    problems, files, keys = [], {}, []
    metas = sorted(root.rglob("*.json"))
    if not metas: return [f"{root}: no metas"], files, keys
    for m in metas:
        meta = json.loads(m.read_text()); stem = m.with_suffix("")
        keys.append([m.parent.name, m.stem])
        need = [m, Path(str(stem) + ".mix.wav"), Path(str(stem) + ".clean.wav")]
        if meta.get("impulse_onsets_s"): need.append(Path(str(stem) + ".twin.mix.wav"))   # burst clips carry a no-burst twin
        lens = {}
        for f in need:
            if not f.exists(): problems.append(f"missing {f}"); continue
            files[f.relative_to(root).as_posix()] = _sha(f)
            if f.suffix == ".wav":
                x, sr = sf.read(f, dtype="float32")
                ch = 1 if x.ndim == 1 else x.shape[1]
                want = 1 if f.name.endswith(".clean.wav") else 2
                if sr != SR or ch != want or not np.isfinite(x).all():
                    problems.append(f"{f}: sr={sr} ch={ch} finite={bool(np.isfinite(x).all())}")
                lens[f.name] = len(x)
        if len(set(lens.values())) > 1: problems.append(f"{stem}: length mismatch {lens}")
    h = root / "EVALSET_HASH"
    if h.exists(): files["EVALSET_HASH"] = _sha(h)
    return problems, files, keys


def check(eval_root: Path, proto: dict):
    """Re-hash both splits and diff against the frozen manifest. Returns a list of problems (empty = intact)."""
    out = []
    for split, rec in proto["splits"].items():
        problems, files, keys = _split_manifest(Path(eval_root) / split)
        out += problems
        if files != rec["files"]: out.append(f"{split}: file set/bytes differ from the frozen manifest")
        if keys != [list(k) for k in rec["keys"]]: out.append(f"{split}: item keys differ from the frozen manifest")
    return out


def freeze(eval_root, anchor_candidates, out_dir, anchor_dir=Path("runs/tier46_anchor")):
    eval_root, out_dir, anchor_dir = Path(eval_root), Path(out_dir), Path(anchor_dir)
    anchor = Path(anchor_candidates[0])   # selection among several is done on val by screen_tier46 (Task 3); the first is the default
    proto = {"anchor": {}, "splits": {}}
    problems = []
    for split in ("test", "val"):
        p, files, keys = _split_manifest(eval_root / split)
        problems += p
        proto["splits"][split] = {"root": (eval_root / split).as_posix(), "n_items": len(keys), "keys": keys, "files": files}
    if not anchor.exists(): problems.append(f"anchor missing: {anchor}")
    if problems:
        print("\n".join(problems)); sys.exit(1)
    sha = _sha(anchor)
    dst = anchor_dir / "best.pt"
    if dst.exists() and _sha(dst) != sha:
        print(f"{dst} exists with a different hash; refusing to overwrite the frozen anchor"); sys.exit(1)
    if not dst.exists():
        anchor_dir.mkdir(parents=True, exist_ok=True); shutil.copyfile(anchor, dst)
    if _sha(dst) != sha: print("anchor copy mismatch"); sys.exit(1)
    try:
        import torch
        cfg = torch.load(anchor, map_location="cpu", weights_only=True).get("config")
    except Exception as e:   # torch absent or foreign checkpoint: the hash still pins it
        cfg = f"unreadable: {e!r}"
    proto["anchor"] = {"source": anchor.as_posix(), "frozen_copy": dst.as_posix(), "sha256": sha, "config": cfg, "git": _git()}
    proto["keys"] = proto["splits"]["test"]["keys"]   # the gate pairs on the test split
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_dir / "anchor.json.tmp"; tmp.write_text(json.dumps(proto, indent=1)); os.replace(tmp, out_dir / "anchor.json")
    print(f"frozen: test {proto['splits']['test']['n_items']} items, val {proto['splits']['val']['n_items']} items, anchor {sha[:12]}")
    return proto


def _git():
    try:
        import subprocess
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze"); f.add_argument("--eval-root", default="data/eval_r2"); f.add_argument("--anchor", required=True)
    f.add_argument("--out", default="results_r2/tier46")
    c = sub.add_parser("check"); c.add_argument("--eval-root", default="data/eval_r2"); c.add_argument("--protocol", default="results_r2/tier46/anchor.json")
    g = sub.add_parser("gate"); g.add_argument("--kind", choices=list(tier46_gate.UTILITY), required=True)
    g.add_argument("--anchor", required=True); g.add_argument("--candidate", required=True)
    g.add_argument("--protocol", default="results_r2/tier46/anchor.json"); g.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "freeze":
        freeze(a.eval_root, [a.anchor], a.out)
    elif a.cmd == "check":
        bad = check(a.eval_root, json.loads(Path(a.protocol).read_text()))
        print("\n".join(bad) if bad else "intact"); sys.exit(1 if bad else 0)
    else:
        r = tier46_gate.compare(a.anchor, a.candidate, json.loads(Path(a.protocol).read_text()), a.kind)
        Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(r, indent=1, default=float))
        print(json.dumps({k: r[k] for k in ("complete", "gates", "pass") if k in r}, indent=1))
        sys.exit(0 if r["pass"] else 1)


if __name__ == "__main__":
    main()
