"""r8 rental-box preflight: every input a queued r8 run reads is present, verified and decodable before GPU time.

usage (repo root):
    python scripts/r8_preflight.py                        # all r8 configs (full + ablation pilots)
    python scripts/r8_preflight.py --config configs/retraining/r8_fe_mini.yaml --sample 50 --smoke 20
    python scripts/r8_preflight.py --fetch-order          # datasets for the first queued jobs, then the rest
    python scripts/r8_preflight.py --mem-summary runs/box_setup/bench_mem_loader.log   # bench memory -> --mem-out JSON

Checks (FAIL blocks the queue, WARN is printed and recorded):
  manifests     exist, have train rows, and a seeded sample of N audio paths resolves and decodes at 16 kHz;
                source_ids compared with the laptop reference copy from the mirror when it is present (WARN)
  banks         data.bank and its sidecars match configs/data/r8_banks.json by sha256 (bank_r3 falls back to the
                published hash in scripts/remote_setup.sh with a WARN when r8_banks.json is absent)
  val           val.eval_root/val exists and scripts/verify_eval_set.py passes against its expected hash
  heldout       data.exclude_groups_file exists and every listed source_id is present in the manifest it names
  init          init_from exists and matches init_sha256
  imports       numba and torch_pesq import (FAIL), faster_whisper (WARN: G4 part 2 / the MAD VAD only)
  cuda          torch sees at least --gpus devices
  disk          free space on the repo's filesystem >= --need-gb
  g1            the G1 result JSON exists, gate_pass is true, it scored >= 200 items on the full configs' bank
  smoke         (--smoke N) N training steps per config into runs/preflight_<name>
Writes runs/preflight.json; exits 1 on any FAIL.
"""
import argparse, glob, hashlib, json, os, random, shutil, subprocess, sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

FULL = ["configs/retraining/r8_fe_mini.yaml", "configs/retraining/r8_refvalid_v2.yaml"]
PILOTS = "configs/retraining/r8_ablations/*.yaml"
BANKS_JSON = "configs/data/r8_banks.json"
DATASETS_YAML = "configs/data/r8_datasets.yaml"
G1_JSON = "results_r2/r8/data_gates/box_g1/v2.json"
G1_MIN_ITEMS = 200
MIRROR = "data/mirror"
# published box copy used by r7 (scripts/remote_setup.sh); the laptop copy differs (99dcfb26...), see banks lane notes
KNOWN_BANKS = {"data/rirs/bank_r3.npz": "e4e67463072e1dca14b94bef97a99bb85da34ebee7ec657a1f1dc7554df1b59f"}
# recomputed by scripts/verify_eval_set.py on the laptop copy, 2026-09-25 (1,480 items)
EVAL_HASHES = {"data/eval_r2/val": "b5f7a4d43bee"}
FIRST_JOBS = ["ab1_fe_mini_s0", "ab1_refvalid_s0"]   # heads of the two GPU queues in scripts/run_r8.sh


class Report:
    def __init__(self):
        self.rows = []

    def add(self, check, status, what, detail=""):
        self.rows.append(dict(check=check, status=status, what=str(what), detail=str(detail)))

    def failed(self):
        return [r for r in self.rows if r["status"] == "FAIL"]


def default_configs(root):
    return [c for c in FULL if (root / c).exists()] + sorted(
        str(Path(p).relative_to(root).as_posix()) for p in glob.glob(str(root / PILOTS)))


def load_cfgs(root, paths):
    return {p: yaml.safe_load(open(root / p, encoding="utf-8")) for p in paths}


def sha256(path, cache=None):
    """sha256 with a (size, mtime) cache so a rerun does not rehash gigabytes."""
    p = Path(path); st = p.stat(); key = f"{p.resolve()}|{st.st_size}|{int(st.st_mtime)}"
    if cache is not None and key in cache:
        return cache[key]
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    if cache is not None:
        cache[key] = h.hexdigest()
    return h.hexdigest()


def check_manifests(root, cfgs, rep, sample, seed=0):
    import soundfile as sf
    from vaani.data import manifests as M
    seen = sorted({m for c in cfgs.values() for m in c["data"]["manifests"]})
    for m in seen:
        p = root / m
        if not p.exists():
            rep.add("manifests", "FAIL", m, "missing"); continue
        df = M.read(p)
        if not (df.split == "train").any():
            rep.add("manifests", "FAIL", m, f"{len(df)} rows, none in split train"); continue
        gone = [q for q in df.path if not (root / q).exists()]
        if gone:
            rep.add("manifests", "FAIL", m, f"{len(gone)}/{len(df)} paths missing, e.g. {gone[:2]}"); continue
        rows = random.Random(seed).sample(range(len(df)), min(sample, len(df)))
        bad = []
        for i in rows:
            a = root / df.path.iloc[i]
            try:
                with sf.SoundFile(str(a)) as f:
                    if f.samplerate != 16000:
                        bad.append(f"{df.path.iloc[i]}: {f.samplerate} Hz"); continue
                    f.read(min(1600, f.frames), dtype="float32")
            except Exception as e:   # noqa: BLE001 - any open/decode error is a bad path
                bad.append(f"{df.path.iloc[i]}: {type(e).__name__}")
        rep.add("manifests", "FAIL" if bad else "ok", m,
                f"{len(bad)}/{len(rows)} sampled paths bad: {bad[:3]}" if bad else f"{len(df)} rows, {len(rows)} sampled decode")
        ref = root / MIRROR / "manifests_laptop" / Path(m).name   # laptop scan, shipped for comparison only
        if ref.exists():
            a, b = set(df.source_id), set(M.read(ref).source_id)
            if a != b:
                rep.add("manifests", "WARN", m, f"differs from the laptop scan: {len(b - a)} missing, {len(a - b)} extra source_ids")


def check_banks(root, cfgs, rep, cache):
    table = json.loads((root / BANKS_JSON).read_text()) if (root / BANKS_JSON).exists() else None
    if table is None:
        rep.add("banks", "WARN", BANKS_JSON, "absent: falling back to the hashes in scripts/remote_setup.sh")
    for b in sorted({c["data"].get("bank") for c in cfgs.values() if c["data"].get("bank")}):
        p = root / b
        if not p.exists():
            rep.add("banks", "FAIL", b, "missing"); continue
        entry = next((e for e in (table or {}).values() if Path(e.get("file", "")).as_posix() == Path(b).as_posix()), None)
        want = entry["sha256"] if entry else KNOWN_BANKS.get(Path(b).as_posix())
        if want is None:
            rep.add("banks", "FAIL", b, f"no sha256 for it in {BANKS_JSON}"); continue
        got = sha256(p, cache)
        rep.add("banks", "ok" if got == want else "FAIL", b, f"sha256 {got[:12]} (want {want[:12]})")
        for side, sw in ((entry or {}).get("sidecars") or {}).items():
            sp = p.parent / side
            if not sp.exists():
                rep.add("banks", "FAIL", sp.relative_to(root).as_posix(), "sidecar missing"); continue
            sg = sha256(sp, cache)
            rep.add("banks", "ok" if sg == sw else "FAIL", sp.relative_to(root).as_posix(), f"sha256 {sg[:12]} (want {sw[:12]})")


def check_val(root, cfgs, rep, overrides):
    sys.path.insert(0, str(root / "scripts")); sys.path.insert(0, str(REPO / "scripts"))
    from verify_eval_set import verify
    for er in sorted({(c.get("val") or {}).get("eval_root") for c in cfgs.values() if (c.get("val") or {}).get("eval_root")}):
        v = Path(er).as_posix() + "/val"; p = root / v
        want = overrides.get(v) or EVAL_HASHES.get(v)
        if not p.exists():
            rep.add("val", "FAIL", v, "missing"); continue
        if want is None:
            rep.add("val", "FAIL", v, "no expected hash (pass --val-hash)"); continue
        probs = verify(p, want)
        rep.add("val", "FAIL" if probs else "ok", v, "; ".join(probs[:3]) if probs else f"hash {want} verified")


def check_heldout(root, cfgs, rep):
    from vaani.data import manifests as M
    done = set()
    for c in cfgs.values():
        f = c["data"].get("exclude_groups_file")
        if not f or f in done:
            continue
        done.add(f); p = root / f
        if not p.exists():
            rep.add("heldout", "FAIL", f, "missing: training would see r8 test-set groups"); continue
        j = json.loads(p.read_text(encoding="utf-8"))
        names = {Path(m).name: m for cc in cfgs.values() if cc["data"].get("exclude_groups_file") == f for m in cc["data"]["manifests"]}
        for src, s in (j.get("sources") or {}).items():
            m = names.get(s.get("manifest", ""))
            if m is None or not (root / m).exists():
                continue   # the source's manifest is not trained on by these configs (or reported missing above)
            ids = set(s.get("source_ids", [])); have = ids & set(M.read(root / m).source_id)
            st = "ok" if have == ids else ("FAIL" if not have else "WARN")
            rep.add("heldout", st, f"{f}:{src}", f"{len(have)}/{len(ids)} listed source_ids found in {m}")


def check_init(root, cfgs, rep, cache):
    for f, want in sorted({(c.get("init_from"), c.get("init_sha256")) for c in cfgs.values()}, key=str):
        if not f:
            continue
        p = root / f
        if not p.exists():
            rep.add("init", "FAIL", f, "missing"); continue
        got = sha256(p, cache)
        rep.add("init", "ok" if not want or got == want else "FAIL", f, f"sha256 {got[:12]}" + (f" (want {want[:12]})" if want else ""))


def check_imports(rep):
    import importlib
    for mod, bad in (("numba", "FAIL"), ("torch_pesq", "FAIL"), ("faster_whisper", "WARN")):
        try:
            importlib.import_module(mod); rep.add("imports", "ok", mod)
        except Exception as e:   # noqa: BLE001
            rep.add("imports", bad, mod, f"{type(e).__name__}: {e}")


def check_cuda(rep, need):
    try:
        import torch
        n = torch.cuda.device_count()
    except Exception as e:   # noqa: BLE001
        rep.add("cuda", "FAIL", "torch", str(e)); return
    rep.add("cuda", "ok" if n >= need else "FAIL", f"{n} device(s)", f"need {need}")


def check_disk(root, rep, need_gb):
    free = shutil.disk_usage(root).free / 1e9
    rep.add("disk", "ok" if free >= need_gb else "FAIL", f"{free:.0f} GB free", f"need {need_gb:g} GB")


def check_g1(root, cfgs, rep, path):
    p = root / path
    if not p.exists():
        rep.add("g1", "FAIL", path, "missing: run the G1 gate on this box first"); return
    j = json.loads(p.read_text())
    items = min((j.get(k) or {}).get("items", 0) for k in ("param", "room") if k in j) if any(k in j for k in ("param", "room")) else 0
    # the gate covers the full runs' bank; a pilot on another bank (gen_r8_configs.py --bank-arm) is not gated by it
    banks = {Path(c["data"]["bank"]).name for s, c in cfgs.items() if c["data"].get("bank") and (s in FULL or not
             any(k in FULL for k in cfgs))}
    probs = []
    if j.get("gate_pass") is not True:
        probs.append(f"gate_pass {j.get('gate_pass')}")
    if items < G1_MIN_ITEMS:
        probs.append(f"{items} items < {G1_MIN_ITEMS}")
    if banks and j.get("bank") not in banks:
        probs.append(f"gated bank {j.get('bank')} is not the configs' {sorted(banks)}")
    # the gate must have mixed what the full runs train on: their mix.v2 block, and no scene overrides (train.py has no
    # way to pass scenes.sample_scene overrides, so a gate that used them gated a mixer nothing trains with)
    if j.get("scene_overrides"):
        probs.append(f"gated with scene overrides {j['scene_overrides']} that training cannot apply")
    for src, c in cfgs.items():
        if src in FULL:
            want = ((c["data"].get("mix") or {}).get("v2") or {})
            if (j.get("v2_overrides") or {}) != want:
                probs.append(f"gated v2 block {j.get('v2_overrides')} != {Path(src).name} mix.v2 {want}")
    rep.add("g1", "FAIL" if probs else "ok", path, "; ".join(probs) or f"pass, seed {j.get('seed')}, {items} items")


def smoke(root, cfgs, rep, steps):
    for src, c in cfgs.items():
        c = json.loads(json.dumps(c)); name = "preflight_" + c["name"]
        c.update(name=name, epochs=1, max_steps=steps, log_every=1, resume=False)
        c["data"]["epoch_len"] = steps * c["batch_size"]
        c["val"] = dict(c.get("val") or {}, dynamic_items=16, composite=dict(every=1, per_bucket=1, limit=6, ilds=[-8, 0]))
        rd = root / "runs" / name
        shutil.rmtree(rd, ignore_errors=True); (root / "runs").mkdir(exist_ok=True)
        cp = root / "runs" / f"{name}.yaml"; yaml.safe_dump(c, open(cp, "w"))
        r = subprocess.run([sys.executable, "-m", "vaani.train", str(cp)], cwd=root, capture_output=True, text=True)
        ok = r.returncode == 0 and (rd / "last.pt").exists()
        (root / "runs" / f"{name}.log").write_text(r.stdout + r.stderr)
        rep.add("smoke", "ok" if ok else "FAIL", src, f"rc {r.returncode}, {steps} steps, log runs/{name}.log")


def g1_manifests():
    """The manifests scripts/data_gates.py reads for the v2 gate (its own lists: not the training list)."""
    sys.path.insert(0, str(REPO / "scripts"))
    import data_gates as g
    return list(g.SPEECH_MANIFESTS) + list(g.V2_NOISE)


def fetch_order(root, first=FIRST_JOBS, extra_manifests=()):
    """(first, rest): the datasets the queue heads and the G1 gate need (by needed_by stem or by manifest file), then
    every other one an r8 config needs, then the others."""
    d = (yaml.safe_load(open(root / DATASETS_YAML, encoding="utf-8")) or {}).get("datasets") or {}
    want = set(extra_manifests)
    for stem in first:
        c = root / "configs/retraining/r8_ablations" / f"{stem}.yaml"
        if c.exists():
            want.update(Path(m).name for m in yaml.safe_load(open(c, encoding="utf-8"))["data"]["manifests"])

    def mans(v):
        m = v.get("manifest") or []
        return {Path(x).name for x in ([m] if isinstance(m, str) else m)}
    need = {k: set(v.get("needed_by") or []) for k, v in d.items()}
    a = [k for k, v in d.items() if need[k] & set(first) or mans(v) & want]
    b = [k for k in d if k not in a and need[k]]
    return a, b + [k for k in d if k not in a and k not in b]


def bank_plan(root, cfgs):
    """(rows, warnings): one (file, sha256, asset, need|opt) row per bank file the box should hold ("need": a config
    trains on it, or it is one of that bank's sidecars). Every entry of r8_banks.json and
    its sidecars, plus any config bank the table lacks, by the published hash in KNOWN_BANKS. A bank is never
    regenerated on the box: pyroomacoustics differs across machines, so a rebuilt bank is a different file."""
    rows, warn = [], []
    table = json.loads((root / BANKS_JSON).read_text()) if (root / BANKS_JSON).exists() else None
    if table is None:
        warn.append(f"{BANKS_JSON} absent: only the configs' banks are fetched, by the hashes in scripts/remote_setup.sh")
    for e in (table or {}).values():
        f = Path(e["file"]).as_posix()
        rows.append((f, e["sha256"], e.get("asset") or Path(f).name))
        for side, sw in (e.get("sidecars") or {}).items():
            rows.append(((Path(f).parent / side).as_posix(), sw, side))
    used = {Path(c["data"]["bank"]).as_posix() for c in cfgs.values() if c["data"].get("bank")}
    have = {r[0] for r in rows}
    for b in sorted(used - have):
        if b in KNOWN_BANKS:
            rows.append((b, KNOWN_BANKS[b], Path(b).name))
            if table is not None:
                warn.append(f"{b} is trained on but not in {BANKS_JSON}: using the remote_setup.sh hash")
        else:
            warn.append(f"{b}: no published sha256 anywhere; the box cannot fetch it")
    stem = lambda f: Path(f).name.split(".")[0]   # bank_r8.speech.npy -> bank_r8
    need = {stem(b) for b in used}
    return [(*r, "need" if stem(r[0]) in need else "opt") for r in rows], warn


def g1_command(root, cfgs, seed, items, out, py=None):
    """argv of the box G1 gate: the full runs' own mix.v2 block and bank, so the gate mixes exactly what trains."""
    full = [(s, c) for s, c in cfgs.items() if s in FULL]
    if not full:
        raise SystemExit("no full r8 config found to gate")
    blocks = {json.dumps(((c["data"].get("mix") or {}).get("v2") or {}), sort_keys=True) for _, c in full}
    banks = {c["data"].get("bank") for _, c in full}
    if len(blocks) > 1 or len(banks) > 1:
        raise SystemExit(f"the full configs disagree on mix.v2 {sorted(blocks)} or bank {sorted(banks)}: one gate cannot cover both")
    return [py or sys.executable, "scripts/data_gates.py", "--versions", "2", "--items", str(items), "--seed", str(seed),
            "--bootstrap", "2000", "--bank", banks.pop(), "--v2", blocks.pop(), "--out", out]


def mem_summary(log):
    """Summary of a memwatch log from r8_box_setup.sh's bench stage. Lines: '# root <pid>', '<t> mem <MemAvailable kB>',
    '<t> proc <pid> <ppid> <rss kB> <pss kB> <private kB> <comm>'. Workers are the root's direct children (DataLoader
    forks them); Rss counts pages shared with the parent and the page cache, Pss splits them, private is USS."""
    root, mem, procs = None, [], []
    for ln in open(log, encoding="utf-8"):
        f = ln.split()
        if ln.startswith("# root") and len(f) >= 3:
            root = f[2].rstrip(";")   # memwatch writes "# root <pid>; lines: ..."
        elif len(f) >= 3 and f[1] == "mem" and f[2].isdigit():
            mem.append((int(f[0]), int(f[2])))
        elif len(f) >= 7 and f[1] == "proc":
            procs.append(dict(t=int(f[0]), pid=f[2], ppid=f[3], rss=int(f[4] or 0), pss=int(f[5] or 0),
                              priv=int(f[6] or 0), comm=f[7] if len(f) > 7 else ""))
    gb = lambda kb: round(kb * 1024 / 1e9, 3)   # noqa: E731 - /proc reports KiB
    main = [p for p in procs if p["pid"] == root]
    work = [p for p in procs if p["ppid"] == root]
    per_t = {}
    for p in work:
        s = per_t.setdefault(p["t"], dict(n=0, pss=0, priv=0)); s["n"] += 1; s["pss"] += p["pss"]; s["priv"] += p["priv"]
    out = dict(log=str(log), samples=len({t for t, _ in mem}), root_pid=root,
               mem_available_gb=dict(first=gb(mem[0][1]), min=gb(min(m for _, m in mem))) if mem else None,
               main=dict(rss_gb_max=gb(max(p["rss"] for p in main)), pss_gb_max=gb(max(p["pss"] for p in main)),
                         private_gb_max=gb(max(p["priv"] for p in main))) if main else None,
               worker=dict(n_max=max(s["n"] for s in per_t.values()),
                           rss_gb_max=gb(max(p["rss"] for p in work)), pss_gb_max=gb(max(p["pss"] for p in work)),
                           private_gb_max=gb(max(p["priv"] for p in work)),
                           pss_gb_sum_max=gb(max(s["pss"] for s in per_t.values())),
                           private_gb_sum_max=gb(max(s["priv"] for s in per_t.values()))) if work else None)
    if mem:   # inferred: the drop also holds page cache the bench faulted in, and anything else the box ran meanwhile
        out["mem_available_drop_gb"] = round(out["mem_available_gb"]["first"] - out["mem_available_gb"]["min"], 3)
    return out


BOX_ENV = {"MIRROR_HF_REPO": "private HF dataset repo with the laptop-only artefacts (scripts/r8_mirror_stage.sh)",
           "HF_TOKEN": "read token for that repo",
           "RIR_BANK_URL": "GitHub release asset base for the RIR banks (configs/data/r8_banks.json)"}


def env_check(root, env=None):
    """Lines naming every missing variable, and whether a missing one blocks the first queued jobs. Values are never
    printed. Dataset credentials come from r8_datasets.yaml (`credentials`); a dataset in the first fetch group
    needs them before the bootstrap starts, the rest only warn."""
    env = os.environ if env is None else env
    lines, block = [], False
    for k, why in BOX_ENV.items():
        ok = bool(env.get(k)); block |= not ok
        lines.append(f"{'ok  ' if ok else 'MISSING'} {k}: {why}")
    if not (root / DATASETS_YAML).exists():
        return lines + [f"MISSING {DATASETS_YAML}: dataset credentials unknown (the datasets stage cannot run)"], True
    d = (yaml.safe_load(open(root / DATASETS_YAML, encoding="utf-8")) or {}).get("datasets") or {}
    try:
        g1 = g1_manifests()
    except Exception:   # noqa: BLE001
        g1 = []
    first, _ = fetch_order(root, extra_manifests=g1)
    for k, v in d.items():
        for c in v.get("credentials") or []:
            ok = bool(env.get(c)); hard = k in first
            block |= hard and not ok
            lines.append(f"{'ok  ' if ok else 'MISSING' if hard else 'WARN'} {c}: {k} (access {v.get('access')}"
                         f"{', first fetch group' if hard else ''})")
        if v.get("access") in ("manual", "request") and k in first:
            lines.append(f"NOTE {k}: access {v.get('access')} - {v.get('notes') or 'see ' + DATASETS_YAML}")
    return lines, block


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", action="append", help="repeatable; default: every r8 config")
    ap.add_argument("--root", default=str(REPO))
    ap.add_argument("--sample", type=int, default=20, help="audio paths decoded per manifest")
    ap.add_argument("--gpus", type=int, default=2)
    ap.add_argument("--need-gb", type=float, default=50.0, help="free disk still needed for runs/ (inferred headroom)")
    ap.add_argument("--g1", default=G1_JSON)
    ap.add_argument("--val-hash", action="append", default=[], metavar="ROOT=HASH")
    ap.add_argument("--skip", default="", help="comma list of checks to skip")
    ap.add_argument("--smoke", type=int, default=0, metavar="N")
    ap.add_argument("--json", default="runs/preflight.json")
    ap.add_argument("--fetch-order", action="store_true", help="print FIRST=<a,b> and REST=<c,d> and exit")
    ap.add_argument("--env-check", action="store_true", help="name missing box environment variables; exit 1 if blocking")
    ap.add_argument("--bank-plan", action="store_true", help="print '<file> <sha256> <asset>' per bank file to fetch")
    ap.add_argument("--g1-cmd", action="store_true", help="print the box G1 gate command (shell-quoted) and exit")
    ap.add_argument("--g1-seed", type=int, default=202)
    ap.add_argument("--g1-items", type=int, default=G1_MIN_ITEMS)
    ap.add_argument("--mem-summary", nargs="+", metavar="LOG", help="summarise bench memwatch logs and exit")
    ap.add_argument("--mem-out", default="results_r2/r8/loader_mem_box.json")
    a = ap.parse_args(argv)
    root = Path(a.root).resolve()
    if a.mem_summary:
        s = {Path(p).stem: mem_summary(p) for p in a.mem_summary}
        out = root / a.mem_out; out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(dict(nproc=os.cpu_count(), logs=s), indent=1))
        for k, v in s.items():
            print(f"MEM {k}: MemAvailable {v['mem_available_gb']}, main {v['main']}, per worker {v['worker']}")
        return 0
    if a.bank_plan or a.g1_cmd:
        cfgs = load_cfgs(root, a.config or default_configs(root))
        if a.g1_cmd:
            import shlex
            print(shlex.join(g1_command(root, cfgs, a.g1_seed, a.g1_items, str(Path(a.g1).parent.as_posix()),
                                        py=os.environ.get("PY")))); return 0
        rows, warn = bank_plan(root, cfgs)
        for w in warn:
            print(f"WARN {w}", file=sys.stderr)
        for r in rows:
            print(" ".join(r))
        return 0 if rows and not any("cannot fetch" in w for w in warn) else 2
    if a.env_check:
        lines, block = env_check(root)
        for ln in lines:
            print(ln)
        return 1 if block else 0
    if a.fetch_order:
        if not (root / DATASETS_YAML).exists():
            print(f"missing {DATASETS_YAML}", file=sys.stderr); return 2
        try:
            g1 = g1_manifests()
        except Exception as e:   # noqa: BLE001 - order by the configs alone
            print(f"G1 manifest list unavailable: {e}", file=sys.stderr); g1 = []
        f, r = fetch_order(root, extra_manifests=g1); print("FIRST=" + ",".join(f)); print("REST=" + ",".join(r)); return 0
    skip = set(filter(None, a.skip.split(",")))
    paths = a.config or default_configs(root)
    cfgs = load_cfgs(root, paths)
    rep = Report()
    cache_p = root / "data" / "rirs" / ".sha256_cache.json"
    cache = json.loads(cache_p.read_text()) if cache_p.exists() else {}
    steps = [("manifests", lambda: check_manifests(root, cfgs, rep, a.sample)),
             ("banks", lambda: check_banks(root, cfgs, rep, cache)),
             ("val", lambda: check_val(root, cfgs, rep, dict(v.split("=", 1) for v in a.val_hash))),
             ("heldout", lambda: check_heldout(root, cfgs, rep)),
             ("init", lambda: check_init(root, cfgs, rep, cache)),
             ("imports", lambda: check_imports(rep)),
             ("cuda", lambda: check_cuda(rep, a.gpus)),
             ("disk", lambda: check_disk(root, rep, a.need_gb)),
             ("g1", lambda: check_g1(root, cfgs, rep, a.g1))]
    for name, fn in steps:
        if name not in skip:
            fn()
    if a.smoke and not rep.failed():   # no GPU steps on a box whose inputs already failed
        smoke(root, cfgs, rep, a.smoke)
    try:
        cache_p.parent.mkdir(parents=True, exist_ok=True); cache_p.write_text(json.dumps(cache))
    except OSError:
        pass
    for r in rep.rows:
        print(f"{r['status']:4} {r['check']:9} {r['what']}  {r['detail']}")
    out = root / a.json; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(configs=paths, rows=rep.rows, failed=len(rep.failed())), indent=1))
    bad = rep.failed()
    print(f"PREFLIGHT {'FAIL' if bad else 'PASS'}: {len(bad)} failed, "
          f"{sum(r['status'] == 'WARN' for r in rep.rows)} warnings, {len(paths)} configs")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
