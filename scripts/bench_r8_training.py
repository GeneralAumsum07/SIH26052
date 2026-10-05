"""Isolated r8 training-step trials, with parity before timing.

Each variant runs in a fresh process: a failed CUDA capture must not contaminate
the next trial. Uses the actual config's model, loss, optimizer, clipping and EMA.
Synthetic waveforms isolate compute; use bench_loader.py and real smoke training
to check the data path. Never interprets unsupported/OOM/failed parity as a win.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

VARIANTS = {
    "lfilter": dict(filters="lfilter", compile=False, cuda_graph=False, gru_kernel="cudnn"),
    "fft": dict(filters="fft", compile=False, cuda_graph=False, gru_kernel="cudnn"),
    "graph": dict(filters="fft", compile=False, cuda_graph=True, gru_kernel="cudnn"),
    "compile": dict(filters="fft", compile=True, cuda_graph=False, gru_kernel="cudnn"),
    "fused": dict(filters="fft", compile=False, cuda_graph=False, gru_kernel="fused"),
    "graph-fused": dict(filters="fft", compile=False, cuda_graph=True, gru_kernel="fused"),  # fused GRU without compile's AMP reordering
    "compile-fused": dict(filters="fft", compile=True, cuda_graph=False, gru_kernel="fused"),
    "graph-compile": dict(filters="fft", compile=True, cuda_graph=True, gru_kernel="cudnn"),  # tiers whose GRU exceeds the fused kernel's shared memory
    "graph-compile-fused": dict(filters="fft", compile=True, cuda_graph=True, gru_kernel="fused"),  # the launch combo
}


def winners(rows):
    """Pick only successful, parity-tested trials, separately for each config/shape."""
    out = {}
    for r in rows:
        if r.get("status") != "passed" or not r.get("parity_passed"):
            continue
        key = f"{r['config']}|B{r['batch']}|{r['crop_s']}s"
        if key not in out or r["median_ms"] < out[key]["median_ms"]:
            out[key] = r
    return out


def trial(a):
    import copy
    import torch
    import yaml
    from vaani import train, runtime
    from vaani.audio_contract import contract_of
    from vaani.enhance_low_delay import build_fe_loss, term_grad_norms
    from vaani.train_graph import GraphedStep

    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the training-step benchmark")
    device = torch.device("cuda")
    cfg = yaml.safe_load(Path(a.config).read_text())
    variant = VARIANTS[a.variant]
    if a.no_amp:  # fp32 parity diagnosis: separates implementation bugs from bf16-AMP rounding
        cfg["amp"] = False
    runtime.tune_backends(device, tf32=not variant["compile"])  # as train.main
    if variant["compile"] or variant["gru_kernel"] == "fused":
        from vaani.models.gru_fused import fused_available
        if not fused_available(device):
            return {"status": "unsupported", "reason": "CUDA Triton is not installed in this environment"}
    batch = a.batch or cfg["batch_size"]
    crop_s = a.crop_s or cfg["data"].get("crop_s", 4.)
    n = int(crop_s * 16000)
    contract = contract_of(cfg.get("model_cfg"))
    low_delay = not contract.is_legacy
    warm = cfg["data"].get("warmup_samples", 0)
    scored = (warm, n) if warm else None

    def setup(v):
        torch.manual_seed(cfg["seed"])
        m = train.build_model(cfg["model"], model_cfg=cfg.get("model_cfg")).to(device).train()
        if v["gru_kernel"] == "fused":
            from vaani.models.gru_fused import use_fused_gru
            use_fused_gru(m)
        lc = copy.deepcopy(cfg["loss_cfg"])
        lc["pesq_filters"] = v["filters"]
        domain = lc.pop("loss_domain", "resynthesis")
        lf = build_fe_loss(lc, cfg.get("model_cfg"), domain) if low_delay else train.losses.build_loss("fe", lc)
        ema = train.EMA(m, cfg["ema"]["decay"], cfg["ema"].get("warmup", True)) if cfg.get("ema") else None
        fwd = torch.compile(m) if v["compile"] else m
        gs = GraphedStep(fwd, lf, device, cfg.get("amp", True), low_delay, scored) if v["cuda_graph"] else None
        groups = train.build_param_groups(m, cfg["optim"])
        opt = torch.optim.AdamW(groups, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: train.cosine_lr_multiplier(s, cfg["optim"].get("warmup", 500), cfg["epochs"] * ((cfg["data"]["epoch_len"] + batch - 1) // batch)))
        return m, lf, fwd, gs, groups, opt, sched, ema

    # Fixed deterministic batches include clean and silent references. The real
    # model input preparation is timed too, including waveform transfer/STFT.
    gen = torch.Generator().manual_seed(4321)
    batches = []
    for j in range(3):
        mix = torch.randn(batch, 2, n, generator=gen) * .05
        clean = mix[:, 0].clone() * .8
        if j == 1:
            clean[0].zero_()
        b = dict(mix=mix.pin_memory(), clean=clean.pin_memory(),
                 meta=[{"clean_bucket": i == 1} for i in range(batch)],
                 ref_avail=torch.ones(batch, n if low_delay else n // 256 + 1).pin_memory())
        if cfg.get("model_cfg", {}).get("inputs") == "pr_nhat":
            b["n_hat"] = torch.zeros(batch, n).pin_memory()
        batches.append(b)

    def step(state, index, update=True, diagnostics=False):
        m, lf, fwd, gs, groups, opt, sched, ema = state
        inputs, target, fw, ic = train.prepare_batch(batches[index % len(batches)], cfg["model"], device, contract=contract)
        fe = getattr(lf, "fe", lf)
        fe.keep_live_terms = diagnostics and gs is None and fwd is m   # as train.py: compile donates backward buffers
        opt.zero_grad(set_to_none=gs is None)
        if gs is not None:
            loss = gs.step(m, inputs, target, ic)
        else:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.get("amp", True)):
                pred = fwd(*inputs)
            loss = lf(pred.float(), target, None, ic, scored=scored) if low_delay else lf(pred.float(), target, fw, ic)
            if diagnostics:
                term_grad_norms(lf, m.parameters())
            loss.backward()
        gn = torch.nn.utils.get_total_norm([p.grad for p in m.parameters() if p.grad is not None])
        if not bool(torch.isfinite(loss) & torch.isfinite(gn)):
            raise RuntimeError(f"nonfinite loss/gradient at step {index}")
        if update:
            train.clip_groups(groups, cfg["optim"].get("clip", 5.))
            opt.step(); sched.step()
            if ema:
                ema.update(m, index + 1)
        return loss.detach(), gn.detach()

    t0 = time.perf_counter()
    candidate = setup(variant)
    # Compare gradients at identical, repeatedly updated weights. This detects
    # stale captured products without mistaking long chaotic trajectories for
    # an implementation error. Each update still exercises optimizer + EMA.
    reference = setup({**VARIANTS["fft"], "filters": variant["filters"]})
    max_loss_rel = max_grad_rel = 0.
    for i in range(a.parity_steps):
        reference[0].load_state_dict(candidate[0].state_dict())
        want, _ = step(reference, i, update=False)
        got, _ = step(candidate, i, update=False)
        rel = float((got - want).abs() / want.abs().clamp(min=1e-8))
        max_loss_rel = max(max_loss_rel, rel)
        # graph replay is bit-exact; compile/fused reorder bf16-AMP arithmetic (measured ~3e-5..8e-5 relative)
        tol = 2e-4 if variant["compile"] or variant["gru_kernel"] == "fused" else 1e-5
        if rel > tol:
            raise AssertionError(f"loss parity {rel} > {tol} at step {i}")
        total = float(sum(g.grad.norm() ** 2 for g in reference[0].parameters() if g.grad is not None) ** .5)
        for (name, p), q in zip(reference[0].named_parameters(), candidate[0].parameters()):
            if (p.grad is None) != (q.grad is None):
                raise AssertionError(f"gradient presence differs: {name}")
            if p.grad is not None:
                err = float((p.grad - q.grad).norm() / p.grad.norm().clamp(min=1e-8))
                max_grad_rel = max(max_grad_rel, err)
                # below 1e-5 of the total norm a gradient is numerically zero and its direction is noise
                if err > 1e-4 and float((p.grad - q.grad).abs().max()) > 1e-7 and float(p.grad.norm()) > 1e-5 * total:
                    # reordering variants: cancellation-heavy sums (pe, a zero-init embedding summed over batch x
                    # tokens) drift 6-8% in norm; require the direction Adam follows, not bitwise equality
                    cos = float(torch.nn.functional.cosine_similarity(p.grad.flatten(), q.grad.flatten(), 0))
                    if tol == 1e-5 or err > .15 or cos < .99:
                        raise AssertionError(f"gradient parity {name} step {i}: {err} (cosine {cos}, |ref| {float(p.grad.norm()):.3g}, |total| {total:.3g})")
        train.clip_groups(candidate[4], cfg["optim"].get("clip", 5.))
        candidate[5].step(); candidate[6].step()
        if candidate[7]:
            candidate[7].update(candidate[0], i + 1)
    del reference
    torch.cuda.synchronize()
    startup = time.perf_counter() - t0
    for i in range(a.warmup):
        step(candidate, i)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    blocks = []
    for repeat in range(a.repeats):
        t0 = time.perf_counter()
        for i in range(a.steps):
            # Include the expensive per-term backward diagnostics at the normal
            # frequency, unless measuring a deliberately quieter profile.
            step(candidate, i + repeat * a.steps,
                 diagnostics=a.gradnorm_every > 0 and (i + 1 + repeat * a.steps) % a.gradnorm_every == 0)
        torch.cuda.synchronize()
        blocks.append(1000 * (time.perf_counter() - t0) / a.steps)
    return dict(status="passed", parity_passed=True, batch=batch, crop_s=crop_s,
                startup_s=startup, loss_relative_error=max_loss_rel, gradient_relative_error=max_grad_rel,
                block_ms=blocks, median_ms=statistics.median(blocks),
                items_per_s=1000 * batch / statistics.median(blocks),
                peak_allocated_gb=torch.cuda.max_memory_allocated() / 1e9,
                peak_reserved_gb=torch.cuda.max_memory_reserved() / 1e9,
                gpu=torch.cuda.get_device_name(), torch=str(torch.__version__), cuda=torch.version.cuda,
                platform=platform.platform(), config_sha256=__import__("hashlib").sha256(Path(a.config).read_bytes()).hexdigest())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", nargs="+", default=["configs/retraining/r8_fe_mini.yaml", "configs/retraining/r8_ld_fe_mini.yaml", "configs/retraining/r8_ld_fe_mini_armb.yaml", "configs/retraining/r8_ld_fe_mini_overparam.yaml"])
    ap.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    ap.add_argument("--thread-counts", nargs="+", type=int, default=[1, 2, 4])
    ap.add_argument("--batch", type=int, default=0, help="0 uses each config's actual batch")
    ap.add_argument("--crop-s", type=float, default=0, help="0 uses the config's actual crop")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--parity-steps", type=int, default=5)
    ap.add_argument("--gradnorm-every", type=int, default=20)
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--no-amp", action="store_true", help="fp32 parity diagnosis (config amp off)")
    ap.add_argument("--parity-out", type=Path, help="write a preflight parity record (pass only if every trial passed)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--config", help=argparse.SUPPRESS)
    ap.add_argument("--variant", help=argparse.SUPPRESS)
    ap.add_argument("--threads", type=int, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if min(a.steps, a.repeats, a.parity_steps, *a.thread_counts) < 1 or a.batch < 0 or a.crop_s < 0:
        ap.error("steps, repeats, parity steps and threads must be positive; batch/crop must be nonnegative")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.worker:
        try:
            row = trial(a)
        except Exception as e:
            row = dict(status="failed", reason=repr(e), traceback=traceback.format_exc())
        row.update(config=a.config, variant=a.variant, threads=a.threads, gradnorm_every=a.gradnorm_every)
        a.out.write_text(json.dumps(row, indent=2))
        return
    rows = []
    logs = a.out.parent / (a.out.stem + "_trials")
    logs.mkdir(exist_ok=True)
    for config in a.configs:
        for threads in a.thread_counts:
            for variant in a.variants:
                name = f"{Path(config).stem}_{variant}_t{threads}"
                out = logs / (name + ".json")
                out.unlink(missing_ok=True)
                cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", "--config", config,
                       "--variant", variant, "--threads", str(threads), "--out", str(out.resolve())]
                for key in ("batch", "crop_s", "steps", "warmup", "repeats", "parity_steps", "gradnorm_every"):
                    cmd += ["--" + key.replace("_", "-"), str(getattr(a, key))]
                cmd += ["--no-amp"] if a.no_amp else []
                env = dict(os.environ)
                for pool in ("OMP", "MKL", "OPENBLAS", "NUMBA", "NUMEXPR"):
                    env[f"{pool}_NUM_THREADS"] = str(threads)
                print(f"trial {name}", flush=True)
                try:
                    with open(logs / (name + ".log"), "w") as f:
                        p = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=a.timeout)
                    row = json.loads(out.read_text()) if out.exists() else dict(status="failed", reason=f"worker exited {p.returncode}")
                except subprocess.TimeoutExpired:
                    row = dict(status="failed", reason=f"timeout after {a.timeout}s")
                row.update(config=config, variant=variant, threads=threads)
                rows.append(row)
                report = dict(scope="synthetic compute, not end-to-end or a guarantee on other hardware",
                              arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}, rows=rows, winners=winners(rows))
                a.out.write_text(json.dumps(report, indent=2))
                print(f"  {row['status']}: {row.get('median_ms', row.get('reason'))}", flush=True)
    if a.parity_out:  # the preflight perf_parity record (r8_preflight.py): every trial must pass its parity check
        ok = bool(rows) and all(r.get("status") == "passed" and r.get("parity_passed") for r in rows)
        a.parity_out.parent.mkdir(parents=True, exist_ok=True)
        a.parity_out.write_text(json.dumps(dict(**{"pass": ok}, fp32=a.no_amp, variants=a.variants, rows=rows), indent=2))
        print(f"parity record {a.parity_out}: pass={ok}", flush=True)
        if not ok:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
