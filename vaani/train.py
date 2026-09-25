"""Config-driven trainer. One YAML == one ablation row.

Inputs per model:
  gtcrn : spec (B,257,T,2) of the primary channel only
  vaani : spec6 (B,257,T,6) [prim, ref, n_hat] + feats (B,T,18)
  vaani_fe : spec (B,257,T,4|6) [prim, ref(, n_hat)] + validity (B,T); no DSP features (the
             limiter/ref_policy front end runs in the workers; the NLMS only for inputs pr_nhat)
The DSP pipeline (NLMS + features + controller) runs on CPU inside the
DataLoader workers via DynamicMixDataset(with_dsp=True); prepare_batch only
moves tensors to the device and takes STFTs there.

r8 options, all off by default (r7 behaviour bit-exact): loss "fe", an EMA shadow of the weights
(`ema: {decay}`), and checkpoint selection `val.select: stoi | composite` (plan 11.2/11.6).
"""
import argparse, copy, hashlib, importlib.util, json, os, subprocess, time
from pathlib import Path

import numpy as np, torch, yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pystoi import stoi

from vaani import losses
from vaani.data import manifests
from vaani.data.dataset import DynamicMixDataset, EpochSampler, RenderedDataset, collate, front_end
from vaani.data.mixer import MixConfig
from vaani.dsp import pipeline, stft
from vaani.models import vaani_fe
from vaani.models.gtcrn import GTCRN
from vaani.models.vaani_net import VaaniNet
from vaani.training_controls import cosine_lr_multiplier, make_schedule_config, validate_resume_schedule, should_stop_for_patience, verify_checkpoint_hash
from vaani.training_controls import composite_key, composite_summary, ema_decay
from vaani import runtime

SR = 16000
ROOT = Path(__file__).resolve().parents[1]  # repo root, so configs work from any cwd
MAX_BAD_STEPS = 20


def _abs(p):
    return None if p is None else (Path(p) if Path(p).is_absolute() else ROOT / p)


def build_model(name, init_from=None, model_cfg=None):
    init_from = _abs(init_from); model_cfg = model_cfg or {}
    if name == "gtcrn":
        m = GTCRN()
        if init_from:
            m.load_state_dict(torch.load(init_from, map_location="cpu", weights_only=True)["model"])
        return m
    if name == "vaani":
        if init_from and init_from.suffix == ".pt":
            ck = torch.load(init_from, map_location="cpu", weights_only=True)
            if ck.get("config", {}).get("model") == "vaani":
                # already-trained vaani run, not a gtcrn seed; warm_start tolerates a narrower source architecture
                return VaaniNet(**model_cfg).warm_start(ck["model"])
        return VaaniNet.from_pretrained_gtcrn(init_from, **model_cfg) if init_from else VaaniNet(**model_cfg)
    if name == "vaani_fe":
        # model_cfg = {tier: mini|mid|large|large_plus, **VaaniFE overrides}; self-contained so export.fe_load rebuilds it
        m = vaani_fe.from_arch(model_cfg)
        if init_from:
            m.load_state_dict(torch.load(init_from, map_location="cpu", weights_only=True)["model"])
        return m
    raise ValueError(name)


def needs_dsp(cfg):
    """The NLMS/feature pipeline runs in the workers only when the model reads n_hat or the 18 features."""
    return cfg["model"] == "vaani" or (cfg["model"] == "vaani_fe" and (cfg.get("model_cfg") or {}).get("inputs") == "pr_nhat")


def frame_weights_from_meta(metas, n_frames, burst_weight=3.0, half_window_s=0.15):
    """(B,T) loss weights: burst_weight on frames within +-half_window_s of an impulse onset."""
    w = torch.ones(len(metas), n_frames)
    hw = int(half_window_s * SR / stft.HOP)
    for b, m in enumerate(metas):
        for on in m.get("impulse_onsets_s", []):
            k = int(on * SR / stft.HOP)
            w[b, max(0, k - hw): k + hw] = burst_weight
    return w


def prepare_batch(batch, model_name, device, burst_weight=1.0):
    """Batch (CPU, from collate) -> (model inputs, target spec, frame weights, is_clean) on device."""
    nb = dict(non_blocking=True)   # pinned host buffers: overlap the copy with compute
    mix, clean, metas = batch["mix"].to(device, **nb), batch["clean"].to(device, **nb), batch["meta"]
    target = stft.stft(clean)  # STFTs on device: cheaper than CPU + transfer of the wider spec
    fw = frame_weights_from_meta(metas, target.shape[2], burst_weight)
    is_clean = torch.tensor([bool(m.get("clean_bucket", False)) for m in metas])
    if model_name == "gtcrn":
        inputs = (stft.stft(mix[:, 0]),)
    elif model_name == "vaani_fe":
        # raw P, R (and n_hat for the pr_nhat arm) STFTs; the model compresses and gates R by validity itself
        chans = [stft.stft(mix[:, 0]), stft.stft(mix[:, 1])]
        if "n_hat" in batch:
            chans.append(stft.stft(batch["n_hat"].to(device, **nb)))
        inputs = (torch.cat(chans, dim=-1), None, batch["ref_avail"].to(device, **nb) if "ref_avail" in batch else None)
    else:
        # n_hat/feats were computed in the dataset workers, which own controller_on
        n_hat = batch["n_hat"].to(device, **nb)
        spec6 = torch.cat([stft.stft(mix[:, 0]), stft.stft(mix[:, 1]), stft.stft(n_hat)], dim=-1)
        inputs = (spec6, batch["feats"].to(device, **nb))
        if "ref_avail" in batch:   # data.ref_corrupt: the capture-path availability label rides along as a third input
            inputs = (*inputs, batch["ref_avail"].to(device, **nb))
    return inputs, target, fw.to(device), is_clean.to(device)


def build_param_groups(model, optim_cfg):
    """AdamW groups. FiLM projections and the zero-init ref_validity conv get lr_new. With lr_df set, the deep-filter head gets its own
    group (lr_df, clipped alone at clip_df): r3 showed its tap gradient is heavy-tailed (per-batch norm
    3..120 on real batches), so Adam's second moment pinned the taps near zero at the shared lr."""
    lr = optim_cfg["lr"]
    is_df = lambda n: n.startswith("df.") and "lr_df" in optim_cfg
    is_new = lambda n: "film" in n or "ref_conv" in n
    groups = [{"params": [p for n, p in model.named_parameters() if not is_new(n) and not is_df(n)], "lr": lr}]
    film = [p for n, p in model.named_parameters() if is_new(n)]
    if film:
        groups.append({"params": film, "lr": optim_cfg.get("lr_new", lr)})
    df = [p for n, p in model.named_parameters() if is_df(n)]
    if df:
        groups.append({"params": df, "lr": optim_cfg["lr_df"], "clip": optim_cfg.get("clip_df")})
    return groups


def clip_groups(groups, clip):
    """Groups carrying their own clip are normalised separately, so a burst batch that blows up the tap
    gradient no longer drags every other parameter's update down with it."""
    own = [g for g in groups if g.get("clip")]
    for g in own:
        torch.nn.utils.clip_grad_norm_(g["params"], g["clip"])
    torch.nn.utils.clip_grad_norm_([p for g in groups if g not in own for p in g["params"]], clip)


def _git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, cwd=ROOT).strip()
    except Exception:
        return "nogit"


def _save(state, path):
    tmp = path.with_suffix(".tmp")  # write-then-rename so a crash never leaves a truncated .pt
    torch.save(state, tmp); os.replace(tmp, path)


@torch.no_grad()
def validate(model, dl, cfg, device):
    if cfg.get("val", {}).get("eval_root"):
        # A shared frozen validation screen makes sampling/loss variants
        # comparable; the old per-run dynamic screen remains the default.
        from vaani.train_refiner import screen_items, score_items
        if not hasattr(dl, "_frozen_screen"):
            dl._frozen_screen = screen_items(cfg["val"]["eval_root"], cfg["val"].get("split", "val"))
        ds, indices = dl._frozen_screen
        values = score_items(model, ds, indices, cfg, device).mean(0)
        dl._last_val_metrics = dict(zip(("snr_out", "stoi", "pesq_wb"), map(float, values)))
        return float(values[1])
    model.eval(); scores = []
    for batch in dl:
        inputs, target, _, _ = prepare_batch(batch, cfg["model"], device)
        pred = model(*inputs).float()
        y = stft.istft(pred, length=batch["clean"].shape[-1]).cpu().numpy()
        for b in range(y.shape[0]):
            scores.append(stoi(batch["clean"][b].numpy(), y[b], SR, extended=False))
    model.train(); return float(np.mean(scores))


class EMA:
    """Shadow weights (plan 11.6): lerp toward the live weights every optimiser step; buffers (BN stats) are copied.
    Scored at every val point next to the raw weights; cheaper than r7's warm restart."""

    def __init__(self, model, decay, warmup=True):
        self.model = copy.deepcopy(model).eval(); self.decay, self.warmup = float(decay), bool(warmup)
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model, step):
        d = ema_decay(self.decay, step, self.warmup)
        torch._foreach_lerp_(list(self.model.parameters()), [p.detach() for p in model.parameters()], 1.0 - d)
        for b, mb in zip(self.model.buffers(), model.buffers()):
            b.copy_(mb)


COMPOSITE_DEFAULTS = dict(
    every=1,                 # val points between composite screens (slow: DSP once, then one forward per clip x condition)
    per_bucket=2, limit=None,   # eval_refvalid.select over val buckets, then the first `limit` clips
    targets=None, ild_max_loss=0.06, margin_db=1.0, include_clean=False,
    baseline="gtcrn_pretrained",   # dSNR reference for the mono / web-stereo filters (None: filters skipped)
    ilds=(-14, -10, -8, -6, -4, -2, 0),
    web_delay=5, web_gain_db=-1.0,   # web-stereo construction: ref = primary 0.31 ms late at -1 dB (our approximation)
)


def _eval_refvalid():
    spec = importlib.util.spec_from_file_location("eval_refvalid", ROOT / "scripts" / "eval_refvalid.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


class CompositeScreen:
    """Composite val selection on a small VAL subset with G3's constructions: nominal (present), mono-duplicated,
    web-stereo and the far-field ILD sweep (scripts/eval_refvalid.py conditions and speech-loss definition).
    Front ends do not depend on the weights, so they are computed once; each score() is one forward per clip x cond."""

    def __init__(self, cfg, run_dir=None):
        from vaani import metrics
        vc = cfg.get("val", {}); c = {**COMPOSITE_DEFAULTS, **(vc.get("composite") or {})}
        if vc.get("split", "val") != "val":
            raise ValueError("composite selection uses val only, never test")
        self.c, self.cfg, self.metrics, self.er = c, cfg, metrics, _eval_refvalid()
        root = _abs(vc.get("eval_root", "data/eval_r2")) / "val"
        idx = self.er.select(root, c["per_bucket"])[: c["limit"]]
        ds = RenderedDataset(root)
        self.conds = ("present", "mono", "web_stereo", *(f"ild_{d}" for d in c["ilds"]))
        dsp, pol = needs_dsp(cfg), (cfg.get("dsp") or {}).get("ref_policy") is not None
        self.items, present = [], []
        for i in idx:
            it = ds[i]; mix, clean = it["mix"].numpy(), it["clean"].numpy(); ci = bool(it["meta"].get("clean_bucket"))
            for cond in self.conds:
                m2, avail = self.construct(cond, mix, clean)
                if cfg["model"] == "gtcrn":
                    fe = dict(mix=m2)
                elif dsp:
                    r = pipeline.run(m2, controller_on=cfg["controller_on"], dsp_cfg=cfg.get("dsp"), ref_avail=avail if pol else None)
                    fe = dict(mix=r["mix"], n_hat=r["n_hat"], feats=r["features"])
                else:
                    m3, fa = front_end(m2, cfg.get("dsp"), avail); fe = dict(mix=m3, ref_avail=fa)
                self.items.append(dict(cond=cond, clean=clean, clean_item=ci, prim=m2[0],
                                       snr_in=metrics.snr_db(clean, m2[0]), **fe))
            if not ci:
                present.append((mix, clean))
        self.base_d_snr = self._baseline(present, run_dir, [str(ds.items[i]) for i in idx])

    def construct(self, cond, mix, clean):
        if cond == "mono":
            return np.stack([mix[0], mix[0]]).astype(np.float32), None
        if cond == "web_stereo":
            r = self.er._shift(mix[0], int(self.c["web_delay"])) * np.float32(10 ** (self.c["web_gain_db"] / 20))
            return np.stack([mix[0], r]).astype(np.float32), None
        m2, avail, _ = self.er.apply_condition(cond, mix, clean)
        return m2, avail

    def _baseline(self, present, run_dir, ids):
        """Mean dSNR of the baseline on the nominal clips; the baseline is mono, so it is also its mono/web dSNR."""
        name = self.c["baseline"]
        if not name:
            return None
        key = hashlib.sha1(json.dumps([name, ids]).encode()).hexdigest()[:12]
        cache = Path(run_dir) / "composite_baseline.json" if run_dir else None
        if cache and cache.exists():
            j = json.loads(cache.read_text())
            if j.get("key") == key:
                return j["d_snr"]
        try:
            from vaani.models import baselines
            b = baselines.get(name)
            d = float(np.mean([self.metrics.snr_db(c, b.enhance(m)) - self.metrics.snr_db(c, m[0]) for m, c in present]))
        except Exception as e:   # a missing baseline checkpoint must not stop training; the filters then skip
            print(f"WARNING: composite baseline {name!r} unavailable ({e!r}); mono/web filters skipped", flush=True)
            return None
        if cache:
            cache.write_text(json.dumps(dict(key=key, baseline=name, d_snr=d, n=len(present))))
        return d

    @torch.no_grad()
    def enhance(self, model, it, device):
        x = torch.from_numpy(it["mix"])[None].to(device); n = x.shape[-1]
        if self.cfg["model"] == "gtcrn":
            out = model(stft.stft(x[:, 0]))
        elif self.cfg["model"] == "vaani_fe":
            chans = [stft.stft(x[:, 0]), stft.stft(x[:, 1])]
            if "n_hat" in it:
                chans.append(stft.stft(torch.from_numpy(it["n_hat"])[None].to(device)))
            fa = torch.from_numpy(it["ref_avail"])[None].to(device) if "ref_avail" in it else None
            out = model(torch.cat(chans, -1), None, fa)
        else:
            spec6 = torch.cat([stft.stft(x[:, 0]), stft.stft(x[:, 1]), stft.stft(torch.from_numpy(it["n_hat"])[None].to(device))], -1)
            out = model(spec6, torch.from_numpy(it["feats"])[None].to(device))
        return stft.istft(out.float(), length=n)[0].cpu().numpy()

    def score(self, model, device):
        was = model.training; model.eval(); rows = []
        for it in self.items:
            y = self.enhance(model, it, device); m = self.metrics
            r = dict(cond=it["cond"], clean_item=it["clean_item"], snr_in=it["snr_in"], snr_out=m.snr_db(it["clean"], y),
                     speech_loss=self.er.frame_stats(it["clean"], y, it["prim"])["speech_loss"], stoi=None, pesq=None)
            if it["cond"] == "present":
                r["stoi"], r["pesq"] = m.stoi(it["clean"], y), m.pesq_wb(it["clean"], y)
            rows.append(r)
        model.train(was)
        c = self.c
        return composite_summary(rows, c["targets"], c["ild_max_loss"], c["margin_db"], self.base_d_snr, c["include_clean"])


def main(config_path):
    cfg = yaml.safe_load(open(config_path))
    verify_checkpoint_hash(_abs(cfg.get("init_from")), cfg.get("init_sha256"))
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    device = torch.device(cfg.get("device", "cuda"))
    run_dir = Path(cfg.get("runs_dir", "runs")) / cfg["name"]; run_dir.mkdir(parents=True, exist_ok=True)
    tb = SummaryWriter(run_dir)
    t_start = time.time()

    d = cfg["data"]; mixcfg = MixConfig(**d.get("mix", {}))
    with_dsp = needs_dsp(cfg)  # gtcrn and VaaniFE (bar pr_nhat) never need n_hat/feats, skip the 150 ms/clip
    dsk = dict(with_dsp=with_dsp, controller_on=cfg["controller_on"], dsp_cfg=cfg.get("dsp"),
               pack_root=d.get("pack", "data/pack"), ref_corrupt=d.get("ref_corrupt"))
    if cfg["model"] == "vaani_fe":
        dsk["fe_inputs"] = True
    for k in ("exclude_groups_file", "scene_weights"):   # r8 keys; absent = the dataset's r7 behaviour
        if k in d:
            dsk[k] = str(_abs(d[k])) if k == "exclude_groups_file" and d[k] else d[k]
    ds = DynamicMixDataset(d["manifests"], "train", d.get("bank"), mixcfg, d.get("crop_s", 4.0),
                           d.get("epoch_len", 20000), cfg["seed"], **dsk)
    vds = DynamicMixDataset(d["manifests"], "val", d.get("bank"), mixcfg, d.get("crop_s", 4.0),
                            cfg.get("val", {}).get("dynamic_items", 200), cfg["seed"] + 1, **dsk)
    # "auto" sizes from the box's core count; $VAANI_WORKERS overrides without editing configs
    nw = runtime.resolve_workers(cfg.get("num_workers", "auto"), share=cfg.get("concurrent_runs", 1))
    runtime.tune_backends(device)
    lk = runtime.loader_kwargs(nw, device)
    # Windows spawns workers: persistent_workers avoids re-importing numba/JIT every epoch;
    # the epoch therefore travels in the sampler's indices, not in dataset attributes
    sampler = EpochSampler(len(ds))
    dl = DataLoader(ds, cfg["batch_size"], sampler=sampler, collate_fn=collate, **lk)
    vdl = DataLoader(vds, cfg["batch_size"], collate_fn=collate, **lk)
    print(f"runtime: {runtime.describe()} num_workers={nw}", flush=True)

    model = build_model(cfg["model"], cfg.get("init_from"), cfg.get("model_cfg")).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    groups = build_param_groups(model, cfg["optim"])
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    total = cfg["max_steps"] if cfg.get("max_steps") else cfg["epochs"] * len(dl)
    warm = cfg["optim"].get("warmup", 500)
    # float(): a numpy scalar in the scheduler state would break weights_only resume
    schedule = make_schedule_config(cfg["epochs"], len(dl), warm, cfg.get("max_steps"))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: cosine_lr_multiplier(s, warm, total))
    sp = cfg["loss"] == "speech_preservation"
    losscfg = cfg.get("loss_cfg", {})  # w_complex / w_mag / p / w_snr; absent = upstream loss verbatim
    fe_loss = cfg["loss"] == "fe"
    if fe_loss:
        loss_fn = losses.build_loss("fe", losscfg)
    else:
        loss_fn = losses.SpeechPreservationLoss(**losscfg) if sp else losses.HybridLoss(**losscfg)
    burst_w = loss_fn.burst_weight if sp else 1.0
    use_amp = bool(cfg.get("amp", True)) and device.type == "cuda"
    ecfg = cfg.get("ema") or {}
    ema = EMA(model, ecfg["decay"], ecfg.get("warmup", True)) if ecfg.get("decay") else None
    select = cfg.get("val", {}).get("select", "stoi")
    if select not in ("stoi", "composite"):
        raise ValueError(f"val.select must be stoi or composite, got {select!r}")
    plain_best = ema is None and select == "stoi"   # r1-r7 checkpoint layout, byte for byte
    screen, best_key = None, None

    step, best, start_epoch, history = 0, -1.0, 0, []
    last = run_dir / "last.pt"
    if cfg.get("resume", True) and last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=True)
        old = ck.get("config", {})
        # Resuming restores optimizer time; changing the budget or recipe here
        # would silently change the experiment. A new run/init_from is a restart.
        for key in ("model", "model_cfg", "data", "loss", "loss_cfg", "optim", "batch_size", "seed", "dsp", "controller_on", "val", "ema"):
            if old.get(key) != cfg.get(key):
                raise RuntimeError(f"Resume configuration changed: {key}; start a new run")
        saved_schedule = ck.get("schedule") or make_schedule_config(old["epochs"], len(dl), old["optim"].get("warmup", 500), old.get("max_steps"))
        validate_resume_schedule(saved_schedule, schedule)
        model.load_state_dict(ck["model"]); step, best = ck["step"], ck.get("best", best)
        history = ck.get("history", [])
        start_epoch = ck.get("epoch", -1) + 1
        if "optim" in ck:
            opt.load_state_dict(ck["optim"]); sched.load_state_dict(ck["sched"])
        if ema is not None:
            ema.model.load_state_dict(ck["ema"])
        if ck.get("best_key") is not None:
            best_key = tuple(ck["best_key"])
        print(f"resumed {last} at step {step}, epoch {start_epoch}")

    vc = cfg.get("val", {})
    evalset_hash = _abs(vc.get("eval_root", "data/eval")) / vc.get("split", "val") / "EVALSET_HASH"
    run_info = dict(name=cfg["name"], config=cfg, git_sha=_git_hash(),
                    config_hash=hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12],
                    manifest_hash=manifests.content_hash(d["manifests"]),
                    evalset_hash=evalset_hash.read_text().strip() if evalset_hash.exists() else "none",
                    init_from=str(_abs(cfg.get("init_from"))) if cfg.get("init_from") else None,
                    params=n_params, seed=cfg["seed"], torch=torch.__version__, cuda=torch.version.cuda,
                    amp=use_amp, start=t_start, best_metric="composite_val" if select == "composite" else "stoi_frozen_val_screen" if vc.get("eval_root") else "stoi_dynamic_val", best_val_stoi=best,
                    steps=step, wall_s=0.0, skipped_steps=0)
    run_info.update(schedule=schedule, history=history)
    json.dump(run_info, open(run_dir / "run.json", "w"), indent=2)

    bad, skipped = 0, 0
    for epoch in range(start_epoch, cfg["epochs"]):
        stop_cfg = cfg.get("early_stopping") or {}
        if should_stop_for_patience(history, stop_cfg.get("patience"), stop_cfg.get("min_delta", 0.)):
            break
        sampler.set_epoch(epoch)
        clamp_sum, clamp_batches = 0., 0
        for batch in dl:
            inputs, target, fw, is_clean = prepare_batch(batch, cfg["model"], device, burst_w)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                pred = model(*inputs)
            if fe_loss:   # the phase term's speech-dominance mask needs the noisy primary spectrum
                noisy = inputs[0][..., :2] if loss_fn.w["phase"] and cfg["model"] != "gtcrn" else None
                loss = loss_fn(pred.float(), target, fw, is_clean, noisy=noisy)
            else:
                loss = loss_fn(pred.float(), target, fw, is_clean)  # loss/iSTFT stay fp32
            if loss_fn.w_snr:
                clamp_sum += float(loss_fn.last_snr_clamp_fraction); clamp_batches += 1
            if not torch.isfinite(loss):
                bad += 1; skipped += 1
                if bad >= MAX_BAD_STEPS:
                    raise RuntimeError(f"{bad} consecutive non-finite losses at step {step}")
                continue
            bad = 0
            opt.zero_grad(set_to_none=True); loss.backward()
            clip_groups(groups, cfg["optim"].get("clip", 5.0))
            opt.step(); sched.step(); step += 1
            if ema is not None:
                ema.update(model, step)
            if step % 20 == 0:
                tb.add_scalar("train/loss", loss.item(), step); tb.add_scalar("train/lr", sched.get_last_lr()[0], step)
                if fe_loss:
                    for k, t in loss_fn.last_terms.items():
                        tb.add_scalar(f"train/loss_{k}", float(t), step)
            if cfg.get("max_steps") and step >= cfg["max_steps"]:
                break
        v = validate(model, vdl, cfg, device); tb.add_scalar("val/stoi", v, step)
        history.append(dict(epoch=epoch, step=step, val_stoi=v, lr=sched.get_last_lr()[0],
                            snr_clamp_fraction=clamp_sum / max(clamp_batches, 1), val_metrics=getattr(vdl, "_last_val_metrics", {"stoi": v})))
        tb.add_scalar("train/snr_clamp_fraction", history[-1]["snr_clamp_fraction"], step)
        cands = {"raw": model}
        if ema is not None:   # the shadow is scored at every val point
            ve = validate(ema.model, vdl, cfg, device); tb.add_scalar("val/stoi_ema", ve, step)
            history[-1].update(val_stoi_ema=ve, val_metrics_ema=getattr(vdl, "_last_val_metrics", {"stoi": ve}))
            cands["ema"] = ema.model
        if plain_best:
            if v > best:
                best = v; _save({"model": model.state_dict(), "config": cfg, "step": step}, run_dir / "best.pt")
        elif select == "stoi":
            scores = {"raw": v, "ema": history[-1]["val_stoi_ema"]}
            pick = max(scores, key=scores.get)   # raw wins ties
            if scores[pick] > best:
                best = scores[pick]
                _save({"model": cands[pick].state_dict(), "config": cfg, "step": step, "weights": pick, "selection": "stoi"}, run_dir / "best.pt")
        else:
            final = epoch == cfg["epochs"] - 1 or bool(cfg.get("max_steps") and step >= cfg["max_steps"])
            every = int((cfg["val"].get("composite") or {}).get("every", COMPOSITE_DEFAULTS["every"]))
            if len(history) % every == 0 or final:
                if screen is None:
                    t0 = time.time(); screen = CompositeScreen(cfg, run_dir)
                    print(f"composite screen: {len(screen.items)} clip-conditions, baseline dSNR {screen.base_d_snr}, "
                          f"built in {time.time() - t0:.1f}s", flush=True)
                for name, m in cands.items():
                    s = screen.score(m, device); history[-1][f"composite_{name}"] = s; key = composite_key(s)
                    tb.add_scalar(f"val/pass_rate_{name}", s["pass_rate"], step)
                    if best_key is None or key > best_key:
                        best_key, best = key, s["pass_rate"]
                        _save({"model": m.state_dict(), "config": cfg, "step": step, "weights": name,
                               "selection": "composite", "composite": s}, run_dir / "best.pt")
        if cfg.get("save_every_epoch"):   # learning-curve pilots score intermediate epochs; off = r1..r7 behaviour
            _save({"model": model.state_dict(), "config": cfg, "step": step, "epoch": epoch}, run_dir / f"epoch{epoch:03d}.pt")
        state = {"model": model.state_dict(), "config": cfg, "step": step, "epoch": epoch, "best": best,
                 "optim": opt.state_dict(), "sched": sched.state_dict(), "schedule": schedule, "history": history}
        if not plain_best:
            state.update(best_key=list(best_key) if best_key is not None else None,
                         **({"ema": ema.model.state_dict()} if ema is not None else {}))
        _save(state, last)
        run_info.update(history=history, best_val_stoi=best, steps=step)
        if best_key is not None:
            run_info["best_key"] = list(best_key)
        json.dump(run_info, open(run_dir / "run.json", "w"), indent=2)
        print(f"epoch {epoch} step {step} val_stoi {v:.4f} best {best:.4f} skipped {skipped}")
        if cfg.get("max_steps") and step >= cfg["max_steps"]:
            break
    # tap-weight norm: a null df result must be diagnosable (untrained taps) rather than believed
    df_norm = float(sum(p.detach().norm() ** 2 for n, p in model.named_parameters() if n.startswith("df.")) ** 0.5)
    run_info.update(end=time.time(), wall_s=time.time() - t_start, best_val_stoi=best, steps=step, skipped_steps=skipped, df_norm=df_norm)
    json.dump(run_info, open(run_dir / "run.json", "w"), indent=2)
    tb.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("config"); main(ap.parse_args().config)
