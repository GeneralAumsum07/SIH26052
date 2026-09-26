"""Every r8 config parses the way vaani.train.main parses it, the ablations differ from their parent in their arm only,
both full configs train the same data (bank, mixer, corpora) and the published bank_r8, and each recipe's dataset
object builds and draws items on synthetic manifests with the recipe's own file names. Nothing here trains."""
import glob, hashlib, json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch
import yaml

from vaani import losses, train
from vaani.data.dataset import DynamicMixDataset
from vaani.data.manifests import COLUMNS
from vaani.data.mixer import V2_DEFAULTS, MixConfig
from tests.test_scenes_r8 import recipe_noise_rows

REPO = Path(__file__).resolve().parents[1]
FULL = {"fe": "configs/retraining/r8_fe_mini.yaml", "refvalid": "configs/retraining/r8_refvalid_v2.yaml"}
ABL = sorted(Path(p).relative_to(REPO).as_posix() for p in glob.glob(str(REPO / "configs/retraining/r8_ablations/*.yaml")))
V2_CONFIGS = list(FULL.values()) + ABL
# the one field (or DSP block) each arm varies; name and epochs (48 = 15 % pilots) and seed are per file
ARM = {"ab1": set(), "ab2_p_": {"model_cfg.inputs"}, "ab2_pr_pld": {"model_cfg.inputs"},
       "ab2_pr_nhat": {"model_cfg.inputs", "dsp.blocking", "dsp.controller.block_margin_db",
                       "dsp.controller.diff_jump_max_db", "dsp.ref_policy.nlms"},
       "ab3_refdrop": {"data.ref_corrupt.p_absent"}, "ab3b": {"data.mix.v2.tail_share"},
       "ab4": {"model_cfg.mask", "model_cfg.df_taps"}, "ab6": {"loss_cfg.kappa"}}
NEW_CORPORA = ["demand_pairs", "avq_drone", "c3gd", "fsd50k", "lombard_grid"]


def _cfg(p):
    return yaml.safe_load(open(REPO / p, encoding="utf-8"))


def _flat(d, pre=""):
    out = {}
    for k, v in d.items():
        out.update(_flat(v, f"{pre}{k}.") if isinstance(v, dict) else {pre + k: v})
    return out


@pytest.mark.parametrize("path", V2_CONFIGS + ["configs/retraining/r8_mini_refvalid.yaml"])
def test_config_parses_like_the_trainer(path):
    cfg = _cfg(path)
    d = cfg["data"]
    MixConfig(**d.get("mix", {}))   # unknown v2 keys raise here, as in train.main
    lc = cfg.get("loss_cfg", {})
    if cfg["loss"] == "fe":
        loss = losses.build_loss("fe", lc)
        assert loss.w["pesq"] == pytest.approx(0.001), "torch_pesq missing: the PESQ term would be zero"
    else:
        losses.HybridLoss(**lc)
    model = train.build_model(cfg["model"], None, cfg.get("model_cfg"))   # init_from checked by its own test
    train.build_param_groups(model, cfg["optim"])
    train.needs_dsp(cfg)
    assert cfg.get("val", {}).get("select", "stoi") in ("stoi", "composite")
    for k in ("name", "seed", "epochs", "batch_size", "controller_on"):
        assert k in cfg, k


@pytest.mark.parametrize("path", V2_CONFIGS)
def test_v2_recipe_is_the_launch_recipe(path):
    cfg = _cfg(path)
    d = cfg["data"]
    assert d["mix"]["version"] == 2 and d["bank"] == "data/rirs/bank_r8.npz"
    assert d["exclude_groups_file"] == "configs/data/r8_heldout_exclude.json"
    assert cfg["val"]["select"] == "composite" and cfg["val"]["eval_root"] == "data/eval_r2"
    assert {Path(m).stem for m in d["manifests"]} >= set(NEW_CORPORA)
    assert "data/manifests/drone.parquet" not in d["manifests"]   # plan 11.5: dropped, AVQ replaces it
    if cfg["loss"] == "fe":
        assert cfg["loss_cfg"]["w_pesq"] == 0.001 and cfg["loss_cfg"]["pesq_required"] is True


def test_full_configs_train_the_g1_confirmed_mixer_on_the_same_data():
    fe, rv = _cfg(FULL["fe"]), _cfg(FULL["refvalid"])
    for k in ("bank", "manifests", "mix", "ref_corrupt", "exclude_groups_file", "crop_s", "epoch_len"):
        assert fe["data"][k] == rv["data"][k], k
    # tail_share 0.40 = the c5 default G1 confirmed on seeds 202 and 5150 (results_r2/r8/data_gates/README.md)
    assert fe["data"]["mix"]["v2"] == {"tail_share": V2_DEFAULTS["tail_share"]} and V2_DEFAULTS["tail_share"] == 0.4
    assert fe["epochs"] == rv["epochs"] == 320 and fe["batch_size"] == rv["batch_size"]


@pytest.mark.parametrize("path", ABL)
def test_ablation_differs_from_its_parent_in_its_arm_only(path):
    a = _flat(_cfg(path))
    par = _flat(_cfg(FULL["refvalid" if "refvalid" in path else "fe"]))
    diff = {k for k in set(a) | set(par) if a.get(k, "<absent>") != par.get(k, "<absent>")} - {"name", "epochs", "seed"}
    arm = next(v for k, v in ARM.items() if Path(path).stem.startswith(k))
    assert diff <= arm, (path, diff - arm)
    assert a["epochs"] == 48 and a["name"] == "r8" + Path(path).stem


def test_bank_is_the_published_r8_bank():
    t = json.loads((REPO / "configs/data/r8_banks.json").read_text(encoding="utf-8"))
    e = t["bank_r8"]
    assert e["file"] == _cfg(FULL["fe"])["data"]["bank"] and len(e["sha256"]) == 64 and len(e["sidecars"]) == 3
    f = REPO / e["file"]
    if not f.exists():
        pytest.skip(f"{e['file']} absent here (it is fetched onto the box by sha256)")
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    assert h.hexdigest() == e["sha256"]


def test_refvalid_init_is_pinned():
    cfg = _cfg(FULL["refvalid"])
    assert cfg["init_from"] == "results_r2/runs/r7_e256_wr64/best.pt" and len(cfg["init_sha256"]) == 64
    p = REPO / cfg["init_from"]
    if not p.exists():
        pytest.skip(f"{cfg['init_from']} absent here")
    train.verify_checkpoint_hash(p, cfg["init_sha256"])


# --- dry build of each recipe's dataset on synthetic manifests ---------------------------------------------------

SPEECH_FMT = {"librispeech_100h": ("librispeech", "ls:{i}", "ls-spk-{i}"), "ears": ("ears", "ears:p9{i}/x", "ears-spk-p9{i}"),
              "cv_hi": ("cv_hi", "cvhi:{i}", "cv-spk-{i}"), "lombard_grid": ("lombard_grid", "lgrid:s{i}/l/s{i}_l_bbaf2n", "lgrid-spk-s{i}")}
# manifest stem -> corpus column, for the noise manifests whose file name is not the corpus
NOISE_CORPUS = {"mad_v2": "mad", "demand_pairs": "demand", "dns_datasets_fullband.noise_fullband.freesound_000.tar": "dns_noise"}


def _write(root, name, i, x):
    p = root / name / f"{i}.flac"; p.parent.mkdir(parents=True, exist_ok=True); sf.write(p, x, 16000)
    return p.as_posix(), len(x) / 16000


def _rows(name, root, rng):
    """Speech: two train rows and one val row per corpus. Noise: every source_id format the scene test proves fills
    each scene role (tests/test_scenes_r8.py), on train, plus one val row. Audio is shaped noise or a short burst."""
    base = dict(licence="t", sha1="")
    if name in SPEECH_FMT:
        corpus, sid, grp = SPEECH_FMT[name]; out = []
        for i, split in enumerate(["train", "train", "val"]):
            path, dur = _write(root, name, i, (rng.normal(0, 0.05, 80000) * np.hanning(80000)).astype(np.float32))
            out.append(dict(base, source_id=sid.format(i=i), corpus=corpus, kind="speech", group_id=grp.format(i=i),
                            speaker_id=str(i), path=path, duration_s=dur, split=split, noise_class=""))
        return out
    corpus = NOISE_CORPUS.get(name, name)
    rows = [r for r in recipe_noise_rows().to_dict("records") if r["corpus"] == corpus]
    assert rows, name
    rows.append(dict(rows[0], source_id=rows[0]["source_id"] + "v", group_id="val-" + name))
    out = []
    for i, r in enumerate(rows):
        if r["noise_class"] == "impulsive":
            x = np.zeros(32000, np.float32); x[800:900] = rng.normal(0, 0.5, 100)
        else:
            x = rng.normal(0, 0.05, 80000).astype(np.float32)
            x = np.stack([x, np.roll(x, 3)], 1) if corpus == "demand" else x
        path, dur = _write(root, name, i, x)
        out.append(dict(base, **r, speaker_id="", path=path, duration_s=dur, split="val" if i == len(rows) - 1 else "train"))
    return out


@pytest.mark.timeout(600)
@pytest.mark.parametrize("path", [FULL["fe"], FULL["refvalid"], "configs/retraining/r8_ablations/ab2_pr_nhat_s0.yaml"])
def test_recipe_dataset_builds_and_draws_on_synthetic_manifests(path, tmp_path):
    cfg = _cfg(path); d = cfg["data"]; rng = np.random.default_rng(0)
    mans = []
    for m in d["manifests"]:
        p = tmp_path / "manifests" / Path(m).name; p.parent.mkdir(exist_ok=True)
        pd.DataFrame(_rows(Path(m).stem, tmp_path / "raw", rng), columns=COLUMNS).to_parquet(p)
        mans.append(str(p))
    dsk = dict(with_dsp=train.needs_dsp(cfg), controller_on=cfg["controller_on"], dsp_cfg=cfg.get("dsp"),
               pack_root=None, ref_corrupt=d.get("ref_corrupt"), exclude_groups_file=str(REPO / d["exclude_groups_file"]))
    if cfg["model"] == "vaani_fe":
        dsk["fe_inputs"] = True
    ds = DynamicMixDataset(mans, "train", None, MixConfig(**d["mix"]), d["crop_s"], 4, cfg["seed"], **dsk)
    tags = set()
    for i in range(4):
        it = ds[i]
        assert torch.isfinite(it["mix"]).all() and torch.isfinite(it["clean"]).all()
        assert it["clean"].shape[-1] == int(d["crop_s"] * 16000)
        assert ("n_hat" in it) == dsk["with_dsp"]
        assert ("ref_avail" in it) == (d.get("ref_corrupt") is not None or ("fe_inputs" in dsk and not dsk["with_dsp"]))
        tags |= {s.get("tag") for s in it["meta"].get("noise_sources", [])}
    assert tags and "fallback" not in tags, tags
