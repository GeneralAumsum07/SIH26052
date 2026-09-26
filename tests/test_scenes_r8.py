"""r8 scene roles for every noise corpus a scanner emits (plan 11.5 / M8): each corpus maps to scene tags, the r8 recipe's
corpora cover every scene's bed and point roles, and DEMAND pairs keep the one-pair manifest's splits."""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from vaani.data import manifests, scenes
from vaani.data.splits import assign
from vaani.data.scenes import SCENES, ScenePool, noise_tags

REPO = Path(__file__).resolve().parents[1]
DEMAND_ENVS = ["DKITCHEN", "DLIVING", "DWASHING", "NFIELD", "NPARK", "NRIVER", "OHALLWAY", "OMEETING", "OOFFICE",
               "PCAFETER", "PRESTO", "PSTATION", "SCAFE", "SPSQUARE", "STRAFFIC", "TBUS", "TCAR", "TMETRO"]


def _scanner_corpora():
    t = (REPO / "vaani/data/sources.py").read_text(encoding="utf-8")
    return set(re.findall(r'_row\([^,]+,\s*"([a-z_0-9]+)",\s*"(noise|speech|rir)"', t))


def _row(sid, corpus, group, ncls="stationary"):
    return dict(source_id=sid, corpus=corpus, kind="noise", group_id=group, noise_class=ncls)


def recipe_noise_rows():
    """One row per source_id format of every noise corpus the r8 recipes train on (formats from the scanners and the
    laptop manifests), several groups where the scanner groups finely."""
    r = []
    for cls in ("shooting", "shelling", "footsteps", "vehicle", "helicopter", "fighter"):
        r.append(_row(f"mad:{cls}/001_1", "mad", f"mad-yt-{cls}"))
    for cat in ("wind", "rain", "helicopter", "engine", "siren", "footsteps", "chainsaw", "keyboard_typing"):
        r.append(_row(f"esc50:{cat}/1-1-A-0", "esc50", f"esc50-{cat}"))
    r.append(_row("esc50:can_opening/1-2-A-34", "esc50", "esc50-can", "impulsive"))
    r.append(_row("gun:glock/x_v0", "gunshots", "gun-x", "impulsive"))
    r.append(_row("dnsn:door_Freesound_validated_1_6", "dns_noise", "dnsn-door_1"))
    for env in DEMAND_ENVS:
        r.append(_row(f"demand:{env}/ch01-09", "demand", f"demand-{env}"))
    r.append(_row("avq_drone:noises-train-drones/n116", "avq_drone", "avq-n116"))
    r.append(_row("c3gd:3-E1-P1/3-E1-P1-M1-F1-C1", "c3gd", "c3gd-E1-P1", "impulsive"))
    for lab in scenes.FSD50K_TAGS:
        imp = "impulsive" if lab in ("Gunshot_and_gunfire", "Explosion") else "stationary"
        r.append(_row(f"fsd50k:{lab}/1", "fsd50k", f"fsd50k-up-{lab}", imp))
    return pd.DataFrame(r)


def test_every_scanner_noise_corpus_has_a_declared_scene_role():
    noise = {c for c, k in _scanner_corpora() if k == "noise"}
    assert noise, "no scanner rows found: the regex no longer matches vaani/data/sources.py"
    assert noise <= set(scenes.NOISE_CORPORA), f"scanner corpora without a scene role: {noise - set(scenes.NOISE_CORPORA)}"


def test_declared_roles_match_noise_tags():
    sid = {"mad": "mad:helicopter/1_1", "gunshots": "gun:a/b", "cadre": "cadre:a/b", "c3gd": "c3gd:g/s",
           "demand": "demand:NPARK/ch01-09", "esc50": "esc50:siren/1-1-A-0", "avq_drone": "avq_drone:t/n116",
           "fsd50k": "fsd50k:Siren/1", "wham": "wham:tr:011a0101_0.061_20dc0109_-0.061", "dns_noise": "dnsn:x",
           "musan": "musan:free-sound/noise-free-sound-0000", "noisex92": "noisex92:leopard", "vehicle_interior":
           "vehicle_interior:car", "drone": "drone:yes_drone/B_S2_D1_067-bebop_000_"}
    for corpus, specific in scenes.NOISE_CORPORA.items():
        t = noise_tags(dict(corpus=corpus, source_id=sid[corpus], noise_class="stationary"))
        if corpus in scenes.V2_EXCLUDED_CORPORA:
            assert t == [], corpus
        elif specific:
            assert t and set(t) != {"general"}, (corpus, t)
        else:
            assert t == ["general"], (corpus, t)


def test_wham_is_babble_not_an_outdoor_bed():
    t = noise_tags(dict(corpus="wham", source_id="wham:tr:x", noise_class="changing"))
    assert "babble" in t and "ambient_outdoor" not in t


def test_r8_recipe_corpora_fill_every_scene_role():
    # before the plan 11.5 additions the pool had no drone, ambient_outdoor or babble rows (the scene fell back)
    pool = ScenePool(recipe_noise_rows())
    for name, sc in SCENES.items():
        for role in sc["beds"] + sc.get("points", []):
            assert any(pool._groups(t, False) for t in role["tags"]), (name, role["tags"])
        ev = sc.get("event")
        if ev:
            assert any(pool._groups(t, True) for t in ev["tags"]), (name, ev["tags"])


def test_drone_scene_draws_a_drone_row_not_the_fallback():
    pool = ScenePool(recipe_noise_rows())
    rng = np.random.default_rng(0)
    for _ in range(20):
        sc = scenes.sample_scene(rng, "drone")
        pool.draw(rng, sc)
        pts = [s for s in sc["sources"] if s["role"] == "point"]
        assert pts and all(s["tag"] == "drone" and s["source_id"].startswith("avq_drone:") for s in pts)


def test_demand_pairs_share_the_one_pair_manifest_splits():
    # split = assign(group_id); scan_demand_pairs reuses scan_demand's group, so a test environment stays test
    one = {e: assign(f"demand-{e}") for e in DEMAND_ENVS}
    pairs = {r["source_id"].split(":")[1].split("/")[0]: assign(r["group_id"])
             for r in recipe_noise_rows().to_dict("records") if r["corpus"] == "demand"}
    assert pairs == one
    lap = REPO / "data/manifests/demand.parquet"
    if lap.exists():   # the laptop scan the r8 test set drew its DEMAND rows from
        m = manifests.read(lap)
        assert {s.split(":")[1]: sp for s, sp in zip(m.source_id, m.split)} == {e: one[e] for e in m.source_id.str[7:]}
