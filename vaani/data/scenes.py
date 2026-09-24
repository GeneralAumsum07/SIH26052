"""Scene-driven sampling for mixer v2 (plan 11.5 M8).

r7 drew noise rows uniformly per manifest row, so drone was 9.4 % of draws from 0.37 h and indoor DNS doors,
fans and typing were 62 % of noise hours. v2 draws a scene first (weights below), then the scene's noise classes,
each with an SPL range; within a class every group is equally likely, so no corpus wins on row count.

Scene level ranges are inferred from verified anchors (battlefield_physics section 7); they are a starting
distribution to be checked against field recordings once hardware exists.

API for the dataset (vaani/data/dataset.py is wired by the trainer stage):
    pool = ScenePool(noise_df)                  # noise rows of one split (manifest columns)
    scene = sample_scene(rng, crop_s=4.0)       # drawn scene parameters (dict, JSON-safe)
    rows, imp_row = pool.draw(rng, scene)       # one manifest row per scene["sources"] entry (None = no row found),
                                                # plus an impulsive row for scene["event"] or None (use impulses.generate)
    mix(rng, speech, [load(r) for r in rows], impulse, onsets, bank, cfg_v2, scene=scene)
"""
import numpy as np

from vaani.data import calib

SCENE_WEIGHTS = {"patrol": .15, "apc": .15, "firefight": .15, "artillery": .10, "helicopter": .10, "drone": .10,
                 "windy_ridge": .10, "command_post": .15}

# levels in dB SPL at the boom mic; beds and point sources in dBA (Leq), events in dB peak, talker speech-active rms.
SCENES = {
    "patrol": dict(rir="outdoor", talker=(80, 92), lombard=False,
                   beds=[dict(tags=["ambient_outdoor", "nature", "general"], spl=(45, 65))],
                   points=[dict(tags=["footsteps", "vehicle", "general"], spl=(40, 60), p=0.3)],
                   event=dict(tags=["gunshot", "gunfire"], peak=[(1.0, (100, 125))], rate_hz=(0.0, 0.2)),
                   wind=dict(p=0.3, speed=(0.0, 8.0))),
    "apc": dict(rir="armoured", talker=(100, 110), lombard=True,
                beds=[dict(tags=["tracked_vehicle", "vehicle", "engine"], spl=(100, 115))],
                points=[],
                event=dict(tags=["clank", "impulsive"], peak=[(1.0, (110, 125))], rate_hz=(0.0, 0.5)),   # clank rate inferred
                wind=dict(p=0.0, speed=(0.0, 0.0))),
    "firefight": dict(rir="outdoor", talker=(100, 110), lombard=True,
                      beds=[dict(tags=["gunfire", "ambient_outdoor"], spl=(70, 90))],
                      points=[],
                      # own weapon 160, squad 140-150, others 115-135 dB peak; the mixture shares are inferred
                      event=dict(tags=["gunshot"], peak=[(0.1, (158, 162)), (0.3, (140, 150)), (0.6, (115, 135))],
                                 rate_hz=(0.5, 5.0)),
                      wind=dict(p=0.2, speed=(0.0, 8.0))),
    "artillery": dict(rir="outdoor", talker=(95, 108), lombard=True,
                      beds=[dict(tags=["ambient_outdoor", "artillery"], spl=(60, 80))],
                      points=[],
                      event=dict(tags=["explosion", "artillery", "gunshot"], peak=[(1.0, (125, 160))], rate_hz=(1 / 60, 6 / 60)),
                      wind=dict(p=0.2, speed=(0.0, 8.0))),
    "helicopter": dict(rir="outdoor", talker=(100, 110), lombard=True,
                       beds=[dict(tags=["helicopter"], spl=(100, 115))],
                       points=[],
                       event=None,
                       wind=dict(p=0.7, speed=(10.0, 25.0))),   # downwash, inferred range
    "drone": dict(rir="outdoor", talker=(85, 95), lombard=False,
                  beds=[dict(tags=["ambient_outdoor", "nature", "general"], spl=(45, 65))],
                  points=[dict(tags=["drone"], spl=(55, 80), p=1.0)],   # drone SPL TBD (primary source paywalled)
                  event=None,
                  wind=dict(p=0.2, speed=(0.0, 8.0))),
    "windy_ridge": dict(rir="outdoor", talker=(89, 100), lombard=False,
                        beds=[dict(tags=["ambient_outdoor", "nature", "general"], spl=(40, 60))],
                        points=[],
                        event=None,
                        wind=dict(p=1.0, speed=(5.0, 20.0))),
    "command_post": dict(rir="room", talker=(85, 92), lombard=False,
                         beds=[dict(tags=["babble", "indoor", "general"], spl=(55, 75))],
                         points=[dict(tags=["radio", "babble"], spl=(70, 85), p=0.6),
                                 dict(tags=["generator", "engine", "machinery"], spl=(60, 75), p=0.5)],
                         event=None,
                         wind=dict(p=0.0, speed=(0.0, 0.0))),
}
P_NEAR = 0.7              # extra near-field source (equipment, own gear) per item: breaks "noise ILD is 0" (G1-tuned)
NEAR_SPL_REL = (0.0, 8.0)   # near source level re the bed, dB (G1-tuned)


def _u(rng, lohi):
    lo, hi = lohi
    return float(rng.uniform(lo, hi)) if hi > lo else float(lo)


def sample_scene(rng, name: str | None = None, weights: dict | None = None, crop_s: float = 4.0,
                 p_near: float = P_NEAR, near_spl_rel: tuple[float, float] = NEAR_SPL_REL) -> dict:
    """One drawn scene: names, levels, roles. JSON-safe so it can ride in the item meta."""
    w = weights or SCENE_WEIGHTS
    names = list(w)
    if name is None:
        p = np.asarray([w[k] for k in names], float)
        name = names[int(rng.choice(len(names), p=p / p.sum()))]
    sc = SCENES[name]
    talker = _u(rng, sc["talker"])
    effort = calib.effort_class(talker - calib.SPEECH_NORMAL_BOOM_DB_SPL)
    sources = []
    for b in sc["beds"]:
        sources.append(dict(role="bed", tags=list(b["tags"]), spl=_u(rng, b["spl"]), weighting="A"))
    for pt in sc["points"]:
        if rng.random() < pt["p"]:
            sources.append(dict(role="point", tags=list(pt["tags"]), spl=_u(rng, pt["spl"]), weighting="A"))
    if rng.random() < p_near:
        b = sources[0]
        sources.append(dict(role="near", tags=list(b["tags"]), spl=b["spl"] + _u(rng, near_spl_rel), weighting="A"))
    wind = 0.0
    if rng.random() < sc["wind"]["p"]:
        wind = _u(rng, sc["wind"]["speed"])
    event = None
    if sc["event"] is not None:
        ev = sc["event"]; rate = _u(rng, ev["rate_hz"])
        shares = np.asarray([s for s, _ in ev["peak"]], float)
        k = int(rng.choice(len(shares), p=shares / shares.sum()))
        event = dict(tags=list(ev["tags"]), p=float(1 - np.exp(-rate * crop_s)), rate_hz=rate, peak_spl=_u(rng, ev["peak"][k][1]))
    return dict(name=name, rir=sc["rir"], speech_spl=talker, effort=effort,
                lombard=bool(sc["lombard"] and effort != "normal"), sources=sources, wind_mps=wind, event=event)


# --- manifest row -> scene noise tags ---
MAD_TAGS = {"shooting": ["gunfire"], "shelling": ["artillery", "explosion"], "footsteps": ["footsteps"],
            "vehicle": ["vehicle", "tracked_vehicle"], "helicopter": ["helicopter"], "fighter": ["jet"]}
DEMAND_TAGS = {"N": ["ambient_outdoor", "nature"], "S": ["ambient_outdoor", "vehicle"], "T": ["vehicle", "engine"],
               "O": ["indoor", "babble"], "P": ["babble", "indoor"], "D": ["indoor"]}
ESC50_TAGS = {"wind": ["wind"], "rain": ["nature"], "thunderstorm": ["nature", "explosion"], "crickets": ["nature"],
              "chirping_birds": ["nature"], "insects": ["nature"], "water_drops": ["nature"], "sea_waves": ["nature"],
              "crackling_fire": ["nature"], "helicopter": ["helicopter"], "airplane": ["jet"], "engine": ["engine", "vehicle"],
              "train": ["vehicle"], "car_horn": ["vehicle"], "chainsaw": ["machinery", "generator"], "hand_saw": ["machinery"],
              "siren": ["siren"], "footsteps": ["footsteps"], "fireworks": ["gunshot", "explosion"],
              "keyboard_typing": ["indoor"], "mouse_click": ["indoor"], "clock_tick": ["indoor"], "vacuum_cleaner": ["indoor"],
              "washing_machine": ["indoor", "machinery"], "door_wood_creaks": ["indoor"], "can_opening": ["clank"]}
# v2 pools: corpora that stay eval-only (NOISEX-92, vehicle_interior) or are dropped (DroneAudioDataset: indoor,
# two toy drones, 0.37 h, no licence) never enter a training scene
V2_EXCLUDED_CORPORA = {"noisex92", "vehicle_interior", "drone"}


def noise_tags(row) -> list[str]:
    """Scene tags for one noise manifest row, from corpus and source_id only (the manifest has no other labels)."""
    corpus, sid = str(row["corpus"]), str(row["source_id"])
    if corpus in V2_EXCLUDED_CORPORA:
        return []
    impulsive = str(row.get("noise_class", "")) == "impulsive"
    if corpus == "mad":
        cls = sid.split(":", 1)[1].split("/", 1)[0]
        return list(MAD_TAGS.get(cls, []))
    if corpus in ("gunshots", "cadre", "c3gd"):
        return ["gunshot", "impulsive"]
    if corpus == "demand":
        env = sid.split(":", 1)[1]
        return list(DEMAND_TAGS.get(env[:1], ["general"]))
    if corpus == "esc50":
        cat = sid.split(":", 1)[1].split("/", 1)[0]
        t = list(ESC50_TAGS.get(cat, ["general"]))
        return t + (["impulsive"] if impulsive else [])
    if corpus == "avq_drone":
        return ["drone"]
    if corpus == "fsd50k":
        lab = sid.split(":", 1)[1].split("/", 1)[0]
        return list(FSD50K_TAGS.get(lab, ["general"])) + (["impulsive"] if impulsive else [])
    if corpus in ("dns_noise", "musan") or corpus.startswith("dns_"):
        return ["general"] + (["impulsive"] if impulsive else [])
    return ["general"]


FSD50K_TAGS = {"Gunshot_and_gunfire": ["gunshot", "impulsive"], "Explosion": ["explosion", "artillery"],
               "Siren": ["siren"], "Wind": ["wind"], "Helicopter": ["helicopter"], "Engine": ["engine", "vehicle"],
               "Vehicle": ["vehicle"], "Aircraft": ["jet"], "Crowd": ["babble"], "Chatter": ["babble"]}


class ScenePool:
    """Tag index over noise rows. Draw order: scene tag list -> a tag uniformly among those with rows (so a
    four-row class is not every item's bed) -> a group uniformly -> a row uniformly within the group."""

    def __init__(self, noise_df):
        df = noise_df.reset_index(drop=True)
        self.df = df
        idx: dict[str, dict[str, list[int]]] = {}
        for i, r in enumerate(df.to_dict("records")):
            for t in noise_tags(r):
                idx.setdefault(t, {}).setdefault(str(r["group_id"]), []).append(i)
        self.index = {t: [np.asarray(v) for _, v in sorted(g.items())] for t, g in idx.items()}
        self.cont = [i for i, r in enumerate(df.to_dict("records")) if noise_tags(r) and r.get("noise_class") != "impulsive"]

    def tags(self) -> dict[str, int]:
        return {t: int(sum(len(g) for g in gs)) for t, gs in sorted(self.index.items())}

    def _groups(self, t, impulsive: bool | None):
        groups = self.index.get(t) or []
        if impulsive is not None and groups:   # keep continuous sources continuous and events impulsive
            groups = [g[(self.df.noise_class.values[g] == "impulsive") == impulsive] for g in groups]
            groups = [g for g in groups if len(g)]
        return groups

    def _pick(self, rng, tags, impulsive: bool | None):
        avail = [(t, g) for t in tags for g in [self._groups(t, impulsive)] if g]
        if not avail:
            return None, None
        t, groups = avail[int(rng.integers(len(avail)))]
        g = groups[int(rng.integers(len(groups)))]
        return self.df.iloc[int(g[int(rng.integers(len(g)))])], t

    def draw(self, rng, scene: dict):
        """Rows for scene["sources"] (None where no tag has rows; the caller may fall back) and an impulsive row
        for the scene event (None: use the synthetic generator). The tag used is written back into the scene."""
        rows = []
        for s in scene["sources"]:
            r, t = self._pick(rng, s["tags"], impulsive=False)
            if r is None and self.cont:   # nothing tagged: any continuous v2-eligible row, flagged in the scene
                r, t = self.df.iloc[self.cont[int(rng.integers(len(self.cont)))]], "fallback"
            s["tag"] = t; s["source_id"] = None if r is None else str(r["source_id"])
            rows.append(r)
        imp = None
        ev = scene.get("event")
        if ev is not None and rng.random() < ev["p"]:
            imp, t = self._pick(rng, ev["tags"], impulsive=True)
            ev["tag"] = t if imp is not None else "synthetic"; ev["fired"] = True
        elif ev is not None:
            ev["fired"] = False
        return rows, imp
