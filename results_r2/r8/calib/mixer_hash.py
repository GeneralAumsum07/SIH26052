"""Hash of mixer v1 and v2-default outputs on synthetic inputs: proves a front-end change leaves the defaults bit-exact.

usage: python results_r2/r8/calib/mixer_hash.py   (prints one sha1 per version; compare before/after an edit)
The v2 meta keys added on 2026-09-25 (past_knee, past_aop, past_rails, peak_db_spl) are left out of the v2 hash.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.environ.get("VAANI_ROOT") or str(Path(__file__).resolve().parents[3]))
from vaani.data import scenes   # noqa: E402
from vaani.data.mixer import MixConfig, mix   # noqa: E402

NEW_KEYS = {"past_knee", "past_aop", "past_rails", "peak_db_spl"}
SR, N = 16000, 3 * 16000


def items(version):
    h = hashlib.sha1()
    for i in range(24):
        rng = np.random.default_rng([7, i])
        t = np.arange(N) / SR
        sp = (np.sin(2 * np.pi * 180 * t) * (1 + np.sin(2 * np.pi * 3 * t)) * 0.1).astype(np.float32)
        nz = [rng.standard_normal(N).astype(np.float32) * 0.05, rng.standard_normal(N).astype(np.float32) * 0.02]
        imp = (rng.standard_normal(800) * np.exp(-np.arange(800) / 80)).astype(np.float32)
        if version == 1:
            cfg = MixConfig()
            out = mix(rng, sp, nz[:1], imp, [0.0], None, cfg)
        else:
            sc = scenes.sample_scene(rng, name=list(scenes.SCENES)[i % len(scenes.SCENES)], crop_s=N / SR)
            cfg = MixConfig(version=2, p_clean=0.0)
            out = mix(rng, sp, nz[: len(sc["sources"])], imp if sc["event"] else None, [0.0], None, cfg, scene=sc)
        m, c, meta = out[:3]
        h.update(np.ascontiguousarray(m).tobytes()); h.update(np.ascontiguousarray(c).tobytes())
        h.update(json.dumps({k: v for k, v in meta.items() if k not in NEW_KEYS}, sort_keys=True, default=str).encode())
    return h.hexdigest()


if __name__ == "__main__":
    for v in (1, 2):
        print(f"v{v}", items(v))
