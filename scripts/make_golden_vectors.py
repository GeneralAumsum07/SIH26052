"""Golden vectors for the DSP port. Deterministic inputs -> expected outputs.

Short (0.5 s) clips + compressed npz so this stays committed under deploy/
(the DSP lead ports features.py/controller.py/nlms.py to C and diffs
against these) instead of the git-ignored /data/ tree.
"""
from pathlib import Path
import numpy as np
import soundfile as sf

from vaani.dsp import pipeline

out = Path("deploy/dsp_reference/vectors")
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)

SR = 16000
DUR = 0.5  # seconds - kept short so the committed vectors stay well under budget
n = int(SR * DUR)
t = np.arange(n) / SR

cases = {}
voiced = (0.3 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
noise = rng.standard_normal(n).astype(np.float32) * 0.05
cases["speech_plus_noise"] = np.stack([voiced + noise, np.roll(voiced, 5) * 0.3 + np.roll(noise, 2)])

imp = np.zeros(n, np.float32)
imp_len = min(800, n)
imp[: imp_len] = np.exp(-np.arange(imp_len) / 100) * rng.standard_normal(imp_len)
imp = np.roll(imp, n // 4)  # place the impulse mid-clip
cases["burst"] = np.stack([voiced * 0.2 + imp, np.roll(voiced, 5) * 0.06 + imp * 0.9])

drop = cases["speech_plus_noise"].copy()
drop[1, n // 2:] *= 0.01
cases["ref_dropout"] = drop

for name, x in cases.items():
    sf.write(out / f"{name}.wav", x.T, SR, subtype="FLOAT")  # PCM16 quantised (and clipped the burst) so the npz never matched the wav
    r = pipeline.run(x)
    np.savez_compressed(out / f"{name}.npz", **r)
    print(name, "burst frames:", int(r["burst"].sum()))
