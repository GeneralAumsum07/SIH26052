"""`post:` systems in vaani.eval: a unity-floor post-filter must reproduce the plain checkpoint path."""
import numpy as np, pytest, soundfile as sf, yaml
from pathlib import Path

from vaani import eval as ev

CLIP = Path("data/eval_r2/test/stationary_5/0000.mix.wav")
ANCHOR = Path("runs/tier46_anchor/best.pt")


@pytest.mark.skipif(not (CLIP.exists() and ANCHOR.exists()), reason="needs the rendered eval_r2 split and the frozen anchor")
def test_unity_postfilter_matches_ckpt_path(tmp_path):
    mix = sf.read(CLIP, dtype="float32")[0].T
    ref = ev.enhance_fn(f"ckpt:{ANCHOR.as_posix()}", device="cpu")(mix)
    y = tmp_path / "pf.yaml"; y.write_text(yaml.safe_dump({"base_checkpoint": ANCHOR.as_posix(), "postfilter": {"gain_floor": 1.0}}))
    got = ev.enhance_fn(f"post:{y.as_posix()}", device="cpu")(mix)
    assert got.shape == ref.shape and np.abs(got - ref).max() < 1e-6
    # a real floor changes the output, and only after the warm-up frames
    y.write_text(yaml.safe_dump({"base_checkpoint": ANCHOR.as_posix(), "postfilter": {"gain_floor": 0.7}}))
    got2 = ev.enhance_fn(f"post:{y.as_posix()}", device="cpu")(mix)
    assert np.abs(got2 - ref).max() > 1e-4 and np.abs(got2[:256 * 14] - ref[:256 * 14]).max() < 1e-6
