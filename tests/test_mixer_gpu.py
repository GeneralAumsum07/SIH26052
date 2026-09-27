"""GPU renderer (plan Task 4b): recipes rendered in float64 against the CPU mixer v2 on the same seeds."""
import numpy as np
import pytest
import torch

from vaani.data import mixer, mixer_gpu, scenes

SR = 16000


class FakeBank:
    """RirBank's sampling semantics (uniform index, armoured selection) over synthetic RIRs."""

    def __init__(self, seed=0, k=6):
        rng = np.random.default_rng(seed)
        dec = np.exp(-np.arange(1200) / 200.0)
        self.speech = [(rng.standard_normal((2, 1200)) * dec).astype(np.float32) for _ in range(k)]
        self.noise = [[(rng.standard_normal((2, 1200)) * dec).astype(np.float32) for _ in range(3)] for _ in range(k)]
        self._pick = {True: np.arange(0, k, 2), False: np.arange(1, k, 2)}

    def __len__(self):
        return len(self.speech)

    def sample(self, rng, armoured=None):
        idx = None if armoured is None else self._pick[bool(armoured)]
        i = int(rng.integers(len(self))) if idx is None else int(idx[rng.integers(len(idx))])
        return {"speech": self.speech[i], "noise": self.noise[i], "rt60": 0.3}


def _speech(rng, n):
    t = np.arange(n) / SR
    f0 = rng.uniform(90, 220)
    x = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 8))
    env = (np.sin(2 * np.pi * rng.uniform(2, 5) * t) > 0).astype(float)
    return (0.1 * x * env + 1e-3 * rng.standard_normal(n)).astype(np.float32)


def _inputs(seed, n):
    rng = np.random.default_rng(10_000 + seed)
    scene = scenes.sample_scene(rng, crop_s=n / SR)
    noises = []
    for s in scene["sources"]:
        L = int(rng.integers(n // 3, 2 * n))
        if s["role"] == "bed" and rng.random() < 0.25:
            noises.append((rng.standard_normal((L, 2)) * 0.05).astype(np.float32))   # measured two-mic bed
        else:
            noises.append((rng.standard_normal(L) * 0.05).astype(np.float32))
    imp, onsets = None, []
    if rng.random() < 0.5:
        L = int(rng.integers(400, 4000))
        imp = (rng.standard_normal(L) * np.exp(-np.arange(L) / 300)).astype(np.float32)
        onsets = [0.0]
    v2 = {}
    r = rng.random()
    if r < 0.15:
        v2 = {"fe_hpf_order": "post"}
    elif r < 0.3:
        v2 = {"fe_curve": "tanh120"}
    elif r < 0.4:
        v2 = {"mic_fs_spl_db": 130.0}
    elif r < 0.5:
        v2 = {"path": "room"}
    cfg = mixer.MixConfig(version=2, v2=v2 or None, p_room=0.6, p_clean=0.05)
    return scene, _speech(rng, n), noises, imp, onsets, cfg


DISCRETE = ("mix_version", "scene", "clean_bucket", "clipped", "ref_dropout", "effort", "lombard", "ref_mode", "path",
            "overloaded", "past_knee", "past_aop", "past_rails", "noise_class", "wind_states", "impulse_onsets_s",
            "mic_gain_db", "noise_sources", "wind_mps", "impulse_peak_db", "speech_spl_db")
NUMERIC = ("ref_speech_gain_db", "snr_db", "snr_achieved_db", "clip_frac", "peak_db_spl", "wind_spl_db", "wind_fc_hz")


TOL = 1e-5   # of full scale (plan Task 4b); never widened
REFERENCE_PRECISION = []   # items where the CPU mixer's own single-precision FFT convolution exceeds TOL


def _cpu_f64_conv(seed, sp, noises, imp, onsets, bank, cfg, scene):
    """mix_v2 with its fftconvolve evaluated in float64 (SciPy runs it in float32 for float32 inputs)."""
    import copy
    orig = mixer.fftconvolve
    mixer.fftconvolve = lambda a, b, *k, **kw: orig(np.asarray(a, np.float64), np.asarray(b, np.float64), *k, **kw)
    try:
        return mixer.mix_v2(np.random.default_rng(seed), sp, noises, imp, onsets, bank, cfg, scene=copy.deepcopy(scene))
    finally:
        mixer.fftconvolve = orig


def _compare(seed, n=8000, device="cpu"):
    scene, sp, noises, imp, onsets, cfg = _inputs(seed, n)
    bank = FakeBank(seed)
    ra, rb = np.random.default_rng(seed), np.random.default_rng(seed)
    import copy
    ref_mix, ref_clean, ref_meta = mixer.mix_v2(ra, sp, noises, imp, onsets, bank, cfg, scene=copy.deepcopy(scene))
    rec = mixer_gpu.mix_v2_recipe(rb, sp, noises, imp, onsets, bank, cfg, scene=copy.deepcopy(scene))
    assert ra.random() == rb.random()      # the same draws, in the same order
    mix, clean, meta = mixer_gpu.render_recipe(rec, device)
    assert mix.shape == ref_mix.shape and mix.dtype == np.float32
    err = float(np.abs(mix - ref_mix).max())
    if err > TOL:
        # allowed only when the reference itself is off by more than TOL (its float32 FFT convolution of an
        # overloading item) and the renderer is within TOL of the same mixer with a float64 convolution
        f64, _, _ = _cpu_f64_conv(seed, sp, noises, imp, onsets, bank, cfg, scene)
        assert float(np.abs(ref_mix - f64).max()) > TOL and float(np.abs(mix - f64).max()) <= TOL, (seed, err)
        REFERENCE_PRECISION.append((seed, err))
    assert np.abs(clean - ref_clean).max() <= 1e-5
    assert set(meta) == set(ref_meta)
    for k in DISCRETE:
        if k in ref_meta:
            assert meta[k] == ref_meta[k], (seed, k)
    for k in NUMERIC:
        if k in ref_meta:
            a, b = meta[k], ref_meta[k]
            assert (np.isinf(a) and np.isinf(b)) or abs(a - b) <= 1e-3 * max(1.0, abs(b)), (seed, k, a, b)


@pytest.mark.parametrize("block", range(10))
def test_1000_recipes_match_the_cpu_mixer(block):
    for seed in range(block * 100, (block + 1) * 100):
        _compare(seed)
    # the renderer is adopted only on an owner decision if any item needed the float64-convolution reference
    print(f"block {block}: reference-precision items {REFERENCE_PRECISION}")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_recipes_match_on_cuda():
    for seed in range(200):
        _compare(seed, device="cuda")


def test_dataset_items_through_render_and_finish(tmp_path):
    """DynamicMixDataset.recipe + render_and_finish == __getitem__ + collate (faults and the compiled limiter on CPU)."""
    from tests.test_train_smoke import _tiny
    from vaani.data.dataset import DynamicMixDataset, collate
    m = _tiny(tmp_path)
    ds = DynamicMixDataset([m], "train", None, mixer.MixConfig(version=2), crop_s=0.5, epoch_len=40, seed=0,
                           fe_inputs=True, dsp_cfg={"limiter": True, "limiter_kernel": "numba"},
                           ref_corrupt={"p": 0.5, "p_absent": 0.3})
    for idx in range(0, 80, 3):
        a = collate([ds[idx]])
        b = mixer_gpu.render_and_finish(ds, [ds.recipe(idx)], "cpu")
        assert float((a["mix"] - b["mix"]).abs().max()) <= 1e-5
        assert float((a["clean"] - b["clean"]).abs().max()) <= 1e-5
        assert torch.equal(a["avail"], b["avail"]) and a["meta"][0]["scene"] == b["meta"][0]["scene"]
        assert a["meta"][0].get("ref_fault") == b["meta"][0].get("ref_fault")
