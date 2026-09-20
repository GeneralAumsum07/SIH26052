import numpy as np, torch
from vaani.dsp import stft
from vaani.models.gtcrn import GTCRN
from vaani.models import gtcrn_stream
from vaani.models.modules.convert import convert_to_stream
from vaani.models.baselines import get


def _load_pretrained(model):
    ck = torch.load("vaani/models/checkpoints/model_trained_on_dns3.tar", map_location="cpu", weights_only=True)
    model.load_state_dict(ck["model"]); return model.eval()


def test_batch_vs_stream_parity():
    torch.manual_seed(0)
    x = torch.randn(1, 16000) * 0.1
    spec = stft.stft(x)                                # (1,257,T,2)
    # StreamGTCRN's Conv2d/ConvTranspose2d submodules nest weights one level deeper
    # (".Conv2d.weight" / ".ConvTranspose2d.weight") than plain GTCRN's flat keys,
    # so a direct load_state_dict on the stream model fails - upstream's own
    # convert_to_stream remaps (and, for the transpose-conv flip case, transforms) them.
    m = _load_pretrained(GTCRN())
    s = gtcrn_stream.StreamGTCRN().eval()
    convert_to_stream(s, m)
    with torch.no_grad():
        y = m(spec)
        caches = gtcrn_stream.init_caches("cpu")
        outs = []
        for t in range(spec.shape[2]):
            o, *caches = s(spec[:, :, t:t + 1], *caches); outs.append(o)
        ys = torch.cat(outs, dim=2)
    assert torch.allclose(y, ys, atol=1e-3)


def test_param_count():
    n = sum(p.numel() for p in GTCRN().parameters())
    assert 47000 < n < 49500, f"unexpected param count {n}"


def test_baselines_enhance_shape():
    x = np.random.default_rng(0).standard_normal((2, 16000)).astype(np.float32) * 0.1
    for name in ("raw", "nlms_only", "gtcrn_pretrained"):
        y = get(name).enhance(x)
        assert y.shape == (16000,) and np.isfinite(y).all()


def test_rnnoise_registered_but_unavailable():
    import pytest
    with pytest.raises(NotImplementedError):
        get("rnnoise").enhance(np.zeros((2, 16000), dtype=np.float32))


def test_gtcrn_cuda_smoke():
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("no CUDA device")
    ck = torch.load("vaani/models/checkpoints/model_trained_on_dns3.tar", map_location="cuda", weights_only=True)
    m = GTCRN().cuda(); m.load_state_dict(ck["model"]); m.eval()
    x = torch.randn(1, 257, 10, 2, device="cuda")
    with torch.no_grad():
        y = m(x)
    assert y.shape == x.shape


def test_hgtcrn_uses_both_mics():
    # H-GTCRN is the one comparator that reads the reference mic: swapping the mics must change the output.
    # A synthetic tone is suppressed as non-speech, so the energy check needs a real rendered clip (skipped if absent).
    import pytest, soundfile as sf
    from pathlib import Path
    f = Path("data/eval_r2/test/stationary_5/0000.mix.wav")
    if not f.exists(): pytest.skip("eval_r2 not rendered")
    mix = sf.read(f, dtype="float32")[0].T
    for name in ("h_gtcrn", "h_gtcrn_iva"):
        y = get(name).enhance(mix)
        assert y.shape == (mix.shape[1],) and np.isfinite(y).all()
        assert not np.allclose(y, get(name).enhance(mix[::-1].copy()), atol=1e-4)
    # the masking-on-noisy variant must keep speech-level energy; the IVA variant is known to collapse on our geometry
    assert np.sqrt(np.mean(get("h_gtcrn").enhance(mix) ** 2)) > 0.3 * np.sqrt(np.mean(mix[0] ** 2))
