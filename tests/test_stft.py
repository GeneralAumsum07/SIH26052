import numpy as np, torch
from vaani.dsp import stft


def test_roundtrip():
    x = torch.randn(2, 16000)
    y = stft.istft(stft.stft(x), length=16000)
    assert torch.allclose(x, y, atol=1e-4)


def test_np_matches_torch():
    x = np.random.randn(8000).astype(np.float32)
    a = stft.np_stft(x)
    b = stft.stft(torch.from_numpy(x))
    b = b[..., 0].numpy() + 1j * b[..., 1].numpy()
    assert a.shape == b.shape
    assert np.allclose(a, b, atol=1e-4)
