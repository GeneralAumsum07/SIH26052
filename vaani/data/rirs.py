"""pyroomacoustics RIR pairs for the headset geometry."""
from pathlib import Path

import numpy as np
import pyroomacoustics as pra

SR = 16000
MIC_SPACING = 0.12       # rigid mount, reference 12 cm from primary
MOUTH_TO_PRIMARY = 0.025
MOUTH_TO_REF = 0.14
MAX_LEN = int(0.6 * SR)  # 600 ms covers RT60 <= 0.5 s


def simulate_pair_set(rng: np.random.Generator, sr: int = SR, n_noise: int = 2) -> dict:
    dims = rng.uniform(2.5, 6.0, size=3); dims[2] = rng.uniform(2.4, 3.2)
    rt60 = float(rng.uniform(0.1, 0.5))
    e_abs, max_order = pra.inverse_sabine(rt60, dims)
    room = pra.ShoeBox(dims, fs=sr, materials=pra.Material(e_abs), max_order=min(max_order, 12))

    # mics roughly at head height near the room centre, random heading
    head = np.array([rng.uniform(0.8, dims[0] - 0.8), rng.uniform(0.8, dims[1] - 0.8), 1.6])
    yaw = rng.uniform(0, 2 * np.pi)
    fwd = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    primary = head + fwd * MOUTH_TO_PRIMARY
    # reference sits behind/below along the same axis at MIC_SPACING from primary
    reference = primary - fwd * MIC_SPACING * 0.9 + np.array([0, 0, -MIC_SPACING * 0.44])
    mouth = head
    room.add_microphone_array(np.stack([primary, reference], axis=1))

    room.add_source(mouth)
    for _ in range(n_noise):
        while True:
            p = np.array([rng.uniform(0.3, dims[0] - 0.3), rng.uniform(0.3, dims[1] - 0.3), rng.uniform(0.5, 2.2)])
            if np.linalg.norm(p - head) >= 0.8:
                break
        room.add_source(p)
    room.compute_rir()

    def pack(src_idx):
        out = np.zeros((2, MAX_LEN), np.float32)
        for m in range(2):
            h = np.asarray(room.rir[m][src_idx], np.float32)[:MAX_LEN]
            out[m, :len(h)] = h
        return out

    return {"speech": pack(0), "noise": np.stack([pack(1 + i) for i in range(n_noise)]),
            "rt60": rt60, "room_dims": dims.astype(np.float32)}


def build_bank(path: Path, n: int = 5000, seed: int = 0, n_noise: int = 3) -> None:
    rng = np.random.default_rng(seed)
    sp, nz, rt = [], [], []
    for _ in range(n):
        s = simulate_pair_set(rng, n_noise=n_noise)
        sp.append(s["speech"]); nz.append(s["noise"]); rt.append(s["rt60"])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, speech=np.stack(sp), noise=np.stack(nz), rt60=np.array(rt, np.float32))


class RirBank:
    def __init__(self, path: Path):
        z = np.load(path)
        self.speech, self.noise, self.rt60 = z["speech"], z["noise"], z["rt60"]

    def __len__(self):
        return len(self.rt60)

    def sample(self, rng: np.random.Generator) -> dict:
        i = int(rng.integers(len(self)))
        return {"speech": self.speech[i], "noise": self.noise[i], "rt60": float(self.rt60[i])}
