"""pyroomacoustics RIR pairs for the headset geometry."""
from pathlib import Path

import os
import numpy as np
import pyroomacoustics as pra

SR = 16000
MIC_SPACING = 0.12       # rigid mount, reference 12 cm from primary
MOUTH_TO_PRIMARY = 0.025
MAX_LEN = int(0.6 * SR)  # 600 ms covers RT60 <= 0.5 s (room mode); armoured banks pass a longer max_len
MAX_SABINE_ATTEMPTS = 50
# armoured mode (plan 2.9): 1.5-2.5 m steel box, RT60 long for the volume. The image method needs order ~200 there,
# so the tail comes from pyroomacoustics' ray tracer on top of a 3rd-order ISM (measured T60 0.81 s vs 0.26 s truncated).
ARMOURED_DIMS = (1.5, 2.5)
ARMOURED_RT60 = (0.4, 0.9)
ARMOURED_RAYS = 10000


def simulate_pair_set(rng: np.random.Generator, sr: int = SR, n_noise: int = 2, armoured: bool = False,
                      max_len: int = MAX_LEN) -> dict:
    # inverse_sabine rejects (dims, rt60) combos where the room is too large for
    # that RT60; resample both together rather than clipping so the drawn RT60 stays honest.
    for _ in range(MAX_SABINE_ATTEMPTS):
        if armoured:
            dims = rng.uniform(*ARMOURED_DIMS, size=3); rt60 = float(rng.uniform(*ARMOURED_RT60))
        else:
            dims = rng.uniform(2.5, 6.0, size=3); dims[2] = rng.uniform(2.4, 3.2); rt60 = float(rng.uniform(0.1, 0.5))
        try:
            e_abs, max_order = pra.inverse_sabine(rt60, dims)
            break
        except ValueError:
            continue
    else:
        raise ValueError(f"no valid (dims, rt60) found in {MAX_SABINE_ATTEMPTS} attempts")
    if armoured:
        room = pra.ShoeBox(dims, fs=sr, materials=pra.Material(e_abs), max_order=3, ray_tracing=True, air_absorption=True)
        room.set_ray_tracing(receiver_radius=0.3, n_rays=ARMOURED_RAYS)
    else:
        room = pra.ShoeBox(dims, fs=sr, materials=pra.Material(e_abs), max_order=min(max_order, 12))

    # mics roughly at head height near the room centre, random heading; seated crew in the small box
    m = 0.3 if armoured else 0.8
    z = rng.uniform(0.9, dims[2] - 0.3) if armoured else 1.6
    head = np.array([rng.uniform(m, dims[0] - m), rng.uniform(m, dims[1] - m), z])
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
            p = np.array([rng.uniform(0.3, dims[0] - 0.3), rng.uniform(0.3, dims[1] - 0.3),
                          rng.uniform(0.5, min(2.2, dims[2] - 0.2))])
            if np.linalg.norm(p - head) >= (0.5 if armoured else 0.8):
                break
        room.add_source(p)
    room.compute_rir()

    def pack(src_idx):
        out = np.zeros((2, max_len), np.float32)
        for mic in range(2):
            h = np.asarray(room.rir[mic][src_idx], np.float32)[:max_len]
            out[mic, :len(h)] = h
        return out

    return {"speech": pack(0), "noise": np.stack([pack(1 + i) for i in range(n_noise)]),
            "rt60": rt60, "room_dims": dims.astype(np.float32), "armoured": armoured}


def build_bank(path: Path, n: int = 5000, seed: int = 0, n_noise: int = 3, armoured_frac: float = 0.0,
               max_len: int = MAX_LEN) -> None:
    # armoured entries are drawn per index so the share is exact and the file order is still seed-reproducible
    rng = np.random.default_rng(seed)
    arm = np.zeros(n, bool); arm[:int(round(n * armoured_frac))] = True; rng.shuffle(arm)
    sp, nz, rt = [], [], []
    for a in arm:
        s = simulate_pair_set(rng, n_noise=n_noise, armoured=bool(a), max_len=max_len)
        sp.append(s["speech"]); nz.append(s["noise"]); rt.append(s["rt60"])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # "armoured" is bank metadata only: RirBank reads KEYS and ignores it, so older banks without it still load
    np.savez_compressed(path, speech=np.stack(sp), noise=np.stack(nz), rt60=np.array(rt, np.float32), armoured=arm)


class RirBank:
    """Memory-mapped so DataLoader workers (spawned on Windows) share one page cache instead of 2 GB each."""
    KEYS = ("speech", "noise", "rt60")

    def __init__(self, path: Path):
        path = Path(path)
        parts = {k: path.with_name(f"{path.stem}.{k}.npy") for k in self.KEYS}
        if not all(f.exists() for f in parts.values()):
            z = np.load(path)
            for k, f in parts.items():
                tmp = f.with_suffix(".tmp.npy")
                np.save(tmp, z[k]); os.replace(tmp, f)
        self.speech, self.noise, self.rt60 = (np.load(parts[k], mmap_mode="r") for k in self.KEYS)

    def __len__(self):
        return len(self.rt60)

    def sample(self, rng: np.random.Generator) -> dict:
        i = int(rng.integers(len(self)))
        return {"speech": self.speech[i], "noise": self.noise[i], "rt60": float(self.rt60[i])}
