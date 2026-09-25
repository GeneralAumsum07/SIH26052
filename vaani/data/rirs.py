"""pyroomacoustics RIR pairs for the headset geometry."""
from pathlib import Path

import hashlib
import json
import os
import shutil
import time
import uuid
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
# M6 (plan 11.5): the legacy 0.3 m receiver sphere is wider than the 0.12 m mic spacing, so both mics integrate the
# same rays. Opt-in radius below half the spacing keeps the spheres disjoint; rays scale by (0.3 / r)^2 so the hit
# count per sphere, and with it the tail's variance, stays that of the legacy setting.
LEGACY_RECEIVER_RADIUS = 0.3
M6_RECEIVER_RADIUS = 0.05
# eval banks draw from their own SeedSequence namespace: an eval bank built with seed k no longer shares its draw
# stream with a training bank built with any seed (plan 3.5: 28 % of eval rooms were bit-identical to training rooms)
EVAL_SEED_NAMESPACE = 0x4556414C   # "EVAL"
# r8 training banks likewise: disjoint from the legacy stream, so no room repeats one in bank.npz (the eval_r2/eval_gen bank)
TRAIN_SEED_NAMESPACE = 0x5452414E  # "TRAN"
SEED_NAMESPACES = {"eval": EVAL_SEED_NAMESPACE, "train": TRAIN_SEED_NAMESPACE}


def draw_room_params(rng: np.random.Generator, n_noise: int = 2, armoured: bool = False) -> dict:
    """Every random draw of one bank entry, in the order the sequential generator always made them. Keeping the
    draws here and the image-source work in simulate_from_params lets build_bank run the simulations in a
    pool while the bank stays byte-identical to the single-process output."""
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

    # mics roughly at head height near the room centre, random heading; seated crew in the small box
    m = 0.3 if armoured else 0.8
    z = rng.uniform(0.9, dims[2] - 0.3) if armoured else 1.6
    head = np.array([rng.uniform(m, dims[0] - m), rng.uniform(m, dims[1] - m), z])
    yaw = rng.uniform(0, 2 * np.pi)
    noise_pos = []
    for _ in range(n_noise):
        while True:
            p = np.array([rng.uniform(0.3, dims[0] - 0.3), rng.uniform(0.3, dims[1] - 0.3),
                          rng.uniform(0.5, min(2.2, dims[2] - 0.2))])
            if np.linalg.norm(p - head) >= (0.5 if armoured else 0.8):
                break
        noise_pos.append(p)
    return {"dims": dims, "rt60": rt60, "e_abs": e_abs, "max_order": max_order, "head": head, "yaw": yaw,
            "noise_pos": noise_pos, "armoured": armoured}


def simulate_from_params(prm: dict, sr: int = SR, max_len: int = MAX_LEN, receiver_radius: float | None = None) -> dict:
    """The deterministic half: build the room from drawn parameters and run the image-source method.
    receiver_radius: armoured ray tracer only; None = the legacy 0.3 m (bit-identical banks)."""
    dims, e_abs, head, yaw, armoured = prm["dims"], prm["e_abs"], prm["head"], prm["yaw"], prm["armoured"]
    if armoured:
        room = pra.ShoeBox(dims, fs=sr, materials=pra.Material(e_abs), max_order=3, ray_tracing=True, air_absorption=True)
        rr = LEGACY_RECEIVER_RADIUS if receiver_radius is None else float(receiver_radius)
        room.set_ray_tracing(receiver_radius=rr, n_rays=int(round(ARMOURED_RAYS * (LEGACY_RECEIVER_RADIUS / rr) ** 2)))
    else:
        room = pra.ShoeBox(dims, fs=sr, materials=pra.Material(e_abs), max_order=min(prm["max_order"], 12))
    fwd = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    primary = head + fwd * MOUTH_TO_PRIMARY
    # reference sits behind/below along the same axis at MIC_SPACING from primary
    reference = primary - fwd * MIC_SPACING * 0.9 + np.array([0, 0, -MIC_SPACING * 0.44])
    room.add_microphone_array(np.stack([primary, reference], axis=1))
    room.add_source(head)   # mouth
    for p in prm["noise_pos"]: room.add_source(p)
    room.compute_rir()
    n_noise, rt60 = len(prm["noise_pos"]), prm["rt60"]

    def pack(src_idx):
        out = np.zeros((2, max_len), np.float32)
        for mic in range(2):
            h = np.asarray(room.rir[mic][src_idx], np.float32)[:max_len]
            out[mic, :len(h)] = h
        return out

    return {"speech": pack(0), "noise": np.stack([pack(1 + i) for i in range(n_noise)]),
            "rt60": rt60, "room_dims": dims.astype(np.float32), "armoured": armoured}


def simulate_pair_set(rng: np.random.Generator, sr: int = SR, n_noise: int = 2, armoured: bool = False,
                      max_len: int = MAX_LEN, receiver_radius: float | None = None) -> dict:
    return simulate_from_params(draw_room_params(rng, n_noise, armoured), sr, max_len, receiver_radius)


def bank_rng(seed: int, seed_namespace: str | None = None) -> np.random.Generator:
    """None = the legacy stream (default_rng(seed)); "eval" / "train" = disjoint namespaces for eval-only and r8+ training banks."""
    if seed_namespace is None:
        return np.random.default_rng(seed)
    if seed_namespace not in SEED_NAMESPACES:
        raise ValueError(f"unknown seed_namespace {seed_namespace!r}")
    return np.random.default_rng(np.random.SeedSequence([SEED_NAMESPACES[seed_namespace], int(seed)]))


def _sim(args):
    prm, max_len, rr = args
    return simulate_from_params(prm, max_len=max_len, receiver_radius=rr)


def _parts_spec(n, seed, n_noise, armoured_frac, max_len, receiver_radius, seed_namespace, params) -> dict:
    # a parts directory may only be resumed by the build that started it: same arguments and the same draws
    h = hashlib.sha256()
    for p in params:
        h.update(np.asarray(p["dims"], np.float64).tobytes()); h.update(np.float64(p["rt60"]).tobytes())
    return {"n": n, "seed": int(seed), "n_noise": n_noise, "armoured_frac": armoured_frac, "max_len": max_len,
            "receiver_radius": receiver_radius, "seed_namespace": seed_namespace, "draws_sha256": h.hexdigest()}


def build_bank(path: Path, n: int = 5000, seed: int = 0, n_noise: int = 3, armoured_frac: float = 0.0,
               max_len: int = MAX_LEN, workers: int | None = None, receiver_radius: float | None = None,
               seed_namespace: str | None = None, parts_dir: Path | None = None, part_size: int = 50) -> None:
    """receiver_radius / seed_namespace (M6) are opt-in; with both None the bank is byte-identical to before.
    parts_dir: finished rooms are saved there in blocks of part_size, and a rerun skips every saved block (an M6
    armoured bank is hours of ray tracing); the directory is removed once the bank is written."""
    # armoured entries are drawn per index so the share is exact and the file order is still seed-reproducible
    rng = bank_rng(seed, seed_namespace)
    arm = np.zeros(n, bool); arm[:int(round(n * armoured_frac))] = True; rng.shuffle(arm)
    # all draws happen here, sequentially, so the parallel simulation below cannot change the bank's content
    params = [draw_room_params(rng, n_noise, bool(a)) for a in arm]
    # filled in place rather than stacked from a list: halves the peak memory of a 5000-room, 1 s bank (5 -> 2.6 GB)
    sp = np.zeros((n, 2, max_len), np.float32); nz = np.zeros((n, n_noise, 2, max_len), np.float32)
    rt = np.zeros(n, np.float32)
    todo = list(range(n))
    if parts_dir is not None:
        parts_dir = Path(parts_dir); parts_dir.mkdir(parents=True, exist_ok=True)
        spec = _parts_spec(n, seed, n_noise, armoured_frac, max_len, receiver_radius, seed_namespace, params)
        spec_f = parts_dir / "spec.json"
        if spec_f.exists() and json.loads(spec_f.read_text()) != spec:
            raise ValueError(f"{parts_dir} belongs to a different build; remove it or pass another parts_dir")
        spec_f.write_text(json.dumps(spec, indent=1))
        for f in sorted(parts_dir.glob("part_*.npz")):
            i0 = int(f.stem.split("_")[1])
            with np.load(f) as z:   # closed at once: Windows cannot remove the directory while a handle is open
                k = len(z["rt60"]); sp[i0:i0 + k], nz[i0:i0 + k], rt[i0:i0 + k] = z["speech"], z["noise"], z["rt60"]
            todo = [i for i in todo if not i0 <= i < i0 + k]
        print(f"parts: {n - len(todo)}/{n} rooms already saved in {parts_dir}", flush=True)
    t0, block, n_done = time.perf_counter(), [], n - len(todo)

    def flush():
        i0, k = block[0], len(block)
        tmp = parts_dir / f"part_{i0:05d}.{uuid.uuid4().hex}.tmp.npz"
        np.savez(tmp, speech=sp[i0:i0 + k], noise=nz[i0:i0 + k], rt60=rt[i0:i0 + k])
        os.replace(tmp, parts_dir / f"part_{i0:05d}.npz")   # a killed build never leaves a half-written block
        block.clear()
        print(f"part {i0:05d}+{k} saved; {n_done}/{n} rooms, {time.perf_counter() - t0:.0f} s", flush=True)

    def take(i, s):
        nonlocal n_done
        sp[i], nz[i], rt[i] = s["speech"], s["noise"], s["rt60"]
        n_done += 1
        if parts_dir is None:
            return
        if block and i != block[-1] + 1:   # a part file is one contiguous index range
            flush()
        block.append(i)
        if (i + 1) % part_size == 0 or i == todo[-1]:
            flush()

    workers = workers or os.cpu_count() or 1
    if workers > 1 and todo:
        from multiprocessing import get_context
        # one BLAS/OpenMP thread per worker: without this each process spawns a thread per core and a 64-worker
        # pool on a 256-vCPU host exhausted the container's pid cgroup (2026-09-21). Spawned children read the
        # environment at start, so the cap is set before the pool exists and restored afterwards.
        caps = {k: "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")}
        saved = {k: os.environ.get(k) for k in caps}; os.environ.update(caps)
        try:
            with get_context("spawn").Pool(workers) as pool:
                # imap keeps the order, so blocks close in index order; a parts build hands out single rooms so the
                # slow armoured ones spread over the pool instead of queueing behind one chunk
                cs = 1 if parts_dir is not None else 8
                for i, s in zip(todo, pool.imap(_sim, [(params[i], max_len, receiver_radius) for i in todo], chunksize=cs)):
                    take(i, s)
        finally:
            for k, v in saved.items():
                if v is None: os.environ.pop(k, None)
                else: os.environ[k] = v
    else:
        for i in todo:
            take(i, simulate_from_params(params[i], max_len=max_len, receiver_radius=receiver_radius))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # "armoured" is bank metadata only: RirBank reads KEYS and ignores it, so older banks without it still load
    extra = {}   # M6 provenance, only when used, so a legacy bank's file is unchanged
    if receiver_radius is not None:
        extra["receiver_radius"] = np.float32(receiver_radius)
    if seed_namespace is not None:
        extra["seed_namespace"] = np.array(seed_namespace)
    np.savez_compressed(path, speech=sp, noise=nz, rt60=rt, armoured=arm, **extra)
    if parts_dir is not None:
        shutil.rmtree(parts_dir)


class RirBank:
    """Memory-mapped so DataLoader workers (spawned on Windows) share one page cache instead of 2 GB each."""
    KEYS = ("speech", "noise", "rt60")

    def __init__(self, path: Path):
        path = Path(path)
        parts = {k: path.with_name(f"{path.stem}.{k}.npy") for k in self.KEYS}
        if not all(f.exists() for f in parts.values()):
            z = np.load(path)
            for k, f in parts.items():
                # The temp name must be unique per writer. It used to be a fixed ".tmp.npy", so two
                # trainers starting together on a fresh box wrote the same temp file and one's
                # os.replace pulled it out from under the other - leaving a truncated .npy behind and
                # a later mmap failing with "length is greater than file size". run_r6.sh's
                # ARMS_PARALLEL starts arms concurrently by design, so this was reachable in normal use.
                # uuid rather than the pid: threads in one process race just as happily.
                tmp = f.with_suffix(f".{uuid.uuid4().hex}.tmp.npy")
                try:
                    np.save(tmp, z[k])
                    try:
                        os.replace(tmp, f)   # replace is atomic within a directory
                    except PermissionError:
                        # Windows refuses to replace a file another writer already published and mmapped.
                        # That file came from the same npz through the same atomic replace, so it is complete: keep it.
                        if not f.exists():
                            raise
                finally:
                    if tmp.exists():
                        tmp.unlink()
        self.speech, self.noise, self.rt60 = (np.load(parts[k], mmap_mode="r") for k in self.KEYS)

    def __len__(self):
        return len(self.rt60)

    def sample(self, rng: np.random.Generator) -> dict:
        i = int(rng.integers(len(self)))
        return {"speech": self.speech[i], "noise": self.noise[i], "rt60": float(self.rt60[i])}
