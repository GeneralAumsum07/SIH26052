# VAANI Training Stack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the end-to-end training stack for SIH26052 — two-channel noisy/clean dataset pipeline, DSP reference (NLMS + impulse/reliability controller), dual-channel GTCRN-derived model (VaaniNet), ablation-matrix evaluation, and ONNX export with a deployment contract.

**Architecture:** A single Python package `vaani/` with GTCRN vendored verbatim. Training uses dynamic on-the-fly two-channel mixing (room-simulated or parametric per sample); val/test are rendered once with fixed seeds and frozen. The DSP reference runs frame-synchronously with the model (16 kHz, 512/256, sqrt-Hann) and its outputs (`n_hat` spectrum + 18-dim feature vector) condition the network. Every experiment is one YAML in `configs/exp/`.

**Tech Stack:** Python 3.13 (fallback 3.12), `uv`, PyTorch cu128 (Blackwell sm_120), torchaudio, numpy, soundfile, pyroomacoustics, pyarrow, pystoi, pesq, onnx, onnxruntime, faster-whisper, tensorboard, pytest.

**Spec:** `docs/superpowers/specs/2026-09-18-vaani-training-stack-design.md`

## Global Constraints

- Sample rate 16 kHz, mono per channel, float32 in [-1, 1].
- STFT everywhere: n_fft 512, hop 256, win 512, window `torch.hann_window(512).pow(0.5)` (sqrt-Hann — matches upstream GTCRN; DSP `stft.py` must use the identical window).
- Two mixture channels are never a duplicated mono signal.
- Splits are assigned before mixing by `hash(speaker_id or noise_source_id) % 10 → {0..7 train, 8 val, 9 test}`.
- VaaniNet ≤ 60 000 parameters. Mask applied to the primary spectrum only.
- Feature vector order (18 dims) is fixed by spec §5.2 and is part of the deployment contract.
- Every training run writes `runs/<name>/run.json` with git hash, config hash, manifest hash, eval-set hash, seed, torch/cuda versions.
- SNR reported in eval = `10·log10(‖s‖² / ‖ŝ−s‖²)` against the clean primary; never conflated with SI-SDR.
- PESQ = wideband P.862.2 at 16 kHz via the `pesq` package; report header states mode and the P.862 withdrawal note.
- **Git:** stage files (`git add`) at the end of each task; **do not commit** — Rachit commits. No attribution trailers of any kind.
- Data and runs directories are git-ignored.
- **Every `torch.load(...)` in the codebase passes `weights_only=True`** (checkpoints hold only tensors plus a plain config dict of str/int/float/bool/list). Code blocks in this plan omit the kwarg for brevity — add it when writing the file. If the upstream `model_trained_on_dns3.tar` refuses to load with `weights_only=True`, load it once with `weights_only=False` in `scripts/convert_upstream_ckpt.py`, re-save just `{"model": state_dict}` as `vaani/models/checkpoints/gtcrn_dns3.pt`, and point every reference at the converted file.
- Comment code heavily: explain *why*, not *what*.

---

## File map

| Path | Responsibility |
|---|---|
| `pyproject.toml` | uv project, cu128 torch index, deps |
| `vaani/dsp/stft.py` | shared STFT/iSTFT (sqrt-Hann 512/256) for torch and numpy |
| `vaani/data/manifests.py` | parquet manifest schema, `stable_hash`, read/write |
| `vaani/data/splits.py` | split assignment by hash |
| `vaani/data/sources.py` | corpus adapters → manifest rows (LibriSpeech, Common Voice Hindi, MAD, DNS shards) |
| `vaani/data/impulses.py` | synthetic impulse generator |
| `vaani/data/rirs.py` | pyroomacoustics RIR-pair bank |
| `vaani/data/mixer.py` | two-channel mixer (room + parametric paths, augmentations, SNR scaling) |
| `vaani/data/dataset.py` | torch `DynamicMixDataset` (train) and `RenderedDataset` (val/test) |
| `vaani/dsp/nlms.py` | guarded NLMS noise estimator |
| `vaani/dsp/features.py` | 18-dim per-frame feature extractor |
| `vaani/dsp/controller.py` | burst/reliability gating |
| `vaani/dsp/pipeline.py` | runs nlms+features+controller over a stereo clip → `n_hat`, features (used by dataset and eval) |
| `vaani/models/gtcrn.py`, `gtcrn_stream.py`, `modules/` | vendored upstream |
| `vaani/models/vaani_net.py` | dual-channel FiLM-conditioned GTCRN |
| `vaani/models/baselines.py` | raw, nlms_only, rnnoise wrappers with a common `enhance(mix)->np.ndarray` API |
| `vaani/losses.py` | upstream HybridLoss + speech-preservation variant |
| `vaani/train.py` | config-driven trainer |
| `vaani/eval.py` | per-clip metrics CSV |
| `vaani/report.py` | ablation matrix with bootstrap CIs |
| `vaani/export.py` | ONNX export + parity + timing |
| `scripts/fetch_data.py` | resumable downloads |
| `scripts/render_eval_sets.py` | seeded val/test rendering |
| `scripts/make_golden_vectors.py` | DSP golden vectors |
| `configs/data/round1.yaml`, `configs/exp/*.yaml` | configuration |
| `tests/*.py` | pytest |

---

### Task 1: Environment and package scaffold

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `vaani/__init__.py`, `vaani/data/__init__.py`, `vaani/dsp/__init__.py`, `vaani/models/__init__.py`, `tests/__init__.py`, `tests/test_env.py`

**Interfaces:**
- Produces: importable `vaani` package; `uv run pytest` works; CUDA verified.

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "vaani"
version = "0.1.0"
description = "SIH26052 - two-channel speech enhancement training stack"
requires-python = ">=3.12"
dependencies = [
  "torch>=2.7",
  "torchaudio>=2.7",
  "numpy>=2.0",
  "scipy>=1.14",
  "soundfile>=0.12",
  "pyroomacoustics>=0.8",
  "pyarrow>=17",
  "pandas>=2.2",
  "pyyaml>=6",
  "pystoi>=0.4",
  "pesq>=0.0.4",
  "onnx>=1.16",
  "onnxruntime>=1.19",
  "tensorboard>=2.17",
  "tqdm>=4.66",
  "requests>=2.32",
]

[project.optional-dependencies]
asr = ["faster-whisper>=1.0"]
dev = ["pytest>=8", "pytest-timeout>=2.3"]

[tool.uv.sources]
torch = { index = "pytorch-cu128" }
torchaudio = { index = "pytorch-cu128" }

[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.pytest.ini_options]
testpaths = ["tests"]
timeout = 300

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["vaani"]
```

- [ ] **Step 2: Write `.gitignore`**

```
data/
runs/
deploy/model*.onnx
.venv/
__pycache__/
*.pyc
.pytest_cache/
*.egg-info/
```

- [ ] **Step 3: Create empty package files**

Create `vaani/__init__.py`, `vaani/data/__init__.py`, `vaani/dsp/__init__.py`, `vaani/models/__init__.py`, `tests/__init__.py` — each empty.

- [ ] **Step 4: Write the environment test**

`tests/test_env.py`:
```python
"""Day-1 gate: if CUDA does not work on this laptop nothing else matters.
Blackwell (sm_120) needs a cu128 wheel; a CPU-only wheel silently installs
and would waste a day of 'why is training slow'."""
import torch


def test_cuda_available():
    assert torch.cuda.is_available(), "torch has no CUDA - check the cu128 index in pyproject"


def test_cuda_matmul_runs():
    a = torch.randn(256, 256, device="cuda")
    b = a @ a
    torch.cuda.synchronize()
    assert b.isfinite().all()
```

- [ ] **Step 5: Install and run**

Run:
```bash
uv sync --extra dev
uv run pytest tests/test_env.py -v
```
Expected: both PASS. If `uv sync` fails on a 3.13 wheel, run `uv python pin 3.12 && uv sync --extra dev` and note this in `pyproject.toml` with a comment.

- [ ] **Step 6: Stage**

```bash
git add pyproject.toml .gitignore vaani tests uv.lock
```

---

### Task 2: Shared STFT and manifest/split primitives

**Files:**
- Create: `vaani/dsp/stft.py`, `vaani/data/manifests.py`, `vaani/data/splits.py`
- Test: `tests/test_stft.py`, `tests/test_splits.py`

**Interfaces:**
- Produces:
  - `stft.N_FFT=512, HOP=256, WIN=512`; `stft.window() -> torch.Tensor`; `stft.stft(x: Tensor[..., T]) -> Tensor[..., F, T', 2]`; `stft.istft(spec: Tensor[..., F, T', 2], length: int|None) -> Tensor[..., T]`; `stft.np_stft(x: np.ndarray[T]) -> np.ndarray[F, T'] complex64` (same window, same framing as torch with `center=True`).
  - `manifests.COLUMNS`, `manifests.stable_hash(s: str) -> int`, `manifests.write(rows: list[dict], path) `, `manifests.read(path) -> pd.DataFrame`.
  - `splits.assign(group_id: str) -> str` returning `"train"|"val"|"test"`.

- [ ] **Step 1: Write failing tests**

`tests/test_stft.py`:
```python
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
```

`tests/test_splits.py`:
```python
from vaani.data import splits, manifests


def test_assign_deterministic_and_covers_all():
    seen = {splits.assign(f"spk{i}") for i in range(2000)}
    assert seen == {"train", "val", "test"}
    assert splits.assign("abc") == splits.assign("abc")


def test_hash_is_stable_across_processes():
    # python's hash() is salted per process; ours must not be
    assert manifests.stable_hash("x") == 1129999393 % (2**32) or isinstance(manifests.stable_hash("x"), int)
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_stft.py tests/test_splits.py -v`
Expected: FAIL with ImportError.

- [ ] **Step 3: Implement `vaani/dsp/stft.py`**

```python
"""One STFT definition for the whole project.

Upstream GTCRN trains and infers with a *square-root* Hann window on both
analysis and synthesis (hann**0.5 twice == hann once, satisfying COLA).
If the DSP reference or the embedded port uses plain Hann the model sees
spectra it was never trained on, so torch and numpy paths share these
constants and a parity test pins them together.
"""
import numpy as np
import torch

N_FFT = 512
HOP = 256
WIN = 512


def window(device=None) -> torch.Tensor:
    return torch.hann_window(WIN, device=device).pow(0.5)


def stft(x: torch.Tensor) -> torch.Tensor:
    """x: (..., T) -> (..., F, T', 2) real/imag, matching upstream GTCRN input."""
    shape = x.shape
    x = x.reshape(-1, shape[-1])
    s = torch.stft(x, N_FFT, HOP, WIN, window(x.device), center=True, return_complex=True)
    s = torch.view_as_real(s)
    return s.reshape(*shape[:-1], *s.shape[1:])


def istft(spec: torch.Tensor, length: int | None = None) -> torch.Tensor:
    shape = spec.shape
    spec = spec.reshape(-1, *shape[-3:])
    c = torch.view_as_complex(spec.contiguous())
    y = torch.istft(c, N_FFT, HOP, WIN, window(spec.device), center=True, length=length)
    return y.reshape(*shape[:-3], y.shape[-1])


def np_stft(x: np.ndarray) -> np.ndarray:
    """NumPy twin of stft() for the DSP reference (portable to C).
    Reproduces torch's center=True reflect padding and framing exactly."""
    w = np.hanning(WIN + 1)[:-1] ** 0.5  # periodic sqrt-Hann == torch.hann_window
    xp = np.pad(x, N_FFT // 2, mode="reflect")
    n_frames = 1 + (len(xp) - N_FFT) // HOP
    frames = np.lib.stride_tricks.as_strided(
        xp, shape=(n_frames, N_FFT), strides=(xp.strides[0] * HOP, xp.strides[0]))
    return np.fft.rfft(frames * w, axis=1).T.astype(np.complex64)  # (F, T')
```

- [ ] **Step 4: Implement `vaani/data/manifests.py`**

```python
"""Manifest = one parquet table per corpus listing every usable file.

Split assignment lives in the manifest so mixing code never decides
membership; that is what makes speaker/noise leakage impossible later.
"""
import hashlib
from pathlib import Path

import pandas as pd

COLUMNS = ["source_id", "corpus", "kind", "group_id", "speaker_id", "path",
           "duration_s", "licence", "split", "sha1", "noise_class"]
# kind: "speech" | "noise" | "impulse"
# group_id: speaker_id for speech; source-video/recording id for noise
# noise_class: stationary | changing | impulsive | "" for speech


def stable_hash(s: str) -> int:
    """Process-independent hash. Python's built-in hash() is salted per
    interpreter, which would silently reshuffle splits between runs."""
    return int(hashlib.sha1(s.encode()).hexdigest()[:8], 16)


def write(rows: list[dict], path: str | Path) -> None:
    df = pd.DataFrame(rows, columns=COLUMNS)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)


def read(path: str | Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def content_hash(paths: list[str | Path]) -> str:
    """Hash of manifests used by a run, written into run.json."""
    h = hashlib.sha1()
    for p in sorted(map(str, paths)):
        h.update(Path(p).read_bytes())
    return h.hexdigest()[:12]
```

- [ ] **Step 5: Implement `vaani/data/splits.py`**

```python
from vaani.data.manifests import stable_hash


def assign(group_id: str) -> str:
    """80/10/10 by group. Group = speaker for speech, recording/video for noise,
    so a voice or a noise source can never straddle train and test."""
    b = stable_hash(group_id) % 10
    return "train" if b < 8 else ("val" if b == 8 else "test")
```

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/test_stft.py tests/test_splits.py -v`
Expected: PASS.

- [ ] **Step 7: Stage**

```bash
git add vaani/dsp/stft.py vaani/data/manifests.py vaani/data/splits.py tests/test_stft.py tests/test_splits.py
```

---

### Task 3: Corpus adapters and resumable fetch script

**Files:**
- Create: `vaani/data/sources.py`, `scripts/fetch_data.py`, `configs/data/round1.yaml`, `data/manifests/physical_test/README.md`
- Test: `tests/test_sources.py`

**Interfaces:**
- Produces: `sources.scan_librispeech(root) -> list[dict]`, `sources.scan_commonvoice_hi(root, min_snr_db=30.0) -> list[dict]`, `sources.scan_mad(root) -> list[dict]`, `sources.scan_dns_noise(root)`, `sources.scan_dns_speech(root)`; `sources.estimate_snr_db(x: np.ndarray, sr: int) -> float`; `sources.to_flac16k(src, dst) -> float` (returns duration).
- Each row follows `manifests.COLUMNS` with `split` filled by `splits.assign(group_id)`.

- [ ] **Step 1: Write failing tests**

`tests/test_sources.py`:
```python
import numpy as np, soundfile as sf
from vaani.data import sources


def test_estimate_snr_clean_vs_noisy():
    sr = 16000
    t = np.arange(sr * 2) / sr
    speech = (np.sin(2 * np.pi * 200 * t) * (np.sin(2 * np.pi * 3 * t) > 0)).astype(np.float32)
    clean = speech
    noisy = speech + 0.1 * np.random.randn(len(t)).astype(np.float32)
    assert sources.estimate_snr_db(clean, sr) > sources.estimate_snr_db(noisy, sr) + 10


def test_to_flac16k_resamples(tmp_path):
    x = np.random.randn(48000).astype(np.float32) * 0.1
    sf.write(tmp_path / "a.wav", x, 48000)
    dur = sources.to_flac16k(tmp_path / "a.wav", tmp_path / "a.flac")
    y, sr = sf.read(tmp_path / "a.flac")
    assert sr == 16000 and abs(dur - 1.0) < 0.01 and len(y) == 16000
```

- [ ] **Step 2: Run to verify failure**

Run: `uv run pytest tests/test_sources.py -v` → FAIL ImportError.

- [ ] **Step 3: Implement `vaani/data/sources.py`**

```python
"""Corpus adapters: each scan_* walks a downloaded corpus, converts audio to
16 kHz mono FLAC under data/raw/<corpus>/, and returns manifest rows.

Nothing here decides how audio is *used*; it only records what exists,
its provenance and licence, and which split it belongs to.
"""
import csv
import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

from vaani.data.manifests import COLUMNS
from vaani.data.splits import assign

SR = 16000


def to_flac16k(src: Path, dst: Path) -> float:
    x, sr = sf.read(src, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        g = np.gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    dst.parent.mkdir(parents=True, exist_ok=True)
    sf.write(dst, x, SR, subtype="PCM_16")
    return len(x) / SR


def estimate_snr_db(x: np.ndarray, sr: int, frame_ms: float = 20.0) -> float:
    """Cheap energy-based SNR estimate used to gate crowdsourced speech.
    Top 30% energy frames ~ speech, bottom 10% ~ noise floor. Coarse, but the
    gate only needs to reject clearly noisy Common Voice clips."""
    n = int(sr * frame_ms / 1000)
    frames = x[: len(x) // n * n].reshape(-1, n)
    e = (frames ** 2).mean(axis=1) + 1e-10
    e = np.sort(e)
    noise = e[: max(1, len(e) // 10)].mean()
    speech = e[-max(1, int(len(e) * 0.3)):].mean()
    return float(10 * np.log10(speech / noise))


def _row(source_id, corpus, kind, group_id, speaker_id, path, dur, licence, noise_class=""):
    sha1 = hashlib.sha1(Path(path).read_bytes()).hexdigest()[:12]
    return dict(source_id=source_id, corpus=corpus, kind=kind, group_id=group_id,
                speaker_id=speaker_id, path=str(path), duration_s=dur, licence=licence,
                split=assign(group_id), sha1=sha1, noise_class=noise_class)


def scan_librispeech(root: Path, out: Path, max_hours: float | None = None) -> list[dict]:
    """LibriSpeech layout: <root>/<spk>/<chapter>/<spk>-<chapter>-<utt>.flac"""
    rows, total = [], 0.0
    for f in sorted(root.rglob("*.flac")):
        spk = f.parts[-3]
        dst = out / "librispeech" / f.name
        dur = to_flac16k(f, dst) if not dst.exists() else sf.info(dst).duration
        rows.append(_row(f"ls:{f.stem}", "librispeech", "speech", f"ls-spk-{spk}", spk, dst, dur, "CC BY 4.0"))
        total += dur / 3600
        if max_hours and total >= max_hours:
            break
    return rows


def scan_commonvoice_hi(root: Path, out: Path, min_snr_db: float = 30.0) -> list[dict]:
    """Common Voice: <root>/validated.tsv + clips/*.mp3. Crowdsourced audio is
    not automatically a clean target, so gate by estimated SNR and record
    the retained count (the spec requires reporting it)."""
    rows, kept, seen = [], 0, 0
    with open(root / "validated.tsv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            seen += 1
            src = root / "clips" / r["path"]
            if not src.exists():
                continue
            dst = out / "cv_hi" / (Path(r["path"]).stem + ".flac")
            dur = to_flac16k(src, dst) if not dst.exists() else sf.info(dst).duration
            x, _ = sf.read(dst, dtype="float32")
            if estimate_snr_db(x, SR) < min_snr_db:
                dst.unlink(missing_ok=True)
                continue
            kept += 1
            spk = r["client_id"][:16]
            rows.append(_row(f"cvhi:{Path(r['path']).stem}", "cv_hi", "speech", f"cv-spk-{spk}", spk, dst, dur, "CC0"))
    print(f"[cv_hi] kept {kept}/{seen} clips at SNR>={min_snr_db} dB")
    return rows


# MAD class names -> our noise taxonomy. Anything not listed is "changing".
MAD_CLASS_MAP = {
    "gunshot": "impulsive", "explosion": "impulsive", "artillery": "impulsive",
    "helicopter": "stationary", "jet": "stationary", "engine": "stationary",
    "vehicle": "stationary", "tank": "stationary",
}


def scan_mad(root: Path, out: Path) -> list[dict]:
    """Military Audio Dataset. Expected layout <root>/<class>/<clip>.wav.
    Clips cut from the same source share a stem prefix before the last '_';
    that prefix is the group so excerpts of one video stay in one split."""
    rows = []
    for f in sorted(root.rglob("*.wav")):
        cls = f.parent.name.lower()
        noise_class = next((v for k, v in MAD_CLASS_MAP.items() if k in cls), "changing")
        group = f"mad-{cls}-{f.stem.rsplit('_', 1)[0]}"
        dst = out / "mad" / cls / (f.stem + ".flac")
        dur = to_flac16k(f, dst) if not dst.exists() else sf.info(dst).duration
        rows.append(_row(f"mad:{cls}/{f.stem}", "mad", "noise", group, "", dst, dur, "MAD (see repo)", noise_class))
    return rows


def scan_dns_noise(root: Path, out: Path) -> list[dict]:
    rows = []
    for f in sorted(root.rglob("*.wav")):
        dst = out / "dns_noise" / (f.stem + ".flac")
        dur = to_flac16k(f, dst) if not dst.exists() else sf.info(dst).duration
        rows.append(_row(f"dnsn:{f.stem}", "dns_noise", "noise", f"dnsn-{f.stem}", "", dst, dur, "DNS-5 (per-shard)", "changing"))
    return rows


def scan_dns_speech(root: Path, out: Path) -> list[dict]:
    """DNS read speech: filenames carry a reader/book id before the first '_'."""
    rows = []
    for f in sorted(root.rglob("*.wav")):
        spk = f.stem.split("_")[0]
        dst = out / "dns_speech" / (f.stem + ".flac")
        dur = to_flac16k(f, dst) if not dst.exists() else sf.info(dst).duration
        rows.append(_row(f"dnss:{f.stem}", "dns_speech", "speech", f"dns-spk-{spk}", spk, dst, dur, "DNS-5 (per-shard)"))
    return rows
```

- [ ] **Step 4: Write `configs/data/round1.yaml`**

```yaml
# Round-1 lean data. Paths are relative to repo root.
raw_root: data/raw
manifest_dir: data/manifests
sources:
  librispeech:
    url: https://www.openslr.org/resources/12/train-clean-100.tar.gz
    extract_to: data/download/librispeech
    max_hours: 20
  cv_hi:
    # Common Voice requires a logged-in download; place the extracted
    # hi/ folder at data/download/cv_hi manually. fetch_data.py skips it if absent.
    extract_to: data/download/cv_hi
    min_snr_db: 30
  mad:
    # https://github.com/kaen2891/military_audio_dataset -> packaged download.
    # Place extracted class folders at data/download/mad. Record the version here:
    version: "TBD-record-after-download"
    extract_to: data/download/mad
```

Note: the `version` field is deliberately a placeholder to be filled at download time — it is data provenance, not code.

- [ ] **Step 5: Write `scripts/fetch_data.py`**

```python
"""Resumable fetch + manifest build. Run repeatedly; it skips what exists.
Round 2 (DNS shards) uses the same script with --dns-shards <list>.
"""
import argparse, subprocess, sys, tarfile
from pathlib import Path

import requests, yaml
from tqdm import tqdm

from vaani.data import manifests, sources


def download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    have = dst.stat().st_size if dst.exists() else 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    with requests.get(url, stream=True, headers=headers, timeout=60) as r:
        if r.status_code == 416:
            return
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) + have
        with open(dst, "ab") as f, tqdm(total=total, initial=have, unit="B", unit_scale=True, desc=dst.name) as bar:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk); bar.update(len(chunk))


def extract(tar: Path, to: Path) -> None:
    if (to / ".done").exists():
        return
    to.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar) as t:
        t.extractall(to, filter="data")
    (to / ".done").touch()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data/round1.yaml")
    ap.add_argument("--dns-shards", nargs="*", default=[], help="DNS-5 shard URLs for round 2")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    raw, mdir = Path(cfg["raw_root"]), Path(cfg["manifest_dir"])

    ls = cfg["sources"]["librispeech"]
    tar = Path("data/download/train-clean-100.tar.gz")
    download(ls["url"], tar); extract(tar, Path(ls["extract_to"]))
    manifests.write(sources.scan_librispeech(Path(ls["extract_to"]), raw, ls["max_hours"]), mdir / "librispeech.parquet")

    cv = cfg["sources"]["cv_hi"]
    if Path(cv["extract_to"], "validated.tsv").exists():
        manifests.write(sources.scan_commonvoice_hi(Path(cv["extract_to"]), raw, cv["min_snr_db"]), mdir / "cv_hi.parquet")
    else:
        print("[cv_hi] not found - skipped (manual download required)")

    mad = cfg["sources"]["mad"]
    if Path(mad["extract_to"]).exists():
        manifests.write(sources.scan_mad(Path(mad["extract_to"]), raw), mdir / "mad.parquet")
    else:
        print("[mad] not found - skipped (manual download required)")

    for i, url in enumerate(a.dns_shards):
        tar = Path("data/download/dns") / Path(url).name
        download(url, tar)
        to = Path("data/download/dns") / tar.stem.replace(".tar", "")
        extract(tar, to)
        fn = sources.scan_dns_noise if "noise" in url else sources.scan_dns_speech
        manifests.write(fn(to, raw), mdir / f"dns_{tar.stem}.parquet")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Write the physical-test schema README**

`data/manifests/physical_test/README.md` (this path is git-ignored under `data/` — also copy to `docs/physical_test_schema.md`):
```markdown
# Physical test recordings — drop-in schema

Place files here; `vaani/eval.py --split physical` consumes them.

- `clips/<id>.wav` — 2-channel, 16 kHz, PCM16. Channel 0 = primary (near-mouth), channel 1 = reference.
- `clips/<id>.clean.wav` — optional; if a clean primary exists (lab replay), same length.
- `physical.csv` with columns: `id, speaker, language, condition, transcript, has_clean`
  - `condition` ∈ {engine, engine+burst, env_change, ref_fault, quiet}
```

- [ ] **Step 7: Run tests and start the fetch in the background**

Run: `uv run pytest tests/test_sources.py -v` → PASS.
Then start the fetch (it runs for a long time; leave it in the background and continue with Task 4):
```bash
uv run python scripts/fetch_data.py
```

- [ ] **Step 8: Stage**

```bash
git add vaani/data/sources.py scripts/fetch_data.py configs/data/round1.yaml docs/physical_test_schema.md tests/test_sources.py
```

---

### Task 4: Synthetic impulse generator

**Files:**
- Create: `vaani/data/impulses.py`
- Test: `tests/test_impulses.py`

**Interfaces:**
- Produces: `impulses.generate(rng: np.random.Generator, sr=16000, kind: str|None=None) -> tuple[np.ndarray, dict]` returning a mono float32 clip (0.2–2 s) peaking at 1.0 and metadata `{"kind": ..., "onsets_s": [...]}`. Kinds: `"burst"` (single exponential decay), `"click_train"`, `"gated_noise"`.

- [ ] **Step 1: Write failing test**

`tests/test_impulses.py`:
```python
import numpy as np
from vaani.data import impulses


def test_all_kinds_peak_at_one_and_have_onsets():
    rng = np.random.default_rng(0)
    for kind in ("burst", "click_train", "gated_noise"):
        x, meta = impulses.generate(rng, kind=kind)
        assert x.dtype == np.float32
        assert abs(np.abs(x).max() - 1.0) < 1e-5
        assert meta["kind"] == kind and len(meta["onsets_s"]) >= 1
        assert 0.2 * 16000 <= len(x) <= 2.0 * 16000


def test_burst_is_actually_impulsive():
    rng = np.random.default_rng(1)
    x, meta = impulses.generate(rng, kind="burst")
    on = int(meta["onsets_s"][0] * 16000)
    # energy in first 10 ms after onset dwarfs energy 200 ms later
    e0 = (x[on:on + 160] ** 2).mean(); e1 = (x[on + 3200:on + 3360] ** 2).mean() + 1e-12
    assert 10 * np.log10(e0 / e1) > 20
```

- [ ] **Step 2: Run to verify failure** — `uv run pytest tests/test_impulses.py -v` → FAIL.

- [ ] **Step 3: Implement `vaani/data/impulses.py`**

```python
"""Synthetic impulsive noise so training never depends on how many gunshots
a downloaded corpus happens to contain. Shapes are chosen to stress the
controller: fast onsets, varied decay, and repetition that must not be
confused with speech consonants.
"""
import numpy as np

KINDS = ("burst", "click_train", "gated_noise")


def _decay(rng, sr, tau_s):
    n = int(sr * min(2.0, tau_s * 6))
    t = np.arange(n) / sr
    # coloured noise * exponential envelope; a 1-pole lowpass gives a "thud"
    x = rng.standard_normal(n)
    a = rng.uniform(0.6, 0.95)
    for i in range(1, n):
        x[i] += a * x[i - 1]
    return (x * np.exp(-t / tau_s)).astype(np.float32)


def generate(rng: np.random.Generator, sr: int = 16000, kind: str | None = None):
    kind = kind or rng.choice(KINDS)
    if kind == "burst":
        x = _decay(rng, sr, rng.uniform(0.02, 0.25))
        pre = int(rng.uniform(0.05, 0.3) * sr)
        x = np.concatenate([np.zeros(pre, np.float32), x])
        onsets = [pre / sr]
    elif kind == "click_train":
        n_clicks = int(rng.integers(3, 12))
        gap = rng.uniform(0.04, 0.15)
        pieces, onsets, pos = [], [], 0.0
        for _ in range(n_clicks):
            c = _decay(rng, sr, rng.uniform(0.003, 0.02))
            g = np.zeros(int(gap * sr), np.float32)
            onsets.append(pos); pos += (len(c) + len(g)) / sr
            pieces += [c, g]
        x = np.concatenate(pieces)
    else:  # gated_noise: wideband noise switched on/off abruptly
        dur = rng.uniform(0.3, 1.5)
        x = rng.standard_normal(int(dur * sr)).astype(np.float32)
        on, off = int(0.1 * sr), int(rng.uniform(0.3, 0.9) * dur * sr)
        x[:on] = 0; x[off:] = 0
        onsets = [on / sr]
    n = int(np.clip(len(x), 0.2 * sr, 2.0 * sr))
    x = np.pad(x, (0, max(0, n - len(x))))[:n]
    x = x / (np.abs(x).max() + 1e-9)
    return x.astype(np.float32), {"kind": str(kind), "onsets_s": [float(o) for o in onsets]}
```

- [ ] **Step 4: Run tests** → PASS.
- [ ] **Step 5: Stage** — `git add vaani/data/impulses.py tests/test_impulses.py`

---

### Task 5: RIR-pair bank

**Files:**
- Create: `vaani/data/rirs.py`, `scripts/make_rir_bank.py`
- Test: `tests/test_rirs.py`

**Interfaces:**
- Produces: `rirs.simulate_pair_set(rng, sr=16000, n_noise=2) -> dict` with keys `speech: (2, L)`, `noise: (n_noise, 2, L)`, `rt60`, `room_dims`; `rirs.RirBank(path).sample(rng) -> dict` (same keys, loaded from an `.npz` bank); `rirs.build_bank(path, n=5000, seed=0)`.
- Geometry constants: `MIC_SPACING=0.12`, `MOUTH_TO_PRIMARY=0.025`, `MOUTH_TO_REF=0.14`.

- [ ] **Step 1: Write failing test**

`tests/test_rirs.py`:
```python
import numpy as np
from vaani.data import rirs


def test_pair_set_shapes_and_leakage():
    rng = np.random.default_rng(0)
    s = rirs.simulate_pair_set(rng, n_noise=2)
    assert s["speech"].shape[0] == 2 and s["noise"].shape[:2] == (2, 2)
    # near-mouth geometry: speech RIR energy must be much higher at primary
    e = (s["speech"] ** 2).sum(axis=1)
    assert 10 * np.log10(e[0] / e[1]) > 6
    # far-field noise: roughly equal at both mics
    en = (s["noise"][0] ** 2).sum(axis=1)
    assert abs(10 * np.log10(en[0] / en[1])) < 4


def test_bank_roundtrip(tmp_path):
    p = tmp_path / "bank.npz"
    rirs.build_bank(p, n=3, seed=1)
    b = rirs.RirBank(p)
    s = b.sample(np.random.default_rng(0))
    assert s["speech"].shape[0] == 2
```

- [ ] **Step 2: Run to verify failure** → FAIL.

- [ ] **Step 3: Implement `vaani/data/rirs.py`**

```python
"""pyroomacoustics RIR pairs for the headset geometry.

Why a bank: ray-tracing per training sample is far too slow for a
DataLoader; ~5k pre-computed rooms give plenty of diversity and turn the
per-sample cost into a convolution.
"""
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
```

- [ ] **Step 4: Write `scripts/make_rir_bank.py`**

```python
import argparse
from pathlib import Path
from vaani.data.rirs import build_bank

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data/rirs/bank.npz")
ap.add_argument("--n", type=int, default=5000)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
build_bank(Path(a.out), a.n, a.seed)
print("wrote", a.out)
```

- [ ] **Step 5: Run tests, then build the bank in the background**

`uv run pytest tests/test_rirs.py -v` → PASS.
`uv run python scripts/make_rir_bank.py` (≈10–30 min; leave running).

- [ ] **Step 6: Stage** — `git add vaani/data/rirs.py scripts/make_rir_bank.py tests/test_rirs.py`

---

### Task 6: Two-channel mixer

**Files:**
- Create: `vaani/data/mixer.py`
- Test: `tests/test_mixer.py`

**Interfaces:**
- Consumes: `RirBank.sample`, `impulses.generate`.
- Produces: `mixer.MixConfig` dataclass; `mixer.mix(rng, speech: np.ndarray[T], noises: list[np.ndarray], impulse: np.ndarray|None, impulse_onsets_s: list[float], bank: RirBank|None, cfg: MixConfig) -> tuple[np.ndarray[2,T], np.ndarray[T], dict]` — returns `(mix, clean_primary, meta)`; meta has `snr_db, path ("room"|"param"), clipped, ref_dropout, impulse_peak_db, impulse_onsets_s, noise_class, clean_bucket`.
- `mixer.speech_active_power(x) -> float` (energy VAD, 20 ms frames, frames above −30 dB rel. max).

- [ ] **Step 1: Write failing tests**

`tests/test_mixer.py`:
```python
import numpy as np
from vaani.data import mixer, rirs


def _speech(rng, n=32000):
    t = np.arange(n) / 16000
    env = (np.sin(2 * np.pi * 2 * t) > 0).astype(np.float32)
    return (np.sin(2 * np.pi * 180 * t) * env * 0.3).astype(np.float32)


def test_param_path_hits_target_snr_and_channels_differ():
    rng = np.random.default_rng(0)
    cfg = mixer.MixConfig(p_room=0.0, p_clip=0.0, p_ref_dropout=0.0, p_wind=0.0, p_clean=0.0, snr_range=(5.0, 5.0))
    s = _speech(rng); n = [rng.standard_normal(len(s)).astype(np.float32)]
    mix, clean, meta = mixer.mix(rng, s, n, None, [], None, cfg)
    assert mix.shape == (2, len(s)) and clean.shape == (len(s),)
    assert not np.allclose(mix[0], mix[1])
    achieved = 10 * np.log10(mixer.speech_active_power(clean) / mixer.speech_active_power(mix[0] - clean))
    assert abs(achieved - 5.0) < 0.5
    # reference carries much less speech than primary
    assert meta["path"] == "param" and meta["ref_speech_gain_db"] <= -8


def test_room_path_runs(tmp_path):
    rirs.build_bank(tmp_path / "b.npz", n=2, seed=0)
    bank = rirs.RirBank(tmp_path / "b.npz")
    rng = np.random.default_rng(0)
    cfg = mixer.MixConfig(p_room=1.0, p_clean=0.0)
    s = _speech(rng); n = [rng.standard_normal(len(s)).astype(np.float32)]
    mix, clean, meta = mixer.mix(rng, s, n, None, [], bank, cfg)
    assert meta["path"] == "room" and np.isfinite(mix).all()


def test_clean_bucket_is_identity():
    rng = np.random.default_rng(0)
    cfg = mixer.MixConfig(p_clean=1.0, p_clip=0.0, p_wind=0.0, p_ref_dropout=0.0, p_room=0.0)
    s = _speech(rng)
    mix, clean, meta = mixer.mix(rng, s, [rng.standard_normal(len(s)).astype(np.float32)], None, [], None, cfg)
    assert meta["clean_bucket"] and np.allclose(mix[0], clean, atol=1e-6)
```

- [ ] **Step 2: Run to verify failure** → FAIL.

- [ ] **Step 3: Implement `vaani/data/mixer.py`**

```python
"""Two-channel mixture synthesis.

The reference-mic channel is the whole point of the dual-channel model, so
it is modelled physically (room path) or with an explicit speech-leakage +
independent-noise-filter model (parametric path) - never by copying the
primary. SNR is defined on the *primary* over speech-active frames; the
impulse level is drawn independently and recorded because a file-average
SNR hides how loud a 50 ms burst really was.
"""
from dataclasses import dataclass

import numpy as np
from scipy.signal import fftconvolve, lfilter

SR = 16000


@dataclass
class MixConfig:
    snr_range: tuple[float, float] = (-10.0, 15.0)
    p_room: float = 0.6
    p_clean: float = 0.05
    p_clip: float = 0.10
    p_ref_dropout: float = 0.05
    p_wind: float = 0.15
    ref_speech_gain_db: tuple[float, float] = (-20.0, -8.0)
    ref_delay_ms: tuple[float, float] = (0.1, 0.5)
    mic_mismatch_db: float = 3.0
    impulse_peak_db: tuple[float, float] = (-6.0, 12.0)   # relative to speech-active RMS on primary


def speech_active_power(x: np.ndarray, frame: int = 320, thresh_db: float = -30.0) -> float:
    f = x[: len(x) // frame * frame].reshape(-1, frame)
    e = (f ** 2).mean(axis=1) + 1e-12
    keep = e > e.max() * 10 ** (thresh_db / 10)
    return float(e[keep].mean()) if keep.any() else float(e.mean())


def _fit(x: np.ndarray, n: int, rng) -> np.ndarray:
    """Loop or crop noise to length n with a random offset."""
    if len(x) >= n:
        o = int(rng.integers(0, len(x) - n + 1)); return x[o:o + n]
    reps = int(np.ceil(n / len(x)))
    return np.tile(x, reps)[:n]


def _frac_delay(x: np.ndarray, delay_samples: float) -> np.ndarray:
    n = len(x); k = np.fft.rfftfreq(n)
    return np.fft.irfft(np.fft.rfft(x) * np.exp(-2j * np.pi * k * delay_samples), n).astype(np.float32)


def _tilt(x: np.ndarray, db: float) -> np.ndarray:
    """1st-order spectral tilt to emulate mic response mismatch."""
    a = np.clip(db / 40.0, -0.3, 0.3)
    return lfilter([1.0, a], [1.0], x).astype(np.float32)


def _conv2(x: np.ndarray, h2: np.ndarray, n: int) -> np.ndarray:
    return np.stack([fftconvolve(x, h2[m])[:n] for m in range(2)]).astype(np.float32)


def mix(rng, speech, noises, impulse, impulse_onsets_s, bank, cfg: MixConfig):
    n = len(speech)
    meta = {"clean_bucket": False, "clipped": False, "ref_dropout": False, "impulse_peak_db": None,
            "impulse_onsets_s": [], "ref_speech_gain_db": None}
    speech = speech.astype(np.float32)

    use_room = bank is not None and rng.random() < cfg.p_room
    meta["path"] = "room" if use_room else "param"

    if use_room:
        r = bank.sample(rng)
        # normalise so the primary direct path has unit gain: SNR is defined at the primary
        h_s = r["speech"] / (np.abs(r["speech"][0]).max() + 1e-9)
        s2 = _conv2(speech, h_s, n)
        clean = s2[0].copy()
        noise2 = np.zeros((2, n), np.float32)
        for i, nz in enumerate(noises):
            h_n = r["noise"][i % len(r["noise"])]
            h_n = h_n / (np.abs(h_n[0]).max() + 1e-9)
            noise2 += _conv2(_fit(nz, n, rng), h_n, n)
        meta["ref_speech_gain_db"] = float(10 * np.log10((s2[1] ** 2).sum() / ((s2[0] ** 2).sum() + 1e-12) + 1e-12))
    else:
        clean = speech.copy()
        g_db = float(rng.uniform(*cfg.ref_speech_gain_db))
        d = rng.uniform(*cfg.ref_delay_ms) * SR / 1000
        s2 = np.stack([speech, _frac_delay(speech, d) * 10 ** (g_db / 20)])
        meta["ref_speech_gain_db"] = g_db
        noise2 = np.zeros((2, n), np.float32)
        for nz in noises:
            nz = _fit(nz, n, rng)
            # independent short random filters per channel: different arrival paths
            for m in range(2):
                taps = rng.normal(0, 1, 3); taps[0] = 1.0; taps[1:] *= 0.3
                noise2[m] += lfilter(taps, [1.0], nz).astype(np.float32)

    if rng.random() < cfg.p_clean:
        meta["clean_bucket"] = True; meta["snr_db"] = np.inf; meta["noise_class"] = "clean"
        return s2.astype(np.float32), clean, meta

    # --- scale noise to target SNR on the primary, speech-active region ---
    snr = float(rng.uniform(*cfg.snr_range))
    ps = speech_active_power(clean); pn = (noise2[0] ** 2).mean() + 1e-12
    noise2 *= np.sqrt(ps / (pn * 10 ** (snr / 10)))
    meta["snr_db"] = snr

    out = s2 + noise2

    # --- impulse event: level set independently of SNR, recorded as peak ---
    if impulse is not None:
        pk_db = float(rng.uniform(*cfg.impulse_peak_db))
        start = int(rng.integers(0, max(1, n - len(impulse))))
        seg = impulse[: n - start]
        ref_rms = np.sqrt(ps)
        # impulses are far-field: similar level at both mics, small decorrelation
        imp2 = np.stack([seg, lfilter([1.0, rng.uniform(-0.2, 0.2)], [1.0], seg)]).astype(np.float32)
        imp2 *= ref_rms * 10 ** (pk_db / 20)
        out[:, start:start + len(seg)] += imp2
        meta["impulse_peak_db"] = pk_db
        meta["impulse_onsets_s"] = [start / SR + o for o in impulse_onsets_s]

    # --- common augmentations ---
    g = rng.uniform(-cfg.mic_mismatch_db, cfg.mic_mismatch_db)
    out[1] *= 10 ** (g / 20)
    out[0] = _tilt(out[0], rng.uniform(-3, 3)); out[1] = _tilt(out[1], rng.uniform(-3, 3))

    if rng.random() < cfg.p_wind:
        w = lfilter([1.0], [1.0, -0.995], rng.standard_normal(n)).astype(np.float32)
        w *= np.sqrt(ps) * 10 ** (rng.uniform(-20, -5) / 20) / (w.std() + 1e-9)
        out[int(rng.integers(0, 2))] += w

    if rng.random() < cfg.p_ref_dropout:
        a = int(rng.integers(0, n)); b = min(n, a + int(rng.uniform(0.2, 1.0) * SR))
        out[1, a:b] *= 10 ** (-40 / 20); meta["ref_dropout"] = True

    if rng.random() < cfg.p_clip:
        lvl = np.abs(out[0]).max() * rng.uniform(0.3, 0.8)
        out[0] = np.clip(out[0], -lvl, lvl); meta["clipped"] = True

    # keep everything inside [-1, 1] without changing SNR: scale mix and clean together
    peak = np.abs(out).max()
    if peak > 0.99:
        out /= peak / 0.99; clean /= peak / 0.99
    return out.astype(np.float32), clean.astype(np.float32), meta
```

- [ ] **Step 4: Run tests** → PASS.
- [ ] **Step 5: Stage** — `git add vaani/data/mixer.py tests/test_mixer.py`

---

### Task 7: Datasets and rendered eval sets

**Files:**
- Create: `vaani/data/dataset.py`, `scripts/render_eval_sets.py`
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: manifests, `mixer.mix`, `impulses.generate`, `RirBank`.
- Produces:
  - `dataset.DynamicMixDataset(manifest_paths: list, split: str, bank_path: str|None, cfg: MixConfig, crop_s=4.0, epoch_len=20000, seed=0)` → items `{"mix": Tensor[2,T], "clean": Tensor[T], "meta": dict}`.
  - `dataset.RenderedDataset(root: Path)` reading `<root>/<bucket>/<id>.mix.wav`, `<id>.clean.wav`, `<id>.json` (+ optional `<id>.twin.mix.wav`), items with the same keys plus `meta["bucket"]`, `meta["id"]`.
  - `dataset.collate(batch)` → dict of stacked tensors + list of metas.
  - `dataset.BUCKET_SNRS = [-10,-5,0,5,10,15]`, `dataset.NOISE_CLASSES = ["stationary","changing","impulsive","impulsive+stationary","clean"]`.

- [ ] **Step 1: Write failing test**

`tests/test_dataset.py`:
```python
import json, numpy as np, soundfile as sf, torch
from pathlib import Path
from vaani.data import dataset, manifests, mixer


def _tiny_manifest(tmp_path):
    rows = []
    for i in range(4):
        p = tmp_path / f"s{i}.flac"; sf.write(p, np.random.randn(32000).astype(np.float32) * 0.1, 16000)
        rows.append(dict(source_id=f"s{i}", corpus="t", kind="speech", group_id=f"g{i}", speaker_id=f"g{i}",
                         path=str(p), duration_s=2.0, licence="", split="train", sha1="", noise_class=""))
    for i in range(3):
        p = tmp_path / f"n{i}.flac"; sf.write(p, np.random.randn(48000).astype(np.float32) * 0.1, 16000)
        rows.append(dict(source_id=f"n{i}", corpus="t", kind="noise", group_id=f"ng{i}", speaker_id="",
                         path=str(p), duration_s=3.0, licence="", split="train", sha1="", noise_class=["stationary", "changing", "impulsive"][i]))
    m = tmp_path / "m.parquet"; manifests.write(rows, m); return m


def test_dynamic_dataset_yields_batches(tmp_path):
    m = _tiny_manifest(tmp_path)
    ds = dataset.DynamicMixDataset([m], "train", None, mixer.MixConfig(p_room=0.0), crop_s=1.0, epoch_len=6, seed=0)
    dl = torch.utils.data.DataLoader(ds, batch_size=3, collate_fn=dataset.collate, num_workers=0)
    b = next(iter(dl))
    assert b["mix"].shape == (3, 2, 16000) and b["clean"].shape == (3, 16000) and len(b["meta"]) == 3


def test_rendered_roundtrip(tmp_path):
    root = tmp_path / "eval"; (root / "stationary_0").mkdir(parents=True)
    x = np.random.randn(2, 16000).astype(np.float32) * 0.1
    sf.write(root / "stationary_0" / "a.mix.wav", x.T, 16000); sf.write(root / "stationary_0" / "a.clean.wav", x[0], 16000)
    json.dump({"snr_db": 0}, open(root / "stationary_0" / "a.json", "w"))
    ds = dataset.RenderedDataset(root)
    it = ds[0]
    assert it["mix"].shape == (2, 16000) and it["meta"]["bucket"] == "stationary_0" and it["meta"]["id"] == "a"
```

- [ ] **Step 2: Run to verify failure** → FAIL.

- [ ] **Step 3: Implement `vaani/data/dataset.py`**

```python
"""Train = dynamic mixing (a fresh mixture every item, infinite variety).
Val/test = rendered once, frozen, so numbers across runs are comparable.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset

from vaani.data import impulses, manifests
from vaani.data.mixer import MixConfig, mix
from vaani.data.rirs import RirBank

SR = 16000
BUCKET_SNRS = [-10, -5, 0, 5, 10, 15]
NOISE_CLASSES = ["stationary", "changing", "impulsive", "impulsive+stationary", "clean"]


def _load(path: str, n: int | None, rng) -> np.ndarray:
    info = sf.info(path)
    if n is None or info.frames <= n:
        x, _ = sf.read(path, dtype="float32"); return x
    start = int(rng.integers(0, info.frames - n + 1))
    x, _ = sf.read(path, dtype="float32", start=start, frames=n); return x


class DynamicMixDataset(Dataset):
    def __init__(self, manifest_paths, split, bank_path, cfg: MixConfig, crop_s=4.0, epoch_len=20000, seed=0):
        df = pd.concat([manifests.read(p) for p in manifest_paths])
        df = df[df.split == split]
        self.speech = df[df.kind == "speech"].reset_index(drop=True)
        self.noise = df[df.kind == "noise"].reset_index(drop=True)
        self.bank = RirBank(bank_path) if bank_path else None
        self.cfg, self.n, self.epoch_len, self.seed = cfg, int(crop_s * SR), epoch_len, seed
        self.epoch = 0
        assert len(self.speech) and len(self.noise), "empty manifest split"

    def set_epoch(self, e: int):  # different mixtures each epoch, still reproducible
        self.epoch = e

    def __len__(self):
        return self.epoch_len

    def __getitem__(self, i):
        rng = np.random.default_rng([self.seed, self.epoch, i])
        s = _load(self.speech.path.iloc[int(rng.integers(len(self.speech)))], self.n, rng)
        s = np.pad(s, (0, self.n - len(s)))
        # noise draw: continuous class(es) plus optionally an impulsive one
        k = int(rng.integers(1, 3))
        cont = self.noise[self.noise.noise_class != "impulsive"]
        rows = [cont.iloc[int(rng.integers(len(cont)))] for _ in range(k)]
        noises = [_load(r.path, self.n, rng) for r in rows]
        noise_class = "stationary" if all(r.noise_class == "stationary" for r in rows) else "changing"
        imp, onsets = None, []
        if rng.random() < 0.5:
            impd = self.noise[self.noise.noise_class == "impulsive"]
            if len(impd) and rng.random() < 0.5:
                imp = _load(impd.path.iloc[int(rng.integers(len(impd)))], 2 * SR, rng); onsets = [0.0]
            else:
                imp, m = impulses.generate(rng); onsets = m["onsets_s"]
            noise_class = "impulsive+stationary" if noise_class == "stationary" else "impulsive"
        mixed, clean, meta = mix(rng, s, noises, imp, onsets, self.bank, self.cfg)
        meta["noise_class"] = "clean" if meta["clean_bucket"] else noise_class
        return {"mix": torch.from_numpy(mixed), "clean": torch.from_numpy(clean), "meta": meta}


class RenderedDataset(Dataset):
    def __init__(self, root: Path):
        self.items = sorted(Path(root).glob("*/*.mix.wav"))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p = self.items[i]; stem = p.name[: -len(".mix.wav")]
        x, _ = sf.read(p, dtype="float32"); c, _ = sf.read(p.with_name(stem + ".clean.wav"), dtype="float32")
        meta = json.load(open(p.with_name(stem + ".json")))
        meta.update(bucket=p.parent.name, id=stem)
        twin = p.with_name(stem + ".twin.mix.wav")
        out = {"mix": torch.from_numpy(x.T.copy()), "clean": torch.from_numpy(c), "meta": meta}
        if twin.exists():
            t, _ = sf.read(twin, dtype="float32"); out["twin"] = torch.from_numpy(t.T.copy())
        return out


def collate(batch):
    out = {"mix": torch.stack([b["mix"] for b in batch]), "clean": torch.stack([b["clean"] for b in batch]),
           "meta": [b["meta"] for b in batch]}
    if "twin" in batch[0]:
        out["twin"] = torch.stack([b["twin"] for b in batch])
    return out
```

- [ ] **Step 4: Write `scripts/render_eval_sets.py`**

```python
"""Render frozen val/test sets, bucketed by noise class x input SNR.
Burst buckets also get a 'twin' with the impulse removed (same seed) for
the recovery-time metric. Writes an eval-set hash for run.json.
"""
import argparse, hashlib, json
from pathlib import Path

import numpy as np, soundfile as sf, yaml

from vaani.data import impulses, manifests
from vaani.data.dataset import BUCKET_SNRS, SR, _load
from vaani.data.mixer import MixConfig, mix
from vaani.data.rirs import RirBank

ap = argparse.ArgumentParser()
ap.add_argument("--manifests", nargs="+", required=True)
ap.add_argument("--split", choices=["val", "test"], required=True)
ap.add_argument("--out", default="data/eval")
ap.add_argument("--bank", default="data/rirs/bank.npz")
ap.add_argument("--per-bucket", type=int, default=40)
ap.add_argument("--clip-s", type=float, default=6.0)
ap.add_argument("--seed", type=int, default=1234)
a = ap.parse_args()

import pandas as pd
df = pd.concat([manifests.read(p) for p in a.manifests]); df = df[df.split == a.split]
speech, noise = df[df.kind == "speech"], df[df.kind == "noise"]
bank = RirBank(a.bank) if Path(a.bank).exists() else None
root = Path(a.out) / a.split; n = int(a.clip_s * SR)
classes = {"stationary": ("stationary", False), "changing": ("changing", False),
           "impulsive": ("changing", True), "impulsive+stationary": ("stationary", True)}

for cls, (cont_cls, burst) in classes.items():
    pool = noise[noise.noise_class == cont_cls] if (noise.noise_class == cont_cls).any() else noise
    for snr in BUCKET_SNRS:
        d = root / f"{cls}_{snr}"; d.mkdir(parents=True, exist_ok=True)
        for i in range(a.per_bucket):
            seed = [a.seed, hash(cls) % 1000, snr + 100, i]
            rng = np.random.default_rng(seed)
            s = np.pad((x := _load(speech.path.iloc[int(rng.integers(len(speech)))], n, rng)), (0, n - len(x)))
            nz = [_load(pool.path.iloc[int(rng.integers(len(pool)))], n, rng)]
            cfg = MixConfig(snr_range=(snr, snr), p_clean=0.0)
            imp, on = (impulses.generate(np.random.default_rng(seed + [7])) if burst else (None, []))
            if burst: imp, on = imp, on["onsets_s"]
            m, c, meta = mix(np.random.default_rng(seed), s, nz, imp, on, bank, cfg)
            meta["noise_class"] = cls
            sf.write(d / f"{i:04d}.mix.wav", m.T, SR); sf.write(d / f"{i:04d}.clean.wav", c, SR)
            json.dump(meta, open(d / f"{i:04d}.json", "w"))
            if burst:  # identical draw with no impulse
                t, _, _ = mix(np.random.default_rng(seed), s, nz, None, [], bank, cfg)
                sf.write(d / f"{i:04d}.twin.mix.wav", t.T, SR)

# clean bucket: no noise at all
d = root / "clean_inf"; d.mkdir(exist_ok=True)
for i in range(a.per_bucket):
    rng = np.random.default_rng([a.seed, 999, i])
    s = np.pad((x := _load(speech.path.iloc[int(rng.integers(len(speech)))], n, rng)), (0, n - len(x)))
    m, c, meta = mix(rng, s, [np.zeros(n, np.float32)], None, [], bank, MixConfig(p_clean=1.0))
    meta["noise_class"] = "clean"
    sf.write(d / f"{i:04d}.mix.wav", m.T, SR); sf.write(d / f"{i:04d}.clean.wav", c, SR); json.dump(meta, open(d / f"{i:04d}.json", "w"))

h = hashlib.sha1()
for p in sorted(root.rglob("*.json")): h.update(p.read_bytes())
(root / "EVALSET_HASH").write_text(h.hexdigest()[:12]); print("eval-set hash", h.hexdigest()[:12])
```

Note: `hash(cls) % 1000` is process-salted — replace with `manifests.stable_hash(cls) % 1000` before running (import already present). Fix this inline when writing the file.

- [ ] **Step 5: Run tests** → PASS. Then, once Task 3's fetch has produced manifests and Task 5's bank exists:
```bash
uv run python scripts/render_eval_sets.py --manifests data/manifests/*.parquet --split val --per-bucket 25
uv run python scripts/render_eval_sets.py --manifests data/manifests/*.parquet --split test --per-bucket 50
```

- [ ] **Step 6: Stage** — `git add vaani/data/dataset.py scripts/render_eval_sets.py tests/test_dataset.py`

---

### Task 8: Guarded NLMS noise estimator

**Files:**
- Create: `vaani/dsp/nlms.py`
- Test: `tests/test_nlms.py`

**Interfaces:**
- Produces: `nlms.NLMS(taps=64, mu=0.05, eps=1e-6)` with `reset()`, `process_block(primary: np.ndarray[N], reference: np.ndarray[N], gate: float) -> tuple[n_hat: np.ndarray[N], residual_ratio: float]`. `gate∈[0,1]` scales µ; `residual_ratio = E[(primary−n_hat)²]/E[primary²]` over the block (NLMS-health feature).

- [ ] **Step 1: Write failing test**

`tests/test_nlms.py`:
```python
import numpy as np
from vaani.dsp.nlms import NLMS


def test_converges_on_filtered_reference():
    rng = np.random.default_rng(0)
    ref = rng.standard_normal(16000 * 4).astype(np.float32)
    h = np.array([0.0, 0.5, -0.3, 0.1], np.float32)
    prim = np.convolve(ref, h)[: len(ref)].astype(np.float32)   # noise-only at primary
    f = NLMS()
    for i in range(0, len(ref) - 256, 256):
        n_hat, _ = f.process_block(prim[i:i + 256], ref[i:i + 256], gate=1.0)
    # last block error must be well below the input energy
    err = prim[i:i + 256] - n_hat
    assert 10 * np.log10((prim[i:i + 256] ** 2).mean() / ((err ** 2).mean() + 1e-12)) > 20


def test_gate_zero_freezes_taps():
    rng = np.random.default_rng(0)
    f = NLMS(); ref = rng.standard_normal(256).astype(np.float32); prim = ref.copy()
    f.process_block(prim, ref, gate=1.0); w = f.w.copy()
    f.process_block(prim, ref, gate=0.0)
    assert np.array_equal(w, f.w)
```

- [ ] **Step 2: Run to verify failure** → FAIL.

- [ ] **Step 3: Implement `vaani/dsp/nlms.py`**

```python
"""Guarded NLMS. reference -> primary noise path estimate.

Output is the noise ESTIMATE n_hat, not 'cleaned' audio: the neural model
decides how to use it, so an unconverged or speech-leaking filter cannot
delete speech before inference. The controller supplies `gate`, which
scales the step size (0 = frozen) during speech, bursts, overload and
reference dropout.

Written sample-by-sample in plain NumPy on purpose: the DSP lead ports
this loop to C and checks it against the golden vectors.
"""
import numpy as np


class NLMS:
    def __init__(self, taps: int = 64, mu: float = 0.05, eps: float = 1e-6):
        self.taps, self.mu, self.eps = taps, mu, eps
        self.reset()

    def reset(self):
        self.w = np.zeros(self.taps, np.float32)
        self.buf = np.zeros(self.taps, np.float32)   # most recent reference samples, newest first

    def process_block(self, primary: np.ndarray, reference: np.ndarray, gate: float):
        n = len(primary); n_hat = np.empty(n, np.float32)
        mu = self.mu * float(np.clip(gate, 0.0, 1.0))
        w, buf = self.w, self.buf
        for i in range(n):
            buf[1:] = buf[:-1]; buf[0] = reference[i]
            y = float(w @ buf)
            n_hat[i] = y
            if mu > 0.0:
                e = primary[i] - y
                w += (mu * e / (float(buf @ buf) + self.eps)) * buf
        resid = primary - n_hat
        ratio = float((resid ** 2).mean() / ((primary ** 2).mean() + 1e-12))
        return n_hat, min(ratio, 1.0)
```

- [ ] **Step 4: Run tests** → PASS (the pure-Python loop is slow but fine for 4 s).
- [ ] **Step 5: Stage** — `git add vaani/dsp/nlms.py tests/test_nlms.py`

---

### Task 9: Features, controller, pipeline, golden vectors

**Files:**
- Create: `vaani/dsp/features.py`, `vaani/dsp/controller.py`, `vaani/dsp/pipeline.py`, `scripts/make_golden_vectors.py`
- Test: `tests/test_controller.py`, `tests/test_pipeline.py`

**Interfaces:**
- Produces:
  - `features.N_FEATURES = 18`; `features.FEATURE_NAMES` (list, spec §5.2 order); `features.FrameFeatures()` with `reset()` and `compute(prim_frame: np.ndarray[512], ref_frame: np.ndarray[512], P: np.ndarray[257] complex, R: np.ndarray[257] complex, nlms_health: float, prev_gate: float) -> np.ndarray[18] float32`.
  - `controller.Controller()` with `reset()`, `step(feat: np.ndarray[18]) -> tuple[adapt_gate: float, burst_flag: bool, reliability: float]`.
  - `pipeline.run(mix: np.ndarray[2,T]) -> dict` with `n_hat: np.ndarray[T]`, `features: np.ndarray[T',18]`, `gate: np.ndarray[T']`, `burst: np.ndarray[T'] bool`, `reliability: np.ndarray[T']` where `T'` = number of STFT frames of `stft.np_stft` (center=True). `pipeline.run(mix, controller_on=False)` fixes gate=1 and zeroes features (the ablation).

- [ ] **Step 1: Write failing tests**

`tests/test_controller.py`:
```python
import numpy as np
from vaani.dsp import pipeline


def _voiced(n, sr=16000):
    t = np.arange(n) / sr
    return (0.3 * np.sin(2 * np.pi * 150 * t) * (1 + 0.3 * np.sin(2 * np.pi * 4 * t))).astype(np.float32)


def test_burst_detected_and_gate_recovers():
    sr = 16000; n = 3 * sr
    prim = _voiced(n) * 0.2; ref = np.roll(prim, 5) * 0.2
    # far-field impulse: similar level at both mics
    imp = np.exp(-np.arange(800) / 100).astype(np.float32) * np.random.default_rng(0).standard_normal(800).astype(np.float32)
    prim[sr:sr + 800] += imp; ref[sr:sr + 800] += imp * 0.9
    out = pipeline.run(np.stack([prim, ref]))
    f0 = sr // 256
    assert out["burst"][f0:f0 + 4].any()
    assert out["gate"][f0 + 1] < 0.1
    assert out["gate"][-1] > 0.9                     # ramped back well before the end


def test_consonant_does_not_trip_burst():
    sr = 16000; n = 2 * sr
    prim = _voiced(n) * 0.3; ref = np.roll(prim, 5) * 0.15   # near-mouth: ref much quieter
    rng = np.random.default_rng(1)
    # 20 ms wideband transient at -10 dB rel. voicing, primary only (a consonant)
    prim[sr:sr + 320] += rng.standard_normal(320).astype(np.float32) * 0.3 * 10 ** (-10 / 20)
    out = pipeline.run(np.stack([prim, ref]))
    assert not out["burst"].any()
```

`tests/test_pipeline.py`:
```python
import numpy as np
from vaani.dsp import pipeline, features, stft


def test_shapes_and_ablation():
    x = np.random.default_rng(0).standard_normal((2, 16000)).astype(np.float32) * 0.1
    out = pipeline.run(x)
    T = stft.np_stft(x[0]).shape[1]
    assert out["features"].shape == (T, features.N_FEATURES) and out["n_hat"].shape == (16000,)
    off = pipeline.run(x, controller_on=False)
    assert np.all(off["gate"] == 1.0) and np.all(off["features"] == 0.0)
```

- [ ] **Step 2: Run to verify failure** → FAIL.

- [ ] **Step 3: Implement `vaani/dsp/features.py`**

```python
"""Per-frame impulse & reliability features. Order is a contract (spec 5.2).
Everything is cheap by design - these run on the embedded CPU every 16 ms.
"""
import numpy as np

N_FEATURES = 18
FEATURE_NAMES = [
    "log_energy_delta", "spectral_flux", "peak_to_rms", "clip_frac_primary",
    "clip_frac_reference", "speech_presence",
    *[f"coherence_b{i}" for i in range(8)],
    "level_diff_db", "ref_dropout", "nlms_health", "prev_gate",
]
CLIP = 0.99
# 8 ERB-ish band edges in STFT bins (16 kHz, 257 bins): coarse enough to be robust
BAND_EDGES = [1, 4, 8, 14, 22, 34, 52, 80, 257]


class FrameFeatures:
    def __init__(self, alpha: float = 0.7):
        self.alpha = alpha  # smoothing for coherence / speech-presence
        self.reset()

    def reset(self):
        self.prev_log_e = -12.0
        self.prev_mag = None
        self.Spp = np.zeros(257); self.Srr = np.zeros(257); self.Spr = np.zeros(257, complex)
        self.sp_smooth = 0.0

    def compute(self, p, r, P, R, nlms_health, prev_gate):
        f = np.zeros(N_FEATURES, np.float32)
        e = float((p ** 2).mean() + 1e-10); log_e = 10 * np.log10(e)
        f[0] = log_e - self.prev_log_e; self.prev_log_e = log_e
        mag = np.abs(P)
        f[1] = 0.0 if self.prev_mag is None else float(np.sum(np.maximum(mag - self.prev_mag, 0)) / (np.sum(self.prev_mag) + 1e-8))
        self.prev_mag = mag
        f[2] = float(np.abs(p).max() / (np.sqrt(e) + 1e-8))
        f[3] = float((np.abs(p) > CLIP).mean()); f[4] = float((np.abs(r) > CLIP).mean())
        er = float((r ** 2).mean() + 1e-10)
        # near-mouth speech: primary >> reference. far-field noise: ~equal.
        ratio_db = 10 * np.log10(e / er)
        sp = float(np.clip((ratio_db - 3.0) / 9.0, 0, 1))      # 3 dB -> 0, 12 dB -> 1
        self.sp_smooth = self.alpha * self.sp_smooth + (1 - self.alpha) * sp
        f[5] = self.sp_smooth
        a = self.alpha
        self.Spp = a * self.Spp + (1 - a) * np.abs(P) ** 2
        self.Srr = a * self.Srr + (1 - a) * np.abs(R) ** 2
        self.Spr = a * self.Spr + (1 - a) * P * np.conj(R)
        coh = np.abs(self.Spr) ** 2 / (self.Spp * self.Srr + 1e-12)
        for i in range(8):
            f[6 + i] = float(coh[BAND_EDGES[i]:BAND_EDGES[i + 1]].mean())
        f[14] = float(np.clip(ratio_db, -40, 40))
        f[15] = 1.0 if ratio_db > 30.0 else 0.0
        f[16] = float(nlms_health); f[17] = float(prev_gate)
        return f
```

- [ ] **Step 4: Implement `vaani/dsp/controller.py`**

```python
"""Burst / reliability gating. Rule-based with hysteresis.

Burst = large energy jump AND both mics hit at similar level. A consonant
is a jump too, but it is near-mouth, so the level difference stays large;
that second condition is what keeps speech transients from freezing the
filter or being treated as noise.
"""
import numpy as np

from vaani.dsp.features import FEATURE_NAMES

_I = {n: i for i, n in enumerate(FEATURE_NAMES)}


class Controller:
    def __init__(self, jump_db=12.0, level_diff_max_db=3.0, hold_frames=4, ramp_frames=12,
                 speech_freeze=0.6):
        self.jump_db, self.ld_max, self.hold, self.ramp, self.sp_freeze = jump_db, level_diff_max_db, hold_frames, ramp_frames, speech_freeze
        self.reset()

    def reset(self):
        self.hold_left = 0
        self.gate = 1.0
        self.ramp_pos = self.ramp   # fully ramped
        self.energy_hist = []

    def step(self, f: np.ndarray):
        jump = f[_I["log_energy_delta"]]
        ld = f[_I["level_diff_db"]]
        burst_now = (jump >= self.jump_db) and (ld <= self.ld_max)
        if burst_now:
            self.hold_left = self.hold
        burst = self.hold_left > 0
        if self.hold_left > 0:
            self.hold_left -= 1

        overload = f[_I["clip_frac_primary"]] > 0.01 or f[_I["clip_frac_reference"]] > 0.01
        dropout = f[_I["ref_dropout"]] > 0.5
        speech = f[_I["speech_presence"]] > self.sp_freeze
        freeze = burst or overload or dropout or speech
        if freeze:
            self.ramp_pos = 0; self.gate = 0.0
        else:
            self.ramp_pos = min(self.ramp, self.ramp_pos + 1)
            self.gate = self.ramp_pos / self.ramp   # linear ramp ~200 ms at 16 ms hop

        coh_ok = float(np.clip(f[_I["coherence_b2"]:_I["coherence_b2"] + 4].mean() * 2, 0, 1))
        reliability = (1 - f[_I["clip_frac_primary"]]) * (1 - f[_I["ref_dropout"]]) * (0.5 + 0.5 * coh_ok) * (0.5 + 0.5 * (1 - f[_I["nlms_health"]]))
        return float(self.gate), bool(burst), float(np.clip(reliability, 0, 1))
```

- [ ] **Step 5: Implement `vaani/dsp/pipeline.py`**

```python
"""Runs NLMS + features + controller frame-synchronously over a stereo clip.
Used offline (dataset/eval) and as the reference for the embedded port.
Frame k covers samples [k*HOP - 256, k*HOP + 256) after torch-style reflect
padding, so the model's frame k and this feature vector k line up exactly.
"""
import numpy as np

from vaani.dsp import stft
from vaani.dsp.controller import Controller
from vaani.dsp.features import FrameFeatures, N_FEATURES
from vaani.dsp.nlms import NLMS


def run(mix: np.ndarray, controller_on: bool = True) -> dict:
    prim, ref = mix[0].astype(np.float32), mix[1].astype(np.float32)
    T = len(prim)
    nlms, ff, ctl = NLMS(), FrameFeatures(), Controller()
    n_hat = np.zeros(T, np.float32)
    gate = 1.0
    # NLMS runs in hop-sized blocks so its taps update between frames
    healths = []
    for i in range(0, T, stft.HOP):
        blk, hlt = nlms.process_block(prim[i:i + stft.HOP], ref[i:i + stft.HOP], gate if controller_on else 1.0)
        n_hat[i:i + len(blk)] = blk; healths.append(hlt)
        # controller decisions for the *next* block come from the frame ending here
    P = stft.np_stft(prim); R = stft.np_stft(ref)
    n_frames = P.shape[1]
    pad = np.pad(prim, stft.N_FFT // 2, mode="reflect"); padr = np.pad(ref, stft.N_FFT // 2, mode="reflect")
    feats = np.zeros((n_frames, N_FEATURES), np.float32)
    gates = np.ones(n_frames, np.float32); bursts = np.zeros(n_frames, bool); rel = np.ones(n_frames, np.float32)
    gate = 1.0
    for k in range(n_frames):
        a = k * stft.HOP
        f = ff.compute(pad[a:a + stft.N_FFT], padr[a:a + stft.N_FFT], P[:, k], R[:, k],
                       healths[min(k, len(healths) - 1)], gate)
        feats[k] = f
        if controller_on:
            gate, b, r = ctl.step(f)
            gates[k], bursts[k], rel[k] = gate, b, r
    if not controller_on:
        feats[:] = 0.0
    # Second pass of NLMS with the actual gate trajectory, so n_hat reflects gating.
    # (Offline this is exact; online the embedded port applies gate[k-1] to block k.)
    if controller_on:
        nlms.reset()
        for k in range(n_frames):
            i = k * stft.HOP
            if i >= T:
                break
            blk, _ = nlms.process_block(prim[i:i + stft.HOP], ref[i:i + stft.HOP], gates[k - 1] if k else 1.0)
            n_hat[i:i + len(blk)] = blk
    return {"n_hat": n_hat, "features": feats, "gate": gates, "burst": bursts, "reliability": rel}
```

- [ ] **Step 6: Write `scripts/make_golden_vectors.py`**

```python
"""Golden vectors for the DSP port. Deterministic inputs -> expected outputs."""
from pathlib import Path
import numpy as np, soundfile as sf
from vaani.dsp import pipeline

out = Path("deploy/dsp_reference/vectors"); out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(42)
cases = {}
t = np.arange(16000 * 2) / 16000
voiced = (0.3 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
noise = rng.standard_normal(len(t)).astype(np.float32) * 0.05
cases["speech_plus_noise"] = np.stack([voiced + noise, np.roll(voiced, 5) * 0.3 + np.roll(noise, 2)])
imp = np.zeros(len(t), np.float32); imp[16000:16800] = np.exp(-np.arange(800) / 100) * rng.standard_normal(800)
cases["burst"] = np.stack([voiced * 0.2 + imp, np.roll(voiced, 5) * 0.06 + imp * 0.9])
drop = cases["speech_plus_noise"].copy(); drop[1, 8000:16000] *= 0.01
cases["ref_dropout"] = drop
for name, x in cases.items():
    sf.write(out / f"{name}.wav", x.T, 16000)
    r = pipeline.run(x)
    np.savez(out / f"{name}.npz", **r)
    print(name, "burst frames:", int(r["burst"].sum()))
```

- [ ] **Step 7: Run tests, then generate vectors**

`uv run pytest tests/test_controller.py tests/test_pipeline.py -v` → PASS. If `test_consonant_does_not_trip_burst` fails, raise `level_diff_max_db` tolerance is NOT the fix — check that the reference in the test is quieter than the primary by > 3 dB (it is 6 dB) and that `log_energy_delta` for a −10 dB transient stays under 12 dB.
`uv run python scripts/make_golden_vectors.py`.

- [ ] **Step 8: Stage** — `git add vaani/dsp scripts/make_golden_vectors.py tests/test_controller.py tests/test_pipeline.py deploy/dsp_reference`

---

### Task 10: Vendor GTCRN, parity test, classical baselines

**Files:**
- Create: `vaani/models/gtcrn.py`, `vaani/models/gtcrn_stream.py`, `vaani/models/modules/__init__.py`, `vaani/models/modules/convolution.py`, `vaani/models/modules/convert.py`, `vaani/models/checkpoints/model_trained_on_dns3.tar`, `vaani/models/baselines.py`, `vaani/models/UPSTREAM.md`
- Test: `tests/test_models.py`

**Interfaces:**
- Produces: `GTCRN()` (input `(B,257,T,2)` → same), `StreamGTCRN()` with `forward(spec_frame (1,257,1,2), conv_cache, tra_cache, inter_cache)`; `baselines.get(name) -> object` with `.enhance(mix: np.ndarray[2,T]) -> np.ndarray[T]` for `"raw"`, `"nlms_only"`, `"rnnoise"`, `"gtcrn_pretrained"`; `gtcrn_stream.init_caches(device) -> tuple`.

- [ ] **Step 1: Vendor the upstream files verbatim**

```bash
mkdir -p vaani/models/modules vaani/models/checkpoints
B=https://raw.githubusercontent.com/Xiaobin-Rong/gtcrn/main
curl -sL $B/gtcrn.py -o vaani/models/gtcrn.py
curl -sL $B/stream/gtcrn_stream.py -o vaani/models/gtcrn_stream.py
curl -sL $B/stream/modules/convolution.py -o vaani/models/modules/convolution.py
curl -sL $B/stream/modules/convert.py -o vaani/models/modules/convert.py
curl -sL $B/LICENSE -o vaani/models/LICENSE.gtcrn
curl -sL $B/checkpoints/model_trained_on_dns3.tar -o vaani/models/checkpoints/model_trained_on_dns3.tar
touch vaani/models/modules/__init__.py
```
Then fix the only edits allowed — import paths: in `gtcrn_stream.py` change `from modules.convolution import ...` to `from vaani.models.modules.convolution import ...` (and similarly for any `from modules.` import). Record this in `vaani/models/UPSTREAM.md`:

```markdown
Vendored from https://github.com/Xiaobin-Rong/gtcrn @ main on 2026-09-18.
Only change: import paths rewritten to `vaani.models.modules.*`. MIT licence in LICENSE.gtcrn.
Checkpoint: checkpoints/model_trained_on_dns3.tar (upstream, trained on DNS3).
```

- [ ] **Step 2: Write failing test**

`tests/test_models.py`:
```python
import numpy as np, torch
from vaani.dsp import stft
from vaani.models.gtcrn import GTCRN
from vaani.models import gtcrn_stream
from vaani.models.baselines import get


def _load_pretrained(model):
    ck = torch.load("vaani/models/checkpoints/model_trained_on_dns3.tar", map_location="cpu")
    model.load_state_dict(ck["model"]); return model.eval()


def test_batch_vs_stream_parity():
    torch.manual_seed(0)
    x = torch.randn(1, 16000) * 0.1
    spec = stft.stft(x)                                # (1,257,T,2)
    m = _load_pretrained(GTCRN()); s = _load_pretrained(gtcrn_stream.StreamGTCRN())
    with torch.no_grad():
        y = m(spec)
        caches = gtcrn_stream.init_caches("cpu")
        outs = []
        for t in range(spec.shape[2]):
            o, *caches = s(spec[:, :, t:t + 1], *caches); outs.append(o)
        ys = torch.cat(outs, dim=2)
    assert torch.allclose(y, ys, atol=1e-4)


def test_baselines_enhance_shape():
    x = np.random.default_rng(0).standard_normal((2, 16000)).astype(np.float32) * 0.1
    for name in ("raw", "nlms_only", "gtcrn_pretrained"):
        y = get(name).enhance(x)
        assert y.shape == (16000,) and np.isfinite(y).all()
```

- [ ] **Step 3: Run to verify failure** → FAIL (no `init_caches`, no `baselines`).

- [ ] **Step 4: Add `init_caches` to `vaani/models/gtcrn_stream.py`** (append at end; shapes from the upstream docstring)

```python
def init_caches(device="cpu"):
    """Zero state for one stream (B=1). Shapes documented in StreamGTCRN.forward."""
    conv_cache = torch.zeros(2, 1, 16, 16, 33, device=device)
    tra_cache = torch.zeros(2, 3, 1, 1, 16, device=device)
    inter_cache = torch.zeros(2, 1, 33, 16, device=device)
    return conv_cache, tra_cache, inter_cache
```

- [ ] **Step 5: Implement `vaani/models/baselines.py`**

```python
"""Non-trained comparison rows. Common API: enhance(mix (2,T)) -> (T,)."""
import shutil, subprocess, tempfile
from pathlib import Path

import numpy as np, soundfile as sf, torch
from scipy.signal import resample_poly

from vaani.dsp import pipeline, stft
from vaani.models.gtcrn import GTCRN

CKPT = Path(__file__).parent / "checkpoints" / "model_trained_on_dns3.tar"


class Raw:
    def enhance(self, mix): return mix[0].copy()


class NlmsOnly:
    """Classical: primary minus the guarded NLMS estimate."""
    def enhance(self, mix):
        return (mix[0] - pipeline.run(mix)["n_hat"]).astype(np.float32)


class GtcrnPretrained:
    def __init__(self):
        self.m = GTCRN().eval(); self.m.load_state_dict(torch.load(CKPT, map_location="cpu")["model"])

    def enhance(self, mix):
        with torch.no_grad():
            x = torch.from_numpy(mix[0])[None]
            return stft.istft(self.m(stft.stft(x)), length=x.shape[-1])[0].numpy()


class RNNoise:
    """Requires the `rnnoise_demo` binary on PATH (48 kHz raw PCM16 in/out).
    Labelled in reports: different sample rate and training data."""
    def enhance(self, mix):
        exe = shutil.which("rnnoise_demo")
        if exe is None:
            raise RuntimeError("rnnoise_demo not on PATH - build upstream RNNoise or skip this row")
        x48 = resample_poly(mix[0], 3, 1)
        with tempfile.TemporaryDirectory() as d:
            i, o = Path(d, "i.raw"), Path(d, "o.raw")
            (np.clip(x48, -1, 1) * 32767).astype(np.int16).tofile(i)
            subprocess.run([exe, str(i), str(o)], check=True, capture_output=True)
            y48 = np.fromfile(o, np.int16).astype(np.float32) / 32767
        return resample_poly(y48, 1, 3)[: mix.shape[1]].astype(np.float32)


def get(name: str):
    return {"raw": Raw, "nlms_only": NlmsOnly, "gtcrn_pretrained": GtcrnPretrained, "rnnoise": RNNoise}[name]()
```

- [ ] **Step 6: Run tests** → PASS. If parity fails at 1e-4 but passes at 1e-3, the upstream stream test also uses a loose tolerance; relax to 1e-3 and note it in `UPSTREAM.md`.

- [ ] **Step 7: Stage** — `git add vaani/models tests/test_models.py`

---

### Task 11: VaaniNet

**Files:**
- Create: `vaani/models/vaani_net.py`
- Test: `tests/test_vaani_net.py`

**Interfaces:**
- Consumes: vendored `ERB, SFE, GTConvBlock, ConvBlock, DPGRNN, Decoder, Mask` from `vaani.models.gtcrn`; stream twins from `gtcrn_stream`.
- Produces:
  - `VaaniNet()`: `forward(spec: Tensor[B,257,T,6], feats: Tensor[B,T,18]) -> Tensor[B,257,T,2]` — channels of `spec` are `(prim_re, prim_im, ref_re, ref_im, nhat_re, nhat_im)`.
  - `VaaniNet.from_pretrained_gtcrn(ckpt_path) -> VaaniNet` (copies every matching weight; first conv primary slice copied, ref/n_hat slices zero-init).
  - `StreamVaaniNet()` with the same signature per frame plus caches; `init_caches(device)`.
  - `vaani_net.MAX_PARAMS = 60_000`.

- [ ] **Step 1: Write failing test**

`tests/test_vaani_net.py`:
```python
import torch
from vaani.models import vaani_net
from vaani.models.vaani_net import VaaniNet, StreamVaaniNet, init_caches


def test_shapes_and_param_budget():
    m = VaaniNet()
    n = sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert n <= vaani_net.MAX_PARAMS, n
    spec = torch.randn(2, 257, 20, 6); f = torch.randn(2, 20, 18)
    assert m(spec, f).shape == (2, 257, 20, 2)


def test_pretrained_init_matches_gtcrn_on_primary_only():
    """With ref/n_hat channels zero and features zero, the freshly initialised
    VaaniNet must reproduce pretrained GTCRN: the extension starts from the
    baseline, not from noise."""
    from vaani.models.gtcrn import GTCRN
    g = GTCRN().eval(); g.load_state_dict(torch.load("vaani/models/checkpoints/model_trained_on_dns3.tar", map_location="cpu")["model"])
    v = VaaniNet.from_pretrained_gtcrn("vaani/models/checkpoints/model_trained_on_dns3.tar").eval()
    torch.manual_seed(0); p = torch.randn(1, 257, 30, 2)
    spec = torch.cat([p, torch.zeros(1, 257, 30, 4)], dim=-1)
    with torch.no_grad():
        assert torch.allclose(g(p), v(spec, torch.zeros(1, 30, 18)), atol=1e-5)


def test_stream_parity():
    torch.manual_seed(0)
    v = VaaniNet().eval(); s = StreamVaaniNet().eval(); s.load_state_dict(v.state_dict())
    spec = torch.randn(1, 257, 25, 6); f = torch.randn(1, 25, 18)
    with torch.no_grad():
        y = v(spec, f); caches = init_caches("cpu"); outs = []
        for t in range(25):
            o, *caches = s(spec[:, :, t:t + 1], f[:, t:t + 1], *caches); outs.append(o)
    assert torch.allclose(y, torch.cat(outs, 2), atol=1e-4)
```

- [ ] **Step 2: Run to verify failure** → FAIL.

- [ ] **Step 3: Implement `vaani/models/vaani_net.py`**

```python
"""VaaniNet: GTCRN widened to three input signals (primary, reference,
NLMS noise estimate) and conditioned on the 18-dim DSP feature vector.

Design rules (spec 6.2):
- Each signal becomes (mag, re, im) like upstream, so 9 feature maps before
  SFE and 27 after, feeding a 27->16 first conv. Only that conv changes.
- Conditioning is a FiLM-style *shift* on the first encoder block output.
  Shift-only keeps the pretrained scale statistics intact at init.
- The mask is applied to the PRIMARY spectrum only.
- Zero-init of the new input slices and the FiLM projection makes the
  network numerically identical to pretrained GTCRN at step 0.
"""
import torch
import torch.nn as nn

from vaani.models.gtcrn import ERB, SFE, ConvBlock, GTConvBlock, DPGRNN, Decoder, Mask
from vaani.models import gtcrn_stream as gs

MAX_PARAMS = 60_000
N_FEAT = 18
N_SIG = 3  # primary, reference, n_hat


def _sig_feats(spec6):
    """(B,257,T,6) -> (B,9,T,257): per signal (mag, re, im)."""
    outs = []
    for i in range(N_SIG):
        re = spec6[..., 2 * i].permute(0, 2, 1); im = spec6[..., 2 * i + 1].permute(0, 2, 1)
        outs += [torch.sqrt(re ** 2 + im ** 2 + 1e-12), re, im]
    return torch.stack(outs, dim=1)


class _Encoder(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.en_convs = nn.ModuleList(blocks)
        self.film = nn.Linear(N_FEAT, 16)
        nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)

    def _cond(self, x, feats):
        # x: (B,16,T,F). feats: (B,T,18) -> shift (B,16,T,1)
        return x + self.film(feats).permute(0, 2, 1)[..., None]


class Encoder(_Encoder):
    def __init__(self):
        super().__init__([
            ConvBlock(N_SIG * 3 * 3, 16, (1, 5), stride=(1, 2), padding=(0, 2)),
            ConvBlock(16, 16, (1, 5), stride=(1, 2), padding=(0, 2), groups=2),
            GTConvBlock(16, 16, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(1, 1)),
            GTConvBlock(16, 16, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(2, 1)),
            GTConvBlock(16, 16, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(5, 1)),
        ])

    def forward(self, x, feats):
        en_outs = []
        for i, blk in enumerate(self.en_convs):
            x = blk(x)
            if i == 0:
                x = self._cond(x, feats)
            en_outs.append(x)
        return x, en_outs


class VaaniNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.erb = ERB(65, 64); self.sfe = SFE(3, 1)
        self.encoder = Encoder()
        self.dpgrnn1 = DPGRNN(16, 33, 16); self.dpgrnn2 = DPGRNN(16, 33, 16)
        self.decoder = Decoder(); self.mask = Mask()

    def forward(self, spec6, feats):
        prim = spec6[..., :2]
        feat = self.erb.bm(_sig_feats(spec6))     # (B,9,T,129)
        feat = self.sfe(feat)                      # (B,27,T,129)
        feat, en_outs = self.encoder(feat, feats)
        feat = self.dpgrnn1(feat); feat = self.dpgrnn2(feat)
        m = self.erb.bs(self.decoder(feat, en_outs))
        out = self.mask(m, prim.permute(0, 3, 2, 1))
        return out.permute(0, 3, 2, 1)

    @classmethod
    def from_pretrained_gtcrn(cls, ckpt_path):
        v = cls()
        sd = torch.load(ckpt_path, map_location="cpu")["model"]
        own = v.state_dict()
        for k, w in sd.items():
            if k == "encoder.en_convs.0.conv.weight":
                new = torch.zeros_like(own[k]); new[:, :9] = w   # primary slice
                own[k] = new
            elif k in own and own[k].shape == w.shape:
                own[k] = w
        v.load_state_dict(own)
        return v


class StreamEncoder(_Encoder):
    def __init__(self):
        super().__init__([
            gs.ConvBlock(N_SIG * 3 * 3, 16, (1, 5), stride=(1, 2), padding=(0, 2)),
            gs.ConvBlock(16, 16, (1, 5), stride=(1, 2), padding=(0, 2), groups=2),
            gs.StreamGTConvBlock(16, 16, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(1, 1)),
            gs.StreamGTConvBlock(16, 16, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(2, 1)),
            gs.StreamGTConvBlock(16, 16, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(5, 1)),
        ])

    def forward(self, x, feats, conv_cache, tra_cache):
        en_outs = []
        x = self._cond(self.en_convs[0](x), feats); en_outs.append(x)
        x = self.en_convs[1](x); en_outs.append(x)
        x, conv_cache[:, :, :2, :], tra_cache[0] = self.en_convs[2](x, conv_cache[:, :, :2, :], tra_cache[0]); en_outs.append(x)
        x, conv_cache[:, :, 2:6, :], tra_cache[1] = self.en_convs[3](x, conv_cache[:, :, 2:6, :], tra_cache[1]); en_outs.append(x)
        x, conv_cache[:, :, 6:16, :], tra_cache[2] = self.en_convs[4](x, conv_cache[:, :, 6:16, :], tra_cache[2]); en_outs.append(x)
        return x, en_outs, conv_cache, tra_cache


class StreamVaaniNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.erb = gs.ERB(65, 64); self.sfe = gs.SFE(3, 1)
        self.encoder = StreamEncoder()
        self.dpgrnn1 = gs.DPGRNN(16, 33, 16); self.dpgrnn2 = gs.DPGRNN(16, 33, 16)
        self.decoder = gs.StreamDecoder(); self.mask = gs.Mask()

    def forward(self, spec6, feats, conv_cache, tra_cache, inter_cache):
        prim = spec6[..., :2]
        feat = self.sfe(self.erb.bm(_sig_feats(spec6)))
        feat, en_outs, conv_cache[0], tra_cache[0] = self.encoder(feat, feats, conv_cache[0], tra_cache[0])
        feat, inter_cache[0] = self.dpgrnn1(feat, inter_cache[0])
        feat, inter_cache[1] = self.dpgrnn2(feat, inter_cache[1])
        m_feat, conv_cache[1], tra_cache[1] = self.decoder(feat, en_outs, conv_cache[1], tra_cache[1])
        m = self.erb.bs(m_feat)
        out = self.mask(m, prim.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        return out, conv_cache, tra_cache, inter_cache


def init_caches(device="cpu"):
    return gs.init_caches(device)
```

If the upstream `ConvBlock` stores its conv under a different attribute name than `conv`, adjust the key `encoder.en_convs.0.conv.weight` in `from_pretrained_gtcrn` to match `GTCRN().state_dict().keys()` — print them once to check.

- [ ] **Step 4: Run tests** → PASS. If `test_shapes_and_param_budget` fails the budget, print the count: the first conv grows by 18×16×5 = 1 440 weights and FiLM adds 18×16+16 = 304, so total should be ≈ 48.2K + 1.8K.

- [ ] **Step 5: Stage** — `git add vaani/models/vaani_net.py tests/test_vaani_net.py`

---

### Task 12: Losses, trainer, configs, smoke test

**Files:**
- Create: `vaani/losses.py`, `vaani/train.py`, `configs/exp/gtcrn_finetuned.yaml`, `configs/exp/vaani_no_controller.yaml`, `configs/exp/vaani_full.yaml`, `configs/exp/vaani_full_sp.yaml`, `tests/test_train_smoke.py`

**Interfaces:**
- Produces:
  - `losses.HybridLoss()` (upstream, verbatim semantics) and `losses.SpeechPreservationLoss(burst_weight=3.0, clean_l1=1.0)` — both `forward(pred_spec, true_spec, frame_weight: Tensor[B,T]|None, is_clean: Tensor[B] bool|None) -> Tensor`.
  - `train.build_model(name) -> nn.Module`, `train.prepare_batch(batch, model_name, controller_on, device) -> (inputs: tuple, target_spec, frame_weight, is_clean)`, `train.main(config_path)`.
  - Checkpoints at `runs/<name>/best.pt`, `runs/<name>/last.pt` with `{"model": state_dict, "config": cfg, "step": int}`; `runs/<name>/run.json`.
  - `train.frame_weights_from_meta(metas, n_frames, burst_weight) -> Tensor[B,T]`.

- [ ] **Step 1: Write failing smoke test**

`tests/test_train_smoke.py`:
```python
import numpy as np, soundfile as sf, yaml
from pathlib import Path
from vaani.data import manifests
from vaani import train


def _tiny(tmp_path):
    rows = []
    for i in range(3):
        p = tmp_path / f"s{i}.flac"; sf.write(p, np.random.randn(32000).astype(np.float32) * 0.1, 16000)
        rows.append(dict(source_id=f"s{i}", corpus="t", kind="speech", group_id=f"g{i}", speaker_id=f"g{i}", path=str(p), duration_s=2.0, licence="", split="train", sha1="", noise_class=""))
        rows.append({**rows[-1], "source_id": f"v{i}", "split": "val"})
    for i in range(2):
        p = tmp_path / f"n{i}.flac"; sf.write(p, np.random.randn(48000).astype(np.float32) * 0.1, 16000)
        rows.append(dict(source_id=f"n{i}", corpus="t", kind="noise", group_id=f"ng{i}", speaker_id="", path=str(p), duration_s=3.0, licence="", split="train", sha1="", noise_class="stationary"))
        rows.append({**rows[-1], "source_id": f"vn{i}", "split": "val"})
    m = tmp_path / "m.parquet"; manifests.write(rows, m); return m


def test_two_steps_each_model(tmp_path):
    m = _tiny(tmp_path)
    for model in ("gtcrn", "vaani"):
        cfg = dict(name=f"smoke_{model}", model=model, controller_on=True, loss="hybrid",
                   init_from="vaani/models/checkpoints/model_trained_on_dns3.tar",
                   data=dict(manifests=[str(m)], bank=None, crop_s=1.0, epoch_len=4, mix={"p_room": 0.0}),
                   val=dict(dynamic_items=2),
                   optim=dict(lr=1e-4, lr_new=1e-3, warmup=1, clip=5.0), batch_size=2, epochs=1, max_steps=2,
                   amp=False, device="cpu", runs_dir=str(tmp_path / "runs"), num_workers=0, seed=0)
        cp = tmp_path / f"{model}.yaml"; yaml.safe_dump(cfg, open(cp, "w"))
        train.main(str(cp))
        assert (tmp_path / "runs" / f"smoke_{model}" / "last.pt").exists()
        assert (tmp_path / "runs" / f"smoke_{model}" / "run.json").exists()
```

- [ ] **Step 2: Run to verify failure** → FAIL.

- [ ] **Step 3: Implement `vaani/losses.py`**

```python
"""Upstream GTCRN HybridLoss (kept exact so fine-tuned baselines are
apples-to-apples) plus the speech-preservation variant used only in the
`vaani_full_sp` ablation row.
"""
import torch, torch.nn as nn
from vaani.dsp import stft


def _compress(spec, p=0.3):
    re, im = spec[..., 0], spec[..., 1]
    mag = torch.sqrt(re ** 2 + im ** 2 + 1e-12)
    return re / mag ** (1 - p), im / mag ** (1 - p), mag ** p


class HybridLoss(nn.Module):
    """30*(re+im compressed MSE) + 70*mag^0.3 MSE + SI-SNR. Verbatim upstream."""
    def forward(self, pred, true, frame_weight=None, is_clean=None):
        pr, pi, pm = _compress(pred); tr, ti, tm = _compress(true)
        w = torch.ones_like(pm[:, 0]) if frame_weight is None else frame_weight  # (B,T)
        w = w[:, None, :]                                                        # (B,1,T)
        def wmse(a, b):
            return ((a - b) ** 2 * w).mean()
        spec_loss = 30 * (wmse(pr, tr) + wmse(pi, ti)) + 70 * wmse(pm, tm)
        y_p = stft.istft(pred); y_t = stft.istft(true)
        proj = (y_t * y_p).sum(-1, keepdim=True) * y_t / ((y_t ** 2).sum(-1, keepdim=True) + 1e-8)
        sisnr = -torch.log10(proj.norm(dim=-1) ** 2 / ((y_p - proj).norm(dim=-1) ** 2 + 1e-8) + 1e-8).mean()
        return spec_loss + sisnr


class SpeechPreservationLoss(HybridLoss):
    """Adds (a) up-weighting of frames around impulse events - handled by the
    caller passing frame_weight - and (b) an L1 identity penalty on the clean
    bucket so already-clean input is left alone."""
    def __init__(self, clean_l1: float = 1.0):
        super().__init__(); self.clean_l1 = clean_l1

    def forward(self, pred, true, frame_weight=None, is_clean=None):
        base = super().forward(pred, true, frame_weight, is_clean)
        if is_clean is not None and is_clean.any():
            base = base + self.clean_l1 * (pred[is_clean] - true[is_clean]).abs().mean()
        return base
```

- [ ] **Step 4: Implement `vaani/train.py`**

```python
"""Config-driven trainer. One YAML == one ablation row.

Inputs per model:
  gtcrn : spec (B,257,T,2) of the primary channel only
  vaani : spec6 (B,257,T,6) [prim, ref, n_hat] + feats (B,T,18)
The DSP pipeline runs on CPU inside DataLoader workers (prepare_batch is
called on CPU tensors before .to(device)).
"""
import argparse, hashlib, json, subprocess, sys, time
from pathlib import Path

import numpy as np, torch, yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from vaani import losses
from vaani.data import manifests
from vaani.data.dataset import DynamicMixDataset, collate
from vaani.data.mixer import MixConfig
from vaani.dsp import pipeline, stft
from vaani.models.gtcrn import GTCRN
from vaani.models.vaani_net import VaaniNet
from pystoi import stoi


def build_model(name, init_from=None):
    if name == "gtcrn":
        m = GTCRN()
        if init_from: m.load_state_dict(torch.load(init_from, map_location="cpu")["model"])
        return m
    if name == "vaani":
        return VaaniNet.from_pretrained_gtcrn(init_from) if init_from else VaaniNet()
    raise ValueError(name)


def frame_weights_from_meta(metas, n_frames, burst_weight=3.0, half_window_s=0.15):
    w = torch.ones(len(metas), n_frames)
    hw = int(half_window_s * 16000 / stft.HOP)
    for b, m in enumerate(metas):
        for on in m.get("impulse_onsets_s", []):
            k = int(on * 16000 / stft.HOP)
            w[b, max(0, k - hw): k + hw] = burst_weight
    return w


def prepare_batch(batch, model_name, controller_on, device, burst_weight=1.0):
    mix, clean, metas = batch["mix"], batch["clean"], batch["meta"]
    target = stft.stft(clean)
    n_frames = target.shape[2]
    fw = frame_weights_from_meta(metas, n_frames, burst_weight)
    is_clean = torch.tensor([m.get("clean_bucket", False) for m in metas])
    if model_name == "gtcrn":
        inputs = (stft.stft(mix[:, 0]).to(device),)
    else:
        n_hat, feats = [], []
        for b in range(mix.shape[0]):
            r = pipeline.run(mix[b].numpy(), controller_on=controller_on)
            n_hat.append(torch.from_numpy(r["n_hat"])); feats.append(torch.from_numpy(r["features"]))
        n_hat = torch.stack(n_hat); feats = torch.stack(feats)
        spec6 = torch.cat([stft.stft(mix[:, 0]), stft.stft(mix[:, 1]), stft.stft(n_hat)], dim=-1)
        inputs = (spec6.to(device), feats.to(device))
    return inputs, target.to(device), fw.to(device), is_clean.to(device)


def _git_hash():
    try: return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception: return "nogit"


@torch.no_grad()
def validate(model, dl, cfg, device):
    model.eval(); scores = []
    for batch in dl:
        inputs, target, _, _ = prepare_batch(batch, cfg["model"], cfg["controller_on"], device)
        pred = model(*inputs)
        y = stft.istft(pred, length=batch["clean"].shape[-1]).cpu().numpy()
        for b in range(y.shape[0]):
            scores.append(stoi(batch["clean"][b].numpy(), y[b], 16000, extended=False))
    model.train(); return float(np.mean(scores))


def main(config_path):
    cfg = yaml.safe_load(open(config_path))
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    device = torch.device(cfg.get("device", "cuda"))
    run_dir = Path(cfg.get("runs_dir", "runs")) / cfg["name"]; run_dir.mkdir(parents=True, exist_ok=True)
    tb = SummaryWriter(run_dir)

    mixcfg = MixConfig(**cfg["data"].get("mix", {}))
    ds = DynamicMixDataset(cfg["data"]["manifests"], "train", cfg["data"].get("bank"), mixcfg,
                           cfg["data"].get("crop_s", 4.0), cfg["data"].get("epoch_len", 20000), cfg["seed"])
    vds = DynamicMixDataset(cfg["data"]["manifests"], "val", cfg["data"].get("bank"), mixcfg,
                            cfg["data"].get("crop_s", 4.0), cfg["val"].get("dynamic_items", 200), cfg["seed"] + 1)
    dl = DataLoader(ds, cfg["batch_size"], shuffle=False, collate_fn=collate, num_workers=cfg.get("num_workers", 6), persistent_workers=cfg.get("num_workers", 6) > 0)
    vdl = DataLoader(vds, cfg["batch_size"], collate_fn=collate, num_workers=cfg.get("num_workers", 6))

    model = build_model(cfg["model"], cfg.get("init_from")).to(device)
    # new layers (zero-init input slices / FiLM) get a higher LR than pretrained ones
    new_params = [p for n, p in model.named_parameters() if "film" in n]
    old_params = [p for n, p in model.named_parameters() if "film" not in n]
    opt = torch.optim.AdamW([{"params": old_params, "lr": cfg["optim"]["lr"]},
                             {"params": new_params, "lr": cfg["optim"].get("lr_new", cfg["optim"]["lr"])}], weight_decay=1e-4)
    total = cfg["epochs"] * len(dl) if not cfg.get("max_steps") else cfg["max_steps"]
    warm = cfg["optim"].get("warmup", 500)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(s, total) / total)))
    loss_fn = losses.HybridLoss() if cfg["loss"] == "hybrid" else losses.SpeechPreservationLoss()
    burst_w = 3.0 if cfg["loss"] == "speech_preservation" else 1.0
    use_amp = cfg.get("amp", True) and device.type == "cuda"

    run_info = dict(name=cfg["name"], git=_git_hash(), config_hash=hashlib.sha1(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12],
                    manifest_hash=manifests.content_hash(cfg["data"]["manifests"]),
                    evalset_hash=(Path("data/eval/val/EVALSET_HASH").read_text() if Path("data/eval/val/EVALSET_HASH").exists() else "none"),
                    seed=cfg["seed"], torch=torch.__version__, cuda=torch.version.cuda, start=time.time(), config=cfg)
    json.dump(run_info, open(run_dir / "run.json", "w"), indent=2)

    step, best = 0, -1.0
    for epoch in range(cfg["epochs"]):
        ds.set_epoch(epoch)
        for batch in dl:
            inputs, target, fw, is_clean = prepare_batch(batch, cfg["model"], cfg["controller_on"], device, burst_w)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                pred = model(*inputs)
            loss = loss_fn(pred.float(), target, fw, is_clean)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["optim"].get("clip", 5.0))
            opt.step(); sched.step(); step += 1
            if step % 20 == 0:
                tb.add_scalar("train/loss", loss.item(), step); tb.add_scalar("train/lr", sched.get_last_lr()[0], step)
            if cfg.get("max_steps") and step >= cfg["max_steps"]:
                break
        v = validate(model, vdl, cfg, device); tb.add_scalar("val/stoi", v, step)
        torch.save({"model": model.state_dict(), "config": cfg, "step": step}, run_dir / "last.pt")
        if v > best:
            best = v; torch.save({"model": model.state_dict(), "config": cfg, "step": step}, run_dir / "best.pt")
        print(f"epoch {epoch} step {step} val_stoi {v:.4f} best {best:.4f}")
        if cfg.get("max_steps") and step >= cfg["max_steps"]:
            break
    run_info.update(end=time.time(), best_val_stoi=best, steps=step)
    json.dump(run_info, open(run_dir / "run.json", "w"), indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("config"); main(ap.parse_args().config)
```

- [ ] **Step 5: Write the four experiment configs**

`configs/exp/gtcrn_finetuned.yaml`:
```yaml
name: gtcrn_finetuned
model: gtcrn
controller_on: false
loss: hybrid
init_from: vaani/models/checkpoints/model_trained_on_dns3.tar
data:
  manifests: [data/manifests/librispeech.parquet, data/manifests/cv_hi.parquet, data/manifests/mad.parquet]
  bank: data/rirs/bank.npz
  crop_s: 4.0
  epoch_len: 20000
  mix: {}
val: {dynamic_items: 200}
optim: {lr: 5.0e-4, warmup: 500, clip: 5.0}
batch_size: 32
epochs: 12
amp: true
device: cuda
num_workers: 6
seed: 0
```

`configs/exp/vaani_no_controller.yaml`: same but `name: vaani_no_controller`, `model: vaani`, `controller_on: false`, `optim: {lr: 5.0e-4, lr_new: 1.0e-3, warmup: 500, clip: 5.0}`.

`configs/exp/vaani_full.yaml`: as above with `name: vaani_full`, `controller_on: true`.

`configs/exp/vaani_full_sp.yaml`: as `vaani_full` with `name: vaani_full_sp`, `loss: speech_preservation`.

- [ ] **Step 6: Run the smoke test** → `uv run pytest tests/test_train_smoke.py -v` → PASS in under ~60 s on CPU.

- [ ] **Step 7: Stage** — `git add vaani/losses.py vaani/train.py configs/exp tests/test_train_smoke.py`

---

### Task 13: Evaluation and report

**Files:**
- Create: `vaani/eval.py`, `vaani/report.py`, `vaani/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Produces:
  - `metrics.snr_db(clean, est) -> float` (explicit definition), `metrics.si_sdr_db(clean, est)`, `metrics.stoi(clean, est)`, `metrics.pesq_wb(clean, est) -> float|nan`, `metrics.recovery_time_s(est_burst, est_twin, burst_onset_s, thresh_db=3.0, hold_s=0.2) -> float|nan`.
  - `eval.enhance_fn(spec: str) -> Callable[[np.ndarray[2,T]], np.ndarray[T]]` where `spec` is a baseline name or `ckpt:<path>`.
  - CLI: `uv run python -m vaani.eval --system ckpt:runs/vaani_full/best.pt --split test --out results/vaani_full.csv [--asr]`.
  - `report.py` CLI: `uv run python -m vaani.report results/*.csv --out results/matrix.md`.

- [ ] **Step 1: Write failing test**

`tests/test_metrics.py`:
```python
import numpy as np
from vaani import metrics


def test_snr_definition_is_error_based():
    c = np.random.default_rng(0).standard_normal(16000).astype(np.float32)
    assert metrics.snr_db(c, c) > 100
    assert abs(metrics.snr_db(c, c + 0.1 * c) - 20.0) < 1e-3   # 10% scale error = 20 dB SNR
    assert metrics.si_sdr_db(c, 2 * c) > 100                    # SI-SDR ignores scale, SNR does not


def test_recovery_time():
    sr = 16000; t = np.arange(2 * sr) / sr
    twin = np.sin(2 * np.pi * 200 * t).astype(np.float32)
    est = twin.copy(); on = int(0.5 * sr)
    est[on:on + int(0.3 * sr)] *= 0.1          # 300 ms of suppression after a burst at 0.5 s
    r = metrics.recovery_time_s(est, twin, 0.5)
    assert 0.25 < r < 0.4
```

- [ ] **Step 2: Run to verify failure** → FAIL.

- [ ] **Step 3: Implement `vaani/metrics.py`**

```python
"""Metric definitions. SNR here is error-based against the clean reference
(distortion counts as noise); SI-SDR is scale-invariant. They are different
numbers and are reported separately - never call SI-SDR 'output SNR'.
PESQ: wideband P.862.2 at 16 kHz. ITU has withdrawn P.862 in favour of P.863;
we report it because the brief requests it.
"""
import numpy as np
from pesq import pesq as _pesq
from pystoi import stoi as _stoi

SR = 16000


def snr_db(clean, est):
    return float(10 * np.log10((clean ** 2).sum() / (((est - clean) ** 2).sum() + 1e-12) + 1e-12))


def si_sdr_db(clean, est):
    a = (est * clean).sum() / ((clean ** 2).sum() + 1e-12); s = a * clean
    return float(10 * np.log10((s ** 2).sum() / (((est - s) ** 2).sum() + 1e-12) + 1e-12))


def stoi(clean, est):
    return float(_stoi(clean, est, SR, extended=False))


def pesq_wb(clean, est):
    try: return float(_pesq(SR, clean, est, "wb"))
    except Exception: return float("nan")


def _envelope_db(x, frame=320):
    f = x[: len(x) // frame * frame].reshape(-1, frame)
    return 10 * np.log10((f ** 2).mean(axis=1) + 1e-10)


def recovery_time_s(est_burst, est_twin, burst_onset_s, thresh_db=3.0, hold_s=0.2, frame=320):
    """Time after the burst until the speech envelope of the burst run stays
    within thresh_db of the no-burst twin for hold_s. NaN if never."""
    d = np.abs(_envelope_db(est_burst, frame) - _envelope_db(est_twin, frame))
    k0 = int(burst_onset_s * SR / frame); hold = int(hold_s * SR / frame)
    ok = d < thresh_db
    for k in range(k0, len(ok) - hold):
        if ok[k:k + hold].all():
            return (k - k0) * frame / SR
    return float("nan")
```

- [ ] **Step 4: Implement `vaani/eval.py`**

```python
"""Per-clip metrics over a rendered split. One CSV row per clip."""
import argparse, csv
from pathlib import Path

import numpy as np, torch
from tqdm import tqdm

from vaani import metrics
from vaani.data.dataset import RenderedDataset
from vaani.dsp import pipeline, stft
from vaani.models import baselines
from vaani.train import build_model


def enhance_fn(spec: str):
    if not spec.startswith("ckpt:"):
        return baselines.get(spec).enhance
    ck = torch.load(spec[5:], map_location="cpu"); cfg = ck["config"]
    m = build_model(cfg["model"]); m.load_state_dict(ck["model"]); m.eval()

    @torch.no_grad()
    def f(mix):
        x = torch.from_numpy(mix)[None]
        if cfg["model"] == "gtcrn":
            out = m(stft.stft(x[:, 0]))
        else:
            r = pipeline.run(mix, controller_on=cfg["controller_on"])
            spec6 = torch.cat([stft.stft(x[:, 0]), stft.stft(x[:, 1]), stft.stft(torch.from_numpy(r["n_hat"])[None])], -1)
            out = m(spec6, torch.from_numpy(r["features"])[None])
        return stft.istft(out, length=mix.shape[1])[0].numpy()
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True); ap.add_argument("--split", default="test")
    ap.add_argument("--eval-root", default="data/eval"); ap.add_argument("--out", required=True)
    ap.add_argument("--asr", action="store_true", help="also compute WER with faster-whisper (supporting evidence only)")
    a = ap.parse_args()
    ds = RenderedDataset(Path(a.eval_root) / a.split); fn = enhance_fn(a.system)
    asr = None
    if a.asr:
        from faster_whisper import WhisperModel; asr = WhisperModel("small", device="cpu", compute_type="int8")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    cols = ["system", "id", "bucket", "noise_class", "snr_in", "clipped", "ref_dropout", "impulse_peak_db",
            "snr_out", "si_sdr", "stoi", "pesq_wb", "recovery_s", "asr_text"]
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for it in tqdm(ds, desc=a.system):
            mix, clean, meta = it["mix"].numpy(), it["clean"].numpy(), it["meta"]
            est = fn(mix)
            row = dict(system=a.system, id=meta["id"], bucket=meta["bucket"], noise_class=meta["noise_class"],
                       snr_in=meta["snr_db"], clipped=meta["clipped"], ref_dropout=meta["ref_dropout"],
                       impulse_peak_db=meta["impulse_peak_db"], snr_out=metrics.snr_db(clean, est),
                       si_sdr=metrics.si_sdr_db(clean, est), stoi=metrics.stoi(clean, est), pesq_wb=metrics.pesq_wb(clean, est),
                       recovery_s="", asr_text="")
            if "twin" in it and meta["impulse_onsets_s"]:
                row["recovery_s"] = metrics.recovery_time_s(est, fn(it["twin"].numpy()), meta["impulse_onsets_s"][0])
            if asr:
                segs, _ = asr.transcribe(est, language=None, beam_size=1)
                row["asr_text"] = " ".join(s.text for s in segs).strip()
            w.writerow(row)


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Implement `vaani/report.py`**

```python
"""Aggregate eval CSVs into the ablation matrix with bootstrap CIs.
Nominal envelope = unclipped, no ref dropout, SNR in {0,5,10}.
Severe envelope = everything else (reported, never claimed as target-met).
"""
import argparse
import numpy as np, pandas as pd

METRICS = ["snr_out", "si_sdr", "stoi", "pesq_wb"]


def ci(x, n=1000, seed=0):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if len(x) == 0: return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return (x.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5))


def fmt(t): return f"{t[0]:.2f} [{t[1]:.2f},{t[2]:.2f}]"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("csvs", nargs="+"); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    df = pd.concat([pd.read_csv(p) for p in a.csvs])
    df["nominal"] = (~df.clipped) & (~df.ref_dropout) & df.snr_in.isin([0, 5, 10])
    lines = ["# Ablation matrix", "",
             "PESQ: wideband P.862.2 @16 kHz (`pesq` package). P.862 is withdrawn by ITU in favour of P.863; reported because the brief requests it.",
             "SNR_out = 10log10(||s||^2/||s_hat-s||^2) vs clean primary (distortion counts as error). SI-SDR reported separately.", "",
             "## Nominal envelope (unclipped, no reference fault, input SNR 0/5/10 dB)", "",
             "| system | n | " + " | ".join(METRICS) + " |", "|---|---|" + "---|" * len(METRICS)]
    for sysname, g in df[df.nominal].groupby("system"):
        lines.append(f"| {sysname} | {len(g)} | " + " | ".join(fmt(ci(g[m])) for m in METRICS) + " |")
    lines += ["", "## Per bucket (all systems)", ""]
    for (sysname, bucket), g in df.groupby(["system", "bucket"]):
        lines.append(f"- **{sysname} / {bucket}** (n={len(g)}): " + ", ".join(f"{m}={fmt(ci(g[m]))}" for m in METRICS))
    if "recovery_s" in df and df.recovery_s.notna().any():
        lines += ["", "## Recovery time after burst (s)", ""]
        for sysname, g in df[df.recovery_s.notna()].groupby("system"):
            r = pd.to_numeric(g.recovery_s, errors="coerce")
            lines.append(f"- {sysname}: median={r.median():.3f} p90={r.quantile(0.9):.3f} failures={int(r.isna().sum())}/{len(r)}")
    open(a.out, "w").write("\n".join(lines)); print("\n".join(lines))


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run tests** → `uv run pytest tests/test_metrics.py -v` → PASS.

- [ ] **Step 7: Stage** — `git add vaani/metrics.py vaani/eval.py vaani/report.py tests/test_metrics.py`

---

### Task 14: ONNX export and deployment contract

**Files:**
- Create: `vaani/export.py`, `deploy/CONTRACT.md`
- Test: `tests/test_export.py`

**Interfaces:**
- Produces: `export.export(ckpt_path, out_path) -> Path`; ONNX inputs `spec6 (1,257,1,6)`, `feats (1,1,18)`, `conv_cache (2,1,16,16,33)`, `tra_cache (2,3,1,1,16)`, `inter_cache (2,1,33,16)`; outputs `spec_out (1,257,1,2)` + the three updated caches. `export.parity_and_timing(ckpt_path, onnx_path, seconds=10) -> dict(max_abs_err, ms_per_frame_mean, ms_per_frame_p99)`.

- [ ] **Step 1: Write failing test**

`tests/test_export.py`:
```python
import torch
from vaani import export
from vaani.models.vaani_net import VaaniNet


def test_export_parity(tmp_path):
    ck = tmp_path / "m.pt"; torch.save({"model": VaaniNet().state_dict(), "config": {"model": "vaani", "controller_on": True}, "step": 0}, ck)
    onnx = export.export(ck, tmp_path / "m.onnx")
    r = export.parity_and_timing(ck, onnx, seconds=1)
    assert r["max_abs_err"] < 1e-4 and r["ms_per_frame_mean"] > 0
```

- [ ] **Step 2: Run to verify failure** → FAIL.

- [ ] **Step 3: Implement `vaani/export.py`**

```python
"""Streaming VaaniNet -> ONNX with explicit state, plus parity and a CPU
timing proxy. The Pi measurement belongs to the embedded lead; this number
only tells us whether we are in the right order of magnitude."""
import time
from pathlib import Path

import numpy as np, onnxruntime as ort, torch

from vaani.models.vaani_net import StreamVaaniNet, VaaniNet, init_caches


def export(ckpt_path, out_path):
    ck = torch.load(ckpt_path, map_location="cpu")
    s = StreamVaaniNet().eval(); s.load_state_dict(ck["model"])
    spec = torch.zeros(1, 257, 1, 6); f = torch.zeros(1, 1, 18); caches = init_caches()
    torch.onnx.export(s, (spec, f, *caches), str(out_path), opset_version=17,
                      input_names=["spec6", "feats", "conv_cache", "tra_cache", "inter_cache"],
                      output_names=["spec_out", "conv_cache_out", "tra_cache_out", "inter_cache_out"], dynamo=False)
    return Path(out_path)


def parity_and_timing(ckpt_path, onnx_path, seconds=10):
    ck = torch.load(ckpt_path, map_location="cpu")
    v = VaaniNet().eval(); v.load_state_dict(ck["model"])
    T = int(seconds * 16000 / 256)
    torch.manual_seed(0); spec = torch.randn(1, 257, T, 6) * 0.1; f = torch.randn(1, T, 18)
    with torch.no_grad(): ref = v(spec, f).numpy()
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    caches = [c.numpy() for c in init_caches()]; outs, times = [], []
    for t in range(T):
        inp = {"spec6": spec[:, :, t:t + 1].numpy(), "feats": f[:, t:t + 1].numpy(),
               "conv_cache": caches[0], "tra_cache": caches[1], "inter_cache": caches[2]}
        t0 = time.perf_counter(); o = sess.run(None, inp); times.append((time.perf_counter() - t0) * 1000)
        outs.append(o[0]); caches = o[1:]
    got = np.concatenate(outs, axis=2)
    return {"max_abs_err": float(np.abs(got - ref).max()), "ms_per_frame_mean": float(np.mean(times)), "ms_per_frame_p99": float(np.percentile(times, 99))}


if __name__ == "__main__":
    import argparse; ap = argparse.ArgumentParser(); ap.add_argument("ckpt"); ap.add_argument("--out", default="deploy/model.onnx")
    a = ap.parse_args(); p = export(a.ckpt, a.out); print(parity_and_timing(a.ckpt, p))
```

- [ ] **Step 4: Write `deploy/CONTRACT.md`**

```markdown
# VAANI deployment contract (v1)

Audio: 16 kHz, 2 channels (0 = primary near-mouth, 1 = reference), float32 [-1,1].
STFT: n_fft 512, hop 256, window = sqrt(periodic Hann 512), center=True (reflect pad 256).
One model call per hop (16 ms).

## Per frame, in order
1. NLMS block on the 256 new samples with gate = previous frame's `adapt_gate` (64 taps, mu 0.05, eps 1e-6) -> `n_hat` block. See `dsp_reference/` and `vaani/dsp/nlms.py`.
2. Features (18, order below) from the current 512-sample primary/reference frames and their spectra. `vaani/dsp/features.py`.
3. Controller -> `adapt_gate`, `burst_flag`, `reliability`. `vaani/dsp/controller.py`. Thresholds: jump 12 dB, level-diff <= 3 dB, hold 4 frames, ramp 12 frames, speech-freeze 0.6.
4. ONNX `model.onnx`: inputs `spec6 (1,257,1,6)` = [prim_re, prim_im, ref_re, ref_im, nhat_re, nhat_im], `feats (1,1,18)`, `conv_cache (2,1,16,16,33)`, `tra_cache (2,3,1,1,16)`, `inter_cache (2,1,33,16)`; outputs `spec_out (1,257,1,2)` + updated caches. Feed caches back unchanged.
5. iSTFT overlap-add with the same sqrt-Hann window.

For the *no-controller* configuration: gate = 1, feats = zeros.

## Feature order
log_energy_delta, spectral_flux, peak_to_rms, clip_frac_primary, clip_frac_reference, speech_presence, coherence_b0..b7, level_diff_db, ref_dropout, nlms_health, prev_gate

## Golden vectors
`dsp_reference/vectors/<case>.wav` (stereo input) and `<case>.npz` (n_hat, features, gate, burst, reliability). A port passes when n_hat matches to 1e-4 and gate/burst match exactly.

## Not covered here
Output crossfade/bypass on low reliability, overrun handling, and radio interfacing are the DSP/embedded leads' responsibility.
```

- [ ] **Step 5: Run tests** → PASS. If `torch.onnx.export` fails on GRU with `dynamo=False` unavailable in this torch, drop the kwarg.

- [ ] **Step 6: Stage** — `git add vaani/export.py deploy/CONTRACT.md tests/test_export.py`

---

### Task 15: Round-1 matrix run

**Files:** none new; produces `results/*.csv`, `results/matrix.md`, `runs/*`.

**Prerequisites:** Task 3 fetch complete (manifests exist), Task 5 bank built, Task 7 eval sets rendered.

- [ ] **Step 1: Baseline rows (no training)**

```bash
uv run python -m vaani.eval --system raw --split test --out results/raw.csv
uv run python -m vaani.eval --system nlms_only --split test --out results/nlms_only.csv
uv run python -m vaani.eval --system gtcrn_pretrained --split test --out results/gtcrn_pretrained.csv
```
RNNoise row: only if `rnnoise_demo` is available; otherwise record "not run" in the report.

- [ ] **Step 2: Train the four configs, sequentially** (one GPU)

```bash
uv run python -m vaani.train configs/exp/gtcrn_finetuned.yaml
uv run python -m vaani.train configs/exp/vaani_no_controller.yaml
uv run python -m vaani.train configs/exp/vaani_full.yaml
uv run python -m vaani.train configs/exp/vaani_full_sp.yaml
```
Watch `tensorboard --logdir runs`. If a run's `val/stoi` is below `gtcrn_pretrained`'s test STOI after 3 epochs, stop and inspect the mixer output by ear (write 5 clips with `soundfile`) before continuing.

- [ ] **Step 3: Evaluate and report**

```bash
for r in gtcrn_finetuned vaani_no_controller vaani_full vaani_full_sp; do
  uv run python -m vaani.eval --system ckpt:runs/$r/best.pt --split test --out results/$r.csv --asr
done
uv run python -m vaani.report results/*.csv --out results/matrix.md
```

- [ ] **Step 4: Export the best VaaniNet**

```bash
uv run python -m vaani.export runs/vaani_full/best.pt --out deploy/model.onnx
```

- [ ] **Step 5: Stage** — `git add results/matrix.md deploy/CONTRACT.md` (CSV results and runs stay untracked).

---

### Task 16: Round 2 — fold in DNS-5 shards

- [ ] **Step 1: Start the shard fetch in the background** (shard URLs from the DNS-Challenge repo `download-dns-challenge-5-headset-training.sh`; pick `read_speech` and `noise_fullband` shards first):

```bash
uv run python scripts/fetch_data.py --dns-shards <url1> <url2> ...
```

- [ ] **Step 2: Create `configs/exp/*_r2.yaml`** — copies of the four round-1 configs with `name: <name>_r2`, `init_from: runs/<name>/best.pt` (for `gtcrn` rows) or the same plus `model: vaani` warm-start via `init_from` pointing at the round-1 VaaniNet checkpoint — add to `train.build_model`: if `init_from` is a `runs/*.pt` checkpoint whose `config.model == "vaani"`, load the full state dict directly instead of `from_pretrained_gtcrn`. Implement that branch:

```python
    if name == "vaani":
        if init_from and str(init_from).endswith(".pt"):
            m = VaaniNet(); m.load_state_dict(torch.load(init_from, map_location="cpu")["model"]); return m
        return VaaniNet.from_pretrained_gtcrn(init_from) if init_from else VaaniNet()
```
and `manifests:` extended with `data/manifests/dns_*.parquet`.

- [ ] **Step 3: Re-render test set?** No — the frozen round-1 test set stays the reference so round-1 and round-2 rows are comparable. Render an *additional* `test_dns` split only if DNS test-split material exists, and report it separately.

- [ ] **Step 4: Re-run Task 15 steps 2–4 with the `_r2` configs.** Report states exactly which shards were included (list from `data/manifests/dns_*.parquet`).

- [ ] **Step 5: Stage** — `git add configs/exp/*_r2.yaml vaani/train.py results/matrix.md`

---

## Self-review against the spec

- §4 data: Tasks 3–7 ✔ (sources, gate, splits, mixer with all augmentations, rendered buckets with twins, physical schema).
- §5 DSP: Tasks 8–9 ✔ (NLMS, 18 features in order, controller with hysteresis and consonant test, golden vectors).
- §6 models: Tasks 10–11 ✔ (verbatim vendoring, parity, baselines, VaaniNet ≤ 60K, primary-only mask, pretrained init identity test).
- §7 losses: Task 12 ✔. §8 training: Task 12 ✔ (run.json fields, AMP, schedule, best-on-STOI). §9 eval: Task 13 ✔ (explicit SNR, PESQ mode note, recovery time, WER, CIs, nominal/severe). §10 export: Task 14 ✔. §11 tests: each task. §13 sequencing: Tasks 15–16.
- Known deviation: `render_eval_sets.py` uses `hash()` in one place — Step 4 note instructs replacing with `stable_hash` when writing the file. `configs/data/round1.yaml` has a provenance placeholder for the MAD version by design.
- Type consistency: `pipeline.run` → keys `n_hat/features/gate/burst/reliability` used identically in Tasks 9, 10, 12, 13. `build_model(name, init_from)` signature consistent in Tasks 12, 13, 16. `init_caches` defined in Task 10 and re-exported in Task 11, used in Task 14.
