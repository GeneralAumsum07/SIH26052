"""DeepFilterNet3 worker: runs under the isolated .venv-dfn (py3.11, torch 2.0.1) because deepfilternet
pins numpy<2 and an old torchaudio. Loads the model once, then serves "in.wav out.wav" lines on stdin
(one clip ≈0.3 s vs ≈20 s per fresh process). Replies "ok"/"err <msg>" per line. Driven by baselines.DeepFilterNet3."""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, soundfile as sf, torch, torchaudio.functional as AF
from df.enhance import init_df, enhance

model, state, _ = init_df(log_level="ERROR", log_file=None)
SR = state.sr()   # 48000: DFN3 is a 48 kHz model; we resample 16 kHz clips in and out
print("ready", flush=True)
for line in sys.stdin:
    try:
        src, dst = line.rstrip("\n").split("\t")
        x, sr = sf.read(src, dtype="float32")
        x = torch.from_numpy(x.T if x.ndim == 2 else x[None])[:1]
        y = enhance(model, state, AF.resample(x, sr, SR))
        y = AF.resample(y, SR, sr)[0, : x.shape[1]].numpy()
        sf.write(dst, np.pad(y, (0, x.shape[1] - len(y))), sr, subtype="FLOAT")
        print("ok", flush=True)
    except Exception as e:  # keep serving; the caller turns this into a NaN row
        print(f"err {e!r}", flush=True)
