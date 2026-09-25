"""Minimal reproducer: in-process pesq 0.0.4 on one eval_r2_relabel/test raw input, in fresh processes.
Each process calls pesq up to --calls times; a native fault ends it with a non-zero return code.
Usage (repo root): .venv/Scripts/python.exe results_r2/r8/native_crash/repro.py changing_5/0018 --procs 6 --calls 10"""
import argparse, json, subprocess, sys

CHILD = r'''
import faulthandler, sys; faulthandler.enable()
import soundfile as sf
from pesq import pesq
d = "data/eval_r2_relabel/test/" + sys.argv[1]
clean, _ = sf.read(d + ".clean.wav"); mix, _ = sf.read(d + ".mix.wav")
for k in range(int(sys.argv[2])):
    print(k, pesq(16000, clean, mix[:, 0], "wb"), flush=True)   # raw input = mix channel 0, as vaani.eval --system raw
'''

ap = argparse.ArgumentParser()
ap.add_argument("clip"); ap.add_argument("--procs", type=int, default=6); ap.add_argument("--calls", type=int, default=10)
a = ap.parse_args()
runs = []
for _ in range(a.procs):
    p = subprocess.run([sys.executable, "-c", CHILD, a.clip, str(a.calls)], capture_output=True, text=True)
    vals = [float(l.split()[1]) for l in p.stdout.split("\n") if l.strip()]
    runs.append(dict(rc=p.returncode, calls_completed=len(vals), values=vals))
print(json.dumps(dict(clip=a.clip, procs=a.procs, calls=a.calls, crashed=sum(r["rc"] != 0 for r in runs), runs=runs), indent=1))
