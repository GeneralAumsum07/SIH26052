# Native crashes in evaluation: pesq 0.0.4 out-of-bounds read

The rc=139 / "Windows fatal exception: access violation" crashes in wf3 were one bug: the PESQ C code
(pesq 0.0.4, ludlows/PESQ, the ITU P.862 reference code) reads before the start of a heap buffer on some
noisy inputs. Timing numbers below are smoke measurements on the shared laptop: not reportable.

## Root cause (established)

In `utterance_split` (pesqmod.c), when an utterance is split and the second part has the larger delay
(`Best_D2 >= Best_D1`), its start is set to `Best_BP - (Best_D2 - Best_D1) / (2 * Downsample)`. Only the
first part's start is clamped afterwards. A spurious huge delay jump makes the second start negative, and the
next loop iteration reads `ref_info->VAD[Utt_SpeechStart]` with that negative index.

- clip `changing_5/0018`: D1 -7385, D2 54688, BP 324 -> start -160
- clip `stationary_-10/0028`: D1 -3084, D2 62012, BP 205 -> start -303 (ASan register rbx = -303)

Both are eval_r2_relabel/test raw inputs (mix channel 0 vs clean), i.e. noise-dominated alignments.
If the address before the buffer is mapped, the read returns heap garbage and PESQ returns a value that
changes from process to process; if not, the process dies. That is why the crash looked random in the clip.

Evidence:
- ASan (WSL Ubuntu, gcc 15.2, `-fsanitize=address`, harness `asan/harness_wav.c` that normalises as
  pesq/_pesq.py does): both clips SEGV every time at `utterance_split` (the VAD read), called from
  `utterance_locate` <- `pesq_measure`. The 2-line instrumented copy (`int g_negstart` counter, incremented when
  `Utt_Start < 0` at the top of the loop) is not committed: the PESQ notice forbids redistributing altered
  code. Recreate it from the description to rerun `asan/sweep.sh`.
- Sweep of all 2,280 eval_r2_relabel/test raw inputs (`asan/sweep_relabel_test.tsv`, columns: id, pesq flag,
  MOS, negstart count, ASan rc, ASan summary): exactly 2 inputs have a negative start and both are the only
  2 ASan failures; the other 2,278 are clean under ASan. Command (job segfault-asan-relabel, rc 0):
  `wsl.exe -d Ubuntu -- bash asan/sweep.sh eval_r2_relabel/test <out.tsv>`.
- Windows, in-process, fresh processes (`repro.py`, JSON beside it), 6 processes x up to 10 calls:

  | clip | processes crashed | distinct values over completed calls |
  |---|---|---|
  | stationary_-10/0028 | 6/6 (after 1-3 calls, rc 3221225477 = access violation) | 3 (1.041160..1.068232) |
  | changing_5/0018 | 0/6 in this run (5 of 6 fresh processes crashed in an earlier 50-call run) | 10 (1.094373..1.378523) |
  | stationary_-10/0021 (control) | 0/6 | 1 (1.355817) |

  `.venv/Scripts/python.exe results_r2/r8/native_crash/repro.py <clip> --procs 6 --calls 10`
- faulthandler stacks end in `pesq/_pesq.py:65 _pesq_inner` in the wf3 crash with faulthandler on (job
  diag-cond2) and in both repro sequences (jobs segfault-rep2a stoi+pesq, segfault-rep2b pesq only, raw
  eval_r2_relabel/test; each died at changing_5/0018 or stationary_-10/0028). diag-rawrelabel2 (no
  faulthandler) segfaulted twice on the same raw sequence. The upstream master `utterance_split` still has no clamp for the second part (checked
  2026-09-25): there is no fixed release to move to.

Ruled out, each by a crash that happens without it: pystoi (pesq-only sequence crashes), torch / numba /
onnxruntime / DNSMOS (the raw system path loads none of them and crashes), multiprocessing and thread safety
(`--workers 0` crashes; so does one bare process), memory pressure (ASan faults deterministically on an idle
WSL VM; the crashing read is a negative index, not an allocation failure). Soundfile is not in any stack.

## Fix

The fault is inside the PESQ C code and depends on its internal alignment, so it cannot be guarded from
Python without changing values. `vaani.metrics.pesq_wb` therefore runs the unchanged C call in one
persistent child process per calling process (`_PesqWorker`):
- a child death or a hang (timeout 60 s + 2 x duration) scores NaN, prints a WARNING with an input hash,
  counts in `metrics.pesq_failures()`, and the next call starts a new child;
- `VAANI_PESQ_CRASH_DIR=<dir>` saves each crashing input as `.npz`; `VAANI_PESQ_ISOLATE=0` restores the old
  in-process call; `metrics.pesq_wb_inproc` is the old function;
- values are bit-identical to the in-process call on every input that does not fault (arrays travel by
  pickle, dtype kept): tests/test_native_isolation.py, tests/test_metrics_known_answer.py.
- overhead (smoke): 0.207 s in-process vs 0.211 s isolated per 6 s clip; child working set about 13 MB.

Training: the r8 composite screen (`CompositeScreen.score`, in-process) and the frozen val screen
(`train_refiner.score_items` -> spawn Pool -> `_metric_item`) both call `metrics.pesq_wb`, so a PESQ fault can
no longer kill the run or a pool worker. The composite summary carries `pesq_failures` and logs a WARNING; a
crashed clip scores NaN and fails its pass target. `validate()` logs PESQ over the scored items plus `pesq_nan`.
STOI and checkpoint selection code are unchanged. vaani.eval already fails loudly on a dead worker
(ProcessPoolExecutor + BrokenProcessPool).

Not fixed: on the offending inputs that do not fault, PESQ is still computed from garbage memory (the value
spread above). Measured rate: 2 of 2,280 raw noisy test inputs (0.09 %); model outputs were not swept.

## Rerun of the crashing command

`scripts/diag_conditioning.py` on the r7 backbone over eval_r2 val (the run that crashed at clip
`recorded_impulsive+stationary_-10/0009`), with numba, faulthandler and VAANI_PESQ_CRASH_DIR set. Its outputs
are evidence only (the committed results_r2/r7/diag files belong to that diagnostic):
docs/impl/2026-09-24/evidence/segfault/ (git-ignored). Command (from the repo root, 23:29-00:10 IST):

    CUDA_VISIBLE_DEVICES=-1 PYTHONFAULTHANDLER=1 VAANI_PESQ_CRASH_DIR=docs/impl/2026-09-24/evidence/segfault/pesq_crash       uv run --with numba python -X faulthandler scripts/diag_conditioning.py --ckpt results_r2/runs/r7_e256_wr64/best.pt       --eval-root data/eval_r2 --split val --n 0 --out docs/impl/2026-09-24/evidence/segfault/conditioning_backbone

Result: rc 0, 1480/1480 clips, 0 skipped (the committed run has 1479, clip 0009 skipped). Three pesq children
died (access violation in `_pesq_inner`, rc 3221225477) and scored NaN, all in the "ref zeroed" variant:
`recorded_impulsive+stationary_-10/0009`, `recorded_impulsive+stationary_-10/0033`, `stationary_-5/0020`
(inputs saved as pesq_crash_*.npz). The other 5,916 rows match results_r2/r7/diag/conditioning_backbone.csv
to <= 4e-15 (STOI exactly). Headline as trained: STOI 0.8910, PESQ-WB 2.3003 (n=1480).
Inferred: the committed PESQ for 0033 and 0020 "ref zeroed" (1.206329, 1.220508) are the non-faulting
garbage-read draws of the same bug; dropping both moves the committed variant mean by about 0.0005.

