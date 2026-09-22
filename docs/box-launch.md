# Box launch — the order of work on the rented GPU

Written 2026-09-22. Everything below is implemented and committed; this is the runbook, not a plan.

## 0. The one thing to know

Training is **dataloader-bound** (~130 ms of a ~144 ms step is CPU). So the box is rented for
cores, not for the GPU, and every optimisation here attacks CPU work:

| change | effect | where |
|---|---|---|
| Packed int16 corpus | no FLAC decode, no file opens in the hot path | `vaani/data/pack.py`, `scripts/pack_corpus.py` |
| Shared batch stream | the epoch sweep costs 256 epochs of data, not 480 | `vaani/train_multi.py` |
| `num_workers: auto` | sizes from the box instead of a constant | `vaani/runtime.py` |
| Direct gated fetch | removes the ~2.5 h wait on the laptop uplink | `scripts/remote_setup.sh` |
| Page-cache warm | first epoch does not pay cold reads | `scripts/remote_setup.sh` |

The packed path is **verified bit-identical and rng-preserving** (`tests/test_pack.py`): it
changes speed and nothing else. If it is ever wrong, training data changed silently, which is
why that test exists.

## 1. Environment to set before `remote_setup.sh`

Only the first is required; the rest each remove a wait.

```bash
export VAANI_WORKERS=              # optional: overrides num_workers everywhere. Leave unset to auto-size.
export RIR_BANK_URL=...            # release asset base; banks fetched by sha256 rather than regenerated
export MIRROR_URL=...              # same idea for noisex92 and the frozen eval_r2 tarball
export KAGGLE_USERNAME= KAGGLE_KEY=    # MAD, straight from Kaggle
export MDC_API_KEY=                # Common Voice Hindi via the Mozilla Data Collective API
```

Without `MIRROR_URL`/`KAGGLE_KEY`/`MDC_API_KEY` the script still works — it falls back to waiting
for the laptop, exactly as before. With them, that stage is minutes.

**A regenerated RIR bank is a different file** (pyroomacoustics is not reproducible across
machines), which silently breaks comparability with every existing result. Set `RIR_BANK_URL`.

## 2. First twenty minutes

```bash
tmux new -s vaani                      # before anything long-running
nproc; free -g; df -h .; nvidia-smi    # record these in the run log
bash scripts/remote_setup.sh           # idempotent; every stage skips if its marker exists
```

Then, before committing to long runs:

```bash
uv run python scripts/profile_loader.py configs/retraining/r6_ctl64.yaml
```

That prints three things: the packed-vs-unpacked per-item cost, a cProfile of where an item's
time goes (**prediction on record: `pipeline.run` first, FLAC decode second**), and a worker
sweep ending in a recommended count.

**The falsification check.** During the first real epoch:

```bash
nvidia-smi dmon -s u -d 5
```

Sustained GPU utilisation **above ~60 %** means this box is not dataloader-bound and the
shared-stream plan needs rethinking. Below ~35 % is the expectation.

## 2b. Why the test suite runs on the box

35 of the 57 test files import torch, and this is the first environment that has the right one.
The dev laptop's venv is Windows-side; the Linux VM the desktop agent works in cannot install
torch at all (its egress proxy allows PyPI but blocks `download.pytorch.org`, and the PyPI wheel
is a CUDA build that will not import without several GB of nvidia packages). Installing a
mismatched torch there would produce failures that are not real on the box, which is worse than
not running them.

So `run_r6.sh` runs `pytest` as a **precondition**, before the scan and before any GPU time.
`SKIP_TESTS=1` bypasses it; don't, unless the failure is already understood.

What this means in practice: the first time `tests/test_pack.py`, `tests/test_train_multi.py` and
the two shared-stream tests in `tests/test_r6_arms.py` execute is on the box, ~60 s in. They are
cheap and they guard the two changes most able to corrupt a run silently — the packed data path
and the shared-stream guard.

## 3. The session

```bash
bash scripts/run_r6.sh                 # SWEEP=0 to skip the epoch sweep; ARMS_PARALLEL=3 to overlap the arms
```

Order, and why:

1. **Preconditions.** The protocol must be committed *and* clean, and its commit hash is written
   to `$OUT/PROTOCOL_COMMIT` so the registration-before-results claim is auditable rather than
   asserted.
2. **Scan, crest-audit, render `eval_gen`.** `render_eval_sets.py` refuses to overwrite a frozen
   set, so `eval_r2` cannot be damaged from here.
3. **Disjointness proven** before the held-out corpus is used.
4. **Epoch sweep + control on one shared stream** — `r6_e32`, `r6_ctl64`, `r6_e128`, `r6_e256`.
   256 epochs of dataloading covers all four budgets, and the comparison between them is
   **paired** (same data, same order), which is how it should be reported.
5. **Corpus arms separately** — `r6_demand64`, `r6_wham64`. These *cannot* share a stream: they
   differ in their manifests, and `train_multi` refuses by design. Sharing would feed them the
   control's data and void the comparison while still printing plausible numbers.
6. **Score everything**, then regenerate the licence table (WHAM! is CC BY-NC and changes the
   transfer story).

## 4. What to report from the run

- Paired, not independent: the four budgets and the control saw identical batches. Say so.
- Seed variants that share a stream vary in **initialisation only**, not in data.
- `df_norm` is written to each `run.json`: a null deep-filter result must be diagnosable as
  under-trained taps rather than believed.

## 5. Cost discipline

Pull checkpoints off after each run that completes, not at the end. Check `df -h` before each
wave. **Stop, don't destroy, between sessions** — the staged corpus and the pack are expensive
to rebuild, and a stopped instance bills only for disk.
