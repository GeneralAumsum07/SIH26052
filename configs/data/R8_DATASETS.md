# r8 datasets on the box

Every corpus r8 trains or gates on downloads straight onto the rented box. Nothing comes from the laptop except
the private mirror (the frozen val set and the VAD-filtered MAD manifest). The machine-readable table is
[`r8_datasets.yaml`](r8_datasets.yaml). The tool that reads it is `scripts/r8_datasets.py`.
`scripts/r8_box_setup.sh` runs it in its dataset stage, starting with the corpora the first queued jobs and
the G1 gate need (`scripts/r8_preflight.py --fetch-order`).

## The one command

```bash
python scripts/r8_datasets.py plan      # what will be fetched, sizes, missing credentials, disk needed
python scripts/r8_datasets.py fetch --parallel 8 && python scripts/r8_datasets.py scan && python scripts/r8_datasets.py verify
```

- **Scope.** With no `--only`, it covers every entry under `datasets:`. `--only a,b` narrows the set. `--all`
  adds the `optional:` entries.
- **Exit code.** The exit is 0 only when everything asked for is present and verified.
- **Resuming.** Downloads resume: rerun the same command after an interruption. A finished step leaves
  `data/raw/<name>_src/.fetched` and `.extracted`, so a rerun skips it.
- **Errors.** One dataset failing never stops the others. The failure is named at the end.

## Needs Rachit (do these before the box starts)

| What | Why | Steps |
|---|---|---|
| `MDC_API_KEY` | Common Voice 27.0 Hindi (`cv_hi`), in the first fetch group, so the box blocks without it | 1. Sign in at mozilladatacollective.com.<br>2. Open Common Voice Scripted Speech 27.0 Hindi and accept its terms.<br>3. Create an API key under the account.<br>4. Export it on the box.<br>The quota is 30 downloads/day per organisation. Each fetch run spends one, and it resumes the partial file. |
| `HF_TOKEN`, `MIRROR_HF_REPO` | The private mirror: the frozen val set and `mad_v2.parquet` | Use a read token for the mirror repo. The repo id is kept out of this public file. |
| `RIR_BANK_URL` | The RIR banks (`configs/data/r8_banks.json`) | Use the release asset base URL. |
| `KAGGLE_USERNAME` + `KAGGLE_KEY` (or `KAGGLE_API_TOKEN`) | MAD. Optional: the anonymous Kaggle download URL is tried first | Only needed if the anonymous URL starts refusing. Create the token at kaggle.com, Settings, API. |
| CADRE registration | `cadre` (optional, eval re-render only) | 1. Register for free at cadreforensics.com/audio/register/.<br>2. Run `fetch --only cadre`. |
| Svarah gate | `svarah` (optional) | Accept the dataset gate on huggingface.co (it is auto-approved) with the account whose `HF_TOKEN` is on the box. |
| IndicTTS | `indictts` (optional) | Licence request to IIT Madras. This tool does not fetch it. |

Export the variables in the box shell, never in a file in the repo. The tool reads them from the environment only
and never prints their values:

```bash
export MDC_API_KEY=...        # Common Voice Hindi
export HF_TOKEN=...           # private mirror (also the Svarah gate and the faster-whisper cache)
export MIRROR_HF_REPO=...     # private mirror repo id
export RIR_BANK_URL=...       # RIR bank release base
# optional: export KAGGLE_USERNAME=... KAGGLE_KEY=...
```

## What the box fetches by default (`datasets:`)

Sizes are archive bytes: probed Content-Length or publisher metadata, with the laptop copy's size where the host
sends none (esc50, drone). A checksum is checked when the publisher gives one. Otherwise, `verify` records the
sha256 of the first fetch and compares it with the laptop copy's hash as an advisory.

| Name | Access | Size (GB) | Checksum | Manifest(s) | Needed by | Licence |
|---|---|---|---|---|---|---|
| librispeech (train-clean-100) | direct | 6.387 | md5 | librispeech_100h | all r8 configs, G1 | CC BY 4.0 |
| ears (p001-p006) | direct | 3.768 | size + laptop sha256 | ears | all r8 configs | CC BY-NC 4.0 |
| cv_hi (CV 27.0 Hindi) | **login** (MDC_API_KEY) | 0.573 | sha256 (MDC) | cv_hi | all r8 configs, G1 | CC0-1.0; MDC terms forbid speaker re-identification and re-hosting |
| esc50 (pinned commit) | direct | 0.646 (est.) | recorded on first fetch | esc50 | all r8 configs, G1 | CC BY-NC 3.0 |
| dns_freesound_000 | direct | 3.470 | size + laptop sha256 | dns_…freesound_000.tar | all r8 configs, G1 | per-clip (DNS README) |
| dns_audioset_000 | direct | 5.365 | size + laptop sha256 | dns_…audioset_000.tar | G1 | per-clip (DNS README) |
| mad | direct (Kaggle) | 1.096 | size + laptop sha256 | mad (+ mad_v2 from the mirror) | all r8 configs, G1, G4 | conflict: Kaggle CC BY-SA 4.0 vs README CC BY 4.0; research only |
| gunshots (Zenodo 7004819) | direct | 1.568 | md5 | gunshots | all r8 configs | CC BY 4.0 |
| demand (Zenodo 1227121) | direct | 2.093 | md5 | demand, demand_pairs | all r8 configs (demand_pairs), G1 (demand) | conflict: CC BY 4.0 vs CC BY-SA 3.0 |
| drone | direct | 0.580 (est.) | recorded on first fetch | drone | r8_mini_refvalid | none stated |
| noisex92 (SPIB .mat) | direct | 0.144 | size | noisex92 | r8_mini_refvalid | unclear: eval only, never ship |
| faster_whisper_small | direct (HF) | 0.486 | sha256 (model.bin) | none (HF cache) | G4 | MIT |
| lombard_grid | direct | 0.653 | md5 | lombard_grid | all r8 configs | CC BY 4.0 |
| fsd50k | direct | 24.679 | md5 | fsd50k | all r8 configs; needs `zip` for the split archive | per clip, CC0/CC BY kept |
| c3gd | direct | 0.771 | md5 | c3gd | all r8 configs | CC BY 4.0 |
| avq_drone | direct | 0.060 | md5 | avq_drone | all r8 configs | CC BY 4.0 |

That is 52.3 GB of archives in all (`r8_datasets.py plan`; 156 GB on disk at the inferred x3 factor). The first group
(everything the r8 configs and G1 read, 51.1 GB) is librispeech, ears, cv_hi, esc50, dns_freesound_000,
dns_audioset_000, mad, gunshots, demand, lombard_grid, fsd50k, c3gd and avq_drone; drone, noisex92 and
faster_whisper_small follow.

## Optional (`optional:`, fetched only with `--only` or `--all`)

| Name | Size (GB) | Use | Licence |
|---|---|---|---|
| ears_more (p007-p107) | 65.4 | plan 11.5 speech | CC BY-NC 4.0 |
| librittsr (clean 100+360) | 37.1 | plan 11.5 speech | CC BY 4.0 |
| musan_noise | 11.1 (noise part extracted) | noise | CC BY 4.0 + per-folder |
| wham_noise | 18.2 | two-mic noise (tr only) | CC BY-NC 4.0 |
| but_reverbdb | 9.31 | measured RIRs | CC BY 4.0 |
| dns_freesound_001, dns_audioset_001 | 0.992, 5.358 | extra DNS noise | per clip |
| dns_rirs | 0.265 | no scanner yet | per DNS README |
| audioset_csv | 0.104 | labels for the DNS AudioSet speech/music filter | CC BY 4.0 |
| cadre | ~0.38 | eval re-render only; **registration** | NIJ grant terms, cite |
| vehicle_interior | 1.229 | held-out generalisation; needs `unrar` or `7z` | CC BY 4.0 |
| svarah | 1.095 | **HF gate** | CC BY 4.0 |
| indictts | TBD | **licence request** | TBD |

## Disk

`plan` prints the need per dataset and in total, and `fetch` refuses to start a dataset that would not fit. The
estimate is the archive size times 3: the archive, the extracted tree, and the 16 kHz FLAC the scanners write
(inferred upper bound, not measured). Keep `--root` on the box's large volume.

## Tools on the box

- **Downloads.** `aria2c` is used when present (parallel, resumable). Otherwise the tool falls back to
  `curl -C -`, then to Python with Range requests.
- **Extraction.**
  - `lbzip2` speeds up the DNS `.tar.bz2` archives.
  - `zip` (Info-ZIP) is needed only for fsd50k.
  - `unrar` or `7z` is needed only for vehicle_interior.
- **Other sources.** The Kaggle fallback needs the `kaggle` CLI, and HF entries need `huggingface_hub`.
