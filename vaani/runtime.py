"""Box-shaped runtime settings: how many dataloader workers, and the free GPU-side switches.

Training here is dataloader-bound (measured ~130 ms of the ~144 ms step is CPU: FLAC decode,
RIR convolution and the NLMS/feature front end in `pipeline.run`). So the worker count is the
single most important runtime knob and it has to follow whatever box we rent, not a constant
baked into a config.
"""
import os

import torch

ENV_WORKERS = "VAANI_WORKERS"


def cpu_quota(root="/sys/fs/cgroup"):
    """The container's CFS CPU quota in cores (cgroup v2 cpu.max, else v1 cfs_quota/period); None if unlimited."""
    try:
        q, p = open(f"{root}/cpu.max").read().split()[:2]
        return None if q == "max" else int(q) / int(p)
    except (OSError, ValueError):
        pass
    try:
        q = int(open(f"{root}/cpu/cpu.cfs_quota_us").read()); p = int(open(f"{root}/cpu/cpu.cfs_period_us").read())
        return None if q <= 0 else q / p
    except (OSError, ValueError):
        return None


def cpu_count() -> int:
    """Cores this process may actually use: affinity, capped by the CFS quota (a vast.ai box shows 512 in
    nproc but may be allowed ~246; sizing workers from nproc oversubscribes it 2x)."""
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:                       # Windows
        n = os.cpu_count() or 1
    q = cpu_quota()
    return max(1, min(n, int(q)) if q else n)


def resolve_workers(spec="auto", share: int = 1, reserve: int = 2) -> int:
    """`spec` is an int, or "auto" to size from the box. `share` divides the cores when several
    trainings run concurrently; the shared-loader trainer keeps share=1 because it has one loader.
    $VAANI_WORKERS overrides everything, so a box can be tuned without editing configs."""
    env = os.environ.get(ENV_WORKERS)
    if env:
        return max(0, int(env))
    if spec != "auto" and spec is not None:
        return max(0, int(spec))
    return max(2, (cpu_count() - reserve) // max(1, share))


def worker_init(_worker_id):
    """One thread per worker. Without this, N workers each open a BLAS/OMP pool of N threads and
    the box thrashes: at 64 workers that is thousands of threads fighting for the same cores."""
    torch.set_num_threads(1)


def limit_worker_threads():
    """Set before torch/numpy import in child processes; safe to call again in the parent."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")


def loader_kwargs(num_workers: int, device, prefetch_factor: int = 4) -> dict:
    """DataLoader settings that cost nothing and are wrong to omit on a CUDA box."""
    kw = dict(num_workers=num_workers, persistent_workers=num_workers > 0,
              pin_memory=(getattr(device, "type", str(device)) == "cuda"))
    if num_workers > 0:
        kw.update(prefetch_factor=prefetch_factor, worker_init_fn=worker_init)
    return kw


def tune_backends(device):
    """TF32 and cudnn autotuning. Shapes are fixed here (batch x 257 x frames), so benchmark mode
    picks kernels once and reuses them; it would be the wrong choice under varying shapes."""
    if getattr(device, "type", str(device)) != "cuda":
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True


def describe() -> str:
    """One line for the run log: what the box actually gave us."""
    import shutil
    total, used, free = shutil.disk_usage(".")
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return (f"cores={cpu_count()} workers={resolve_workers()} gpu={gpu} "
            f"disk_free={free / 1e9:.0f}GB")
