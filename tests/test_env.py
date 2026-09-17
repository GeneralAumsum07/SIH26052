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
