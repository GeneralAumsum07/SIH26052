"""Day-1 CUDA gate: cu128 wheel required for Blackwell sm_120.
CPU-only wheel silently installs and wastes days of debugging."""
import torch


def test_cuda_available():
    assert torch.cuda.is_available(), "torch has no CUDA - check the cu128 index in pyproject"


def test_cuda_matmul_runs():
    a = torch.randn(256, 256, device="cuda")
    b = a @ a
    torch.cuda.synchronize()
    assert b.isfinite().all()
