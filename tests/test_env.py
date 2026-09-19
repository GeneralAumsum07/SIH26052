"""Day-1 CUDA gate: cu128 wheel required for Blackwell sm_120.
CPU-only wheel silently installs and wastes days of debugging."""
import os

import pytest
import torch

# opt-in on the training box (VAANI_REQUIRE_CUDA=1); a CPU-only checkout (CI, judges' laptops) must still pass
pytestmark = pytest.mark.skipif(os.environ.get("VAANI_REQUIRE_CUDA") != "1" and not torch.cuda.is_available(),
                                reason="no CUDA and VAANI_REQUIRE_CUDA unset")


def test_cuda_available():
    assert torch.cuda.is_available(), "torch has no CUDA - check the cu128 index in pyproject"


def test_cuda_matmul_runs():
    a = torch.randn(256, 256, device="cuda")
    b = a @ a
    torch.cuda.synchronize()
    assert b.isfinite().all()
