"""Suite-wide setup: every test sees the repo root as its working directory and on sys.path.

Many tests name repo files relatively ("configs/retraining/...", "deploy/r7/...", "data/manifests/..."),
and some do it at import time (skipif conditions, module constants). Changing directory here, when pytest
loads this conftest and before any test module is imported, makes the suite pass from any working
directory instead of only from the repo root.
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# module level, not a fixture: collection-time Path(...).exists() checks run before any fixture would
os.chdir(REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))   # `from scripts import ...` / `import vaani` without an installed package
