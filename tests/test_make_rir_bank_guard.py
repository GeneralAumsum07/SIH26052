"""scripts/make_rir_bank.py must keep its work behind a __main__ guard.

vaani.data.rirs.build_bank simulates in a `spawn` pool. Unlike fork, spawn re-imports the
script that started the pool in every child process, so anything at module level runs again
in each one. When build_bank() sat at module level, each child re-entered it and opened a
pool of its own; on a 128-core box the recursion pinned load at 117, wrote a 111 MB log of
bootstrap tracebacks, and never produced a bank. The fetch-by-hash path masked it until a
box ran without RIR_BANK_URL set and fell through to the builder.

This is a static check on purpose: reproducing the fault means actually starting the bomb.
"""
import ast
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_rir_bank.py"
# anything that must not run at import time in a spawned child
UNSAFE = {"build_bank", "parse_args"}


def _is_main_guard(node: ast.stmt) -> bool:
    return (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__")


def _called_name(call: ast.Call) -> str:
    f = call.func
    return f.id if isinstance(f, ast.Name) else getattr(f, "attr", "")


def test_make_rir_bank_entrypoint_is_guarded():
    tree = ast.parse(SCRIPT.read_text())
    guards = [n for n in tree.body if _is_main_guard(n)]
    assert guards, f"{SCRIPT.name} has no `if __name__ == \'__main__\'` guard"

    guarded = {id(n) for g in guards for n in ast.walk(g)}
    bodies = [b for n in tree.body if isinstance(n, ast.FunctionDef) for b in ast.walk(n)]
    in_function = {id(n) for n in bodies}

    offenders = [
        _called_name(n) for n in ast.walk(tree)
        if isinstance(n, ast.Call) and _called_name(n) in UNSAFE
        and id(n) not in guarded and id(n) not in in_function
    ]
    assert not offenders, (
        f"{SCRIPT.name} calls {sorted(set(offenders))} at import time; spawn re-imports this "
        "module in every pool child, so each one would start another pool"
    )
