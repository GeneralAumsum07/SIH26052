import hashlib
from vaani.data import splits, manifests


def test_assign_deterministic_and_covers_all():
    seen = {splits.assign(f"spk{i}") for i in range(2000)}
    assert seen == {"train", "val", "test"}
    assert splits.assign("abc") == splits.assign("abc")


def test_hash_is_stable_across_processes():
    # python's hash() is salted per process; ours must not be
    expected = int(hashlib.sha1(b"x").hexdigest()[:8], 16)
    assert manifests.stable_hash("x") == expected
    assert splits.assign("abc") in {"train", "val", "test"}
