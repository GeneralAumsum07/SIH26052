import hashlib, json, shutil

from scripts.verify_eval_set import verify


def _item(d, i, twin=False):
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{i:04d}.json").write_text(json.dumps({"i": i}))
    (d / f"{i:04d}.mix.wav").write_bytes(b"x"); (d / f"{i:04d}.clean.wav").write_bytes(b"x")
    if twin: (d / f"{i:04d}.twin.mix.wav").write_bytes(b"x")


def test_complete_set_verifies_and_partial_copies_fail(tmp_path):
    root = tmp_path / "test"
    for i in range(3): _item(root / "stationary_0", i)
    for i in range(2): _item(root / "fault_burst_0", i, twin=True)
    h = hashlib.sha1()
    for p in sorted(root.rglob("*.json")): h.update(p.read_bytes())
    expected = h.hexdigest()[:12]
    (root / "EVALSET_HASH").write_text(expected)   # present but never trusted
    assert verify(root, expected) == []
    # a missing bucket changes the recomputed digest even though the copied hash file still reads "expected"
    shutil.rmtree(root / "fault_burst_0")
    assert any("recomputed hash" in p for p in verify(root, expected))
    # a meta whose WAV did not arrive is reported by name
    for i in range(2): _item(root / "fault_burst_0", i, twin=True)
    (root / "stationary_0/0001.mix.wav").unlink()
    assert any(p.endswith("0001.mix.wav") for p in verify(root, expected))
    # a bucket that lost some twins is flagged
    (root / "stationary_0/0001.mix.wav").write_bytes(b"x"); (root / "fault_burst_0/0001.twin.mix.wav").unlink()
    assert any("1/2 twin" in p for p in verify(root, expected))
