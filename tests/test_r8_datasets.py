"""scripts/r8_datasets.py on local fixtures only: a Range-capable http.server, file:// URLs, a fake kaggle on PATH, a
fake huggingface_hub and a canned MDC response. No network, no real corpus."""
import hashlib, importlib.util, io, json, os, shutil, sys, tarfile, threading, types, zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("r8_datasets", REPO / "scripts/r8_datasets.py")
R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)


# ---------------------------------------------------------------------------------------------------------------
# the real registry against the recipes
# ---------------------------------------------------------------------------------------------------------------

def _r8_configs():
    import glob
    out = {}
    for c in glob.glob(str(REPO / "configs/retraining/r8_*.yaml")) + glob.glob(str(REPO / "configs/retraining/r8_ablations/*.yaml")):
        d = yaml.safe_load(open(c, encoding="utf-8"))
        if "data" in d:   # the refiner config inherits its data block from a base run
            out[Path(c).stem] = d
    return out


def test_registry_schema_valid():
    reg = R.load_registry()
    assert R.validate(reg) == []
    assert reg["datasets"] and reg["optional"] and reg["hosts"]["zenodo.org"]["connections"] == 1


def test_registry_covers_every_r8_manifest_and_the_gates():
    reg = R.load_registry()
    have = {m for v in reg["datasets"].values() for m in R.manifest_list(v)}
    want = {m for c in _r8_configs().values() for m in c["data"]["manifests"]}
    assert want and want <= have, sorted(want - have)
    sys.path.insert(0, str(REPO / "scripts"))
    import data_gates as g
    g1 = {Path(m).name for m in list(g.SPEECH_MANIFESTS) + list(g.V2_NOISE)}   # the gate lists bare file names
    assert g1 <= {Path(m).name for m in have}, sorted(g1 - {Path(m).name for m in have})


def test_registry_needed_by_names_real_configs_and_scans_write_listed_manifests():
    reg = R.load_registry()
    stems = set(_r8_configs())
    for k, v in R.entries(reg).items():
        v = v[1]
        for n in v["needed_by"]:
            assert ":" in n or n in stems, f"{k}: needed_by {n} is not an r8 config"
        # every manifest a config reads is either scanned here or installed by the mirror
        scanned = {m for _, m, _ in R.scan_specs(v)} | set(v.get("mirror_manifests") or [])
        assert set(R.manifest_list(v)) <= scanned, k


def test_preflight_fetch_order_ignores_optional():
    sys.path.insert(0, str(REPO / "scripts"))
    import r8_preflight as p
    first, rest = p.fetch_order(REPO, extra_manifests=p.g1_manifests())
    reg = R.load_registry()
    assert set(first) | set(rest) == set(reg["datasets"])
    assert {"librispeech", "ears", "cv_hi", "esc50", "mad", "gunshots", "demand", "dns_audioset_000"} <= set(first)


def test_public_guide_has_no_secret_values():
    md = (REPO / "configs/data/R8_DATASETS.md").read_text(encoding="utf-8")
    for bad in ("hf_", "huggingface.co/datasets/", "KAGGLE_KEY=" + "x"):
        assert bad not in md.replace("huggingface.co (", "")   # names the site, never a private repo id or a token
    reg = R.load_registry()
    for k, (_, v) in R.entries(reg).items():
        for c in (v.get("credentials") or []) + (v.get("optional_credentials") or []):
            assert c.isupper() and "=" not in c


def test_validate_catches_bad_entries():
    reg = {"hosts": {}, "optional": {}, "datasets": {"x": {
        "access": "sometimes", "credentials": ["lower"], "files": [{"name": "a", "md5": "1", "sha256": "2"}],
        "dest": "elsewhere/x", "extract": "cab", "scanner": "vaani.data.sources.scan_nope",
        "manifest": "data/manifests/x.parquet", "needed_by": [], "licence": "?"}}}
    errs = "\n".join(R.validate(reg))
    for s in ("access 'sometimes'", "extract 'cab'", "dest must be", "credential 'lower'", "both md5 and sha256",
              "has no url", "scan_nope does not exist"):
        assert s in errs


# ---------------------------------------------------------------------------------------------------------------
# fixtures: a local server and tiny archives
# ---------------------------------------------------------------------------------------------------------------

class Server:
    """Serves bytes by path with Range, and scripted faults: 429 once (Retry-After 0), 403 always."""

    def __init__(self):
        self.files, self.once429, self.deny, self.log = {}, set(), set(), []
        srv = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                path = self.path.split("?")[0]
                rng = self.headers.get("Range")
                srv.log.append((path, rng))
                if path in srv.deny:
                    self.send_response(403); self.end_headers(); return
                if path in srv.once429:
                    srv.once429.discard(path)
                    self.send_response(429); self.send_header("Retry-After", "0"); self.end_headers(); return
                body = srv.files.get(path)
                if body is None:
                    self.send_response(404); self.end_headers(); return
                start, end = 0, len(body) - 1
                if rng and rng.startswith("bytes="):
                    a, _, b = rng[6:].partition("-")
                    start, end = int(a), min(int(b) if b else len(body) - 1, len(body) - 1)
                    if start >= len(body):
                        self.send_response(416); self.end_headers(); return
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{end}/{len(body)}")
                else:
                    self.send_response(200)
                self.send_header("Content-Length", str(end + 1 - start)); self.send_header("Accept-Ranges", "bytes")
                self.end_headers(); self.wfile.write(body[start:end + 1])

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def add(self, path, body):
        self.files[path] = body
        return self.url + path


@pytest.fixture
def server():
    s = Server()
    yield s
    s.httpd.shutdown()


def _wav_bytes(n=16000, sr=44100, seed=0, tone=False):
    x = np.random.default_rng(seed).standard_normal(n).astype(np.float32) * 0.1
    if tone:   # loud half + near-silent half: a high energy-SNR "speech" clip
        x[: n // 2] = 0.5 * np.sin(2 * np.pi * 220 * np.arange(n // 2) / sr); x[n // 2:] *= 0.001
    b = io.BytesIO(); sf.write(b, x, sr, format="WAV", subtype="PCM_16"); return b.getvalue()


def _zip(entries):
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        for k, v in entries.items():
            z.writestr(k, v)
    return b.getvalue()


def _tgz(entries):
    b = io.BytesIO()
    with tarfile.open(fileobj=b, mode="w:gz") as t:
        for k, v in entries.items():
            v = v.encode() if isinstance(v, str) else v
            ti = tarfile.TarInfo(k); ti.size = len(v); t.addfile(ti, io.BytesIO(v))
    return b.getvalue()


def _mat(name, seed):
    from scipy.io import savemat
    x = (np.random.default_rng(seed).standard_normal(19980) * 3000).astype(np.int16)
    b = io.BytesIO(); savemat(b, {name: x[:, None]}); return b.getvalue(), x


def _md5(b):
    return hashlib.md5(b).hexdigest()


def _esc50_zip():
    meta = "filename,fold,target,category,esc10,src_file,take\n1-100-A-0.wav,1,0,dog,True,100,A\n1-100-B-0.wav,1,0,dog,True,100,B\n"
    return _zip({"ESC-50-abc/meta/esc50.csv": meta, "ESC-50-abc/audio/1-100-A-0.wav": _wav_bytes(seed=1),
                 "ESC-50-abc/audio/1-100-B-0.wav": _wav_bytes(seed=2)})


def _mad_zip():
    csv = "path,label,youtube url\ntraining/7/0.wav,4,https://youtu.be/abcdefghijk\n"
    return _zip({"MAD_dataset/training.csv": csv, "MAD_dataset/test.csv": "path,label,youtube url\n",
                 "MAD_dataset/training/7/0.wav": _wav_bytes(seed=3)})


def _entry(name, files, extract, scanner=None, manifest=None, **kw):
    e = {"access": "direct", "credentials": [], "urls": [], "files": files, "size_gb": 0.0,
         "dest": f"data/raw/{name}_src", "extract": extract, "scanner": scanner,
         "manifest": manifest, "needed_by": [], "licence": "test", "notes": ""}
    e.update(kw)
    return e


def _write_reg(tmp, datasets, optional=None):
    p = tmp / "reg.yaml"
    p.write_text(yaml.safe_dump({"hosts": {"default": {"max_files": 2, "connections": 2}},
                                 "datasets": datasets, "optional": optional or {}}), encoding="utf-8")
    return p


def _run(reg, root, *a):
    return R.main([*a, "--registry", str(reg), "--root", str(root), "--downloader", "python", "--retry-wait", "0",
                   "--retries", "2"])


# ---------------------------------------------------------------------------------------------------------------
# fetch -> extract -> scan -> verify end to end
# ---------------------------------------------------------------------------------------------------------------

def test_fetch_scan_verify_http_file_and_mat2wav(tmp_path, server, capsys):
    esc = _esc50_zip()
    mats = {n: _mat(n, i) for i, n in enumerate(("white", "pink"))}
    local = tmp_path / "srv"; local.mkdir()
    (local / "pink.mat").write_bytes(mats["pink"][0])
    reg = _write_reg(tmp_path, {
        "esc50": _entry("esc50", [{"name": "esc.zip", "url": server.add("/esc.zip", esc), "size_bytes": None,
                                   "sha256": None, "laptop_sha256": hashlib.sha256(esc).hexdigest()}],
                        "zip", "vaani.data.sources.scan_esc50", "data/manifests/esc50.parquet", root_find="meta/esc50.csv"),
        "noisex92": _entry("noisex92", [
            {"name": "white.mat", "url": server.add("/white.mat", mats["white"][0]), "size_bytes": len(mats["white"][0]),
             "md5": _md5(mats["white"][0])},
            {"name": "pink.mat", "url": (local / "pink.mat").as_uri(), "size_bytes": len(mats["pink"][0])}],
            "mat2wav", "vaani.data.sources.scan_noisex92", "data/manifests/noisex92.parquet", root=".", mat_rate=19980)})
    root = tmp_path / "data"
    assert _run(reg, root, "fetch", "--parallel", "2") == 0
    d = root / "raw/esc50_src"
    f = json.loads((d / ".fetched").read_text())
    assert f["files"]["esc.zip"]["sha256"] == hashlib.sha256(esc).hexdigest()   # recorded on first fetch
    assert f["files"]["esc.zip"]["laptop_match"] is True
    assert Path(json.loads((d / ".extracted").read_text())["root"]).name == "ESC-50-abc"
    # mat2wav is sample-exact at the stated rate
    y, sr = sf.read(root / "raw/noisex92_src/white.wav", dtype="int16")
    assert sr == 19980 and np.array_equal(y, mats["white"][1])
    assert _run(reg, root, "scan", "--parallel", "1") == 0
    assert _run(reg, root, "verify") == 0
    from vaani.data import manifests
    m = manifests.read(root / "manifests/esc50.parquet")
    assert len(m) == 2 and set(m.group_id) == {"esc50-100"} and m.path.str.contains("/raw/esc50/dog/").all()
    assert len(manifests.read(root / "manifests/noisex92.parquet")) == 2
    # a rerun downloads nothing
    n = len(server.log)
    assert _run(reg, root, "fetch") == 0 and len(server.log) == n
    out = capsys.readouterr().out
    assert "OK   esc50" in out and "OK   noisex92" in out


def test_resume_after_partial_download(tmp_path, server):
    body = os.urandom(300_000)
    reg = _write_reg(tmp_path, {"blob": _entry("blob", [{"name": "b.bin", "url": server.add("/b.bin", body),
                                                          "size_bytes": len(body), "md5": _md5(body)}], "none")})
    root = tmp_path / "data"; d = root / "raw/blob_src"; d.mkdir(parents=True)
    (d / "b.bin.part").write_bytes(body[:123_456])   # an interrupted earlier run
    assert _run(reg, root, "fetch") == 0
    assert (d / "b.bin").read_bytes() == body and not (d / "b.bin.part").exists()
    assert server.log == [("/b.bin", "bytes=123456-")]
    assert _run(reg, root, "verify") == 0


@pytest.mark.parametrize("tool", ["curl", "aria2c"])
def test_resume_with_external_downloader(tmp_path, server, tool):
    if not shutil.which(tool):
        pytest.skip(f"{tool} not on PATH")
    body = os.urandom(200_000)
    reg = _write_reg(tmp_path, {"blob": _entry("blob", [{"name": "b.bin", "url": server.add("/b.bin", body),
                                                          "size_bytes": len(body), "md5": _md5(body)}], "none")})
    root = tmp_path / "data"; d = root / "raw/blob_src"; d.mkdir(parents=True)
    (d / "b.bin.part").write_bytes(body[:50_000])
    rc = R.main(["fetch", "--registry", str(reg), "--root", str(root), "--downloader", tool, "--retry-wait", "0"])
    assert rc == 0 and (d / "b.bin").read_bytes() == body
    # curl resumes at the part length; aria2c at the piece boundary below it (49152 here), after a probe GET
    assert any(r and 0 < int(r[6:].split("-")[0]) <= 50_000 for _, r in server.log)


def test_429_retry_after_is_honoured(tmp_path, server):
    body = b"x" * 1000
    url = server.add("/r.bin", body); server.once429.add("/r.bin")
    reg = _write_reg(tmp_path, {"blob": _entry("blob", [{"name": "r.bin", "url": url, "size_bytes": 1000}], "none")})
    assert _run(reg, tmp_path / "data", "fetch") == 0
    assert [p for p, _ in server.log] == ["/r.bin", "/r.bin"]


def test_errors_are_per_dataset_and_mirrors_are_tried(tmp_path, server, capsys):
    good, bad = b"g" * 100, b"b" * 100
    reg = _write_reg(tmp_path, {
        "good": _entry("good", [{"name": "g", "url": server.url + "/404", "mirrors": [server.add("/g", good)],
                                 "size_bytes": 100}], "none"),
        "wrongsize": _entry("wrongsize", [{"name": "b", "url": server.add("/b", bad), "size_bytes": 99}], "none"),
        "wrongmd5": _entry("wrongmd5", [{"name": "c", "url": server.add("/c", bad), "md5": "0" * 32}], "none"),
        "gone": _entry("gone", [{"name": "x", "url": server.url + "/nothing"}], "none")})
    root = tmp_path / "data"
    assert _run(reg, root, "fetch") == 1
    out = capsys.readouterr().out
    assert "OK   good" in out
    assert "FAIL wrongsize: b: 100 B, registry says 99 B" in out
    assert "FAIL wrongmd5: c: md5" in out and "FAIL gone: HTTP 404" in out
    assert _run(reg, root, "verify", "--only", "good") == 0
    assert _run(reg, root, "verify") == 1


def test_missing_credentials_block_only_that_dataset_and_never_print_values(tmp_path, server, monkeypatch, capsys):
    cv = _tgz({"cv-corpus-27.0/hi/validated.tsv": "client_id\tpath\nspk0000000000000001\ta.wav\n",
               "cv-corpus-27.0/hi/clips/a.wav": _wav_bytes(16000, 48000, tone=True)})
    dl = server.add("/presigned/cv.tgz", cv)
    canned = tmp_path / "mdc.json"; canned.write_text(json.dumps({"downloadUrl": dl + "?X-Sig=secret"}))
    cvh = _entry("cv_hi", [{"name": "cv.tgz", "size_bytes": len(cv), "sha256": hashlib.sha256(cv).hexdigest()}], "tar",
                 "vaani.data.sources.scan_commonvoice_hi", "data/manifests/cv_hi.parquet",
                 access="login", credentials=["MDC_API_KEY"], fetch="mdc", mdc_dataset="abc",
                 mdc_api=canned.as_uri(), root_find="validated.tsv", scan_args={"min_snr_db": 10})
    other = _entry("other", [{"name": "o", "url": server.add("/o", b"o"), "size_bytes": 1}], "none")
    reg = _write_reg(tmp_path, {"cv_hi": cvh, "other": other})
    root = tmp_path / "data"
    monkeypatch.delenv("MDC_API_KEY", raising=False)
    assert _run(reg, root, "plan") == 1
    assert "MISSING MDC_API_KEY" in capsys.readouterr().out
    assert _run(reg, root, "fetch") == 1
    out = capsys.readouterr().out
    assert "FAIL cv_hi: missing credentials MDC_API_KEY" in out and "OK   other" in out
    assert not any("/presigned" in p for p, _ in server.log)   # no request went out for it
    monkeypatch.setenv("MDC_API_KEY", "sekrit-value-123")
    assert _run(reg, root, "fetch") == 0 and _run(reg, root, "scan", "--parallel", "1") == 0
    assert _run(reg, root, "verify") == 0
    out = capsys.readouterr().out
    assert "sekrit-value-123" not in out and "X-Sig" not in out
    for m in (root / "raw/cv_hi_src").glob(".*"):
        assert "sekrit" not in m.read_text() and "X-Sig" not in m.read_text()   # nothing secret lands on disk


def _fake_kaggle(tmp, zbytes, monkeypatch):
    bindir = tmp / "bin"; bindir.mkdir()
    src = tmp / "mad_fixture.zip"; src.write_bytes(zbytes)
    py = bindir / "fake_kaggle.py"
    py.write_text(
        "import os, shutil, sys\n"
        "a = sys.argv[1:]\n"
        "assert a[:2] == ['datasets', 'download'], a\n"
        "assert os.environ.get('KAGGLE_KEY') or os.environ.get('KAGGLE_API_TOKEN')\n"
        "p = a[a.index('-p') + 1]\n"
        f"shutil.copy(r'{src}', os.path.join(p, a[2].split('/')[1] + '.zip'))\n")
    if os.name == "nt":
        (bindir / "kaggle.cmd").write_text(f'@"{sys.executable}" "{py}" %*\r\n')
    else:
        k = bindir / "kaggle"; k.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{py}" "$@"\n'); k.chmod(0o755)
    monkeypatch.setenv("PATH", str(bindir) + os.pathsep + os.environ["PATH"])


def test_kaggle_fallback_links_and_mirror_manifest(tmp_path, server, monkeypatch, capsys):
    z = _mad_zip()
    url = server.add("/api/v1/datasets/download/o/mad", z); server.deny.add("/api/v1/datasets/download/o/mad")
    mad = _entry("mad", [{"name": "mad.zip", "url": url, "size_bytes": len(z)}], "zip", "vaani.data.sources.scan_mad",
                 ["data/manifests/mad.parquet", "data/manifests/mad_v2.parquet"], kaggle="o/mad",
                 optional_credentials=["KAGGLE_USERNAME", "KAGGLE_KEY"], root_find="training.csv",
                 links={"data/download/mad/MAD_dataset": "."},
                 scans=[{"scanner": "vaani.data.sources.scan_mad", "manifest": "data/manifests/mad.parquet"}],
                 mirror_manifests=["data/manifests/mad_v2.parquet"])
    reg = _write_reg(tmp_path, {"mad": mad})
    root = tmp_path / "data"
    for k in ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN"):
        monkeypatch.delenv(k, raising=False)
    assert _run(reg, root, "fetch") == 1
    assert "neither KAGGLE_API_TOKEN nor KAGGLE_USERNAME+KAGGLE_KEY is set" in capsys.readouterr().out
    _fake_kaggle(tmp_path, z, monkeypatch)
    monkeypatch.setenv("KAGGLE_USERNAME", "u"); monkeypatch.setenv("KAGGLE_KEY", "k")
    assert _run(reg, root, "fetch") == 0
    assert (root / "download/mad/MAD_dataset/training.csv").exists()   # the G4/score_real default MAD root
    assert _run(reg, root, "scan", "--parallel", "1") == 0
    assert _run(reg, root, "verify") == 0   # mad_v2 comes from the mirror later: noted, not a failure
    assert "mad_v2.parquet not installed yet" in capsys.readouterr().out
    from vaani.data import manifests
    m = manifests.read(root / "manifests/mad.parquet")
    assert list(m.source_id) == ["mad:vehicle/7_0"] and list(m.group_id) == ["mad-7"]
    bad = m.copy(); bad["path"] = "data/raw/mad/nowhere.flac"
    manifests.write(bad.to_dict("records"), root / "manifests/mad_v2.parquet")
    assert _run(reg, root, "verify") == 1   # once installed, its paths must resolve


def test_hf_snapshot_into_cache(tmp_path, monkeypatch):
    cache = tmp_path / "hfcache/snapshots/rev123"; cache.mkdir(parents=True)
    model = b"model-bytes" * 100
    calls = []

    def snapshot_download(**kw):
        calls.append(kw)
        (cache / "model.bin").write_bytes(model); (cache / "config.json").write_text("{}")
        return str(cache)
    monkeypatch.setitem(sys.modules, "huggingface_hub", types.SimpleNamespace(snapshot_download=snapshot_download))
    monkeypatch.setenv("HF_TOKEN", "tok")
    e = _entry("fw", [{"name": "model.bin", "size_bytes": len(model), "sha256": hashlib.sha256(model).hexdigest()},
                      {"name": "config.json", "size_bytes": 2}], "none", fetch="hf", hf_repo="o/fw", hf_cache=True,
               hf_revision="rev123", kind="model")
    reg = _write_reg(tmp_path, {"fw": e})
    root = tmp_path / "data"
    assert _run(reg, root, "fetch") == 0 and _run(reg, root, "verify") == 0
    assert calls[0]["repo_id"] == "o/fw" and calls[0]["token"] == "tok" and "local_dir" not in calls[0]
    assert "tok" not in (root / "raw/fw_src/.fetched").read_text()


def test_extract_members_into_and_missing_tools(tmp_path, server, monkeypatch, capsys):
    tg = _tgz({"musan/noise/free/a.wav": _wav_bytes(seed=5), "musan/speech/x/b.wav": _wav_bytes(seed=6)})
    z1 = _zip({"Glock_Zoom/ZM_1A_S1.wav": _wav_bytes(seed=7)})
    reg = _write_reg(tmp_path, {
        "musan": _entry("musan", [{"name": "musan.tgz", "url": server.add("/m.tgz", tg)}], "tar", members=["musan/noise"],
                        root="musan"),
        "cadre": _entry("cadre", [{"name": "Glock_zoom.zip", "url": server.add("/g.zip", z1)}], "zip",
                        extract_into="{stem}", root="."),
        "visc": _entry("visc", [{"name": "v.rar", "url": server.add("/v.rar", b"Rar!")}], "rar")})
    root = tmp_path / "data"
    monkeypatch.setattr(R.shutil, "which", lambda n: None)   # no aria2c/curl/unrar/7z/zip
    assert _run(reg, root, "fetch") == 1
    out = capsys.readouterr().out
    assert "FAIL visc: a .rar archive needs unrar or 7z" in out
    assert (root / "raw/musan_src/musan/noise/free/a.wav").exists()
    assert not (root / "raw/musan_src/musan/speech").exists()
    assert (root / "raw/cadre_src/Glock_zoom/Glock_Zoom/ZM_1A_S1.wav").exists()


def test_disk_guard_and_unknown_names(tmp_path, server, monkeypatch, capsys):
    reg = _write_reg(tmp_path, {"big": _entry("big", [{"name": "b", "url": server.add("/b", b"1"), "size_bytes": 10**15}],
                                              "zip")})
    root = tmp_path / "data"
    assert _run(reg, root, "fetch") == 1
    assert "needs" in capsys.readouterr().out and not server.log
    with pytest.raises(SystemExit):
        _run(reg, root, "fetch", "--only", "nope")


def test_list_and_dry_run_touch_nothing(tmp_path, server, capsys):
    reg = _write_reg(tmp_path, {"a": _entry("a", [{"name": "a", "url": server.add("/a", b"a")}], "none")},
                     {"opt": _entry("opt", [{"name": "o", "url": server.add("/o", b"o")}], "none"),
                      "req": _entry("req", [], "none", access="request")})
    root = tmp_path / "data"
    assert _run(reg, root, "list") == 0
    out = capsys.readouterr().out
    assert "datasets  a" in out and "optional  opt" in out and "optional  req" in out
    assert _run(reg, root, "fetch", "--dry-run", "--all") == 0
    assert not server.log and not root.exists()
    out = capsys.readouterr().out
    assert "would fetch o" in out and "req" not in out.split("OK")[-1]   # request-only entries are not in --all
