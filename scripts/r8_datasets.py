"""r8 datasets straight onto the box: fetch, verify, extract and scan every corpus in configs/data/r8_datasets.yaml.

usage (repo root):
    python scripts/r8_datasets.py list                         # the access table (datasets: and optional:)
    python scripts/r8_datasets.py plan [--only a,b] [--all]    # what would be fetched, sizes, missing credentials, disk
    python scripts/r8_datasets.py fetch [--only a,b] [--parallel N] [--dry-run]   # download + verify + extract
    python scripts/r8_datasets.py scan [--only a,b]            # run the scanners -> data/manifests/*.parquet
    python scripts/r8_datasets.py verify [--only a,b]          # markers, sizes/checksums, manifests and their paths

Selection: no --only = every entry under `datasets:` (what the r8 configs and the G1/G4 gates read); --all adds
`optional:`; --only takes names from either. --root data maps dest data/raw/x -> <root>/raw/x (manifests likewise).
Exit 0 only if everything asked for is present and verified; one dataset failing never stops the others.

Markers: <dest>/.fetched (JSON: size, sha256, md5 per file) and <dest>/.extracted (JSON: the scanner root) and
<dest>/.scanned (rows per manifest). A rerun skips finished steps and resumes partial downloads (<name>.part).
Credentials come only from environment variables and are never printed or written; presigned URLs are not recorded.
Downloads: aria2c (-x/-s, --continue) when on PATH, else curl -C -, else Python with Range requests. Per-host limits
come from the registry's `hosts:` block (zenodo.org: 2 files, 1 connection each); HTTP 429 honours Retry-After.
"""
import argparse, hashlib, importlib, importlib.util, json, os, shutil, subprocess, sys, threading, time
import urllib.error, urllib.parse, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
REGISTRY = REPO / "configs/data/r8_datasets.yaml"
GUIDE = "configs/data/R8_DATASETS.md"
EXTRACTS = {"none", "tar", "zip", "zip_split", "rar", "mat2wav"}
ACCESS = {"direct", "login", "request", "manual"}
FETCHES = {"http", "mdc", "hf"}
CHUNK = 1 << 20
UA = "vaani-r8-datasets/1 (+research download; resumable)"


class DatasetError(Exception):
    """One dataset's failure: reported and counted, never raised past that dataset."""


# ---------------------------------------------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------------------------------------------

def load_registry(path=REGISTRY):
    reg = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    reg.setdefault("hosts", {}); reg.setdefault("datasets", {}); reg.setdefault("optional", {})
    return reg


def entries(reg):
    """name -> (section, entry) over datasets: then optional:."""
    out = {}
    for sec in ("datasets", "optional"):
        for k, v in (reg.get(sec) or {}).items():
            out[k] = (sec, v)
    return out


def manifest_list(v):
    m = v.get("manifest") or []
    return [m] if isinstance(m, str) else list(m)


def scan_specs(v):
    """[(scanner dotted path, manifest path, scan_args)]: `scans` when one download feeds several manifests."""
    if v.get("scans"):
        return [(s["scanner"], s["manifest"], s.get("scan_args") or v.get("scan_args") or {}) for s in v["scans"]]
    if v.get("scanner"):
        mans = manifest_list(v)
        return [(v["scanner"], mans[0], v.get("scan_args") or {})] if mans else []
    return []


def validate(reg):
    """Schema problems as strings (empty = valid). Also checks every scanner resolves in vaani.data.sources."""
    errs = []
    names = entries(reg)
    for k, (sec, v) in names.items():
        p = f"{sec}.{k}"
        for f in ("access", "credentials", "files", "dest", "extract", "needed_by", "licence"):
            if f not in v:
                errs.append(f"{p}: missing {f}")
        if v.get("access") not in ACCESS:
            errs.append(f"{p}: access {v.get('access')!r} not in {sorted(ACCESS)}")
        if v.get("extract") not in EXTRACTS:
            errs.append(f"{p}: extract {v.get('extract')!r} not in {sorted(EXTRACTS)}")
        if v.get("fetch", "http") not in FETCHES:
            errs.append(f"{p}: fetch {v.get('fetch')!r} not in {sorted(FETCHES)}")
        if not str(v.get("dest", "")).startswith("data/raw/"):
            errs.append(f"{p}: dest must be data/raw/<dir>")
        for c in (v.get("credentials") or []) + (v.get("optional_credentials") or []):
            if not str(c).isupper():
                errs.append(f"{p}: credential {c!r} must be an ENV_VAR name")
        for f in v.get("files") or []:
            if "name" not in f:
                errs.append(f"{p}: a file has no name")
            if v.get("fetch", "http") == "http" and not f.get("url") and not v.get("kaggle"):
                errs.append(f"{p}: {f.get('name')} has no url")
            if f.get("md5") and f.get("sha256"):
                errs.append(f"{p}: {f['name']} carries both md5 and sha256")
        if v.get("access") in ("direct", "login") and not v.get("files"):
            errs.append(f"{p}: access {v['access']} but no files")
        for m in manifest_list(v):
            if not str(m).startswith("data/manifests/") or not str(m).endswith(".parquet"):
                errs.append(f"{p}: manifest {m} must be data/manifests/<file>.parquet")
        for scanner, man, _ in scan_specs(v):
            mod, _, fn = scanner.rpartition(".")
            if mod != "vaani.data.sources":
                errs.append(f"{p}: scanner {scanner} is not a vaani.data.sources function")
            elif not hasattr(importlib.import_module(mod), fn):
                errs.append(f"{p}: scanner {scanner} does not exist")
            if man not in manifest_list(v):
                errs.append(f"{p}: scans manifest {man} not listed in manifest")
        if v.get("audioset_filter") and "audioset_csv" not in names:
            errs.append(f"{p}: audioset_filter needs an audioset_csv entry")
    return errs


# ---------------------------------------------------------------------------------------------------------------
# paths
# ---------------------------------------------------------------------------------------------------------------

class Paths:
    """Registry paths are written against the default root `data`; --root moves all of them together."""

    def __init__(self, root):
        self.root = Path(root)

    def map(self, p):
        parts = Path(p).parts
        return self.root.joinpath(*parts[1:]) if parts and parts[0] == "data" else Path(p)

    def dest(self, v):
        return self.map(v["dest"])

    @property
    def out(self):   # scanners write <out>/<corpus>/...; with root `data` the manifest paths match the laptop's
        return self.root / "raw"


def read_json(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def write_json(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True), encoding="utf-8")
    os.replace(tmp, p)   # atomic: a marker is never half-written


def hashes(path):
    md5, sha = hashlib.md5(), hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(CHUNK), b""):
            md5.update(b); sha.update(b)
    return md5.hexdigest(), sha.hexdigest()


def gb(n):
    return f"{n / 1e9:.3f}" if n else "?"


# ---------------------------------------------------------------------------------------------------------------
# downloads
# ---------------------------------------------------------------------------------------------------------------

class Hosts:
    """Per-host semaphores (max files in flight) and per-file connection counts from the registry's hosts: block."""

    def __init__(self, table):
        self.table = table or {}
        self.default = self.table.get("default") or {"max_files": 4, "connections": 8}
        self._sem, self._lock = {}, threading.Lock()

    def key(self, url):
        h = (urllib.parse.urlparse(url).hostname or "").lower()
        for k in self.table:
            if k != "default" and (h == k or h.endswith("." + k)):
                return k
        return h

    def conf(self, url):
        return {**self.default, **(self.table.get(self.key(url)) or {})}

    def sem(self, url):
        k = self.key(url)
        with self._lock:
            if k not in self._sem:
                self._sem[k] = threading.BoundedSemaphore(int(self.conf(url).get("max_files") or 1))
            return self._sem[k]


def pick_downloader(choice="auto"):
    if choice != "auto":
        return choice
    if shutil.which("aria2c"):
        return "aria2c"
    return "curl" if shutil.which("curl") else "python"


def _python_get(url, part, headers, conns, retries, wait, log):
    """Resumable GET into `part` with Range; a server that ignores Range (200) restarts the file from zero."""
    for attempt in range(retries + 1):
        have = part.stat().st_size if part.exists() else 0
        h = {"User-Agent": UA, **headers}
        if have:
            h["Range"] = f"bytes={have}-"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=120) as r:
                mode = "ab" if have and r.status == 206 else "wb"
                with open(part, mode) as fh:
                    shutil.copyfileobj(r, fh, CHUNK)
            return
        except urllib.error.HTTPError as e:
            if e.code == 416 and have:   # already complete: nothing left to range
                return
            if e.code in (429, 500, 502, 503, 504) and attempt < retries:
                ra = e.headers.get("Retry-After") if e.headers else None
                delay = float(ra) if ra and ra.isdigit() else wait * (2 ** attempt)
                log(f"HTTP {e.code}; retrying in {delay:.0f} s")
                time.sleep(delay); continue
            raise DatasetError(f"HTTP {e.code} from {redact(url)}") from None
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            if attempt < retries:
                log(f"{type(e).__name__}: {e}; retrying in {wait * (2 ** attempt):.0f} s")
                time.sleep(wait * (2 ** attempt)); continue
            raise DatasetError(f"{redact(url)}: {e}") from None


def _file_url_get(url, part):
    """file:// (tests, local staging): copy with the same resume rule as HTTP."""
    src = Path(urllib.request.url2pathname(urllib.parse.urlparse(url).path))
    if not src.exists():
        raise DatasetError(f"{url}: no such file")
    have = part.stat().st_size if part.exists() else 0
    with open(src, "rb") as fi, open(part, "ab" if have else "wb") as fo:
        fi.seek(have); shutil.copyfileobj(fi, fo, CHUNK)


def redact(url):
    """Scheme, host and path only: query strings carry signatures on presigned URLs."""
    u = urllib.parse.urlparse(url)
    return f"{u.scheme}://{u.netloc}{u.path}" if u.scheme in ("http", "https") else url


def download(url, part, downloader, conns, headers=None, retries=4, wait=15.0, log=print):
    part.parent.mkdir(parents=True, exist_ok=True)
    if url.startswith("file:"):
        return _file_url_get(url, part)
    if downloader == "aria2c" and not headers:
        # --allow-overwrite=true would restart from zero when the .aria2 control file is gone; false resumes the part
        cmd = ["aria2c", "--continue=true", f"-x{conns}", f"-s{conns}", "-k1M", "--file-allocation=none",
               "--auto-file-renaming=false", "--allow-overwrite=false", f"--max-tries={retries + 1}",
               f"--retry-wait={int(wait)}", "--summary-interval=0", "--console-log-level=warn",
               f"--user-agent={UA}", "-d", str(part.parent), "-o", part.name, url]
    elif downloader in ("aria2c", "curl") and shutil.which("curl") and not headers:
        # curl --retry honours Retry-After on 429; -C - resumes from the .part length
        cmd = ["curl", "-fL", "--retry", str(retries), "--retry-delay", str(int(wait)), "-sS", "-A", UA,
               "-C", "-", "-o", str(part), url]
    else:
        return _python_get(url, part, headers or {}, conns, retries, wait, log)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0:
        return
    if cmd[0] == "curl" and r.returncode == 33 and part.exists():   # range not satisfiable: the part is complete
        return
    tail = (r.stderr or r.stdout or "").strip().splitlines()[-1:] or [""]
    raise DatasetError(f"{cmd[0]} exit {r.returncode} for {redact(url)}: {tail[0][:200]}")


# ---------------------------------------------------------------------------------------------------------------
# fetch one dataset
# ---------------------------------------------------------------------------------------------------------------

class Ctx:
    def __init__(self, args, reg):
        self.args, self.reg = args, reg
        self.paths = Paths(args.root)
        self.hosts = Hosts(reg.get("hosts"))
        self.downloader = pick_downloader(args.downloader)
        self.lock = threading.Lock()
        self.reserved = 0
        self.env = os.environ


def check_file(dest, f, path, log):
    """Size and published checksum; the sha256 is always computed and recorded; the laptop hash is advisory."""
    size = path.stat().st_size
    if f.get("size_bytes") and size != f["size_bytes"]:
        raise DatasetError(f"{f['name']}: {size} B, registry says {f['size_bytes']} B")
    md5, sha = hashes(path)
    if f.get("md5") and md5 != f["md5"]:
        raise DatasetError(f"{f['name']}: md5 {md5} != published {f['md5']}")
    if f.get("sha256") and sha != f["sha256"]:
        raise DatasetError(f"{f['name']}: sha256 {sha} != published {f['sha256']}")
    lap = f.get("laptop_sha256")
    rec = {"size": size, "sha256": sha, "md5": md5,
           "checked": "md5" if f.get("md5") else "sha256" if f.get("sha256") else "size" if f.get("size_bytes") else "recorded",
           "laptop_match": None if not lap else sha == lap}
    if lap and sha != lap:
        log(f"{f['name']}: sha256 differs from the laptop copy (advisory; the laptop manifests came from that copy)")
    return rec


def need_credentials(v, env):
    return [c for c in v.get("credentials") or [] if not env.get(c)]


def fetch_http_file(ctx, name, v, f, dest, log):
    final = dest / f["name"]; part = dest / (f["name"] + ".part")
    urls = [f["url"]] + list(f.get("mirrors") or []) if f.get("url") else []
    errors = []
    for u in urls:
        conf = ctx.hosts.conf(u)
        with ctx.hosts.sem(u):
            try:
                log(f"GET {f['name']} <- {redact(u)} ({ctx.downloader}, {conf.get('connections')} conn)")
                download(u, part, ctx.downloader, int(conf.get("connections") or 1),
                         retries=ctx.args.retries, wait=ctx.args.retry_wait, log=log)
            except DatasetError as e:
                errors.append(str(e)); continue
        os.replace(part, final)
        return
    if v.get("kaggle"):
        return fetch_kaggle(ctx, v, f, dest, log, errors)
    raise DatasetError("; ".join(errors) or f"{f['name']}: no url")


def fetch_kaggle(ctx, v, f, dest, log, errors):
    """Fallback after the anonymous API URL: the kaggle CLI with the env credentials (no resume; the zip is small)."""
    env = ctx.env
    if not (env.get("KAGGLE_API_TOKEN") or (env.get("KAGGLE_USERNAME") and env.get("KAGGLE_KEY"))):
        raise DatasetError("; ".join(errors) + " | anonymous Kaggle download failed and neither KAGGLE_API_TOKEN nor "
                           f"KAGGLE_USERNAME+KAGGLE_KEY is set (see {GUIDE})")
    exe = shutil.which("kaggle")
    if not exe:
        raise DatasetError("; ".join(errors) + " | Kaggle credentials are set but the kaggle CLI is not on PATH "
                           "(pip install kaggle)")
    tmp = dest / ".kaggle_tmp"
    shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
    log(f"kaggle datasets download {v['kaggle']}")
    r = subprocess.run([exe, "datasets", "download", v["kaggle"], "-p", str(tmp), "-o", "-q"],
                       capture_output=True, text=True, env=dict(env))
    zips = sorted(tmp.glob("*.zip"))
    if r.returncode != 0 or not zips:
        raise DatasetError(f"kaggle CLI exit {r.returncode}: {(r.stderr or r.stdout or '').strip()[-200:]}")
    os.replace(zips[0], dest / f["name"]); shutil.rmtree(tmp, ignore_errors=True)


def mdc_url(ctx, v):
    """POST the MDC download API with the Bearer key; the presigned downloadUrl lives in memory only."""
    key = ctx.env.get("MDC_API_KEY")
    api = v["mdc_api"].format(id=v["mdc_dataset"])
    if api.startswith("file:"):   # tests: a canned JSON response
        body = Path(urllib.request.url2pathname(urllib.parse.urlparse(api).path)).read_text()
    else:
        req = urllib.request.Request(api, data=b"", method="POST",
                                     headers={"Authorization": f"Bearer {key}", "User-Agent": UA,
                                              "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                body = r.read().decode()
        except urllib.error.HTTPError as e:
            raise DatasetError(f"MDC API HTTP {e.code} for {v['mdc_dataset']} (key rejected, terms not accepted, "
                               "or the 30/day quota is spent)") from None
    try:
        d = json.loads(body)
    except ValueError:
        raise DatasetError("MDC API returned non-JSON") from None
    u = d.get("downloadUrl") or d.get("url")
    if not u:
        raise DatasetError(f"MDC API response has no downloadUrl (keys: {sorted(d)})")
    return u


def fetch_mdc_file(ctx, v, f, dest, log):
    final = dest / f["name"]; part = dest / (f["name"] + ".part")
    for attempt in (1, 2):   # a presigned link can expire mid-run: one fresh POST, then give up
        u = mdc_url(ctx, v)
        conf = ctx.hosts.conf(u)
        with ctx.hosts.sem(u):
            try:
                log(f"GET {f['name']} <- MDC presigned link (attempt {attempt})")
                download(u, part, ctx.downloader, int(conf.get("connections") or 1),
                         retries=ctx.args.retries, wait=ctx.args.retry_wait, log=log)
                os.replace(part, final); return
            except DatasetError as e:
                if attempt == 2:
                    raise DatasetError(f"MDC download failed: {e}") from None
                log(f"{e}; requesting a fresh link")


def fetch_hf(ctx, name, v, dest, log):
    """huggingface_hub snapshot of the listed files. hf_cache: into the HF cache at the repo's current main (what
    vaani.asr.load_whisper resolves), checked against the pinned hashes; otherwise into dest."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise DatasetError("huggingface_hub is not installed (pip install huggingface_hub)") from None
    names = [f["name"] for f in v["files"]]
    kw = dict(repo_id=v["hf_repo"], repo_type=v.get("hf_repo_type", "model"), allow_patterns=names,
              token=ctx.env.get("HF_TOKEN") or None)
    if not v.get("hf_cache"):
        kw.update(local_dir=str(dest), revision=v.get("hf_revision"))
    log(f"huggingface_hub snapshot {v['hf_repo']} ({len(names)} files{', HF cache' if v.get('hf_cache') else ''})")
    try:
        snap = Path(snapshot_download(**kw))
    except Exception as e:   # noqa: BLE001 - gated/401/404 all land here; the message names the cause
        raise DatasetError(f"huggingface_hub: {type(e).__name__}: {str(e)[:200]}") from None
    rec = {}
    for f in v["files"]:
        p = snap / f["name"]
        if not p.exists():
            raise DatasetError(f"{f['name']} missing from the snapshot {snap}")
        rec[f["name"]] = check_file(dest, f, p, log)
    if v.get("hf_cache") and v.get("hf_revision") and snap.name != v["hf_revision"]:
        log(f"HF main is {snap.name}, registry pins {v['hf_revision']}; the pinned hashes matched, so the files agree")
    return rec, str(snap)


def disk_need(ctx, v, dest):
    """Bytes still to land: missing archive bytes plus the extraction estimate (archive x (factor - 1))."""
    if (dest / ".fetched").exists() and (dest / ".extracted").exists():
        return 0
    tot = sum((f.get("size_bytes") or f.get("size_bytes_estimate") or 0) for f in v.get("files") or [])
    have = sum((dest / f["name"]).stat().st_size for f in v.get("files") or [] if (dest / f["name"]).exists())
    have += sum((dest / (f["name"] + ".part")).stat().st_size for f in v.get("files") or []
                if (dest / (f["name"] + ".part")).exists())
    extract = 0 if (dest / ".extracted").exists() or v.get("extract") == "none" else tot * (ctx.args.disk_factor - 1)
    return max(0, tot - have) + int(extract)


def free_bytes(p):
    p = Path(p)
    while not p.exists():
        p = p.parent
    return shutil.disk_usage(p).free


def fetch_dataset(ctx, name, v, file_pool):
    """Download every file, verify, write .fetched, then extract. Raises DatasetError with the reason."""
    dest = ctx.paths.dest(v)
    log = lambda m: print(f"[{name}] {m}", flush=True)
    if v.get("access") in ("request", "manual") and not v.get("files"):
        raise DatasetError(f"access {v['access']}: not fetchable by this tool. {v.get('notes', '')}")
    miss = need_credentials(v, ctx.env)
    if miss:
        raise DatasetError(f"missing credentials {', '.join(miss)} (access {v['access']}; see {GUIDE})")
    if ctx.args.dry_run:
        for f in v["files"]:
            src = "MDC API" if v.get("fetch") == "mdc" else "HF " + v["hf_repo"] if v.get("fetch") == "hf" else redact(f.get("url", ""))
            log(f"would fetch {f['name']} ({gb(f.get('size_bytes') or f.get('size_bytes_estimate'))} GB) <- {src}")
        log(f"would extract ({v['extract']}) into {dest} and scan {[s[1] for s in scan_specs(v)]}")
        return "dry-run"
    need = disk_need(ctx, v, dest)
    with ctx.lock:
        free = free_bytes(dest) - ctx.reserved
        if need > free:
            raise DatasetError(f"needs {gb(need)} GB (archives + x{ctx.args.disk_factor} extraction), "
                               f"{gb(free)} GB free after other downloads' reservations")
        ctx.reserved += need
    try:
        dest.mkdir(parents=True, exist_ok=True)
        done = read_json(dest / ".fetched")
        if not done:
            done = _fetch_files(ctx, name, v, dest, file_pool, log)
        extract_dataset(ctx, name, v, dest, done, log)
        return "ok"
    finally:
        with ctx.lock:
            ctx.reserved -= need


def _fetch_files(ctx, name, v, dest, file_pool, log):
    if v.get("fetch") == "hf":
        rec, snap = fetch_hf(ctx, name, v, dest, log)
        done = {"files": rec, "snapshot": snap, "time": time.strftime("%F %T")}
        write_json(dest / ".fetched", done); return done
    state_p = dest / ".fetch_state.json"
    state = read_json(state_p) or {}

    def one(f):
        final = dest / f["name"]
        prev = state.get(f["name"])
        if final.exists() and prev and prev.get("size") == final.stat().st_size:
            return f["name"], prev
        if not final.exists():
            if v.get("fetch") == "mdc":
                fetch_mdc_file(ctx, v, f, dest, log)
            else:
                fetch_http_file(ctx, name, v, f, dest, log)
        try:
            rec = check_file(dest, f, final, log)
        except DatasetError:
            os.replace(final, final.with_name(final.name + ".bad"))   # kept for inspection; a rerun downloads afresh
            raise
        with ctx.lock:
            state[f["name"]] = rec; write_json(state_p, state)
        log(f"{f['name']}: {rec['size']} B, {rec['checked']} ok")
        return f["name"], rec

    futs = [file_pool.submit(one, f) for f in v["files"]]
    rec, errs = {}, []
    for fu in futs:
        try:
            k, r = fu.result(); rec[k] = r
        except DatasetError as e:
            errs.append(str(e))
    if errs:
        raise DatasetError("; ".join(errs))
    done = {"files": rec, "time": time.strftime("%F %T"), "downloader": ctx.downloader}
    write_json(dest / ".fetched", done)
    state_p.unlink(missing_ok=True)
    return done


# ---------------------------------------------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------------------------------------------

def _tar(archive, dest, members):
    """GNU tar (lbzip2 for .bz2 when present) on POSIX; Python tarfile with the safe `data` filter otherwise."""
    if os.name != "nt" and shutil.which("tar"):
        cmd = ["tar"]
        if str(archive).endswith(".bz2") and shutil.which("lbzip2"):
            cmd += ["-I", "lbzip2"]
        cmd += ["-xf", str(archive), "-C", str(dest), *members]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            raise DatasetError(f"tar exit {r.returncode} on {archive.name}: {r.stderr.strip()[-200:]}")
        return
    import tarfile
    with tarfile.open(archive) as t:
        sel = [m for m in t.getmembers() if not members or any(m.name == p or m.name.startswith(p.rstrip("/") + "/")
                                                              for p in members)]
        t.extractall(dest, members=sel, filter="data")


def _zip(archive, into):
    import zipfile
    with zipfile.ZipFile(archive) as z:
        z.extractall(into)   # zipfile strips absolute and .. components


def _mat2wav(mat, dest, rate):
    """SPIB NOISEX .mat: one int16 variable named after the file, written as PCM_16 at the stated rate, sample-exact."""
    import numpy as np
    import soundfile as sf
    from scipy.io import loadmat
    d = {k: val for k, val in loadmat(mat).items() if not k.startswith("__")}
    if len(d) != 1:
        raise DatasetError(f"{mat.name}: expected one variable, found {sorted(d)}")
    x = np.asarray(next(iter(d.values()))).squeeze()
    if x.dtype.kind != "i" or x.dtype.itemsize != 2:   # SPIB stores big-endian int16 (>i2): same samples
        raise DatasetError(f"{mat.name}: {x.dtype}, expected int16 (not an SPIB original?)")
    x = x.astype(np.int16)
    sf.write(dest / (mat.stem + ".wav"), x, int(rate), subtype="PCM_16")


def find_root(base, pattern, depth=6):
    """The shallowest directory D under base with glob(D/pattern) non-empty (ties broken by name)."""
    level = [Path(base)]
    for _ in range(depth + 1):
        for d in sorted(level):
            if any(d.glob(pattern)):
                return d
        level = [c for d in level for c in d.iterdir() if c.is_dir() and not c.name.startswith(".")]
        if not level:
            break
    return None


def make_link(link, target):
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        if link.resolve() == target.resolve():
            return
        raise DatasetError(f"link {link} exists and points elsewhere")
    try:
        os.symlink(target.resolve(), link, target_is_directory=True)
    except OSError:
        if os.name != "nt":
            raise
        r = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target.resolve())], capture_output=True, text=True)
        if r.returncode:
            raise DatasetError(f"could not link {link} -> {target}: {r.stderr.strip()}") from None


def extract_dataset(ctx, name, v, dest, fetched, log):
    ex = read_json(dest / ".extracted")
    if ex:
        root = Path(ex["root"])
    else:
        kind = v["extract"]
        files = [dest / f["name"] for f in v["files"]]
        if kind == "tar":
            for a in files:
                log(f"untar {a.name}"); _tar(a, dest, list(v.get("members") or []))
        elif kind == "zip":
            for a in files:
                into = dest / v["extract_into"].format(stem=a.stem) if v.get("extract_into") else dest
                log(f"unzip {a.name}"); _zip(a, into)
        elif kind == "zip_split":
            for a in files:
                if a.suffix != ".zip":
                    continue
                if a.with_suffix(".z01").exists():
                    if not shutil.which("zip"):
                        raise DatasetError(f"{a.name} is a split zip: Info-ZIP `zip` is needed to join it (apt install zip)")
                    joined = dest / (a.stem + ".joined.zip")
                    r = subprocess.run(["zip", "-q", "-s", "0", str(a), "--out", str(joined)], capture_output=True, text=True)
                    if r.returncode:
                        raise DatasetError(f"zip -s 0 {a.name}: exit {r.returncode} {r.stderr.strip()[-200:]}")
                    log(f"unzip {joined.name}"); _zip(joined, dest); joined.unlink()
                else:
                    log(f"unzip {a.name}"); _zip(a, dest)
        elif kind == "rar":
            tool = shutil.which("unrar") or shutil.which("7z") or shutil.which("7za")
            if not tool:
                raise DatasetError("a .rar archive needs unrar or 7z on PATH (apt install unrar or p7zip-full)")
            for a in files:
                cmd = [tool, "x", "-o+", "-y", str(a), str(dest) + os.sep] if "unrar" in Path(tool).name else \
                      [tool, "x", "-y", f"-o{dest}", str(a)]
                log(f"unrar {a.name}")
                r = subprocess.run(cmd, capture_output=True, text=True)
                if r.returncode:
                    raise DatasetError(f"{Path(tool).name} exit {r.returncode} on {a.name}: {r.stderr.strip()[-200:]}")
        elif kind == "mat2wav":
            for a in files:
                _mat2wav(a, dest, v.get("mat_rate") or 19980)
            log(f"mat2wav {len(files)} files at {v.get('mat_rate') or 19980} Hz")
        base = Path(fetched["snapshot"]) if fetched.get("snapshot") else dest
        if v.get("root_find"):
            root = find_root(base, v["root_find"])
            if root is None:
                raise DatasetError(f"no directory under {base} holds {v['root_find']} (archive layout changed?)")
        else:
            root = base / (v.get("root") or ".")
        if not root.exists():
            raise DatasetError(f"scanner root {root} does not exist after extraction")
        write_json(dest / ".extracted", {"root": str(root), "time": time.strftime("%F %T")})
        log(f"extracted; scanner root {root}")
    for link, rel in (v.get("links") or {}).items():
        make_link(ctx.paths.map(link), root / rel)
    return root


# ---------------------------------------------------------------------------------------------------------------
# scan
# ---------------------------------------------------------------------------------------------------------------

def audioset_csvs(ctx):
    names = entries(ctx.reg)
    if "audioset_csv" not in names:
        return None
    v = names["audioset_csv"][1]; dest = ctx.paths.dest(v)
    if not (dest / ".fetched").exists():
        return None
    return [dest / f["name"] for f in v["files"] if f["name"].endswith("_segments.csv")]


def _scan_one(scanner, root, out, man, args, csvs):
    """One scanner -> one manifest (module-level so a process pool can run it)."""
    from vaani.data import manifests
    mod, _, fn = scanner.rpartition(".")
    rows = getattr(importlib.import_module(mod), fn)(Path(root), Path(out), **(args or {}))
    if csvs is not None:
        from vaani.data.sources import dns_audioset_filter
        n0 = len(rows)
        rows = dns_audioset_filter(rows, [Path(c) for c in csvs])
        print(f"[audioset filter] {n0} -> {len(rows)} rows ({n0 - len(rows)} Speech/Music clips dropped)")
    if not rows:
        raise DatasetError(f"{fn} returned no rows from {root}")
    Path(man).parent.mkdir(parents=True, exist_ok=True)
    manifests.write(rows, man)
    return len(rows)


def scan_dataset(ctx, name, v, pool=None):
    dest = ctx.paths.dest(v)
    ex = read_json(dest / ".extracted")
    if not ex:
        raise DatasetError("not extracted yet (run fetch first)")
    specs = scan_specs(v)
    if not specs:
        return "no scanner"
    marker = read_json(dest / ".scanned") or {}
    csvs = None
    if v.get("audioset_filter"):
        csvs = audioset_csvs(ctx)
        if not csvs:
            raise DatasetError("audioset_filter is true but audioset_csv is not fetched (fetch --only audioset_csv)")
    done = {}
    for scanner, man, sargs in specs:
        mp = ctx.paths.map(man)
        if not ctx.args.force and marker.get(man) and mp.exists():
            done[man] = marker[man]; continue
        print(f"[{name}] {scanner.rpartition('.')[2]}({ex['root']}) -> {mp}", flush=True)
        job = (scanner, ex["root"], str(ctx.paths.out), str(mp), sargs, [str(c) for c in csvs] if csvs else None)
        done[man] = pool.submit(_scan_one, *job).result() if pool else _scan_one(*job)
    write_json(dest / ".scanned", {**marker, **done})
    return ", ".join(f"{Path(m).name} {n} rows" for m, n in done.items())


# ---------------------------------------------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------------------------------------------

def _paths_ok(mp):
    from vaani.data import manifests
    df = manifests.read(mp)
    missing = [p for p in df.path if not Path(p).exists()]
    return len(df), missing


def verify_dataset(ctx, name, v):
    dest = ctx.paths.dest(v)
    fetched = read_json(dest / ".fetched")
    if not fetched:
        raise DatasetError("no .fetched marker")
    for f in v.get("files") or []:
        r = fetched["files"].get(f["name"])
        if not r:
            raise DatasetError(f"{f['name']} not in .fetched")
        p = (Path(fetched["snapshot"]) if fetched.get("snapshot") else dest) / f["name"]
        if not p.exists() or p.stat().st_size != r["size"]:
            raise DatasetError(f"{f['name']}: missing or resized since fetch")
        for alg in ("md5", "sha256"):
            if f.get(alg) and r.get(alg) != f[alg]:
                raise DatasetError(f"{f['name']}: recorded {alg} does not match the registry")
        if ctx.args.deep:
            md5, sha = hashes(p)
            if sha != r["sha256"]:
                raise DatasetError(f"{f['name']}: sha256 changed since fetch")
    ex = read_json(dest / ".extracted")
    if not ex or not Path(ex["root"]).exists():
        raise DatasetError("not extracted (no .extracted marker or its root is gone)")
    notes = []
    for _, man, _ in scan_specs(v):
        mp = ctx.paths.map(man)
        if not mp.exists():
            raise DatasetError(f"{mp} missing (run scan)")
        n, missing = _paths_ok(mp)
        if not n or missing:
            raise DatasetError(f"{mp}: {n} rows, {len(missing)} paths missing (first {missing[:1]})")
        notes.append(f"{mp.name} {n} rows")
    for man in v.get("mirror_manifests") or []:
        mp = ctx.paths.map(man)
        if not mp.exists():   # the mirror stage runs beside the fetch; preflight enforces it before training
            notes.append(f"{mp.name} not installed yet (mirror)"); continue
        n, missing = _paths_ok(mp)
        if missing:
            raise DatasetError(f"{mp}: {len(missing)}/{n} paths missing (first {missing[:1]})")
        notes.append(f"{mp.name} {n} rows")
    for link, rel in (v.get("links") or {}).items():
        if not ctx.paths.map(link).exists():
            raise DatasetError(f"link {ctx.paths.map(link)} missing")
    return "; ".join(notes) or "files ok"


# ---------------------------------------------------------------------------------------------------------------
# list / plan
# ---------------------------------------------------------------------------------------------------------------

def cmd_list(ctx, sel):
    print(f"{'section':9} {'name':22} {'access':8} {'GB':>8}  {'credentials':28} {'manifest':40} licence")
    for k, (sec, v) in entries(ctx.reg).items():
        mans = ",".join(Path(m).name for m in manifest_list(v)) or "-"
        creds = ",".join(v.get("credentials") or []) or ("(" + ",".join(v["optional_credentials"]) + ")"
                                                        if v.get("optional_credentials") else "-")
        print(f"{sec:9} {k:22} {v.get('access', '?'):8} {gb((v.get('size_gb') or 0) * 1e9):>8}  {creds:28} "
              f"{mans:40} {v.get('licence', '')}")
    return 0


def tool_notes(v):
    t = []
    if v["extract"] == "zip_split" and not shutil.which("zip"):
        t.append("needs Info-ZIP zip")
    if v["extract"] == "rar" and not (shutil.which("unrar") or shutil.which("7z") or shutil.which("7za")):
        t.append("needs unrar or 7z")
    if v.get("fetch") == "hf" and importlib.util.find_spec("huggingface_hub") is None:
        t.append("needs huggingface_hub")
    if v["extract"] == "tar" and any(f["name"].endswith(".bz2") for f in v["files"]) and not shutil.which("lbzip2"):
        t.append("lbzip2 absent (slower)")
    return t


def cmd_plan(ctx, sel):
    print(f"downloader: {ctx.downloader}; root {ctx.paths.root}; disk factor x{ctx.args.disk_factor}")
    print(f"{'name':22} {'access':8} {'files':>5} {'GB':>8} {'need GB':>8}  state / notes")
    tot = need_tot = 0; blocked = False
    for k in sel:
        sec, v = entries(ctx.reg)[k]
        dest = ctx.paths.dest(v)
        size = sum((f.get("size_bytes") or f.get("size_bytes_estimate") or 0) for f in v.get("files") or [])
        need = disk_need(ctx, v, dest) if v.get("files") else 0
        state = "scanned" if (dest / ".scanned").exists() else "extracted" if (dest / ".extracted").exists() else \
                "fetched" if (dest / ".fetched").exists() else "partial" if dest.exists() else "to fetch"
        miss = need_credentials(v, ctx.env)
        notes = [state] + ([f"MISSING {','.join(miss)}"] if miss else []) + tool_notes(v)
        if v.get("access") in ("request", "manual") and not v.get("files"):
            notes.append(f"access {v['access']}: not fetched by this tool")
        if any(not f.get("size_bytes") for f in v.get("files") or []):
            notes.append("some sizes are estimates")
        blocked |= bool(miss)
        tot += size; need_tot += need
        print(f"{k:22} {v['access']:8} {len(v.get('files') or []):>5} {gb(size):>8} {gb(need):>8}  {'; '.join(notes)}")
    free = free_bytes(ctx.paths.root)
    print(f"total {gb(tot)} GB of archives; still needed on disk {gb(need_tot)} GB; free {gb(free)} GB at {ctx.paths.root}")
    if need_tot > free:
        print("NOT ENOUGH DISK"); blocked = True
    if blocked:
        print(f"blocked: see {GUIDE}")
    return 1 if blocked else 0


# ---------------------------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------------------------

def select(reg, args):
    names = entries(reg)
    if args.only:
        want = [s.strip() for s in args.only.split(",") if s.strip()]
        bad = [w for w in want if w not in names]
        if bad:
            raise SystemExit(f"unknown dataset(s): {', '.join(bad)} (see `list`)")
        return want, True
    sel = list(reg["datasets"])
    if args.all:   # request/manual entries have nothing to fetch: listed by plan, skipped here
        sel += [k for k, v in reg["optional"].items() if v.get("files")]
    return sel, False


def run_each(sel, fn, parallel=1):
    ok, fail = {}, {}

    def one(k):
        try:
            ok[k] = fn(k)
        except DatasetError as e:
            fail[k] = str(e)
        except Exception as e:   # noqa: BLE001 - an unexpected bug in one dataset must not stop the rest
            fail[k] = f"{type(e).__name__}: {e}"
    if parallel > 1 and len(sel) > 1:
        with ThreadPoolExecutor(parallel) as ex:
            list(ex.map(one, sel))
    else:
        for k in sel:
            one(k)
    for k in sel:
        print(f"{'OK  ' if k in ok else 'FAIL'} {k}: {ok.get(k) if k in ok else fail[k]}")
    return 0 if not fail else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=["list", "plan", "fetch", "scan", "verify"])
    ap.add_argument("--only", default="", help="comma-separated dataset names (datasets: or optional:)")
    ap.add_argument("--all", action="store_true", help="also the optional: entries")
    ap.add_argument("--parallel", type=int, default=4, help="files in flight (fetch) / scanner processes (scan)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--root", default="data")
    ap.add_argument("--registry", default=str(REGISTRY))
    ap.add_argument("--downloader", default="auto", choices=["auto", "aria2c", "curl", "python"])
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--retry-wait", type=float, default=15.0)
    ap.add_argument("--disk-factor", type=float, default=3.0, help="disk per archive byte: archive + tree + FLAC")
    ap.add_argument("--force", action="store_true", help="scan: rewrite manifests that already exist")
    ap.add_argument("--deep", action="store_true", help="verify: re-hash every archive")
    args = ap.parse_args(argv)
    cwd = os.getcwd()
    os.chdir(REPO)   # a relative --root is the repo's data/: manifest paths stay repo-relative, as on the laptop
    try:
        reg = load_registry(args.registry)
        errs = validate(reg)
        if errs:
            print("registry invalid:\n  " + "\n  ".join(errs)); return 2
        ctx = Ctx(args, reg)
        if args.cmd == "list":
            return cmd_list(ctx, None)
        sel, _ = select(reg, args)
        if args.cmd == "plan":
            return cmd_plan(ctx, sel)
        E = entries(reg)
        if args.cmd == "fetch":
            with ThreadPoolExecutor(max(1, args.parallel)) as files:
                # dataset threads only wait on their files and then extract; the file pool bounds the downloads
                return run_each(sel, lambda k: fetch_dataset(ctx, k, E[k][1], files), parallel=max(1, len(sel)))
        if args.cmd == "scan":
            if args.parallel > 1:
                from concurrent.futures import ProcessPoolExecutor
                with ProcessPoolExecutor(args.parallel) as pool:
                    return run_each(sel, lambda k: scan_dataset(ctx, k, E[k][1], pool), parallel=args.parallel)
            return run_each(sel, lambda k: scan_dataset(ctx, k, E[k][1]))
        return run_each(sel, lambda k: verify_dataset(ctx, k, E[k][1]))
    finally:
        os.chdir(cwd)


if __name__ == "__main__":
    sys.exit(main())
