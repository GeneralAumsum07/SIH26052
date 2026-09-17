import importlib.util
import sys
from pathlib import Path

# scripts/ isn't a package; load fetch_data.py directly by path.
_spec = importlib.util.spec_from_file_location(
    "fetch_data", Path(__file__).parents[1] / "scripts" / "fetch_data.py"
)
fetch_data = importlib.util.module_from_spec(_spec)
sys.modules["fetch_data"] = fetch_data
_spec.loader.exec_module(fetch_data)


class _FakeResponse:
    """Minimal stand-in for requests.Response used as a context manager."""

    def __init__(self, status_code, body: bytes, content_length: int | None = None):
        self.status_code = status_code
        self._body = body
        self.headers = {"content-length": str(content_length if content_length is not None else len(body))}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        yield self._body


def test_download_appends_on_206(tmp_path, monkeypatch):
    dst = tmp_path / "f.bin"
    dst.write_bytes(b"AAAA")  # 4 bytes already "downloaded"

    def fake_get(url, stream, headers, timeout):
        assert headers == {"Range": "bytes=4-"}
        return _FakeResponse(206, b"BBBB")

    monkeypatch.setattr(fetch_data.requests, "get", fake_get)
    fetch_data.download("http://x/f.bin", dst)
    assert dst.read_bytes() == b"AAAABBBB"


def test_download_restarts_when_server_ignores_range(tmp_path, monkeypatch):
    dst = tmp_path / "f.bin"
    dst.write_bytes(b"AAAA")  # partial file present

    def fake_get(url, stream, headers, timeout):
        assert headers == {"Range": "bytes=4-"}
        # Server ignores Range and sends the full body with 200 instead of 206.
        return _FakeResponse(200, b"FULLBODY")

    monkeypatch.setattr(fetch_data.requests, "get", fake_get)
    fetch_data.download("http://x/f.bin", dst)
    assert dst.read_bytes() == b"FULLBODY"
