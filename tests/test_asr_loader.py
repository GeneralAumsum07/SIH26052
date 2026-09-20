"""Keep the optional ASR loader portable without downloading weights in unit tests."""
import sys
from types import SimpleNamespace

import pytest
import torch  # load native libraries before simulating another platform's DLL API

from vaani import asr


@pytest.mark.parametrize("windows_dll_api", [False, True])
def test_cuda_loader_only_registers_dll_directory_when_supported(monkeypatch, windows_dll_api):
    directories, calls = [], []
    if windows_dll_api:
        monkeypatch.setattr(asr.os, "add_dll_directory", directories.append, raising=False)
    else:
        monkeypatch.delattr(asr.os, "add_dll_directory", raising=False)
    sentinel = object()

    def model(size, **kwargs):
        calls.append((size, kwargs))
        return sentinel

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=model))
    assert asr.load_whisper("cuda", threads=4) is sentinel
    assert len(directories) == int(windows_dll_api)
    assert calls == [("small", {"device": "cuda", "compute_type": "float16", "num_workers": 4})]
