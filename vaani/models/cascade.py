"""Frozen VaaniNet + ResidualRefiner as one module and one checkpoint (Tier 4.6 plan §6).

The cascade checkpoint embeds both state dicts and the first stage's config + SHA256, so evaluation and export never
depend on a movable first-stage file. `train()` is overridden: the first stage stays in eval mode with grads off, so
its BatchNorm running statistics cannot drift while the refiner trains.
"""
import copy, hashlib

import torch
from torch import nn

from vaani.models.residual_refiner import ResidualRefiner, init_refine_cache
from vaani.models.vaani_net import StreamVaaniNet, VaaniNet, init_caches

MODEL_NAME = "vaani_cascade"


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


class FrozenCascade(nn.Module):
    def __init__(self, first_model_cfg: dict | None = None, refiner_cfg: dict | None = None):
        super().__init__()
        self.first = VaaniNet(**(first_model_cfg or {}))
        for p in self.first.parameters(): p.requires_grad_(False)
        self.refiner_cfg = dict(refiner_cfg or {})
        self.refiner = ResidualRefiner(**self.refiner_cfg)
        self.first.eval()

    def train(self, mode: bool = True):
        super().train(mode); self.first.eval()   # the frozen stage never leaves eval, whatever the parent does
        return self

    def forward(self, spec6, feats):
        with torch.no_grad():
            y = self.first(spec6, feats)
        return self.refiner(spec6[..., 0:2], spec6[..., 2:4], y)

    @classmethod
    def from_first_stage(cls, ckpt_path, refiner_cfg=None):
        """New cascade around a trained first stage; returns (module, cascade config) ready for torch.save."""
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=True); cfg = ck["config"]
        assert cfg["model"] == "vaani", cfg["model"]
        m = cls(cfg.get("model_cfg"), refiner_cfg); m.first.load_state_dict(ck["model"])
        ccfg = copy.deepcopy(cfg)   # controller_on / dsp stay top-level so vaani.eval runs the DSP exactly as the first stage trained
        ccfg["model"] = MODEL_NAME
        ccfg["refiner_cfg"] = dict(refiner_cfg or {})
        ccfg["first_stage"] = {"source": str(ckpt_path), "sha256": _sha(ckpt_path), "config": cfg}
        return m, ccfg

    @classmethod
    def from_config(cls, cfg):
        assert cfg["model"] == MODEL_NAME, cfg["model"]
        return cls(cfg.get("model_cfg"), cfg.get("refiner_cfg"))


class StreamCascade(nn.Module):
    """Streaming twin for export: the existing seven inputs + refine_cache, six outputs + refine_cache_out."""

    def __init__(self, first_model_cfg=None, refiner_cfg=None):
        super().__init__()
        self.first = StreamVaaniNet(**(first_model_cfg or {})); self.refiner = ResidualRefiner(**(refiner_cfg or {}))

    def forward(self, spec6, feats, *caches):
        y, *first_caches = self.first(spec6, feats, *caches[:-1])
        z, refine_cache = self.refiner.step(spec6[..., 0:2], spec6[..., 2:4], y, caches[-1])
        return z, *first_caches, refine_cache


def init_cascade_caches(device="cpu", first_model_cfg=None, refiner_cfg=None):
    mc, rc = first_model_cfg or {}, refiner_cfg or {}
    return (*init_caches(device, channels=mc.get("channels", 16), noise_floor=mc.get("noise_floor", False)),
            init_refine_cache(device, hidden=rc.get("hidden", 16), past=rc.get("past", 2)))
