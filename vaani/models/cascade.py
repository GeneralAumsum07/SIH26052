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
    def __init__(self, first_model_cfg: dict | None = None):
        super().__init__()
        self.first = VaaniNet(**(first_model_cfg or {}))
        for p in self.first.parameters(): p.requires_grad_(False)
        self.refiner = ResidualRefiner()
        self.first.eval()

    def train(self, mode: bool = True):
        super().train(mode); self.first.eval()   # the frozen stage never leaves eval, whatever the parent does
        return self

    def forward(self, spec6, feats):
        with torch.no_grad():
            y = self.first(spec6, feats)
        return self.refiner(spec6[..., 0:2], spec6[..., 2:4], y)

    @classmethod
    def from_first_stage(cls, ckpt_path):
        """New cascade around a trained first stage; returns (module, cascade config) ready for torch.save."""
        ck = torch.load(ckpt_path, map_location="cpu", weights_only=True); cfg = ck["config"]
        assert cfg["model"] == "vaani", cfg["model"]
        m = cls(cfg.get("model_cfg")); m.first.load_state_dict(ck["model"])
        ccfg = copy.deepcopy(cfg)   # controller_on / dsp stay top-level so vaani.eval runs the DSP exactly as the first stage trained
        ccfg["model"] = MODEL_NAME
        ccfg["first_stage"] = {"source": str(ckpt_path), "sha256": _sha(ckpt_path), "config": cfg}
        return m, ccfg

    @classmethod
    def from_config(cls, cfg):
        assert cfg["model"] == MODEL_NAME, cfg["model"]
        return cls(cfg.get("model_cfg"))


class StreamCascade(nn.Module):
    """Streaming twin for export: the existing seven inputs + refine_cache, six outputs + refine_cache_out."""

    def __init__(self, first_model_cfg=None):
        super().__init__()
        self.first = StreamVaaniNet(**(first_model_cfg or {})); self.refiner = ResidualRefiner()

    def forward(self, spec6, feats, conv_cache, tra_cache, inter_cache, df_cache, coh_cache, refine_cache):
        y, conv_cache, tra_cache, inter_cache, df_cache, coh_cache = self.first(spec6, feats, conv_cache, tra_cache, inter_cache, df_cache, coh_cache)
        z, refine_cache = self.refiner.step(spec6[..., 0:2], spec6[..., 2:4], y, refine_cache)
        return z, conv_cache, tra_cache, inter_cache, df_cache, coh_cache, refine_cache


def init_cascade_caches(device="cpu"):
    return (*init_caches(device), init_refine_cache(device))
