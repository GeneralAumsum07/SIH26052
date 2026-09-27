"""Audio contracts: the framing a VaaniFE network was trained for (low-delay plan, Section 3.1-3.2).

A contract fixes the analysis/FFT length K, the hop H, the synthesis support L (the algorithmic delay),
the windows, the alignment rule, the frontend policy and the deployment resampler. A network is only ever
run on the contract it was trained with: trained masks are specific to their synthesis support.

Registered contracts:
  vaanife_r8_control_legacy512_h256_v1   C0: the centered 512/256 sqrt-Hann STFT of vaani/dsp/stft.py
  vaanife_ld_asym512_h96_s{160,144,128}_v1   Arm A at L = 10, 9 or 8 ms
  vaanife_ld_asym512_h128_s160_v1        Arm B

The object is never called `profile` or `profile_id`: those already mean the tier name and the stream-state
identifier (vaani/backend.py). `model_cfg.audio_contract` names it; absent = C0.

Low-delay windows (K = 512, hop H, support L, crossfade X = L - H; H < L <= 2H):
  a[:K-H] = sin(pi n / 2(K-H)),  a[K-H:] = cos(pi n / 2H)        analysis, sum(a^2) = 256 for every H
  p = 0 before K-L, raised-cosine crossfades of X samples, 1 between   product window, overlap-adds to 1
  s = p / a where a > 0                                            synthesis
Coefficient hashes are taken over the windows rounded to 12 decimals, so a last-ulp libm difference between
machines cannot change them while any real edit does.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json

import numpy as np

SR = 16000
K = 512
N_BINS = K // 2 + 1

LEGACY_ID = "vaanife_r8_control_legacy512_h256_v1"
KIND_LEGACY, KIND_LD = "legacy_centered", "low_delay_asym"
RESAMPLER_R0 = "r0_linphase_kaiser193_v1"     # vaani/live.py: 193-tap linear-phase Kaiser sinc, 4.0 ms pair
RESAMPLER_R1 = "r1_minphase_kaiser193_v1"     # its minimum-phase equivalent, same magnitude, 0.4 ms pair (D7 default)
RESAMPLER_R2 = "r2_cdelay_ls193_v1"           # constrained-delay near-linear-phase LS FIR (D7 alternative)
RAMP_SAMPLES = 3072                           # reconnect ramp: 192 ms, the r8 ramp_frames 12 x 256
LIMITER_SUB = 32                              # r8 limiter sub-block
GUARD_WINDOW = 256                            # guards' sliding statistics window, independent of the hop


def _round_hash(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.round(np.asarray(a, np.float64), 12).tobytes())
    return h.hexdigest()


def ld_windows(k: int, hop: int, support: int):
    """(a, p, s) float64 windows of the low-delay asymmetric pair (Section 3.1)."""
    if not (0 < hop < support <= 2 * hop) or support > k or hop >= k:
        raise ValueError(f"need H < L <= 2H and L <= K, got K={k} H={hop} L={support}")
    x = support - hop
    a = np.zeros(k)
    a[:k - hop] = np.sin(np.pi * np.arange(k - hop) / (2 * (k - hop)))
    a[k - hop:] = np.cos(np.pi * np.arange(hop) / (2 * hop))
    p = np.zeros(k)
    p[k - support:] = 1.0
    p[k - support:k - support + x] = np.sin(np.pi * np.arange(x) / (2 * x)) ** 2
    p[k - x:] = np.cos(np.pi * np.arange(x) / (2 * x)) ** 2
    s = np.divide(p, a, out=np.zeros(k), where=a > 0)
    return a, p, s


def legacy_windows(k: int = K):
    """(a, p, s) of the legacy periodic sqrt-Hann pair (torch.hann_window(512).sqrt())."""
    a = np.hanning(k + 1)[:-1] ** 0.5
    return a, a * a, a.copy()


@dataclasses.dataclass(frozen=True)
class AudioContract:
    audio_contract_id: str
    kind: str
    k: int
    hop: int
    support: int
    crossfade: int
    window_id: str
    window_hash: str
    window_energy: float
    alignment_version: int
    offset: int
    hops_per_s: float
    deadline_ms: float
    limiter_sub: int
    ramp_samples: int
    guard_window: int
    resampler_id: str
    sr: int = SR

    # ---- derived ---------------------------------------------------------------------------
    @property
    def is_legacy(self) -> bool:
        return self.kind == KIND_LEGACY

    @property
    def n_bins(self) -> int:
        return self.k // 2 + 1

    @property
    def history(self) -> int:
        """Retained analysis history per channel (samples)."""
        return self.k - self.hop

    @property
    def algorithmic_delay_ms(self) -> float:
        return 1000.0 * self.support / self.sr

    @property
    def hop_ms(self) -> float:
        return 1000.0 * self.hop / self.sr

    @property
    def lookahead_ms(self) -> float:
        """Average model look-ahead per output sample, (L-1)/2 samples (Section 2.1)."""
        return 1000.0 * (self.support - 1) / 2 / self.sr

    @property
    def release_lead(self) -> int:
        """Samples the first streamed release precedes stream sample 0 (pre-start positions, discarded offline)."""
        return self.support - self.hop

    def n_frames(self, n: int) -> int:
        if self.is_legacy:
            return n // self.hop + 1
        return (n + self.hop - 1) // self.hop + 1

    def windows(self):
        if self.is_legacy:
            return legacy_windows(self.k)
        return ld_windows(self.k, self.hop, self.support)

    # ---- serialisation ---------------------------------------------------------------------
    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @property
    def contract_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()[:16]

    @classmethod
    def from_dict(cls, d: dict) -> "AudioContract":
        c = cls(**d)
        c.validate()
        return c

    def validate(self):
        """Raise on any inconsistent field, including an altered coefficient hash."""
        if self.k != K or self.sr != SR:
            raise ValueError(f"{self.audio_contract_id}: K must be {K} and sr {SR}")
        if not (0 < self.hop < self.support <= 2 * self.hop):
            raise ValueError(f"{self.audio_contract_id}: synthesis support L={self.support} outside (H, 2H] for H={self.hop}")
        if self.crossfade != self.support - self.hop:
            raise ValueError(f"{self.audio_contract_id}: crossfade must be L - H")
        if abs(self.hops_per_s - self.sr / self.hop) > 1e-9:
            raise ValueError(f"{self.audio_contract_id}: hops_per_s must be sr / H")
        if self.kind not in (KIND_LEGACY, KIND_LD):
            raise ValueError(f"{self.audio_contract_id}: unknown kind {self.kind!r}")
        if self.kind == KIND_LD and self.offset != self.hop:
            raise ValueError(f"{self.audio_contract_id}: low-delay sequence offset must equal H")
        a, _, s = self.windows()
        if _round_hash(a, s) != self.window_hash or REGISTERED_HASHES.get(self.audio_contract_id, self.window_hash) != self.window_hash:
            raise ValueError(f"{self.audio_contract_id}: window coefficient hash differs from the registered one")
        if abs(float((a * a).sum()) - self.window_energy) > 1e-9:
            raise ValueError(f"{self.audio_contract_id}: analysis window energy differs")
        if self.limiter_sub <= 0 or self.hop % self.limiter_sub:
            raise ValueError(f"{self.audio_contract_id}: hop must be a whole number of limiter sub-blocks")
        return self


def _make(cid, kind, hop, support, window_id, resampler_id, offset):
    a, _, s = (legacy_windows() if kind == KIND_LEGACY else ld_windows(K, hop, support))
    return AudioContract(audio_contract_id=cid, kind=kind, k=K, hop=hop, support=support, crossfade=support - hop,
                         window_id=window_id, window_hash=_round_hash(a, s), window_energy=float(round((a * a).sum(), 9)),
                         alignment_version=1, offset=offset, hops_per_s=SR / hop, deadline_ms=1000.0 * hop / SR,
                         limiter_sub=LIMITER_SUB, ramp_samples=RAMP_SAMPLES, guard_window=GUARD_WINDOW,
                         resampler_id=resampler_id)


def ld_contract_id(hop: int, support: int) -> str:
    return f"vaanife_ld_asym512_h{hop}_s{support}_v1"


# Registered window hashes (Section 3.2). A contract whose windows no longer hash to these is rejected, so an edit
# to the window formulas cannot silently re-label trained artifacts. Regenerate only with a new contract version.
REGISTERED_HASHES = {
    "vaanife_r8_control_legacy512_h256_v1": "7921aeb533efd8721570accff24b38e4c5caabef69bc6b459a29d97845521348",
    "vaanife_ld_asym512_h96_s160_v1": "2b30ec549944d38841be191687fe954a5e98892a75f0e0f96d56d9462f1156a6",
    "vaanife_ld_asym512_h96_s144_v1": "c4e823467b45d62db16af7cbd6f490bb014ff65f84e9b25e78ffabae2342bcb5",
    "vaanife_ld_asym512_h96_s128_v1": "5d48631de42c140c7e5182a15f07e27a6947c78ff023702bf6d5917aa13431e1",
    "vaanife_ld_asym512_h128_s160_v1": "a4f697af744c035fc3a4ac1d04bbb501dfe3aba01a84337ee8f316534be40c52",
}

_REGISTRY: dict[str, AudioContract] = {}


def _register(c: AudioContract):
    c.validate()
    if REGISTERED_HASHES.get(c.audio_contract_id) != c.window_hash:
        raise ValueError(f"{c.audio_contract_id}: windows do not match the registered coefficient hash")
    _REGISTRY[c.audio_contract_id] = c


_register(_make(LEGACY_ID, KIND_LEGACY, 256, 512, "sqrt_hann_periodic_512", RESAMPLER_R0, 0))
for _h, _l in ((96, 160), (96, 144), (96, 128), (128, 160)):
    _register(_make(ld_contract_id(_h, _l), KIND_LD, _h, _l, f"asym_sqrt_sin_cos_512_h{_h}_x{_l - _h}",
                    RESAMPLER_R1, _h))
del _h, _l

ARM_A_IDS = tuple(ld_contract_id(96, s) for s in (160, 144, 128))
ARM_B_ID = ld_contract_id(128, 160)


def get_audio_contract(audio_contract_id: str | None) -> AudioContract:
    """Registered contract by ID; None = C0 (legacy). Unknown IDs are rejected."""
    if audio_contract_id is None:
        audio_contract_id = LEGACY_ID
    if isinstance(audio_contract_id, AudioContract):
        return audio_contract_id
    try:
        return _REGISTRY[audio_contract_id]
    except KeyError:
        raise ValueError(f"unknown audio contract {audio_contract_id!r}; registered: {sorted(_REGISTRY)}") from None


def registered() -> dict[str, AudioContract]:
    return dict(_REGISTRY)


def contract_of(model_cfg: dict | None) -> AudioContract:
    """The contract a model_cfg names (C0 when the field is absent)."""
    return get_audio_contract((model_cfg or {}).get("audio_contract"))


def verify_record(d: dict) -> AudioContract:
    """A serialized contract (sidecar, checkpoint, ONNX metadata) must equal the registered one exactly."""
    c = AudioContract.from_dict(d)
    reg = get_audio_contract(c.audio_contract_id)
    if reg != c:
        raise ValueError(f"contract record {c.audio_contract_id} differs from the registered contract")
    return reg


# ---- artifact metadata (Task 5) ---------------------------------------------------------------------
# ONNX metadata_props keys stamped by vaani.export.export_fe before the graph hashes are taken. The board path reads
# them through onnxruntime's custom_metadata_map, so this module stays numpy-only.
META_ID = "audio_contract_id"
META_HASH = "audio_contract_hash"
META_RECORD = "audio_contract"          # the full contract record, JSON
META_PROFILE = "vaani_profile"          # tier / network name, e.g. "mini", "mini_p18", "mini_df96"
META_MODEL_CFG = "vaani_model_cfg"      # the exported (folded) model_cfg, JSON


def onnx_metadata(contract, profile: str | None = None, model_cfg: dict | None = None) -> dict:
    c = get_audio_contract(contract)
    meta = {META_ID: c.audio_contract_id, META_HASH: c.contract_hash,
            META_RECORD: json.dumps(c.to_dict(), sort_keys=True)}
    if profile:
        meta[META_PROFILE] = profile
    if model_cfg is not None:
        meta[META_MODEL_CFG] = json.dumps(model_cfg, sort_keys=True, default=list)
    return meta


def contract_from_metadata(meta: dict | None, expected: str | None = None, where: str = "artifact") -> AudioContract:
    """The contract an artifact's metadata declares, verified against the registry.

    Metadata without a contract is a known legacy artifact (C0, explicit legacy dispatch); it is never accepted as a
    low-delay artifact. `expected` (a contract ID, e.g. from the sidecar) must agree when given."""
    meta = meta or {}
    exp = None if expected is None else get_audio_contract(expected)
    if META_ID not in meta:
        if exp is not None and not exp.is_legacy:
            raise ValueError(f"{where}: no {META_ID} metadata; an unstamped artifact is never accepted as the "
                             f"low-delay contract {exp.audio_contract_id}")
        return get_audio_contract(None)
    if META_RECORD not in meta or META_HASH not in meta:
        raise ValueError(f"{where}: {META_ID} is stamped without its record and hash")
    rec = verify_record(json.loads(meta[META_RECORD]))
    if rec.audio_contract_id != meta[META_ID] or rec.contract_hash != meta[META_HASH]:
        raise ValueError(f"{where}: contract metadata disagree (id {meta[META_ID]}, hash {meta[META_HASH]}, "
                         f"record {rec.audio_contract_id}/{rec.contract_hash})")
    if exp is not None and exp != rec:
        raise ValueError(f"{where}: stamped for {rec.audio_contract_id} ({rec.contract_hash}) but "
                         f"{exp.audio_contract_id} ({exp.contract_hash}) is expected")
    return rec


def config_extra(contract) -> dict | None:
    """The contract's part of vaani.backend.config_hash: None for C0, so legacy hashes are unchanged."""
    c = get_audio_contract(contract)
    return None if c.is_legacy else {META_ID: c.audio_contract_id, META_HASH: c.contract_hash}


def sidecar_fields(contract) -> dict:
    """model_config.json fields of a low-delay artifact (empty for C0: legacy sidecars are unchanged)."""
    c = get_audio_contract(contract)
    return {} if c.is_legacy else {META_ID: c.audio_contract_id, META_HASH: c.contract_hash, META_RECORD: c.to_dict()}


def contract_from_sidecar(cfg: dict, where: str = "model_config") -> AudioContract:
    """The contract of a model_config.json: model_cfg's audio_contract, which must agree with the recorded fields.
    A low-delay model_cfg without its recorded contract is refused."""
    c = contract_of(cfg.get("model_cfg"))
    if META_RECORD not in cfg and META_ID not in cfg:
        if not c.is_legacy:
            raise ValueError(f"{where}: model_cfg names {c.audio_contract_id} but the contract record is missing")
        return c
    rec = verify_record(cfg[META_RECORD]) if META_RECORD in cfg else get_audio_contract(cfg[META_ID])
    if rec != c or cfg.get(META_ID, rec.audio_contract_id) != rec.audio_contract_id \
            or cfg.get(META_HASH, rec.contract_hash) != rec.contract_hash:
        raise ValueError(f"{where}: recorded contract {cfg.get(META_ID)} / {cfg.get(META_HASH)} disagrees with "
                         f"model_cfg's {c.audio_contract_id} / {c.contract_hash}")
    return rec
