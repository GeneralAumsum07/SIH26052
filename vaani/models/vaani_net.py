"""VaaniNet: GTCRN widened to three input signals (primary, reference, NLMS
noise estimate) and conditioned on the 18-dim DSP feature vector.

Design rules (spec 6.2):
- Each signal becomes (mag, re, im) like upstream, so 9 feature maps before
  SFE and 27 after, feeding a 27->16 first conv. Only that conv changes.
- Conditioning is a FiLM-style *shift* on the first encoder block output.
  Shift-only keeps the pretrained scale statistics intact at init.
- The mask is applied to the PRIMARY spectrum only.
- Zero-init of the new input slices and the FiLM projection makes the
  network numerically identical to pretrained GTCRN at step 0.

Encoder/StreamEncoder are re-declared here (not in the vendored files) only to
change the first block's in-channels and insert the FiLM shift.
"""
import torch
import torch.nn as nn

from vaani.models.gtcrn import ERB, SFE, ConvBlock, GTConvBlock, DPGRNN, Decoder, Mask
from vaani.models import gtcrn_stream as gs

MAX_PARAMS = 60_000
N_FEAT = 18
N_SIG = 3  # primary, reference, n_hat
N_PRIM = 9  # primary's slice of the 27 first-conv input channels (SFE keeps signal order)


def _sig_feats(spec6):
    """(B,257,T,6) -> (B,9,T,257): per signal (mag, re, im)."""
    outs = []
    for i in range(N_SIG):
        re = spec6[..., 2 * i].permute(0, 2, 1); im = spec6[..., 2 * i + 1].permute(0, 2, 1)
        outs += [torch.sqrt(re ** 2 + im ** 2 + 1e-12), re, im]
    return torch.stack(outs, dim=1)


class _Encoder(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.en_convs = nn.ModuleList(blocks)
        self.film = nn.Linear(N_FEAT, 16)
        nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)  # inert at step 0

    def _cond(self, x, feats):
        # x: (B,16,T,F). feats: (B,T,18) -> shift (B,16,T,1)
        return x + self.film(feats).permute(0, 2, 1)[..., None]


class Encoder(_Encoder):
    def __init__(self):
        super().__init__([
            ConvBlock(N_SIG * 3 * 3, 16, (1, 5), stride=(1, 2), padding=(0, 2)),
            ConvBlock(16, 16, (1, 5), stride=(1, 2), padding=(0, 2), groups=2),
            GTConvBlock(16, 16, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(1, 1)),
            GTConvBlock(16, 16, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(2, 1)),
            GTConvBlock(16, 16, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(5, 1)),
        ])

    def forward(self, x, feats):
        en_outs = []
        for i, blk in enumerate(self.en_convs):
            x = blk(x)
            if i == 0:
                x = self._cond(x, feats)
            en_outs.append(x)
        return x, en_outs


class VaaniNet(nn.Module):
    """forward(spec6 (B,257,T,6), feats (B,T,18)) -> enhanced primary (B,257,T,2)."""

    def __init__(self):
        super().__init__()
        self.erb = ERB(65, 64); self.sfe = SFE(3, 1)
        self.encoder = Encoder()
        self.dpgrnn1 = DPGRNN(16, 33, 16); self.dpgrnn2 = DPGRNN(16, 33, 16)
        self.decoder = Decoder(); self.mask = Mask()

    def forward(self, spec6, feats):
        prim = spec6[..., :2]
        feat = self.erb.bm(_sig_feats(spec6))     # (B,9,T,129)
        feat = self.sfe(feat)                      # (B,27,T,129)
        feat, en_outs = self.encoder(feat, feats)
        feat = self.dpgrnn1(feat); feat = self.dpgrnn2(feat)
        m = self.erb.bs(self.decoder(feat, en_outs))
        out = self.mask(m, prim.permute(0, 3, 2, 1))
        return out.permute(0, 3, 2, 1)

    @classmethod
    def from_pretrained_gtcrn(cls, ckpt_path):
        """Copy every GTCRN weight; first conv gets primary slice, rest stays zero."""
        v = cls()
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)["model"]
        own = v.state_dict()
        for k, w in sd.items():
            if k == "encoder.en_convs.0.conv.weight":
                new = torch.zeros_like(own[k]); new[:, :N_PRIM] = w   # primary slice
                own[k] = new
            elif k in own and own[k].shape == w.shape:
                own[k] = w
        v.load_state_dict(own)
        return v


class StreamEncoder(_Encoder):
    def __init__(self):
        super().__init__([
            gs.ConvBlock(N_SIG * 3 * 3, 16, (1, 5), stride=(1, 2), padding=(0, 2)),
            gs.ConvBlock(16, 16, (1, 5), stride=(1, 2), padding=(0, 2), groups=2),
            gs.StreamGTConvBlock(16, 16, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(1, 1)),
            gs.StreamGTConvBlock(16, 16, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(2, 1)),
            gs.StreamGTConvBlock(16, 16, (3, 3), stride=(1, 1), padding=(0, 1), dilation=(5, 1)),
        ])

    def forward(self, x, feats, conv_cache, tra_cache):
        en_outs = []
        x = self._cond(self.en_convs[0](x), feats); en_outs.append(x)
        x = self.en_convs[1](x); en_outs.append(x)
        x, conv_cache[:, :, :2, :], tra_cache[0] = self.en_convs[2](x, conv_cache[:, :, :2, :], tra_cache[0]); en_outs.append(x)
        x, conv_cache[:, :, 2:6, :], tra_cache[1] = self.en_convs[3](x, conv_cache[:, :, 2:6, :], tra_cache[1]); en_outs.append(x)
        x, conv_cache[:, :, 6:16, :], tra_cache[2] = self.en_convs[4](x, conv_cache[:, :, 6:16, :], tra_cache[2]); en_outs.append(x)
        return x, en_outs, conv_cache, tra_cache


class StreamVaaniNet(nn.Module):
    """Frame-by-frame twin; load weights via convert_to_stream(stream, batch). Caches as StreamGTCRN."""

    def __init__(self):
        super().__init__()
        self.erb = gs.ERB(65, 64); self.sfe = gs.SFE(3, 1)
        self.encoder = StreamEncoder()
        self.dpgrnn1 = gs.DPGRNN(16, 33, 16); self.dpgrnn2 = gs.DPGRNN(16, 33, 16)
        self.decoder = gs.StreamDecoder(); self.mask = gs.Mask()

    def forward(self, spec6, feats, conv_cache, tra_cache, inter_cache):
        prim = spec6[..., :2]
        feat = self.sfe(self.erb.bm(_sig_feats(spec6)))  # ERB/SFE act per frame, so streaming is exact here
        feat, en_outs, conv_cache[0], tra_cache[0] = self.encoder(feat, feats, conv_cache[0], tra_cache[0])
        feat, inter_cache[0] = self.dpgrnn1(feat, inter_cache[0])
        feat, inter_cache[1] = self.dpgrnn2(feat, inter_cache[1])
        m_feat, conv_cache[1], tra_cache[1] = self.decoder(feat, en_outs, conv_cache[1], tra_cache[1])
        m = self.erb.bs(m_feat)
        out = self.mask(m, prim.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        return out, conv_cache, tra_cache, inter_cache


def init_caches(device="cpu"):
    return gs.init_caches(device)
