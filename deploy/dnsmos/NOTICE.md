# DNSMOS model: attribution and licence

`deploy/dnsmos/sig_bak_ovr.onnx` is the non-personalised DNSMOS P.835 model published by Microsoft
in the DNS Challenge repository (`microsoft/DNS-Challenge`, `DNSMOS/` directory). `vaani/dnsmos.py`
mirrors that repository's `dnsmos_local.py` (9.01 s windows, polynomial fits for
`is_personalized_MOS=False`).

| field | value |
|---|---|
| file | `deploy/dnsmos/sig_bak_ovr.onnx` |
| sha256 | `269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd` |
| origin | Microsoft, `microsoft/DNS-Challenge`, `DNSMOS/` |
| use in this repo | evaluation only (SIG / BAK / OVRL scores in the eval CSVs); never part of the enhancement path or the board runtime |
| licence | TBD: which licence file of `microsoft/DNS-Challenge` covers the `DNSMOS/*.onnx` model files, and at which commit was this copy taken? Not established from anything in this repository. |
| redistribution | TBD: follows from the licence above. Until it is confirmed, treat the file as third-party material that is not covered by this repository's MIT licence. |

## Citation

Reddy, C. K. A., Gopal, V., and Cutler, R. "DNSMOS P.835: A Non-Intrusive Perceptual Objective
Speech Quality Metric to Evaluate Noise Suppressors." ICASSP 2022.

## Scope

DNSMOS numbers reported by this project are a non-intrusive proxy computed by Microsoft's model;
they are not listening-test MOS. The model is copyright Microsoft and is attributed here, not
relicensed.
