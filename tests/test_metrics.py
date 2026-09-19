import numpy as np
import torch
from vaani import metrics


def test_snr_definition_is_error_based():
    c = np.random.default_rng(0).standard_normal(16000).astype(np.float32)
    assert metrics.snr_db(c, c) > 100
    assert abs(metrics.snr_db(c, c + 0.1 * c) - 20.0) < 1e-3   # 10% scale error = 20 dB SNR
    assert metrics.si_sdr_db(c, 2 * c) > 100                    # SI-SDR ignores scale, SNR does not


def test_recovery_time():
    sr = 16000; t = np.arange(2 * sr) / sr
    twin = np.sin(2 * np.pi * 200 * t).astype(np.float32)
    est = twin.copy(); on = int(0.5 * sr)
    est[on:on + int(0.3 * sr)] *= 0.1          # 300 ms of suppression after a burst at 0.5 s
    r = metrics.recovery_time_s(est, twin, 0.5)
    assert 0.25 < r < 0.4


def _mini_eval_set(tmp_path):
    """Tiny rendered eval set via render_bucket_item, mirroring scripts/render_eval_sets.py."""
    import json
    import pandas as pd
    import soundfile as sf
    from scripts.render_eval_sets import render_bucket_item
    from vaani.data.rirs import build_bank

    sr = 16000
    rng = np.random.default_rng(0)
    speech_dir = tmp_path / "speech"; noise_dir = tmp_path / "noise"
    speech_dir.mkdir(); noise_dir.mkdir()
    speech_path = speech_dir / "s0.wav"; noise_path = noise_dir / "n0.wav"
    sf.write(speech_path, rng.standard_normal(sr).astype(np.float32), sr)
    sf.write(noise_path, rng.standard_normal(sr).astype(np.float32), sr)
    speech_df = pd.DataFrame({"path": [str(speech_path)]})
    pool_df = pd.DataFrame({"path": [str(noise_path)]})

    bank_path = tmp_path / "bank.npz"
    build_bank(bank_path, n=2, seed=0, n_noise=1)
    from vaani.data.rirs import RirBank
    bank = RirBank(bank_path)

    root = tmp_path / "eval" / "test" / "stationary_0"; root.mkdir(parents=True)
    m, c, meta, twin = render_bucket_item([0, 1, 100, 0], speech_df, pool_df, sr, 0.0, False, bank)
    meta["noise_class"] = "stationary"
    sf.write(root / "0000.mix.wav", m.T, sr); sf.write(root / "0000.clean.wav", c, sr)
    json.dump(meta, open(root / "0000.json", "w"))
    return tmp_path / "eval"


def test_enhance_fn_baselines(tmp_path):
    from vaani.eval import enhance_fn
    mix = np.random.default_rng(0).standard_normal((2, 16000)).astype(np.float32)
    for name in ("raw", "nlms_only"):
        est = enhance_fn(name)(mix)
        assert est.shape == (16000,)
        assert np.isfinite(est).all()


def test_eval_main_writes_csv(tmp_path, monkeypatch):
    import sys
    from vaani import eval as vaani_eval

    eval_root = _mini_eval_set(tmp_path)
    out = tmp_path / "raw.csv"
    argv = ["eval.py", "--system", "raw", "--split", "test", "--eval-root", str(eval_root), "--out", str(out)]
    monkeypatch.setattr(sys, "argv", argv)
    vaani_eval.main()

    import pandas as pd
    df = pd.read_csv(out)
    assert len(df) == 1
    for col in ("system", "id", "bucket", "snr_out", "si_sdr", "stoi", "pesq_wb"):
        assert col in df.columns
    assert df.system.iloc[0] == "raw"


def test_report_main_writes_markdown(tmp_path):
    import sys
    from vaani import report

    csv_path = tmp_path / "raw.csv"
    csv_path.write_text(
        "system,id,bucket,noise_class,snr_in,clipped,ref_dropout,impulse_peak_db,"
        "snr_out,si_sdr,stoi,pesq_wb,recovery_s,asr_text\n"
        "raw,0000,stationary_0,stationary,0,False,False,,16.0,10.0,0.9,3.0,,\n"
        "raw,0001,stationary_0,stationary,5,False,False,,20.0,12.0,0.95,3.5,,\n"
    )
    out = tmp_path / "matrix.md"
    argv = ["report.py", str(csv_path), "--out", str(out)]
    import sys as _sys
    old = _sys.argv; _sys.argv = argv
    try:
        report.main()
    finally:
        _sys.argv = old

    text = out.read_text(encoding="utf-8")
    assert "# Ablation matrix" in text
    assert "raw" in text
    assert "✓" in text  # both rows clear all three targets


def _train_two_steps(tmp_path, model_name):
    """Reuses tests/test_train_smoke.py's fixture: train 2 real steps, return the checkpoint path."""
    import yaml
    from vaani import train
    from tests.test_train_smoke import _tiny

    m = _tiny(tmp_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = dict(name=f"eval_smoke_{model_name}", model=model_name, controller_on=True, loss="hybrid",
               init_from="vaani/models/checkpoints/model_trained_on_dns3.tar",
               data=dict(manifests=[str(m)], bank=None, crop_s=1.0, epoch_len=4, mix={"p_room": 0.0}),
               val=dict(dynamic_items=2),
               optim=dict(lr=1e-4, lr_new=1e-3, warmup=1, clip=5.0), batch_size=2, epochs=1, max_steps=2,
               amp=False, device=device, runs_dir=str(tmp_path / "runs"), num_workers=0, seed=0)
    cp = tmp_path / f"{model_name}.yaml"; yaml.safe_dump(cfg, open(cp, "w"))
    train.main(str(cp))
    return tmp_path / "runs" / f"eval_smoke_{model_name}" / "best.pt"


def test_enhance_fn_ckpt_gtcrn(tmp_path):
    import torch as _torch
    from vaani.eval import enhance_fn

    ck = _train_two_steps(tmp_path, "gtcrn")
    mix = np.random.default_rng(0).standard_normal((2, 32000)).astype(np.float32)
    est = enhance_fn(f"ckpt:{ck}")(mix)
    assert est.shape == (32000,)
    assert _torch.isfinite(_torch.from_numpy(est)).all()


def test_enhance_fn_ckpt_vaani(tmp_path):
    import torch as _torch
    from vaani.eval import enhance_fn

    ck = _train_two_steps(tmp_path, "vaani")
    mix = np.random.default_rng(0).standard_normal((2, 32000)).astype(np.float32)
    est = enhance_fn(f"ckpt:{ck}")(mix)
    assert est.shape == (32000,)
    assert _torch.isfinite(_torch.from_numpy(est)).all()


def test_eval_main_skips_bad_clip(tmp_path, monkeypatch):
    """A clip that raises during enhancement must not abort the run - row is written with NaN metrics."""
    import sys
    from vaani import eval as vaani_eval

    eval_root = _mini_eval_set(tmp_path)
    out = tmp_path / "raw.csv"
    argv = ["eval.py", "--system", "raw", "--split", "test", "--eval-root", str(eval_root), "--out", str(out), "--workers", "0"]  # in-process so the monkeypatch reaches enhance_fn
    monkeypatch.setattr(sys, "argv", argv)
    def _boom(mix): raise RuntimeError("boom")
    monkeypatch.setattr(vaani_eval, "enhance_fn", lambda spec, device=None: _boom)
    vaani_eval.main()

    import pandas as pd
    df = pd.read_csv(out)
    assert len(df) == 1
    assert np.isnan(df.snr_out.iloc[0])
    assert str(df.id.iloc[0]) in ("0000", "0")
