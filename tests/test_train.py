

def test_build_model_warm_starts_from_vaani_checkpoint(tmp_path):
    import torch
    from vaani.train import build_model
    from vaani.models.vaani_net import VaaniNet
    src = VaaniNet(); ck = tmp_path / "best.pt"
    torch.save({"model": src.state_dict(), "config": {"model": "vaani"}}, ck)
    m = build_model("vaani", ck)
    assert torch.equal(m.encoder.film.weight, src.encoder.film.weight)
