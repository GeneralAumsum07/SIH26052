import numpy as np
import torch
import yaml

from vaani.eval import enhance_fn
from vaani.models.cascade import FrozenCascade


def test_conditional_audio_eval_is_checkpoint_self_contained(tmp_path):
    cfg = dict(model="vaani_cascade", model_cfg=dict(channels=8, noise_floor=True),
               refiner_cfg=dict(hidden=5, past=4), controller_on=True)
    m = FrozenCascade.from_config(cfg)
    with torch.no_grad():
        m.refiner.c2.weight.normal_(0, .01)
    ck = tmp_path / "cascade.pt"
    torch.save(dict(config=cfg, model=m.state_dict()), ck)
    policy = tmp_path / "policy.yaml"
    policy.write_text(yaml.safe_dump(dict(base_checkpoint=str(ck), conditional=dict(mode="always"))))
    ordinary = enhance_fn(f"cascade:{ck}", device="cpu")
    conditional = enhance_fn(f"conditional:{policy}", device="cpu")
    mix = np.random.default_rng(0).normal(0, .03, (2, 8000)).astype(np.float32)
    np.testing.assert_allclose(ordinary(mix), conditional(mix), atol=1e-5, rtol=1e-4)
    assert conditional.conditional_runtime.stats["fire_rate"] == 1.
