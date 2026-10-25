"""Training boundary integration: real GPU physics with forced mixed resets."""

from argparse import Namespace
import json
import math

import pytest
import torch

from src.diff_loco.train import soft_continuation, train
from src.diff_loco.task import LocomotionTask


def test_soft_survival_restores_height_and_tilt():
    height = torch.tensor(0.6, requires_grad=True)
    angle = torch.tensor(1.0, requires_grad=True)
    q = torch.stack((height * 0, height * 0, height,
                     torch.cos(angle / 2), torch.sin(angle / 2), angle * 0, angle * 0))[None]
    survival = soft_continuation(q, 1.0)
    dh, da = torch.autograd.grad(survival.sum(), (height, angle))
    assert 0 < float(survival) < 1
    assert dh > 0 and da < 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_train_mixed_resets_and_critic_checkpoint(tmp_path, monkeypatch):
    seen = {"fall": 0, "timeout": 0}

    def forced_termination(self, q):
        fallen = torch.zeros_like(self.age, dtype=torch.bool)
        timeout = torch.zeros_like(fallen)
        fallen[0] = self.age[0] >= 3
        timeout[1] = self.age[1] >= 2
        seen["fall"] += int(fallen.sum())
        seen["timeout"] += int(timeout.sum())
        return fallen, timeout

    monkeypatch.setattr(LocomotionTask, "termination", forced_termination)
    args = Namespace(robot="go1", num_envs=2, nominal=False, device="cuda:0",
                     max_speed=0.2, rsi=0.8, seed=44, lr=3e-4, critic_lr=1e-3,
                     discount_seconds=1.0, critic_epochs=2, lam=0.95, resume=None,
                     output=str(tmp_path), updates=2, horizon=4, alpha=0.95,
                     grad_clip=1.0, log_every=1, save_every=1)
    checkpoint = train(args)
    state = torch.load(checkpoint, weights_only=False, map_location="cpu")
    rows = [json.loads(line) for line in (tmp_path / "metrics.jsonl").read_text().splitlines()]
    assert seen["fall"] and seen["timeout"]
    assert state["iteration"] == 2 and state["critic_optimizer"]["state"]
    assert state["actor"]["obs_scale"].numel() == 47
    for row in rows:
        assert math.isfinite(row["loss"]) and math.isfinite(row["critic_loss"])
        assert 0 < row["reward"] <= 1 and 0 < row["gradient_norm"] < 1e5
        assert row["resets"] > row["falls"] > 0
    # Loading both optimizers and critic must remain compatible with a new rollout.
    args.resume, args.updates = str(checkpoint), 1
    train(args)
    assert torch.load(checkpoint, weights_only=False, map_location="cpu")["iteration"] == 3
