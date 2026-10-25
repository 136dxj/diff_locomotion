"""Hardware-only observations and rollout graph semantics."""

from types import SimpleNamespace

import pytest
import torch

from src.diff_loco.policy import Actor, align, observation_schema
from src.diff_loco.task import LocomotionTask


@pytest.mark.parametrize("n", [12, 29])
def test_actor_ignores_privileged_state_and_world_heading(n):
    cfg = SimpleNamespace(randomize=False)
    task = SimpleNamespace(
        cfg=cfg, encoder_bias=torch.zeros(2, n), commands=torch.tensor([[0.5, 0.0, 0.1]]).expand(2, -1),
        q0=torch.zeros(n), phase=lambda: torch.tensor([0.2, 0.2]),
    )
    q = torch.zeros(2, n + 7)
    q[:, 3] = 1
    q[1, :3] = torch.tensor([10., -5., 2.])
    q[1, 3:7] = torch.tensor([0.5**0.5, 0., 0., 0.5**0.5])
    v = torch.zeros(2, n + 6)
    v[1, :3] = torch.tensor([5., -2., 1.])
    obs = LocomotionTask.observation(task, q, v, torch.zeros(2, n))
    torch.testing.assert_close(obs[0], obs[1])
    assert obs.shape == (2, 11 + 3 * n)
    assert sum(x["size"] for x in observation_schema(n)) == obs.shape[-1]
    actor = Actor(n)
    torch.testing.assert_close(actor(obs)[0], actor(obs)[1])
    # A changed physical joint reading must affect the observation.
    q[1, 7] += 0.1
    changed = LocomotionTask.observation(task, q, v, torch.zeros(2, n))
    assert not torch.equal(changed[0], changed[1])


def test_alignment_forward_values_and_gradients_and_reset_rows():
    predicted = torch.randn(3, 5, requires_grad=True)
    actual = torch.randn(3, 5)
    aligned = align(actual, predicted, 0.8)
    assert torch.equal(aligned.detach(), actual)
    reset = torch.tensor([False, True, False])[:, None]
    next_state = torch.where(reset, torch.zeros_like(aligned), aligned)
    next_state.sum().backward()
    torch.testing.assert_close(predicted.grad[0], torch.full((5,), 0.8))
    assert predicted.grad[1].count_nonzero() == 0
    assert next_state.detach().grad_fn is None


def test_standing_phase_is_zero_and_phase_is_pure():
    from src.diff_loco.policy import observe
    z3, zn = torch.zeros(1, 3), torch.zeros(1, 12)
    phase = torch.tensor([0.2])
    obs = observe(z3, z3, z3, phase, zn, zn, zn, zn)
    assert obs[:, 9:11].count_nonzero() == 0
    assert phase.item() == pytest.approx(0.2)
