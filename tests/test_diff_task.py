"""GPU integration: randomized worlds agree with independent CPU constants."""

import copy

import mujoco
import numpy as np
import pytest
import torch

from src.diff_loco.task import LocomotionTask, TaskCfg
from src.diff_loco.model import RobotModel


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("robot", ["go1", "g1"])
def test_randomization_constants_and_subset_reset(robot):
    torch.manual_seed(17)
    task = LocomotionTask(TaskCfg(robot=robot, num_envs=2))
    derived = ("body_subtreemass", "dof_invweight0", "body_invweight0")
    for env in range(2):
        cpu = copy.copy(task.sim.mj_model)
        for field in task.fields:
            getattr(cpu, field)[:] = getattr(task.sim.model, field)[env].cpu().numpy()
        mujoco.mj_setConst(cpu, mujoco.MjData(cpu))
        for field in derived:
            actual = getattr(task.sim.model, field)
            assert actual.shape[0] == 2
            np.testing.assert_allclose(actual[env].cpu().numpy(), getattr(cpu, field),
                                       rtol=5e-4, atol=2e-5)
    dynamics = RobotModel(task.sim.mj_model, foot_sites=task.foot_sites,
                          contact_geoms=task.contact_geoms, device=task.device).float()
    params, kp, kd, effort = task.parameters(dynamics.contact_geom_ids)
    np.testing.assert_allclose(params["friction"].cpu(),
                               task.sim.model.geom_friction[:, dynamics.contact_geom_ids, 0].cpu())
    assert params["friction"].shape == (2, dynamics.num_contacts)
    predicted = dynamics.step(*task.state(), task.q0, kp, kd, effort, task.cfg.dt, parameters=params)
    assert all(torch.isfinite(x).all() for x in predicted)
    for _ in range(10):
        task.step(task.q0.expand(2, -1))
    q, v = task.state()
    physical = {f: getattr(task.sim.model, f)[1].clone() for f in task.fields}
    task.reset(torch.tensor([0], device=task.device))
    torch.testing.assert_close(task.state()[0][1], q[1], rtol=0, atol=0)
    torch.testing.assert_close(task.state()[1][1], v[1], rtol=0, atol=0)
    for field, value in physical.items():
        torch.testing.assert_close(getattr(task.sim.model, field)[1], value, rtol=0, atol=0)
    for _ in range(10):
        q, v = task.step(task.q0.expand(2, -1))
    assert torch.isfinite(q).all() and torch.isfinite(v).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("robot", ["go1", "g1"])
def test_privileged_context_does_not_enter_actor(robot):
    task = LocomotionTask(TaskCfg(robot=robot, num_envs=2))
    q, v = task.state()
    action = torch.zeros(2, task.n, device=task.device)
    before = task.observation(q, v, action)
    context = task.critic_context(task.contact_ids)
    # Unobservable root translation/velocity and physical masses belong only to critic.
    altered_q, altered_v = q.clone(), v.clone()
    altered_q[:, :3] += 0.1
    altered_v[:, :3] += 0.3
    task.sim.model.body_mass[:, 1:] *= 1.05
    after = task.observation(altered_q, altered_v, action)
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert before.shape[-1] == 11 + 3 * task.n
    assert not torch.equal(context, task.critic_context(task.contact_ids))
    assert not context.requires_grad and torch.isfinite(context).all()
