"""Physical loss contracts; static FK/autograd only, with no simulator rollout.

Run in the integrated repository's CUDA environment:
    python -B -m pytest -p no:cacheprovider tests/test_diff_loss.py -q
"""

import copy
import math

import mujoco
import numpy as np
import pytest
import torch

from src.diff_loco.model import RobotModel
from src.diff_loco.task import LocomotionTask, TaskCfg


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA LocomotionTask")


@pytest.fixture(scope="module")
def go1_template():
    task = LocomotionTask(TaskCfg(robot="go1", num_envs=1, randomize=False))
    dynamics = RobotModel(task.sim.mj_model, foot_sites=task.foot_sites,
                          contact_geoms=task.contact_geoms, device=task.device).float()
    return task, dynamics


@pytest.fixture
def go1(go1_template):
    original, dynamics = go1_template
    task = copy.copy(original)
    task.initial = original.initial.clone()
    task.q0 = task.initial[7:]
    task.commands = torch.zeros_like(original.commands)
    task.age = torch.zeros_like(original.age)
    task.phase_offset = torch.zeros_like(original.phase_offset)
    return task, dynamics


@pytest.fixture
def walking_loss_cls():
    # Let machines without CUDA skip before importing the training/reference stack.
    from src.diff_loco.train import WalkingLoss
    return WalkingLoss


def _cpu_geometry(task, q):
    """Independent reference: MuJoCo sites and the Go1 foot-sphere bottoms."""
    model = task.sim.mj_model
    data = mujoco.MjData(model)
    data.qpos[:] = q.detach().cpu().numpy()
    mujoco.mj_forward(model, data)
    sites = [model.site(name).id for name in task.foot_sites]
    geoms = [model.geom(name).id for name in task.contact_geoms]
    assert np.all(model.geom_type[geoms] == mujoco.mjtGeom.mjGEOM_SPHERE)
    lowest = (data.geom_xpos[geoms, 2] - model.geom_size[geoms, 0]).min()
    return data.site_xpos[sites].copy(), float(lowest)


def _grounded(task):
    q = task.initial.clone()
    _, lowest = _cpu_geometry(task, q)
    q[2] -= lowest
    return q


def _evaluate(loss, q):
    task = loss.task
    action = q.new_zeros(1, task.n)
    _, kp, kd, effort = task.parameters(loss.dynamics.contact_geom_ids)
    return loss(q[None], q.new_zeros(1, task.n + 6), action, action,
                task.target(action), kp, kd, effort)


def _safety_and_gradient(loss, q, coordinate):
    _, terms = _evaluate(loss, q)
    safety = terms["safety"].sum()
    gradient, = torch.autograd.grad(safety, coordinate)
    assert torch.isfinite(safety) and torch.isfinite(gradient)
    return float(safety.detach()), float(gradient.detach())


def test_ground_reference_uses_contact_surface_and_initial_body_frame(go1, walking_loss_cls):
    task, dynamics = go1
    # Identity attitude would conceal a world/body reference-frame error.
    yaw = 0.63
    task.initial[:3] += task.initial.new_tensor([0.31, -0.27, 0.11])
    task.initial[3:7] = task.initial.new_tensor([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
    task.height = float(task.initial[2])
    feet, lowest = _cpu_geometry(task, task.initial)
    loss = walking_loss_cls(task, dynamics)
    assert float(loss.height) == pytest.approx(task.height - lowest, abs=2e-6)
    # The site is the sphere center, so its correct ground height is positive.
    assert np.all(feet[:, 2] - lowest > 0.01)
    grounded = _grounded(task)
    assert abs(_cpu_geometry(task, grounded)[1]) < 2e-6
    _, terms = _evaluate(loss, grounded)
    assert float(terms["height"]) < 1e-10
    assert float(terms["gait"]) < 1e-10
    # Check observable loss behavior, regardless of where GaitReference stores rest feet.
    # A different global heading must preserve the same body-frame standing target.
    grounded[3:7] = grounded.new_tensor([math.cos(-0.41 / 2), 0, 0, math.sin(-0.41 / 2)])
    _, turned = _evaluate(loss, grounded)
    assert float(turned["height"]) < 1e-10
    assert float(turned["gait"]) < 1e-10


def test_gait_uses_reference_targets_and_world_vertical_error(go1, walking_loss_cls):
    task, dynamics = go1
    loss = walking_loss_cls(task, dynamics)
    q = _grounded(task)
    _, nominal = _evaluate(loss, q)
    assert float(nominal["gait"]) < 1e-10
    assert float(nominal["safety"]) == pytest.approx(0, abs=1e-9)
    for dz in (-0.02, 0.04):
        displaced = q.clone()
        displaced[2] += dz
        displaced.requires_grad_()
        _, terms = _evaluate(loss, displaced)
        gradient, = torch.autograd.grad(terms["gait"].sum(), displaced)
        # Body-relative foot coordinates are unchanged; world Z must still count.
        assert float(terms["gait"]) > 1e-5
        assert torch.isfinite(gradient).all()
        assert float(gradient[2]) * dz > 0  # Descent moves back toward the ground.
        assert float(terms["gait"]) == pytest.approx(dz**2 / 3, rel=2e-5, abs=1e-9)
        assert float(gradient[2]) == pytest.approx(2 * dz / 3, rel=2e-5, abs=1e-7)

    # Reuse the reference's public targets; IK/RSI correctness has its own tests.
    # Nonzero yaw/translation expose world/body XY confusion, while two different
    # clock phases expose a loss that ignores the shared reference trajectory.
    yaw = -0.41
    q[:2] += q.new_tensor([0.31, -0.27])
    q[3:7] = q.new_tensor([math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)])
    feet, _ = _cpu_geometry(task, q)
    rotation = np.array([[math.cos(yaw), -math.sin(yaw)],
                         [math.sin(yaw), math.cos(yaw)]])
    task.commands[0] = q.new_tensor([0.25, -0.06, 0.12])
    task.phase_offset.fill_(0.07)
    for ticks in (13, 43):
        task.age.fill_(ticks)
        desired = loss.reference.targets(task.phase(), task.commands)
        assert desired.shape == (1, len(task.foot_sites), 3)
        world_target = desired[0].detach().cpu().numpy().copy()
        world_target[:, :2] = (world_target[:, :2] @ rotation.T
                              + q[:2].detach().cpu().numpy())
        world_target[:, 2] += float(loss.height)
        expected = np.mean((feet - world_target)**2)
        _, terms = _evaluate(loss, q)
        assert float(terms["gait"]) == pytest.approx(expected, rel=2e-4, abs=2e-8)


def test_low_height_safety_is_continuous_and_pushes_up_with_weight_ten(go1, walking_loss_cls):
    task, dynamics = go1
    loss = walking_loss_cls(task, dynamics)
    q = _grounded(task)

    def sample(fraction):
        height_fraction = q.new_tensor(fraction, requires_grad=True)
        state = torch.cat((q[:2], (height_fraction * loss.height)[None], q[3:]))
        return _safety_and_gradient(loss, state, height_fraction)

    assert sample(1.0) == pytest.approx((0, 0), abs=1e-9)
    assert sample(0.85 + 1e-4) == pytest.approx((0, 0), abs=1e-9)
    unsafe, gradient = sample(0.70)
    assert unsafe > 0 and gradient < 0
    assert (unsafe, gradient) == pytest.approx((0.25, -10 / 3), rel=2e-5, abs=1e-6)
    near, near_gradient = sample(0.85 - 1e-4)
    assert 0 < near < unsafe * 1e-4
    assert 0 < -near_gradient < -gradient * 0.01

    low = q.clone()
    low[2] = 0.70 * loss.height
    low.requires_grad_()
    total, terms = _evaluate(loss, low)
    assert loss.weights["safety"] == 10.0
    other = sum(loss.weights[name] * value for name, value in terms.items() if name != "safety")
    weighted_safety = total - other
    torch.testing.assert_close(weighted_safety, 10 * terms["safety"], rtol=1e-5, atol=1e-5)
    weighted_gradient, = torch.autograd.grad(weighted_safety.sum(), low, retain_graph=True)
    safety_gradient, = torch.autograd.grad(terms["safety"].sum(), low)
    assert float(safety_gradient[2]) < 0
    torch.testing.assert_close(weighted_gradient, 10 * safety_gradient, rtol=1e-5, atol=1e-5)


def test_tilt_safety_is_continuous_and_restores_roll_and_pitch(go1, walking_loss_cls):
    task, dynamics = go1
    loss = walking_loss_cls(task, dynamics)
    q = _grounded(task)

    def sample(angle, axis):
        theta = q.new_tensor(angle, requires_grad=True)
        vector = q.new_zeros(3)
        vector[axis] = 1
        quaternion = torch.cat(((theta / 2).cos()[None], vector * (theta / 2).sin()))
        state = torch.cat((q[:3], quaternion, q[7:]))
        return _safety_and_gradient(loss, state, theta)

    boundary = math.acos(0.94)
    for axis in (0, 1):
        for sign in (-1, 1):
            assert sample(sign * (boundary - 1e-4), axis) == pytest.approx((0, 0), abs=1e-9)
            unsafe, gradient = sample(sign * 0.55, axis)
            assert unsafe > 0 and sign * gradient > 0
            near, near_gradient = sample(sign * (boundary + 1e-4), axis)
            assert 0 < near < unsafe * 1e-4
            assert 0 < sign * near_gradient < sign * gradient * 0.01


def test_go1_thirty_mm_clearance_has_a_bounded_action_witness(go1):
    task, _ = go1
    assert task.clearance == pytest.approx(0.03)
    q = _grounded(task)
    before, _ = _cpu_geometry(task, q)
    action = q.new_zeros(1, task.n)
    # Symmetric thigh/calf flexion raises the foot without changing its XY.
    # The continuous path from zero action to this tuck brackets 30 mm.
    for leg in ("FR", "FL", "RR", "RL"):
        action[0, task.joint_names.index(f"{leg}_thigh_joint")] = 0.4
        action[0, task.joint_names.index(f"{leg}_calf_joint")] = -0.8
    assert bool((action.abs() < 1).all())
    target = task.target(action)
    torch.testing.assert_close(target, task.q0[None] + task.scale * action, rtol=0, atol=1e-7)
    assert bool(((target >= task.limits[:, 0]) & (target <= task.limits[:, 1])).all())
    tucked = q.clone()
    tucked[7:] = target[0]
    torch.testing.assert_close(tucked[:7], q[:7], rtol=0, atol=0)
    after, _ = _cpu_geometry(task, tucked)
    np.testing.assert_allclose(after[:, :2], before[:, :2], rtol=0, atol=2e-6)
    lift = after[:, 2] - before[:, 2]
    assert np.all(lift >= task.clearance), f"Foot lifts {lift} cannot reach {task.clearance} m"
