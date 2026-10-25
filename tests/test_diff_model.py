"""Numerical contracts for the portable surrogate; no mjlab import is needed.

Run from the repository root with:
    python -B -m pytest -p no:cacheprovider tests/test_diff_model.py -q
Robot-asset cases skip only when the corresponding asset is not installed.
"""

from pathlib import Path

import mujoco
import numpy as np
import pytest
import torch

from src.diff_loco.model import RobotModel


TREE_XML = """
<mujoco>
  <compiler angle="radian"/>
  <option gravity="0 0 -9.81"/>
  <default><joint armature="0.015" damping="0.07"/></default>
  <worldbody>
    <body name="fixed" pos="0.4 0.2 0.1" quat="0.9 0.1 0.2 0.3">
      <geom type="box" size="0.1 0.1 0.1"/>
    </body>
    <body name="base" pos="0.1 -0.2 0.9" quat="0.9 0.1 -0.2 0.3">
      <freejoint/>
      <inertial pos="0.02 -0.03 0.04" quat="0.9 0.2 0.1 -0.2" mass="5" diaginertia="0.3 0.4 0.5"/>
      <body name="leg" pos="0.1 0.08 -0.1" quat="0.96 0.1 0.2 0.1">
        <joint name="hip" type="hinge" pos="0.03 -0.02 0.01" axis="0.2 0.9 0.3" ref="0.2"/>
        <joint name="extension" type="slide" axis="0.3 0.1 0.8" ref="0.03"/>
        <inertial pos="0.02 0 -0.1" quat="0.9 0.2 0.3 0.1" mass="1.2" diaginertia="0.03 0.04 0.05"/>
        <body name="ankle" pos="0.15 0 -0.25" quat="0.95 0.1 -0.2 0.1">
          <joint name="ankle_roll" axis="1 0.2 -0.1" pos="0.02 0.01 0.03" ref="-0.15"/>
          <inertial pos="0.04 0 -0.03" mass="0.8" diaginertia="0.02 0.03 0.04"/>
          <body name="sole_frame" pos="0.03 -0.01 -0.02" quat="0.97 0.1 0.05 0.15">
            <inertial pos="0.02 0 -0.01" mass="0.1" diaginertia="0.001 0.002 0.002"/>
            <site name="foot" pos="0.04 0 -0.05" quat="0.9 0.1 0.2 -0.1"/>
            <geom name="sole" type="box" pos="0.04 0 -0.03" size="0.1 0.05 0.02" friction="0.7 0.01 0.001"/>
            <geom name="capsule" type="capsule" fromto="-0.06 0 -0.03 0.12 0 -0.03" size="0.02"/>
            <geom name="sphere" type="sphere" pos="0.04 0 -0.04" size="0.02"/>
          </body>
        </body>
      </body>
      <body name="arm" pos="-0.1 0.2 0.2" quat="0.95 0.2 -0.1 0.1">
        <joint name="arm_slide" type="slide" axis="0.2 0.9 0.3" ref="-0.02"/>
        <inertial pos="0.1 0.02 0" quat="0.9 -0.1 0.2 0.1" mass="0.6" diaginertia="0.01 0.02 0.025"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _tree():
    return mujoco.MjModel.from_xml_string(TREE_XML)


def _states(m, batch=3):
    rng = np.random.default_rng(7)
    q = np.broadcast_to(m.qpos0, (batch, m.nq)).copy()
    for i in range(batch):
        tangent = rng.normal(0, 0.25, m.nv)
        mujoco.mj_integratePos(m, q[i], tangent, 1.0)
    return torch.tensor(q, dtype=torch.float64)


def _dense_mass(m, data):
    result = np.empty((m.nv, m.nv))
    try:  # MuJoCo >= 3.4 takes MjData; older releases take the packed array.
        mujoco.mj_fullM(m, data, result)
    except TypeError:
        mujoco.mj_fullM(m, result, data.qM)
    return result


def _compare_mujoco(m, robot, q):
    k = robot.kinematics(q)
    mass, gravity = robot.mass_matrix(q), robot.gravity(q)
    assert torch.linalg.cholesky_ex(mass).info.eq(0).all()
    torch.testing.assert_close(mass, mass.transpose(-1, -2), rtol=1e-12, atol=1e-12)
    for b, row in enumerate(q.numpy()):
        data = mujoco.MjData(m)
        data.qpos[:] = row
        mujoco.mj_forward(m, data)
        np.testing.assert_allclose(k["body_pos"][b], data.xpos, atol=2e-10, rtol=2e-10)
        np.testing.assert_allclose(k["body_rot"][b], data.xmat.reshape(-1, 3, 3), atol=2e-10, rtol=2e-10)
        np.testing.assert_allclose(k["com_pos"][b], data.xipos, atol=2e-10, rtol=2e-10)
        np.testing.assert_allclose(mass[b], _dense_mass(m, data), atol=2e-9, rtol=2e-9)
        np.testing.assert_allclose(gravity[b], -data.qfrc_bias, atol=2e-9, rtol=2e-9)
        for f, name in enumerate(robot.foot_sites):
            sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, name)
            np.testing.assert_allclose(k["foot_pos"][b, f], data.site_xpos[sid], atol=2e-10, rtol=2e-10)
            np.testing.assert_allclose(k["foot_rot"][b, f], data.site_xmat[sid].reshape(3, 3), atol=2e-10, rtol=2e-10)
            jac = np.zeros((3, m.nv))
            mujoco.mj_jacSite(m, data, jac, None, sid)
            np.testing.assert_allclose(k["foot_jac"][b, f], jac, atol=2e-10, rtol=2e-10)
        for c, body in enumerate(robot.contact_body_ids.tolist()):
            jac = np.zeros((3, m.nv))
            mujoco.mj_jac(m, data, jac, None, k["contact_pos"][b, c].numpy(), body)
            np.testing.assert_allclose(k["contact_jac"][b, c], jac, atol=2e-10, rtol=2e-10)


def test_tree_fk_jacobians_mass_gravity_match_mujoco():
    m = _tree()
    robot = RobotModel(m, foot_sites=["foot"])
    assert robot.num_joints == 4
    assert robot.num_contacts == 11  # eight corners, two cap supports, one sphere
    _compare_mujoco(m, robot, _states(m))


def test_material_point_jacobians_match_autograd_in_velocity_coordinates():
    m = _tree()
    robot = RobotModel(m, foot_sites=["foot"], contact_geoms=["sole"])
    q = _states(m, 2)
    direction = torch.linspace(-0.3, 0.5, 2 * m.nv, dtype=q.dtype).reshape(2, m.nv)
    def positions(tangent):
        k = robot.kinematics(robot.integrate(q, tangent, 1.0))
        return torch.cat((k["foot_pos"], k["contact_pos"]), 1)
    _, numerical = torch.autograd.functional.jvp(positions, torch.zeros_like(direction), direction)
    k = robot.kinematics(q)
    jac = torch.cat((k["foot_jac"], k["contact_jac"]), 1)
    expected = (jac @ direction[:, None, :, None]).squeeze(-1)
    torch.testing.assert_close(numerical, expected, rtol=2e-10, atol=2e-10)


def test_integrate_matches_mujoco_and_has_finite_zero_rotation_gradient():
    m = _tree()
    robot = RobotModel(m, foot_sites=[], contact_geoms=[])
    q = _states(m, 2)
    velocity = torch.linspace(-1, 1, 2 * m.nv, dtype=q.dtype).reshape(2, m.nv)
    expected = q.numpy().copy()
    for b in range(2):
        mujoco.mj_integratePos(m, expected[b], velocity[b].numpy(), 0.03)
    np.testing.assert_allclose(robot.integrate(q, velocity, 0.03), expected, rtol=2e-12, atol=2e-12)
    zero = torch.zeros_like(velocity, requires_grad=True)
    derivative, = torch.autograd.grad(robot.integrate(q, zero, 0.01).sum(), zero)
    assert torch.isfinite(derivative).all()


def test_absolute_dr_arrays_match_modified_mujoco_and_broadcast():
    m = _tree()
    robot = RobotModel(m, foot_sites=["foot"], contact_geoms=[])
    q = _states(m, 2)
    p = {}
    for name in ("body_mass", "body_inertia", "body_ipos", "dof_armature", "dof_damping"):
        value = torch.tensor(np.array(getattr(m, name)), dtype=q.dtype)
        p[name] = value.unsqueeze(0).repeat((2,) + (1,) * value.ndim)
    p["body_mass"][1] *= 1.3
    p["body_inertia"][1] *= 1.17
    p["body_ipos"][1, 2:] += torch.tensor([0.02, -0.01, 0.015])
    p["dof_armature"][1] += 0.01
    p["dof_damping"][1] += 0.12
    matrices, gravity = robot.mass_matrix(q, parameters=p), robot.gravity(q, parameters=p)
    for b in range(2):
        modified = _tree()
        for name, value in p.items():
            getattr(modified, name)[:] = value[b].numpy()
        data = mujoco.MjData(modified)
        mujoco.mj_setConst(modified, data)
        data.qpos[:] = q[b].numpy()
        mujoco.mj_forward(modified, data)
        np.testing.assert_allclose(matrices[b], _dense_mass(modified, data), rtol=3e-9, atol=3e-9)
        np.testing.assert_allclose(gravity[b], -data.qfrc_bias, rtol=3e-9, atol=3e-9)
    scale = torch.tensor([[0.8], [1.4]], dtype=q.dtype, requires_grad=True)
    offset = torch.tensor([[0.01, -0.02, 0.03], [-0.01, 0.02, 0.04]], dtype=q.dtype, requires_grad=True)
    combined = {**p, "mass_scale": scale, "com_offset": offset}
    absolute = dict(p)
    absolute["body_mass"] = p["body_mass"] * scale
    absolute["body_inertia"] = p["body_inertia"] * scale[..., None]
    absolute["body_ipos"] = p["body_ipos"] + offset[:, None] * robot.root_mask[None, :, None]
    torch.testing.assert_close(robot.mass_matrix(q, parameters=combined), robot.mass_matrix(q, parameters=absolute))
    value = robot.mass_matrix(q, parameters=combined).square().sum() + robot.gravity(q, parameters=combined).square().sum()
    gradients = torch.autograd.grad(value, (scale, offset))
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients)
    with pytest.raises(ValueError, match="Unsupported dynamics parameters"):
        robot.step(q, q.new_zeros(2, m.nv), q[:, 7:], 1, 1, 1, 0.01, parameters={"body_mas": 1})


def test_implicit_pd_and_damping_balance_momentum_and_respect_effort():
    m = _tree()
    robot = RobotModel(m, foot_sites=[], contact_geoms=[])
    q = _states(m, 2)
    velocity = torch.linspace(-0.2, 0.2, 2 * m.nv, dtype=q.dtype).reshape(2, m.nv)
    dt = 0.02
    mass = robot.mass_matrix(q)
    damping = torch.linspace(0.1, 0.4, m.nv, dtype=q.dtype).expand(2, -1)
    target = q[:, 7:] + 100
    effort = torch.tensor([0.1, 0.3, 0.0, 0.2], dtype=q.dtype)
    _, next_velocity = robot.step(q, velocity, target, 500, 12, effort, dt, parameters={"dof_damping": damping})
    recovered = (mass @ (next_velocity - velocity)[..., None]).squeeze(-1) / dt
    recovered += damping * next_velocity + 1e-9 * next_velocity / dt - robot.gravity(q)
    torch.testing.assert_close(recovered[:, :6], torch.zeros_like(recovered[:, :6]), atol=1e-10, rtol=0)
    assert torch.all(recovered[:, 6:].abs() <= effort + 1e-9)
    torch.testing.assert_close(recovered[:, 8], torch.zeros(2, dtype=q.dtype), atol=1e-10, rtol=0)
    # Infinite effort recovers the exact linear implicit-PD solution.
    kp, kd = 50.0, 3.0
    target = q[:, 7:] + 0.1
    _, actual = robot.step(q, velocity, target, kp, kd, float("inf"), dt, parameters={"dof_damping": damping})
    stiff = torch.cat((q.new_zeros(2, 6), q.new_full((2, robot.num_joints), dt * kd + dt**2 * kp)), -1)
    matrix = mass + torch.diag_embed(dt * damping + 1e-9 + stiff)
    rhs = (mass @ velocity[..., None]).squeeze(-1) + dt * robot.gravity(q)
    rhs += dt * torch.cat((q.new_zeros(2, 6), kp * (target - q[:, 7:])), -1)
    expected = torch.linalg.solve(matrix, rhs[..., None]).squeeze(-1)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)


def test_contact_cones_support_and_joint_moments():
    m = _tree()
    robot = RobotModel(m, foot_sites=["foot"], contact_geoms=["sole"])
    q = _states(m, 2)
    q[:, 2] -= robot.kinematics(q)["contact_pos"][..., 2].min(-1).values + 0.002
    k = robot.kinematics(q)
    matrix = robot.mass_matrix(q)
    jac = k["contact_jac"]
    jt = jac.flatten(1, 2).transpose(-1, -2)
    response = torch.linalg.solve(matrix, jt).reshape(2, m.nv, robot.num_contacts, 3).transpose(1, 2)
    velocity = q.new_zeros(2, m.nv)
    velocity[:, :3] = q.new_tensor([0.4, -0.2, -0.5])
    friction = q.new_tensor([[0.0], [0.6]]).expand(2, robot.num_contacts)
    impulse = robot._contact_impulses(velocity, jac, response, k["contact_pos"], friction, q.new_tensor([[0.01]]))
    assert torch.isfinite(impulse).all()
    assert (impulse[..., 2] >= 0).all()
    assert (torch.linalg.vector_norm(impulse[..., :2], dim=-1) <= friction * impulse[..., 2] + 1e-12).all()
    generalized = (jt @ impulse.flatten(1)[..., None]).squeeze(-1)
    assert (generalized[:, 2] > 0).all()
    assert (generalized[:, 8].abs() > 1e-4).all()  # ankle hinge moment
    free = RobotModel(m, foot_sites=["foot"], contact_geoms=[])
    args = (q, velocity, q[:, 7:], 5, 0.3, 10, 0.01)
    _, contact_velocity = robot.step(*args, parameters={"friction": friction})
    _, free_velocity = free.step(*args)
    assert torch.linalg.vector_norm(contact_velocity[:, :6] - free_velocity[:, :6]) > 0.01
    assert torch.linalg.vector_norm(contact_velocity[:, 6:] - free_velocity[:, 6:]) > 0.01


def test_contact_step_gradcheck_state_actions_and_dr():
    m = _tree()
    robot = RobotModel(m, foot_sites=["foot"], contact_geoms=["sole"])
    q = _states(m, 1)
    q[:, 2] -= robot.kinematics(q)["contact_pos"][..., 2].min(-1).values + 0.003
    velocity = q.new_zeros(1, m.nv)
    velocity[:, 0], velocity[:, 2] = 0.15, -0.1
    target = (q[:, 7:] + 0.025).requires_grad_()
    friction = q.new_tensor([[0.6]], requires_grad=True)
    mass_scale = q.new_tensor([[1.1]], requires_grad=True)
    tangent = torch.zeros_like(velocity, requires_grad=True)
    def step(dq, action, mu, scale):
        state = robot.integrate(q, dq, 1.0)
        return robot.step(state, velocity, action, 12, 1, 20, 0.005,
                          parameters={"friction": mu, "mass_scale": scale})
    assert torch.autograd.gradcheck(step, (tangent, target, friction, mass_scale),
                                    eps=1e-6, atol=2e-5, rtol=3e-4, fast_mode=True)


def test_batched_steps_equal_individual_steps_and_all_actions_have_gradients():
    m = _tree()
    robot = RobotModel(m, foot_sites=["foot"])
    q = _states(m, 3)
    q[:, 2] -= robot.kinematics(q)["contact_pos"][..., 2].min(-1).values
    velocity = q.new_zeros(3, m.nv)
    target = (q[:, 7:] + 0.02).requires_grad_()
    p = {"mass_scale": q.new_tensor([[0.8], [1.0], [1.2]]),
         "friction": q.new_tensor([[0.2], [0.5], [0.8]])}
    result = robot.step(q, velocity, target, 15, 0.8, 30, 0.005, parameters=p)
    for b in range(3):
        individual = robot.step(q[b:b+1], velocity[b:b+1], target[b:b+1], 15, 0.8, 30, 0.005,
                                parameters={key: value[b:b+1] for key, value in p.items()})
        for all_values, one in zip(result, individual):
            torch.testing.assert_close(all_values[b:b+1], one, rtol=1e-9, atol=1e-9)
    grad, = torch.autograd.grad(result[1][:, 6:].square().sum(), target)
    assert torch.isfinite(grad).all() and (grad.abs() > 1e-9).all()


def test_site_fallback_and_unsupported_topology():
    m = mujoco.MjModel.from_xml_string(TREE_XML.replace('name="sole"', 'name="sole" contype="0" conaffinity="0"')
                                      .replace('name="capsule"', 'name="capsule" contype="0" conaffinity="0"')
                                      .replace('name="sphere"', 'name="sphere" contype="0" conaffinity="0"'))
    robot = RobotModel(m, foot_sites=["foot"])
    k = robot.kinematics(_states(m, 1))
    assert robot.num_contacts == 1
    torch.testing.assert_close(k["foot_pos"], k["contact_pos"])
    torch.testing.assert_close(k["foot_jac"], k["contact_jac"])
    with pytest.raises(ValueError, match="Unknown"):
        RobotModel(m, foot_sites=["missing"])
    ball = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><freejoint/><geom size=".1"/><body><joint type="ball"/><geom size=".1"/></body></body></worldbody></mujoco>')
    with pytest.raises(ValueError, match="hinge/slide"):
        RobotModel(ball, foot_sites=[])


@pytest.mark.parametrize("name,joints,sites,geoms", [
    ("unitree_g1", 29, ["left_foot", "right_foot"], None),
    ("unitree_go1", 12, ["FR", "FL", "RR", "RL"],
     [f"{leg}_foot_collision" for leg in ("FR", "FL", "RR", "RL")]),
])
def test_robot_assets_fk_mass_and_every_action_gradient(name, joints, sites, geoms):
    root = Path(__file__).resolve().parents[1]
    path = root / "src" / "assets" / "robots" / name / "xmls" / (name.removeprefix("unitree_") + ".xml")
    if not path.exists():
        pytest.skip(f"Robot asset not installed: {path}")
    m = mujoco.MjModel.from_xml_path(str(path))
    robot = RobotModel(str(path), foot_sites=sites, contact_geoms=geoms)
    assert robot.num_joints == joints
    q = _states(m, 2)
    _compare_mujoco(m, robot, q)
    # A sole patch must create a nonzero ankle moment from a normal force.
    if name == "unitree_g1":
        assert robot.num_contacts >= 4
        k = robot.kinematics(q)
        for joint in ("left_ankle_pitch_joint", "left_ankle_roll_joint",
                      "right_ankle_pitch_joint", "right_ankle_roll_joint"):
            index = robot.joint_names.index(joint) + 6
            assert k["contact_jac"][..., 2, index].abs().max() > 0.01
    for dtype in (torch.float64, torch.float32):
        state = q.to(dtype)
        state[:, 2] -= robot.kinematics(state)["contact_pos"][..., 2].min(-1).values
        velocity = state.new_zeros(2, robot.nv)
        target = (state[:, 7:] + 0.01).requires_grad_()
        mass = robot.mass_matrix(state)
        assert torch.linalg.cholesky_ex(mass).info.eq(0).all()
        # Check a short rollout, not just a shape or a single forward pass.
        for _ in range(3):
            state, velocity = robot.step(state, velocity, target, 10, 1, 40, 0.002)
        grad, = torch.autograd.grad(velocity[:, 6:].square().sum() + state[:, 7:].square().sum(), target)
        assert torch.isfinite(state).all() and torch.isfinite(velocity).all()
        assert torch.isfinite(grad).all() and (grad.abs() > 1e-8).all()
        torch.testing.assert_close(state[:, 3:7].norm(dim=-1), torch.ones(2, dtype=dtype))
