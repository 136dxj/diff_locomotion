"""CPU MuJoCo checks of model-derived RSI; no GPU simulation or rollout.

Run from the integrated source directory:
    uv run --frozen python -B -m pytest -p no:cacheprovider tests/test_diff_gait_reference.py tests/test_diff_reference.py -q

Fixtures compile the project's actual EntityCfg and use its configured nominal
pose, action scales and clearance, without test-only clearance overrides.
Command corners cover Go1 vx=.1/.25, G1 vx=.1/.35, and all vy/yaw sign
combinations at +/-.1.
Infeasibility must be reported, never hidden by clipping the returned reference.
"""

from itertools import product
import math
from pathlib import Path
import re
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from src.diff_loco.model import RobotModel
from src.diff_loco.reference import GaitReference


@pytest.fixture(scope="module", params=["go1", "g1"])
def robot(request):
    from mjlab.entity.entity import Entity
    from src.diff_loco.task import robot_config

    path = Path(__file__).resolve().parents[1] / "src/assets/robots" / ("unitree_" + request.param)
    if not path.exists():
        pytest.skip(f"Requires the integrated {request.param} assets: {path}")
    cfg, feet, geoms, offsets, scale, clearance = robot_config(request.param)
    m = Entity(cfg).spec.compile()
    names = [m.joint(i).name for i in range(1, m.njnt)]
    assert m.nv - 6 == (29 if request.param == "g1" else 12)
    assert m.nkey, "Entity must compile its configured initial pose"
    q = torch.tensor(m.key_qpos[0], dtype=torch.float64)
    # Nonzero translation and yaw expose otherwise invisible frame mistakes.
    yaw = .57
    q[:3] += q.new_tensor([.31, -.27, .13])
    q[3:7] = q.new_tensor([math.cos(yaw/2), 0, 0, math.sin(yaw/2)])
    values = [next(v for pattern, v in scale.items() if re.fullmatch(pattern, name))
              for name in names] if isinstance(scale, dict) else [scale] * len(names)
    return SimpleNamespace(name=request.param, model=m, initial=q, feet=feet, geoms=geoms,
                           offsets=offsets, limits=torch.tensor(m.jnt_range[1:]),
                           scale=torch.tensor(values, dtype=torch.float64),
                           clearance=clearance, flat=len(feet) == 2)


def reference(robot, *, dtype=torch.float64, dt=.01, scale=None, clearance=None):
    model = RobotModel(robot.model, foot_sites=robot.feet, contact_geoms=robot.geoms).to(dtype=dtype)
    return GaitReference(model, robot.initial.to(dtype), robot.limits.to(dtype),
                         robot.scale.to(dtype) if scale is None else scale, robot.offsets,
                         .6, robot.clearance if clearance is None else clearance,
                         .65 if robot.flat else .6, flat_feet=robot.flat, dt=dt)


def geometry(robot, q):
    """Independent site poses and exact plane-support samples from MuJoCo."""
    m = robot.model
    d = mujoco.MjData(m)
    d.qpos[:] = np.asarray(q)
    mujoco.mj_forward(m, d)
    points, owners = [], []
    for name in robot.geoms:
        g = m.geom(name).id
        size, kind = m.geom_size[g], m.geom_type[g]
        if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            local, radius = np.zeros((1, 3)), size[0]
        elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            local, radius = np.array([[0, 0, -size[1]], [0, 0, size[1]]]), size[0]
        elif kind == mujoco.mjtGeom.mjGEOM_BOX:
            local, radius = np.array(list(product((-1, 1), repeat=3))) * size, 0
        else:
            raise AssertionError(f"Unexpected contact geometry: {name}, type {kind}")
        world = local @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        world[:, 2] -= radius
        points.extend(world)
        owners.extend([m.geom_bodyid[g]] * len(world))
    sites = [m.site(name).id for name in robot.feet]
    return d, d.site_xpos[sites].copy(), d.site_xmat[sites].reshape(-1, 3, 3).copy(), np.array(points), np.array(owners)


def world_targets(ref, q, phase, commands):
    target = ref.targets(phase, commands).detach().cpu().numpy().copy()
    for b, pose in enumerate(q):
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, np.asarray(pose[3:7]))
        target[b, :, :2] = target[b, :, :2] @ rotation.reshape(3, 3)[:2, :2].T + np.asarray(pose[:2])
    target[..., 2] += float(ref.height)
    return target


def continue_root(model, q, command, seconds):
    """Independent constant body planar twist, including curved translation."""
    q = np.asarray(q).copy()
    if np.linalg.norm(command) < .1:
        return q
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, q[3:7])
    angle = command[2] * seconds
    c, s = math.cos(angle/2), math.sin(angle/2)
    travel = seconds * np.sinc(angle/(2*math.pi))
    local = travel * np.array([c*command[0] - s*command[1],
                               s*command[0] + c*command[1], 0])
    q[:3] += rotation.reshape(3, 3) @ local
    velocity = np.zeros(model.nv)
    velocity[5] = command[2]
    mujoco.mj_integratePos(model, q, velocity, seconds)
    return q


def test_ground_height_and_standing_are_model_derived(robot):
    original = robot.initial.clone()
    ref = reference(robot)
    _, _, _, contacts, _ = geometry(robot, original)
    assert float(ref.height) == pytest.approx(float(original[2]) - contacts[:, 2].min(), abs=2e-8)
    torch.testing.assert_close(robot.initial, original, rtol=0, atol=0)
    commands = torch.zeros(4, 3, dtype=torch.float64)
    sample = ref.sample(commands)
    torch.testing.assert_close(sample["qvel"], torch.zeros_like(sample["qvel"]), rtol=0, atol=1e-12)
    torch.testing.assert_close(sample["action_history"], torch.zeros_like(sample["action_history"]), rtol=0, atol=1e-12)
    expected = world_targets(ref, sample["qpos"], sample["phase"], commands)
    for b, q in enumerate(sample["qpos"]):
        _, feet, _, points, _ = geometry(robot, q)
        np.testing.assert_allclose(feet, expected[b], rtol=0, atol=2e-8)
        assert points[:, 2].min() == pytest.approx(0, abs=2e-8)
        if robot.name == "go1":
            assert np.all(feet[:, 2] > .01), "Sphere-center foot sites must remain above ground"
    # A new global heading changes neither the body target nor its ground Z.
    turned = sample["qpos"][:1].clone()
    turned[0, 3:7] = turned.new_tensor([math.cos(-.41/2), 0, 0, math.sin(-.41/2)])
    _, feet, _, _, _ = geometry(robot, turned[0])
    np.testing.assert_allclose(feet, world_targets(ref, turned, sample["phase"][:1], commands[:1])[0], atol=2e-8)


def test_gate_periodicity_clearance_and_no_slip_stance_targets(robot):
    ref = reference(robot)
    phase = torch.tensor([.83, .83, .83, .83], dtype=torch.float64)
    commands = torch.tensor([[0, 0, 0], [.099, 0, 0], [0, 0, .099], [0, 0, .1]], dtype=torch.float64)
    target = ref.targets(phase, commands)
    torch.testing.assert_close(target[:3], target[:1].expand(3, -1, -1), rtol=0, atol=0)
    assert not torch.equal(target[2], target[3]), "Yaw-only norm=.1 must enable gait"
    torch.testing.assert_close(ref.targets(phase + 1, commands), target, rtol=0, atol=1e-14)
    peak = torch.tensor([(1 + ref.duty)/2 - robot.offsets[0]], dtype=torch.float64)
    lift = ref.targets(peak, commands[3:])[0, 0, 2] - ref.targets(peak, commands[:1])[0, 0, 2]
    assert float(lift) == pytest.approx(robot.clearance, abs=1e-12)
    for boundary in (ref.duty, 1.):
        eps = 1e-7
        before = ref.targets(torch.tensor([boundary-eps]), commands[3:])
        after = ref.targets(torch.tensor([boundary+eps]), commands[3:])
        torch.testing.assert_close(before, after, rtol=0, atol=1e-7)
    # Actual world positions of an interior stance must stay fixed under a
    # moving, turning root. This checks the command frame and yaw contribution.
    phase = torch.tensor([.07], dtype=torch.float64)
    command = torch.tensor([[.12, -.04, .09]], dtype=torch.float64)
    h = 1e-4
    q = ref.initial.detach().numpy()
    pair = np.array([continue_root(robot.model, q, command[0].numpy(), t) for t in (-h, h)])
    phases = phase + torch.tensor([-h, h], dtype=torch.float64) / ref.period
    worlds = world_targets(ref, pair, phases, command.expand(2, -1))
    stance = ((float(phase[0]) + np.array(robot.offsets)) % 1) < ref.duty - .01
    assert stance.any()
    np.testing.assert_allclose((worlds[1] - worlds[0])[stance] / (2*h), 0, rtol=0, atol=1e-8)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("speed,vy,yaw", list(product(("low", "high"), (-.1, .1), (-.1, .1))))
def test_actual_robot_bounded_ik_matches_independent_fk_and_flat_contacts(robot, dtype, speed, vy, yaw):
    ref = reference(robot, dtype=dtype)
    phase = torch.arange(32, dtype=dtype) / 32
    # Include each foot's swing onset, exact peak lift and cycle wrap as well
    # as the uniform grid; the hardest clearance can fall between grid points.
    critical = ref.initial.new_tensor([ref.duty, (1 + ref.duty)/2, 1 - 1e-5])
    critical = (critical[:, None] - ref.offsets[None]) % 1
    phase = torch.cat((phase, critical.flatten())).unique()
    vx = .1 if speed == "low" else {"go1": .25, "g1": .35}[robot.name]
    command = ref.initial.new_tensor([vx, vy, yaw]).expand(len(phase), -1)
    q = ref._configuration(phase, command)
    expected = world_targets(ref, q, phase, command)
    raw = (q[:, 7:] - ref.initial[7:]) / ref.scale
    assert raw.abs().max() <= 1 + 1e-6
    assert (q[:, 7:] >= robot.limits[:, 0] - 1e-6).all()
    assert (q[:, 7:] <= robot.limits[:, 1] + 1e-6).all()
    inactive = ~ref.dynamics.ancestry[ref.dynamics.foot_body_ids].any(0)
    torch.testing.assert_close(q[:, 7:][:, inactive], ref.initial[7:][inactive].expand(len(q), -1), rtol=0, atol=0)
    for b, pose in enumerate(q):
        _, feet, rotations, points, owners = geometry(robot, pose)
        np.testing.assert_allclose(feet, expected[b], rtol=0, atol=ref.position_tolerance)
        assert points[:, 2].min() >= -2 * ref.position_tolerance
        if robot.flat:
            np.testing.assert_allclose(rotations[:, :, 2], np.tile([0, 0, 1], (2, 1)), atol=ref.rotation_tolerance)
            # Every sampled sole point, not just the site, shares its foot's height.
            for f, name in enumerate(robot.feet):
                body = robot.model.site_bodyid[robot.model.site(name).id]
                sole = points[owners == body, 2]
                assert np.ptp(sole) < 5e-5
                assert abs(sole.min() - feet[f, 2]) < 5e-5


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_go1_ten_centimeter_swing_is_reachable(robot, dtype):
    if robot.name != "go1":
        pytest.skip("10 cm swing is configured for Go1")
    ref = reference(robot, dtype=dtype, clearance=.1, scale=.6)
    phase = torch.unique(torch.cat((torch.arange(120, dtype=dtype) / 120,
                                    ref.initial.new_tensor([.3, .8]))))
    corners = ref.initial.new_tensor(list(product((.1, .25), (-.1, .1), (-.1, .1))))
    commands = corners.repeat_interleave(len(phase), 0)
    phase = phase.repeat(len(corners))
    q = ref._configuration(phase, commands)
    expected = world_targets(ref, q, phase, commands)
    assert ((q[:, 7:] - ref.initial[7:]) / ref.scale).abs().max() < .95
    for pose, target in zip(q, expected):
        _, feet, _, points, _ = geometry(robot, pose)
        np.testing.assert_allclose(feet, target, rtol=0, atol=ref.position_tolerance)
        assert points[:, 2].min() >= -2 * ref.position_tolerance
    peak = ref.targets(ref.initial.new_tensor([.8]), corners[:1])[0, 0, 2] - ref.nominal[0, 2]
    assert float(peak) == pytest.approx(.1, abs=1e-7)
    sample = ref.sample(corners.repeat(16, 1))
    assert all(torch.isfinite(value).all() for value in sample.values())
    assert sample["action_history"].abs().max() < 1


def test_sample_velocity_and_delay_history_follow_one_periodic_path(robot):
    ref = reference(robot)
    commands = torch.tensor([[.1, .01, .03], [-.1, -.01, -.03], [0, 0, .1],
                             [0, 0, 0], [.099, 0, 0]], dtype=torch.float64)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(31)
        state = ref.sample(commands)
    assert set(state) == {"qpos", "qvel", "phase", "action_history"}
    assert all(torch.isfinite(value).all() and not value.requires_grad for value in state.values())
    assert ((state["phase"] >= 0) & (state["phase"] < 1)).all()
    assert state["action_history"].abs().max() <= 1
    previous = ref._configuration((state["phase"] - ref.dt/ref.period) % 1, commands)
    future = ref._configuration((state["phase"] + ref.dt/ref.period) % 1, commands)
    older = ref._configuration((state["phase"] - 2*ref.dt/ref.period) % 1, commands)
    for index, poses in enumerate((previous, older)):
        reconstructed = ref.initial[7:] + ref.scale * state["action_history"][:, index]
        torch.testing.assert_close(reconstructed, poses[:, 7:], rtol=0, atol=2e-5)
        for b in range(len(commands)):
            q = state["qpos"][b].clone()
            q[7:] = reconstructed[b]
            _, feet, _, _, _ = geometry(robot, q)
            phase = (state["phase"][b:b+1] - (index+1)*ref.dt/ref.period) % 1
            np.testing.assert_allclose(feet, world_targets(ref, q[None], phase, commands[b:b+1])[0], atol=2e-5)
    for b, command in enumerate(commands.numpy()):
        minus = continue_root(robot.model, previous[b], command, -ref.dt)
        plus = continue_root(robot.model, future[b], command, ref.dt)
        expected = np.zeros(robot.model.nv)
        mujoco.mj_differentiatePos(robot.model, expected, 2*ref.dt, minus, plus)
        np.testing.assert_allclose(state["qvel"][b], expected, rtol=0, atol=2e-3)
        # Check physical site velocity using MuJoCo Jacobians, independently
        # of the surrogate Jacobian used by IK (central FD has O(dt^2) error).
        data, _, _, _, _ = geometry(robot, state["qpos"][b])
        _, foot_minus, _, _, _ = geometry(robot, minus)
        _, foot_plus, _, _, _ = geometry(robot, plus)
        for f, name in enumerate(robot.feet):
            jacp, jacr = np.zeros((3, robot.model.nv)), np.zeros((3, robot.model.nv))
            mujoco.mj_jacSite(robot.model, data, jacp, jacr, robot.model.site(name).id)
            np.testing.assert_allclose(jacp @ state["qvel"][b].numpy(),
                                       (foot_plus[f] - foot_minus[f])/(2*ref.dt), atol=4e-3, rtol=.02)
    assert state["qvel"][:3, 6:].abs().max() > .01
    torch.testing.assert_close(state["qvel"][3:], torch.zeros_like(state["qvel"][3:]), rtol=0, atol=1e-12)
    # Mutating outputs cannot corrupt future references or the caller's q0.
    state["qpos"].zero_()
    state["action_history"].fill_(.9)
    standing = ref.sample(torch.zeros(1, 3))
    torch.testing.assert_close(standing["qpos"][0], ref.initial, rtol=0, atol=1e-12)


def test_infeasible_reference_is_rejected_and_empty_batch_supported(robot):
    ref = reference(robot, scale=torch.zeros_like(robot.scale))
    phase = ref.initial.new_tensor([(1 + ref.duty)/2 - robot.offsets[0]])
    with pytest.raises(ValueError, match="Gait IK is unreachable.*residual"):
        ref._configuration(phase, ref.initial.new_tensor([[.1, 0, 0]]))
    empty = ref.sample(torch.empty(0, 3))
    assert {key: tuple(value.shape) for key, value in empty.items()} == {
        "qpos": (0, robot.model.nq), "qvel": (0, robot.model.nv),
        "phase": (0,), "action_history": (0, 2, len(robot.scale)),
    }
    with pytest.raises(ValueError, match="finite"):
        ref.sample(torch.tensor([[float("nan"), 0, 0]]))
