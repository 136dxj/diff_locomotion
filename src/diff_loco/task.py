"""Thin mjlab adapter; robot semantics live here, never in the dynamics model."""

from dataclasses import dataclass, replace
import math
from pathlib import Path
import re

import mujoco
import numpy as np
import torch

from .policy import observe, quat_inverse_rotate


@dataclass
class TaskCfg:
    robot: str = "go1"
    num_envs: int = 32
    device: str = "cuda:0"
    dt: float = 0.01
    substeps: int = 5
    randomize: bool = True
    episode_seconds: float = 10.0
    period: float = 0.6
    max_speed: float = 0.25
    rsi_probability: float = 0.0
    clearance: float | None = None
    action_scale: float | None = None

    def __post_init__(self):
        if self.clearance is not None and (not math.isfinite(self.clearance) or self.clearance < 0):
            raise ValueError("clearance must be finite and nonnegative")
        if self.action_scale is not None and (not math.isfinite(self.action_scale) or self.action_scale <= 0):
            raise ValueError("action_scale must be finite and positive")


def robot_config(name):
    if name == "g1":
        from src.assets.robots.unitree_g1.g1_constants import get_g1_robot_cfg, G1_ACTION_SCALE
        entity = get_g1_robot_cfg()
        # A modest crouch leaves IK clearance inside the existing action bounds.
        entity.init_state = replace(entity.init_state, joint_pos={**entity.init_state.joint_pos,
            ".*_hip_pitch_joint": -0.3, ".*_knee_joint": 0.6, ".*_ankle_pitch_joint": -0.3})
        return (entity, ["left_foot", "right_foot"],
                [f"{s}_foot{i}_collision" for s in ("left", "right") for i in range(1, 8)],
                [0.0, 0.5], G1_ACTION_SCALE, 0.03)
    if name == "go1":
        from src.assets.robots.unitree_go1.go1_constants import get_go1_robot_cfg, GO1_ACTION_SCALE
        return (get_go1_robot_cfg(), ["FR", "FL", "RR", "RL"],
                [f"{s}_foot_collision" for s in ("FR", "FL", "RR", "RL")],
                [0.0, 0.5, 0.5, 0.0], GO1_ACTION_SCALE, 0.03)
    raise ValueError(f"Unknown robot {name!r}; add only its EntityCfg and foot/gait metadata here.")


class LocomotionTask:
    def __init__(self, cfg: TaskCfg):
        from mjlab.scene import Scene, SceneCfg
        from mjlab.sim import Simulation, SimulationCfg, MujocoCfg
        from mjlab.terrains import TerrainEntityCfg

        self.cfg, self.device = cfg, cfg.device
        entity, feet, geoms, offsets, scale, self.clearance = robot_config(cfg.robot)
        if cfg.clearance is not None:
            self.clearance = cfg.clearance
        if cfg.action_scale is not None:
            scale = cfg.action_scale
        self.foot_sites = ["robot/" + name for name in feet]
        self.contact_geoms = ["robot/" + name for name in geoms]
        self.offsets = self.tensor(offsets)
        self.scene = Scene(SceneCfg(num_envs=cfg.num_envs, entities={"robot": entity},
                           terrain=TerrainEntityCfg(terrain_type="plane")), device=cfg.device)
        self.sim = Simulation(num_envs=cfg.num_envs, model=self.scene.compile(), device=cfg.device,
                              cfg=SimulationCfg(nconmax=64, njmax=256,
                                  mujoco=MujocoCfg(timestep=cfg.dt / cfg.substeps,
                                                  iterations=10, ls_iterations=10)))
        self.scene.initialize(mj_model=self.sim.mj_model, model=self.sim.model, data=self.sim.data)
        m = self.sim.mj_model
        self.n = m.nv - 6
        # The model adapter is intentionally strict about unsupported transmissions.
        if m.nq != 7 + self.n or m.nu != self.n or m.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("Expected one free root and one position actuator per scalar joint")
        aj = m.actuator_trnid[:, 0]
        if (set(m.jnt_qposadr[aj]) != set(range(7, m.nq))
                or not np.all(m.actuator_gear[:, 0] == 1)
                or not np.all(m.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)):
            raise ValueError("Only bijective, unit-gear joint actuators are supported")
        self.actuator_joint = torch.tensor(m.jnt_qposadr[aj] - 7, device=self.device)
        self.joint_actuator = torch.argsort(self.actuator_joint)
        self.joint_names = [m.joint(i).name.removeprefix("robot/") for i in range(1, m.njnt)]
        self.initial = self.tensor(m.key_qpos[0] if m.nkey else m.qpos0)
        self.q0 = self.initial[7:]
        self.height = float(self.initial[2])
        self.limits = self.tensor(m.jnt_range[1:])
        self.contact_ids = torch.tensor([m.geom(name).id for name in self.contact_geoms], device=self.device)
        self.scale = self.tensor([next((v for k, v in scale.items() if re.fullmatch(k, name)), 0.25)
                                  for name in self.joint_names] if isinstance(scale, dict)
                                 else [scale] * self.n)
        self.fields = ("body_mass", "body_inertia", "body_ipos", "dof_armature",
                       "dof_damping", "geom_friction", "actuator_gainprm",
                       "actuator_biasprm", "actuator_forcerange")
        # set_const writes these too; shared buffers race across randomized worlds.
        derived = ("body_subtreemass", "dof_invweight0", "body_invweight0",
                   "tendon_length0", "tendon_invweight0")
        self.sim.expand_model_fields(self.fields + derived)
        self.defaults = {f: self.tensor(getattr(m, f)) for f in self.fields}
        self.age = torch.zeros(cfg.num_envs, dtype=torch.long, device=self.device)
        self.commands = torch.zeros(cfg.num_envs, 3, device=self.device)
        self.encoder_bias = torch.zeros(cfg.num_envs, self.n, device=self.device)
        self.delay = torch.zeros(cfg.num_envs, dtype=torch.long, device=self.device)
        self.phase_offset = torch.zeros(cfg.num_envs, device=self.device)
        self.initial_actions = torch.zeros(cfg.num_envs, 2, self.n, device=self.device)
        self.from_reference = torch.zeros(cfg.num_envs, dtype=torch.bool, device=self.device)
        self.reference = None  # Training installs the model-derived gait after model creation.
        self.command_scale = 1.0
        self.reset(torch.arange(cfg.num_envs, device=self.device))

    def tensor(self, x):
        return torch.as_tensor(np.array(x, copy=True), device=self.device, dtype=torch.float32)

    def install_reference(self, reference):
        """Use the same contact-derived standing height for RSI and ordinary resets."""
        self.reference = reference
        self.initial[2] = reference.height
        self.height = float(reference.height)

    def uniform(self, shape, low, high):
        return low + (high - low) * torch.rand(shape, device=self.device)

    @torch.no_grad()
    def randomize(self, ids):
        from mjlab.managers.event_manager import RecomputeLevel

        b, m = len(ids), self.sim.model
        for f, nominal in self.defaults.items():
            getattr(m, f)[ids] = nominal
        self.encoder_bias[ids] = 0
        self.delay[ids] = 0
        if self.cfg.randomize:
            mass_scale = self.uniform((b, self.sim.mj_model.nbody), 0.9, 1.1)
            m.body_mass[ids] *= mass_scale
            m.body_inertia[ids] *= mass_scale[..., None]
            base_id = self.sim.mj_model.jnt_bodyid[0]
            m.body_ipos[ids, base_id] += self.uniform((b, 3), -0.015, 0.015)
            m.dof_armature[ids] *= self.uniform((b, 1), 0.9, 1.1)
            m.dof_damping[ids] *= self.uniform((b, 1), 0.8, 1.2)
            m.geom_friction[ids, :, 0] = self.uniform((b, 1), 0.5, 1.25)
            kp_scale, kd_scale = self.uniform((b, self.n), 0.8, 1.2), self.uniform((b, self.n), 0.8, 1.2)
            m.actuator_gainprm[ids, :, 0] *= kp_scale
            m.actuator_biasprm[ids, :, 1] *= kp_scale
            m.actuator_biasprm[ids, :, 2] *= kd_scale
            m.actuator_forcerange[ids] *= self.uniform((b, self.n, 1), 0.85, 1.0)
            self.encoder_bias[ids] = self.uniform((b, self.n), -0.01, 0.01)
            self.delay[ids] = torch.randint(0, 3, (b,), device=self.device)
        self.sim.recompute_constants(RecomputeLevel.set_const)

    @torch.no_grad()
    def reset(self, ids):
        if not len(ids):
            return
        self.randomize(ids)
        self.sim.reset(ids)
        self.sim.data.qpos[ids] = self.initial
        self.sim.data.qvel[ids] = 0
        self.age[ids] = 0
        self.phase_offset[ids] = 0
        self.initial_actions[ids] = 0
        self.from_reference[ids] = False
        self.commands[ids, 0] = self.uniform((len(ids),), min(0.1, self.cfg.max_speed),
                                           self.cfg.max_speed) * self.command_scale
        self.commands[ids, 1:] = self.uniform((len(ids), 2), -0.1, 0.1) * self.command_scale
        self.commands[ids] *= (torch.rand(len(ids), 1, device=self.device) > 0.1)
        if self.reference is not None and self.cfg.rsi_probability:
            chosen = ids[torch.rand(len(ids), device=self.device) < self.cfg.rsi_probability]
            if len(chosen):
                state = self.reference.sample(self.commands[chosen])
                self.sim.data.qpos[chosen] = state["qpos"]
                self.sim.data.qvel[chosen] = state["qvel"]
                self.phase_offset[chosen] = state["phase"]
                self.initial_actions[chosen] = state["action_history"]
                self.from_reference[chosen] = True
        if self.cfg.randomize:
            self.sim.data.qpos[ids, 7:] += self.uniform((len(ids), self.n), -0.02, 0.02)
            self.sim.data.qpos[ids, 7:] = self.sim.data.qpos[ids, 7:].clamp(
                self.limits[:, 0], self.limits[:, 1])
        self.sim.data.ctrl[ids] = self.target(self.initial_actions[ids, 0])[:, self.actuator_joint]
        self.sim.forward()

    def state(self):
        return self.sim.data.qpos.clone(), self.sim.data.qvel.clone()

    def parameters(self, contact_geom_ids=None):
        # Clones prevent reset-time randomization from mutating a live autograd graph.
        m = self.sim.model
        params = {f: getattr(m, f).clone() for f in self.fields[:5]}
        # A capsule generates two support points, a box eight; repeat by sample.
        ids = self.contact_ids if contact_geom_ids is None else contact_geom_ids
        params["friction"] = m.geom_friction[:, ids, 0].clone()
        j = self.joint_actuator
        kp = m.actuator_gainprm[:, j, 0].clone()
        kd = -m.actuator_biasprm[:, j, 2].clone()
        effort = m.actuator_forcerange[:, j, 1].clone()
        return params, kp, kd, effort

    def observation(self, q, v, last_action, noise=False):
        gravity = quat_inverse_rotate(q[:, 3:7], q.new_tensor([0, 0, -1]).expand(len(q), -1))
        joint_pos = q[:, 7:] + self.encoder_bias.clone()
        gyro, joint_vel = v[:, 3:6], v[:, 6:]
        if noise and self.cfg.randomize:
            gyro = gyro + self.uniform(gyro.shape, -0.03, 0.03)
            gravity = gravity + self.uniform(gravity.shape, -0.015, 0.015)
            joint_pos = joint_pos + self.uniform(joint_pos.shape, -0.005, 0.005)
            joint_vel = joint_vel + self.uniform(joint_vel.shape, -0.2, 0.2)
        return observe(gyro, gravity, self.commands.clone(), self.phase(),
                       joint_pos, joint_vel, last_action, self.q0)

    @torch.no_grad()
    def critic_context(self, contact_geom_ids):
        """Privileged physical/latency context, never supplied to the actor."""
        m, b = self.sim.model, self.cfg.num_envs
        pieces = []
        for field in ("body_mass", "body_inertia", "dof_armature", "dof_damping"):
            nominal = self.defaults[field]
            ratio = (getattr(m, field) - nominal) / nominal.abs().clamp_min(1e-6)
            pieces.append(ratio.flatten(1))
        pieces.extend(((m.body_ipos - self.defaults["body_ipos"]).reshape(b, -1) / 0.02,
                       m.geom_friction[:, contact_geom_ids, 0],
                       self.encoder_bias / 0.01, self.delay[:, None] / 2))
        for field, index in (("actuator_gainprm", 0), ("actuator_biasprm", 2),
                             ("actuator_forcerange", 1)):
            nominal = self.defaults[field][:, index]
            pieces.append((getattr(m, field)[:, :, index] - nominal) / nominal.abs().clamp_min(1e-6))
        return torch.cat(pieces, -1).clone()

    def phase(self):
        return (self.phase_offset + self.age.float() * self.cfg.dt / self.cfg.period) % 1

    def target(self, action):
        return torch.clamp(self.q0 + self.scale * action, self.limits[:, 0], self.limits[:, 1])

    @torch.no_grad()
    def step(self, target):
        self.sim.data.ctrl[:] = target.detach()[:, self.actuator_joint]
        # Low-rate planar pushes, with no privileged disturbance input to the actor.
        if self.cfg.randomize:
            pushed = (self.age > 0) & (self.age % round(5.0 / self.cfg.dt) == 0)
            self.sim.data.qvel[pushed, :2] += self.uniform((int(pushed.sum()), 2), -0.3, 0.3)
        for _ in range(self.cfg.substeps):
            self.sim.step()
        self.age += 1
        return self.state()

    def termination(self, q):
        gravity = quat_inverse_rotate(q[:, 3:7], q.new_tensor([0, 0, -1]).expand(len(q), -1))
        fallen = (q[:, 2] < self.height * 0.55) | (gravity[:, 2] > -0.45)
        timeout = self.age >= round(self.cfg.episode_seconds / self.cfg.dt)
        return fallen, timeout

    def save_scene(self, path):
        self.scene.write(Path(path))
