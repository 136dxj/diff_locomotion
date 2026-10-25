"""Model-derived PyTorch dynamics for short differentiable rollout gradients.

FK, material-point Jacobians, the generalized mass matrix, and gravity use the
compiled MuJoCo tree, including joint anchors/references and inertial frames.
The step is deliberately a surrogate: frozen-configuration implicit PD/passive
damping, smooth effort saturation, and four smoothed contact GS sweeps against
the horizontal z=0 plane. It omits Coriolis/centrifugal/gyroscopic forces, joint
limits, springs, tendons, actuator transmissions/delays, equality constraints,
self-collision, restitution, and rolling/torsional friction. Multiple contact
points still generate moments through their full generalized Jacobians.

Intended for external forward-state alignment as in Song et al.,
arXiv:2403.14864, Sec. III-F; this uses coupled articulated inertia rather than
that paper's split single-body approximation. The cross-contact AND final
impulse sigmoid gates follow the idea of Schwarke et al., arXiv:2404.02887v2,
Alg. 1. Our smooth cone contraction, penetration bias, controller, and fixed
contact samples are approximations, NOT a reproduction of its NCP simulator.
No MuJoCo stepping, finite differences, mjlab, or detach occurs in the tensor
path. Loading is a one-time copy; later MjModel changes require parameters.
"""

from __future__ import annotations

from itertools import product
from os import PathLike

import mujoco
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _quat_matrix(q: Tensor) -> Tensor:
    q = F.normalize(q, dim=-1)
    w, x, y, z = q.unbind(-1)
    return torch.stack((
        1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w),
        2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w),
        2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y),
    ), -1).reshape(q.shape[:-1] + (3, 3))


def _quat_product(a: Tensor, b: Tensor) -> Tensor:
    return torch.cat((a[..., :1]*b[..., :1] - (a[..., 1:]*b[..., 1:]).sum(-1, keepdim=True),
                      a[..., :1]*b[..., 1:] + b[..., :1]*a[..., 1:]
                      + torch.linalg.cross(a[..., 1:], b[..., 1:])), -1)


def _rotation_quat(vector: Tensor) -> Tensor:
    squared = vector.square().sum(-1, keepdim=True)
    angle = squared.clamp_min(1e-12).sqrt()
    # Both branches have finite derivatives at zero, including JVP/double backward.
    real = torch.where(squared < 1e-8, 1 - squared/8 + squared.square()/384,
                       torch.cos(angle / 2))
    scale = torch.where(squared < 1e-8, 0.5 - squared/48 + squared.square()/3840,
                        0.5 * torch.sinc(angle / (2 * torch.pi)))
    return torch.cat((real, scale * vector), -1)


def _mv(matrix: Tensor, vector: Tensor) -> Tensor:
    return (matrix @ vector.unsqueeze(-1)).squeeze(-1)


class RobotModel(nn.Module):
    """One free root plus any hinge/slide tree, in canonical MuJoCo order.

    qpos: [B, 7+N], world translation, wxyz quaternion, scalar joints.
    qvel: [B, 6+N], world linear velocity, BODY angular velocity, joints.
    Inputs must share a device and use float32 or float64. Constants retain
    MuJoCo's double precision and are cast to the input dtype during use;
    ``model.to(dtype=torch.float32)`` avoids those casts during training.

    ``foot_sites`` is an ordered list of semantic site names. With
    ``contact_geoms=None``, use collidable geoms on those sites' bodies, falling
    back to a site point for a body without collision geoms. An explicit list
    selects exactly those geoms; [] disables contacts. Supported contact geoms
    are spheres (one support point), capsules (two end-sphere support points),
    and boxes (eight corners). Unsupported shapes raise, so callers can choose
    site contacts explicitly using a model with no collidable foot geoms.
    Sphere/capsule support points move down by their radius in WORLD z; their
    contact_jac is the material-point velocity Jacobian at that position, not
    the position derivative of a changing geometric support point.

    ``step``/``mass_matrix``/``gravity`` accept these exact parameter keys:
      body_mass: [B,nbody], body_inertia: [B,nbody,3] principal moments,
      body_ipos: [B,nbody,3] absolute COM locations in BODY frames,
      dof_armature, dof_damping: [B,nv],
      friction: [B,1] or [B,C], indexed by generated contact point,
      mass_scale: [B,1] or [B,nbody], multiplies mass AND inertia,
      com_offset: [B,3] additive root-body COM shift, or [B,nbody,3].
    Unbatched trailing shapes broadcast too (root com_offset also accepts [3]).
    Body arrays include world body 0; inertial rotations stay model-derived.
    Absolute overrides apply first, then convenience scale/offset modifiers.
    Mass/inertia must describe a physical model; gains, damping, armature,
    friction and effort must be nonnegative. Unknown keys raise ValueError.
    dt is positive, scalar or [B,1]. kp/kd/effort/target broadcast to [B,N].
    Effort is in generalized joint units, NOT actuator control units.
    """

    _parameter_keys = frozenset({
        "body_mass", "body_inertia", "body_ipos", "dof_armature",
        "dof_damping", "friction", "mass_scale", "com_offset",
    })

    def __init__(self, model: mujoco.MjModel | str, *, foot_sites: list[str],
                 contact_geoms: list[str] | None = None, device="cpu"):
        super().__init__()
        m = mujoco.MjModel.from_xml_path(str(model)) if isinstance(model, (str, PathLike)) else model
        if not isinstance(m, mujoco.MjModel):
            raise TypeError("model must be mujoco.MjModel or an XML path")
        free = np.flatnonzero(m.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        scalar = {int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)}
        if len(free) != 1 or free[0] != 0 or any(int(t) not in scalar for t in m.jnt_type[1:]):
            raise ValueError("Expected one leading free joint followed only by hinge/slide joints")
        self.root_body = int(m.jnt_bodyid[0])
        if m.body_parentid[self.root_body] != 0 or np.any(m.body_mocapid >= 0):
            raise ValueError("The free root must be a world child; mocap bodies are unsupported")
        self.nq, self.nv, self.nbody = m.nq, m.nv, m.nbody
        self.num_joints = m.njnt - 1
        if self.nq != 7 + self.num_joints or self.nv != 6 + self.num_joints:
            raise ValueError("Unsupported generalized coordinate layout")
        self.joint_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j)
                            or f"joint_{j}" for j in range(1, m.njnt)]
        self.foot_sites = list(foot_sites)
        self.parents = tuple(int(p) for p in m.body_parentid)
        self.body_joints = tuple(tuple(range(int(m.body_jntadr[b]),
                                             int(m.body_jntadr[b] + m.body_jntnum[b])))
                                 for b in range(m.nbody))
        self.joint_types = tuple(int(t) for t in m.jnt_type)
        self.contact_iterations = 4
        self.contact_width = 0.01  # m; sigmoid stiffness = 1 / width
        self.contact_beta = 0.2   # soft penetration correction per step
        self.contact_velocity_width = 0.01  # m/s, normal rectification

        def buffer(name, value, dtype=torch.float64):
            self.register_buffer(name, torch.as_tensor(np.array(value, copy=True),
                                                       dtype=dtype, device=device))

        for name in ("qpos0", "body_pos", "body_quat", "body_ipos", "body_iquat",
                     "body_mass", "body_inertia", "jnt_pos", "jnt_axis",
                     "dof_armature", "dof_damping"):
            buffer(name, getattr(m, name))
        buffer("gravity_vector", m.opt.gravity)
        # All axes on a body affect that body; fixed descendants inherit them.
        ancestry = np.zeros((m.nbody, self.num_joints), dtype=bool)
        dynamic = np.zeros(m.nbody, dtype=bool)
        for b in range(1, m.nbody):
            ancestry[b] = ancestry[self.parents[b]]
            dynamic[b] = b == self.root_body or dynamic[self.parents[b]]
            for j in self.body_joints[b]:
                if j:
                    if not dynamic[b]:
                        raise ValueError("Every scalar joint must descend from the free root")
                    ancestry[b, j - 1] = True
        buffer("ancestry", ancestry, torch.bool)
        buffer("dynamic_bodies", dynamic, torch.bool)
        buffer("hinges", m.jnt_type[1:] == mujoco.mjtJoint.mjJNT_HINGE, torch.bool)
        buffer("body_ids", np.arange(m.nbody), torch.long)

        def resolve(kind, names):
            ids = [mujoco.mj_name2id(m, kind, n) for n in names]
            if any(i < 0 for i in ids):
                raise ValueError(f"Unknown {kind.name} name in {names}")
            if len(set(ids)) != len(ids):
                raise ValueError("Duplicate semantic names are not allowed")
            return ids

        sites = resolve(mujoco.mjtObj.mjOBJ_SITE, self.foot_sites)
        buffer("foot_body_ids", m.site_bodyid[sites], torch.long)
        buffer("foot_local_pos", m.site_pos[sites])
        buffer("foot_local_rot", _quat_matrix(torch.as_tensor(m.site_quat[sites])).numpy())
        fallback = []
        if contact_geoms is None:
            geoms = []
            for sid in sites:
                selected = [g for g in range(m.ngeom) if m.geom_bodyid[g] == m.site_bodyid[sid]
                            and (m.geom_contype[g] or m.geom_conaffinity[g])]
                geoms.extend(g for g in selected if g not in geoms)
                if not selected:
                    fallback.append(sid)
        else:
            geoms = resolve(mujoco.mjtObj.mjOBJ_GEOM, contact_geoms)
        self.contact_geom_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
                                   or f"geom_{g}" for g in geoms]
        cb, cp, cr, cf, cg = [], [], [], [], []
        for g in geoms:
            kind, size = int(m.geom_type[g]), m.geom_size[g]
            if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
                points, radius = [(0, 0, 0)], size[0]
            elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
                points, radius = [(0, 0, -size[1]), (0, 0, size[1])], size[0]
            elif kind == mujoco.mjtGeom.mjGEOM_BOX:
                points, radius = np.array(list(product((-1, 1), repeat=3))) * size, 0.0
            else:
                raise ValueError(f"Contact geom {g} must be sphere/capsule/box; use semantic site points otherwise")
            rotation = _quat_matrix(torch.as_tensor(m.geom_quat[g])).numpy()
            for point in points:
                cb.append(int(m.geom_bodyid[g]))
                cp.append(m.geom_pos[g] + rotation @ np.asarray(point))
                cr.append(radius)
                cf.append(m.geom_friction[g, 0])
                cg.append(g)
        for sid in fallback:
            cb.append(int(m.site_bodyid[sid]))
            cp.append(m.site_pos[sid])
            cr.append(0.0)
            cf.append(0.8)  # a site has no MuJoCo friction; override via parameters
            cg.append(-1)
        if any(not dynamic[b] for b in cb):
            raise ValueError("Contact points must belong to the free-root tree")
        self.num_contacts = len(cb)
        buffer("contact_body_ids", cb, torch.long)
        buffer("contact_geom_ids", cg, torch.long)
        buffer("contact_local_pos", np.asarray(cp).reshape(-1, 3))
        buffer("contact_radius", cr)
        buffer("contact_friction", cf)
        buffer("root_mask", np.arange(m.nbody) == self.root_body, torch.bool)

    def _checked_parameters(self, parameters):
        p = {} if parameters is None else parameters
        unknown = p.keys() - self._parameter_keys
        if unknown:
            raise ValueError(f"Unsupported dynamics parameters: {sorted(unknown)}")
        return p

    @staticmethod
    def _tensor(value, like: Tensor) -> Tensor:
        return torch.as_tensor(value, dtype=like.dtype, device=like.device)

    def _properties(self, q: Tensor, p: dict):
        def value(name, shape):
            return self._tensor(p.get(name, getattr(self, name)), q).expand((q.shape[0],) + shape)
        scale = self._tensor(p.get("mass_scale", 1.0), q).expand(q.shape[0], self.nbody)
        mass = value("body_mass", (self.nbody,)) * scale
        inertia = value("body_inertia", (self.nbody, 3)) * scale[..., None]
        ipos = value("body_ipos", (self.nbody, 3))
        if "com_offset" in p:
            offset = self._tensor(p["com_offset"], q)
            if offset.ndim <= 2:  # [3] or [B,3] is a root-body offset
                offset = offset.reshape(-1, 1, 3) * self.root_mask[None, :, None]
            ipos = ipos + offset
        return mass, inertia, ipos, value("dof_armature", (self.nv,)), value("dof_damping", (self.nv,))

    def _point_jacobians(self, point: Tensor, bodies: Tensor, k: dict):
        batch, count = point.shape[:2]
        dynamic = self.dynamic_bodies[bodies][None, :, None, None]
        eye = torch.eye(3, dtype=point.dtype, device=point.device).expand(batch, count, 3, 3)
        root_axes = k["body_rot"][:, self.root_body].transpose(-1, -2)[:, None]
        base_rotation = torch.linalg.cross(root_axes.expand(-1, count, -1, -1),
                                           (point - k["body_pos"][:, self.root_body, None])[:, :, None].expand(-1, -1, 3, -1))
        axes = k["joint_axis"][:, None].expand(-1, count, -1, -1)
        hinge_linear = torch.linalg.cross(axes, point[:, :, None] - k["joint_pos"][:, None])
        joint_linear = torch.where(self.hinges[None, None, :, None], hinge_linear, axes)
        mask = self.ancestry[bodies][None, :, :, None]
        linear = torch.cat((eye * dynamic, base_rotation.transpose(-1, -2) * dynamic,
                            (joint_linear * mask).transpose(-1, -2)), -1)
        angular = torch.cat((torch.zeros_like(eye),
                             root_axes.transpose(-1, -2).expand(-1, count, -1, -1) * dynamic,
                             (axes * mask * self.hinges[None, None, :, None]).transpose(-1, -2)), -1)
        return linear, angular

    def _kinematics(self, q: Tensor, ipos: Tensor, inertia: Tensor, *, feet=True) -> dict:
        if q.ndim != 2 or q.shape[-1] != self.nq:
            raise ValueError(f"qpos must have shape [B,{self.nq}]")
        batch = q.shape[0]
        eye = torch.eye(3, dtype=q.dtype, device=q.device).expand(batch, 3, 3)
        positions, rotations = [q.new_zeros(batch, 3)], [eye]
        axes, anchors = [], []
        bpos, brot = self.body_pos.to(q), _quat_matrix(self.body_quat.to(q))
        jpos, jaxis = self.jnt_pos.to(q), self.jnt_axis.to(q)
        # Independent joint rotations share one batched kernel sequence. The
        # tree traversal below still applies them in exactly the same order.
        displacement = q[:, 7:] - self.qpos0[7:].to(q)
        joint_rot = _quat_matrix(_rotation_quat(jaxis[1:] * displacement[..., None]))
        for b in range(1, self.nbody):
            if b == self.root_body:
                pos, rot = q[:, :3], _quat_matrix(q[:, 3:7])
            else:
                parent = self.parents[b]
                pos = positions[parent] + _mv(rotations[parent], bpos[b])
                rot = rotations[parent] @ brot[b]
                for j in self.body_joints[b]:
                    axis, anchor = _mv(rot, jaxis[j]), pos + _mv(rot, jpos[j])
                    axes.append(axis)
                    anchors.append(anchor)
                    if self.joint_types[j] == mujoco.mjtJoint.mjJNT_SLIDE:
                        pos = pos + axis * displacement[:, j - 1, None]
                    else:
                        rot = rot @ joint_rot[:, j - 1]
                        pos = anchor - _mv(rot, jpos[j])
            positions.append(pos)
            rotations.append(rot)
        k = {"body_pos": torch.stack(positions, 1), "body_rot": torch.stack(rotations, 1),
             "joint_axis": torch.stack(axes, 1) if axes else q.new_zeros(batch, 0, 3),
             "joint_pos": torch.stack(anchors, 1) if anchors else q.new_zeros(batch, 0, 3)}
        k["com_pos"] = k["body_pos"] + _mv(k["body_rot"], ipos)
        inertial_rot = k["body_rot"] @ _quat_matrix(self.body_iquat.to(q))
        k["inertia_world"] = (inertial_rot * inertia[..., None, :]) @ inertial_rot.transpose(-1, -2)
        k["com_jac"], k["body_ang_jac"] = self._point_jacobians(k["com_pos"], self.body_ids, k)
        if feet:
            k["foot_rot"] = k["body_rot"][:, self.foot_body_ids] @ self.foot_local_rot.to(q)
            k["foot_pos"] = k["body_pos"][:, self.foot_body_ids] + _mv(k["body_rot"][:, self.foot_body_ids], self.foot_local_pos.to(q))
            k["foot_jac"], _ = self._point_jacobians(k["foot_pos"], self.foot_body_ids, k)
        center = k["body_pos"][:, self.contact_body_ids] + _mv(k["body_rot"][:, self.contact_body_ids], self.contact_local_pos.to(q))
        down = q.new_tensor([0, 0, 1]) * self.contact_radius.to(q)[:, None]
        k["contact_pos"] = center - down
        k["contact_jac"], _ = self._point_jacobians(k["contact_pos"], self.contact_body_ids, k)
        return k

    def kinematics(self, qpos: Tensor) -> dict[str, Tensor]:
        """World poses/Jacobians; body and COM arrays include world body 0."""
        _, inertia, ipos, _, _ = self._properties(qpos, {})
        return self._kinematics(qpos, ipos, inertia)

    @staticmethod
    def _mass(k: dict, mass: Tensor, armature: Tensor) -> Tensor:
        jv, jw = k["com_jac"], k["body_ang_jac"]
        return ((jv.transpose(-1, -2) @ (mass[..., None, None] * jv)
                 + jw.transpose(-1, -2) @ k["inertia_world"] @ jw).sum(1)
                + torch.diag_embed(armature))

    def mass_matrix(self, qpos: Tensor, *, parameters: dict | None = None) -> Tensor:
        """Full [B,nv,nv] physical inertia including armature, without jitter."""
        mass, inertia, ipos, armature, _ = self._properties(qpos, self._checked_parameters(parameters))
        return self._mass(self._kinematics(qpos, ipos, inertia, feet=False), mass, armature)

    def _gravity(self, k: dict, mass: Tensor) -> Tensor:
        return (k["com_jac"] * (mass[..., None] * self.gravity_vector.to(mass))[:, :, :, None]).sum((1, 2))

    def gravity(self, qpos: Tensor, *, parameters: dict | None = None) -> Tensor:
        """Generalized gravitational force (negative MuJoCo bias at qvel=0)."""
        mass, inertia, ipos, _, _ = self._properties(qpos, self._checked_parameters(parameters))
        return self._gravity(self._kinematics(qpos, ipos, inertia, feet=False), mass)

    def integrate(self, qpos: Tensor, qvel: Tensor, dt) -> Tensor:
        """MuJoCo-compatible configuration integration, with body-frame omega."""
        h = self._tensor(dt, qpos).reshape(-1, 1)
        quat = F.normalize(_quat_product(F.normalize(qpos[:, 3:7], dim=-1),
                                         _rotation_quat(h * qvel[:, 3:6])), dim=-1)
        return torch.cat((qpos[:, :3] + h * qvel[:, :3], quat,
                          qpos[:, 7:] + h * qvel[:, 6:]), -1)

    def _contact_impulses(self, velocity: Tensor, jac: Tensor, response: Tensor,
                          position: Tensor, friction: Tensor, h: Tensor) -> Tensor:
        """Fixed-count GS; response [B,C,nv,3] = (M+hD)^-1 Jc.T."""
        count = self.num_contacts
        diagonal = jac @ response
        eye = torch.eye(3, dtype=jac.dtype, device=jac.device)
        diagonal = diagonal + 1e-6 * diagonal.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None] * eye
        inverse = torch.linalg.solve(diagonal, eye.expand_as(diagonal))
        gap = position[..., 2]
        gate = torch.sigmoid(-gap / self.contact_width)
        desired_z = self.contact_beta * self.contact_width * F.softplus(-gap / self.contact_width) / h
        desired = torch.stack((torch.zeros_like(gap), torch.zeros_like(gap), desired_z), -1)
        impulses = [velocity.new_zeros(velocity.shape[0], 3) for _ in range(count)]
        delta = torch.zeros_like(velocity)
        for _ in range(self.contact_iterations):
            for c in range(count):
                # Remove self; every other impulse is sigmoid gated (Alg. 1).
                without = delta - _mv(response[:, c], gate[:, c, None] * impulses[c])
                raw = _mv(inverse[:, c], desired[:, c] - _mv(jac[:, c], velocity + without))
                width = self.contact_velocity_width / diagonal[:, c, 2, 2].clamp_min(1e-12)
                normal = width * F.softplus(raw[:, 2] / width)
                limit = friction[:, c] * normal
                tangent = raw[:, :2] * (limit / torch.sqrt(limit.square() + raw[:, :2].square().sum(-1) + 1e-16))[:, None]
                impulses[c] = torch.cat((tangent, normal[:, None]), -1)
                delta = without + _mv(response[:, c], gate[:, c, None] * impulses[c])
        # Final gate is necessary even with gated cross-contact interactions.
        return gate[..., None] * torch.stack(impulses, 1)

    def step(self, qpos: Tensor, qvel: Tensor, target, kp, kd, effort, dt, *,
             parameters: dict | None = None) -> tuple[Tensor, Tensor]:
        """One smooth surrogate step. Contact forces affect base AND all joints.

        PD uses an implicit unconstrained predictor followed by tanh effort
        saturation and a passive-damping solve. Contact then freezes that
        motor impulse; it does not re-solve the saturated controller. Positive
        torque limits are respected exactly, but this approximation is not an
        unconditional stability guarantee under arbitrarily large dt/contact.
        """
        p = self._checked_parameters(parameters)
        if qvel.shape != (qpos.shape[0], self.nv):
            raise ValueError(f"qvel must have shape [B,{self.nv}]")
        h = self._tensor(dt, qpos).reshape(-1, 1)
        target, kp, kd, effort = [self._tensor(x, qpos).expand(qpos.shape[0], self.num_joints)
                                 for x in (target, kp, kd, effort)]
        mass, inertia, ipos, armature, damping = self._properties(qpos, p)
        k = self._kinematics(qpos, ipos, inertia, feet=False)
        matrix = self._mass(k, mass, armature)
        # Tiny solve-only regularization; never a replacement for physical inertia.
        passive = matrix + torch.diag_embed(h * damping + 1e-9)
        zeros = qpos.new_zeros(qpos.shape[0], 6)
        stiffness = torch.cat((zeros, h * kd + h.square() * kp), -1)
        error = target - qpos[:, 7:]
        momentum = _mv(matrix, qvel) + h * self._gravity(k, mass)
        predictor = torch.linalg.solve(passive + torch.diag_embed(stiffness),
                                        (momentum + h * torch.cat((zeros, kp * error), -1)).unsqueeze(-1)).squeeze(-1)
        raw = kp * (error - h * predictor[:, 6:]) - kd * predictor[:, 6:]
        finite = torch.isfinite(effort)
        limit = torch.where(finite, effort, torch.ones_like(effort))
        torque = torch.where(finite, limit * torch.tanh(raw / limit.clamp_min(1e-12)), raw)
        rhs = momentum + h * torch.cat((zeros, torque), -1)
        jac = k["contact_jac"]
        jt = jac.reshape(qpos.shape[0], 3 * self.num_contacts, self.nv).transpose(-1, -2)
        solved = torch.linalg.solve(passive, torch.cat((rhs[..., None], jt), -1))
        velocity = solved[..., 0]
        if self.num_contacts:
            response = solved[..., 1:].reshape(qpos.shape[0], self.nv, self.num_contacts, 3).transpose(1, 2)
            friction = self._tensor(p.get("friction", self.contact_friction), qpos).expand(qpos.shape[0], self.num_contacts)
            impulse = self._contact_impulses(velocity, jac, response, k["contact_pos"], friction, h)
            velocity = velocity + _mv(solved[..., 1:], impulse.flatten(1))
        return self.integrate(qpos, velocity, h), velocity
