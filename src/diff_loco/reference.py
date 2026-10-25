"""Model-derived kinematic gait references and optional NPZ state banks."""

import math

import numpy as np
import torch


class GaitReference:
    """Flat-ground, fixed-height gait seeds; no robot names or fitted geometry.

    ``initial`` is an upright nominal qpos (arbitrary world XY/yaw). Foot sites
    and support geometry come from ``dynamics``. ``height`` grounds its lowest
    selected contact surface, including sphere/capsule radii. ``targets`` uses
    body XY and world foot Z minus height, matching WalkingLoss. Commands are
    body vx/vy and yaw rate; norm < .1 means a stationary nominal pose.

    Stance follows a constant planar body twist exactly. A C2 horizontal swing
    reconnects stance, with a C1 vertical lift of ``clearance``. ``flat_feet``
    preserves nominal sole roll/pitch and compensates yaw in stance, using
    body angular Jacobians. The supplied nominal soles must already be flat.

    RSI uses bounded damped Jacobian IK, with all joints outside the foot
    chains held at nominal. Joint bounds intersect mechanical limits with
    initial +/- scale; zero scales freeze joints. Unreachable requests RAISE
    with residuals, never return a silently clipped pose or history. In
    particular, a near-straight leg with small action scales may not reach
    the requested clearance. Tune clearance/commands/nominal posture explicitly.

    This is a kinematic seed, not a dynamically balanced walk or inverse-PD
    solution: no torque, velocity-limit, collision, or DR feasibility guarantee.
    Only selected ground contacts are checked. A sphere's changing support
    point need not have zero material slip. Sampling is intentionally detached;
    targets remain differentiable away from the explicit command gate.
    """

    def __init__(self, dynamics, initial, limits, scale, offsets, period,
                 clearance, duty, flat_feet=False, dt=.01):
        self.dynamics = dynamics
        self.initial = torch.as_tensor(initial, device=dynamics.qpos0.device,
                                       dtype=dynamics.qpos0.dtype).detach().clone()
        self.n, self.feet = dynamics.num_joints, len(dynamics.foot_sites)
        if (self.initial.shape != (7 + self.n,) or not self.feet
                or not torch.isfinite(self.initial).all()):
            raise ValueError("initial must be a finite model qpos with at least one foot site")
        self.period, self.clearance, self.duty, self.dt = map(float, (period, clearance, duty, dt))
        if (not all(math.isfinite(x) for x in (self.period, self.clearance, self.duty, self.dt))
                or self.period <= 0 or self.clearance < 0 or not 0 < self.duty < 1
                or not 0 < self.dt < self.period / 2):
            raise ValueError("Require period > 0, clearance >= 0, 0 < duty < 1, 0 < dt < period/2")
        self.flat_feet = bool(flat_feet)
        self.scale = torch.as_tensor(scale, device=self.initial.device,
                                     dtype=self.initial.dtype).detach().clone()
        if self.scale.ndim == 0:
            self.scale = self.scale.expand(self.n).clone()
        self.offsets = torch.as_tensor(offsets, device=self.initial.device,
                                       dtype=self.initial.dtype).detach().clone()
        limits = torch.as_tensor(limits, device=self.initial.device,
                                 dtype=self.initial.dtype).detach().clone()
        if (self.scale.shape != (self.n,) or not torch.isfinite(self.scale).all()
                or (self.scale < 0).any()):
            raise ValueError("scale must be finite, nonnegative and scalar or [N]")
        if self.offsets.shape != (self.feet,) or not torch.isfinite(self.offsets).all():
            raise ValueError("offsets must be finite [F] cycle fractions")
        if (limits.shape != (self.n, 2) or torch.isnan(limits).any()
                or (limits[:, 0] > limits[:, 1]).any()
                or (self.initial[7:] < limits[:, 0]).any()
                or (self.initial[7:] > limits[:, 1]).any()):
            raise ValueError("limits must be ordered [N,2] and contain the initial joints")
        norm = self.initial[3:7].norm()
        if abs(float(norm) - 1) > 1e-4:
            raise ValueError("initial must have a unit wxyz quaternion")
        self.initial[3:7] /= norm
        with torch.no_grad():
            k = dynamics.kinematics(self.initial[None])
            self.rotation = k["body_rot"][0, dynamics.root_body].clone()
            if not torch.allclose(self.rotation[:, 2], self.initial.new_tensor([0, 0, 1]),
                                  rtol=0, atol=1e-5):
                raise ValueError("The nominal root must be upright (world yaw is arbitrary)")
            if not dynamics.num_contacts:
                raise ValueError("A ground reference needs selected contact geometry or site contacts")
            lowest = k["contact_pos"][0, :, 2].min()
            self.height = (self.initial[2] - lowest).clone()
            if self.height <= 0:
                raise ValueError("Selected contact surfaces must lie below the nominal root")
            relative = k["foot_pos"][0] - self.initial[:3]
            self.nominal = relative @ self.rotation
            # Z is a world-height offset, not a rotated body-height coordinate.
            self.nominal[:, 2] = relative[:, 2]
            self.foot_rotation = k["foot_rot"][0].clone()
            self.initial[2] = self.height
        self._active = (dynamics.ancestry[dynamics.foot_body_ids].any(0)
                        & (self.scale > 0)).nonzero().flatten()
        safe_scale = self.scale.clamp_min(torch.finfo(self.initial.dtype).tiny)
        self._lower = ((limits[:, 0] - self.initial[7:]) / safe_scale).clamp_min(-1)
        self._upper = ((limits[:, 1] - self.initial[7:]) / safe_scale).clamp_max(1)
        self.position_tolerance, self.rotation_tolerance = 2e-5, 2e-4

    def _inputs(self, phase, commands):
        phase = torch.as_tensor(phase, device=self.initial.device, dtype=self.initial.dtype)
        commands = torch.as_tensor(commands, device=self.initial.device, dtype=self.initial.dtype)
        if phase.ndim != 1 or commands.shape != (len(phase), 3):
            raise ValueError("Expected phase [B] and commands [B,3]")
        return phase, commands

    def _trajectory(self, phase, commands):
        phase, commands = self._inputs(phase, commands)
        moving = (commands.norm(dim=-1) >= .1).to(commands.dtype)
        commands = commands * moving[:, None]
        cycle = (phase[:, None] + self.offsets) % 1
        swing = ((cycle - self.duty) / (1 - self.duty)).clamp(0, 1)
        smooth = swing**3 * (10 + swing * (-15 + 6 * swing))
        # Sweep has derivative -1 in stance and at both ends of swing.
        sweep = self.period * torch.where(cycle < self.duty, self.duty / 2 - cycle,
                    -self.duty / 2 - (1 - self.duty) * swing + smooth)
        angle = commands[:, 2, None] * sweep
        c, s = angle.cos(), angle.sin()
        x, y = self.nominal[:, 0], self.nominal[:, 1]
        vx, vy = commands[:, 0, None], commands[:, 1, None]
        half_c, half_s = (angle / 2).cos(), (angle / 2).sin()
        travel = sweep * torch.sinc(angle / (2 * math.pi))
        xy = torch.stack((c*x - s*y + travel*(half_c*vx - half_s*vy),
                          s*x + c*y + travel*(half_s*vx + half_c*vy)), -1)
        z = self.nominal[:, 2] + self.clearance * (math.pi*swing).sin().square() * moving[:, None]
        return torch.cat((xy, z[..., None]), -1), angle

    def targets(self, phase, command):
        """[B,F,3]: root/body XY; desired world foot Z minus ``height``."""
        return self._trajectory(phase, command)[0]

    @torch.no_grad()
    def _configuration(self, phase, commands):
        """Solve the periodic joint path at a fixed world root pose."""
        target, angle = self._trajectory(phase, commands)
        q = self.initial.expand(len(phase), -1).clone()
        if not len(phase):
            return q
        desired = target @ self.rotation.T + self.initial[:3]
        desired[..., 2] = target[..., 2] + self.height
        c, s = angle.cos(), angle.sin()
        zero, one = torch.zeros_like(c), torch.ones_like(c)
        yaw = torch.stack((c, -s, zero, s, c, zero, zero, zero, one), -1).reshape(-1, self.feet, 3, 3)
        rotations = yaw @ self.foot_rotation
        actions = q.new_zeros(len(q), self.n)
        active = self._active
        rows = self.feet * (6 if self.flat_feet else 3)
        eye = torch.eye(rows, dtype=q.dtype, device=q.device)
        for _ in range(32):
            k = self.dynamics.kinematics(q)
            error = desired - k["foot_pos"]
            jac = k["foot_jac"][..., 6:]
            angle_error = (rotations - k["foot_rot"]).abs().amax((1, 2, 3))
            if (error.norm(dim=-1).amax() < self.position_tolerance / 4
                    and (not self.flat_feet or angle_error.max() < self.rotation_tolerance / 4)):
                break
            if not active.numel():
                break
            if self.flat_feet:
                # Sum cross-products of frame columns: world angular error.
                orient = .5 * torch.linalg.cross(k["foot_rot"].transpose(-1, -2),
                                                 rotations.transpose(-1, -2)).sum(-2)
                angular = k["body_ang_jac"][:, self.dynamics.foot_body_ids, :, 6:]
                # 0.1 m/rad balances position and orientation in the DLS solve.
                error = torch.cat((error, .1 * orient), -1)
                jac = torch.cat((jac, .1 * angular), -2)
            jac = jac[..., active].reshape(len(q), rows, -1) * self.scale[active]
            rhs = error.reshape(len(q), rows, 1)
            delta = (jac.transpose(1, 2) @ torch.linalg.solve(
                jac @ jac.transpose(1, 2) + 1e-8 * eye, rhs)).squeeze(-1)
            actions[:, active] = (actions[:, active] + delta.clamp(-.5, .5)).clamp(
                self._lower[active], self._upper[active])
            q[:, 7:] = self.initial[7:] + self.scale * actions
        k = self.dynamics.kinematics(q)
        error = (desired - k["foot_pos"]).norm(dim=-1).amax(-1)
        angle_error = (rotations - k["foot_rot"]).abs().amax((1, 2, 3))
        good = torch.isfinite(q).all(-1) & (error <= self.position_tolerance)
        if self.flat_feet:
            good &= angle_error <= self.rotation_tolerance
        # Catch ground penetration caused by unconstrained foot rotation too.
        good &= k["contact_pos"][..., 2].amin(-1) >= -2 * self.position_tolerance
        if not good.all():
            bad = (~good).nonzero().flatten()[:3]
            raise ValueError(
                f"Gait IK is unreachable within joint/action bounds or penetrates ground: "
                f"max position residual={float(error.max()):.3g} m, "
                f"rotation residual={float(angle_error.max()):.3g}, "
                f"min contact z={float(k['contact_pos'][..., 2].min()):.3g} m; "
                f"phase={phase[bad].tolist()}, commands={commands[bad].tolist()}, "
                f"clearance={self.clearance:g}. Reduce clearance/speed or adjust initial/scale."
            )
        return q

    @torch.no_grad()
    def sample(self, commands):
        """Random phase, coherent MuJoCo qpos/qvel and raw t-1,t-2 actions.

        All four joint configurations use the same command and periodic path:
        t-2dt, t-dt, t, t+dt. Joint velocities are centered finite differences
        over 2dt. Root XY/yaw at reset retain the nominal world pose; the local
        continuation has the commanded planar twist (world linear/body angular).
        History stores position-target actions, without inverse-dynamics lead.
        """
        commands = torch.as_tensor(commands, device=self.initial.device, dtype=self.initial.dtype)
        if commands.ndim != 2 or commands.shape[1] != 3 or not torch.isfinite(commands).all():
            raise ValueError("commands must be finite [B,3]")
        batch = len(commands)
        phase = torch.rand(batch, device=self.initial.device, dtype=self.initial.dtype)
        times = self.initial.new_tensor([-2, -1, 0, 1]) * self.dt / self.period
        phases = (phase[:, None] + times) % 1
        poses = self._configuration(phases.flatten(), commands[:, None].expand(-1, 4, -1).reshape(-1, 3))
        poses = poses.reshape(batch, 4, 7 + self.n)
        qpos = poses[:, 2].clone()
        qvel = qpos.new_zeros(batch, 6 + self.n)
        moving = (commands.norm(dim=-1) >= .1).to(commands.dtype)[:, None]
        linear = torch.cat((commands[:, :2], commands.new_zeros(batch, 1)), -1) * moving
        angular = torch.cat((commands.new_zeros(batch, 2), commands[:, 2:]), -1) * moving
        qvel[:, :3] = linear @ self.rotation.T
        qvel[:, 3:6] = angular @ self.rotation
        qvel[:, 6:] = (poses[:, 3, 7:] - poses[:, 1, 7:]) / (2 * self.dt)
        safe_scale = torch.where(self.scale > 0, self.scale, torch.ones_like(self.scale))
        history = (poses[:, [1, 0], 7:] - self.initial[7:]) / safe_scale
        # Reconstructing actions from qpos can round a boundary a few ulps out.
        # Reject substantive violations, and round ONLY those numerical ulps.
        if (not torch.isfinite(qvel).all() or not torch.isfinite(history).all()
                or (history.abs() > 1 + 8 * torch.finfo(history.dtype).eps).any()):
            raise ValueError("Nonfinite gait velocity or action history outside [-1,1]")
        return dict(qpos=qpos, qvel=qvel, phase=phase, action_history=history.clamp(-1, 1))


class ReferenceStates:
    """Load ordered joint_names, qpos[F,7+N], qvel[F,6+N], phase[F],
    command[F,3], scalar period, and optional action_history[F,2,N].

    qpos uses world XYZ and unit wxyz; qvel follows MuJoCo coordinates.
    History contains raw actions at t-1, t-2. Period must match within 1e-6
    relative tolerance. Samples are independent float32 tensors on device.
    """

    def __init__(self, path, joint_names, device, period):
        expected = np.asarray(joint_names)
        if (expected.ndim != 1 or expected.dtype.kind not in "US" or not expected.size
                or len(set(expected.tolist())) != expected.size or np.any(expected == "")):
            raise ValueError("joint_names must be nonempty, unique strings in model order")
        expected = expected.astype(str)
        requested_period = np.asarray(period)
        archive = np.load(path, allow_pickle=False)
        if not isinstance(archive, np.lib.npyio.NpzFile):
            raise ValueError("Expected an NPZ reference archive")
        with archive:
            required = {"joint_names", "qpos", "qvel", "phase", "command", "period"}
            if missing := required.difference(archive.files):
                raise ValueError(f"Missing reference fields: {sorted(missing)}")
            names = archive["joint_names"]
            if (names.dtype.kind not in "US" or names.shape != expected.shape
                    or not np.array_equal(names.astype(str), expected)):
                raise ValueError("Reference joint_names do not match model joint order")
            saved_period = archive["period"]
            for value in (requested_period, saved_period):
                if (value.shape != () or value.dtype.kind not in "fiu"
                        or not np.isfinite(value) or value <= 0):
                    raise ValueError("period must be a finite positive scalar")
            if not np.isclose(saved_period, requested_period, rtol=1e-6, atol=0):
                raise ValueError("Reference period does not match task period")
            arrays = {key: archive[key] for key in ("qpos", "qvel", "phase", "command")}
            frames = arrays["qpos"].shape[0] if arrays["qpos"].ndim == 2 else 0
            if frames == 0:
                raise ValueError("qpos must contain a nonempty matrix of reference states")
            n = len(expected)
            shapes = dict(qpos=(frames, 7 + n), qvel=(frames, 6 + n),
                          phase=(frames,), command=(frames, 3), action_history=(frames, 2, n))
            arrays["action_history"] = (archive["action_history"] if "action_history" in archive
                                        else np.zeros(shapes["action_history"], dtype=np.float32))
        for key, value in arrays.items():
            if value.shape != shapes[key] or value.dtype.kind not in "fiu":
                raise ValueError(f"{key} must be a real numeric array of shape {shapes[key]}")
            if not np.isfinite(value).all():
                raise ValueError(f"{key} must contain only finite values")
        if np.any((arrays["phase"] < 0) | (arrays["phase"] >= 1)):
            raise ValueError("phase must lie in [0, 1)")
        norms = np.linalg.norm(arrays["qpos"][:, 3:7].astype(np.float64), axis=-1)
        if not np.allclose(norms, 1, rtol=0, atol=1e-4):
            raise ValueError("qpos quaternion norm must be within 1e-4 of one")
        if np.any(np.abs(arrays["action_history"].astype(np.float64)) > 1):
            raise ValueError("action_history must lie in [-1, 1]")
        self.device, self.num_frames = torch.device(device), frames
        self.joint_names, self.period = tuple(expected.tolist()), float(saved_period)
        self._states = {key: torch.tensor(value, dtype=torch.float32, device=self.device)
                        for key, value in arrays.items()}
        if any(not torch.isfinite(value).all() for value in self._states.values()):
            raise ValueError("Reference values must be finite when represented as float32")
        if (self._states["phase"] >= 1).any():
            raise ValueError("phase must remain below one when represented as float32")

    def sample(self, count):
        """Uniform sampling with replacement, including an empty batch for count=0."""
        if isinstance(count, bool) or not isinstance(count, (int, np.integer)) or count < 0:
            raise ValueError("sample count must be a nonnegative integer")
        indices = torch.randint(self.num_frames, (int(count),), device=self.device)
        # index_select allocates independent storage, even for repeated indices.
        return {key: value.index_select(0, indices) for key, value in self._states.items()}
