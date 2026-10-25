"""The complete deployable actor contract. No simulator types are used here."""

import math

import torch
from torch import Tensor, nn


def observation_schema(num_joints: int) -> list[dict]:
    return [
        {"name": name, "size": size, "source": source}
        for name, size, source in (
            ("gyro_body", 3, "IMU, rad/s, body frame"),
            ("projected_gravity", 3, "IMU attitude: R_body_to_world.T @ [0,0,-1]"),
            ("velocity_command", 3, "controller command vx, vy, yaw rate"),
            ("phase_sin_cos", 2, "controller clock; zero when command norm < 0.1"),
            ("joint_pos_rel", num_joints, "encoders minus default pose, rad"),
            ("joint_vel", num_joints, "encoder velocity, rad/s"),
            ("last_action", num_joints, "previous raw policy output"),
        )
    ]


def observe(
    gyro_body: Tensor, projected_gravity: Tensor, command: Tensor,
    phase: Tensor, joint_pos: Tensor, joint_vel: Tensor,
    last_action: Tensor, default_pos: Tensor,
) -> Tensor:
    """Raw physical units; phase is cycles, not radians. Pure, no clock mutation."""
    angle = 2 * math.pi * phase
    gait = torch.stack((angle.sin(), angle.cos()), -1)
    gait = gait * (command.norm(dim=-1, keepdim=True) >= 0.1)
    return torch.cat((gyro_body, projected_gravity, command, gait,
                      joint_pos - default_pos, joint_vel, last_action), -1)


def quat_rotate(q: Tensor, v: Tensor) -> Tensor:
    """Rotate by a unit wxyz quaternion."""
    t = 2 * torch.linalg.cross(q[..., 1:], v, dim=-1)
    return v + q[..., :1] * t + torch.linalg.cross(q[..., 1:], t, dim=-1)


def quat_inverse_rotate(q: Tensor, v: Tensor) -> Tensor:
    return quat_rotate(torch.cat((q[..., :1], -q[..., 1:]), -1), v)


class Actor(nn.Module):
    """Single-frame MLP; fixed scaling is inside the exported network exactly once."""

    def __init__(self, num_joints: int):
        super().__init__()
        self.num_obs = 11 + 3 * num_joints
        scale = [0.25] * 3 + [1.0] * 3 + [2.0, 2.0, 0.25] + [1.0] * 2
        scale += [1.0] * num_joints + [0.05] * num_joints + [1.0] * num_joints
        self.register_buffer("obs_scale", torch.tensor(scale))
        self.net = nn.Sequential(
            nn.Linear(self.num_obs, 128), nn.ELU(),
            nn.Linear(128, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(), nn.Linear(64, num_joints), nn.Tanh(),
        )
        nn.init.uniform_(self.net[-2].weight, -0.01, 0.01)
        nn.init.zeros_(self.net[-2].bias)

    def forward(self, obs: Tensor) -> Tensor:
        return self.net(torch.clamp(obs * self.obs_scale, -10, 10))


def align(actual: Tensor, predicted: Tensor, alpha: float = 1.0) -> Tensor:
    """Exact simulator values with the surrogate derivative (paper 2403.14864)."""
    return actual.detach() + alpha * (predicted - predicted.detach())
