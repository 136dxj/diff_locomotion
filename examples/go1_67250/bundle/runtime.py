"""Portable inference: NumPy + ONNX Runtime, no simulator or training imports."""

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort


def build_observation(gyro, quaternion_wxyz, joint_pos, joint_vel, command,
                      last_action, default_pos, tick, dt, period):
    q = np.asarray(quaternion_wxyz, dtype=np.float32)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    gravity = np.array([2 * (w*y - x*z), -2 * (y*z + w*x),
                        -1 + 2 * (x*x + y*y)], dtype=np.float32)
    phase = (tick * dt / period) % 1.0
    gait = [np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)]
    if np.linalg.norm(command) < 0.1:
        gait = [0.0, 0.0]
    return np.concatenate((gyro, gravity, command, gait,
                           np.asarray(joint_pos) - default_pos, joint_vel, last_action)).astype(np.float32)


class Policy:
    def __init__(self, directory):
        root = Path(directory)
        self.meta = json.loads((root / "metadata.json").read_text())
        self.session = ort.InferenceSession(str(root / "exported/policy.onnx"),
                                            providers=["CPUExecutionProvider"])
        self.q0 = np.asarray(self.meta["default_joint_pos"], dtype=np.float32)
        self.scale = np.asarray(self.meta["action_scale"], dtype=np.float32)
        self.limits = np.asarray(self.meta["joint_limits"], dtype=np.float32)
        if self.session.get_inputs()[0].shape != [1, self.meta["num_observations"]]:
            raise ValueError("ONNX input does not match the saved observation contract")
        self.reset()

    def reset(self):
        self.tick = 0
        self.last_action = np.zeros_like(self.q0)

    def act(self, gyro, quaternion_wxyz, joint_pos, joint_vel, command):
        obs = build_observation(gyro, quaternion_wxyz, joint_pos, joint_vel, command,
                                self.last_action, self.q0, self.tick,
                                self.meta["control_dt"], self.meta["gait_period"])
        if obs.shape != (self.meta["num_observations"],) or not np.isfinite(obs).all():
            raise ValueError("Invalid physical observation")
        self.last_action = self.session.run(["action"], {"obs": obs[None]})[0][0]
        self.tick += 1
        return np.clip(self.q0 + self.scale * self.last_action, self.limits[:, 0], self.limits[:, 1])
