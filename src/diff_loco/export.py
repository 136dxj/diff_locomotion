"""Export a self-contained ONNX/CPU-MuJoCo bundle and verify actor parity."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import shutil

import numpy as np
import onnxruntime as ort
import torch

from .policy import Actor, observation_schema, observe, quat_inverse_rotate
from .runtime import build_observation
from .task import LocomotionTask, TaskCfg


def export(checkpoint, directory, device="cuda:0"):
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state["task"]["robot"] == "g1" and "initial_qpos" not in state:
        raise ValueError("Legacy G1 checkpoint lacks nominal pose metadata; use its original task configuration")
    task = LocomotionTask(replace(TaskCfg(**state["task"]), num_envs=1, randomize=False,
                                   rsi_probability=0.0, device=device))
    if "initial_qpos" in state:
        task.initial.copy_(state["initial_qpos"].to(task.device))
        task.height = float(task.initial[2])
    actor = Actor(task.n).eval()
    actor.load_state_dict(state["actor"])
    root = Path(directory)
    (root / "exported").mkdir(parents=True, exist_ok=True)
    example = torch.zeros(1, actor.num_obs)
    torch.onnx.export(actor, (example,), str(root / "exported/policy.onnx"),
                      input_names=["obs"], output_names=["action"], opset_version=17, dynamo=False)
    torch.jit.trace(actor, example).save(str(root / "exported/policy.pt"))
    task.save_scene(root / "scene")
    _, kp, kd, effort = task.parameters()
    array = lambda x: x.detach().cpu().tolist()
    meta = dict(
        format_version=1, robot=task.cfg.robot, iteration=state["iteration"],
        num_observations=actor.num_obs, num_actions=task.n,
        observations=observation_schema(task.n), history_length=1,
        normalization="fixed scaling and clipping embedded in the ONNX graph; feed raw physical units",
        observation_scale=array(actor.obs_scale), quaternion_order="wxyz",
        control_dt=task.cfg.dt, physics_dt=task.cfg.dt / task.cfg.substeps,
        substeps=task.cfg.substeps, gait_period=task.cfg.period, gait_clearance=task.clearance,
        physics_options={name: (int(getattr(task.sim.mj_model.opt, name))
                                if name in ("integrator", "solver", "cone", "iterations", "ls_iterations")
                                else float(getattr(task.sim.mj_model.opt, name)))
                         for name in ("integrator", "solver", "cone", "iterations", "ls_iterations",
                                      "tolerance", "ls_tolerance", "impratio")},
        joint_names=task.joint_names, actuator_joint_indices=task.actuator_joint.tolist(),
        default_joint_pos=array(task.q0), initial_qpos=array(task.initial),
        action_scale=array(task.scale), joint_limits=array(task.limits),
        kp=array(kp[0]), kd=array(kd[0]), effort_limits=array(effort[0]),
        action_semantics="clip(q0 + scale * raw_action, joint_limits); qd_des=0; tau_ff=0",
        deployment_status="offline bundle; walking and hardware acceptance require measured evaluation",
    )
    (root / "metadata.json").write_text(json.dumps(meta, indent=2))
    shutil.copyfile(Path(__file__).with_name("runtime.py"), root / "runtime.py")
    session = ort.InferenceSession(str(root / "exported/policy.onnx"), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(171)
    errors, observation_errors = [], []
    for tick in (0, 1, 59, 60, 1001):
        q = rng.normal(size=4).astype(np.float32)
        q /= np.linalg.norm(q)
        gyro, command = rng.normal(size=(2, 3)).astype(np.float32)
        if tick == 0:
            command[:] = 0
        jp, jv, prev = rng.normal(size=(3, task.n)).astype(np.float32)
        raw = build_observation(gyro, q, jp, jv, command, prev, np.array(meta["default_joint_pos"]),
                                tick, task.cfg.dt, task.cfg.period)
        tensor = lambda a: torch.tensor(a, dtype=torch.float32)[None]
        gravity = quat_inverse_rotate(tensor(q), tensor([0, 0, -1]))
        expected_obs = observe(tensor(gyro), gravity, tensor(command),
                               torch.tensor([(tick * task.cfg.dt / task.cfg.period) % 1]),
                               tensor(jp), tensor(jv), tensor(prev), tensor(meta["default_joint_pos"]))
        observation_errors.append(float(np.max(np.abs(raw - expected_obs.numpy()[0]))))
        with torch.no_grad():
            expected = actor(torch.from_numpy(raw[None])).numpy()
        errors.append(float(np.max(np.abs(expected - session.run(["action"], {"obs": raw[None]})[0]))))
    result = dict(onnx_max_abs_error=max(errors), observation_max_abs_error=max(observation_errors))
    if max(errors) > 2e-5 or max(observation_errors) > 2e-5:
        raise AssertionError(f"Export parity failed: {result}")
    (root / "parity.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    return root


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("checkpoint")
    p.add_argument("output")
    args = p.parse_args()
    export(args.checkpoint, args.output)


if __name__ == "__main__":
    main()
