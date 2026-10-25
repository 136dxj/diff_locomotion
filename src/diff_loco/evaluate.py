"""Independent CPU MuJoCo evaluation of exported actors, without any robot SDK."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from .runtime import Policy


def evaluate(args):
    policy = Policy(args.bundle)
    meta = policy.meta
    model = mujoco.MjModel.from_xml_path(str(Path(args.bundle) / "scene/scene.xml"))
    model.opt.timestep = meta["physics_dt"]
    for name, value in meta["physics_options"].items():
        setattr(model.opt, name, value)
        assert getattr(model.opt, name) == value
    data = mujoco.MjData(model)
    rng = np.random.default_rng(args.seed)
    fields = ("body_mass", "body_inertia", "body_ipos", "dof_armature", "dof_damping",
              "geom_friction", "actuator_gainprm", "actuator_biasprm", "actuator_forcerange")
    nominal = {f: getattr(model, f).copy() for f in fields}
    n, dt = meta["num_actions"], meta["control_dt"]
    command = np.array([args.vx, args.vy, args.yaw], dtype=np.float32)
    renderer = writer = None
    if args.video:
        import imageio.v2 as imageio
        from PIL import Image, ImageDraw

        Path(args.video).parent.mkdir(parents=True, exist_ok=True)
        renderer = mujoco.Renderer(model, height=480, width=640)
        writer = imageio.get_writer(args.video, fps=50, codec="libx264",
                                    ffmpeg_params=["-movflags", "+faststart"])
        camera = mujoco.MjvCamera()
        camera.distance, camera.azimuth, camera.elevation = 1.8, 135, -20
    result = []
    for episode in range(args.episodes):
        for f, value in nominal.items():
            getattr(model, f)[:] = value
        bias, delay = np.zeros(n), 0
        if args.randomized:
            scale = rng.uniform(0.85, 1.15, model.nbody)
            model.body_mass[:] *= scale
            model.body_inertia[:] *= scale[:, None]
            model.body_ipos[model.jnt_bodyid[0]] += rng.uniform(-0.02, 0.02, 3)
            model.dof_armature[:] *= rng.uniform(0.85, 1.15)
            model.dof_damping[:] *= rng.uniform(0.75, 1.25)
            model.geom_friction[:, 0] = rng.uniform(0.45, 1.35)
            kp_scale, kd_scale = rng.uniform(0.75, 1.25, (2, n))
            model.actuator_gainprm[:, 0] *= kp_scale
            model.actuator_biasprm[:, 1] *= kp_scale
            model.actuator_biasprm[:, 2] *= kd_scale
            model.actuator_forcerange[:] *= rng.uniform(0.8, 1.0, (n, 1))
            bias, delay = rng.uniform(-0.01, 0.01, n), int(rng.integers(3))
        mujoco.mj_setConst(model, data)
        mujoco.mj_resetData(model, data)
        data.qpos[:] = meta["initial_qpos"]
        if args.randomized:
            data.qpos[7:] += rng.uniform(-0.02, 0.02, n)
        data.ctrl[:] = policy.q0[np.array(meta["actuator_joint_indices"])]
        mujoco.mj_forward(model, data)
        policy.reset()
        history = [policy.q0.copy(), policy.q0.copy()]
        errors, velocities, heights = [], [], []
        start_xy = data.qpos[:2].copy()
        fallen = False
        failure_reason = None
        for step in range(round(args.seconds / dt)):
            gyro, jp, jv = data.qvel[3:6].copy(), data.qpos[7:].copy() + bias, data.qvel[6:].copy()
            if args.randomized:
                gyro += rng.uniform(-0.03, 0.03, 3)
                jp += rng.uniform(-0.005, 0.005, n)
                jv += rng.uniform(-0.2, 0.2, n)
            target = policy.act(gyro, data.qpos[3:7], jp, jv, command)
            if args.zero_action:
                target = policy.q0.copy()
            applied = [target, *history][delay]
            history = [target, history[0]]
            data.ctrl[:] = applied[np.array(meta["actuator_joint_indices"])]
            if args.randomized and step > 0 and step % round(5 / dt) == 0:
                data.qvel[:2] += rng.uniform(-0.35, 0.35, 2)
            for _ in range(meta["substeps"]):
                previous_time = data.time
                mujoco.mj_step(model, data)
                bad = any(data.warning[int(w)].number for w in (
                    mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL,
                    mujoco.mjtWarning.mjWARN_BADQACC))
                if bad or not np.isclose(data.time - previous_time, model.opt.timestep):
                    failure_reason = "invalid_physics_or_automatic_reset"
                    break
            rotation = np.empty(9)
            mujoco.mju_quat2Mat(rotation, data.qpos[3:7])
            body_vel = rotation.reshape(3, 3).T @ data.qvel[:3]
            errors.append(float(np.sum((body_vel[:2] - command[:2]) ** 2)))
            velocities.append(body_vel.tolist())
            heights.append(float(data.qpos[2]))
            fallen = bool(failure_reason or data.qpos[2] < meta["initial_qpos"][2] * 0.55 or rotation[8] < 0.45
                          or not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all())
            if writer and (step % 2 == 0 or fallen):
                camera.lookat[:] = data.qpos[:3]
                camera.lookat[2] = meta["initial_qpos"][2] * 0.7
                renderer.update_scene(data, camera=camera)
                frame = Image.fromarray(renderer.render())
                draw = ImageDraw.Draw(frame)
                draw.rectangle((0, 0, 640, 55), fill=(20, 24, 30))
                draw.text((12, 8), f"{meta['robot'].upper()} | iteration {meta['iteration']} | "
                          f"episode {episode + 1} | t={(step + 1) * dt:.2f}s", fill="white")
                draw.text((12, 30), f"command vx={args.vx:.2f} m/s | measured vx={body_vel[0]:.2f} m/s | "
                          f"{'FALL' if fallen else 'RUNNING'}", fill="orange" if fallen else "white")
                writer.append_data(np.asarray(frame))
                if fallen:
                    for _ in range(50):  # Hold the failure frame for one second.
                        writer.append_data(np.asarray(frame))
            if fallen:
                break
        result.append(dict(episode=episode, survived_seconds=(step + 1) * dt, fallen=fallen,
                           tracking_rmse=float(np.sqrt(np.mean(errors))),
                           mean_body_velocity=np.mean(velocities, axis=0).tolist(),
                           displacement_xy=(data.qpos[:2] - start_xy).tolist(),
                           min_height=min(heights), latency_ticks=delay,
                           failure_reason=failure_reason or ("fall" if fallen else None)))
    if writer:
        writer.close()
        renderer.close()
    report = dict(robot=meta["robot"], checkpoint_iteration=meta["iteration"],
                  randomized=args.randomized, zero_action=args.zero_action, seed=args.seed,
                  command=command.tolist(), requested_seconds=args.seconds, episodes=result,
                  survival_fraction=float(np.mean([not e["fallen"] for e in result])),
                  mean_tracking_rmse=float(np.mean([e["tracking_rmse"] for e in result])))
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("bundle")
    p.add_argument("--output", required=True)
    p.add_argument("--episodes", type=int, default=8)
    p.add_argument("--seconds", type=float, default=10)
    p.add_argument("--vx", type=float, default=0.3)
    p.add_argument("--vy", type=float, default=0)
    p.add_argument("--yaw", type=float, default=0)
    p.add_argument("--seed", type=int, default=2027)
    p.add_argument("--randomized", action="store_true")
    p.add_argument("--zero-action", action="store_true")
    p.add_argument("--video", help="Optional MP4 of all episodes, 50 fps at the 100 Hz control rate")
    args = p.parse_args()
    if args.episodes < 1 or args.seconds <= 0:
        p.error("episodes and seconds must be positive")
    evaluate(args)


if __name__ == "__main__":
    main()
