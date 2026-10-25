"""Aligned short-horizon actor-critic with model-derived dynamics and IK RSI."""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import torch

from .model import RobotModel
from .policy import Actor, align, quat_inverse_rotate
from .reference import GaitReference
from .task import LocomotionTask, TaskCfg
from .value import Critic, frozen_parameters, lambda_returns


class WalkingLoss:
    def __init__(self, task, dynamics):
        self.task, self.dynamics = task, dynamics
        self.duty = 0.65 if len(task.foot_sites) == 2 else 0.6
        self.reference = GaitReference(dynamics, task.initial, task.limits, task.scale,
                                       task.offsets, task.cfg.period, task.clearance, self.duty,
                                       flat_feet=len(task.foot_sites) == 2, dt=task.cfg.dt)
        self.height = self.reference.height
        self.weights = dict(track=2.0, yaw=0.5, upright=4.0, height=30.0, gait=40.0,
                            safety=10.0, balance=2.0 if len(task.foot_sites) == 2 else 0.1,
                            vertical=0.05, smooth=0.02, pose=0.03, speed=0.0005,
                            effort=0.005, limits=1.0, slip=0.1)

    def __call__(self, q, v, action, previous, target, kp, kd, effort):
        t = self.task
        command = t.commands.clone()
        moving = (command.norm(dim=-1) >= 0.1).float()
        vel = quat_inverse_rotate(q[:, 3:7], v[:, :3])
        gravity = quat_inverse_rotate(q[:, 3:7], q.new_tensor([0, 0, -1]).expand(len(q), -1))
        k = self.dynamics.kinematics(q)
        feet = k["foot_pos"]
        relative = feet - q[:, None, :3]
        relative = quat_inverse_rotate(q[:, None, 3:7].expand(-1, len(t.foot_sites), -1), relative)
        phase = (t.phase()[:, None] + t.offsets) % 1
        stance = (phase < self.duty).float()
        desired_feet = self.reference.targets(t.phase(), command)
        # Ground-relative vertical targets agree with the contact-derived root height.
        gait_error = torch.cat((relative[..., :2] - desired_feet[..., :2],
                               (feet[..., 2] - desired_feet[..., 2] - self.height)[..., None]), -1)
        support = stance * moving[:, None] + (1 - moving[:, None])
        center = (feet[:, :, :2] * support[..., None]).sum(1) / support.sum(1, keepdim=True).clamp_min(1)
        contact_velocity = (k["contact_jac"] @ v[:, None, :, None]).squeeze(-1)
        contact = (k["contact_pos"][..., 2] < 0.005).float()
        torque = (kp * (target - q[:, 7:]) - kd * v[:, 6:]) / effort.clamp_min(1)
        margin = 0.05 * (t.limits[:, 1] - t.limits[:, 0])
        joint_violation = ((t.limits[:, 0] + margin - q[:, 7:]).clamp_min(0)
                           + (q[:, 7:] - t.limits[:, 1] + margin).clamp_min(0))
        terms = {
            "track": (vel[:, :2] - command[:, :2]).square().sum(-1),
            "yaw": (v[:, 5] - command[:, 2]).square(),
            "upright": gravity[:, :2].square().sum(-1) + (gravity[:, 2] + 1).square(),
            "height": (q[:, 2] - self.height).square(),
            "gait": gait_error.square().mean((1, 2)),
            # A binary fall flag has no pathwise action gradient. Act before failure.
            "safety": ((0.85 * self.height - q[:, 2]).clamp_min(0) / (0.3 * self.height)).square()
                      + ((gravity[:, 2] + 0.94).clamp_min(0) / 0.49).square(),
            "balance": (q[:, :2] - center).square().sum(-1),
            "vertical": v[:, 2].square() + 0.1 * v[:, 3:5].square().sum(-1),
            "smooth": (action - previous).square().mean(-1),
            "pose": (q[:, 7:] - t.q0).square().mean(-1),
            "speed": v[:, 6:].square().mean(-1),
            "effort": torque.square().mean(-1),
            "limits": joint_violation.square().mean(-1),
            "slip": (contact_velocity[..., :2].square().sum(-1) * contact).mean(-1),
        }
        return sum(self.weights[name] * value for name, value in terms.items()), terms


def delayed_action(action, history, delay):
    stack = torch.stack([action, *history], 1)
    return stack[torch.arange(len(action), device=action.device), delay.clone()]


def critic_observation(task, q, v, history, actor_scale, context):
    proprio = (task.observation(q, v, history[0]) * actor_scale).clamp(-10, 10)
    velocity = quat_inverse_rotate(q[:, 3:7], v[:, :3])
    return torch.cat((proprio, 2 * velocity, q[:, 2:3] / task.height - 1,
                      history[1], context), -1)


def soft_continuation(q, height):
    """Analytic height/tilt survival; this is not a learned world model."""
    gravity = quat_inverse_rotate(q[:, 3:7], q.new_tensor([0, 0, -1]).expand(len(q), -1))
    return torch.sigmoid((q[:, 2] / height - 0.55) / 0.05) * torch.sigmoid((-gravity[:, 2] - 0.45) / 0.05)


def train(args):
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    state = torch.load(args.resume, map_location=args.device, weights_only=False) if args.resume else None
    # Explicit CLI overrides win; an ordinary resume preserves checkpoint gait settings.
    for name in ("clearance", "action_scale"):
        value = getattr(args, name, None)
        if value is None and state is not None:
            value = state["task"].get(name)
        setattr(args, name, value)
    cfg = TaskCfg(robot=args.robot, num_envs=args.num_envs, randomize=not args.nominal, device=args.device,
                  max_speed=args.max_speed, rsi_probability=args.rsi,
                  clearance=args.clearance, action_scale=args.action_scale)
    task = LocomotionTask(cfg)
    dynamics = RobotModel(task.sim.mj_model, foot_sites=task.foot_sites,
                          contact_geoms=task.contact_geoms, device=cfg.device).float()
    loss_fn = WalkingLoss(task, dynamics)
    task.install_reference(loss_fn.reference)
    task.reset(torch.arange(cfg.num_envs, device=cfg.device))
    actor = Actor(task.n).to(cfg.device)
    optimizer = torch.optim.Adam(actor.parameters(), lr=args.lr)
    context = task.critic_context(dynamics.contact_geom_ids)
    history = [task.initial_actions[:, i].clone() for i in range(2)]
    q, v = task.state()
    features = critic_observation(task, q, v, history, actor.obs_scale, context)
    critic = Critic(features.shape[-1]).to(cfg.device)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    gamma = math.exp(-cfg.dt / args.discount_seconds)
    first = 0
    if state is not None:
        if state["task"]["robot"] != args.robot:
            raise ValueError("Checkpoint robot does not match task")
        if args.robot == "g1" and "initial_qpos" not in state:
            raise ValueError("Legacy G1 checkpoint uses a different nominal pose; start a fresh run")
        if "initial_qpos" in state and not torch.allclose(state["initial_qpos"][7:], task.q0):
            raise ValueError("Checkpoint nominal joint pose does not match task")
        actor.load_state_dict(state["actor"])
        optimizer.load_state_dict(state["optimizer"])
        if "critic" in state:
            critic.load_state_dict(state["critic"])
            critic_optimizer.load_state_dict(state["critic_optimizer"])
        first = state["iteration"]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(dict(task=asdict(cfg), training=vars(args),
        loss_weights=loss_fn.weights, gamma=gamma, horizon_seconds=args.horizon * cfg.dt,
        reward="exp(-weighted_cost); critic uses (1-gamma)*reward", critic_observations=features.shape[-1]), indent=2))
    start = time.monotonic()
    with (out / "metrics.jsonl").open("a", buffering=1) as log:
        for iteration in range(first + 1, first + args.updates + 1):
            tick = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            critic_optimizer.zero_grad(set_to_none=True)
            q, v = q.detach(), v.detach()
            history = [x.detach() for x in history]
            params, kp, kd, effort = task.parameters(dynamics.contact_geom_ids)
            context = task.critic_context(dynamics.contact_geom_ids)
            segment = q.new_zeros(cfg.num_envs)
            weight = q.new_ones(cfg.num_envs)
            objective = q.new_zeros(())
            segments = 0
            rollout = {key: [] for key in ("obs", "reward", "next_value", "continuation", "fallen", "timeout")}
            metrics = {}
            falls = resets = falls_rsi = 0
            completed_seconds = fallen_seconds = mean_reward = mean_cost = 0.0
            with frozen_parameters(critic):
                for step in range(args.horizon):
                    rollout["obs"].append(critic_observation(task, q, v, history, actor.obs_scale, context).detach())
                    previous = history[0]
                    action = actor(task.observation(q, v, previous, noise=True))
                    applied = delayed_action(action, history, task.delay)
                    target = task.target(applied)
                    predicted_q, predicted_v = dynamics.step(q, v, target, kp, kd, effort, cfg.dt, parameters=params)
                    actual_q, actual_v = task.step(target)
                    if not (torch.isfinite(actual_q).all() and torch.isfinite(actual_v).all()):
                        raise FloatingPointError(f"Non-finite simulator state at iteration {iteration}")
                    q, v = align(actual_q, predicted_q, args.alpha), align(actual_v, predicted_v, args.alpha)
                    cost, terms = loss_fn(q, v, action, previous, target, kp, kd, effort)
                    reward = torch.exp(-cost)
                    continuation = soft_continuation(q, task.height)
                    fallen, timeout = task.termination(actual_q)
                    done = fallen | timeout
                    history = [action, previous]
                    # Always evaluate the true successor, before reset changes phase or DR.
                    successor = critic_observation(task, q, v, history, actor.obs_scale, context)
                    next_value = critic(successor)
                    scaled_reward = (1 - gamma) * reward
                    segment = segment + weight * scaled_reward
                    weight = weight * gamma * continuation
                    boundary = done | (step == args.horizon - 1)
                    objective = objective + ((segment + weight * next_value * (~fallen))[boundary]).sum()
                    segments += int(boundary.sum())
                    segment = torch.where(boundary, torch.zeros_like(segment), segment)
                    weight = torch.where(boundary, torch.ones_like(weight), weight)
                    for name, value in (("reward", scaled_reward), ("next_value", next_value),
                                        ("continuation", continuation), ("fallen", fallen), ("timeout", timeout)):
                        rollout[name].append(value.detach())
                    mean_reward += float(reward.detach().mean()) / args.horizon
                    mean_cost += float(cost.detach().mean()) / args.horizon
                    for name, value in terms.items():
                        metrics[name] = metrics.get(name, 0) + value.detach().mean() / args.horizon
                    falls += int(fallen.sum())
                    resets += int(done.sum())
                    falls_rsi += int((fallen & task.from_reference).sum())
                    completed_seconds += float(task.age[done].sum()) * cfg.dt
                    fallen_seconds += float(task.age[fallen].sum()) * cfg.dt
                    if done.any():
                        task.reset(done.nonzero().flatten())
                        reset_q, reset_v = task.state()
                        q, v = torch.where(done[:, None], reset_q, q), torch.where(done[:, None], reset_v, v)
                        history = [torch.where(done[:, None], task.initial_actions[:, i], x)
                                   for i, x in enumerate(history)]
                        params, kp, kd, effort = task.parameters(dynamics.contact_geom_ids)
                        context = task.critic_context(dynamics.contact_geom_ids)
                # Normalize the window's discounted reward mass to keep gradient scale usable.
                actor_loss = -objective / segments / (1 - gamma ** args.horizon)
                if not torch.isfinite(actor_loss):
                    raise FloatingPointError(f"Non-finite actor loss at iteration {iteration}")
                actor_loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(actor.parameters(), args.grad_clip, error_if_nonfinite=True)
                optimizer.step()
            batch = {key: torch.stack(value) for key, value in rollout.items()}
            returns = lambda_returns(batch["reward"], batch["next_value"], batch["continuation"],
                                     batch["fallen"], batch["timeout"], gamma, args.lam)
            inputs, targets = batch["obs"].flatten(0, 1), returns.flatten()
            for _ in range(args.critic_epochs):
                critic_optimizer.zero_grad(set_to_none=True)
                critic_loss = (critic(inputs) - targets).square().mean()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), args.grad_clip, error_if_nonfinite=True)
                critic_optimizer.step()
            row = dict(iteration=iteration, loss=float(actor_loss.detach()), cost=mean_cost, reward=mean_reward,
                       critic_loss=float(critic_loss.detach()), value_target=float(targets.mean()),
                       gradient_norm=float(grad), falls_from_reference=falls_rsi, falls_from_stand=falls-falls_rsi,
                       falls=falls, resets=resets, elapsed_seconds=time.monotonic() - start,
                       iteration_seconds=time.monotonic() - tick,
                       mean_live_episode_seconds=float(task.age.float().mean()) * cfg.dt,
                       mean_completed_episode_seconds=completed_seconds / resets if resets else None,
                       mean_fallen_episode_seconds=fallen_seconds / falls if falls else None,
                       weighted_terms={name: float(value) * loss_fn.weights[name] for name, value in metrics.items()},
                       **{name: float(value) for name, value in metrics.items()})
            log.write(json.dumps(row) + "\n")
            if iteration == first + 1 or iteration % args.log_every == 0 or iteration == first + args.updates:
                print(json.dumps(row), flush=True)
            if iteration % args.save_every == 0 or iteration == first + args.updates:
                checkpoint = out / "last.pt.tmp"
                torch.save(dict(actor=actor.state_dict(), optimizer=optimizer.state_dict(),
                                critic=critic.state_dict(), critic_optimizer=critic_optimizer.state_dict(),
                                initial_qpos=task.initial.cpu(), task=asdict(cfg), iteration=iteration,
                                training=vars(args)), checkpoint)
                checkpoint.replace(out / "last.pt")
    return out / "last.pt"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--robot", choices=["go1", "g1"], required=True)
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--updates", type=int, default=500)
    p.add_argument("--horizon", type=int, default=24)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--critic-lr", type=float, default=1e-3)
    p.add_argument("--critic-epochs", type=int, default=4)
    p.add_argument("--discount-seconds", type=float, default=1.0, help="Discount e-folding time in seconds")
    p.add_argument("--lam", type=float, default=0.95, help="TD(lambda) mixture for detached critic targets")
    p.add_argument("--alpha", type=float, default=0.95)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--output", required=True)
    p.add_argument("--nominal", action="store_true")
    p.add_argument("--max-speed", type=float, default=0.25, help="Forward command limit in m/s; bounded Go1 IK supports 0.25")
    p.add_argument("--rsi", type=float, default=0.8, help="Probability of an IK gait-reference reset; 0 disables RSI")
    p.add_argument("--clearance", type=float, help="Swing lift in metres; defaults to checkpoint or robot setting")
    p.add_argument("--action-scale", type=float, help="Joint target amplitude in radians; defaults to checkpoint or robot setting")
    p.add_argument("--resume")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=50)
    args = p.parse_args()
    if args.horizon < 1 or args.updates < 1 or args.num_envs < 1:
        p.error("horizon, updates and num-envs must be positive")
    if args.log_every < 1 or args.save_every < 1:
        p.error("log-every and save-every must be positive")
    if not 0 <= args.rsi <= 1 or args.max_speed <= 0:
        p.error("rsi must lie in [0,1], and max-speed must be positive")
    if args.discount_seconds <= 0 or not 0 <= args.lam <= 1 or args.critic_epochs < 1:
        p.error("discount-seconds and critic-epochs must be positive, lam must lie in [0,1]")
    if not 0 <= args.alpha <= 1 or min(args.lr, args.critic_lr, args.grad_clip) <= 0:
        p.error("alpha must lie in [0,1], learning rates and grad-clip must be positive")
    train(args)


if __name__ == "__main__":
    main()
