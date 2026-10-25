"""Generic value network and episode-aware, detached critic targets."""

import math
from collections.abc import Iterator
from contextlib import contextmanager
from numbers import Integral, Real

import torch
from torch import Tensor, nn


class Critic(nn.Module):
    """Predict sigmoid-bounded normalized returns without detaching inputs."""

    def __init__(self, obs_dim: int):
        super().__init__()
        if isinstance(obs_dim, bool) or not isinstance(obs_dim, Integral) or obs_dim <= 0:
            raise ValueError("obs_dim must be a positive integer")
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128), nn.ELU(),
            nn.Linear(128, 128), nn.ELU(),
            nn.Linear(128, 64), nn.ELU(), nn.Linear(64, 1),
        )
        nn.init.uniform_(self.net[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, obs: Tensor) -> Tensor:
        return self.net(obs).squeeze(-1).sigmoid()


@contextmanager
def frozen_parameters(module: nn.Module) -> Iterator[None]:
    """Freeze weights, preserve input gradients, and restore every original flag."""
    parameters = tuple(module.parameters())
    flags = tuple(parameter.requires_grad for parameter in parameters)
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, flag in zip(parameters, flags):
            parameter.requires_grad_(flag)


@torch.no_grad()
def lambda_returns(
    rewards: Tensor, next_values: Tensor, continuation: Tensor,
    terminated: Tensor, truncated: Tensor, gamma: float, lam: float,
) -> Tensor:
    """Return detached [T, B] targets in rewards' dtype/device and reward units.

    All tensors share a nonempty [T, B] shape and device; rewards, next_values
    and continuation are floating point, while the two done flags are boolean.
    next_values[t] must value the true successor BEFORE any environment reset.
    Termination retains only the current reward. Truncation bootstraps from
    that successor and cuts recursion into the next episode. If both flags
    are set, termination wins. The final horizon step also uses its successor.
    No reward rescaling or gradient propagation is performed here.
    """
    if rewards.ndim != 2 or 0 in rewards.shape:
        raise ValueError("rewards must have nonempty shape [T, B]")
    for name, tensor in (("next_values", next_values), ("continuation", continuation),
                         ("terminated", terminated), ("truncated", truncated)):
        if tensor.shape != rewards.shape:
            raise ValueError(f"{name} must have the same [T, B] shape as rewards")
        if tensor.device != rewards.device:
            raise ValueError(f"{name} must be on the same device as rewards")
    if not all(t.is_floating_point() for t in (rewards, next_values, continuation)):
        raise TypeError("rewards, next_values and continuation must be floating point")
    if terminated.dtype != torch.bool or truncated.dtype != torch.bool:
        raise TypeError("terminated and truncated must be boolean tensors")
    for name, scalar in (("gamma", gamma), ("lam", lam)):
        if (isinstance(scalar, bool) or not isinstance(scalar, Real)
                or not math.isfinite(scalar) or not 0 <= scalar <= 1):
            raise ValueError(f"{name} must be a finite scalar in [0, 1]")
    if not bool(((continuation >= 0) & (continuation <= 1)).all()):
        raise ValueError("continuation must contain finite values in [0, 1]")

    targets = torch.empty_like(rewards)
    future = next_values[-1]
    for t in range(rewards.shape[0] - 1, -1, -1):
        bootstrap = torch.where(
            truncated[t], next_values[t], (1 - lam) * next_values[t] + lam * future,
        )
        future = torch.where(
            terminated[t], rewards[t], rewards[t] + gamma * continuation[t] * bootstrap,
        )
        targets[t] = future
    return targets
