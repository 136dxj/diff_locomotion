"""CPU-only value/return contracts, independent of simulators and training."""

import pytest
import torch

from src.diff_loco.value import Critic, frozen_parameters, lambda_returns


def test_manual_batched_lambda_returns_and_soft_continuation():
    rewards = torch.tensor([[1., 10.], [2., 20.], [3., 30.]], dtype=torch.float64)
    values = torch.tensor([[4., 40.], [6., 60.], [8., 80.]], dtype=torch.float64)
    continuation = torch.tensor([[1., .5], [.5, 1.], [1., .25]], dtype=torch.float64)
    done = torch.zeros_like(rewards, dtype=torch.bool)
    actual = lambda_returns(rewards, values, continuation, done, done, .5, .5)
    # Hand calculation: last row (7, 40), middle (3.625, 45), then first.
    expected = rewards.new_tensor([[2.90625, 20.625], [3.625, 45.], [7., 40.]])
    torch.testing.assert_close(actual, expected)
    assert actual.shape == rewards.shape and actual.dtype == rewards.dtype


@pytest.mark.parametrize("terminated,truncated,expected", [
    (True, False, [4., 2., 149.5]),
    (False, True, [4.25, 3., 149.5]),
    (True, True, [4., 2., 149.5]),
])
def test_done_retains_reward_uses_true_successor_and_isolates_reset(
    terminated, truncated, expected,
):
    rewards = torch.tensor([[1.], [2.], [100.]])
    values = torch.tensor([[10.], [8.], [99.]])
    continuation = torch.tensor([[1.], [.25], [1.]])
    term = torch.tensor([[False], [terminated], [False]])
    trunc = torch.tensor([[False], [truncated], [False]])
    actual = lambda_returns(rewards, values, continuation, term, trunc, .5, .5)
    torch.testing.assert_close(actual, rewards.new_tensor(expected)[:, None])
    # Timeout: 2 + .5 * .25 * 8 = 3, using its pre-reset successor value.
    # Nothing about the new episode may change either preceding target.
    rewards[2] = 10000
    values[2] = 20000
    changed = lambda_returns(rewards, values, continuation, term, trunc, .5, .5)
    torch.testing.assert_close(changed[:2], actual[:2])
    assert not torch.equal(changed[2], actual[2])


@pytest.mark.parametrize("lam", [0., .4, 1.])
def test_single_step_horizon_bootstrap_is_independent_of_lambda(lam):
    rewards = torch.tensor([[2., 3., 4.]])
    values = torch.tensor([[10., 20., 30.]])
    continuation = torch.tensor([[.25, .5, .75]])
    term = torch.tensor([[False, True, False]])
    trunc = torch.tensor([[False, False, True]])
    actual = lambda_returns(rewards, values, continuation, term, trunc, .8, lam)
    torch.testing.assert_close(actual, rewards.new_tensor([[4., 3., 22.]]))


@pytest.mark.parametrize("gamma,lam,expected", [
    (0., .5, [1., 2.]),
    (.5, 0., [3., 5.]),
    (.5, 1., [3.5, 5.]),
])
def test_discount_and_lambda_endpoints(gamma, lam, expected):
    rewards = torch.tensor([[1.], [2.]])
    values = torch.tensor([[4.], [6.]])
    continuation = torch.ones_like(rewards)
    done = torch.zeros_like(rewards, dtype=torch.bool)
    actual = lambda_returns(rewards, values, continuation, done, done, gamma, lam)
    torch.testing.assert_close(actual, rewards.new_tensor(expected)[:, None])


def test_target_detaches_every_input_and_zero_continuation_retains_reward():
    rewards = torch.tensor([[.02], [.03]], requires_grad=True)
    values = torch.tensor([[4.], [6.]], requires_grad=True)
    continuation = torch.zeros_like(rewards, requires_grad=True)
    done = torch.zeros_like(rewards, dtype=torch.bool)
    actual = lambda_returns(rewards, values, continuation, done, done, .99, .95)
    torch.testing.assert_close(actual, rewards.detach())
    assert not actual.requires_grad and actual.grad_fn is None
    assert all(t.requires_grad and t.grad is None for t in (rewards, values, continuation))


@pytest.mark.parametrize("batch_size", [1, 4])
def test_frozen_critic_weights_preserve_observation_gradients(batch_size):
    critic = Critic(5)
    with torch.no_grad():
        for parameter in critic.parameters():
            parameter.fill_(.01)
    obs = torch.ones(batch_size, 5, requires_grad=True)
    with frozen_parameters(critic):
        values = critic(obs)
        assert values.shape == (batch_size,)
        assert ((values > 0) & (values < 1)).all()
        values.sum().backward()
        assert all(not p.requires_grad and p.grad is None for p in critic.parameters())
    assert all(p.requires_grad for p in critic.parameters())
    assert torch.isfinite(obs.grad).all() and (obs.grad > 0).all()


def test_critic_starts_near_half_with_small_final_weights():
    critic = Critic(5)
    final = critic.net[-1]
    assert (final.weight.abs() <= .01).all()
    assert torch.count_nonzero(final.bias) == 0
    values = critic(torch.zeros(3, 5))
    assert ((values > .45) & (values < .55)).all()
    values.sum().backward()
    assert final.weight.grad is not None and torch.isfinite(final.weight.grad).all()


def test_frozen_parameters_restores_mixed_flags_after_exception():
    critic = Critic(3)
    next(critic.parameters()).requires_grad_(False)
    original = [p.requires_grad for p in critic.parameters()]
    with pytest.raises(RuntimeError, match="interrupted"):
        with frozen_parameters(critic):
            assert not any(p.requires_grad for p in critic.parameters())
            raise RuntimeError("interrupted")
    assert [p.requires_grad for p in critic.parameters()] == original


@pytest.mark.parametrize("obs_dim", [0, -1, True, 2.5])
def test_critic_rejects_invalid_dimensions(obs_dim):
    with pytest.raises(ValueError, match="obs_dim"):
        Critic(obs_dim)


@pytest.fixture
def valid_inputs():
    return dict(
        rewards=torch.ones(2, 2), next_values=torch.ones(2, 2),
        continuation=torch.ones(2, 2), terminated=torch.zeros(2, 2, dtype=torch.bool),
        truncated=torch.zeros(2, 2, dtype=torch.bool), gamma=.99, lam=.95,
    )


@pytest.mark.parametrize("name", ["gamma", "lam"])
@pytest.mark.parametrize("bad", [-.01, 1.01, float("nan"), float("inf"), True, [0.5]])
def test_rejects_invalid_scalars(valid_inputs, name, bad):
    valid_inputs[name] = bad
    with pytest.raises(ValueError, match=name):
        lambda_returns(**valid_inputs)


@pytest.mark.parametrize("bad", [-.01, 1.01, float("nan"), float("inf")])
def test_rejects_invalid_continuation(valid_inputs, bad):
    valid_inputs["continuation"][0, 0] = bad
    with pytest.raises(ValueError, match="continuation"):
        lambda_returns(**valid_inputs)


@pytest.mark.parametrize("name", ["rewards", "next_values", "continuation",
                                  "terminated", "truncated"])
def test_rejects_shape_mismatch(valid_inputs, name):
    valid_inputs[name] = valid_inputs[name][0]
    with pytest.raises(ValueError, match="shape"):
        lambda_returns(**valid_inputs)


def test_rejects_empty_rollout(valid_inputs):
    for name in ("rewards", "next_values", "continuation", "terminated", "truncated"):
        valid_inputs[name] = valid_inputs[name][:0]
    with pytest.raises(ValueError, match="nonempty"):
        lambda_returns(**valid_inputs)


@pytest.mark.parametrize("name", ["terminated", "truncated"])
def test_rejects_non_boolean_done_flags(valid_inputs, name):
    valid_inputs[name] = valid_inputs[name].float()
    with pytest.raises(TypeError, match="boolean"):
        lambda_returns(**valid_inputs)
