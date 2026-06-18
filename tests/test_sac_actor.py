import numpy as np
import torch

from rl.sac_agent import SACAgent


def test_actor_action_bounded():
    """tanh squashing must keep |action| < 1 on every dim."""
    agent = SACAgent(state_dim=32, action_dim=64)
    state = torch.randn(16, 32)
    action, log_prob, det = agent.actor.sample(state)
    assert action.shape == (16, 64)
    assert torch.all(action.abs() <= 1.0)
    assert torch.all(det.abs() <= 1.0)


def test_logprob_shape_and_finite():
    """log_prob is summed over action dims -> shape [batch], all finite."""
    agent = SACAgent(state_dim=8, action_dim=128)
    state = torch.randn(10, 8)
    _, log_prob, _ = agent.actor.sample(state)
    assert log_prob.shape == (10,)
    assert torch.all(torch.isfinite(log_prob))


def test_select_action_eval_bounded():
    agent = SACAgent(state_dim=12, action_dim=20)
    a = agent.select_action(np.random.randn(12).astype(np.float32), evaluate=True)
    assert a.shape == (20,)
    assert np.all(np.abs(a) <= 1.0)
