import numpy as np
import pytest

from utils.common import set_seed


def _fake_pools(self):
    rng = np.random.default_rng(0)
    self.physio_pool = rng.standard_normal((12, self.physio_dim)).astype(np.float32)
    self.user_pool = rng.standard_normal((7, self.user_dim)).astype(np.float32)
    self._physio_std = np.std(self.physio_pool, axis=0).astype(np.float32)
    margin = 0.5 * self._physio_std
    self._physio_min = self.physio_pool.min(axis=0).astype(np.float32) - margin
    self._physio_max = self.physio_pool.max(axis=0).astype(np.float32) + margin


@pytest.fixture
def make_env(monkeypatch):
    from simulator import gym_env
    monkeypatch.setattr(gym_env.MusicEnv, "_load_pools", _fake_pools)

    def _build():
        set_seed(42)  # so the random-init world/context models match across builds
        return gym_env.MusicEnv()
    return _build


def test_reset_step_contract(make_env):
    env = make_env()
    state, info = env.reset(seed=1)
    assert state.shape == (env.state_dim,)
    assert state.dtype == np.float32
    action = env.action_space.sample()
    nxt, reward, terminated, truncated, _ = env.step(action)
    assert nxt.shape == (env.state_dim,)
    assert np.isfinite(reward)
    assert isinstance(terminated, bool)


def test_seed_determinism(make_env):
    env1 = make_env()
    env2 = make_env()
    s1, _ = env1.reset(seed=7)
    s2, _ = env2.reset(seed=7)
    assert np.allclose(s1, s2)

    env1.action_space.seed(7)
    env2.action_space.seed(7)
    for _ in range(5):
        a1 = env1.action_space.sample()
        a2 = env2.action_space.sample()
        assert np.allclose(a1, a2)
        n1, r1, _, _, _ = env1.step(a1)
        n2, r2, _, _, _ = env2.step(a2)
        assert np.allclose(n1, n2, atol=1e-5)
        assert np.isclose(r1, r2, atol=1e-6)


def test_reward_nan_guard(make_env):
    env = make_env()
    env.reset(seed=3)
    env.current_physio = np.full(env.physio_dim, np.nan, dtype=np.float32)
    env.target_physio = np.zeros(env.physio_dim, dtype=np.float32)
    r = env._calculate_reward()
    assert np.isfinite(r)


def test_step_handles_nonfinite_action(make_env):
    env = make_env()
    env.reset(seed=4)
    bad = np.full(env.action_dim, np.inf, dtype=np.float32)
    nxt, reward, _, _, _ = env.step(bad)
    assert np.all(np.isfinite(nxt))
    assert np.isfinite(reward)
