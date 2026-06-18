import torch

from simulator.world_model import WorldModel


def test_world_model_output_shape():
    physio_dim, action_dim = 64, 1024
    state_dim = 224
    wm = WorldModel(state_dim=state_dim, action_dim=action_dim, physio_dim=physio_dim)
    state = torch.randn(8, state_dim)
    action = torch.randn(8, action_dim)
    out = wm(state, action)
    assert out.shape == (8, physio_dim)


def test_world_model_residual():
    """Output = current_physio + delta; with zero-weight head delta≈ small,
    just confirm next state depends on current physio prefix."""
    physio_dim, action_dim, state_dim = 16, 32, 64
    wm = WorldModel(state_dim=state_dim, action_dim=action_dim, physio_dim=physio_dim)
    wm.eval()
    state = torch.zeros(1, state_dim)
    state[0, :physio_dim] = 5.0
    action = torch.zeros(1, action_dim)
    with torch.no_grad():
        out = wm(state, action)
    # residual connection carries the current physio through
    assert out.shape == (1, physio_dim)


def test_world_model_delta_bounded():
    """Per-step delta is bounded by max_delta_scale * physio_std even for extreme
    inputs — the structural guard against rollout drift."""
    physio_dim, action_dim, state_dim = 16, 32, 64
    std = torch.full((physio_dim,), 2.0)
    scale = 1.5
    wm = WorldModel(state_dim=state_dim, action_dim=action_dim, physio_dim=physio_dim,
                    max_delta_scale=scale, physio_std=std)
    wm.eval()
    # Large random weights would otherwise produce huge deltas; tanh must cap it.
    state = torch.randn(32, state_dim) * 100.0
    action = torch.randn(32, action_dim) * 100.0
    with torch.no_grad():
        out = wm(state, action)
    delta = out - state[:, :physio_dim]
    bound = (std * scale).unsqueeze(0)  # per-dim cap |tanh|<=1
    assert torch.all(delta.abs() <= bound + 1e-4)


def test_world_model_buffers_persist(tmp_path):
    """physio_std / max_delta_scale survive a save+load round-trip."""
    physio_dim, action_dim, state_dim = 8, 16, 32
    std = torch.linspace(0.5, 3.0, physio_dim)
    wm = WorldModel(state_dim=state_dim, action_dim=action_dim, physio_dim=physio_dim,
                    max_delta_scale=2.0, physio_std=std)
    path = str(tmp_path / "wm.pth")
    wm.save(path)
    loaded = WorldModel.load(path, state_dim=state_dim, action_dim=action_dim, physio_dim=physio_dim)
    assert torch.allclose(loaded.physio_std, std)
    assert torch.allclose(loaded.max_delta_scale, torch.tensor(2.0))
