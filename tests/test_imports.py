import importlib

import pytest

MODULES = [
    "utils.common",
    "data.windows",
    "audio.faiss_index",
    "audio.mert_embedder",
    "rl.sac_agent",
    "rl.train_agent",
    "rl.wolpertinger",
    "simulator.world_model",
    "simulator.gym_env",
    "simulator.train_simulator",
    "context.sequence_model",
    "user.train_profile",
    "user.dcn_profile",
    "physio.encoder",
    "scripts.align_and_slice",
    "scripts.inference",
]


@pytest.mark.parametrize("mod", MODULES)
def test_import_smoke(mod):
    importlib.import_module(mod)
