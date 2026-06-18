import numpy as np

import audio.faiss_index as fi
from audio.faiss_index import MusicRetrieval


def _make_index(tmp_path, monkeypatch, n=20, d=1024):
    emb = np.random.RandomState(0).randn(n, d).astype(np.float32) * 13.0
    ids = np.array([f"song_{i}" for i in range(n)])
    epath = tmp_path / "emb.npy"
    ipath = tmp_path / "ids.npy"
    np.save(epath, emb)
    np.save(ipath, ids)
    monkeypatch.setattr(fi, "EMBEDDING_PATH", str(epath))
    monkeypatch.setattr(fi, "ID_MAP_PATH", str(ipath))
    return n, d


def test_index_is_normalized_cosine(tmp_path, monkeypatch):
    n, d = _make_index(tmp_path, monkeypatch)
    r = MusicRetrieval()
    assert r.index is not None
    assert r.index.ntotal == n
    # stored embeddings are on the unit sphere
    norms = np.linalg.norm(r.embeddings, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_search_shapes_and_unit_candidates(tmp_path, monkeypatch):
    n, d = _make_index(tmp_path, monkeypatch)
    r = MusicRetrieval()
    query = np.random.randn(d).astype(np.float32) * 50.0  # unnormalized proto
    ids, vecs = r.search_candidates(query, k=5)
    assert len(ids) == 5
    assert vecs.shape == (5, d)
    # returned candidate vectors are unit-norm (match world-model action space)
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)
