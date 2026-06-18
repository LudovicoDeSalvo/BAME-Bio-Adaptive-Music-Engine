import numpy as np
import faiss
import os

from utils.common import l2_normalize, resolve_path, parse_clip_id

EMBEDDING_PATH = resolve_path("data/processed/song_embeddings.npy")
ID_MAP_PATH = resolve_path("data/processed/song_id_map.npy")


def _collapse_songs(emb, ids):
    """Collapse per-(user,chunk) clip vectors to ONE representative per underlying
    song_id so the retrieval catalog is a clean song set, not many near-duplicate
    crops of the same track.

    clip_id format is "{sid}_{pid}_{song_no}_{chunk}". All crops AND replays of a
    song are averaged (then re-normalized onto the unit sphere) into a single
    song-level vector. Averaging gives a STABLE representative — the old
    "smallest chunk_idx" pick was an arbitrary single crop, so the action vector
    the agent applies at inference could differ noticeably from any one chunk the
    world model trained on. The mean sits at the centroid of the song's crops,
    minimizing that train/inference manifold gap.

    Ids that don't match the 4-field pattern are treated as already-unique (each
    is its own song), so a clean per-song id set passes through unchanged.
    Returns (collapsed_embeddings [M, d] unit-norm, collapsed_ids [M]).
    Insertion order is preserved for determinism.
    """
    groups = {}  # key(sid) -> [row_index, ...]
    order = []
    for i, raw_id in enumerate(ids):
        s = str(raw_id)
        try:
            sid, _pid, _sno, _chunk = parse_clip_id(s)
            key = sid  # collapse all crops/replays of a song to its sid
        except ValueError:
            key = s  # non-conforming id -> already its own unique song
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(i)

    rows = [emb[groups[key]].mean(axis=0) for key in order]
    collapsed = l2_normalize(np.stack(rows), axis=1).astype("float32")
    return collapsed, np.array(order)


class MusicRetrieval:
    def __init__(self):
        self.index = None
        self.song_ids = None
        self.embeddings = None
        self.load_index()

    def load_index(self):

        if not os.path.exists(EMBEDDING_PATH):
            print(f" !!! Warning: embeddings not found at {EMBEDDING_PATH}")
            return

        # load data
        try:
            raw = np.load(EMBEDDING_PATH, allow_pickle=False).astype('float32')
            ids_all = np.load(ID_MAP_PATH, allow_pickle=False)

            # L2-normalize the action manifold: the index, the returned
            # candidate vectors, and the actor's proto-action all live on the
            # unit sphere so cosine (inner-product) retrieval is direction-based,
            # not dominated by raw MERT magnitude.
            emb = l2_normalize(raw, axis=1)

            # Collapse per-(user,chunk) clip duplicates so the agent selects among
            # distinct songs, not many crops of the same track. Each song's crops
            # are averaged into one stable unit-norm representative.
            self.embeddings, self.song_ids = _collapse_songs(emb, ids_all)

            d = self.embeddings.shape[1]
            self.index = faiss.IndexFlatIP(d)  # inner product == cosine on unit vectors
            self.index.add(self.embeddings)

            print(f"MusicRetrieval: index built with {self.index.ntotal} songs (cosine)")
        except Exception as e:
            print(f" !!! Error loading index: {e}")

    def search_candidates(self, query_vector, k=10):
        """
        returns (ids, vectors)
        """
        if self.index is None:
            return [], []

        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        # Normalize the proto-action onto the same unit sphere as the index.
        query_vector = l2_normalize(query_vector.astype('float32'), axis=1)

        dists, indices = self.index.search(query_vector, k)
        
        # indices are [1, k]
        valid_indices = indices[0]
        
        # index is -1 handling (not found)
        valid_indices = [i for i in valid_indices if i != -1 and i < len(self.embeddings)]
        
        if not valid_indices:
            return [], []

        candidate_vectors = self.embeddings[valid_indices] 
        candidate_ids = [self.song_ids[i] for i in valid_indices]
        
        return candidate_ids, candidate_vectors