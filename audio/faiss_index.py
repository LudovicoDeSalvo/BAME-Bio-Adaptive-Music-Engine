import numpy as np
import faiss
import os

EMBEDDING_PATH = "data/processed/song_embeddings.npy"
ID_MAP_PATH = "data/processed/song_id_map.npy"

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
            self.embeddings = np.load(EMBEDDING_PATH, allow_pickle=True).astype('float32')
            self.song_ids = np.load(ID_MAP_PATH, allow_pickle=True)
            
            d = self.embeddings.shape[1] 
            self.index = faiss.IndexFlatL2(d)
            self.index.add(self.embeddings)

            print(f"MusicRetrieval: index built with {self.index.ntotal} songs")
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
            
        dists, indices = self.index.search(query_vector.astype('float32'), k)
        
        # indices are [1, k]
        valid_indices = indices[0]
        
        # index is -1 handling (not found)
        valid_indices = [i for i in valid_indices if i != -1 and i < len(self.embeddings)]
        
        if not valid_indices:
            return [], []

        candidate_vectors = self.embeddings[valid_indices] 
        candidate_ids = [self.song_ids[i] for i in valid_indices]
        
        return candidate_ids, candidate_vectors