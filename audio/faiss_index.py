import numpy as np
import faiss
import os

EMBEDDING_PATH = "data/processed/song_embeddings.npy"
ID_MAP_PATH = "data/processed/song_id_map.npy"

class MusicRetrieval:
    def __init__(self):
        self.index = None
        self.song_ids = None
        self.load_index()

    def load_index(self):
        """Loads pre-computed embeddings and builds FAISS index."""
        if not os.path.exists(EMBEDDING_PATH):
            raise FileNotFoundError(f"Run 'process-audio' first! Missing: {EMBEDDING_PATH}")

        # Load data
        self.embeddings = np.load(EMBEDDING_PATH).astype('float32')
        self.song_ids = np.load(ID_MAP_PATH)
        
        # Dimension of MERT embeddings (usually 768 or 1024 depending on layer)
        d = self.embeddings.shape[1] 
        
        # Build Index (L2 Distance = Euclidean)
        self.index = faiss.IndexFlatL2(d)
        self.index.add(self.embeddings)
        print(f"MusicRetrieval: Index built with {self.index.ntotal} songs.")

    def search(self, query_vector, k=1):
        """
        Input: query_vector (numpy array of shape (1, dim))
        Output: list of song_ids
        """
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
            
        distances, indices = self.index.search(query_vector.astype('float32'), k)
        
        # Map indices back to filenames/IDs
        retrieved_ids = [self.song_ids[i] for i in indices[0]]
        return retrieved_ids

# Test functionality
if __name__ == "__main__":
    retriever = MusicRetrieval()
    # Fake query
    fake_vec = np.random.rand(1, 768).astype('float32')
    result = retriever.search(fake_vec)
    print(f"Test Search Result: Nearest song is {result}")