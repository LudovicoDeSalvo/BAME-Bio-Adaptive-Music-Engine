import torch
import numpy as np
from audio.faiss_index import MusicRetrieval

class WolpertingerPolicy:
    def __init__(self, k_neighbors=10, device="cpu"):
        self.retriever = MusicRetrieval()
        self.k = k_neighbors
        self.device = device

    def select_action(self, proto_action, critic, state_tensor):
        """      
        args:
            proto_action: output from actor [1, 1024]
            critic: SAC critic network
            state_tensor: current state [1, state_dim]
            
        returns:
            best_action: chosen song embeddings
            song_id: chosen song ID
        """
        # FAISS: get k nearest real songs
        ids, vectors = self.retriever.search_candidates(proto_action, k=self.k)
        
        # critic eval
        k = len(vectors)
        
        expanded_state = state_tensor.repeat(k, 1)
        
        candidate_tensor = torch.FloatTensor(vectors).to(self.device)
        
        with torch.no_grad():
            q1, q2 = critic(expanded_state, candidate_tensor)
            min_q = torch.min(q1, q2)
            
        # selection
        best_idx = torch.argmax(min_q).item()
        
        best_song_id = ids[best_idx]
        best_action_vec = vectors[best_idx]
        
        return best_action_vec, best_song_id