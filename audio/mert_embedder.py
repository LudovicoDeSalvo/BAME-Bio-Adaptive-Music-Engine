import os
import torch
import torchaudio
import numpy as np
import glob
from transformers import Wav2Vec2FeatureExtractor, AutoModel
from tqdm import tqdm

from utils.common import resolve_path

# --- Configuration ---

MODEL_NAME = "m-a-p/MERT-v1-330M"
TARGET_SR = 24000
AUDIO_DIR = resolve_path("data/processed/audio_clips")
OUTPUT_PATH = resolve_path("data/processed/song_embeddings.npy")
ID_MAP_PATH = resolve_path("data/processed/song_id_map.npy")

class MERTExtractor:

    def __init__(self, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f">> loading MERT model on {self.device}...")
        
        try:
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to(self.device)
            self.model.eval()

        except OSError:
            print(f" !!! Connection error")
            raise

    def process_audio(self, file_path):
        """ loads audio, resamples, and extracts embedding"""
        try:
            # audio loading
            waveform, sample_rate = torchaudio.load(file_path)
            
            # resampling
            if sample_rate != TARGET_SR:
                resampler = torchaudio.transforms.Resample(sample_rate, TARGET_SR)
                waveform = resampler(waveform)
            
            # mono mix
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # truncation
            max_samples = 30 * TARGET_SR
            if waveform.shape[1] > max_samples:
                waveform = waveform[:, :max_samples]
                
            inputs = self.processor(waveform.squeeze().numpy(), sampling_rate=TARGET_SR, return_tensors="pt")
            input_values = inputs["input_values"].to(self.device)

            # inference
            with torch.no_grad():
                outputs = self.model(input_values)
                last_hidden_state = outputs.last_hidden_state
                embedding = torch.mean(last_hidden_state, dim=1).squeeze().cpu().numpy()
                
            return embedding

        except Exception:
            # Let the extraction loop log + skip this file once (it already does);
            # re-raising here without printing avoids the double error message.
            raise

def extract_all_embeddings():

    if not os.path.exists(resolve_path("data/processed")):
        os.makedirs(resolve_path("data/processed"))

    if not os.path.exists(AUDIO_DIR):
        print(f" !!! Error: audio directory not found at: {AUDIO_DIR}")
        return

    files = []
    for ext in ["*.mp3", "*.MP3", "*.wav", "*.WAV"]:
        files.extend(glob.glob(os.path.join(AUDIO_DIR, ext)))
    
    if not files:
        print(f" !!! Error: No audio files found in {AUDIO_DIR}")
        return

    print(f"Found {len(files)} songs. initializing model...")
    extractor = MERTExtractor()
    
    embeddings = []
    song_ids = []

    print("--- Starting Extraction ---")

    failed_count = 0
    for i, f_path in enumerate(tqdm(files)):
        try:
            filename = os.path.basename(f_path)
            song_id = os.path.splitext(filename)[0]
            
            emb = extractor.process_audio(f_path)

            embeddings.append(emb)
            song_ids.append(song_id)

        except Exception as e:
            # One bad file must not abort the whole extraction.
            print(f" !!! Error on file: {f_path} ({e}) -- skipping")
            failed_count += 1
            continue

    if len(embeddings) == 0:
        print(" !!! No embeddings were generated")
        return

    # save: L2-normalize onto the unit sphere so song_embeddings.npy is the
    # canonical action manifold shared by FAISS retrieval and world-model
    # training (the actor's proto-action is normalized to match).
    from utils.common import l2_normalize

    embeddings_np = l2_normalize(np.vstack(embeddings), axis=1)
    song_ids_np = np.array(song_ids)

    np.save(OUTPUT_PATH, embeddings_np)
    np.save(ID_MAP_PATH, song_ids_np)

    print(f" SUCCESS! Saved {embeddings_np.shape[0]} embeddings to {OUTPUT_PATH} "
          f"({failed_count} files skipped)")

if __name__ == "__main__":
    extract_all_embeddings()