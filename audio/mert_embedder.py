import os
import torch
import torchaudio
import numpy as np
import glob
from transformers import Wav2Vec2FeatureExtractor, AutoModel
from tqdm import tqdm

# --- Configuration ---
MODEL_NAME = "m-a-p/MERT-v1-330M"
TARGET_SR = 24000
# Ensure this path exactly matches your folder structure
AUDIO_DIR = os.path.join("data", "raw", "HKU956", "2. audio_signals")
OUTPUT_PATH = os.path.join("data", "processed", "song_embeddings.npy")
ID_MAP_PATH = os.path.join("data", "processed", "song_id_map.npy")

class MERTExtractor:
    def __init__(self, device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading MERT model on {self.device}...")
        
        try:
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(MODEL_NAME, trust_remote_code=True).to(self.device)
            self.model.eval()
        except OSError:
            print(f"❌ Connection Error: Could not download {MODEL_NAME}. Check internet or HuggingFace status.")
            raise

    def process_audio(self, file_path):
        """Loads audio, resamples, and extracts embedding."""
        try:
            # 1. Load Audio
            # This is where it usually fails if ffmpeg is missing
            waveform, sample_rate = torchaudio.load(file_path)
            
            # 2. Resample
            if sample_rate != TARGET_SR:
                resampler = torchaudio.transforms.Resample(sample_rate, TARGET_SR)
                waveform = resampler(waveform)
            
            # 3. Mono Mix
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # 4. Truncate (Max 30s)
            max_samples = 30 * TARGET_SR
            if waveform.shape[1] > max_samples:
                waveform = waveform[:, :max_samples]
                
            inputs = self.processor(waveform.squeeze().numpy(), sampling_rate=TARGET_SR, return_tensors="pt")
            input_values = inputs["input_values"].to(self.device)

            # 5. Inference
            with torch.no_grad():
                outputs = self.model(input_values)
                last_hidden_state = outputs.last_hidden_state
                embedding = torch.mean(last_hidden_state, dim=1).squeeze().cpu().numpy()
                
            return embedding

        except Exception as e:
            print(f"\n❌ CRITICAL ERROR processing {os.path.basename(file_path)}")
            print(f"   Reason: {e}")
            # Reraise the error for the first file so the user sees it immediately
            raise e 

def extract_all_embeddings():
    if not os.path.exists("data/processed"):
        os.makedirs("data/processed")

    # 1. Check Directory
    if not os.path.exists(AUDIO_DIR):
        print(f"❌ Error: Audio directory not found at: {AUDIO_DIR}")
        print("   Please check where you moved the 'HKU956' folder.")
        return

    # 2. Find Files (Case Insensitive)
    files = []
    for ext in ["*.mp3", "*.MP3", "*.wav", "*.WAV"]:
        files.extend(glob.glob(os.path.join(AUDIO_DIR, ext)))
    
    if not files:
        print(f"❌ No audio files found in {AUDIO_DIR}")
        return

    print(f"Found {len(files)} songs. initializing model...")
    extractor = MERTExtractor()
    
    embeddings = []
    song_ids = []

    print("Starting extraction (Press Ctrl+C to stop)...")

    # We use a loop that breaks on the first error to help debugging
    failed_count = 0
    for i, f_path in enumerate(tqdm(files)):
        try:
            filename = os.path.basename(f_path)
            song_id = os.path.splitext(filename)[0]
            
            emb = extractor.process_audio(f_path)
            
            if emb is not None:
                embeddings.append(emb)
                song_ids.append(song_id)
        except Exception as e:
            print(f"Aborting due to error on file: {f_path}")
            return # Stop completely so user can fix the dependency

    if len(embeddings) == 0:
        print("❌ No embeddings were generated. Check the error messages above.")
        return

    # Convert and Save
    embeddings_np = np.vstack(embeddings)
    song_ids_np = np.array(song_ids)

    np.save(OUTPUT_PATH, embeddings_np)
    np.save(ID_MAP_PATH, song_ids_np)

    print(f"✅ Success! Saved {embeddings_np.shape[0]} embeddings to {OUTPUT_PATH}")

if __name__ == "__main__":
    extract_all_embeddings()