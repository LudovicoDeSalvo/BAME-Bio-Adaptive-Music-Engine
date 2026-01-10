import pandas as pd
import requests
import time
import os

# --- Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(SCRIPT_DIR, '../data/raw/HKU956/2. original_song_audio.csv')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '../data/raw/HKU956/2. audio_files')
CLIENT_ID = '01ade0a2'

def execution():
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    # check if CSV exists
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"CSV file not found at: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    # verify CSV structure
    id_column = 'song_id' 
    if id_column not in df.columns:
        raise ValueError(f"Column '{id_column}' not found. Available columns: {df.columns.tolist()}")

    track_ids = df[id_column].unique()
    print(f"Found {len(track_ids)} unique tracks available")

    # download
    for track_id in track_ids:
        file_path = os.path.join(OUTPUT_DIR, f"{track_id}.mp3")
        
        if os.path.exists(file_path):
            print(f"Skipping {track_id} (already exists)")
            continue
            
        url = f"https://api.jamendo.com/v3.0/tracks/file?client_id={CLIENT_ID}&id={track_id}"
        
        try:
            response = requests.get(url, stream=True, timeout=10)

            if response.status_code == 200:
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            f.write(chunk)
                print(f"Downloaded: {track_id}")

            elif response.status_code == 404:
                print(f"Failed: {track_id} (404)")

            else:
                print(f"Error {response.status_code} for {track_id}")
                
        except Exception as e:
            print(f"Exception for {track_id}: {e}")
            
        time.sleep(0.2)

    print("Process finished.")

if __name__ == "__main__":
    execution()