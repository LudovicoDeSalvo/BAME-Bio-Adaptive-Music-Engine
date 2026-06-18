import sys
import os
import time
import importlib
import argparse
import glob
import numpy as np
import zipfile

# --- Configuration ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR = os.path.join(ROOT_DIR, "data", "raw", "HKU956")

# --- Utility functions ---

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    print("="*65)
    print("      BIO-ADAPTIVE MUSIC ENGINE | PIPELINE CONTROLLER")
    print(f"      Root: {ROOT_DIR}")
    print("="*65)

def safe_run(module_path, function_name, **kwargs):
    """Dynamically imports and runs a function with error handling."""

    # load
    try:
        mod = importlib.import_module(module_path)
        func = getattr(mod, function_name)

    except (ModuleNotFoundError, AttributeError) as e:
        print(f"\n>> !!!  ERROR: Could not load '{function_name}' from '{module_path}'.")
        print(f"   Reason: {e}")
        return

    # run
    try:
        print(f"\n >> Launching {module_path}.{function_name}()...")
        time.sleep(1)
        func(**kwargs)
        print(f"\n>> Task {function_name} completed!")

    except Exception as e:
        print(f"\n>> !!! RUNTIME ERROR in {function_name}: {e}")
        import traceback
        traceback.print_exc() 
    
    input("\n[Press Enter to return to menu]")


def check_system_health():
    print("\n--- SYSTEM HEALTH CHECK ---")
    
    checks = [
        ("Raw Data", "data/raw/HKU956/1. physiological_signals", "Check if HKU956 is in place"),
        ("Audio Clips", "data/processed/audio_clips", "Run option [1] to generate clips"),
        ("Physio Pool", "data/processed/physio_cache.npz", "Run option [1] to generate physio"),
        ("Song Embeddings", "data/processed/song_embeddings.npy", "Run option [2] to encode audio"),
        ("Physio Embeddings", "data/processed/physio_embeddings.npz", "Run option [4] to train encoder"),
        ("User Embeddings", "data/processed/user_embeddings.npz", "Run option [5] to train profiler"),
        ("Context Model", "context/checkpoints/context_model.pth", "Run option [6] to train context"),
        ("World Model", "simulator/checkpoints/world_model.pth", "Run option [7] to train simulator"),
        ("Agent Checkpoint", "rl/checkpoints/sac_final_actor.pth", "Run option [8] to train agent")
    ]
    
    all_pass = True

    for name, path, tip in checks:

        full_path = os.path.join(ROOT_DIR, path)
        exists = os.path.exists(full_path)

        if not exists and "audio_clips" in path:
             exists = os.path.isdir(full_path) and len(os.listdir(full_path)) > 0
             
        status = " PASS" if exists else " MISSING"

        print(f" {status} | {name:<18} | {path}")
        if not exists:
            print(f"    -> Tip: {tip}")
            all_pass = False
            
    if all_pass:
        print("\n>> SYSTEM READY! All components are built.")
    else:
        print("\n>> !!! System incomplete. Run the missing steps above.")
    
    input("\n[Press Enter to continue]")


# --- Task Wrappers ---

def setup_project():
    """Creates folder structure, unzips dataset, and optionally downloads songs"""

    # folders
    folders = [
        "data/raw", "data/processed/audio_clips", 
        "audio", "physio", "user", "context", "simulator", "rl", "scripts", "configs"
    ]

    for f in folders:
        os.makedirs(os.path.join(ROOT_DIR, f), exist_ok=True)

    print(">> Folders created")
    
    # unzip dataset
    zip_path = os.path.join(ROOT_DIR, "HKU956.zip")
    extract_path = os.path.join(ROOT_DIR, "data", "raw")
    target_folder = os.path.join(extract_path, "HKU956")

    if os.path.exists(zip_path):
        if not os.path.exists(target_folder):

            print(f">> Found HKU956.zip. Extracting to {extract_path}...")
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)
                print(">> Dataset extracted successfully")
                
            except zipfile.BadZipFile:
                print(">> !!! Error: HKU956.zip is corrupted")

            except Exception as e:
                print(f">> !!! Error extracting dataset: {e}")
        else:
            print(">> Dataset already extracted")
    else:
        print(f">> !!! HKU956.zip not found in root")

    time.sleep(1)

    # download songs
    choice = input("Run the song downloader? (y/n): ").strip().lower()
    
    if choice == 'y':
        safe_run("scripts.download_songs", "execution")
    else:
        print(">> Skipping song download.")

    time.sleep(1)

def align_and_slice():

    confirm = input("This will overwrite existing clips in 'data/processed. Continue? (y/n): ")
    if confirm.lower() == 'y':

        clips_dir = os.path.join(ROOT_DIR, "data/processed/audio_clips")

        if os.path.exists(clips_dir):
            files = glob.glob(os.path.join(clips_dir, "*.wav"))
            for f in files: os.remove(f)
            
        safe_run("scripts.align_and_slice", "align_and_process")

def process_audio():
    if not os.path.exists(os.path.join(ROOT_DIR, "data/processed/audio_clips")):
        print("!!! Warning: 'audio_clips' folder missing")
        time.sleep(2)
    safe_run("audio.mert_embedder", "extract_all_embeddings")

def verify_data():

    path = os.path.join(ROOT_DIR, "data/processed/physio_cache.npz")

    if not os.path.exists(path):
        print(f"!!! File not found: {path}")

    else:
        try:
            data = np.load(path, allow_pickle=True)
            print(f"\n--- DATASET STATISTICS ---")
            print(f"Total samples (clips): {len(data['window_features'])}")
            print(f"Window tensor shape:   {data['window_features'].shape}")
            print(f"Physio feature dim:    {data['window_features'].shape[2]}")
            print(f"Unique songs:          {len(np.unique(data['song_ids']))}")
            print(f"Unique participants:   {len(np.unique(data['participant_ids']))}")
            print("OK! Data aligned and ready for training.")
        except Exception as e:
            print(f"!!! Error reading cache: {e}")
    
    input("\n[Press Enter]")

def train_physio():
    epochs = input("Enter epochs (default 150): ") or 150
    safe_run("physio.train_encoder", "train_physio_model", epochs=int(epochs))

def train_user():
    safe_run("user.train_profile", "train_user_model")

def train_context():
    safe_run("context.train_context", "train_context_model")

def train_world():
    safe_run("simulator.train_simulator", "train_world_model")

def train_agent():
    # Default must exceed the warmup (START_STEPS=1000) or the agent does 0 updates.
    steps = input("Enter training steps (default 20000): ") or 20000
    safe_run("rl.train_agent", "train_sac_agent", steps=int(steps))

def run_inference():
    safe_run("scripts.inference", "run_inference_protocol")

def run_hyperparam_search():
    steps = input("Steps per trial (default 6000): ") or 6000
    method = (input("Method [grid/random] (default grid): ") or "grid").strip().lower()
    trials = input("Trials for random method (default 12): ") or 12
    safe_run("scripts.hyperparam_search", "run_hyperparam_search",
             steps=int(steps), method=method, trials=int(trials))

def run_controllability_probe():
    targets = input("Number of (start,target) pairs (default 20): ") or 20
    safe_run("scripts.controllability_probe", "run_controllability_probe",
             targets=int(targets))

# --- Interactive Menu ---

def interactive_menu():
    while True:
        clear_screen()
        print_header()
        
        print("\n--- DATA PREPARATION ---")
        print(" [0] Preparation: Create Folders, Unzip Dataset, Download Songs")
        print(" [1] Align and Slice Data")
        print(" [2] Extract Audio Embeddings (MERT)")
        print(" [3] Verify Aligned Dataset")

        print("\n--- COMPONENT TRAINING ---")
        print(" [4] Train Physiological Encoder")
        print(" [5] Train User Profiler")
        print(" [6] Train Context Transformer")

        print("\n--- SIMULATION and AGENT ---")
        print(" [7] Train User Simulator")
        print(" [8] Train SAC Agent")

        print("\n--- EVALUATION ---")
        print(" [9] Run Inference (Holdout User Evaluation)")

        print("\n--- UTILITIES ---")
        print(" [S] Hyperparameter Sweep (SAC Agent)")
        print(" [P] Controllability Probe (Env Diagnostic)")
        print(" [H] System Health Check")
        print(" [Q] Quit")
        
        choice = input("\nSelect an option >> ").strip().lower()

        if choice == '0': setup_project()
        elif choice == '1': align_and_slice()
        elif choice == '2': process_audio()
        elif choice == '3': verify_data()
        elif choice == '4': train_physio()
        elif choice == '5': train_user()
        elif choice == '6': train_context() 
        elif choice == '7': train_world()   
        elif choice == '8': train_agent()
        elif choice == '9': run_inference()
        elif choice == 's': run_hyperparam_search()
        elif choice == 'p': run_controllability_probe()
        elif choice == 'h': check_system_health()
        elif choice == 'q': sys.exit(0)
        else: print("Invalid selection.")

if __name__ == "__main__":
    interactive_menu()