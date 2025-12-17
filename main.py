# 🚀 CLI Entry point (The Controller)
import sys
import os
import time
import importlib
import argparse

# --- Configuration Constants ---
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR = os.path.join(ROOT_DIR, "data", "raw", "HKU956")

# --- Utility Functions ---

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    print("="*65)
    print("      BIO-ADAPTIVE MUSIC ENGINE | CLI CONTROLLER")
    print(f"      Root: {ROOT_DIR}")
    print("="*65)

def check_structure():
    """Verifies that the HKU956 dataset is correctly placed."""
    required_paths = [
        os.path.join(DATA_RAW_DIR, "1. physiological_signals"),
        os.path.join(DATA_RAW_DIR, "2. audio_signals"),
        os.path.join(DATA_RAW_DIR, "3. AV_ratings.csv"),
        os.path.join(DATA_RAW_DIR, "4. participant_personality.csv")
    ]
    
    missing = [p for p in required_paths if not os.path.exists(p)]
    
    if missing:
        print("\n⚠️  STRUCTURE WARNING: HKU956 Dataset not found in expected path.")
        print(f"   Expected location: {DATA_RAW_DIR}")
        print("   Missing files/folders:")
        for m in missing:
            print(f"    - {os.path.basename(m)}")
        print("\n   ACTION REQUIRED: Move your 'HKU956' folder into 'data/raw/'.")
        input("   Press Enter to continue anyway (or Ctrl+C to exit and fix)...")

def safe_run(module_path, function_name, **kwargs):
    """Dynamically imports and runs a function with error handling."""
    try:
        mod = importlib.import_module(module_path)
        func = getattr(mod, function_name)
        
        print(f"\n>> 🚀 Launching {module_path}.{function_name}()...")
        time.sleep(1)
        func(**kwargs)
        print(f"\n>> ✅ Task {function_name} completed.")
        
    except ModuleNotFoundError as e:
        print(f"\n>> ❌ ERROR: Module '{e.name}' not found.")
        print(f"   Has the file '{module_path.replace('.', '/')}.py' been created?")
    except AttributeError:
        print(f"\n>> ❌ ERROR: Function '{function_name}' not found in '{module_path}'.")
    except Exception as e:
        print(f"\n>> ❌ RUNTIME ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    input("\n[Press Enter to return to menu]")

# --- Task Wrappers ---

def setup_project():
    """Creates the empty folder structure."""
    folders = [
        "data/raw", "data/processed",
        "audio", "physio", "user", "context", "simulator", "rl", "scripts", "configs"
    ]
    for f in folders:
        os.makedirs(os.path.join(ROOT_DIR, f), exist_ok=True)
        # Create __init__.py so they are treated as modules
        init_file = os.path.join(ROOT_DIR, f, "__init__.py")
        if not os.path.exists(init_file):
            open(init_file, 'a').close()
            
    print(">> ✅ Project structure created.")
    print(">> NOW: Move your existing 'HKU956' folder into 'data/raw/'.")
    time.sleep(2)

def process_data():
    """Person B: Parse HKU956."""
    safe_run("data.hku956_loader", "process_and_cache_data")

def train_physio():
    """Person B: Train CNN Encoder."""
    epochs = input("Enter epochs (default 20): ") or 20
    safe_run("physio.train_encoder", "train_physio_model", epochs=int(epochs))

def train_user():
    """Person B: Train DCN Profile."""
    safe_run("user.train_profile", "train_user_model")

def train_world():
    """Person B: Train World Simulator."""
    safe_run("simulator.train_simulator", "train_world_model")

def process_audio():
    """Person A: Extract MERT Embeddings."""
    safe_run("audio.mert_embedder", "extract_all_embeddings")

def train_context():
    """Person A: Train Transformer Context."""
    safe_run("context.train_context", "train_context_model")

def train_agent():
    """Person A: Train SAC Agent."""
    steps = input("Enter training steps (default 10000): ") or 10000
    safe_run("rl.train_agent", "train_sac_agent", steps=int(steps))

# --- Interactive Menu ---

def interactive_menu():
    while True:
        clear_screen()
        print_header()
        
        print("\n--- 🛠  SETUP ---")
        print(" [0] Initialize Folder Structure (Run this first!)")
        
        print("\n--- 👤 PERSON B: WORLD BUILDER (Environment) ---")
        print(" [1] Process Raw HKU956 Data (Loader)")
        print(" [2] Train Module B (Physio Encoder)")
        print(" [3] Train Module C (User Profiles)")
        print(" [4] Train The User Simulator (World Model)")

        print("\n--- 👤 PERSON A: AGENT ARCHITECT (Policy) ---")
        print(" [5] Pre-compute MERT Audio Embeddings (Module A)")
        print(" [6] Train Context Transformer (Module E)")
        print(" [7] Train SAC Agent (Offline RL)")
        
        print("\n--- 🏁 SYSTEM ---")
        print(" [Q] Quit")
        
        choice = input("\nSelect an option >> ").strip().lower()

        if choice == '0': setup_project()
        elif choice == '1': process_data()
        elif choice == '2': train_physio()
        elif choice == '3': train_user()
        elif choice == '4': train_world()
        elif choice == '5': process_audio()
        elif choice == '6': train_context()
        elif choice == '7': train_agent()
        elif choice == 'q': sys.exit(0)
        else: print("Invalid selection.")

if __name__ == "__main__":
    check_structure()
    interactive_menu()