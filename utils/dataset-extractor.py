import zipfile
import os
import sys

# --- CONFIGURATION ---
ZIP_FILENAME = 'HKU956.zip'  # <--- REPLACE with your actual zip file name

# Get the directory where this script is located (/utils)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define paths relative to the script location
# Go up one level (..) to reach the project root, then into /dataset
INPUT_DIR = os.path.join(SCRIPT_DIR, '..')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..')

# Full path to the zip file
zip_file_path = os.path.join(INPUT_DIR, ZIP_FILENAME)

# --- EXECUTION ---
def extract_zip():
    # 1. Validate Input
    if not os.path.exists(zip_file_path):
        print(f"Error: Zip file not found at: {zip_file_path}")
        print(f"Current working directory: {os.getcwd()}")
        sys.exit(1)

    # 2. Prepare Output Directory
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")
    else:
        print(f"Output directory exists: {OUTPUT_DIR}")

    # 3. Extract
    print(f"Extracting '{ZIP_FILENAME}'...")
    try:
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            # List files (optional, for verification)
            # print(f"Files found: {zip_ref.namelist()[:5]} ...")
            
            zip_ref.extractall(OUTPUT_DIR)
            
        print(f"Success! Extracted to: {OUTPUT_DIR}")

    except zipfile.BadZipFile:
        print("Error: The file is a bad zip file (corrupted or not a zip).")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    extract_zip()