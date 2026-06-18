import os
import sys

# CPU-only test suite: hide CUDA before torch initializes so get_device()
# resolves to CPU. Keeps tests deterministic and runnable without a GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Make the project root importable when running `pytest` from anywhere.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
