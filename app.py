"""
Root Entrypoint for Streamlit Cloud and Hugging Face Spaces.
Launches the full interactive Predictive Maintenance 3D Twin & Fleet Command Dashboard.
"""

from pathlib import Path
import sys
import runpy

# Ensure root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Launch main dashboard runner
demo_runner_path = ROOT_DIR / "dashboard" / "demo_runner.py"
runpy.run_path(str(demo_runner_path), run_name="__main__")
