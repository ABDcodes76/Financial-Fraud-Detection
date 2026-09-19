# FINSEC AI - Streamlit Community Cloud Root Entrypoint
import sys
from pathlib import Path

# Add project root directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Launch master dashboard application
import dashboard.app
