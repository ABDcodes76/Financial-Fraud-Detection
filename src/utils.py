import os
import sys
import json
import joblib
import logging
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("FraudDetectionSystem")

def get_project_root() -> Path:
    """Returns the absolute root directory of the project."""
    return Path(__file__).resolve().parent.parent

def get_data_path(filename: str = "transactions.csv") -> Path:
    """Returns the absolute path to a file in the dataset directory."""
    return get_project_root() / "dataset" / filename

def get_models_dir() -> Path:
    """Returns the absolute path to the models directory, ensuring it exists."""
    models_dir = get_project_root() / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    return models_dir

def save_model(model_obj, filename: str):
    """Saves a model object using joblib."""
    filepath = get_models_dir() / filename
    joblib.dump(model_obj, filepath)
    logger.info(f"Model saved successfully to {filepath}")
    return filepath

def load_model(filename: str):
    """Loads a model object using joblib."""
    filepath = get_models_dir() / filename
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found at {filepath}. Please train models first.")
    return joblib.load(filepath)

def save_json(data: dict, filename: str):
    """Saves dictionary data to a JSON file."""
    filepath = get_models_dir() / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    logger.info(f"JSON data saved to {filepath}")
    return filepath

def load_json(filename: str) -> dict:
    """Loads dictionary data from a JSON file."""
    filepath = get_models_dir() / filename
    if not filepath.exists():
        raise FileNotFoundError(f"File not found at {filepath}")
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)
