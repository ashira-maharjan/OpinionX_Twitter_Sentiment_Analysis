import os
import sys
import pickle
import numpy as np
from src.logger import get_logger
from src.exception import CustomException

logger = get_logger(__name__)

ARTIFACTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def save_object(file_path: str, obj):
    """Serialize and save any Python object to disk."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
        logger.info(f"Object saved at: {file_path}")
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path: str):
    """Load a pickled object from disk."""
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise CustomException(e, sys)


def get_artifact_path(filename: str) -> str:
    """Return the full path for an artifact file."""
    return os.path.join(ARTIFACTS_DIR, filename)


def map_cluster_to_sentiment(cluster_label: int, label_mapping: dict) -> str:
    """Map a numeric cluster label to a sentiment string using the label mapping."""
    return label_mapping.get(cluster_label, "Unknown")
