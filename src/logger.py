import logging
import os
from datetime import datetime

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "training.log")

logging.basicConfig(
    filename=LOG_FILE,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        fmt = logging.Formatter("[ %(asctime)s ] %(name)s - %(levelname)s - %(message)s")
        console.setFormatter(fmt)
        logger.addHandler(console)
    return logger
