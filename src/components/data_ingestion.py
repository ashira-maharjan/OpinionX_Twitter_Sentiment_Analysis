import os
import sys
import pandas as pd
from dataclasses import dataclass
from src.logger import get_logger
from src.exception import CustomException

logger = get_logger(__name__)

@dataclass
class DataIngestionConfig:
    # Dynamically find project root
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    train_data_path: str = os.path.join(base_dir, "notebook", "data", "twitter_training.csv")
    val_data_path: str = os.path.join(base_dir, "notebook", "data", "twitter_validation.csv")
    columns: list = None

    def __post_init__(self):
        self.columns = ["id", "entity", "sentiment", "text"]

class DataIngestion:
    def __init__(self):
        self.config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logger.info("Starting data ingestion...")
        try:
            train_df = pd.read_csv(self.config.train_data_path, header=None, names=self.config.columns)
            val_df = pd.read_csv(self.config.val_data_path, header=None, names=self.config.columns)

            # Drop rows with missing tweet text
            train_df.dropna(subset=["text"], inplace=True)
            val_df.dropna(subset=["text"], inplace=True)

            logger.info(f"Ingested Train: {train_df.shape}, Val: {val_df.shape}")
            return train_df, val_df

        except Exception as e:
            raise CustomException(e, sys)