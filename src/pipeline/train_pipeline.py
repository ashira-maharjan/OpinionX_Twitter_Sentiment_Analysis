import sys
from src.logger import get_logger
from src.exception import CustomException
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

logger = get_logger(__name__)

class TrainPipeline:
    def __init__(self):
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation()
        self.model_trainer = ModelTrainer()

    def run(self):
        try:
            logger.info("========== STARTING SUPERVISED TRAINING ==========")
            
            train_df, val_df = self.data_ingestion.initiate_data_ingestion()
            
            X_train, X_val, train_df, val_df = self.data_transformation.initiate_data_transformation(
                train_df, val_df
            )
            
            model, mapping, metrics = self.model_trainer.initiate_model_training(
                X_train, X_val, train_df, val_df
            )
            
            print("\n" + "="*40)
            print("   LOGISTIC REGRESSION RESULTS")
            print("="*40)
            print(f"   Accuracy: {metrics['validation_accuracy_pct']}%")
            print(f"   Mapping:  {metrics['label_mapping']}")
            print("="*40 + "\n")
            
            return metrics
        except Exception as e:
            raise CustomException(e, sys)