import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

from src.logger import get_logger
from src.exception import CustomException
from src.utils import save_object, get_artifact_path

logger = get_logger(__name__)

class ModelTrainer:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        # multi_class='multinomial' handles 4-class sentiment perfectly
        self.model = LogisticRegression(
            max_iter=1000, 
            multi_class='multinomial', 
            random_state=self.random_state
        )
        self.label_encoder = LabelEncoder()

    def initiate_model_training(self, X_train, X_val, train_df, val_df):
        try:
            logger.info("Starting Logistic Regression training...")
            
            # Encode 'Positive', 'Negative', etc. into 0, 1, 2, 3
            y_train = self.label_encoder.fit_transform(train_df["sentiment"])
            y_val = self.label_encoder.transform(val_df["sentiment"])

            # Train the model
            self.model.fit(X_train, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred) * 100
            
            # Dynamic mapping for the prediction pipeline
            label_mapping = {int(i): str(label) for i, label in enumerate(self.label_encoder.classes_)}
            logger.info(f"Training Complete. Accuracy: {accuracy:.2f}%")
            logger.info(f"Label Mapping: {label_mapping}")

            # Save artifacts
            save_object(get_artifact_path("model.pkl"), self.model)
            save_object(get_artifact_path("label_mapping.pkl"), label_mapping)
            
            metrics = {
                "validation_accuracy_pct": round(accuracy, 2),
                "label_mapping": label_mapping,
                "n_samples_train": X_train.shape[0]
            }
            
            return self.model, label_mapping, metrics

        except Exception as e:
            raise CustomException(e, sys)