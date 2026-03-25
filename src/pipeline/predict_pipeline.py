import sys
from src.logger import get_logger
from src.exception import CustomException
from src.utils import load_object, get_artifact_path
from src.components.data_transformation import clean_tweet

class PredictPipeline:
    def __init__(self):
        try:
            self.model = load_object(get_artifact_path("model.pkl"))
            self.vectorizer = load_object(get_artifact_path("tfidf_vectorizer.pkl"))
            self.label_mapping = load_object(get_artifact_path("label_mapping.pkl"))
        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, tweet: str):
        try:
            cleaned = clean_tweet(tweet)
            if not cleaned.strip():
                return {"sentiment": "Neutral", "original_text": tweet}

            vec = self.vectorizer.transform([cleaned])
            # Predict the class index
            pred_idx = int(self.model.predict(vec)[0])
            
            return {
                "original_text": tweet,
                "sentiment": self.label_mapping.get(pred_idx, "Unknown"),
                "cleaned_text": cleaned
            }
        except Exception as e:
            raise CustomException(e, sys)