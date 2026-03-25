import re, sys, string, nltk, pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from src.logger import get_logger
from src.exception import CustomException
from src.utils import save_object, get_artifact_path

logger = get_logger(__name__)

def initialize_nltk_resources():
    resources = ['stopwords', 'wordnet', 'omw-1.4', 'punkt']
    for res in resources:
        try:
            nltk.data.find(f'corpora/{res}' if res != 'punkt' else f'tokenizers/{res}')
        except LookupError:
            nltk.download(res, quiet=True)

initialize_nltk_resources()
STOPWORDS_SET = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

def clean_tweet(text: str) -> str:
    """Core cleaning logic used by both Trainer and Predictor."""
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    
    tokens = [LEMMATIZER.lemmatize(w) for w in text.split() 
              if w not in STOPWORDS_SET and len(w) > 2]
    return " ".join(tokens)

class DataTransformation:
    def __init__(self, max_features: int = 5000, ngram_range: tuple = (1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            min_df=3,
        )

    def initiate_data_transformation(self, train_df: pd.DataFrame, val_df: pd.DataFrame, target_column: str = "text"):
        try:
            logger.info("Transforming text data...")
            train_df["cleaned_text"] = train_df[target_column].apply(clean_tweet)
            val_df["cleaned_text"] = val_df[target_column].apply(clean_tweet)

            # Filter empty results
            train_df = train_df[train_df["cleaned_text"].str.strip() != ""].reset_index(drop=True)
            val_df = val_df[val_df["cleaned_text"].str.strip() != ""].reset_index(drop=True)

            X_train = self.vectorizer.fit_transform(train_df["cleaned_text"])
            X_val = self.vectorizer.transform(val_df["cleaned_text"])

            save_object(get_artifact_path("tfidf_vectorizer.pkl"), self.vectorizer)
            return X_train, X_val, train_df, val_df
        except Exception as e:
            raise CustomException(e, sys)