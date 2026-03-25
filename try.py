"""
try_app.py  —  Quick smoke-test for the full pipeline.

Run:
    python try_app.py
"""
import sys
import os

# Ensure root is on PYTHONPATH when running from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline.train_pipeline import TrainPipeline
from src.pipeline.predict_pipeline import PredictPipeline

# ── STEP 1: Train ──────────────────────────────────────────────────────────────
print("\n🚀  Running Training Pipeline...")
train_pipeline = TrainPipeline()
metrics = train_pipeline.run()

# ── STEP 2: Predict ────────────────────────────────────────────────────────────
print("\n🔍  Running Prediction Pipeline on sample tweets...")
predict_pipeline = PredictPipeline()

sample_tweets = [
    "I absolutely love this game! Best experience ever!",
    "This product is terrible and broken. Waste of money.",
    "Just updated my software today. Works fine I guess.",
    "Amazon delivery was super fast, great service!",
    "Microsoft keeps crashing every single day. So frustrating.",
    "Nothing special happened. Pretty average day.",
]

print("\n" + "=" * 60)
print("  SAMPLE PREDICTIONS")
print("=" * 60)
for tweet in sample_tweets:
    result = predict_pipeline.predict(tweet)
    emoji = {"Positive": "😊", "Negative": "😠", "Neutral": "😐"}.get(result["sentiment"], "❓")
    print(f"  {emoji}  [{result['sentiment']:10s}]  {tweet[:55]}...")
print("=" * 60 + "\n")
