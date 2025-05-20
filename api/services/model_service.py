import os
import json
import pandas as pd
import joblib

DATA_PATH = "data/properties_simple.json"
VECTORIZER_PATH = "models/vectorizer.joblib"
CLUSTER_MODEL_PATH = "models/cluster_model.joblib"

def load_data():
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data)
    else:
        return pd.DataFrame()

def save_data(df: pd.DataFrame):
    with open(DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(df.to_dict(orient="records"), f, ensure_ascii=False, indent=2)

def load_vectorizer():
    if os.path.exists(VECTORIZER_PATH):
        return joblib.load(VECTORIZER_PATH)
    return None

def load_cluster_model():
    if os.path.exists(CLUSTER_MODEL_PATH):
        return joblib.load(CLUSTER_MODEL_PATH)
    return None

# Pre-load shared data and models (if needed globally)
df = load_data()
vectorizer = load_vectorizer()
cluster_model = load_cluster_model()
