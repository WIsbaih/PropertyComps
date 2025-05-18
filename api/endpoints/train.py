from fastapi import APIRouter, HTTPException
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans

router = APIRouter()

@router.post("/train")
def train_model():
    data_path = Path(__file__).resolve().parent.parent.parent / "data" / "properties_simple.json"
    model_path = Path(__file__).resolve().parent.parent.parent / "models" / "model.pkl"

    try:
        with open(data_path, "r") as f:
            properties = json.load(f)
    except Exception:
        raise HTTPException(status_code=500, detail="Could not load training data")

    if not properties:
        raise HTTPException(status_code=400, detail="No data to train on")

    df = pd.DataFrame(properties)

    feature_columns = [col for col in df.columns if col not in ["id", "cluster"]]
    df[feature_columns] = df[feature_columns].fillna(0)

    kmeans = KMeans(n_clusters=2, random_state=42)
    df["cluster"] = kmeans.fit_predict(df[feature_columns])

    knn = NearestNeighbors(n_neighbors=2)
    knn.fit(df[feature_columns])

    joblib.dump((knn, kmeans, df, feature_columns), model_path)

    return {"message": "Model trained successfully."}