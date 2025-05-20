import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import joblib
from pathlib import Path
import json
from typing import Dict, Any, Tuple

MODELS_DIR = Path("models")
VECTORIZER_PATH = MODELS_DIR / "vectorizer.joblib"
CLUSTER_MODEL_PATH = MODELS_DIR / "cluster_model.joblib"
KNN_MODEL_PATH = MODELS_DIR / "knn_model.joblib"
PROPERTIES_FILE = Path("data/properties.json")

def train_models() -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Train all models using properties from properties.json file.
    Returns training statistics and the processed DataFrame.
    """
    # Load properties from JSON file
    if not PROPERTIES_FILE.exists():
        raise ValueError("Properties file not found")
        
    with open(PROPERTIES_FILE, "r", encoding="utf-8") as f:
        properties = json.load(f)

    if not properties:
        raise ValueError("No properties to train on")

    # Remove duplicates by ID
    unique_props = {}
    for prop in properties:
        prop_id = prop.get("id")
        if prop_id:
            unique_props[prop_id] = prop

    # Convert to DataFrame
    df = pd.DataFrame(unique_props.values())
    
    # Fill missing values
    for column in df.columns:
        if df[column].dtype == "object":
            df[column] = df[column].fillna("")
        elif pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].fillna(0)
        else:
            df[column] = df[column].fillna("unknown")

    # Select or combine text columns
    #text_columns = df.select_dtypes(include=["object", "string"]).columns
    #if "description" in df.columns:
    #    text_column = "description"
    #elif len(text_columns) > 0:
    #    text_column = text_columns[0]
    #else:
    df["combined_text"] = df.astype(str).agg(" ".join, axis=1)
    text_column = "combined_text"

    # TF-IDF vectorization and KMeans clustering
    vectorizer = TfidfVectorizer(stop_words="english", max_features=100)
    X_text = vectorizer.fit_transform(df[text_column])
    kmeans = KMeans(n_clusters=5, random_state=42, n_init='auto')
    kmeans.fit(X_text)
    df["cluster"] = kmeans.labels_

    # Save text-based models
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(kmeans, CLUSTER_MODEL_PATH)

    # KNN on Numeric Features
    numeric_columns = df.select_dtypes(include=["number"]).columns
    X_numeric = df[numeric_columns]

    # Standardize + KNN pipeline
    knn_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=5))
    ])

    # Use clusters as labels for supervised KNN training
    knn_pipeline.fit(X_numeric, df["cluster"])

    # Save KNN pipeline
    joblib.dump(knn_pipeline, KNN_MODEL_PATH)

    # Save clustered data
    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Save full JSON
    clustered_data_path = data_dir / "properties_clustered.json"
    df.to_json(clustered_data_path, orient="records", indent=2)
    
    # Save full CSV
    df.to_csv(data_dir / "properties_clustered.csv", index=False)
    
    # Save simplified CSV with selected columns
    columns_to_save = ["id", text_column, "cluster"] + list(numeric_columns)
    df[columns_to_save].to_csv(data_dir / "properties_clustered_simple.csv", index=False)

    # Prepare training statistics
    stats = {
        "cluster_distribution": df["cluster"].value_counts().sort_index().to_dict(),
        "sample_predictions": knn_pipeline.predict(X_numeric.head(5)).tolist(),
        "properties_count": len(df),
        "text_column_used": text_column,
        "numeric_features_used": list(numeric_columns)
    }

    return stats, df 