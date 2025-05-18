import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any
import numpy as np

from api.services.model_service import load_data, load_vectorizer, load_cluster_model

def to_serializable(val):
    if isinstance(val, (np.generic, np.ndarray)):
        return val.item() if val.ndim == 0 else val.tolist()
    return val

def get_text_column(df: pd.DataFrame) -> str:
    """Detect or build the text column used for vectorization."""
    if "description" in df.columns:
        return "description"
    
    text_columns = df.select_dtypes(include=["object", "string"]).columns
    if len(text_columns) > 0:
        return text_columns[0]
    
    # Fallback: create a combined text column
    df["combined_text"] = df.astype(str).agg(" ".join, axis=1)
    return "combined_text"

def get_recommendations(property_id: int, top_n: int = 5) -> Dict[str, Any]:
    df = load_data()
    vectorizer = load_vectorizer()
    cluster_model = load_cluster_model()

    if property_id not in df.index:
        raise ValueError(f"Property ID {property_id} not found.")

    # Ensure the text column used in training exists
    text_column = get_text_column(df)

    # Fill missing text values
    df[text_column] = df[text_column].fillna("")

    # Vectorize the selected column
    feature_matrix = vectorizer.transform(df[text_column])

    # Predict cluster if not already
    if 'cluster' not in df.columns:
        df["cluster"] = cluster_model.predict(feature_matrix)

    # Find similar properties in the same cluster
    index = property_id
    property_cluster = df.loc[index, "cluster"]
    cluster_df = df[df["cluster"] == property_cluster]

    # Compute cosine similarity
    target_vector = feature_matrix[index]
    cluster_vectors = feature_matrix[cluster_df.index]
    similarity_scores = cosine_similarity(target_vector, cluster_vectors).flatten()

    # Get top N recommendations (excluding itself)
    similar_indices = cluster_df.index[similarity_scores.argsort()[::-1]]
    similar_indices = [i for i in similar_indices if i != index][:top_n]

    neighbors = df.loc[similar_indices]
    scores = similarity_scores[[list(cluster_df.index).index(i) for i in similar_indices]]

    recommendations = neighbors.copy()
    recommendations["similarity"] = [round(float(score), 4) for score in scores]

    # Replace NaN with null-safe JSON values
    recommendations = recommendations.replace({np.nan: None})

    formatted = [
        {k: to_serializable(v) for k, v in row.items()}
        for _, row in recommendations.iterrows()
    ]

    return {
        "property_id": property_id,
        "cluster": int(property_cluster),
        "recommendations": formatted
    }
