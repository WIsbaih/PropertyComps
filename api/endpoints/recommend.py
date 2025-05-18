from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from ..services.recommendation_service import get_recommendations

router = APIRouter()

@router.get("/recommend")
def recommend(property_id: int = Query(..., description="ID of the property to get recommendations for"),
              top_n: Optional[int] = Query(5, description="Number of similar properties to return")):
    try:
        return get_recommendations(property_id, top_n)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")
    
# def to_serializable(obj):
#     if isinstance(obj, np.generic):
#         return obj.item()
#     return obj

# @router.get("/recommend")
# def recommend(
#     property_id: int = Query(...),
#     count: int = Query(5, ge=1, le=20)
# ):
#     try:
#         # Load the saved model and data
#         model_path = __import__('pathlib').Path(__file__).resolve().parent.parent.parent / "models" / "model.pkl"
#         knn, kmeans, df, feature_columns = joblib.load(model_path)
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Model could not be loaded: {str(e)}")

#     if property_id not in df["id"].values:
#         raise HTTPException(status_code=404, detail="Property ID not found")

#     # Locate the index of the property
#     row_index = df[df["id"] == property_id].index[0]

#     # Get the feature vector
#     property_vector = df.loc[row_index, feature_columns].values.reshape(1, -1)

#     # Get the cluster of the property
#     property_cluster = kmeans.predict(property_vector)[0]

#     # Filter only properties from the same cluster
#     same_cluster_df = df[df["cluster"] == property_cluster]

#     if same_cluster_df.shape[0] <= 1:
#         raise HTTPException(status_code=404, detail="No other properties found in the same cluster")

#     # Re-run KNN on filtered cluster
#     from sklearn.neighbors import NearestNeighbors
#     cluster_features = same_cluster_df[feature_columns].values
#     cluster_knn = NearestNeighbors(n_neighbors=min(count + 1, len(same_cluster_df)))
#     cluster_knn.fit(cluster_features)

#     distances, indices = cluster_knn.kneighbors(property_vector)

#     # Get actual indices and remove the property itself
#     result_indices = same_cluster_df.iloc[indices[0]].index
#     neighbors = same_cluster_df.loc[result_indices]

#     if row_index in neighbors.index:
#         neighbors = neighbors.drop(row_index)
#         distances = distances[0][1:]  # skip first
#     else:
#         distances = distances[0]

#     # Add similarity score (1 - normalized distance)
#     max_dist = max(distances) if distances.any() else 1
#     similarity_scores = [1 - (d / max_dist) for d in distances]

#     # Assuming `neighbors` is a DataFrame and `similarity_scores` is a list or array
#     recommendations = neighbors.copy()
#     recommendations["similarity"] = similarity_scores
    
#     # Convert to records and ensure all values are serializable
#     recommendations_list = [
#     {k: to_serializable(v) for k, v in rec.items()}
#     for rec in recommendations.to_dict(orient="records")
#     ]

#     return {
#     "property_id": int(property_id),  # also make sure this is serializable
#     "cluster": to_serializable(property_cluster),
#     "recommendations": recommendations_list
#     }

