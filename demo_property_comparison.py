import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def combine_property_features(property_data):
    """Combine all property features into a single text string."""
    return " ".join(str(v) for v in property_data.values() if v is not None)

def find_similar_properties(subject_property, candidate_properties, top_n=3):
    """
    Find similar properties using TF-IDF and cosine similarity.
    
    Args:
        subject_property (dict): The property to find matches for
        candidate_properties (list): List of candidate properties
        top_n (int): Number of similar properties to return
    
    Returns:
        list: Top N most similar properties
    """
    # Combine all properties into text
    all_properties = [subject_property] + candidate_properties
    property_texts = [combine_property_features(prop) for prop in all_properties]
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(property_texts)
    
    # Calculate similarity between subject and candidates
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
    # Get top N most similar properties
    top_indices = similarities.argsort()[::-1][:top_n]
    similar_properties = [candidate_properties[i] for i in top_indices]
    
    return similar_properties

# Example usage
if __name__ == "__main__":
    # Sample subject property
    subject = {
        "address": "123 Main St",
        "bedrooms": 3,
        "bathrooms": 2,
        "square_feet": 2000,
        "price": 350000,
        "year_built": 2010
    }
    
    # Sample candidate properties
    candidates = [
        {
            "address": "456 Oak Ave",
            "bedrooms": 3,
            "bathrooms": 2,
            "square_feet": 2100,
            "price": 360000,
            "year_built": 2012
        },
        {
            "address": "789 Pine Rd",
            "bedrooms": 4,
            "bathrooms": 3,
            "square_feet": 2500,
            "price": 400000,
            "year_built": 2015
        },
        {
            "address": "321 Elm St",
            "bedrooms": 2,
            "bathrooms": 1,
            "square_feet": 1500,
            "price": 300000,
            "year_built": 2008
        }
    ]
    
    # Find similar properties
    similar_properties = find_similar_properties(subject, candidates)
    
    # Print results
    print("\nSubject Property:")
    print(subject)
    print("\nMost Similar Properties:")
    for i, prop in enumerate(similar_properties, 1):
        print(f"\n{i}. {prop}") 