import os
import json
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import pandas as pd
import numpy as np
from .property_service import update_properties

APPRAISALS_FILE = Path("data/appraisals.json")
VECTORIZER_PATH = Path("models/vectorizer.joblib")
APPRAISALS_FILE.parent.mkdir(parents=True, exist_ok=True)

# Initialize appraisals file if not exists
if not APPRAISALS_FILE.exists():
    APPRAISALS_FILE.write_text(json.dumps({}))

def _load_appraisals():
    return json.loads(APPRAISALS_FILE.read_text())

def _save_appraisals(data):
    APPRAISALS_FILE.write_text(json.dumps(data, indent=2))

def _select_comps_ml(subject: dict, candidates: list[dict], top_n=3):
    """
    Select comparable properties using TF-IDF and cosine similarity.
    A simplified approach that focuses on text-based similarity.
    """
    def combine_text(prop):
        """Combine all property features into a single text string."""
        return " ".join(str(v) for v in prop.values() if v is not None)

    # Load pre-trained vectorizer if exists, otherwise create new one
    if VECTORIZER_PATH.exists():
        vectorizer = joblib.load(VECTORIZER_PATH)
    else:
        vectorizer = TfidfVectorizer(stop_words="english")
        # Train vectorizer on all properties
        all_texts = [combine_text(subject)] + [combine_text(p) for p in candidates]
        vectorizer.fit(all_texts)
        # Save vectorizer for future use
        VECTORIZER_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(vectorizer, VECTORIZER_PATH)

    # Transform all properties
    texts = [combine_text(subject)] + [combine_text(p) for p in candidates]
    tfidf_matrix = vectorizer.transform(texts)
    
    # Calculate similarity between subject and candidates
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    
    # Get top N most similar properties
    top_indices = similarities.argsort()[::-1][:top_n]
    selected_comps = [candidates[i] for i in top_indices]
    
    return selected_comps

def _get_next_appraisal_id():
    appraisals = _load_appraisals()
    if not appraisals:
        return 1
    return max(int(id) for id in appraisals.keys()) + 1

def create_appraisal(subject: dict, candidates: list[dict]):
    appraisal_id = str(_get_next_appraisal_id())
    comps = _select_comps_ml(subject, candidates)
    status = "completed"

    # Save appraisal
    appraisals = _load_appraisals()
    appraisals[appraisal_id] = {
        "id": appraisal_id,
        "status": status,
        "subject": subject,
        "comps": comps
    }
    _save_appraisals(appraisals)

    # Add/replace properties in properties.json using centralized service
    all_props = [subject] + candidates
    update_properties(all_props)

    return appraisal_id, status

def get_appraisal_by_id(appraisal_id: str):
    appraisals = _load_appraisals()
    return appraisals.get(appraisal_id)