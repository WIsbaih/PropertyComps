import os
import json
import uuid
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

APPRAISALS_FILE = Path("data/appraisals.json")
PROPERTIES_FILE = Path("data/properties.json")

APPRAISALS_FILE.parent.mkdir(parents=True, exist_ok=True)
PROPERTIES_FILE.parent.mkdir(parents=True, exist_ok=True)

# Initialize appraisals file if not exists
if not APPRAISALS_FILE.exists():
    APPRAISALS_FILE.write_text(json.dumps({}))

# Initialize properties file if not exists
if not PROPERTIES_FILE.exists():
    PROPERTIES_FILE.write_text(json.dumps([]))

def _load_appraisals():
    return json.loads(APPRAISALS_FILE.read_text())

def _save_appraisals(data):
    APPRAISALS_FILE.write_text(json.dumps(data, indent=2))

def _load_properties():
    return json.loads(PROPERTIES_FILE.read_text())

def _save_properties(properties):
    PROPERTIES_FILE.write_text(json.dumps(properties, indent=2))

def _update_properties(new_properties: list[dict]):
    existing = _load_properties()
    existing_map = {prop["id"]: prop for prop in existing if "id" in prop}

    for prop in new_properties:
        if "id" in prop:
            existing_map[prop["id"]] = prop  # add or replace

    updated = list(existing_map.values())
    _save_properties(updated)

def _select_comps(subject: dict, candidates: list[dict], top_n=3):
    def combine_text(prop):
        return " ".join(str(v) for v in prop.values() if v is not None)

    texts = [combine_text(subject)] + [combine_text(p) for p in candidates]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(texts)

    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    top_indices = similarities.argsort()[::-1][:top_n]
    selected_comps = [candidates[i] for i in top_indices]

    return selected_comps

def create_appraisal(subject: dict, candidates: list[dict]):
    appraisal_id = str(uuid.uuid4())
    comps = _select_comps(subject, candidates)
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

    # Add/replace properties in properties.json
    all_props = [subject] + candidates
    _update_properties(all_props)

    return appraisal_id, status

def get_appraisal_by_id(appraisal_id: str):
    appraisals = _load_appraisals()
    return appraisals.get(appraisal_id)