from pathlib import Path
import json
from typing import Dict, Any, List

PROPERTIES_FILE = Path("data/properties.json")

# Initialize properties file if not exists
PROPERTIES_FILE.parent.mkdir(parents=True, exist_ok=True)
if not PROPERTIES_FILE.exists():
    PROPERTIES_FILE.write_text(json.dumps([]))

def load_properties() -> List[Dict[str, Any]]:
    """Load all properties from the JSON file."""
    return json.loads(PROPERTIES_FILE.read_text())

def save_properties(properties: List[Dict[str, Any]]):
    """Save properties to the JSON file."""
    PROPERTIES_FILE.write_text(json.dumps(properties, indent=2))

def update_properties(new_properties: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Update or add new properties to the existing properties.
    Returns the updated list of properties.
    """
    existing = load_properties()
    existing_map = {prop["id"]: prop for prop in existing if "id" in prop}

    for prop in new_properties:
        if "id" in prop:
            existing_map[prop["id"]] = prop  # add or replace

    updated = list(existing_map.values())
    save_properties(updated)
    return updated