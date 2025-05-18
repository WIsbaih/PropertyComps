from fastapi import APIRouter, HTTPException, Body
from typing import Dict, Any
from pathlib import Path
import json

from api.utils.validator import validate_property_input

router = APIRouter()

@router.post("/add")
async def add_property(new_property: Dict[str, Any] = Body(...)):
    validate_property_input(new_property)

    data_path = Path(__file__).resolve().parent.parent.parent / "data" / "properties_simple.json"

    if not data_path.exists():
        properties = []
    else:
        with open(data_path, "r") as f:
            properties = json.load(f)

    if any(p["id"] == new_property["id"] for p in properties):
        raise HTTPException(status_code=400, detail="Property ID already exists")

    properties.append(new_property)

    with open(data_path, "w") as f:
        json.dump(properties, f, indent=2)

    return {"message": "Property added. Retrain model to apply changes."}