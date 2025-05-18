from fastapi import HTTPException
from typing import Dict, Any

def validate_property_input(data: Dict[str, Any]):
    if "id" not in data:
        raise HTTPException(status_code=422, detail="'id' field is required")

    if not isinstance(data["id"], int):
        raise HTTPException(status_code=422, detail="'id' must be an integer")

    for key, value in data.items():
        if key == "id":
            continue

        if value is None:
            raise HTTPException(status_code=422, detail=f"'{key}' cannot be null")

        if not isinstance(value, (int, float)):
            raise HTTPException(status_code=422, detail=f"'{key}' must be a number")
