from fastapi import HTTPException
from typing import Dict, Any

def validate_property_input(data: Dict[str, Any]):
  from fastapi import HTTPException
from typing import Dict, Any

def validate_property_input(data: Dict[str, Any]):
    """Validate property input data."""
    # Required fields
    required_fields = ["id", "address", "bedrooms", "gla"]
    
    # Convert nulls to defaults for required fields
    for field in required_fields:
        if field not in data:
            raise HTTPException(status_code=422, detail=f"'{field}' field is required")
        if data[field] is None:
            if field in ["id", "bedrooms", "gla"]:
                data[field] = 0
            else:  # address
                data[field] = ""

    # Validate ID is an integer
    if not isinstance(data["id"], int):
        raise HTTPException(status_code=422, detail="'id' must be an integer")

    # Validate bedrooms is a number
    if not isinstance(data["bedrooms"], (int, float)):
        raise HTTPException(status_code=422, detail="'bedrooms' must be a number")

    # Validate gla (Gross Living Area) is a number
    if not isinstance(data["gla"], (int, float)):
        raise HTTPException(status_code=422, detail="'gla' must be a number")

    # Validate address is a string
    if not isinstance(data["address"], str):
        raise HTTPException(status_code=422, detail="'address' must be a string")

    return data
