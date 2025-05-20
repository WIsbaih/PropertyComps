from fastapi import APIRouter, HTTPException, Body
from typing import Dict, Any, List, Optional

from api.utils.validator import validate_property_input
from api.services.property_service import update_properties
from api.services.training_service import train_models

router = APIRouter()

@router.post("/properties")
async def add_properties(
    properties: List[Dict[str, Any]] = Body(...),
    retrain: Optional[bool] = Body(False, description="Whether to retrain models after adding properties")
):
    # Validate all properties
    for prop in properties:
        validate_property_input(prop)

    # Update properties using the service
    update_properties(properties)

    # Retrain models if requested
    if retrain:
        try:
            training_stats, _ = train_models()
            return {
                "message": "Properties added and models retrained successfully",
                "current_properties_count": len(properties),
                "training_stats": training_stats
            }
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Properties added but model retraining failed: {str(e)}"
            )

    return {
        "message": "Properties added successfully",
        "total_properties_count": len(properties)
    }