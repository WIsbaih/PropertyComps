from fastapi import APIRouter, HTTPException
from uuid import uuid4
from ..services.appraisal_service import create_appraisal, get_appraisal_by_id

router = APIRouter()

@router.post("/appraisals")
def post_appraisal(subject: dict, properties: list[dict]):
    appraisal_id, status = create_appraisal(subject, properties)
    return {"appraisal_id": appraisal_id, "status": status}

@router.get("/appraisals/{appraisal_id}")
def get_appraisal(appraisal_id: str):
    appraisal = get_appraisal_by_id(appraisal_id)
    if not appraisal:
        raise HTTPException(status_code=404, detail="Appraisal not found")
    return appraisal
