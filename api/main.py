from fastapi import FastAPI
from api.endpoints import add, recommend, train, appraisal, property

app = FastAPI()

# Register all routers
app.include_router(property.router, prefix="/api", tags=["properties"])
app.include_router(appraisal.router, prefix="/api", tags=["appraisals"])
