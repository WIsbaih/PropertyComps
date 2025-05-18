from fastapi import FastAPI
from api.endpoints import add, recommend, train, appraisal

app = FastAPI()

# Register all routers
app.include_router(add.router)
app.include_router(recommend.router)
app.include_router(train.router)
app.include_router(appraisal.router)
