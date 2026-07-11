# FastAPI entrypoint
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from backend.app.api.router import api_router
from backend.app.config.config import settings
from backend.app.shared.logging import logger, setup_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize logger and system resources
    setup_logging()
    logger.info("Initializing Traffic Violation Detection System backend...")
    yield
    logger.info("Shutting down backend system...")


app = FastAPI(
    title=settings.PROJECT_NAME, lifespan=lifespan, docs_url="/docs", redoc_url="/redoc"
)


# Root Endpoint
@app.get("/")
def read_root():
    return {"message": "Traffic Violation Detection System Backend is operational."}


# Register API Router
app.include_router(api_router, prefix=settings.API_V1_STR)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
