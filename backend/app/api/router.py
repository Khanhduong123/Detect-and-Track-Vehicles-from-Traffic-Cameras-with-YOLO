# Main API Router
from fastapi import APIRouter

from backend.app.api.v1 import dashboard, videos

api_router = APIRouter()

api_router.include_router(videos.router, prefix="/videos", tags=["videos"])
api_router.include_router(dashboard.router, prefix="/dashboard", tags=["dashboard"])
