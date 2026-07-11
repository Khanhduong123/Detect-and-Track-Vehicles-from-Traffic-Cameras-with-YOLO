# Dashboard Analytical endpoints
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.session import get_db
from backend.app.shared.logging import logger

router = APIRouter()


@router.get("/metrics")
async def get_metrics(db: AsyncSession = Depends(get_db)):
    """
    Retrieves system violations metrics, counts, and summaries for the analyst dashboard.
    """
    logger.info("Fetching violation metrics for analyst dashboard")
    # In real deployment, execute SQLAlchemy counts on ViolationEvent table
    # query = select(func.count(ViolationEvent.id)).filter(...)

    # Mock data output
    return {
        "total_violations": 128,
        "violations_by_type": {"wrong_way": 45, "speeding": 83},
        "monitored_cameras": ["cam_01", "cam_02", "cam_03"],
    }
