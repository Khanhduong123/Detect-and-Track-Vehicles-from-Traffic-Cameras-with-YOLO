# Dashboard Analytical endpoints
from fastapi import APIRouter

from backend.app.shared.logging import logger
from backend.app.shared.state import mock_db_store

router = APIRouter()


@router.get("/metrics")
async def get_metrics():
    """
    Retrieves system violations metrics, counts, and summaries for the analyst dashboard.
    """
    logger.info("Fetching violation metrics for analyst dashboard")
    speeding_count = sum(1 for e in mock_db_store if e.violation_type == "speeding")
    wrong_way_count = sum(1 for e in mock_db_store if e.violation_type == "wrong_way")

    return {
        "total_violations": len(mock_db_store),
        "violations_by_type": {
            "wrong_way": wrong_way_count,
            "speeding": speeding_count,
        },
        "monitored_cameras": ["cam_01"],
    }


@router.get("/violations")
async def get_violations():
    """
    Retrieves all logged violation events.
    """
    events = []
    for e in mock_db_store:
        events.append(
            {
                "id": e.id,
                "violation_type": e.violation_type,
                "timestamp": e.timestamp.isoformat(),
                "metadata_json": e.metadata_json,
                "track_id": e.track_id,
                "camera_id": e.camera_id,
            }
        )
    return events
