# Shared global state for job status and mock database storage
from typing import Any, Dict, List

# Stores background job statistics and progress
# Format: { "video_id": { "status": "processing" | "completed" | "failed", "progress": float, "total_vehicles": int, "speed_violations": int, "wrong_way_violations": int, "fps": float } }
jobs_status: Dict[str, Any] = {}

# In-memory database store for ViolationEvent mock records
mock_db_store: List[Any] = []
