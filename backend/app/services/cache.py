# Redis Track Cache Client
import json
from typing import Dict, Optional

from backend.app.shared.logging import logger


class RedisTrackCache:
    def __init__(self):
        # In a real environment, initialize redis client:
        # self.client = redis.from_url(settings.REDIS_URL)
        self._mock_db: Dict[str, str] = {}

    def get_track(self, track_id: int) -> Optional[dict]:
        """
        Retrieve track trajectory and status from cache.
        """
        key = f"track:{track_id}"
        val = self._mock_db.get(key)
        if val:
            return json.loads(val)
        return None

    def update_track(self, track_id: int, data: dict, ttl: int = 3600) -> None:
        """
        Store tracking information in Redis with a TTL.
        """
        key = f"track:{track_id}"
        self._mock_db[key] = json.dumps(data)
        logger.debug("Updated track cache in Redis", track_id=track_id)


redis_cache = RedisTrackCache()
