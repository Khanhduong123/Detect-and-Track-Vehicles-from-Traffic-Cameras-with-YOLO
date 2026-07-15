# Redis Track Cache Client
import json
from typing import Dict, Optional

from backend.app.config.config import settings
from backend.app.shared.logging import logger


class RedisTrackCache:
    def __init__(self):
        try:
            import redis

            self.client = redis.from_url(settings.REDIS_URL)
            self.client.ping()
            self.use_mock = False
            logger.info("Successfully connected to Redis cache", url=settings.REDIS_URL)
        except Exception:
            logger.warning("Redis connection failed, falling back to mock cache")
            self._mock_db: Dict[str, str] = {}
            self.use_mock = True

    def get_track(self, track_id: int, video_id: str = "default") -> Optional[dict]:
        """
        Retrieve track trajectory and status from cache.
        """
        key = f"track:{video_id}:{track_id}"
        if self.use_mock:
            val = self._mock_db.get(key)
        else:
            try:
                redis_val = self.client.get(key)
                val = redis_val.decode("utf-8") if redis_val is not None else None
            except Exception as e:
                logger.error("Failed to read from Redis cache", error=str(e))
                val = None
        if val:
            return json.loads(val)
        return None

    def update_track(
        self, track_id: int, data: dict, video_id: str = "default", ttl: int = 3600
    ) -> None:
        """
        Store tracking information in Redis with a TTL.
        """
        key = f"track:{video_id}:{track_id}"
        val = json.dumps(data)
        if self.use_mock:
            self._mock_db[key] = val
        else:
            try:
                self.client.set(key, val, ex=ttl)
            except Exception as e:
                logger.error("Failed to write to Redis cache", error=str(e))
        logger.debug(
            "Updated track cache in Redis", track_id=track_id, video_id=video_id
        )


redis_cache = RedisTrackCache()
