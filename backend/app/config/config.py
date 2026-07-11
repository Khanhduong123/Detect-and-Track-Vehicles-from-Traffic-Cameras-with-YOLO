from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    PROJECT_NAME: str = "Traffic Violation Detection System"
    API_V1_STR: str = "/api/v1"

    # Data Persistence
    DATABASE_URL: str = (
        "postgresql+asyncpg://postgres:postgres@localhost:5432/traffic_violations"  # pragma: allowlist secret
    )

    # Redis (Cache for ByteTrack Track Cache & Celery broker)
    REDIS_URL: str = "redis://localhost:6379/0"

    # Cloud Storage (AWS S3 / GCS)
    STORAGE_BUCKET_NAME: str = "traffic-violation-evidence"
    AWS_ACCESS_KEY_ID: str = "mock_key"
    AWS_SECRET_ACCESS_KEY: str = "mock_secret"

    # AI / ML Triton Inference Server
    TRITON_SERVER_URL: str = "localhost:8001"
    YOLO_MODEL_NAME: str = "yolov8_tensorrt"

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=True
    )


settings = Settings()
