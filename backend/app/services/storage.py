# AWS S3 / GCS Mock Service for Evidence Clips storage.py
from backend.app.config.config import settings
from backend.app.shared.logging import logger


class StorageService:
    def __init__(self):
        self.bucket = settings.STORAGE_BUCKET_NAME
        # Initialize boto3 S3 or google-cloud-storage client here

    async def upload_evidence_clip(self, file_path: str, filename: str) -> str:
        """
        Uploads a 3-5 seconds evidence video clip to S3/GCS.
        Returns the public/signed URL of the uploaded evidence.
        """
        logger.info(
            "Uploading video evidence clip to bucket", bucket=self.bucket, file=filename
        )
        # Mock successful upload return URL
        return f"https://s3.amazonaws.com/{self.bucket}/{filename}"


storage_service = StorageService()
