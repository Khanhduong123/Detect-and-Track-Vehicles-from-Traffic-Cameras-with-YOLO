import datetime
from typing import Dict, Optional

from sqlalchemy import DateTime
from sqlalchemy.dialects.postgresql import JSONB
from sqlmodel import Column, Field, SQLModel


class ViolationEvent(SQLModel, table=True):
    __tablename__ = "violation_events"

    id: Optional[int] = Field(default=None, primary_key=True, index=True)
    violation_type: str = Field(index=True)  # wrong_way, speeding, etc.
    timestamp: datetime.datetime = Field(
        default_factory=datetime.datetime.utcnow,
        sa_column=Column(DateTime, default=datetime.datetime.utcnow, nullable=False),
    )

    # Store dynamic metadata such as bounding box coordinates, speeds, license plates, confidence scores
    metadata_json: Optional[Dict] = Field(
        default=None, sa_column=Column(JSONB, nullable=True)
    )

    # S3 URI pointing to the 3-5 seconds evidence video clip
    evidence_clip_url: Optional[str] = Field(default=None, nullable=True)

    # Tracking identification logs related to the infraction
    track_id: Optional[int] = Field(default=None, index=True, nullable=True)
    camera_id: str = Field(index=True)
