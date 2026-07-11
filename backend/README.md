# Traffic Violation Detection System - Backend

This directory houses the backend codebase for the Traffic Violation Detection System, structured in accordance with the system architecture layout (API Service, ML Inference cluster communication, Behavior Rule Engine, Ingestion, and Observability).

## Modules Overview

- **`app/main.py`**: API service main entry point (FastAPI).
- **`app/api/`**: API gateway routes (`/upload` for video streams, `/dashboard` for traffic stats).
- **`app/config/`**: Central system setup (configuration loading & structured logger initialization).
- **`app/db/`**: Persistence settings for PostgreSQL storing events metadata and logs.
- **`app/services/`**: Communication clients for external services (S3/GCS Storage, Redis track cache, Alert Notification).
- **`app/workers/`**: Async queue handlers (extraction of frames via OpenCV/FFmpeg).
- **`app/engine/`**: ML Inference endpoints interfacing with Triton Inference Server (YOLOv8 + ByteTrack).
- **`app/rules/`**: Rule engine logic (speed thresholds, wrong way movement check, trigger events).

## Getting Started

### Local Setup
1. Create a virtual environment and activate:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the development server:
   ```bash
   uvicorn app.main:app --reload
   ```

### Docker
Build and run via Docker:
```bash
docker build -t traffic-violation-backend .
docker run -p 8000:8000 traffic-violation-backend
```
