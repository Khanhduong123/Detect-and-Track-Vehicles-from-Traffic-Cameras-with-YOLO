# 🚦 Traffic AI Portal - Run & Installation Guide

This repository contains a real-time, distributed Traffic Violation Detection and Tracking System. The architecture consists of three core layers:

1. **Frontend (Streamlit)**: A premium interactive dashboard for video upload, real-time KPI streaming, and visual alerts.
2. **Backend (FastAPI)**: High-performance async API server handling background frames extraction, ByteTrack tracking, rule engine validation, and video annotation.
3. **ML Inference Server (Triton)**: High-throughput YOLOv8 ONNX model serving.

---

## 🏗️ Architecture Overview

```mermaid
graph TD
    A[Streamlit Web UI] <-->|HTTP / REST| B[FastAPI Backend]
    B -->|Frames Extraction| C[Celery / Async Workers]
    C <-->|gRPC / HTTP| D[Triton Inference Server]
    D -->|YOLOv8 ONNX| E[(model_repository)]
    C -->|ByteTrack / Rule Engine| F[Violation Logger]
    F -->|In-memory / DB fallback| G[(Mock Database)]
```

---

## ⚡ Quick Start: How to Run

### Step 1: Start Triton Inference Server

Triton is used to serve the YOLOv8 model for object detection. Ensure Docker is installed on your system.

1. **Model Check**: The model file `model.onnx` is already located at `model_repository/yolov8_onnx/1/model.onnx`.
2. **Start the Triton Container**:
   Run the following command from the **repository root directory** to start Triton:

   * **Option A: Run on CPU (Default/Testing)**
     ```bash
     docker run --name triton_traffic_yolo \
       -p 8000:8000 -p 8001:8001 -p 8002:8002 \
       -v $(pwd)/model_repository:/models \
       nvcr.io/nvidia/tritonserver:24.04-py3 \
       tritonserver --model-repository=/models
     ```

   * **Option B: Run on GPU (Recommended for Production)**
     ```bash
     docker run --gpus all --name triton_traffic_yolo \
       -p 8000:8000 -p 8001:8001 -p 8002:8002 \
       -v $(pwd)/model_repository:/models \
       nvcr.io/nvidia/tritonserver:24.04-py3 \
       tritonserver --model-repository=/models
     ```

   > [!NOTE]
   > Triton uses port **8000** for HTTP requests, **8001** for gRPC, and **8002** for metrics. The backend client will automatically translate gRPC port configurations to HTTP requests.

---

### Step 2: Start the FastAPI Backend

The FastAPI backend handles orchestration, video processing queue, and vehicle tracking.

1. **Create and Activate a Virtual Environment** (from repository root):
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```
2. **Install Dependencies**:
   ```bash
   pip install -r backend/requirements.txt
   ```
3. **Run the Backend Server**:
   By default, the backend runs on port `8020` to avoid conflicts with Triton's HTTP port. Run the following command from the repository root:
   ```bash
   PYTHONPATH=. uvicorn backend.app.main:app --host 0.0.0.0 --port 8020 --reload
   ```

   > [!TIP]
   > **Out-of-the-box Fallbacks**: The backend is configured to fall back to an in-memory database and mock services automatically if PostgreSQL or Redis are not detected. You do not need database setups for quick testing!

---

### Step 3: Run the Streamlit Frontend

The frontend allows you to upload traffic videos and view vehicle speeds, wrong-way violations, and tracking bounding boxes in real-time.

1. **Return to the Repository Root**:
   ```bash
   cd ..
   ```
2. **Activate the Virtual Environment (if not active)**:
   ```bash
   source backend/venv/bin/activate
   ```
3. **Install Frontend Dependencies**:
   Make sure you have `streamlit`, `requests`, and `protobuf` installed (to avoid any protobuf dependency errors):
   ```bash
   pip install streamlit requests protobuf
   ```
4. **Start Streamlit**:
   Since the backend is running on port `8020` (which is the default port the frontend expects), you can start the application directly:
   ```bash
   streamlit run app.py
   ```
5. **Access the Web Portal**:
   Open your browser and navigate to the URL printed in the terminal (typically `http://localhost:8501`).

---

## ⚙️ Configuration & Environment Variables

You can customize port numbers, database URLs, and server endpoints by creating a `.env` file in the `backend/` directory. Refer to `backend/.env.example`:

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| `DATABASE_URL` | `postgresql+asyncpg://postgres:postgres@localhost:5432/traffic_violations` <!-- pragma: allowlist secret --> | Connection string for PostgreSQL database (falls back to mock if unavailable) |
| `REDIS_URL` | `redis://localhost:6379/0` | Cache endpoint for tracking cache & task queuing |
| `TRITON_SERVER_URL` | `localhost:8001` | URL of the Triton Inference Server endpoint |
| `YOLO_MODEL_NAME` | `yolov8_onnx` | Model name matching the Triton directory |

---

## 🚦 Testing the Flow

1. Open the **Traffic AI Portal** in your browser.
2. Select a traffic video file (e.g. from the `data/` or `dataset/` directories) and click **Start Processing**.
3. Watch the progress bar advance while real-time speed and wrong-way violation alerts stream in on the right-hand panel.
4. Once completed, play or download the annotated video output!
