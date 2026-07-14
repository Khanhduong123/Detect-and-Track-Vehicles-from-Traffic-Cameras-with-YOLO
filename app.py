import os
import time

import requests
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Traffic AI Portal",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

BACKEND_URL = os.environ.get("BACKEND_URL", "http://localhost:8020")

# Custom Premium Styling (Premium Light Mode theme)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=Space+Grotesk:wght@400;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }

    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
    }

    /* Title Styling */
    .title-container {
        background: linear-gradient(135deg, #ffffff 0%, #f1f5f9 100%);
        border: 1px solid rgba(0, 0, 0, 0.06);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.04);
    }
    .main-title {
        font-family: 'Space Grotesk', sans-serif;
        font-weight: 800;
        font-size: 2.8rem;
        background: linear-gradient(90deg, #0284c7 0%, #7c3aed 50%, #db2777 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }
    .subtitle {
        color: #64748b;
        font-size: 1.1rem;
        font-weight: 300;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #f1f5f9;
        border-right: 1px solid rgba(0, 0, 0, 0.06);
    }

    /* Glowing Stat Cards */
    .kpi-card {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(0, 0, 0, 0.06);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.03);
        transition: all 0.3s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: rgba(124, 58, 237, 0.3);
        box-shadow: 0 8px 30px rgba(124, 58, 237, 0.08);
    }
    .kpi-title {
        color: #64748b;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .kpi-value {
        font-size: 2.2rem;
        font-weight: 800;
        margin-top: 5px;
        font-family: 'Space Grotesk', sans-serif;
    }

    /* Color code the KPI values */
    .val-blue { color: #0284c7; text-shadow: 0 2px 4px rgba(2, 132, 199, 0.1); }
    .val-purple { color: #7c3aed; text-shadow: 0 2px 4px rgba(124, 58, 237, 0.1); }
    .val-red { color: #dc2626; text-shadow: 0 2px 4px rgba(220, 38, 38, 0.1); }
    .val-orange { color: #ea580c; text-shadow: 0 2px 4px rgba(234, 88, 12, 0.1); }

    /* Alerts Panel */
    .alerts-container {
        background: #ffffff;
        border: 1px solid rgba(0, 0, 0, 0.06);
        border-radius: 12px;
        padding: 16px;
        height: 480px;
        overflow-y: auto;
    }
    .alert-item {
        background: rgba(220, 38, 38, 0.05);
        border-left: 4px solid #dc2626;
        color: #991b1b;
        padding: 10px 14px;
        border-radius: 4px;
        margin-bottom: 10px;
        animation: fadeIn 0.4s ease;
    }
    .alert-wrong-way {
        background: rgba(234, 88, 12, 0.05);
        border-left: 4px solid #ea580c;
        color: #9a3412;
    }
    .alert-time {
        font-size: 0.75rem;
        color: #64748b;
    }
    .alert-text {
        font-size: 0.95rem;
        font-weight: 600;
        margin-top: 2px;
        color: #0f172a;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(4px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Streamlit overrides */
    div.stButton > button:first-child {
        background: linear-gradient(90deg, #0284c7 0%, #7c3aed 100%);
        color: white;
        border: none;
        padding: 8px 24px;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s;
    }
    div.stButton > button:first-child:hover {
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.4);
        transform: scale(1.02);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------- SESSION STATE SETUP -----------------
if "video_id" not in st.session_state:
    st.session_state.video_id = None
if "status" not in st.session_state:
    st.session_state.status = "idle"
if "stats" not in st.session_state:
    st.session_state.stats = {
        "progress": 0.0,
        "total_vehicles": 0,
        "speed_violations": 0,
        "wrong_way_violations": 0,
        "fps": 0.0,
    }

# ----------------- HEADER & APP TITLE -----------------
st.markdown(
    """
    <div class="title-container">
        <div class="main-title">🚦 Traffic AI Portal</div>
        <div class="subtitle">Distributed Real-Time Vehicle Detection, Tracking, and Intelligent Violation Rule Engine</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------- SIDEBAR CONFIGURATION -----------------
st.sidebar.markdown("### ⚙️ Engine Settings")
st.sidebar.info("🤖 Inference Engine: Triton Server (yolov8_onnx)")
st.sidebar.info(f"🔗 Backend API: {BACKEND_URL}")

# Check Backend connection
try:
    response = requests.get(f"{BACKEND_URL}/")
    if response.status_code == 200:
        st.sidebar.success("✅ Backend Connected")
    else:
        st.sidebar.error("❌ Backend Connection Issue")
except Exception:
    st.sidebar.error("❌ Backend Offline")

# ----------------- METRICS DASHBOARD ROW -----------------
kpi_cols = st.columns(4)

with kpi_cols[0]:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Active Vehicles Count</div>
            <div class="kpi-value val-blue">{st.session_state.stats["total_vehicles"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_cols[1]:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Speeding Violations</div>
            <div class="kpi-value val-red">{st.session_state.stats["speed_violations"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_cols[2]:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Wrong Way Detections</div>
            <div class="kpi-value val-orange">{st.session_state.stats["wrong_way_violations"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with kpi_cols[3]:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Processing Speed</div>
            <div class="kpi-value val-purple">{st.session_state.stats["fps"]:.1f} FPS</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# ----------------- MAIN LAYOUT columns -----------------
col_feed, col_alerts = st.columns([2, 1])

with col_feed:
    st.markdown("### 📹 Video Processing Control")
    uploaded_file = st.file_uploader(
        "Upload a Traffic Video File", type=["mp4", "avi", "mov", "mkv"]
    )

    if uploaded_file is not None:
        if st.session_state.status == "idle":
            if st.button("▶️ Start Processing"):
                # Upload to backend
                with st.spinner("Uploading video to backend API..."):
                    try:
                        files = {
                            "file": (
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                uploaded_file.type,
                            )
                        }
                        res = requests.post(
                            f"{BACKEND_URL}/api/v1/videos/upload", files=files
                        )
                        if res.status_code == 200:
                            data = res.json()
                            st.session_state.video_id = data["video_id"]
                            st.session_state.status = "processing"
                            st.success(
                                f"Video uploaded successfully. Video ID: {st.session_state.video_id}"
                            )
                            st.rerun()
                        else:
                            st.error(f"Upload failed: {res.text}")
                    except Exception as e:
                        st.error(f"Error connecting to backend: {str(e)}")

        elif st.session_state.status == "processing":
            st.info("Video is currently processing in the background...")
            progress_bar = st.progress(0.0)

            # Polling placeholder
            status_text = st.empty()

            # Polling Loop
            while st.session_state.status == "processing":
                try:
                    res = requests.get(
                        f"{BACKEND_URL}/api/v1/videos/status/{st.session_state.video_id}"
                    )
                    if res.status_code == 200:
                        status_data = res.json()
                        progress = status_data.get("progress", 0.0)

                        st.session_state.stats = {
                            "progress": progress,
                            "total_vehicles": status_data.get("total_vehicles", 0),
                            "speed_violations": status_data.get("speed_violations", 0),
                            "wrong_way_violations": status_data.get(
                                "wrong_way_violations", 0
                            ),
                            "fps": status_data.get("fps", 0.0),
                        }

                        progress_bar.progress(progress / 100.0)
                        status_text.text(
                            f"Progress: {progress:.1f}% | FPS: {st.session_state.stats['fps']:.1f}"
                        )

                        if status_data.get("status") == "completed":
                            st.session_state.status = "completed"
                            st.success("Processing completed!")
                            st.rerun()
                        elif status_data.get("status") == "failed":
                            st.session_state.status = "failed"
                            st.error("Processing failed on backend.")
                            break
                    else:
                        st.error("Error retrieving status from backend.")
                        break
                except Exception as e:
                    st.error(f"Connection lost: {str(e)}")
                    break
                time.sleep(1.0)

        elif st.session_state.status == "completed":
            st.success("🎉 Video processing is finished!")
            video_url = (
                f"{BACKEND_URL}/api/v1/videos/download/{st.session_state.video_id}"
            )
            st.video(video_url)

            if st.button("🔄 Process Another Video"):
                st.session_state.status = "idle"
                st.session_state.video_id = None
                st.session_state.stats = {
                    "progress": 0.0,
                    "total_vehicles": 0,
                    "speed_violations": 0,
                    "wrong_way_violations": 0,
                    "fps": 0.0,
                }
                st.rerun()

with col_alerts:
    st.markdown("### 🔔 Event Violations Log")
    alerts_placeholder = st.empty()

    # Render Alerts from backend /dashboard/violations
    try:
        res = requests.get(f"{BACKEND_URL}/api/v1/dashboard/violations")
        if res.status_code == 200:
            violations = res.json()
            # Sort violations descending by timestamp or id
            violations.reverse()

            alert_html = '<div class="alerts-container">'
            if not violations:
                alert_html += '<div style="color: #94a3b8; text-align: center; margin-top: 40px;">No violations reported yet.</div>'
            else:
                for v in violations[:25]:
                    alert_cls = (
                        "alert-wrong-way" if v["violation_type"] == "wrong_way" else ""
                    )
                    time_str = (
                        v["timestamp"].split("T")[1][:8]
                        if "T" in v["timestamp"]
                        else v["timestamp"]
                    )

                    if v["violation_type"] == "speeding":
                        msg = f"Vehicle #{v['track_id']} exceeded speed limit! Speed: {v['metadata_json'].get('speed', 0.0):.1f} km/h"
                    else:
                        msg = f"Vehicle #{v['track_id']} detected driving WRONG WAY!"

                    alert_html += f"""
                    <div class="alert-item {alert_cls}">
                        <div class="alert-time">⏱️ {time_str} | {v['violation_type'].upper()}</div>
                        <div class="alert-text">{msg}</div>
                    </div>
                    """
            alert_html += "</div>"
            alerts_placeholder.markdown(alert_html, unsafe_allow_html=True)
    except Exception:
        alerts_placeholder.error("Could not fetch violation events log from backend.")
