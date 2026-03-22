"""
SIMS — Solapur Intelligent Mobility SaaS
Streamlit Command Centre Dashboard
Stage 1: Video ingestion
Stage 4: Live metrics + annotated feed + alert log
"""

import time
import tempfile
import os
from datetime import datetime
from pathlib import Path

import cv2
import streamlit as st

from ai_engine import TrafficAnalyzer

FLAGGED_DIR = "flagged_frames"
os.makedirs(FLAGGED_DIR, exist_ok=True)

# ------------------------------------------------------------------
# (a) Page config + session state initialisation
# ------------------------------------------------------------------

st.set_page_config(
    page_title="SIMS — Solapur Intelligent Mobility Command Centre",
    page_icon="🚦",
    layout="wide",
)

# Session state keys
if "alert_log" not in st.session_state:
    st.session_state.alert_log: list[dict] = []   # {"time": str, "msg": str}

if "total_vehicles_final" not in st.session_state:
    st.session_state.total_vehicles_final: int = 0

if "video_complete" not in st.session_state:
    st.session_state.video_complete: bool = False

if "last_density" not in st.session_state:
    st.session_state.last_density: float = 88.0

if "frame_counter" not in st.session_state:
    st.session_state.frame_counter: int = 0

if "density_history" not in st.session_state:
    st.session_state.density_history: list[float] = []

if "vehicle_history" not in st.session_state:
    st.session_state.vehicle_history: list[int] = []

if "class_counts" not in st.session_state:
    st.session_state.class_counts: dict[str, int] = {
        "car/auto": 0,
        "motorcycle/2W": 0,
        "ST bus": 0,
        "truck/tempo": 0,
    }

# ------------------------------------------------------------------
# (b) Sidebar — file uploader, confidence slider, system status
# ------------------------------------------------------------------

with st.sidebar:
    st.markdown("# 🚦 SIMS")
    st.markdown("**Solapur Intelligent Mobility SaaS**")
    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Upload Traffic Video",
        type=["mp4"],
        help="Upload an MP4 traffic video to begin analysis.",
    )

    confidence = st.slider(
        "Detection Confidence Threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.5,
        step=0.05,
    )

    st.markdown("---")
    st.info(
        "Simulating RTSP feed from Solapur ICCC. "
        "In production, this connects to 1,200 live CCTV streams."
    )

    st.markdown("**System Status**")
    st.markdown("- Model: `YOLOv11s`")
    st.markdown("- Tracker: `ByteTrack`")
    st.markdown("- Algorithm: `Max-Pressure`")
    st.markdown("---")

    # ------------------------------------------------------------------
    # (c) Sidebar expanders
    # ------------------------------------------------------------------

    with st.expander("📐 Max-Pressure Formula"):
        st.markdown(
            """
**Max-Pressure Traffic Signal Control**

```
p_z(k) = upstream_occupancy(k) − downstream_occupancy(k)

if p_z(k) > threshold:
    extend_green()
else:
    maintain_cycle()
```

**Plain English:** Each junction measures how many vehicles are
queued *upstream* versus how many have already cleared *downstream*.
When the upstream pressure exceeds the threshold, the green phase
is extended by 15 seconds.

Because every junction makes this calculation simultaneously and
shares occupancy data, signals coordinate network-wide — a **"Hive
Mind"** that prevents spillback gridlock across all junctions in
Solapur without any central controller bottleneck.
"""
        )

    with st.expander("📱 WhatsApp Enforcement Alerts (Production Feature)"):
        st.markdown(
            """
When a parking violation is confirmed, SIMS automatically sends a
WhatsApp message to officers in the duty roster via the
**WhatsApp Business API**. Officers review evidence and approve
E-Challans directly from WhatsApp — no separate dashboard needed.
"""
        )
        st.markdown(
            """
<div style="
    background-color: #dcf8c6;
    border-radius: 12px;
    padding: 14px 18px;
    font-family: monospace;
    font-size: 0.85rem;
    line-height: 1.6;
    max-width: 320px;
    box-shadow: 1px 1px 4px rgba(0,0,0,0.15);
">
🚨 <b>SIMS Alert — Solapur Traffic Police</b><br>
Violation: Illegal Parking<br>
Location: Camera 47 — Railway Station Road<br>
Vehicle ID: #12<br>
Duration: 3 min 20 sec<br>
Action: Tap to review evidence → [Link]
</div>
""",
            unsafe_allow_html=True,
        )

    with st.expander("📋 E-Challan Workflow (Production Feature)"):
        st.markdown(
            """
Every confirmed violation generates an **evidence package**:
- Timestamped video clip
- Detection metadata (vehicle ID, zone, duration)
- Confidence score

Officers review and approve with **one click**. SIMS never issues
challans autonomously — it is an evidence tool only.

✅ Compliant with **DPDP Act 2023** — non-violating vehicle faces
are Gaussian-blurred before storage or transmission.
"""
        )

    with st.expander("🔁 Active Learning Loop"):
        st.markdown(
            """
Detections below **0.70 confidence** are flagged automatically and
routed to **Label Studio** for human annotation.

**Target:** 500 Solapur-specific frames within 30 days of
deployment — covering ST buses, 6-seater autos, and tempos not
well-represented in standard COCO training data.

Annotated frames feed back into **YOLOv11 fine-tuning**, continuously
improving accuracy for Solapur's unique traffic mix.
"""
        )

# ------------------------------------------------------------------
# (d) Welcome screen — shown only when no file has been uploaded
# ------------------------------------------------------------------

if uploaded_file is None:
    st.title("🚦 SIMS — Solapur Intelligent Mobility SaaS")
    st.markdown(
        "#### Smart City Command Centre · SAMVED 2026 · MIT Vishwaprayag University"
    )
    st.markdown("---")

    col_prob, col_sol = st.columns(2)

    with col_prob:
        st.markdown("### 🏙️ The Problem")
        st.markdown(
            """
**Solapur — 1.1 million people, one broken intersection.**

- **Saat Rasta** junction: 7 roads converge with no coordinated signals
- **Railway Station Road**: chronic illegal parking blocks two of four lanes
- **Result**: peak-hour travel times 3× longer than necessary
- **1,200 CCTV cameras** installed city-wide — all passive, zero intelligence

The cameras watch. Nothing changes.
"""
        )

    with col_sol:
        st.markdown("### ⚡ The Solution")
        st.markdown(
            """
**SIMS transforms every camera into an active traffic agent.**

Zero new hardware. Zero civil works. Pure software upgrade to the
existing Solapur ICCC Smart City Command Centre.
"""
        )
        st.markdown(
            """
| Stage | What happens |
|-------|-------------|
| **1 — Ingestion** | RTSP frames from CCTV arrive in real time |
| **2 — Vision AI** | YOLOv11n detects cars, buses, trucks, motorcycles |
| **3 — Spatial Logic** | LineZone counts, PolygonZone flags illegal parking |
| **4 — Command** | Max-Pressure signals + WhatsApp alerts + E-Challans |
"""
        )

    st.markdown("---")
    st.info("👈 **Upload a traffic video in the sidebar to start the live demo.**")

# ------------------------------------------------------------------
# (e) Main layout — two-tab system
#     (only rendered when a file has been uploaded)
# ------------------------------------------------------------------

else:
    st.title("🚦 SIMS — Live Command Centre")

    tab1, tab2 = st.tabs(["📹 Live Feed", "🗺️ Network Map"])

    with tab1:
        # --- Metric row ---
        col1, col2, col3, col4 = st.columns(4)
        metric_total    = col1.empty()
        metric_density  = col2.empty()
        metric_signal   = col3.empty()
        metric_parking  = col4.empty()

        # --- Video feed placeholder ---
        st.markdown("#### 📹 Live Camera Feed")
        frame_placeholder = st.empty()

        # --- Alert section ---
        st.markdown("#### 🔔 Live Alert Feed")
        alert_placeholder = st.empty()

        # --- Charts + impact panel (placeholders for in-loop update) ---
        chart_placeholder = st.empty()
        class_chart_placeholder = st.empty()
        impact_placeholder = st.empty()

        # ------------------------------------------------------------------
        # (f) Video processing loop
        # ------------------------------------------------------------------

        # Reset state when a new file arrives
        st.session_state.video_complete = False
        st.session_state.alert_log = []
        st.session_state.frame_counter = 0
        st.session_state.density_history = []
        st.session_state.vehicle_history = []
        st.session_state.class_counts = {
            "car/auto": 0, "motorcycle/2W": 0,
            "ST bus": 0, "truck/tempo": 0,
        }

        # Write the uploaded bytes to a temp file so OpenCV can open it
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        analyzer = TrafficAnalyzer(confidence=confidence)
        cap = cv2.VideoCapture(tmp_path)

        if not cap.isOpened():
            st.error("Could not open video file. Please upload a valid MP4.")
        else:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Stage 2 + 3 + 4 — run the AI pipeline
                annotated_frame, metrics = analyzer.process_frame(frame)

                # BGR → RGB for Streamlit
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                # Track density for map tab
                st.session_state.last_density = metrics["density_percentage"]
                st.session_state.frame_counter += 1

                # Density + vehicle history for timeline chart (Task a)
                st.session_state.density_history.append(metrics["density_percentage"])
                st.session_state.vehicle_history.append(metrics["total_vehicles"])
                st.session_state.density_history = st.session_state.density_history[-100:]
                st.session_state.vehicle_history = st.session_state.vehicle_history[-100:]

                # Accumulate class counts (Task b)
                for cls, count in metrics.get("class_counts", {}).items():
                    st.session_state.class_counts[cls] = (
                        st.session_state.class_counts.get(cls, 0) + count
                    )

                # --- Update metric cards ---
                metric_total.metric("Total Vehicles", metrics["total_vehicles"])
                metric_density.metric("Density", f'{metrics["density_percentage"]}%')
                metric_parking.metric("Parking Violations", metrics["parking_violations"])

                # Signal action with colour logic
                signal = metrics["signal_action"]
                if signal == "Extend Green 15s":
                    metric_signal.markdown(
                        f"**Signal Decision**\n\n"
                        f'<p style="color:#28a745; font-size:1.4rem; font-weight:700;">'
                        f"🟢 {signal}</p>",
                        unsafe_allow_html=True,
                    )
                else:
                    metric_signal.metric("Signal Decision", signal)

                # --- Update video frame ---
                frame_placeholder.image(rgb_frame, use_container_width=True)

                # --- Update alert log ---
                _new_alerts = metrics["alerts"]
                if _new_alerts:
                    ts = datetime.now().strftime("%H:%M:%S")
                    for msg in _new_alerts:
                        st.session_state.alert_log.append({"time": ts, "msg": msg})
                    st.session_state.alert_log = st.session_state.alert_log[-10:]

                # Render alert feed
                with alert_placeholder.container():
                    if st.session_state.alert_log:
                        for entry in reversed(st.session_state.alert_log):
                            st.error(f'[{entry["time"]}] {entry["msg"]}')
                    else:
                        st.success("All clear — no violations detected")

                # --- Task (a): Density timeline chart ---
                if len(st.session_state.density_history) > 1:
                    import pandas as pd
                    with chart_placeholder.container():
                        st.markdown("#### Traffic Density Over Time")
                        chart_data = pd.DataFrame({
                            "Density %": st.session_state.density_history,
                        })
                        st.line_chart(chart_data, use_container_width=True, height=150)

                # --- Task (b): Vehicle class breakdown bar ---
                import pandas as pd
                with class_chart_placeholder.container():
                    st.markdown("#### Vehicle Class Breakdown (Cumulative)")
                    counts_df = pd.DataFrame({
                        "Vehicle Type": list(st.session_state.class_counts.keys()),
                        "Count": list(st.session_state.class_counts.values()),
                    })
                    st.bar_chart(
                        counts_df.set_index("Vehicle Type"),
                        use_container_width=True,
                        height=180,
                    )

                # --- Task (c): SIMS Impact Panel ---
                d = metrics["density_percentage"]
                if d >= 80:
                    without_sims = f"~{int(d * 0.4 + 10)} min avg wait"
                    with_sims = f"~{int(d * 0.4 + 10) - 8} min avg wait"
                    reduction = "20-30% reduction"
                    action = "🟢 Green extended +15s — clearing spillback now"
                elif d >= 50:
                    without_sims = f"~{int(d * 0.2 + 3)} min avg wait"
                    with_sims = f"~{int(d * 0.2 + 3) - 3} min avg wait"
                    reduction = "15-20% reduction"
                    action = "🟡 Monitoring — pre-emptive coordination active"
                else:
                    without_sims = "~2 min avg wait"
                    with_sims = "~1.5 min avg wait"
                    reduction = "Normal flow"
                    action = "✅ Free flow — standard cycle maintained"

                with impact_placeholder.container():
                    st.markdown("#### SIMS Impact — Live Estimate")
                    ic1, ic2, ic3 = st.columns(3)
                    ic1.metric("Without SIMS", without_sims, delta=None)
                    ic2.metric(
                        "With SIMS",
                        with_sims,
                        delta=f"-{reduction}",
                        delta_color="inverse",
                    )
                    ic3.metric("Current Action", action)

                # --- Task (d): Save flagged crops for Active Learning gallery ---
                for i, crop in enumerate(metrics.get("flagged_crops", [])):
                    ts = datetime.now().strftime("%H%M%S%f")
                    crop_path = os.path.join(FLAGGED_DIR, f"flag_{ts}_{i}.jpg")
                    cv2.imwrite(crop_path, crop)
                    existing = sorted(os.listdir(FLAGGED_DIR))
                    if len(existing) > 20:
                        for old in existing[:-20]:
                            os.remove(os.path.join(FLAGGED_DIR, old))

                # Throttle to prevent UI freezing
                time.sleep(0.01)

            cap.release()
            Path(tmp_path).unlink(missing_ok=True)

            # Record final vehicle count for completion screen
            st.session_state.total_vehicles_final = metrics["total_vehicles"]
            st.session_state.video_complete = True

        # ------------------------------------------------------------------
        # (h) Post-video completion screen
        # ------------------------------------------------------------------

        if st.session_state.video_complete:
            st.markdown("---")
            st.success(
                f"✅ Video processing complete. "
                f"Total vehicles counted: **{st.session_state.total_vehicles_final}**"
            )
            st.markdown(
                "Upload another video in the sidebar to run a new analysis session."
            )

    with tab2:
        from map_view import render_map

        last_density = st.session_state.get("last_density", 88.0)
        render_map(last_density)

        # --- Flagged Frames Gallery ---
        st.markdown("#### Flagged for Active Learning Review")
        st.caption(
            "Low-confidence detections (<0.70) captured this session. "
            "In production these route to Label Studio for annotation."
        )
        flagged_files = sorted(os.listdir(FLAGGED_DIR))[-9:]
        if flagged_files:
            cols = st.columns(3)
            for i, fname in enumerate(flagged_files):
                fpath = os.path.join(FLAGGED_DIR, fname)
                img = cv2.imread(fpath)
                if img is not None:
                    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    cols[i % 3].image(
                        rgb,
                        caption=f"Flagged #{i + 1}",
                        use_container_width=True,
                    )
        else:
            st.info("No frames flagged yet — upload a video to begin analysis.")
