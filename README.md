# SIMS — Solapur Intelligent Mobility SaaS

> Transforms Solapur's 1,200 passive CCTV cameras into an active intelligent traffic management network — zero new hardware.

Built for **SAMVED 2026** Smart City Hackathon by MIT Vishwaprayag University.

---

## Prerequisites

- Python 3.9 or higher
- pip

---

## Setup

```bash
# 1. Clone the repository
git clone <repo-url>
cd sims

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Run

```bash
streamlit run app.py
```

Open the URL shown in your terminal (default: `http://localhost:8501`).

---

## YOLOv11n Model Weights

No manual download needed. On first run, `ultralytics` automatically downloads `yolo11n.pt` (~6 MB) and caches it locally. Subsequent runs use the cache.

---

## Usage

1. Launch the app with `streamlit run app.py`
2. In the sidebar, upload any traffic MP4 video
3. Adjust the confidence threshold slider if needed (default: 0.5)
4. Watch the live annotated feed, metrics, and alert log update in real time

---

## 4-Stage Pipeline

| Stage | Component | What it does |
|-------|-----------|-------------|
| **1 — Ingestion** | `app.py` + OpenCV | Accepts MP4 upload, reads frames via `VideoCapture` |
| **2 — Vision AI** | YOLOv11n (ultralytics) | Detects cars, buses, trucks, motorcycles with bounding boxes |
| **3 — Spatial Logic** | supervision `LineZone` + `PolygonZone` + `ByteTrack` | Counts vehicles crossing a virtual line; flags stationary vehicles in no-parking zones; tracks IDs across frames |
| **4 — Command** | `app.py` Streamlit dashboard | Displays live feed, metrics, Max-Pressure signal decisions, and parking violation alerts |

### Key Parameters (hardcoded for demo)

| Parameter | Value | Meaning |
|-----------|-------|---------|
| Lane capacity | 20 vehicles | Denominator for density % |
| Stationary threshold | 150 frames (~5 s) | Frames before parking violation is raised |
| Density threshold | 80% | Triggers "Extend Green 15s" |
| Active Learning threshold | confidence < 0.70 | Flags detections for human review |

---

## Project Structure

```
sims/
├── ai_engine.py        # TrafficAnalyzer class — the AI pipeline brain
├── app.py              # Streamlit dashboard — ingestion + command centre
├── requirements.txt    # Pinned dependencies
└── README.md
```

---

## Team

**fSociety** — MIT Academy of Engineering, Pune
SAMVED 2026 Smart City Hackathon — MIT Vishwaprayag University
