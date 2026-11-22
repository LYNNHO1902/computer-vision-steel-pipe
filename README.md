# computer-vision-steel-pipe
# Autonomous Pipeline Defect Detection & Tracking API (YOLO11s + ONNX + ByteTrack)

End-to-end Industrial Computer Vision microservice designed for autonomous robotic crawlers and drone pipeline inspection. Built with **Ultralytics YOLO11s**, **ONNX Runtime**, **ByteTrack**, **FastAPI**, and **Docker**.

---

## Architecture Overview

```text
[Crawler / Drone Feed] ──► [FastAPI Microservice] ──► [YOLO11s ONNX Runtime (36.2 MB)]
                                                                │
[JSON Alert Report] ◄── [Persistence Filter (>= 3 frames)] ◄── [ByteTrack Multi-Object Tracker]
```

## Model Evaluation (Unseen Test Set — 2,138 Images / 2,773 Instances)

Full fine-tuning of **YOLO11s** (9.4M parameters, 21.4 GFLOPs) over 40 epochs at `640x640` resolution:

| Defect Class | Precision | Recall | mAP@50 | mAP@50-95 |
| :--- | :---: | :---: | :---: | :---: |
| **All Classes (6)** | **94.1%** | **91.2%** | **95.2%** | **73.6%** |
| **Disconnet** (Joint Separation) | 97.2% | 97.2% | 99.2% | 88.3% |
| **Misalignment** (Offset Joint) | 97.5% | 95.6% | 97.6% | 80.3% |
| **Desposition** (Sediment / Buildup) | 93.9% | 96.3% | 96.8% | 76.4% |
| **Obstacle** (Blockage / Debris) | 94.4% | 91.8% | 95.2% | 74.9% |
| **Deformation** (Wall Buckling) | 91.7% | 84.1% | 93.5% | 61.8% |
| **Rupture** (Crack / Collapse) | 89.9% | 82.1% | 89.0% | 60.2% |

## Engineering & Field Validation Highlights

* **Edge-Ready ONNX Export:** Converted from PyTorch (`.pt`) to graph-simplified **ONNX** (`models/best.onnx`, 36.2 MB), removing heavy PyTorch dependencies for fast, portable CPU/Edge inference (~4.3 ms/frame on T4 GPU).
* **Temporal Multi-Object Tracking (`ByteTrack`):** Tracks defects across consecutive video frames using composite keys (`ID_<track_id>_<class_name>`) to prevent ID switches on abrupt camera cuts and filter transient false positives (`min_frames_alert >= 3`).
* **Out-of-Distribution CCTV Field Testing:** Validated on real-world crawler/endoscope inspection footage outside the training distribution—accurately isolating telescoping pipe disconnects (`Disconnet`, 90% confidence) and wall ruptures (`Rupture`, 89.2% confidence), while documenting domain-shift boundaries (metallic vs. PVC/root intrusion) for future Active Learning loops.
* **Production Microservice:** Containerized with Docker Compose, exposing a **Live MJPEG Operator Dashboard** (`/`) alongside REST endpoints (`/predict/image`, `/simulate/video`).

## Project Structure

```text
pipe-defect-vision-api/
├── app/
│   ├── __init__.py
│   └── main.py              # FastAPI server, ONNX inference, ByteTrack & Live Dashboard
├── metrics                 # notebook Evaluation plots & prediction grids
├── models/
│   └── best.onnx            # Simplified YOLO11s ONNX weights (36.2 MB)
├── notebooks/               # Training & evaluation notebook
├── .dockerignore
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Quickstart (Docker)

```bash
git clone https://github.com
cd pipe-defect-vision-api
docker compose up --build
```

Once running, open your browser:

* **Interactive Swagger API Docs:** `http://localhost:8000/docs`
