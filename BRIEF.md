# Handball AI System — Original Project Brief

> Act as a Senior AI Computer Vision Engineer, System Architect, and Sports Analytics Expert.
> I am building a fully offline, real-time AI performance analysis system for Handball matches.

---

## Hardware & Setup

- **Camera:** Hikvision DS-2CD2686G2-LZS (4K, Motorized Varifocal) — RTSP stream ingestion
- **Server:** Dell Precision 7740 · Core i9 9th Gen · 32GB RAM · NVIDIA RTX 3000
- **Network:** 100% OFFLINE — laptop acts as local server + Wi-Fi hotspot; phone and smartwatch connect to hotspot for live stats and notifications
- **Edge Devices:** NO Raspberry Pi or microcomputers — everything runs on the Dell laptop

---

## Software Constraints & Rules

- **AI Model:** YOLOv8 via Ultralytics — CUDA on RTX 3000
- **Backend:** Python + FastAPI
- **Paths:** STRICT — all directory paths must be dynamic (`pathlib` / `os.path`). No hardcoded absolute paths (system must be portable)
- **Communication:** WebSockets — FastAPI backend pushes live stats and notifications to a local HTML/JS dashboard

---

## System Capabilities (Phased)

| Phase | Goal |
|-------|------|
| 1 | RTSP stream ingestion, CUDA init, YOLOv8 player + ball detection |
| 2 | Team classification — Red vs Blue via HSV color extraction on bounding boxes |
| 3 | Court zoning + event logging (shots, attacking zone, score/stats) |
| 4 | Live notifications via WebSocket → phone browser dashboard |

---

## Deliverables Requested

1. Python project directory structure
2. `requirements.txt`
3. `main.py` — FastAPI + WebSocket setup with dynamic paths
4. `vision_engine.py` — RTSP ingestion (OpenCV), YOLOv8 inference forcing CUDA
5. Team color detection logic explanation for next iteration

**Code standard:** clean, highly commented, production-grade Python.
