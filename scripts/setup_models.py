"""
Offline Model Setup — download ALL required AI models once, then run offline.

Run this script ONCE on any machine with internet access.
After that, the full pipeline runs with NO internet required.

Models downloaded
─────────────────
  1. YOLOv8n.pt          — person + ball detection (6 MB)
  2. YOLOv8s.pt          — higher accuracy option (22 MB)
  3. osnet_ain_x1_0_msmt17.pt  — strongest ReID, attention-based (11 MB)
  4. osnet_x1_0_msmt17.pt      — standard strong ReID (7 MB)
  5. osnet_x0_25_msmt17.pt     — lightweight ReID fallback (5 MB)

Usage
─────
    cd "Handball project"
    python scripts/setup_models.py

    # To download everything including YOLOv8m (better detection, 50MB):
    python scripts/setup_models.py --all
"""

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT   = Path(__file__).parent.parent
MODEL_DIR = ROOT   # models stored in project root (where boxmot/ultralytics look)

MODELS = {
    "yolov8n.pt": {
        "url":  "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
        "size": "6 MB",
        "desc": "YOLOv8 nano — fastest detection",
        "required": True,
    },
    "yolov8s.pt": {
        "url":  "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt",
        "size": "22 MB",
        "desc": "YOLOv8 small — better accuracy",
        "required": False,
    },
    "yolov8m.pt": {
        "url":  "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt",
        "size": "50 MB",
        "desc": "YOLOv8 medium — best accuracy (slower)",
        "required": False,
    },
    "osnet_ain_x1_0_msmt17.pt": {
        "url":  ("https://drive.google.com/uc?export=download"
                 "&id=1SigwBE6mPdqiJMqhuIY4aqC7--5CsMal"),
        "size": "11 MB",
        "desc": "OSNet-AIN x1.0 — strongest ReID, robust to appearance changes",
        "required": True,
        "gdrive": True,
        "boxmot_name": "osnet_ain_x1_0_msmt17",
    },
    "osnet_x1_0_msmt17.pt": {
        "url":  ("https://drive.google.com/uc?export=download"
                 "&id=1IosIFlLiulGIjwW3H8uMCC13LPUL73Ee"),
        "size": "7 MB",
        "desc": "OSNet x1.0 — strong ReID baseline",
        "required": False,
        "gdrive": True,
        "boxmot_name": "osnet_x1_0_msmt17",
    },
    "osnet_x0_25_msmt17.pt": {
        "url":  ("https://drive.google.com/uc?export=download"
                 "&id=1z1UghYvOTtjx7kEoRfmqSMu-4SJ2iYah"),
        "size": "5 MB",
        "desc": "OSNet x0.25 — lightweight ReID fallback",
        "required": False,
        "gdrive": True,
        "boxmot_name": "osnet_x0_25_msmt17",
    },
}


def download_via_boxmot(name: str) -> bool:
    """Use boxmot's built-in downloader for ReID models."""
    try:
        from boxmot.utils.checks import TestRequirements
        from boxmot.appearance.reid_model_factory import get_model_name
        from pathlib import Path
        import torch

        print(f"  Downloading {name} via boxmot...")
        # Instantiate BotSort which triggers auto-download
        from boxmot.trackers import BotSort
        _ = BotSort(
            reid_weights=Path(name),
            device=torch.device("cpu"),
            half=False,
        )
        print(f"  ✓ {name} downloaded by boxmot")
        return True
    except Exception as e:
        print(f"  ! boxmot download failed: {e}")
        return False


def download_yolo(name: str, url: str) -> bool:
    """Download YOLO model via ultralytics."""
    try:
        from ultralytics import YOLO
        print(f"  Downloading {name} via ultralytics...")
        model = YOLO(name)   # triggers auto-download if not present
        print(f"  ✓ {name} ready at {Path(name).resolve()}")
        return True
    except Exception as e:
        print(f"  ! ultralytics download failed: {e}")
        return False


def check_model(name: str) -> bool:
    """Check if a model file exists and has non-zero size."""
    p = MODEL_DIR / name
    if p.exists() and p.stat().st_size > 1024:
        return True
    # Also check current working directory
    p2 = Path(name)
    return p2.exists() and p2.stat().st_size > 1024


def main():
    parser = argparse.ArgumentParser(
        description="Download all offline AI models for Handball Coach")
    parser.add_argument("--all", action="store_true",
                        help="Download all models including large ones")
    parser.add_argument("--check", action="store_true",
                        help="Only check which models are present")
    args = parser.parse_args()

    print("=" * 60)
    print("  Handball AI Coach — Offline Model Setup")
    print("=" * 60)

    missing_required = []

    for name, info in MODELS.items():
        present = check_model(name)
        status  = "✓ present" if present else "✗ missing"
        req_tag = "[REQUIRED]" if info["required"] else "[optional]"
        print(f"\n  {req_tag} {name}  ({info['size']})")
        print(f"  {info['desc']}")
        print(f"  Status: {status}")

        if not present and info["required"]:
            missing_required.append(name)

    if args.check:
        if missing_required:
            print(f"\n⚠ Missing required models: {missing_required}")
            sys.exit(1)
        else:
            print("\n✓ All required models present.")
            sys.exit(0)

    print("\n" + "=" * 60)
    print("  Downloading models...")
    print("=" * 60)

    failed = []
    for name, info in MODELS.items():
        should_download = info["required"] or args.all
        if not should_download:
            continue
        if check_model(name):
            print(f"\n  ✓ {name} already present — skipping")
            continue

        print(f"\n  Downloading: {name} ({info['size']})...")
        ok = False

        if "gdrive" in info:
            ok = download_via_boxmot(name)
        elif name.endswith(".pt") and "yolo" in name.lower():
            ok = download_yolo(name, info["url"])

        if not ok:
            print(f"  ✗ Failed to download {name}")
            failed.append(name)

    print("\n" + "=" * 60)
    if failed:
        print(f"  ✗ Failed: {failed}")
        print("  Try running: pip install boxmot ultralytics --upgrade")
        sys.exit(1)
    else:
        print("  ✓ All models downloaded. System is ready for offline use.")
        print("  Run: python src/pipeline.py")


if __name__ == "__main__":
    main()
