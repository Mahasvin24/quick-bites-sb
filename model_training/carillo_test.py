"""
Test script: run Carrillo counter on carrillo_previous.jpg and carrillo_current.jpg,
save marked-up outputs to the same output dir, and optionally display them.
Run from repo root: python3 model_training/carillo_test.py
Or from model_training: python3 carillo_test.py
"""

import os
import sys

import cv2

# Add parent so we can import from carrillo_counter when run from repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from carrillo_counter import (
    OUTPUT_DIR,
    FrameProcessor,
    draw_annotations,
)

# Paths relative to this script's directory (model_training/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATHS = [
    os.path.join(SCRIPT_DIR, "images", "carrillo_previous.jpg"),
    os.path.join(SCRIPT_DIR, "images", "carrillo_current.jpg"),
]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    processor = FrameProcessor()
    prev_frame_data = []

    for path in IMAGE_PATHS:
        if not os.path.isfile(path):
            print(f"Missing: {path}")
            continue
        img = cv2.imread(path)
        if img is None:
            print(f"Could not read: {path}")
            continue
        h, w = img.shape[:2]
        detections = processor.detect_people(img)
        in_zone = processor.filter_entrance_zone(detections, img.shape)
        entered, exited, updated = processor.classify_direction(
            in_zone, img, prev_frame_data, w, debug=False
        )
        processor.update_counts(entered, exited)
        annotated = draw_annotations(
            img,
            updated,
            img.shape,
            processor.total_entered,
            processor.total_exited,
            processor.net_occupancy,
        )
        out_name = os.path.basename(path)
        out_path = os.path.join(OUTPUT_DIR, out_name)
        cv2.imwrite(out_path, annotated)
        print(f"Saved: {out_path}")
        prev_frame_data = [
            {"centroid": d["centroid"], "area": d["area"], "direction": d.get("direction")}
            for d in updated
        ]

    print(f"\nEntered: {processor.total_entered} | Exited: {processor.total_exited} | Net: {processor.net_occupancy}")
    print(f"Outputs in: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == "__main__":
    main()
