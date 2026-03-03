"""
Carrillo dining hall entry/exit counter using YOLOv8 detection + pose.
Continuously analyzes carrillo_previous.jpg and carrillo_current.jpg every
SAMPLE_INTERVAL seconds, keeps a running count, and writes annotated frames to OUTPUT_DIR.
"""

import argparse
import os
import time

import cv2
import numpy as np
from ultralytics import YOLO

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_PATHS = [
    os.path.join(SCRIPT_DIR, "images", "carrillo_previous.jpg"),
    os.path.join(SCRIPT_DIR, "images", "carrillo_current.jpg"),
]
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output", "carrillo")
SAMPLE_INTERVAL = 5
ENTRANCE_ZONE_X = (0.0, 1.0)
YOLO_MODEL = "yolov8n.pt"
YOLO_POSE_MODEL = "yolov8n-pose.pt"
CONFIDENCE_THRESHOLD = 0.5
MATCH_CENTROID_FRACTION = 0.2
NOSE_CONF_THRESHOLD = 0.5
SHOULDER_CONF_THRESHOLD = 0.5

# COCO keypoint indices
KPT_NOSE = 0
KPT_LEFT_SHOULDER = 5
KPT_RIGHT_SHOULDER = 6
COCO_PERSON_CLS_ID = 0


def _iou_box(box1_xyxy, box2_xyxy):
    """Compute IoU between two boxes in xyxy format."""
    x1 = max(box1_xyxy[0], box2_xyxy[0])
    y1 = max(box1_xyxy[1], box2_xyxy[1])
    x2 = min(box1_xyxy[2], box2_xyxy[2])
    y2 = min(box1_xyxy[3], box2_xyxy[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = (box1_xyxy[2] - box1_xyxy[0]) * (box1_xyxy[3] - box1_xyxy[1])
    a2 = (box2_xyxy[2] - box2_xyxy[0]) * (box2_xyxy[3] - box2_xyxy[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


class FrameProcessor:
    def __init__(self):
        self.total_entered = 0
        self.total_exited = 0
        self._det_model = YOLO(YOLO_MODEL)
        self._pose_model = YOLO(YOLO_POSE_MODEL)
        self._prev_frame_data = []

    def detect_people(self, image_bgr):
        """Run YOLOv8 detection; return list of person detections."""
        results = self._det_model(image_bgr, verbose=False)[0]
        detections = []
        if results.boxes is None:
            return detections
        boxes = results.boxes
        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i].item())
            if cls_id != COCO_PERSON_CLS_ID:
                continue
            conf = float(boxes.conf[i].item())
            if conf < CONFIDENCE_THRESHOLD:
                continue
            xyxy = boxes.xyxy[i].cpu().numpy()
            x1, y1, x2, y2 = xyxy
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            area = (x2 - x1) * (y2 - y1)
            detections.append({
                "bbox_xyxy": xyxy,
                "confidence": conf,
                "centroid": (float(cx), float(cy)),
                "area": float(area),
            })
        return detections

    def filter_entrance_zone(self, detections, frame_shape):
        """Return all detections (use full-frame, no horizontal cropping)."""
        # Previous behavior limited to a horizontal entrance band; we now keep
        # all detections so the model considers the entire image.
        return list(detections)

    def _pose_direction(self, kpts, debug=False, person_idx=0):
        """
        Classify direction from keypoints. Returns 'entering', 'exiting', or None.
        Nose (0) between shoulders (5,6) y -> entering (facing camera, walking in).
        Nose not usable but both shoulders visible -> exiting (facing away, walking out).
        """
        if kpts is None or len(kpts) < 17:
            return None
        nose = kpts[KPT_NOSE]
        left_sh = kpts[KPT_LEFT_SHOULDER]
        right_sh = kpts[KPT_RIGHT_SHOULDER]
        nose_conf = float(nose[2]) if len(nose) > 2 else 0.0
        left_conf = float(left_sh[2]) if len(left_sh) > 2 else 0.0
        right_conf = float(right_sh[2]) if len(right_sh) > 2 else 0.0
        if debug:
            print(f"  person {person_idx}: nose conf={nose_conf:.3f} xy=({nose[0]:.1f},{nose[1]:.1f}) "
                  f"L_sh conf={left_conf:.3f} R_sh conf={right_conf:.3f}")
        if nose_conf > NOSE_CONF_THRESHOLD and left_conf > 0 and right_conf > 0:
            nose_y = float(nose[1])
            left_y = float(left_sh[1])
            right_y = float(right_sh[1])
            y_lo = min(left_y, right_y)
            y_hi = max(left_y, right_y)
            if y_lo <= nose_y <= y_hi:
                return "entering"  # facing camera = walking in
        if nose_conf <= NOSE_CONF_THRESHOLD and left_conf > SHOULDER_CONF_THRESHOLD and right_conf > SHOULDER_CONF_THRESHOLD:
            return "exiting"  # facing away = walking out
        return None

    def classify_direction(self, person_detections, image_bgr, prev_frame_data, frame_width, debug=False):
        """
        Run pose for each person, classify direction; use bbox-size fallback for ambiguous.
        Returns (entered_count, exited_count, list of dicts with 'direction' and keypoints).
        """
        if not person_detections:
            return 0, 0, []
        pose_results = self._pose_model(image_bgr, verbose=False)[0]
        pose_boxes = pose_results.boxes
        pose_keypoints = pose_results.keypoints
        if pose_boxes is None or pose_keypoints is None:
            for d in person_detections:
                d["direction"] = None
                d["keypoints"] = None
            return self._resolve_and_count(person_detections, prev_frame_data, frame_width, debug)
        kpt_data = pose_keypoints.data.cpu().numpy()
        det_xyxy = [d["bbox_xyxy"] for d in person_detections]
        for person_idx, d in enumerate(person_detections):
            d["direction"] = None
            d["keypoints"] = None
            best_iou = 0
            best_idx = -1
            for idx in range(len(pose_boxes)):
                if int(pose_boxes.cls[idx].item()) != COCO_PERSON_CLS_ID:
                    continue
                pb = pose_boxes.xyxy[idx].cpu().numpy()
                iou = _iou_box(d["bbox_xyxy"], pb)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_idx >= 0 and best_iou > 0.1:
                kpts = kpt_data[best_idx]
                d["keypoints"] = kpts
                d["direction"] = self._pose_direction(kpts, debug=debug, person_idx=person_idx)
        return self._resolve_and_count(person_detections, prev_frame_data, frame_width, debug)

    def _resolve_and_count(self, person_detections, prev_frame_data, frame_width, debug):
        """Resolve ambiguous with bbox-size; compute entered/exited counts."""
        match_dist = MATCH_CENTROID_FRACTION * frame_width
        for d in person_detections:
            if d["direction"] is not None:
                continue
            cx, cy = d["centroid"]
            area = d["area"]
            best_dist = float("inf")
            best_prev = None
            for p in prev_frame_data:
                pcx, pcy = p.get("centroid", (0, 0))
                dist = np.hypot(cx - pcx, cy - pcy)
                if dist < match_dist and dist < best_dist:
                    best_dist = dist
                    best_prev = p
            if best_prev is not None:
                prev_area = best_prev.get("area", 0)
                if area > prev_area:
                    d["direction"] = "entering"  # closer/larger = coming in
                elif area < prev_area:
                    d["direction"] = "exiting"  # farther/smaller = going out
        entered = sum(1 for d in person_detections if d["direction"] == "entering")
        exited = sum(1 for d in person_detections if d["direction"] == "exiting")
        return entered, exited, person_detections

    def update_counts(self, entered_count, exited_count):
        self.total_entered += entered_count
        self.total_exited += exited_count

    @property
    def net_occupancy(self):
        return self.total_entered - self.total_exited


def draw_annotations(image_bgr, person_detections, frame_shape, entered, exited, net_occupancy):
    """Draw boxes (green=entering, red=exiting), keypoints, and HUD."""
    out = image_bgr.copy()
    for d in person_detections:
        xyxy = d["bbox_xyxy"]
        x1, y1, x2, y2 = map(int, xyxy)
        color = (0, 255, 0) if d.get("direction") == "entering" else (0, 0, 255) if d.get("direction") == "exiting" else (128, 128, 128)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        direction = d.get("direction")
        if direction in ("entering", "exiting"):
            label = "enter" if direction == "entering" else "exit"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            tx = x1
            ty = max(y1 - 4, th + 2)
            cv2.rectangle(out, (tx, ty - th - 2), (tx + tw + 4, ty + 2), color, -1)
            cv2.putText(out, label, (tx + 2, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        kpts = d.get("keypoints")
        if kpts is not None:
            for i in range(len(kpts)):
                x, y, c = kpts[i][0], kpts[i][1], kpts[i][2] if kpts[i].size > 2 else 0
                if c > 0.3:
                    cv2.circle(out, (int(x), int(y)), 3, (0, 255, 255), -1)
    hud = f"Entered: {entered} | Exited: {exited} | Net occupancy: {net_occupancy}"
    cv2.putText(out, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(out, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
    return out


def main():
    parser = argparse.ArgumentParser(description="Carrillo entry/exit counter (YOLOv8) – continuous mode")
    parser.add_argument("--debug", action="store_true", help="Print keypoint coords and confidence per person")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    processor = FrameProcessor()
    prev_frame_data = []

    print(f"Running every {SAMPLE_INTERVAL}s on: {IMAGE_PATHS[0]!r}, {IMAGE_PATHS[1]!r}")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
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
                    in_zone, img, prev_frame_data, w, debug=args.debug
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
                prev_frame_data = [
                    {"centroid": d["centroid"], "area": d["area"], "direction": d.get("direction")}
                    for d in updated
                ]
            print(
                f"Entered: {processor.total_entered} | Exited: {processor.total_exited} | "
                f"Net: {processor.net_occupancy}"
            )
            time.sleep(SAMPLE_INTERVAL)
    except KeyboardInterrupt:
        print("\nStopped.")
        print(f"  Total entered:  {processor.total_entered}")
        print(f"  Total exited:   {processor.total_exited}")
        print(f"  Net occupancy:  {processor.net_occupancy}")


if __name__ == "__main__":
    main()
