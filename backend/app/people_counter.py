"""Generic people counter using YOLOv8 detection + pose.

This module is a backend-friendly extraction of the logic from
`model_training/carrillo_counter.py`, adapted to:

- Work with arbitrary image frames (np.ndarray, BGR).
- Be reusable for any dining hall (no file-path assumptions).
- Load YOLO models once per process.

Key entry point:

- `PeopleCounter.process_frame(image_bgr: np.ndarray, debug: bool = False)`
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple

import numpy as np
import cv2
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Configuration (can be overridden via environment variables later if needed)
# ---------------------------------------------------------------------------
YOLO_MODEL = "yolov8n.pt"
YOLO_POSE_MODEL = "yolov8n-pose.pt"
CONFIDENCE_THRESHOLD = 0.5
MATCH_CENTROID_FRACTION = 0.2
# Bias slightly toward "entering" and require stronger evidence for "exiting".
NOSE_CONF_THRESHOLD = 0.4
SHOULDER_CONF_THRESHOLD = 0.6

# COCO keypoint indices
KPT_NOSE = 0
KPT_LEFT_SHOULDER = 5
KPT_RIGHT_SHOULDER = 6
COCO_PERSON_CLS_ID = 0


def detect_people_with_model(
    det_model: YOLO, image_bgr: np.ndarray
) -> List[dict[str, Any]]:
    """Run YOLOv8 detection with a provided model and return person detections.

    This helper is intentionally model-agnostic so it can be reused by
    alternative counters (e.g. Ortega-specific Re-ID) without duplicating the
    detection logic.
    """
    results = det_model(image_bgr, verbose=False)[0]
    detections: List[dict[str, Any]] = []
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
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        area = (x2 - x1) * (y2 - y1)
        detections.append(
            {
                "bbox_xyxy": xyxy,
                "confidence": conf,
                "centroid": (float(cx), float(cy)),
                "area": float(area),
            }
        )
    return detections


def _iou_box(box1_xyxy: np.ndarray, box2_xyxy: np.ndarray) -> float:
    """Compute IoU between two boxes in xyxy format."""
    x1 = max(float(box1_xyxy[0]), float(box2_xyxy[0]))
    y1 = max(float(box1_xyxy[1]), float(box2_xyxy[1]))
    x2 = min(float(box1_xyxy[2]), float(box2_xyxy[2]))
    y2 = min(float(box1_xyxy[3]), float(box2_xyxy[3]))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    a1 = (box1_xyxy[2] - box1_xyxy[0]) * (box1_xyxy[3] - box1_xyxy[1])
    a2 = (box2_xyxy[2] - box2_xyxy[0]) * (box2_xyxy[3] - box2_xyxy[1])
    union = a1 + a2 - inter
    return float(inter / union) if union > 0 else 0.0


class _FrameProcessor:
    """Internal helper that mirrors the original Carrillo FrameProcessor."""

    def __init__(self, initial_entered: int = 0, initial_exited: int = 0) -> None:
        self.total_entered = int(initial_entered)
        self.total_exited = int(initial_exited)
        self._det_model = YOLO(YOLO_MODEL)
        self._pose_model = YOLO(YOLO_POSE_MODEL)
        self._prev_frame_data: List[dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # Detection & pose helpers
    # ------------------------------------------------------------------ #
    def detect_people(self, image_bgr: np.ndarray) -> List[dict[str, Any]]:
        """Run YOLOv8 detection; return list of person detections."""
        return detect_people_with_model(self._det_model, image_bgr)

    def filter_entrance_zone(
        self, detections: List[dict[str, Any]], frame_shape: Tuple[int, int, int]
    ) -> List[dict[str, Any]]:
        """Return all detections (use full-frame, no horizontal cropping)."""
        _ = frame_shape  # unused but kept for API symmetry
        return list(detections)

    def _pose_direction(
        self, kpts: np.ndarray | None, debug: bool = False, person_idx: int = 0
    ) -> str | None:
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
            print(
                f"  person {person_idx}: nose conf={nose_conf:.3f} xy=({nose[0]:.1f},{nose[1]:.1f}) "
                f"L_sh conf={left_conf:.3f} R_sh conf={right_conf:.3f}"
            )
        # Prefer classifying "entering" when we have a reasonably confident nose
        # and at least somewhat visible shoulders.
        if nose_conf > NOSE_CONF_THRESHOLD and left_conf > 0 and right_conf > 0:
            nose_y = float(nose[1])
            left_y = float(left_sh[1])
            right_y = float(right_sh[1])
            y_lo = min(left_y, right_y)
            y_hi = max(left_y, right_y)
            if y_lo <= nose_y <= y_hi:
                return "entering"  # facing camera = walking in
        # If the nose is strong but the simple geometric test fails (e.g. tilted
        # or off-center), still treat as entering instead of leaving it ambiguous.
        if nose_conf > NOSE_CONF_THRESHOLD and (left_conf > 0 or right_conf > 0):
            return "entering"
        # Make "exiting" stricter so we only count very clear away-facing poses.
        if (
            nose_conf < (NOSE_CONF_THRESHOLD * 0.7)
            and left_conf > SHOULDER_CONF_THRESHOLD
            and right_conf > SHOULDER_CONF_THRESHOLD
        ):
            return "exiting"  # facing away = walking out
        return None

    def classify_direction(
        self,
        person_detections: List[dict[str, Any]],
        image_bgr: np.ndarray,
        frame_width: int,
        debug: bool = False,
    ) -> Tuple[int, int, List[dict[str, Any]]]:
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
            return self._resolve_and_count(person_detections, frame_width, debug)

        kpt_data = pose_keypoints.data.cpu().numpy()

        for person_idx, d in enumerate(person_detections):
            d["direction"] = None
            d["keypoints"] = None
            best_iou = 0.0
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
                d["direction"] = self._pose_direction(
                    kpts, debug=debug, person_idx=person_idx
                )

        return self._resolve_and_count(person_detections, frame_width, debug)

    def _resolve_and_count(
        self,
        person_detections: List[dict[str, Any]],
        frame_width: int,
        debug: bool = False,
    ) -> Tuple[int, int, List[dict[str, Any]]]:
        """Resolve ambiguous with bbox-size; compute entered/exited counts."""
        _ = debug  # currently unused in this path
        match_dist = MATCH_CENTROID_FRACTION * frame_width
        for d in person_detections:
            if d.get("direction") is not None:
                continue
            cx, cy = d["centroid"]
            area = d["area"]
            best_dist = float("inf")
            best_prev: dict[str, Any] | None = None
            for p in self._prev_frame_data:
                pcx, pcy = p.get("centroid", (0.0, 0.0))
                dist = float(np.hypot(cx - pcx, cy - pcy))
                if dist < match_dist and dist < best_dist:
                    best_dist = dist
                    best_prev = p
            if best_prev is not None:
                prev_area = best_prev.get("area", 0.0)
                # Require a noticeable size change; treat small changes as ambiguous.
                if area > prev_area * 1.05:
                    d["direction"] = "entering"  # closer/larger = coming in
                elif area < prev_area * 0.95:
                    d["direction"] = "exiting"  # farther/smaller = going out

        entered = sum(1 for d in person_detections if d.get("direction") == "entering")
        exited = sum(1 for d in person_detections if d.get("direction") == "exiting")
        return entered, exited, person_detections

    # ------------------------------------------------------------------ #
    # Public-ish API used by PeopleCounter
    # ------------------------------------------------------------------ #
    def process_frame(
        self, image_bgr: np.ndarray, debug: bool = False
    ) -> Tuple[int, int, List[dict[str, Any]]]:
        """Process a single frame and update internal previous-frame data."""
        h, w = image_bgr.shape[:2]
        detections = self.detect_people(image_bgr)
        in_zone = self.filter_entrance_zone(detections, image_bgr.shape)
        entered, exited, updated = self.classify_direction(
            in_zone, image_bgr, w, debug=debug
        )
        self.total_entered += entered
        self.total_exited += exited
        # store minimal previous-frame data for next iteration
        self._prev_frame_data = [
            {
                "centroid": d["centroid"],
                "area": d["area"],
                "direction": d.get("direction"),
            }
            for d in updated
        ]
        return entered, exited, updated


def draw_annotations(
    image_bgr: np.ndarray,
    person_detections: List[dict[str, Any]],
    entered: int,
    exited: int,
    net_occupancy: int,
) -> np.ndarray:
    """Draw boxes (green=entering, red=exiting), keypoints, and HUD."""
    out = image_bgr.copy()
    for d in person_detections:
        xyxy = d["bbox_xyxy"]
        x1, y1, x2, y2 = map(int, xyxy)
        direction = d.get("direction")
        if direction == "entering":
            color = (0, 255, 0)
        elif direction == "exiting":
            color = (0, 0, 255)
        else:
            color = (128, 128, 128)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        if direction in ("entering", "exiting"):
            label = "enter" if direction == "entering" else "exit"
            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            tx = x1
            ty = max(y1 - 4, th + 2)
            cv2.rectangle(
                out,
                (tx, ty - th - 2),
                (tx + tw + 4, ty + 2),
                color,
                -1,
            )
            cv2.putText(
                out,
                label,
                (tx + 2, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        kpts = d.get("keypoints")
        if kpts is not None:
            for i in range(len(kpts)):
                x, y, c = kpts[i][0], kpts[i][1], kpts[i][2] if kpts[i].size > 2 else 0
                if c > 0.3:
                    cv2.circle(out, (int(x), int(y)), 3, (0, 255, 255), -1)

    hud = f"Entered: {entered} | Exited: {exited} | Net occupancy: {net_occupancy}"
    cv2.putText(
        out,
        hud,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        out,
        hud,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )
    return out


@dataclass
class PeopleCounterSnapshot:
    """Single snapshot of counter state after processing one frame."""

    entered: int
    exited: int
    total_entered: int
    total_exited: int
    net_occupancy: int


class PeopleCounter:
    """Stateful people counter reusable per dining hall."""

    def __init__(self, initial_entered: int = 0, initial_exited: int = 0) -> None:
        self._processor = _FrameProcessor(initial_entered=initial_entered, initial_exited=initial_exited)

    @property
    def total_entered(self) -> int:
        return int(self._processor.total_entered)

    @property
    def total_exited(self) -> int:
        return int(self._processor.total_exited)

    @property
    def net_occupancy(self) -> int:
        # Clamp at zero so occupancy never goes negative.
        return max(int(self.total_entered - self.total_exited), 0)

    def process_frame(
        self, image_bgr: np.ndarray, *, debug: bool = False
    ) -> Tuple[PeopleCounterSnapshot, np.ndarray, List[dict[str, Any]]]:
        """
        Process a single frame and return:

        - PeopleCounterSnapshot with deltas and totals.
        - Annotated image (np.ndarray, BGR).
        - Raw detection dicts with keypoints/directions.
        """
        entered, exited, detections = self._processor.process_frame(
            image_bgr, debug=debug
        )
        snap = PeopleCounterSnapshot(
            entered=entered,
            exited=exited,
            total_entered=self.total_entered,
            total_exited=self.total_exited,
            net_occupancy=self.net_occupancy,
        )
        annotated = draw_annotations(
            image_bgr,
            detections,
            snap.total_entered,
            snap.total_exited,
            snap.net_occupancy,
        )
        return snap, annotated, detections

