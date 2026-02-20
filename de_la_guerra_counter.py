import json
import time
from datetime import datetime, timezone

import cv2
import numpy as np


DEFAULT_CONFIG = {
    # These are normalized coordinates (x/width, y/height).
    # Tune this polygon once you verify where de-la-guerra's entrance/exit appears.
    "entrance_polygon_norm": [
        (0, 0.42),
        (0.63, 0.42),
        (0.46, 1),
        (0, 1),
    ],
    "binary_threshold": 25,
    "min_motion_area": 350,
    "morph_kernel_size": 5,
    # With lower snapshot rates, people can move far between frames.
    # A larger association radius avoids dropping tracks before crossings are detected.
    "max_track_distance_px": 520,
    "track_ttl_frames": 8,
    "crossing_cooldown_sec": 5,
    "max_crossings_per_cycle": 8,
    # Ignore tiny centroid jitter near the polygon boundary.
    "min_direction_distance_px": 12,
    # Require motion to have a meaningful component toward/away from zone center.
    "min_projection_toward_zone_px": 6,
}


def _distance(point_a, point_b):
    return ((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2) ** 0.5


def _dot(vector_a, vector_b):
    return vector_a[0] * vector_b[0] + vector_a[1] * vector_b[1]


class DeLaGuerraCounter:
    def __init__(self, config=None, output_path="images/de-la-guerra_occupancy.json"):
        self.config = config or DEFAULT_CONFIG
        self.output_path = output_path
        self.debug_overlay_path = "images/de-la-guerra_debug_overlay.jpg"
        self.occupancy_state = {
            "occupancy": 0,
            "total_in": 0,
            "total_out": 0,
        }
        self.tracking_state = {
            "next_track_id": 1,
            "tracks": {},
        }

    def warmup(self):
        payload = self._build_payload(cycle_in=0, cycle_out=0, zone_polygon=np.array([]))
        self._write_payload(payload)
        return payload

    def process(self, previous_image_bytes, current_image_bytes):
        previous_frame = self._decode_jpeg_to_bgr(previous_image_bytes)
        current_frame = self._decode_jpeg_to_bgr(current_image_bytes)
        if previous_frame is None or current_frame is None:
            return None

        zone_polygon = self._polygon_from_normalized(
            self.config["entrance_polygon_norm"], current_frame.shape
        )
        moving_centroids = self._extract_motion_centroids(previous_frame, current_frame)
        now_unix = time.time()
        cycle_in, cycle_out = self._update_tracks_and_count_crossings(
            moving_centroids, zone_polygon, now_unix
        )

        self.occupancy_state["total_in"] += cycle_in
        self.occupancy_state["total_out"] += cycle_out
        self.occupancy_state["occupancy"] = max(
            0, self.occupancy_state["occupancy"] + cycle_in - cycle_out
        )
        self._write_debug_overlay(
            current_frame=current_frame,
            zone_polygon=zone_polygon,
            moving_centroids=moving_centroids,
            cycle_in=cycle_in,
            cycle_out=cycle_out,
        )

        payload = self._build_payload(cycle_in, cycle_out, zone_polygon)
        self._write_payload(payload)
        return payload

    def _decode_jpeg_to_bgr(self, image_bytes):
        array = np.frombuffer(image_bytes, dtype=np.uint8)
        return cv2.imdecode(array, cv2.IMREAD_COLOR)

    def _polygon_from_normalized(self, normalized_points, frame_shape):
        height, width = frame_shape[:2]
        points = []
        for x_norm, y_norm in normalized_points:
            points.append((int(x_norm * width), int(y_norm * height)))
        return np.array(points, dtype=np.int32)

    def _point_inside_polygon(self, point, polygon):
        return cv2.pointPolygonTest(polygon, point, False) >= 0

    def _polygon_centroid(self, polygon):
        moments = cv2.moments(polygon)
        if moments["m00"] == 0:
            return (0.0, 0.0)
        return (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])

    def _is_motion_direction_consistent(
        self, previous_centroid, new_centroid, zone_centroid, crossing_type
    ):
        move_vector = (
            new_centroid[0] - previous_centroid[0],
            new_centroid[1] - previous_centroid[1],
        )
        movement_distance = _distance(previous_centroid, new_centroid)
        if movement_distance < self.config["min_direction_distance_px"]:
            return False

        to_zone_vector = (
            zone_centroid[0] - previous_centroid[0],
            zone_centroid[1] - previous_centroid[1],
        )
        projection = _dot(move_vector, to_zone_vector) / max(
            _distance(previous_centroid, zone_centroid), 1e-6
        )
        min_projection = self.config["min_projection_toward_zone_px"]
        if crossing_type == "in":
            return projection >= min_projection
        return projection <= -min_projection

    def _extract_motion_centroids(self, previous_frame, current_frame):
        prev_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        prev_blur = cv2.GaussianBlur(prev_gray, (5, 5), 0)
        curr_blur = cv2.GaussianBlur(curr_gray, (5, 5), 0)

        frame_delta = cv2.absdiff(prev_blur, curr_blur)
        _, thresholded = cv2.threshold(
            frame_delta, self.config["binary_threshold"], 255, cv2.THRESH_BINARY
        )

        kernel_size = self.config["morph_kernel_size"]
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        cleaned = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        centroids = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.config["min_motion_area"]:
                continue

            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue

            centroid_x = int(moments["m10"] / moments["m00"])
            centroid_y = int(moments["m01"] / moments["m00"])
            centroids.append((centroid_x, centroid_y))

        return centroids

    def _update_tracks_and_count_crossings(self, centroids, polygon, now_unix):
        tracks = self.tracking_state["tracks"]
        unmatched_track_ids = set(tracks.keys())
        unmatched_centroid_indices = set(range(len(centroids)))
        zone_centroid = self._polygon_centroid(polygon)

        cycle_in = 0
        cycle_out = 0

        while unmatched_track_ids and unmatched_centroid_indices:
            best_track_id = None
            best_centroid_idx = None
            best_distance = None

            for track_id in unmatched_track_ids:
                track = tracks[track_id]
                track_centroid = track["centroid"]
                velocity = track["velocity"]
                predicted_centroid = (
                    track_centroid[0] + velocity[0],
                    track_centroid[1] + velocity[1],
                )
                for centroid_idx in unmatched_centroid_indices:
                    centroid = centroids[centroid_idx]
                    distance = _distance(predicted_centroid, centroid)
                    if best_distance is None or distance < best_distance:
                        best_distance = distance
                        best_track_id = track_id
                        best_centroid_idx = centroid_idx

            if best_distance is None or best_distance > self.config["max_track_distance_px"]:
                break

            track = tracks[best_track_id]
            new_centroid = centroids[best_centroid_idx]
            old_centroid = track["centroid"]
            was_inside = track["inside"]
            is_inside = self._point_inside_polygon(new_centroid, polygon)

            can_count_crossing = (
                now_unix - track["last_crossing_ts"] >= self.config["crossing_cooldown_sec"]
            )
            if can_count_crossing and was_inside != is_inside:
                if not was_inside and is_inside:
                    if self._is_motion_direction_consistent(
                        old_centroid, new_centroid, zone_centroid, "in"
                    ):
                        cycle_in += 1
                        track["last_crossing_ts"] = now_unix
                elif was_inside and not is_inside:
                    if self._is_motion_direction_consistent(
                        old_centroid, new_centroid, zone_centroid, "out"
                    ):
                        cycle_out += 1
                        track["last_crossing_ts"] = now_unix

            track["velocity"] = (
                new_centroid[0] - old_centroid[0],
                new_centroid[1] - old_centroid[1],
            )
            track["centroid"] = new_centroid
            track["inside"] = is_inside
            track["last_seen_ts"] = now_unix
            track["missed_frames"] = 0

            unmatched_track_ids.remove(best_track_id)
            unmatched_centroid_indices.remove(best_centroid_idx)

        for centroid_idx in unmatched_centroid_indices:
            centroid = centroids[centroid_idx]
            track_id = self.tracking_state["next_track_id"]
            self.tracking_state["next_track_id"] += 1
            tracks[track_id] = {
                "centroid": centroid,
                "inside": self._point_inside_polygon(centroid, polygon),
                "velocity": (0.0, 0.0),
                "last_seen_ts": now_unix,
                "last_crossing_ts": -1e9,
                "missed_frames": 0,
            }

        stale_track_ids = []
        for track_id in unmatched_track_ids:
            tracks[track_id]["missed_frames"] += 1
            if tracks[track_id]["missed_frames"] > self.config["track_ttl_frames"]:
                stale_track_ids.append(track_id)

        for track_id in stale_track_ids:
            del tracks[track_id]

        max_per_cycle = self.config["max_crossings_per_cycle"]
        cycle_in = min(cycle_in, max_per_cycle)
        cycle_out = min(cycle_out, max_per_cycle)

        return cycle_in, cycle_out

    def _build_payload(self, cycle_in, cycle_out, zone_polygon):
        return {
            "hall": "de-la-guerra",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "occupancy": self.occupancy_state["occupancy"],
            "total_in": self.occupancy_state["total_in"],
            "total_out": self.occupancy_state["total_out"],
            "cycle_in": cycle_in,
            "cycle_out": cycle_out,
            "entrance_polygon_px": zone_polygon.tolist(),
        }

    def _write_debug_overlay(
        self, current_frame, zone_polygon, moving_centroids, cycle_in, cycle_out
    ):
        debug_frame = current_frame.copy()

        overlay = debug_frame.copy()
        cv2.fillPoly(overlay, [zone_polygon], (0, 180, 0))
        debug_frame = cv2.addWeighted(overlay, 0.22, debug_frame, 0.78, 0)
        cv2.polylines(debug_frame, [zone_polygon], isClosed=True, color=(0, 255, 0), thickness=2)

        # Raw motion detections from frame differencing.
        for centroid_x, centroid_y in moving_centroids:
            cv2.circle(debug_frame, (int(centroid_x), int(centroid_y)), 4, (0, 255, 255), -1)

        tracks = self.tracking_state["tracks"]
        for track_id, track in tracks.items():
            centroid_x, centroid_y = track["centroid"]
            velocity_x, velocity_y = track["velocity"]
            is_inside = track["inside"]
            track_color = (0, 255, 0) if is_inside else (0, 100, 255)

            cv2.circle(debug_frame, (int(centroid_x), int(centroid_y)), 6, track_color, 2)
            cv2.putText(
                debug_frame,
                f"T{track_id}",
                (int(centroid_x) + 7, int(centroid_y) - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                track_color,
                1,
                cv2.LINE_AA,
            )

            arrow_end = (int(centroid_x + velocity_x), int(centroid_y + velocity_y))
            cv2.arrowedLine(
                debug_frame,
                (int(centroid_x), int(centroid_y)),
                arrow_end,
                (255, 255, 255),
                1,
                tipLength=0.25,
            )

        cv2.putText(
            debug_frame,
            f"cycle in:+{cycle_in} out:-{cycle_out} occ:{self.occupancy_state['occupancy']}",
            (12, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            debug_frame,
            f"tracks:{len(tracks)} detections:{len(moving_centroids)}",
            (12, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imwrite(self.debug_overlay_path, debug_frame)

    def _write_payload(self, payload):
        with open(self.output_path, "w", encoding="utf-8") as output_file:
            json.dump(payload, output_file, indent=2)
