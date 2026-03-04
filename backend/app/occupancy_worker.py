"""Background worker that keeps per-hall occupancy up to date."""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import cv2
import numpy as np

from app.camera_fetcher import CameraFetcher, get_default_image_dir
from app.occupancy_store import HALL_CAPACITY, HallOccupancy, save_occupancy
from app.people_counter import PeopleCounter


def _get_poll_interval() -> float:
    try:
        return float(os.getenv("OCCUPANCY_POLL_INTERVAL_SECONDS", "5"))
    except ValueError:
        return 5.0


class OccupancyWorker:
    """Async background worker that tracks occupancy for all halls."""

    def __init__(self) -> None:
        self._stopped = asyncio.Event()
        self._camera_fetcher = CameraFetcher(get_default_image_dir())
        # When the backend starts, assume each hall is already at a fixed
        # percentage of its maximum capacity (all counted as "entered", none
        # as "exited").
        self._counters: Dict[str, PeopleCounter] = {}
        for hall in self._camera_fetcher.halls:
            max_cap = HALL_CAPACITY.get(hall, 0)
            # Per-hall starting occupancy as a fraction of capacity.
            if hall == "de-la-guerra":
                frac = 0.63
            elif hall == "carrillo":
                frac = 0.29
            elif hall == "portola":
                frac = 0.76
            elif hall == "ortega":
                frac = 0.5
            else:
                frac = 0.5

            baseline_entered = int(round(max_cap * frac)) if max_cap > 0 else 0
            baseline_exited = 0
            self._counters[hall] = PeopleCounter(
                initial_entered=baseline_entered, initial_exited=baseline_exited
            )
        self._poll_interval = _get_poll_interval()

    async def run(self) -> None:
        """Run until stopped."""
        while not self._stopped.is_set():
            try:
                await self._camera_fetcher.update_once()
                self._process_latest_frames()
            except Exception as exc:  # noqa: BLE001
                # Log and keep going; we do not want the worker to crash the backend.
                print(f"[occupancy_worker] Error in loop: {exc}")

            try:
                await asyncio.wait_for(self._stopped.wait(), timeout=self._poll_interval)
            except asyncio.TimeoutError:
                continue

    def stop(self) -> None:
        self._stopped.set()

    def _process_latest_frames(self) -> None:
        frames_by_hall = self._camera_fetcher.get_frames_snapshot()
        now_iso = datetime.now(timezone.utc).isoformat()
        for hall, frames in frames_by_hall.items():
            counter = self._counters[hall]
            last_annotated = None

            # Mirror the original logic: process previous, then current frame if present.
            for img_bytes in (frames.previous, frames.current):
                if not img_bytes:
                    continue
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                if img_bgr is None:
                    continue
                snapshot, annotated, _detections = counter.process_frame(img_bgr)
                last_annotated = annotated

            if counter.total_entered == 0 and counter.total_exited == 0:
                # Nothing processed yet for this hall.
                continue

            snapshot = HallOccupancy(
                hall=hall,
                timestamp=now_iso,
                total_entered=counter.total_entered,
                total_exited=counter.total_exited,
                net_occupancy=counter.net_occupancy,
                max_capacity=HALL_CAPACITY.get(hall, 0),
            )
            save_occupancy(snapshot)

            if last_annotated is not None:
                try:
                    root = Path(os.getenv("OCCUPANCY_DATA_ROOT", ".")).resolve()
                    annotated_dir = root / "data" / "annotated"
                    annotated_dir.mkdir(parents=True, exist_ok=True)
                    out_path = annotated_dir / f"{hall}.jpg"
                    cv2.imwrite(str(out_path), last_annotated)
                except Exception as exc:  # noqa: BLE001
                    print(f"[occupancy_worker] Failed to write annotated image for {hall}: {exc}")


async def start_occupancy_worker() -> OccupancyWorker:
    """Helper to start the worker in the background."""
    worker = OccupancyWorker()
    asyncio.create_task(worker.run())
    return worker

