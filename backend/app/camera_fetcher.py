"""Camera still fetcher for UCSB dining halls.

This is a backend-local version of `model_training/update-images.py` that:

- Fetches still images for all supported dining halls.
- Maintains in-memory previous/current bytes per hall.
- Optionally writes JPEGs to disk for debugging.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import httpx

from app.config import UCSB_API_KEY, UCSB_BASE_URL


DINING_HALLS = ["carrillo", "de-la-guerra", "ortega", "portola"]
FETCH_INTERVAL_SECONDS = 5


@dataclass
class HallFrames:
    previous: Optional[bytes] = None
    current: Optional[bytes] = None


class CameraFetcher:
    """Fetch and manage latest frames for all dining halls."""

    def __init__(self, image_dir: Path) -> None:
        if not UCSB_API_KEY:
            raise RuntimeError(
                "UCSB_API_KEY/CONSUMER_KEY not configured; cannot fetch camera stills."
            )
        self._base_url = f"{UCSB_BASE_URL.rstrip('/')}/dining/cams/v2/still"
        self._image_dir = image_dir
        self._image_dir.mkdir(parents=True, exist_ok=True)
        self._frames: Dict[str, HallFrames] = {
            hall: HallFrames() for hall in DINING_HALLS
        }

    @property
    def halls(self) -> list[str]:
        return list(self._frames.keys())

    def get_frames_snapshot(self) -> Dict[str, HallFrames]:
        """Return a shallow copy of the current frames mapping."""
        return {hall: HallFrames(f.previous, f.current) for hall, f in self._frames.items()}

    def _hall_url(self, hall: str) -> str:
        return f"{self._base_url}/{hall}?ucsb-api-key={UCSB_API_KEY}"

    async def update_once(self) -> None:
        """Fetch one new frame for each hall (in parallel) and update internal state."""
        async with httpx.AsyncClient(timeout=10.0) as client:
            tasks = {
                hall: client.get(self._hall_url(hall))
                for hall in self._frames.keys()
            }
            results = await asyncio.gather(
                *tasks.values(), return_exceptions=True
            )

        for (hall, _), result in zip(tasks.items(), results, strict=False):
            if isinstance(result, Exception):
                print(f"{hall} request error: {result}")
                continue

            res = result
            if res.status_code != 200:
                print(f"{hall} error: {res.status_code}")
                continue

            hall_frames = self._frames[hall]
            hall_frames.previous = hall_frames.current
            hall_frames.current = res.content

            # Optional: write JPEGs for debugging/inspection.
            try:
                if hall_frames.previous:
                    with open(self._image_dir / f"{hall}_previous.jpg", "wb") as f:
                        f.write(hall_frames.previous)
                if hall_frames.current:
                    with open(self._image_dir / f"{hall}_current.jpg", "wb") as f:
                        f.write(hall_frames.current)
            except OSError as e:
                print(f"Failed to write images for {hall}: {e}")


def get_default_image_dir() -> Path:
    """Return default directory for saving camera images."""
    root = Path(os.getenv("OCCUPANCY_DATA_ROOT", ".")).resolve()
    return root / "data" / "images"


