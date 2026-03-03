"""File-based persistence for per-hall occupancy snapshots."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


HALL_CAPACITY: dict[str, int] = {
    "de-la-guerra": 350,
    "carrillo": 300,
    "ortega": 40,
    "portola": 500,
}


@dataclass
class HallOccupancy:
    hall: str
    timestamp: str  # ISO 8601 UTC
    total_entered: int
    total_exited: int
    net_occupancy: int
    max_capacity: int


def _get_data_dir() -> Path:
    root = Path(os.getenv("OCCUPANCY_DATA_ROOT", ".")).resolve()
    data_dir = root / "data" / "occupancy"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def _hall_path(hall: str) -> Path:
    return _get_data_dir() / f"{hall}.json"


def save_occupancy(snapshot: HallOccupancy) -> None:
    """Persist a snapshot atomically to disk."""
    path = _hall_path(snapshot.hall)
    tmp_path = path.with_suffix(".json.tmp")
    payload = asdict(snapshot)
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def load_occupancy(hall: str) -> Optional[HallOccupancy]:
    """Load the latest snapshot for a hall, if present."""
    path = _hall_path(hall)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    try:
        total_entered = int(data.get("total_entered", 0))
        total_exited = int(data.get("total_exited", 0))
        # Always recompute net_occupancy from totals and clamp at 0 so we
        # are not dependent on any stale or incorrect stored value.
        net_occupancy = max(total_entered - total_exited, 0)
        max_capacity = int(data.get("max_capacity", HALL_CAPACITY.get(hall, 0)))
        return HallOccupancy(
            hall=str(data.get("hall", hall)),
            timestamp=str(data.get("timestamp", "")),
            total_entered=total_entered,
            total_exited=total_exited,
            net_occupancy=net_occupancy,
            max_capacity=max_capacity,
        )
    except (TypeError, ValueError):
        return None


