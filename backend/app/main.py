"""FastAPI app: UCSB dining menu proxy and health."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from app.occupancy_store import HallOccupancy, load_occupancy
from app.occupancy_worker import start_occupancy_worker

from app.config import MENU_CACHE_TTL_SECONDS, UCSB_API_KEY
from app.ucsb_dining import fetch_menu

app = FastAPI(title="QuickBiteSB Backend", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "OPTIONS"],
    allow_headers=["*"],
)

# In-memory cache: key = (hall_slug, date_str), value = (payload, expiry_ts)
_menu_cache: dict[tuple[str, str], tuple[dict[str, Any], float]] = {}

_occupancy_worker = None

def _cache_get(hall: str, date_str: str) -> dict[str, Any] | None:
    key = (hall, date_str)
    if key not in _menu_cache:
        return None
    payload, expiry = _menu_cache[key]
    if expiry <= datetime.now(timezone.utc).timestamp():
        del _menu_cache[key]
        return None
    return payload


def _cache_set(hall: str, date_str: str, payload: dict[str, Any]) -> None:
    import time
    expiry = time.time() + MENU_CACHE_TTL_SECONDS
    _menu_cache[(hall, date_str)] = (payload, expiry)


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok"}


ALLOWED_HALL_SLUGS = {"carrillo", "de-la-guerra", "ortega", "portola"}


@app.on_event("startup")
async def _startup() -> None:
    # Fire-and-forget occupancy worker; it keeps counting in the background.
    global _occupancy_worker  # noqa: PLW0603
    try:
        _occupancy_worker = await start_occupancy_worker()
    except Exception as exc:  # noqa: BLE001
        # Do not prevent backend from starting if the worker fails.
        print(f"[startup] Failed to start occupancy worker: {exc}")


@app.get("/v1/menus/{hall}")
async def get_menus(
    hall: str,
    date: str | None = Query(None, description="YYYY-MM-DD (default: today)"),
    meal: str | None = Query(None, description="Filter by meal name (optional)"),
    include_raw: bool = Query(False, description="Include raw UCSB response for debugging"),
) -> dict[str, Any]:
    """Return menu for a dining hall. Optional date and meal filter."""
    hall_slug = hall.lower().strip()
    if hall_slug not in ALLOWED_HALL_SLUGS:
        raise HTTPException(status_code=404, detail=f"Unknown dining hall: {hall}")
    date_str = date
    if not date_str:
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    cached = _cache_get(hall_slug, date_str)
    if cached is not None:
        out = dict(cached)
        if not include_raw:
            out.pop("raw", None)
        if meal:
            out["meals"] = [m for m in out.get("meals", []) if (m.get("name") or "").lower() == meal.lower()]
        return out

    payload, err = await fetch_menu(hall_slug, date_str)
    if err is not None:
        raise HTTPException(status_code=502, detail=err)

    _cache_set(hall_slug, date_str, payload)
    out = dict(payload)
    if not include_raw:
        out.pop("raw", None)
    if meal:
        out["meals"] = [m for m in out.get("meals", []) if (m.get("name") or "").lower() == meal.lower()]
    return out


def _serialize_occupancy(snapshot: HallOccupancy) -> Dict[str, Any]:
    percent_full = 0.0
    if snapshot.max_capacity > 0:
        percent_full = max(min(snapshot.net_occupancy / snapshot.max_capacity * 100.0, 100.0), 0.0)
    return {
        "hall": snapshot.hall,
        "timestamp": snapshot.timestamp,
        "total_entered": snapshot.total_entered,
        "total_exited": snapshot.total_exited,
        "net_occupancy": snapshot.net_occupancy,
        "max_capacity": snapshot.max_capacity,
        "percent_full": percent_full,
    }


@app.get("/v1/occupancy/{hall}")
async def get_occupancy(hall: str) -> Dict[str, Any]:
    """Return latest occupancy snapshot for a dining hall."""
    hall_slug = hall.lower().strip()
    if hall_slug not in ALLOWED_HALL_SLUGS:
        raise HTTPException(status_code=404, detail=f"Unknown dining hall: {hall}")

    snapshot = load_occupancy(hall_slug)
    if snapshot is None:
        raise HTTPException(status_code=404, detail="No occupancy data available yet")
    return _serialize_occupancy(snapshot)


@app.get("/v1/occupancy")
async def list_occupancy() -> List[Dict[str, Any]]:
    """Return latest occupancy snapshot for all halls that have data."""
    results: List[Dict[str, Any]] = []
    for hall in sorted(ALLOWED_HALL_SLUGS):
        snap = load_occupancy(hall)
        if snap is not None:
            results.append(_serialize_occupancy(snap))
    return results
