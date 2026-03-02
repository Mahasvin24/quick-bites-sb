"""FastAPI app: UCSB dining menu proxy and health."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .config import MENU_CACHE_TTL_SECONDS, UCSB_API_KEY
from .ucsb_dining import fetch_menu

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
