"""UCSB Dining Menu API client and response normalization."""
from __future__ import annotations

import json
from typing import Any

import httpx

from .config import UCSB_API_KEY, UCSB_BASE_URL

# From UCSB docs: https://api.ucsb.edu/dining/menu/v1/{date-time}/{dining-common-code}
MENU_PATH_TEMPLATE = "/dining/menu/v1/{date_time}/{dining_common_code}"
REQUEST_TIMEOUT = 15.0


def _normalize_menu(raw: Any) -> dict[str, Any]:
    """Normalize UCSB menu response into a consistent shape for the frontend."""
    if not isinstance(raw, dict):
        return {"hall": "", "date": "", "meals": [], "raw": raw}

    meals_out = []
    # Common shapes: array of meal periods, or nested under a key like "menu" / "meals"
    candidates = (
        raw.get("meals")
        or raw.get("menu")
        or (raw if isinstance(raw.get("items"), list) else None)
    )
    if isinstance(candidates, list):
        for m in candidates:
            if not isinstance(m, dict):
                continue
            name = m.get("name") or m.get("mealName") or m.get("period") or "Meal"
            stations = m.get("stations") or m.get("station") or []
            if not isinstance(stations, list):
                stations = [stations] if stations else []
            items = []
            station_names = set()
            for st in stations:
                if isinstance(st, dict):
                    station_name = st.get("name") or st.get("stationName") or "Station"
                    station_names.add(station_name)
                    station_items = st.get("items") or st.get("menuItems") or []
                    if isinstance(station_items, list):
                        for it in station_items:
                            if isinstance(it, str):
                                items.append({"name": it, "station": station_name})
                            elif isinstance(it, dict):
                                items.append({
                                    "name": it.get("name") or it.get("itemName") or str(it),
                                    "station": station_name,
                                })
                elif isinstance(st, str):
                    station_names.add("General")
                    items.append({"name": st, "station": "General"})
            meals_out.append({
                "name": name,
                "stations": sorted(station_names),
                "items": items,
            })
    # Flatten: sometimes API returns a single list of items
    if not meals_out and isinstance(raw.get("items"), list):
        all_items = []
        for it in raw["items"]:
            name = it.get("name") or it.get("itemName") if isinstance(it, dict) else str(it)
            all_items.append({"name": name, "station": "General"})
        if all_items:
            meals_out.append({"name": "Menu", "stations": ["General"], "items": all_items})

    return {
        "hall": raw.get("diningCommons") or raw.get("hall") or raw.get("diningCommonCode") or "",
        "date": raw.get("date") or raw.get("serviceDate") or "",
        "meals": meals_out,
        "raw": raw,
    }


async def fetch_menu(hall_slug: str, date_str: str) -> tuple[dict[str, Any] | None, str | None]:
    """
    Fetch menu from UCSB API. Returns (normalized_dict, error_message).
    date_str should be YYYY-MM-DD.
    """
    if not UCSB_API_KEY:
        return None, "UCSB API key not configured (set CONSUMER_KEY or UCSB_API_KEY)"

    url = f"{UCSB_BASE_URL.rstrip('/')}{MENU_PATH_TEMPLATE.format(date_time=date_str, dining_common_code=hall_slug)}"
    headers = {
        "ucsb-api-key": UCSB_API_KEY,
        "ucsb-api-version": "1.0",
        "Accept": "application/json",
    }
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            resp = await client.get(url, headers=headers)
    except httpx.TimeoutException:
        return None, "UCSB API request timed out"
    except Exception as e:
        return None, f"UCSB API request failed: {e!s}"

    if resp.status_code == 401:
        return None, "Invalid or missing UCSB API key"
    if resp.status_code == 404:
        return None, "Menu not found for this hall or date"
    if resp.status_code >= 500:
        return None, f"UCSB API error: {resp.status_code}"
    if resp.status_code != 200:
        return None, f"UCSB API returned {resp.status_code}"

    try:
        raw = resp.json()
    except json.JSONDecodeError:
        return None, "Invalid JSON from UCSB API"

    normalized = _normalize_menu(raw)
    normalized["hall"] = normalized.get("hall") or hall_slug
    normalized["date"] = normalized.get("date") or date_str
    return normalized, None
