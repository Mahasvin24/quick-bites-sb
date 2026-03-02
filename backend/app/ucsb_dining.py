"""UCSB Dining Menu API client and response normalization."""
from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx

from app.config import UCSB_API_KEY, UCSB_BASE_URL

# UCSB API: two endpoints:
# 1) GET /dining/menu/v1/{date}/{hall} -> list of {name, code} (meal periods only, no items)
# 2) GET /dining/menu/v1/{date}/{hall}/{meal_code} -> list of {name, station} (items for that meal)
MENU_LIST_PATH = "/dining/menu/v1/{date_time}/{dining_common_code}"
MENU_ITEMS_PATH = "/dining/menu/v1/{date_time}/{dining_common_code}/{meal_code}"
REQUEST_TIMEOUT = 15.0


def _parse_meal_items(raw_items: Any) -> list[dict[str, Any]]:
    """Parse a list of item dicts (each with name, station) into normalized items."""
    if not isinstance(raw_items, list):
        return []
    items = []
    for it in raw_items:
        if isinstance(it, dict):
            item_name = it.get("name") or it.get("itemName") or str(it)
            station = it.get("station") or it.get("stationName") or "General"
            items.append({"name": item_name, "station": station})
        elif isinstance(it, str):
            items.append({"name": it, "station": "General"})
    return items


def _build_meals_with_items(
    meal_list: list[dict[str, Any]],
    items_by_meal_code: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Build normalized meals list: each meal has name, stations, items."""
    meals_out = []
    for m in meal_list:
        if not isinstance(m, dict):
            continue
        meal_name = m.get("name") or m.get("mealName") or m.get("mealCode") or "Meal"
        meal_code = (m.get("code") or m.get("mealCode") or "").strip().lower()
        items = items_by_meal_code.get(meal_code) or []
        station_names = sorted({it.get("station") or "General" for it in items})
        meals_out.append({
            "name": meal_name,
            "stations": station_names,
            "items": items,
        })
    return meals_out


async def fetch_menu(hall_slug: str, date_str: str) -> tuple[dict[str, Any] | None, str | None]:
    """
    Fetch menu from UCSB API. Returns (normalized_dict, error_message).
    date_str should be YYYY-MM-DD.
    Uses two endpoints: meal list, then items per meal (API does not return items in the list endpoint).
    """
    if not UCSB_API_KEY:
        return None, "UCSB API key not configured (set CONSUMER_KEY or UCSB_API_KEY)"

    base = UCSB_BASE_URL.rstrip("/")
    headers = {
        "ucsb-api-key": UCSB_API_KEY,
        "ucsb-api-version": "1.0",
        "Accept": "application/json",
    }

    # Step 1: get meal periods (name, code) for this hall/date
    list_url = f"{base}{MENU_LIST_PATH.format(date_time=date_str, dining_common_code=hall_slug)}"
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            list_resp = await client.get(list_url, headers=headers)
    except httpx.TimeoutException:
        return None, "UCSB API request timed out"
    except Exception as e:
        return None, f"UCSB API request failed: {e!s}"

    if list_resp.status_code == 401:
        return None, "Invalid or missing UCSB API key"
    if list_resp.status_code == 404:
        return None, "Menu not found for this hall or date"
    if list_resp.status_code >= 500:
        return None, f"UCSB API server error: {list_resp.status_code}"
    if list_resp.status_code != 200:
        return None, f"UCSB API returned {list_resp.status_code}"

    try:
        meal_list = list_resp.json()
    except json.JSONDecodeError:
        return None, "Invalid JSON from UCSB API"

    if not isinstance(meal_list, list):
        meal_list = []

    # Step 2: fetch items for each meal (per-meal endpoint returns list of {name, station})
    async def fetch_meal_items(meal_code: str) -> tuple[str, list[dict[str, Any]]]:
        url = f"{base}{MENU_ITEMS_PATH.format(date_time=date_str, dining_common_code=hall_slug, meal_code=meal_code)}"
        try:
            async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
                r = await client.get(url, headers=headers)
        except Exception:
            return meal_code, []
        if r.status_code != 200:
            return meal_code, []
        try:
            raw = r.json()
        except json.JSONDecodeError:
            return meal_code, []
        return meal_code, _parse_meal_items(raw if isinstance(raw, list) else [])

    meal_codes = [
        (m.get("code") or m.get("mealCode") or "").strip().lower()
        for m in meal_list
        if isinstance(m, dict)
    ]
    results = await asyncio.gather(*[fetch_meal_items(code) for code in meal_codes if code])
    items_by_meal_code = dict(results)

    meals_out = _build_meals_with_items(meal_list, items_by_meal_code)
    normalized = {
        "hall": hall_slug,
        "date": date_str,
        "meals": meals_out,
        "raw": meal_list,
    }

    # #region agent log
    try:
        _log_path = "/Users/mahasvin/Github/quick-bites-sb/.cursor/debug-61549c.log"
        _summary = [{"name": m.get("name"), "items_count": len(m.get("items") or [])} for m in meals_out]
        with open(_log_path, "a") as _f:
            _f.write(json.dumps({"sessionId": "61549c", "hypothesisId": "H5", "location": "ucsb_dining.py:normalized", "message": "normalized_summary", "runId": "post-fix", "data": {"meals_count": len(meals_out), "meals_summary": _summary}, "timestamp": __import__("time").time() * 1000}) + "\n")
    except Exception:
        pass
    # #endregion
    return normalized, None