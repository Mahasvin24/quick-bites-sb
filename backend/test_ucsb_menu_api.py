"""Small script to hit the UCSB Dining Menu API directly.

Reads CONSUMER_KEY / UCSB_API_KEY from backend/.env and prints
status + a snippet of the JSON response, to help troubleshoot.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv


def load_env_from_backend() -> None:
    backend_root = Path(__file__).resolve().parent
    env_path = backend_root / ".env"
    load_dotenv(env_path)


def main() -> None:
    load_env_from_backend()

    api_key = os.getenv("CONSUMER_KEY") or os.getenv("UCSB_API_KEY")
    if not api_key:
        print("ERROR: Missing CONSUMER_KEY / UCSB_API_KEY in backend/.env")
        sys.exit(1)

    date = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y-%m-%d")
    hall = sys.argv[2] if len(sys.argv) > 2 else "carrillo"

    url = f"https://api.ucsb.edu/dining/menu/v1/{date}/{hall}"
    headers = {
        "ucsb-api-key": api_key,
        "ucsb-api-version": "1.0",
        "Accept": "application/json",
    }

    print(f"Requesting: {url}")
    try:
        resp = httpx.get(url, headers=headers, timeout=15.0)
    except Exception as e:  # noqa: BLE001
        print(f"Request error: {e}")
        sys.exit(1)

    print(f"Status: {resp.status_code}")
    print(f"Content-Type: {resp.headers.get('content-type')}")
    print("--- Body (first 4000 chars) ---")
    text = resp.text
    print(text[:4000])
    if len(text) > 4000:
        print("... (truncated) ...")


if __name__ == "__main__":
    main()

