"""Small script to hit the FastAPI backend menu endpoint.

Useful to confirm that the backend is able to talk to UCSB and
that normalization works, independently of the Next.js frontend.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

import httpx


def main() -> None:
    backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
    hall = sys.argv[1] if len(sys.argv) > 1 else "carrillo"
    date = sys.argv[2] if len(sys.argv) > 2 else datetime.now().strftime("%Y-%m-%d")

    url = f"{backend_url.rstrip('/')}/v1/menus/{hall}"
    params = {"date": date}

    print(f"Requesting: {url} with params {params}")
    try:
        resp = httpx.get(url, params=params, timeout=15.0)
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

