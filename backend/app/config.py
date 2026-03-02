"""Backend configuration from environment."""

import os
from pathlib import Path

from dotenv import load_dotenv


def _load_backend_env() -> None:
    """
    Load environment variables from backend/.env if present.

    Supports starting the app from the repo root or from the backend directory.
    """
    # backend/app/config.py -> backend/.env
    backend_root = Path(__file__).resolve().parent.parent
    env_path = backend_root / ".env"
    # load_dotenv is safe to call even if file is missing
    load_dotenv(env_path)


_load_backend_env()


# UCSB API: prefer CONSUMER_KEY (matches model_training) or UCSB_API_KEY
def _get_api_key() -> str | None:
    return os.getenv("CONSUMER_KEY") or os.getenv("UCSB_API_KEY")


UCSB_API_KEY: str | None = _get_api_key()
UCSB_BASE_URL: str = os.getenv("UCSB_BASE_URL", "https://api.ucsb.edu")
MENU_CACHE_TTL_SECONDS: int = int(os.getenv("MENU_CACHE_TTL_SECONDS", "300"))  # 5 min
