# QuickBiteSB Backend

FastAPI service that proxies the UCSB Dining Menu API and returns normalized menu data for the frontend.

## Endpoints

- **GET /healthz** — Liveness check.
- **GET /v1/menus/{hall}** — Menu for a dining hall.
  - Path: `hall` = `carrillo` | `de-la-guerra` | `ortega` | `portola`
  - Query: `date` (optional) = `YYYY-MM-DD` (default: today)
  - Query: `meal` (optional) = filter by meal name
  - Query: `include_raw=1` (optional) = include raw UCSB response in the payload

## Environment variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| **CONSUMER_KEY** or **UCSB_API_KEY** | Yes (for menu) | — | UCSB API Consumer Key from [developer.ucsb.edu](https://developer.ucsb.edu) |
| **UCSB_BASE_URL** | No | `https://api.ucsb.edu` | UCSB API base URL |
| **MENU_CACHE_TTL_SECONDS** | No | `300` | In-memory cache TTL for menu responses |

## Run locally

From the **repository root**:

```bash
pip install -r backend/requirements.txt
export CONSUMER_KEY="your-consumer-key"
uvicorn backend.app.main:app --reload --port 8000
```

From the **backend** directory:

```bash
pip install -r requirements.txt
export CONSUMER_KEY="your-consumer-key"
uvicorn app.main:app --reload --port 8000
```

CORS is set to allow `http://localhost:3000` and `http://127.0.0.1:3000` for the Next.js dev server.
