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
| **CONSUMER_KEY** or **UCSB_API_KEY** | Yes (for menu & cameras) | — | UCSB API Consumer Key from [developer.ucsb.edu](https://developer.ucsb.edu) |
| **UCSB_BASE_URL** | No | `https://api.ucsb.edu` | UCSB API base URL |
| **MENU_CACHE_TTL_SECONDS** | No | `300` | In-memory cache TTL for menu responses |
| **OCCUPANCY_POLL_INTERVAL_SECONDS** | No | `5` | Interval (seconds) between camera polls and occupancy updates |
| **OCCUPANCY_DATA_ROOT** | No | `.` (current working directory) | Root directory under which `data/occupancy` and `data/images` are created |

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

The backend also runs a background occupancy worker which:

- Periodically fetches dining hall camera stills.
- Uses YOLOv8 to count people entering/exiting each hall.
- Persists per-hall occupancy snapshots under `data/occupancy/{hall}.json`.
- Writes the latest annotated frame (with entering/exiting overlays) per hall to `data/annotated/{hall}.jpg`.

You can query the latest occupancy via:

- `GET /v1/occupancy/{hall}` — Latest snapshot for a single hall.
- `GET /v1/occupancy` — Latest snapshots for all halls with data.

Each occupancy snapshot includes:

- `hall`: hall slug (`carrillo`, `de-la-guerra`, `ortega`, `portola`)
- `timestamp`: ISO 8601 UTC string
- `total_entered`, `total_exited`: cumulative counts (non-decreasing)
- `net_occupancy`: current in-hall count, clamped at a minimum of 0
- `max_capacity`: configured maximum capacity for that hall
- `percent_full`: `0–100` percentage, derived from `net_occupancy` / `max_capacity`
