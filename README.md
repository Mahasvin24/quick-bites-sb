# QuickBiteSB

Live dining commons snapshot for UCSB: occupancy view and today’s menu per hall.

## What’s in this repo

- **Frontend** — Next.js app: dining hall selector, occupancy “water tank” display, and today’s menu.
- **Backend** — FastAPI service that proxies the [UCSB Dining Menu API](https://developer.ucsb.edu/apis/dining/dining-menu) and optionally caches responses.
- **Model training / scripts** — Optional Python tools: fetch dining cam stills, Carrillo entry/exit counter (YOLOv8), and image utilities. These are separate from the web stack; the UI currently shows a placeholder occupancy value.

## Prerequisites

- **Node.js** (v18+) and **npm** — for the frontend.
- **Python 3.10+** — for the backend and model scripts.
- **UCSB API key** — [Get a Consumer Key](https://developer.ucsb.edu/docs/security/api-key) from the UCSB Developer Portal (required for menus and, if you use them, cam images).

## Quick start: run the full app

You need the **backend** and **frontend** running. Use two terminals.

### 1. Backend (menu API proxy)

From the **repository root**:

```bash
# Optional: virtualenv
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r backend/requirements.txt
```

Create `backend/.env` with your UCSB API key:

```bash
# backend/.env
CONSUMER_KEY=your-ucsb-consumer-key
```

Start the API:

```bash
uvicorn backend.app.main:app --reload --port 8000
```

- API base: **http://localhost:8000**
- Health: **GET** `http://localhost:8000/healthz`
- Menus: **GET** `http://localhost:8000/v1/menus/{hall}` (e.g. `carrillo`, `de-la-guerra`, `ortega`, `portola`)

### 2. Frontend

In a **second terminal**, from the repo root:

```bash
cd frontend
npm install
npm run dev
```

Open **http://localhost:3000**. The app will use the backend at `http://localhost:8000` by default.

To use a different backend URL:

```bash
export NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
npm run dev
```

---

## Environment reference

### Backend (`backend/.env`)

| Variable | Required | Description |
|----------|----------|-------------|
| `CONSUMER_KEY` or `UCSB_API_KEY` | Yes (for menus) | UCSB API Consumer Key from the developer portal. |
| `UCSB_BASE_URL` | No | Base URL for UCSB API; default `https://api.ucsb.edu`. |
| `MENU_CACHE_TTL_SECONDS` | No | Menu cache TTL in seconds; default `300` (5 minutes). |

### Frontend

| Variable | Required | Description |
|----------|----------|-------------|
| `NEXT_PUBLIC_BACKEND_URL` | No | Backend base URL; default `http://localhost:8000`. |

---

## Project layout

| Path | Description |
|------|-------------|
| `frontend/` | Next.js app (App Router), React, Tailwind, shadcn-style UI. Dining hall selector, water-tank occupancy widget, menu panel. |
| `backend/` | FastAPI app: UCSB Dining Menu proxy, in-memory cache, CORS for local frontend. |
| `model_training/` | Optional Python scripts: fetch cam stills, Carrillo entry/exit counter (YOLOv8), occupancy-related utilities. Not required to run the web app. |
| `outline-gen/` | Optional image outline utility (e.g. for preprocessing). |

---

## Optional: model training and cam images

These steps are **not** required to run the main app. They are for fetching dining cam stills and running the Carrillo occupancy counter.

### Shared setup

- **UCSB API key**: same Consumer Key as the backend. Put it in a `.env` in the directory from which you run the scripts (e.g. `model_training/` or repo root), as `CONSUMER_KEY=...`.
- **Python deps** (from repo root or `model_training/`):

```bash
pip install -r model_training/requirements.txt
```

### Fetch dining cam stills

`update-images.py` fetches current (and previous) stills from the UCSB dining cams and saves them under `model_training/images/` (e.g. `carrillo_current.jpg`, `carrillo_previous.jpg`). Run from **model_training/** so that `images/` is created there:

```bash
cd model_training
# .env in model_training/ or repo root with CONSUMER_KEY=
python3 update-images.py
```

Runs continuously (Ctrl+C to stop). Uses the same key for `https://api.ucsb.edu/dining/cams/v2/still/{hall}`.

### Carrillo entry/exit counter (YOLOv8)

- **Continuous mode** (reads `model_training/images/carrillo_previous.jpg` and `carrillo_current.jpg`, writes annotated frames to `model_training/output/carrillo/`):

  From **model_training/**:

  ```bash
  python3 carrillo_counter.py
  ```

  Optional: `python3 carrillo_counter.py --debug` for extra keypoint/direction debug output.

- **One-off test** (same inputs/outputs, single pass):

  From repo root or **model_training/**:

  ```bash
  python3 model_training/carrillo_test.py
  ```

The counter uses YOLOv8 detection + pose; it downloads model weights on first run. Occupancy is derived from entry/exit counts; the web UI does not read this yet (it shows a fixed placeholder).

### Other scripts

- `model_training/counter.py` — Intended to run a Carrillo counter loop that reads images from disk and prints occupancy; it depends on a `CarrilloCounter` API that may not match the current `carrillo_counter` module (use `carrillo_counter.py` / `carrillo_test.py` as above).
- `outline-gen/script.py` — Standalone script to generate outlines from a pre-processed image (e.g. `pre-carillo.png` → `carillo.png`). Run from `outline-gen/` with the expected input file.

---

## Tech stack

- **Frontend**: Next.js (App Router), React, Tailwind CSS, shadcn-style components.
- **Backend**: FastAPI (Python), httpx, in-memory menu cache.
- **Model scripts**: OpenCV, Ultralytics YOLOv8, NumPy; cam fetcher uses `requests` and `python-dotenv`.

---

## Troubleshooting

- **Menus don’t load / 502 from backend**  
  Ensure `CONSUMER_KEY` or `UCSB_API_KEY` is set in `backend/.env` and that the key is valid in the [UCSB Developer Portal](https://developer.ucsb.edu/docs/security/api-key).

- **Frontend can’t reach the API**  
  Confirm the backend is running on port 8000 and that `NEXT_PUBLIC_BACKEND_URL` (if set) matches it. Restart the Next dev server after changing env vars.

- **Carrillo counter “Missing: …”**  
  Run `update-images.py` from `model_training/` first so that `images/carrillo_previous.jpg` and `images/carrillo_current.jpg` exist.

- **Occupancy always 72%**  
  The UI uses a fixed placeholder. Real occupancy would require wiring the frontend to an endpoint that serves data from the model_training pipeline (not implemented in this repo yet).
