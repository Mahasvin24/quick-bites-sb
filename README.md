# QuickBiteSB

Live dining commons snapshot for UCSB: occupancy view and today’s menu per hall.

## Stack

- **Frontend**: Next.js (App Router), React, Tailwind CSS, shadcn-style components
- **Backend**: FastAPI (Python) — proxies [UCSB Dining Menu API](https://developer.ucsb.edu/apis/dining/dining-menu)

## Local development

### 1. Backend (menu API proxy)

From the repo root:

```bash
# Optional: use a virtualenv
python3 -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows

pip install -r backend/requirements.txt
export CONSUMER_KEY="your-ucsb-developer-consumer-key"
uvicorn backend.app.main:app --reload --port 8000
```

- **CONSUMER_KEY** (or **UCSB_API_KEY**): Your [UCSB API Consumer Key](https://developer.ucsb.edu/docs/security/api-key) from the developer portal (required for menu data).
- **UCSB_BASE_URL**: Optional; default `https://api.ucsb.edu`.
- **MENU_CACHE_TTL_SECONDS**: Optional; default `300` (5 minutes).

API: `http://localhost:8000` (health: `GET /healthz`, menus: `GET /v1/menus/{hall}`).

### 2. Frontend

```bash
cd frontend
npm install
npm run dev
```

Set the backend URL if it’s not on the default port:

```bash
export NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
npm run dev
```

Open [http://localhost:3000](http://localhost:3000).

## Project layout

- `frontend/` — Next.js app (dining hall selector, water-tank occupancy, menu panel)
- `backend/` — FastAPI app (UCSB Dining Menu proxy, in-memory cache)
- `model_training/` — occupancy/counter and image scripts (separate from the web stack)
