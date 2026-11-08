# Video analytics dashboard (frontend)

React 18 + TypeScript + Vite dashboard for the RTSP/YOLOv8 video analytics
pipeline. See the repo root [README.md](../README.md#dashboard) "Dashboard"
section for what this shows, how it talks to the FastAPI backend, and how
to run it (`npm install && npm run dev`).

Quick reference:

```bash
npm install
npm run dev         # Vite dev server on :5173, proxies /events + /ws to :8000
npx tsc --noEmit     # type-check
npm run build        # production build -> dist/
npm run preview      # serve the production build locally
```

- `src/types.ts` — TypeScript interfaces mirroring `src/analytics/schemas.py`'s `VisionEvent`.
- `src/App.tsx` — the dashboard: live event table, summary strip, Recharts bar chart, Detections panel.
- `vite.config.ts` — dev/preview proxy config (see comments).
- `nginx.conf` / `Dockerfile` — production image (`docker-compose.yml`'s `dashboard-ui` service).
