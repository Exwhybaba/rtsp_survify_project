import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// During `npm run dev` the app is served from :5173 while the FastAPI
// backend runs separately on :8000 (see README "Dashboard" section). This
// proxy makes the frontend's relative fetch("/events")/WebSocket("/ws/live")
// calls transparently reach the backend without CORS config or hardcoding a
// backend origin into the app code. Production (docker-compose) achieves the
// same relative-path behaviour via an nginx reverse proxy instead — see
// frontend/nginx.conf. BACKEND_ORIGIN is overridable via env (e.g. by
// scripts/capture_dashboard_screenshot.py, which runs the backend on a
// non-default port to avoid colliding with other local services).
// Use IPv4 explicitly on Windows. Some systems resolve localhost to ::1
// while the local FastAPI process is listening on IPv4.
const BACKEND_ORIGIN = process.env.BACKEND_ORIGIN || "http://127.0.0.1:8000";

const proxyConfig = {
  "/events": BACKEND_ORIGIN,
  "/api": BACKEND_ORIGIN,
  "/health": BACKEND_ORIGIN,
  "/ws": {
    target: BACKEND_ORIGIN.replace("http", "ws"),
    ws: true,
  },
};

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: proxyConfig,
  },
  // `vite preview` (serving the built dist/, e.g. for the screenshot
  // harness in scripts/capture_dashboard_screenshot.py) gets the same
  // proxying as `vite dev` so the production-style bundle can still reach
  // the backend without a hardcoded origin.
  preview: {
    proxy: proxyConfig,
  },
});
