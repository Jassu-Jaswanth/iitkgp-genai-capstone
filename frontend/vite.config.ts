import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

// `npm run dev`: the UI on :5173, /api proxied to the backend on :8100 (same paths as nginx in the container).
export default defineConfig(({ mode }) => {
  const api = loadEnv(mode, ".", "").API_URL || "http://localhost:8100";
  return {
    plugins: [react()],
    server: { proxy: { "/api": { target: api, rewrite: (p: string) => p.replace(/^\/api/, "") } } },
  };
});
