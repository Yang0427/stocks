import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// During development, proxy /api calls to the FastAPI backend on :8000 so the
// React app and the Python brain run side by side.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
});
