import path from 'node:path'
import { tmpdir } from 'node:os'
import { fileURLToPath } from 'node:url'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

// Keep Vite's cache outside `node_modules` (OneDrive / Windows often locks `.vite/deps`).
const cacheDir = path.join(tmpdir(), 'vite-cache-ML_Project2-frontend')

// https://vite.dev/config/
const apiProxy = {
  '/api': {
    target: 'http://127.0.0.1:8000',
    changeOrigin: true,
    rewrite: (path) => path.replace(/^\/api/, ''),
  },
}

export default defineConfig({
  cacheDir,
  plugins: [react()],
  server: {
    proxy: { ...apiProxy },
  },
  // `vite preview` does not use `server.proxy` unless mirrored here — without it, `/api/*` returns 404.
  preview: {
    proxy: { ...apiProxy },
  },
})
