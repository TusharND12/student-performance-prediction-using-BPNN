/**
 * Resolve API URLs for dev (Vite `/api` proxy), preview, and production.
 * If `VITE_API_URL` ends with `/api` (common mistake copying the dev prefix), strip it so
 * paths like `/import/csv` hit FastAPI instead of `/api/import/csv` (404).
 */
export function backendOrigin() {
  const env = import.meta.env.VITE_API_URL;
  if (!env) return null;
  let b = String(env).trim().replace(/\/$/, '');
  if (b.endsWith('/api')) {
    b = b.slice(0, -4).replace(/\/$/, '');
  }
  return b || null;
}

/** @param {string} path e.g. `predict`, `meta/version`, `labs/sensitivity` */
export function apiUrl(path) {
  const p = String(path || '').replace(/^\//, '');
  const origin = backendOrigin();
  if (origin) return `${origin}/${p}`;
  return `/api/${p}`;
}
