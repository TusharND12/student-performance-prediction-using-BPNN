/**
 * Session PDF via FastAPI + ReportLab (dashboard-themed multipage report).
 * Dev: Vite proxies `/api/*` → Uvicorn. Prod: `VITE_API_URL` origin + `/report/pdf`.
 */
import axios from 'axios';
import { apiUrl } from './apiRoutes.js';

function reportPdfUrl() {
  return apiUrl('report/pdf');
}

async function parseErrorFromBlob(blob) {
  try {
    const text = await blob.text();
    const j = JSON.parse(text);
    if (typeof j.detail === 'string') return j.detail;
    if (Array.isArray(j.detail)) {
      return j.detail.map((d) => (typeof d === 'string' ? d : d.msg || JSON.stringify(d))).join('; ');
    }
    return text || 'PDF request failed';
  } catch {
    return 'PDF request failed';
  }
}

/**
 * @param {object} opts
 * @param {Array} opts.predictionHistory
 * @param {object|null} opts.improvementSnapshot
 * @param {object} opts.improvementPlan — from buildImprovementPlan()
 * @param {boolean} opts.darkMode
 */
export async function downloadEduPredictPdf({
  predictionHistory,
  improvementSnapshot,
  improvementPlan,
  darkMode,
}) {
  const res = await axios.post(
    reportPdfUrl(),
    {
      generated_at: new Date().toLocaleString(),
      dark_mode: Boolean(darkMode),
      predictions: predictionHistory,
      improvement_plan: improvementPlan,
      improvement_snapshot: improvementSnapshot,
    },
    { responseType: 'blob', validateStatus: () => true },
  );

  const ct = (res.headers['content-type'] || '').toLowerCase();
  if (res.status !== 200 || !ct.includes('application/pdf')) {
    const msg =
      res.data instanceof Blob ? await parseErrorFromBlob(res.data) : `HTTP ${res.status}`;
    throw new Error(msg);
  }

  const blob = res.data;
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `edupredict-session-report-${Date.now()}.pdf`;
  a.rel = 'noopener';
  a.click();
  URL.revokeObjectURL(url);
}
