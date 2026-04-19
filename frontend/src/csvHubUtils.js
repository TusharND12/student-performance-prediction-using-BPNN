/**
 * CSV Hub helpers — all computed client-side from predictionHistory.
 * No backend changes required; each CSV-imported row in predictionHistory already has
 * `formData`, `numeric`, `passProb`, `csvMeta.{displayName, displayId, fileName, ...}`.
 */

export const CSV_CLASSES_KEY = 'edupredict-csv-classes';
export const CSV_AUDIT_KEY = 'edupredict-csv-audit';

/** Read/write classes (saved cohorts) */
export function readClasses() {
  try {
    const raw = localStorage.getItem(CSV_CLASSES_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}
export function writeClasses(classes) {
  try {
    localStorage.setItem(CSV_CLASSES_KEY, JSON.stringify(classes));
  } catch { /* ignore quota */ }
}

/** Audit log entries */
export function readAudit() {
  try {
    const raw = localStorage.getItem(CSV_AUDIT_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}
export function appendAudit(entry) {
  const list = readAudit();
  list.unshift(entry);
  try {
    localStorage.setItem(CSV_AUDIT_KEY, JSON.stringify(list.slice(0, 500)));
  } catch { /* ignore quota */ }
}
export function clearAudit() {
  try {
    localStorage.removeItem(CSV_AUDIT_KEY);
  } catch { /* ignore */ }
}

/** Group CSV-imported rows by source file name. */
export function groupByFile(rows) {
  const groups = new Map();
  for (const r of rows) {
    if (r.source !== 'csv') continue;
    const name = r.csvMeta?.fileName || 'unknown.csv';
    if (!groups.has(name)) groups.set(name, []);
    groups.get(name).push(r);
  }
  return groups;
}

/** Key numeric features the dashboard cares about (must match training names). */
export const KEY_FEATURES = [
  'attendance_rate',
  'study_hours_daily',
  'sleep_hours',
  'mental_stress',
  'previous_gpa',
  'assignment_avg',
  'quiz_avg',
  'project_score',
  'standardized_exam_score',
  'math_score',
  'science_score',
  'english_score',
  'history_score',
  'computer_score',
];

/** Pretty label. */
export function prettyFeatureName(key) {
  return key.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}

/** Detect out-of-range ("impossible") values. */
const RANGES = {
  attendance_rate: { min: 0, max: 1.0, soft: true },
  sleep_hours: { min: 0, max: 24 },
  study_hours_daily: { min: 0, max: 24 },
  mental_stress: { min: 0, max: 10 },
  previous_gpa: { min: 0, max: 4.0 },
  math_score: { min: 0, max: 100 },
  science_score: { min: 0, max: 100 },
  english_score: { min: 0, max: 100 },
  history_score: { min: 0, max: 100 },
  computer_score: { min: 0, max: 100 },
  assignment_avg: { min: 0, max: 100 },
  quiz_avg: { min: 0, max: 100 },
  project_score: { min: 0, max: 100 },
  standardized_exam_score: { min: 0, max: 100 },
};

export function rowFlags(row) {
  const flags = [];
  const n = row.numeric || {};
  for (const [k, r] of Object.entries(RANGES)) {
    const v = n[k];
    if (typeof v !== 'number' || !Number.isFinite(v)) continue;
    if (v < r.min || v > r.max) flags.push(`${k} out of range (${v})`);
  }
  return flags;
}

/** Provided vs imputed: a feature is "provided" when formData[key] is a non-blank non-zero numeric string. */
export function featureProvided(row, key) {
  const raw = row.formData?.[key];
  if (raw == null) return false;
  const s = String(raw).trim();
  if (s === '' || s === '-' || s === '.') return false;
  if (key === 'parent_involvement') return true;
  const n = parseFloat(s);
  return Number.isFinite(n) && n !== 0;
}

/** Overall import quality score (0-100) — higher is better. */
export function qualityScore(rows) {
  if (!rows.length) return { score: 0, components: [] };
  const cov = rows.reduce((s, r) => s + KEY_FEATURES.filter((k) => featureProvided(r, k)).length, 0)
    / (rows.length * KEY_FEATURES.length);
  const outRows = rows.filter((r) => rowFlags(r).length > 0).length / rows.length;
  const ids = rows
    .map((r) => (r.csvMeta?.displayId || '').trim())
    .filter(Boolean);
  const dupRate = ids.length ? 1 - new Set(ids).size / ids.length : 0;
  const labelRate = rows.filter((r) => r.csvMeta?.fileExpectPass != null).length / rows.length;

  const components = [
    { key: 'coverage', label: 'Feature coverage', value: Math.round(cov * 100) },
    { key: 'validity', label: 'Valid rows (no outliers)', value: Math.round((1 - outRows) * 100) },
    { key: 'uniqueness', label: 'ID uniqueness', value: Math.round((1 - dupRate) * 100) },
    { key: 'labels', label: 'Ground-truth labels', value: Math.round(labelRate * 100) },
  ];
  // Weighted combine: coverage 40, validity 25, uniqueness 20, labels 15
  const score = Math.round(
    cov * 40 + (1 - outRows) * 25 + (1 - dupRate) * 20 + labelRate * 15,
  );
  return { score: Math.max(0, Math.min(100, score)), components };
}

/** Confusion matrix + metrics given rows with .passProb and .csvMeta.fileExpectPass. */
export function confusionMatrix(rows, threshold) {
  let TP = 0;
  let FP = 0;
  let TN = 0;
  let FN = 0;
  for (const r of rows) {
    const truth = r.csvMeta?.fileExpectPass;
    if (truth == null) continue;
    const p = typeof r.passProb === 'number' ? r.passProb : null;
    if (p == null) continue;
    const pred = p >= threshold;
    if (pred && truth) TP += 1;
    else if (pred && !truth) FP += 1;
    else if (!pred && !truth) TN += 1;
    else if (!pred && truth) FN += 1;
  }
  const n = TP + FP + TN + FN;
  if (n === 0) return null;
  const acc = (TP + TN) / n;
  const prec = TP + FP === 0 ? 0 : TP / (TP + FP);
  const rec = TP + FN === 0 ? 0 : TP / (TP + FN);
  const f1 = prec + rec === 0 ? 0 : (2 * prec * rec) / (prec + rec);
  const bal = (TP / Math.max(1, TP + FN) + TN / Math.max(1, TN + FP)) / 2;
  return { TP, FP, TN, FN, n, acc, prec, rec, f1, bal };
}

/** Scan thresholds to find the one with best match rate (= accuracy on labeled rows). */
export function suggestBestThreshold(rows) {
  const labeled = rows.filter(
    (r) => r.csvMeta?.fileExpectPass != null && typeof r.passProb === 'number',
  );
  if (!labeled.length) return null;
  let best = { t: 0.5, acc: 0 };
  for (let t = 0.02; t <= 0.98; t += 0.01) {
    let tp = 0;
    for (const r of labeled) {
      const pred = r.passProb >= t;
      if (pred === r.csvMeta.fileExpectPass) tp += 1;
    }
    const acc = tp / labeled.length;
    if (acc > best.acc) best = { t: +t.toFixed(2), acc };
  }
  return best;
}

/** Simple k-means on a chosen feature subset (Lloyd's, k random seeds). */
export function kmeans(rows, k = 3, features = ['attendance_rate', 'study_hours_daily', 'sleep_hours', 'mental_stress']) {
  const pts = [];
  const indices = [];
  rows.forEach((r, i) => {
    const v = features.map((f) => (typeof r.numeric?.[f] === 'number' ? r.numeric[f] : NaN));
    if (v.every((x) => Number.isFinite(x))) {
      pts.push(v);
      indices.push(i);
    }
  });
  if (pts.length < k) return null;
  // Min-max scale
  const dim = features.length;
  const mins = Array(dim).fill(Infinity);
  const maxs = Array(dim).fill(-Infinity);
  for (const p of pts) {
    p.forEach((v, d) => {
      if (v < mins[d]) mins[d] = v;
      if (v > maxs[d]) maxs[d] = v;
    });
  }
  const norm = pts.map((p) => p.map((v, d) => (maxs[d] === mins[d] ? 0 : (v - mins[d]) / (maxs[d] - mins[d]))));

  // Seed with deterministic evenly spaced choices
  const centroids = [];
  for (let i = 0; i < k; i += 1) {
    centroids.push([...norm[Math.floor((i * norm.length) / k)]]);
  }
  let labels = new Array(norm.length).fill(0);
  for (let iter = 0; iter < 20; iter += 1) {
    // Assign
    const newLabels = norm.map((p) => {
      let best = 0;
      let bestDist = Infinity;
      for (let c = 0; c < k; c += 1) {
        const d = p.reduce((s, v, di) => s + (v - centroids[c][di]) ** 2, 0);
        if (d < bestDist) {
          bestDist = d;
          best = c;
        }
      }
      return best;
    });
    // Recompute
    const sums = Array.from({ length: k }, () => Array(dim).fill(0));
    const counts = Array(k).fill(0);
    newLabels.forEach((lab, i) => {
      counts[lab] += 1;
      norm[i].forEach((v, d) => { sums[lab][d] += v; });
    });
    const nextCentroids = sums.map((s, i) => (counts[i] ? s.map((v) => v / counts[i]) : centroids[i]));
    let stable = true;
    for (let c = 0; c < k; c += 1) {
      for (let d = 0; d < dim; d += 1) {
        if (Math.abs(nextCentroids[c][d] - centroids[c][d]) > 1e-4) stable = false;
        centroids[c][d] = nextCentroids[c][d];
      }
    }
    labels = newLabels;
    if (stable) break;
  }
  // Label each cluster with a qualitative name based on centroid feature mix
  const clusters = Array.from({ length: k }, (_, c) => {
    const members = indices.filter((_, i) => labels[i] === c).map((i) => rows[i]);
    const avg = {};
    features.forEach((f, d) => { avg[f] = mins[d] + centroids[c][d] * (maxs[d] - mins[d]); });
    return { c, members, centroid: avg };
  });
  // Name clusters by simple heuristic
  const named = clusters.slice().sort((a, b) => (b.centroid.attendance_rate ?? 0) - (a.centroid.attendance_rate ?? 0));
  const names = ['Strivers', 'Mid pack', 'At risk', 'Irregulars', 'Quiet'];
  named.forEach((cl, i) => { cl.name = names[i] || `Cluster ${i + 1}`; });
  return { clusters, features, indices, labels };
}

/** Convert array of objects to CSV text (utf8 with BOM). */
export function toCsv(rows, columns) {
  const header = columns.map((c) => c.label).join(',');
  const lines = rows.map((r) =>
    columns
      .map((c) => {
        const val = c.get(r);
        if (val == null) return '';
        const s = String(val);
        return /[,"\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
      })
      .join(','),
  );
  return `\uFEFF${[header, ...lines].join('\r\n')}`;
}

export function downloadBlob(name, mime, content) {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 200);
}

/** Hash a blob of text for file fingerprinting (simple, not crypto). */
export function simpleHash(text) {
  let h = 0;
  for (let i = 0; i < text.length; i += 1) {
    h = ((h << 5) - h) + text.charCodeAt(i);
    h |= 0;
  }
  return (h >>> 0).toString(16).padStart(8, '0');
}
