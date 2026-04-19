/**
 * CSV Hub — post-import workspace with 10 tabs:
 * Overview · Preview · Quality · Heatmap · Confusion · At-risk · Scatter · Archetypes · Compare · Export
 * Uses predictionHistory (CSV-sourced rows) + localStorage for saved classes + audit log.
 */
import React, { useMemo, useState, useEffect } from 'react';
import {
  Layers,
  FileSpreadsheet,
  Sparkles,
  AlertTriangle,
  CheckCircle2,
  Target,
  Users,
  Download,
  History,
  Grid3x3,
  Gauge,
  Activity,
  PieChart,
} from 'lucide-react';
import {
  KEY_FEATURES,
  prettyFeatureName,
  featureProvided,
  rowFlags,
  qualityScore,
  confusionMatrix,
  suggestBestThreshold,
  kmeans,
  groupByFile,
  readClasses,
  writeClasses,
  readAudit,
  clearAudit,
  toCsv,
  downloadBlob,
} from './csvHubUtils.js';

function TabBtn({ active, onClick, icon, label }) {
  const TabIcon = icon;
  return (
    <button
      type="button"
      className={`csvhub-tab ${active ? 'csvhub-tab--active' : ''}`}
      onClick={onClick}
    >
      <TabIcon size={14} strokeWidth={1.9} />
      {label}
    </button>
  );
}

function EmptyState({ children }) {
  return <div className="csvhub-empty">{children}</div>;
}

/* ── Overview ───────────────────────────────────────── */
function OverviewPanel({ rows, threshold }) {
  const n = rows.length;
  const passN = rows.filter((r) => (r.passProb ?? 0) >= threshold).length;
  const failN = n - passN;
  const avgProb = n ? rows.reduce((s, r) => s + (r.passProb ?? 0), 0) / n : 0;
  const cm = confusionMatrix(rows, threshold);
  return (
    <div className="csvhub-grid csvhub-grid--4">
      <div className="csvhub-card">
        <span className="csvhub-card-label">Students imported</span>
        <strong className="csvhub-card-val">{n}</strong>
      </div>
      <div className="csvhub-card">
        <span className="csvhub-card-label">Predicted Pass</span>
        <strong className="csvhub-card-val csvhub-pass">{passN}</strong>
        <span className="csvhub-card-sub">{n ? `${Math.round((100 * passN) / n)}%` : '—'}</span>
      </div>
      <div className="csvhub-card">
        <span className="csvhub-card-label">Predicted Fail</span>
        <strong className="csvhub-card-val csvhub-fail">{failN}</strong>
        <span className="csvhub-card-sub">{n ? `${Math.round((100 * failN) / n)}%` : '—'}</span>
      </div>
      <div className="csvhub-card">
        <span className="csvhub-card-label">Avg P(Pass)</span>
        <strong className="csvhub-card-val">{n ? `${(avgProb * 100).toFixed(1)}%` : '—'}</strong>
        {cm ? (
          <span className="csvhub-card-sub">
            Match vs file: {Math.round(cm.acc * 100)}%
          </span>
        ) : null}
      </div>
    </div>
  );
}

/* ── Preview (first 20 rows, raw numeric + flags) ─────────────────────── */
function PreviewPanel({ rows }) {
  const sample = rows.slice(0, 20);
  if (!sample.length) return <EmptyState>No imported rows.</EmptyState>;
  return (
    <div className="csvhub-table-wrap">
      <table className="csvhub-table">
        <thead>
          <tr>
            <th>#</th>
            <th>Name</th>
            <th>ID</th>
            <th>P(Pass)</th>
            <th>Math</th>
            <th>Attendance</th>
            <th>Sleep</th>
            <th>Study</th>
            <th>Stress</th>
            <th>Issues</th>
          </tr>
        </thead>
        <tbody>
          {sample.map((r, i) => {
            const flags = rowFlags(r);
            return (
              <tr key={r.id}>
                <td className="tabnums">{r.csvMeta?.rowIndex ?? i + 1}</td>
                <td>{r.csvMeta?.displayName || '—'}</td>
                <td className="tabnums">{r.csvMeta?.displayId || '—'}</td>
                <td className="tabnums">
                  {typeof r.passProb === 'number' ? `${(r.passProb * 100).toFixed(1)}%` : '—'}
                </td>
                <td className="tabnums">{r.numeric?.math_score ?? '—'}</td>
                <td className="tabnums">{r.numeric?.attendance_rate ?? '—'}</td>
                <td className="tabnums">{r.numeric?.sleep_hours ?? '—'}</td>
                <td className="tabnums">{r.numeric?.study_hours_daily ?? '—'}</td>
                <td className="tabnums">{r.numeric?.mental_stress ?? '—'}</td>
                <td style={{ color: flags.length ? '#b91c1c' : 'var(--text-light)' }}>
                  {flags.length ? flags.join(' · ') : 'ok'}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/* ── Quality score ─────────────────────────── */
function QualityPanel({ rows }) {
  const { score, components } = qualityScore(rows);
  const dupFlag = components.find((c) => c.key === 'uniqueness');
  const outFlag = components.find((c) => c.key === 'validity');
  const covFlag = components.find((c) => c.key === 'coverage');
  return (
    <div className="csvhub-quality">
      <div className="csvhub-quality-score">
        <strong>{score}</strong>
        <span>/ 100</span>
      </div>
      <ul className="csvhub-quality-list">
        {components.map((c) => (
          <li key={c.key}>
            <span>{c.label}</span>
            <span className="csvhub-quality-bar">
              <span style={{ width: `${c.value}%` }} />
            </span>
            <span className="csvhub-quality-val">{c.value}%</span>
          </li>
        ))}
      </ul>
      <div className="csvhub-quality-tips">
        {covFlag && covFlag.value < 40 ? (
          <div><AlertTriangle size={14} /> Low coverage — add attendance, sleep, stress columns for better predictions.</div>
        ) : null}
        {outFlag && outFlag.value < 90 ? (
          <div><AlertTriangle size={14} /> Outliers detected — open the <strong>Preview</strong> tab to review.</div>
        ) : null}
        {dupFlag && dupFlag.value < 100 ? (
          <div><AlertTriangle size={14} /> Duplicate student IDs — deduplicate upstream or merge.</div>
        ) : null}
      </div>
    </div>
  );
}

/* ── Missing-data heatmap ─────────────────── */
function HeatmapPanel({ rows }) {
  const sample = rows.slice(0, 80);
  if (!sample.length) return <EmptyState>No imported rows.</EmptyState>;
  const cellH = 14;
  const cellW = 22;
  const featCount = KEY_FEATURES.length;
  const W = 180 + featCount * cellW;
  const H = sample.length * cellH + 80;
  return (
    <div className="csvhub-scroll">
      <svg viewBox={`0 0 ${W} ${H}`} className="csvhub-svg">
        {KEY_FEATURES.map((f, i) => (
          <text
            key={f}
            x={180 + i * cellW + cellW / 2}
            y={18}
            transform={`rotate(-35 ${180 + i * cellW + cellW / 2} 18)`}
            fontSize="9"
            fill="#64748b"
            textAnchor="end"
          >
            {prettyFeatureName(f)}
          </text>
        ))}
        {sample.map((r, rowIdx) => (
          <g key={r.id}>
            <text x={0} y={60 + rowIdx * cellH} fontSize="10" fill="#0f172a">
              {(r.csvMeta?.displayName || r.csvMeta?.displayId || `#${rowIdx + 1}`).slice(0, 22)}
            </text>
            {KEY_FEATURES.map((f, i) => (
              <rect
                key={f}
                x={180 + i * cellW + 1}
                y={52 + rowIdx * cellH}
                width={cellW - 2}
                height={cellH - 2}
                rx={2}
                fill={featureProvided(r, f) ? '#10b981' : 'rgba(148,163,184,0.25)'}
                opacity={featureProvided(r, f) ? 0.85 : 0.9}
              />
            ))}
          </g>
        ))}
      </svg>
      <div className="csvhub-legend">
        <span><i style={{ background: '#10b981' }} /> provided</span>
        <span><i style={{ background: 'rgba(148,163,184,0.6)' }} /> imputed (median)</span>
      </div>
    </div>
  );
}

/* ── Confusion matrix + suggested threshold ─────────── */
function ConfusionPanel({ rows, threshold, onApplyThreshold }) {
  const cm = confusionMatrix(rows, threshold);
  const best = suggestBestThreshold(rows);
  if (!cm) return <EmptyState>Include a <code>result</code> / <code>pass_fail</code> column in the CSV to see this.</EmptyState>;
  return (
    <div className="csvhub-confusion-wrap">
      <div className="csvhub-confusion">
        <div />
        <div className="csvhub-cell csvhub-cell-head">Pred Pass</div>
        <div className="csvhub-cell csvhub-cell-head">Pred Fail</div>
        <div className="csvhub-cell csvhub-cell-head">Actual Pass</div>
        <div className="csvhub-cell csvhub-cell--tp">TP · {cm.TP}</div>
        <div className="csvhub-cell csvhub-cell--fn">FN · {cm.FN}</div>
        <div className="csvhub-cell csvhub-cell-head">Actual Fail</div>
        <div className="csvhub-cell csvhub-cell--fp">FP · {cm.FP}</div>
        <div className="csvhub-cell csvhub-cell--tn">TN · {cm.TN}</div>
      </div>
      <div className="csvhub-metrics">
        <div>
          <span>Accuracy</span>
          <strong>{(cm.acc * 100).toFixed(1)}%</strong>
        </div>
        <div>
          <span>Precision</span>
          <strong>{(cm.prec * 100).toFixed(1)}%</strong>
        </div>
        <div>
          <span>Recall</span>
          <strong>{(cm.rec * 100).toFixed(1)}%</strong>
        </div>
        <div>
          <span>F1</span>
          <strong>{(cm.f1 * 100).toFixed(1)}%</strong>
        </div>
        <div>
          <span>Balanced accuracy</span>
          <strong>{(cm.bal * 100).toFixed(1)}%</strong>
        </div>
      </div>
      {best ? (
        <div className="csvhub-suggest">
          <Sparkles size={14} /> Best match at threshold <strong>{best.t.toFixed(2)}</strong> → {(best.acc * 100).toFixed(1)}% accuracy.
          <button type="button" className="btn btn-sm btn-outline" onClick={() => onApplyThreshold(best.t)}>
            Apply
          </button>
        </div>
      ) : null}
    </div>
  );
}

/* ── At-risk list ─────────────────────────── */
function AtRiskPanel({ rows, threshold, kmResult, indexToCluster }) {
  const atRisk = rows
    .filter((r) => typeof r.passProb === 'number' && r.passProb < threshold)
    .sort((a, b) => a.passProb - b.passProb);
  if (!atRisk.length) return <EmptyState>No students below the current threshold — lower the slider on Student Records to see more.</EmptyState>;
  return (
    <div className="csvhub-table-wrap">
      <table className="csvhub-table">
        <thead>
          <tr>
            <th>Rank</th>
            <th>Name</th>
            <th>ID</th>
            <th>Archetype</th>
            <th>P(Pass)</th>
            <th>Gap</th>
            <th>Avg subject %</th>
          </tr>
        </thead>
        <tbody>
          {atRisk.slice(0, 100).map((r, i) => {
            const cl = indexToCluster?.get(r.id);
            const clName = cl != null ? kmResult?.clusters?.find((x) => x.c === cl)?.name : null;
            return (
              <tr key={r.id}>
                <td className="tabnums">#{i + 1}</td>
                <td>{r.csvMeta?.displayName || '—'}</td>
                <td className="tabnums">{r.csvMeta?.displayId || '—'}</td>
                <td>{clName ?? '—'}</td>
                <td className="tabnums csvhub-fail">{(r.passProb * 100).toFixed(1)}%</td>
                <td className="tabnums">{((threshold - r.passProb) * 100).toFixed(1)} pts</td>
                <td className="tabnums">{r.compositeLabel ?? '—'}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/* ── Scatter (subj avg × attendance) ─────── */
function ScatterPanel({ rows }) {
  const pts = rows
    .map((r) => {
      const x = typeof r.composite === 'number' ? r.composite : null;
      const y = typeof r.numeric?.attendance_rate === 'number' ? r.numeric.attendance_rate * 100 : null;
      const p = typeof r.passProb === 'number' ? r.passProb : null;
      if (x == null || y == null || p == null) return null;
      return { x, y, p, name: r.csvMeta?.displayName };
    })
    .filter(Boolean);
  if (pts.length < 2) return <EmptyState>Need more rows with subject averages + attendance.</EmptyState>;
  const W = 520;
  const H = 260;
  const sx = (x) => 30 + (Math.min(100, Math.max(0, x)) / 100) * (W - 40);
  const sy = (y) => H - 24 - (Math.min(100, Math.max(0, y)) / 100) * (H - 36);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="csvhub-svg">
      <line x1={30} y1={H - 24} x2={W - 10} y2={H - 24} stroke="#94a3b8" strokeWidth={0.5} />
      <line x1={30} y1={10} x2={30} y2={H - 24} stroke="#94a3b8" strokeWidth={0.5} />
      {pts.map((p, i) => {
        const r = 2 + p.p * 5;
        const hue = Math.round(p.p * 140);
        return <circle key={i} cx={sx(p.x)} cy={sy(p.y)} r={r} fill={`hsl(${hue}, 65%, 50%)`} fillOpacity={0.75} />;
      })}
      <text x={W - 10} y={H - 8} textAnchor="end" fontSize="10" fill="#64748b">subject avg %</text>
      <text x={34} y={14} fontSize="10" fill="#64748b">attendance %</text>
    </svg>
  );
}

/* ── Archetypes (k-means) ─────────────────── */
function ArchetypesPanel({ kmResult }) {
  if (!kmResult) return <EmptyState>Not enough feature-rich rows to cluster. Include attendance, study, sleep, stress.</EmptyState>;
  const total = kmResult.clusters.reduce((s, c) => s + c.members.length, 0) || 1;
  return (
    <div className="csvhub-archetype-grid">
      {kmResult.clusters.map((cl) => (
        <div key={cl.c} className="csvhub-archetype">
          <h4>{cl.name}</h4>
          <strong>{cl.members.length} students</strong>
          <span className="csvhub-archetype-meta">{Math.round((100 * cl.members.length) / total)}%</span>
          <ul>
            {kmResult.features.map((f) => (
              <li key={f}>
                <span>{prettyFeatureName(f)}</span>
                <strong>{cl.centroid[f]?.toFixed(2)}</strong>
              </li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  );
}

/* ── Compare (two saved classes) ──────────── */
function ComparePanel({ classes, currentRows, saveCurrent, removeClass }) {
  const [aId, setAId] = useState('');
  const [bId, setBId] = useState('');
  const a = classes.find((c) => c.id === aId);
  const b = classes.find((c) => c.id === bId);

  const stats = (rows) => {
    if (!rows?.length) return null;
    const n = rows.length;
    const avgP = rows.reduce((s, r) => s + (r.passProb ?? 0), 0) / n;
    const passN = rows.filter((r) => (r.passProb ?? 0) >= 0.5).length;
    return {
      n,
      avgP,
      pass: passN,
      fail: n - passN,
    };
  };
  const aStat = stats(a?.rows);
  const bStat = stats(b?.rows);

  return (
    <>
      <div className="csvhub-compare-controls">
        <button type="button" className="btn btn-sm btn-outline" onClick={saveCurrent} disabled={!currentRows.length}>
          Save current import as class
        </button>
      </div>
      <div className="csvhub-compare-pick">
        <label>
          Class A
          <select value={aId} onChange={(e) => setAId(e.target.value)}>
            <option value="">—</option>
            {classes.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
          </select>
          {a ? <button type="button" className="csvhub-remove" onClick={() => removeClass(a.id)}>✕</button> : null}
        </label>
        <label>
          Class B
          <select value={bId} onChange={(e) => setBId(e.target.value)}>
            <option value="">—</option>
            {classes.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
          </select>
          {b ? <button type="button" className="csvhub-remove" onClick={() => removeClass(b.id)}>✕</button> : null}
        </label>
      </div>
      {aStat && bStat ? (
        <div className="csvhub-compare-grid">
          {[
            { label: 'Students', a: aStat.n, b: bStat.n },
            { label: 'Pass @ thr 0.5', a: aStat.pass, b: bStat.pass },
            { label: 'Fail @ thr 0.5', a: aStat.fail, b: bStat.fail },
            { label: 'Avg P(Pass)', a: `${(aStat.avgP * 100).toFixed(1)}%`, b: `${(bStat.avgP * 100).toFixed(1)}%` },
          ].map((row) => (
            <div key={row.label} className="csvhub-compare-row">
              <span>{row.label}</span>
              <strong>{row.a}</strong>
              <strong>{row.b}</strong>
            </div>
          ))}
        </div>
      ) : (
        <EmptyState>Save at least two classes to compare.</EmptyState>
      )}
    </>
  );
}

/* ── Export tab ──────────────────────────── */
function ExportPanel({ rows, threshold }) {
  const passN = rows.filter((r) => (r.passProb ?? 0) >= threshold).length;
  const failN = rows.length - passN;
  const common = {
    name: (r) => r.csvMeta?.displayName ?? '',
    id: (r) => r.csvMeta?.displayId ?? '',
    composite: (r) => r.compositeLabel ?? '',
    pass_probability: (r) => (typeof r.passProb === 'number' ? r.passProb.toFixed(4) : ''),
    predicted: (r) => ((r.passProb ?? 0) >= threshold ? 'Pass' : 'Fail'),
    file_label: (r) => r.csvMeta?.fileTruthLabel ?? '',
    match: (r) =>
      r.csvMeta?.fileExpectPass == null
        ? ''
        : ((r.passProb ?? 0) >= threshold) === r.csvMeta.fileExpectPass
          ? 'Yes'
          : 'No',
  };
  const columns = [
    { label: 'row_index', get: (r) => r.csvMeta?.rowIndex ?? '' },
    { label: 'name', get: common.name },
    { label: 'student_id', get: common.id },
    { label: 'subject_avg', get: common.composite },
    { label: 'pass_probability', get: common.pass_probability },
    { label: 'predicted_outcome', get: common.predicted },
    { label: 'file_label', get: common.file_label },
    { label: 'match', get: common.match },
  ];
  const featureCols = KEY_FEATURES.map((f) => ({ label: f, get: (r) => r.numeric?.[f] ?? '' }));
  const extraCols = [
    ...columns.slice(0, 8),
    ...featureCols,
  ];
  const downloadScored = () => downloadBlob('edupredict_scored.csv', 'text/csv', toCsv(rows, columns));
  const downloadExpanded = () => downloadBlob('edupredict_scored_features.csv', 'text/csv', toCsv(rows, extraCols));
  const atRiskRows = rows.filter((r) => (r.passProb ?? 0) < threshold);
  const downloadAtRisk = () => downloadBlob('edupredict_at_risk.csv', 'text/csv', toCsv(atRiskRows, columns));

  return (
    <div className="csvhub-export">
      <div className="csvhub-export-grid">
        <div>
          <strong>Scored rows</strong>
          <p>Predictions + file labels + match column for every imported student.</p>
          <button type="button" className="btn btn-sm btn-primary" onClick={downloadScored}>
            <Download size={14} /> scored.csv
          </button>
        </div>
        <div>
          <strong>Full feature dump</strong>
          <p>Above, plus every model input (for spreadsheet analysis).</p>
          <button type="button" className="btn btn-sm btn-outline" onClick={downloadExpanded}>
            <Download size={14} /> scored_features.csv
          </button>
        </div>
        <div>
          <strong>At-risk shortlist</strong>
          <p>Only students with P(Pass) below the current threshold ({(threshold * 100).toFixed(0)}%).</p>
          <button type="button" className="btn btn-sm btn-outline" onClick={downloadAtRisk}>
            <Download size={14} /> at_risk.csv
          </button>
        </div>
      </div>
      <p className="csvhub-export-summary">
        {rows.length} total · {passN} Pass · {failN} Fail · {atRiskRows.length} at-risk
      </p>
    </div>
  );
}

/* ── Audit tab ───────────────────────────── */
function AuditPanel() {
  const [entries, setEntries] = useState(() => readAudit());
  return (
    <>
      <div className="csvhub-audit-head">
        <span><History size={14} /> {entries.length} imports logged</span>
        {entries.length ? (
          <button
            type="button"
            className="btn btn-sm btn-outline"
            onClick={() => {
              clearAudit();
              setEntries([]);
            }}
          >
            Clear log
          </button>
        ) : null}
      </div>
      {entries.length === 0 ? (
        <EmptyState>No imports audited yet.</EmptyState>
      ) : (
        <div className="csvhub-table-wrap">
          <table className="csvhub-table">
            <thead>
              <tr>
                <th>When</th>
                <th>File</th>
                <th>Rows</th>
                <th>Match %</th>
                <th>Threshold</th>
              </tr>
            </thead>
            <tbody>
              {entries.map((e, i) => (
                <tr key={i}>
                  <td>{e.at}</td>
                  <td>{e.file}</td>
                  <td className="tabnums">{e.rows}</td>
                  <td className="tabnums">{e.matchPct != null ? `${e.matchPct}%` : '—'}</td>
                  <td className="tabnums">{e.threshold ?? '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </>
  );
}

export default function ImportHubPage({ predictionHistory }) {
  const csvRows = useMemo(
    () => predictionHistory.filter((r) => r.source === 'csv'),
    [predictionHistory],
  );
  const groups = useMemo(() => groupByFile(csvRows), [csvRows]);
  const fileNames = Array.from(groups.keys());
  const [activeFile, setActiveFile] = useState(fileNames[0] || '');
  const effectiveFile = activeFile || fileNames[0] || '';

  const rows = useMemo(
    () => (effectiveFile ? groups.get(effectiveFile) ?? [] : csvRows),
    [effectiveFile, groups, csvRows],
  );

  const [threshold, setThreshold] = useState(() => {
    const raw = Number(localStorage.getItem('edupredict-records-threshold'));
    return Number.isFinite(raw) && raw > 0 && raw < 1 ? raw : 0.5;
  });
  useEffect(() => {
    try {
      localStorage.setItem('edupredict-records-threshold', String(threshold));
    } catch { /* ignore */ }
  }, [threshold]);

  const [tab, setTab] = useState('overview');

  const km = useMemo(() => (rows.length >= 6 ? kmeans(rows, 3) : null), [rows]);
  const indexToCluster = useMemo(() => {
    if (!km) return null;
    const map = new Map();
    km.indices.forEach((origIdx, i) => {
      map.set(rows[origIdx]?.id, km.labels[i]);
    });
    return map;
  }, [km, rows]);

  const [classes, setClasses] = useState(() => readClasses());
  const saveCurrent = () => {
    if (!rows.length) return;
    const name = prompt('Name this class:', effectiveFile || 'Class');
    if (!name) return;
    const entry = {
      id: `cls-${Date.now()}`,
      name,
      savedAt: new Date().toISOString(),
      rows: rows.map((r) => ({
        id: r.id,
        passProb: r.passProb,
        numeric: r.numeric,
        composite: r.composite,
        csvMeta: r.csvMeta,
      })),
    };
    const next = [entry, ...classes].slice(0, 20);
    setClasses(next);
    writeClasses(next);
  };
  const removeClass = (id) => {
    const next = classes.filter((c) => c.id !== id);
    setClasses(next);
    writeClasses(next);
  };

  const TABS = [
    ['overview', 'Overview', Layers],
    ['preview', 'Preview', FileSpreadsheet],
    ['quality', 'Quality', Gauge],
    ['heatmap', 'Heatmap', Grid3x3],
    ['confusion', 'Confusion', Target],
    ['atrisk', 'At-risk', AlertTriangle],
    ['scatter', 'Scatter', Activity],
    ['archetypes', 'Archetypes', PieChart],
    ['compare', 'Compare', Users],
    ['export', 'Export', Download],
    ['audit', 'Audit', History],
  ];

  return (
    <>
      <div className="page-hero">
        <div className="page-title-row page-title-row--flush">
          <div>
            <h1>CSV Hub</h1>
            <p>
              Deep dive on imported student files — quality, explanations, archetypes, and downloads.
              Driven entirely by data already in your session; no re-upload needed.
            </p>
          </div>
        </div>
      </div>

      <div className="panel panel--elevated">
        <div className="panel-header">
          <div>
            <h3>
              {effectiveFile || 'All CSV imports'}{' '}
              <span style={{ color: 'var(--text-light)', fontWeight: 400, fontSize: 13 }}>
                · {rows.length} row{rows.length !== 1 ? 's' : ''}
              </span>
            </h3>
          </div>
          <div className="csvhub-controls">
            {fileNames.length > 1 ? (
              <select value={effectiveFile} onChange={(e) => setActiveFile(e.target.value)} className="csvhub-file-select">
                {fileNames.map((n) => <option key={n} value={n}>{n}</option>)}
              </select>
            ) : null}
            <div className="csvhub-threshold">
              <span>thr</span>
              <input
                type="range"
                min="0.02"
                max="0.98"
                step="0.01"
                value={threshold}
                onChange={(e) => setThreshold(parseFloat(e.target.value))}
              />
              <strong>{threshold.toFixed(2)}</strong>
            </div>
          </div>
        </div>

        <div className="panel-body">
          {rows.length === 0 ? (
            <EmptyState>
              <CheckCircle2 size={36} />
              <h4>Import a CSV to fill this hub</h4>
              <p>Drop a file on the sidebar upload zone. Every tab below will populate using the same batch.</p>
            </EmptyState>
          ) : (
            <>
              <div className="csvhub-tabs">
                {TABS.map(([id, label, icon]) => (
                  <TabBtn key={id} active={tab === id} onClick={() => setTab(id)} icon={icon} label={label} />
                ))}
              </div>
              <div className="csvhub-panel">
                {tab === 'overview' && <OverviewPanel rows={rows} threshold={threshold} />}
                {tab === 'preview' && <PreviewPanel rows={rows} />}
                {tab === 'quality' && <QualityPanel rows={rows} />}
                {tab === 'heatmap' && <HeatmapPanel rows={rows} />}
                {tab === 'confusion' && (
                  <ConfusionPanel rows={rows} threshold={threshold} onApplyThreshold={setThreshold} />
                )}
                {tab === 'atrisk' && (
                  <AtRiskPanel rows={rows} threshold={threshold} kmResult={km} indexToCluster={indexToCluster} />
                )}
                {tab === 'scatter' && <ScatterPanel rows={rows} />}
                {tab === 'archetypes' && <ArchetypesPanel kmResult={km} />}
                {tab === 'compare' && (
                  <ComparePanel classes={classes} currentRows={rows} saveCurrent={saveCurrent} removeClass={removeClass} />
                )}
                {tab === 'export' && <ExportPanel rows={rows} threshold={threshold} />}
                {tab === 'audit' && <AuditPanel />}
              </div>
            </>
          )}
        </div>
      </div>
    </>
  );
}
