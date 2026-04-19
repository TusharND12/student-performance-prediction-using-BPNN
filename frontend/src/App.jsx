import React, { useEffect, useId, useMemo, useRef, useState } from 'react';
import axios from 'axios';
import { downloadEduPredictPdf } from './pdfReport.js';
import LabsPage, { buildSessionNarrativeArc } from './LabsPage.jsx';
import { postStudyBudget } from './labsApi.js';
import { postCompleteness, getCalibration, getMetaVersion, postGoalSeek } from './metaApi.js';
import { apiUrl, backendOrigin } from './apiRoutes.js';
import NetworkDiagram from './NetworkDiagram.jsx';
import AnalyticsExtraGraphs from './AnalyticsCharts.jsx';
import ImportHubPage from './ImportHubPage.jsx';
import { appendAudit } from './csvHubUtils.js';
import './importHub.css';
import {
  BarChart3, Users, GraduationCap, Brain,
  BookOpen, Clock, Activity, Target, Zap, Search,
  Bell, Settings, ChevronRight, CheckCircle2, AlertCircle,
  LayoutDashboard, FileText, PieChart,
  Moon, Sun, Hourglass, Download, Shield, Sparkles,
  LifeBuoy, ExternalLink, RefreshCw, FileDown,
  Lightbulb, ListChecks, ArrowRight,
  Server, Code2, Database,   Workflow, ChevronDown, FlaskConical,
  CalendarDays, UploadCloud, FileSpreadsheet, Loader2, X,
} from 'lucide-react';

// Dev: Vite proxies `/api/*` → `http://127.0.0.1:8000/*` (see vite.config.js; same for `vite preview`).
// Prod: set `VITE_API_URL` to the API origin only, e.g. `https://api.example.com` (not `.../api`).
function predictUrl() {
  return apiUrl('predict');
}

/** Server-side CSV parse + BPNN batch + label agreement (multipart upload). */
function importCsvUrl() {
  return apiUrl('import/csv');
}

function apiBaseUrl() {
  return backendOrigin() ?? 'http://127.0.0.1:8000';
}

/** v2 key: legacy `edupredict-skip-landing` is no longer read so the intro shows again after upgrade. */
const LANDING_SKIP_STORAGE_KEY = 'edupredict-skip-landing-v2';
const LANDING_SKIP_LEGACY_KEY = 'edupredict-skip-landing';

function readShowLandingOnStartup() {
  try {
    return localStorage.getItem(LANDING_SKIP_STORAGE_KEY) !== '1';
  } catch {
    return true;
  }
}

const PAGE_BREADCRUMB = {
  dashboard: 'Dashboard',
  predict: 'Performance Predictor',
  students: 'Student Records',
  imports: 'CSV Hub',
  reports: 'Reports',
  analytics: 'Analytics',
  improvement: 'Improvement coach',
  settings: 'Settings',
  help: 'Help & Support',
  labs: 'Innovation Labs',
  timetable: 'Study timetable',
};

/** Tiny monochrome sparkline for dashboard (confidence % over recent runs, oldest → newest). */
function MiniSparkline({ values, width = 80, height = 24 }) {
  const v = (values || []).map(Number).filter((n) => Number.isFinite(n));
  if (v.length < 2) return null;
  const min = Math.min(...v, 0);
  const max = Math.max(...v, 100);
  const span = Math.max(1e-6, max - min);
  const pad = 2;
  const w = width;
  const h = height;
  const pts = v.map((val, i) => {
    const x = pad + (i / (v.length - 1)) * (w - pad * 2);
    const y = pad + (1 - (val - min) / span) * (h - pad * 2);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  const d = `M ${pts.join(' L ')}`;
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} className="dashboard-kite-sparkline" aria-hidden>
      <path d={d} fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

// ─── Sidebar ──────────────────────────────────────────
function Sidebar({ activePage, onNavigate, onCsvFileSelect, csvImportState }) {
  const link = (id, Icon, label) => {
    const isActive = activePage === id;
    return (
      <button
        type="button"
        className={`sidebar-link ${isActive ? 'active' : ''}`}
        aria-current={isActive ? 'page' : undefined}
        onClick={() => onNavigate(id)}
      >
        <Icon className="sidebar-link-icon" strokeWidth={1.75} aria-hidden />
        {label}
      </button>
    );
  };

  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <div className="sidebar-brand-icon">
          <Sparkles size={18} strokeWidth={2} />
        </div>
        <div>
          <h1>EduPredict</h1>
          <span>Student success suite</span>
        </div>
      </div>

      <nav className="sidebar-nav" aria-label="Main navigation">
        <div className="sidebar-section">
          <div className="sidebar-section-title">Main</div>
          {link('dashboard', LayoutDashboard, 'Dashboard')}
          {link('predict', Brain, 'Performance Predictor')}
          {link('improvement', Lightbulb, 'Improvement coach')}
          {link('timetable', CalendarDays, 'Study timetable')}
        </div>

        <div className="sidebar-section">
          <div className="sidebar-section-title">Management</div>
          {link('students', Users, 'Student Records')}
          {link('imports', FileSpreadsheet, 'CSV Hub')}
          {link('reports', FileText, 'Reports')}
          {link('analytics', PieChart, 'Analytics')}
          {link('labs', FlaskConical, 'Innovation Labs')}
        </div>

        <div className="sidebar-section">
          <div className="sidebar-section-title">System</div>
          {link('settings', Settings, 'Settings')}
          {link('help', LifeBuoy, 'Help & Support')}
        </div>
      </nav>

      <div className="sidebar-footer">
        <CsvUploadZone onFileSelect={onCsvFileSelect} csvImportState={csvImportState} />
        <div className="sidebar-user">
          <div className="sidebar-avatar">EP</div>
          <div className="sidebar-user-info">
            <p>EduPredict</p>
            <small>BPNN workspace</small>
          </div>
        </div>
      </div>
    </aside>
  );
}

function CsvUploadZone({ onFileSelect, csvImportState }) {
  const inputRef = useRef(null);
  const [dragOver, setDragOver] = useState(false);
  const [pickedName, setPickedName] = useState('');
  const loading = !!csvImportState?.loading;
  const disabled = loading;

  const handleFile = (file) => {
    if (!file) return;
    if (!/\.csv$/i.test(file.name) && file.type !== 'text/csv' && file.type !== 'application/vnd.ms-excel') {
      return;
    }
    setPickedName(file.name);
    onFileSelect?.(file);
  };

  const onDrop = (e) => {
    e.preventDefault();
    setDragOver(false);
    if (disabled) return;
    handleFile(e.dataTransfer?.files?.[0]);
  };

  const onPick = () => {
    if (disabled) return;
    inputRef.current?.click();
  };

  const msg = csvImportState?.message;
  const summary = csvImportState?.summary;
  const error = !!csvImportState?.error;
  const [showDetails, setShowDetails] = useState(false);
  // Collapse details while a new upload is running; using a derived flag below.
  const effectiveShowDetails = loading ? false : showDetails;

  return (
    <div className="csv-upload-card">
      <div className="csv-upload-head">
        <span className="csv-upload-title">Load students CSV</span>
        {pickedName ? (
          <button
            type="button"
            className="csv-upload-clear"
            onClick={(e) => {
              e.stopPropagation();
              setPickedName('');
            }}
            aria-label="Clear selected file"
            title="Clear"
          >
            <X size={12} strokeWidth={2.25} />
          </button>
        ) : null}
      </div>

      <button
        type="button"
        className={`csv-dropzone ${dragOver ? 'is-drag' : ''} ${loading ? 'is-loading' : ''} ${error ? 'is-error' : ''}`}
        onClick={onPick}
        onDragOver={(e) => {
          e.preventDefault();
          if (!disabled) setDragOver(true);
        }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        disabled={disabled}
      >
        <span className="csv-dropzone-icon" aria-hidden>
          {loading ? (
            <Loader2 size={22} strokeWidth={2} className="csv-spin" />
          ) : pickedName ? (
            <FileSpreadsheet size={22} strokeWidth={1.75} />
          ) : (
            <UploadCloud size={24} strokeWidth={1.75} />
          )}
        </span>
        <span className="csv-dropzone-text">
          {loading
            ? 'Processing…'
            : pickedName || (dragOver ? 'Drop CSV to import' : 'Click or drop CSV here')}
        </span>
        {!loading && !pickedName ? (
          <span className="csv-dropzone-hint">.csv · UTF-8</span>
        ) : null}
      </button>

      <input
        id="sidebar-csv-upload"
        ref={inputRef}
        type="file"
        accept=".csv,text/csv"
        className="csv-upload-input"
        onChange={(e) => {
          handleFile(e.target.files?.[0]);
          e.target.value = '';
        }}
        disabled={disabled}
      />

      {summary || msg ? (
        <div className={`csv-status ${error ? 'is-error' : 'is-ok'}`}>
          <button
            type="button"
            className="csv-status-head"
            onClick={() => setShowDetails((s) => !s)}
            aria-expanded={effectiveShowDetails}
          >
            <span className="csv-status-dot" aria-hidden />
            <span className="csv-status-summary">{summary || (msg?.split('\n')[0] ?? '')}</span>
            {msg ? (
              <ChevronDown
                size={14}
                className="csv-status-chevron"
                style={{ transform: effectiveShowDetails ? 'rotate(180deg)' : 'none' }}
                aria-hidden
              />
            ) : null}
          </button>
          {effectiveShowDetails && msg ? (
            <div className="csv-status-body">
              {msg.split('\n').map((line, i) => (
                <p key={i}>{line}</p>
              ))}
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

// ─── Header ───────────────────────────────────────────
function Header({ breadcrumb, searchQuery, onSearchChange, darkMode, onToggleTheme }) {
  return (
    <header className="header">
      <div className="header-left">
        <div className="header-breadcrumb">
          <span className="header-crumb-home">EduPredict</span>
          <ChevronRight size={14} className="header-crumb-chev" />
          <span className="header-crumb-current">{breadcrumb}</span>
        </div>
      </div>
      <div className="header-right">
        <div className="header-search">
          <Search size={16} className="header-search-icon" />
          <input
            value={searchQuery}
            onChange={(e) => onSearchChange(e.target.value)}
            placeholder="Filter records (Student Records page)…"
            aria-label="Search and filter"
          />
        </div>
        <button type="button" className="header-btn" title="Notifications" aria-label="Notifications">
          <Bell size={18} />
        </button>
        <button
          type="button"
          className="header-btn header-btn-theme"
          onClick={onToggleTheme}
          title={darkMode ? 'Light mode' : 'Dark mode'}
          aria-label={darkMode ? 'Switch to light mode' : 'Switch to dark mode'}
        >
          {darkMode ? <Sun size={18} /> : <Moon size={18} />}
        </button>
      </div>
    </header>
  );
}

// ─── Dashboard Page ───────────────────────────────────
function DashboardPage({ onNavigate, predictionHistory }) {
  const recent = predictionHistory.slice(0, 5);
  const passCount = predictionHistory.filter((r) => r.outcome === 'Pass').length;
  const failCount = predictionHistory.length - passCount;
  const narrative = useMemo(() => buildSessionNarrativeArc(predictionHistory), [predictionHistory]);
  const sparkConfidences = useMemo(
    () =>
      [...predictionHistory]
        .filter((r) => typeof r.confidence === 'number' && Number.isFinite(r.confidence))
        .reverse()
        .slice(-12)
        .map((r) => r.confidence),
    [predictionHistory],
  );

  return (
    <div className="dashboard-kite">
      <header className="dashboard-kite-hero">
        <div>
          <h1>Dashboard</h1>
          <p>Session overview · BPNN runs &amp; model context</p>
        </div>
        <button type="button" className="dashboard-kite-btn-primary" onClick={() => onNavigate('predict')}>
          <Brain size={15} strokeWidth={2} aria-hidden />
          New prediction
        </button>
      </header>

      <div className="dashboard-kite-stats" aria-label="Session summary">
        <button type="button" className="dashboard-kite-stat dashboard-kite-stat--link" onClick={() => onNavigate('students')}>
          <div className="dashboard-kite-stat-label">Predictions logged</div>
          <div className="dashboard-kite-stat-value tabnums">{predictionHistory.length}</div>
          {sparkConfidences.length >= 2 && (
            <div className="dashboard-kite-spark-wrap">
              <span className="dashboard-kite-stat-micro">Confidence trend</span>
              <MiniSparkline values={sparkConfidences} />
            </div>
          )}
          <div className="dashboard-kite-stat-sub">Open student records</div>
        </button>
        <div className="dashboard-kite-stat">
          <div className="dashboard-kite-stat-label">Pass / fail (session)</div>
          <div className="dashboard-kite-stat-value tabnums">
            {passCount} <span className="dashboard-kite-stat-sep">/</span> {failCount}
          </div>
          <div className="dashboard-kite-stat-sub dashboard-kite-stat-sub--muted">From saved runs</div>
        </div>
        <div className="dashboard-kite-stat">
          <div className="dashboard-kite-stat-label">Inference stack</div>
          <div className="dashboard-kite-stat-value dashboard-kite-stat-value--sm">TF · FastAPI</div>
          <div className="dashboard-kite-stat-sub dashboard-kite-stat-sub--muted">Low-latency predict</div>
        </div>
        <button type="button" className="dashboard-kite-stat dashboard-kite-stat--link" onClick={() => onNavigate('analytics')}>
          <div className="dashboard-kite-stat-label">Analytics</div>
          <div className="dashboard-kite-stat-value dashboard-kite-stat-value--sm">Breakdown</div>
          <div className="dashboard-kite-stat-sub">View charts</div>
        </button>
      </div>

      {predictionHistory.length > 0 && (
        <div className="panel panel--elevated dashboard-kite-narrative">
          <div className="panel-header">
            <div>
              <h3>
                <Sparkles size={14} className="dashboard-kite-narrative-icon" strokeWidth={2} aria-hidden />
                Session narrative
              </h3>
              <p>Three beats from your logged runs — context only, not a forecast.</p>
            </div>
          </div>
          <div className="panel-body dashboard-kite-narrative-body">
            <ul className="dashboard-kite-narrative-list">
              {narrative.beats.map((b, i) => (
                <li key={i}>{b}</li>
              ))}
            </ul>
            <button type="button" className="dashboard-kite-btn-outline" onClick={() => onNavigate('labs')}>
              Innovation Labs — sensitivity, cohort, adversary
            </button>
          </div>
        </div>
      )}

      <div className="dashboard-two-col dashboard-kite-split">
        <div className="panel panel--elevated">
          <div className="panel-header">
            <div>
              <h3>Recent predictions</h3>
              <p>Last five runs · Performance Predictor</p>
            </div>
            <button type="button" className="dashboard-kite-btn-text" onClick={() => onNavigate('students')}>
              View all
            </button>
          </div>
          <div className="panel-body panel-body--flush">
            {recent.length === 0 ? (
              <div className="empty-inline dashboard-kite-empty">
                <Activity size={36} className="empty-inline-icon" strokeWidth={1.25} />
                <h4>No runs yet</h4>
                <p>Use Performance Predictor — results sync here automatically.</p>
                <button type="button" className="dashboard-kite-btn-primary" onClick={() => onNavigate('predict')}>
                  Go to predictor
                </button>
              </div>
            ) : (
              <ul className="recent-list dashboard-kite-recent">
                {recent.map((r) => (
                  <li key={r.id} className="recent-list-item">
                    <span
                      className={`recent-tag ${
                        r.outcome === 'Pass' ? 'recent-tag--pass' : r.outcome === 'Fail' ? 'recent-tag--fail' : ''
                      }`}
                      style={r.outcome !== 'Pass' && r.outcome !== 'Fail' ? { opacity: 0.85 } : undefined}
                    >
                      {r.outcome}
                    </span>
                    <div className="recent-list-meta">
                      <span className="recent-list-time">{r.atLabel}</span>
                      <span className="recent-list-sub">
                        {r.csvMeta?.displayName ? `${r.csvMeta.displayName} · ` : ''}
                        Subj {r.compositeLabel ?? '—'}
                      </span>
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>

        <div className="panel panel--elevated">
          <div className="panel-header">
            <div>
              <h3>Model</h3>
              <p>Architecture &amp; artifacts</p>
            </div>
          </div>
          <div className="panel-body panel-body--dense">
            {[
              ['Architecture', 'Back-propagation neural network (Dense)'],
              ['Layers', '64 → BN → Dropout → 32 → Dropout → Sigmoid'],
              ['Optimizer', 'Adam · binary cross-entropy'],
              ['Training data', 'student_data.csv'],
              ['API', 'FastAPI · model.h5 + scaler.pkl'],
            ].map(([k, v]) => (
              <div key={k} className="model-info-row">
                <span>{k}</span>
                <span>{v}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Student Records (prediction log) ─────────────────
function csvRecordSearchBlob(r) {
  const parts = [
    r.outcome,
    r.atLabel,
    r.confidence != null ? String(r.confidence) : '',
    r.compositeLabel ?? '',
    r.csvMeta?.displayName ?? '',
    r.csvMeta?.displayId ?? '',
    r.csvMeta?.fileName ?? '',
    r.csvMeta?.fileTruthLabel ?? '',
    r.csvMeta?.fileTruthRaw ?? '',
    (r.csvMeta?.extras ?? []).map((e) => `${e.label} ${e.value}`).join(' '),
    r.csvMeta?.batchError ?? '',
  ];
  return parts.join(' ').toLowerCase();
}

function StudentRecordsPage({ predictionHistory, searchQuery, onNavigate }) {
  const q = searchQuery.trim().toLowerCase();
  // Local threshold slider so the user can cut rows by a stricter probability cutoff than the model's.
  const [threshold, setThreshold] = useState(() => {
    const raw = Number(localStorage.getItem('edupredict-records-threshold'));
    return Number.isFinite(raw) && raw > 0 && raw < 1 ? raw : 0.5;
  });
  useEffect(() => {
    try {
      localStorage.setItem('edupredict-records-threshold', String(threshold));
    } catch { /* ignore */ }
  }, [threshold]);

  const filteredBase = predictionHistory.filter((r) => !q || csvRecordSearchBlob(r).includes(q));
  const rows = useMemo(
    () =>
      filteredBase.map((r) => {
        if (r.outcome === 'Error') return r;
        const p = typeof r.passProb === 'number' && Number.isFinite(r.passProb) ? r.passProb : null;
        if (p == null) return r;
        const outcome = p >= threshold ? 'Pass' : 'Fail';
        return { ...r, outcome };
      }),
    [filteredBase, threshold],
  );

  // Counts and match % based on the (possibly overridden) outcomes
  const overriddenPass = rows.filter((r) => r.outcome === 'Pass').length;
  const overriddenFail = rows.filter((r) => r.outcome === 'Fail').length;
  let compared = 0;
  let matched = 0;
  for (const r of rows) {
    const ft = r.csvMeta?.fileExpectPass;
    if (ft == null || r.outcome === 'Error') continue;
    compared += 1;
    if ((r.outcome === 'Pass') === ft) matched += 1;
  }
  const matchPct = compared ? Math.round((100 * matched) / compared) : null;

  return (
    <>
      <div className="page-hero">
        <div className="page-title-row page-title-row--flush">
          <div>
            <h1>Student Records</h1>
          </div>
          <button type="button" className="btn btn-primary" onClick={() => onNavigate('predict')}>
            <Brain size={16} /> New prediction
          </button>
        </div>
      </div>

      <div className="panel panel--elevated">
        <div className="panel-header">
          <div>
            <h3>Session log</h3>
            <p>{rows.length} record{rows.length !== 1 ? 's' : ''}</p>
          </div>
        </div>
        <div className="panel-body panel-body--flush">
          <div
            style={{
              display: 'flex',
              flexWrap: 'wrap',
              alignItems: 'center',
              gap: 16,
              padding: '12px 14px',
              margin: '0 0 14px',
              borderRadius: 10,
              border: '1px solid var(--border)',
              background: 'var(--surface)',
              fontSize: 13,
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, flex: '1 1 360px' }}>
              <span style={{ fontWeight: 600, color: 'var(--text)' }}>Decision threshold</span>
              <input
                type="range"
                min="0.02"
                max="0.98"
                step="0.01"
                value={threshold}
                onChange={(e) => setThreshold(parseFloat(e.target.value))}
                style={{ flex: 1, accentColor: '#4f46e5' }}
              />
              <span
                style={{
                  fontFamily: 'ui-monospace, SFMono-Regular, Menlo, monospace',
                  background: 'rgba(99, 102, 241, 0.1)',
                  padding: '2px 10px',
                  borderRadius: 6,
                  minWidth: 56,
                  textAlign: 'center',
                }}
              >
                {threshold.toFixed(2)}
              </span>
              <button
                type="button"
                className="btn btn-sm btn-outline"
                onClick={() => setThreshold(0.5)}
                title="Reset to neutral 0.50"
              >
                Reset
              </button>
            </div>
            <div
              style={{
                display: 'flex',
                gap: 12,
                fontSize: 12,
                color: 'var(--text-light)',
                alignItems: 'center',
              }}
            >
              <span><strong style={{ color: '#047857' }}>{overriddenPass}</strong> Pass</span>
              <span><strong style={{ color: '#b91c1c' }}>{overriddenFail}</strong> Fail</span>
              {matchPct != null ? (
                <span>
                  Model vs file: <strong style={{ color: 'var(--text)' }}>{matchPct}%</strong> match over {compared}
                </span>
              ) : null}
            </div>
          </div>
          {rows.length === 0 ? (
            <div className="empty-inline">
              <Users size={40} className="empty-inline-icon" />
              <h4>{predictionHistory.length === 0 ? 'No predictions yet' : 'No matches'}</h4>
              <p>{predictionHistory.length === 0 ? 'Run the predictor or load a CSV from the sidebar.' : 'Try a different search in the header.'}</p>
            </div>
          ) : (
            <div className="table-wrap table-wrap--sticky">
              <table className="data-table">
                <thead>
                  <tr>
                    <th>#</th>
                    <th>Student / name</th>
                    <th>ID</th>
                    <th>Other columns</th>
                    <th>When</th>
                    <th>Outcome</th>
                    <th>Subj avg %</th>
                    <th>In file</th>
                    <th>Match</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((r) => {
                    const badgeClass =
                      r.outcome === 'Pass'
                        ? 'table-badge--pass'
                        : r.outcome === 'Fail'
                          ? 'table-badge--fail'
                          : 'table-badge--neutral';
                    const extras =
                      r.csvMeta?.extras?.length > 0
                        ? r.csvMeta.extras.map((e) => `${e.label}: ${e.value}`).join(' · ')
                        : '—';
                    const ft = r.csvMeta?.fileExpectPass;
                    const predPass = r.outcome === 'Pass';
                    const matchCell =
                      ft == null || r.outcome === 'Error' ? '—' : predPass === ft ? 'Yes' : 'No';
                    return (
                      <tr key={r.id}>
                        <td className="tabnums">{r.csvMeta?.rowIndex ?? '—'}</td>
                        <td>{r.csvMeta?.displayName || '—'}</td>
                        <td className="tabnums">{r.csvMeta?.displayId || '—'}</td>
                        <td style={{ maxWidth: 320, fontSize: 12, color: 'var(--text-light)' }} title={extras}>
                          {extras}
                        </td>
                        <td>{r.atLabel}</td>
                        <td>
                          <span className={`table-badge ${badgeClass}`}>{r.outcome}</span>
                          {r.csvMeta?.batchError ? (
                            <div style={{ fontSize: 11, marginTop: 4, maxWidth: 240, color: 'var(--text-light)' }} title={r.csvMeta.batchError}>
                              {r.csvMeta.batchError}
                            </div>
                          ) : null}
                        </td>
                        <td>{r.compositeLabel ?? '—'}</td>
                        <td className="tabnums">
                          {r.csvMeta?.fileTruthLabel ? (
                            <span title={`Raw: ${r.csvMeta?.fileTruthRaw ?? ''}`}>{r.csvMeta.fileTruthLabel}</span>
                          ) : (
                            '—'
                          )}
                        </td>
                        <td className="tabnums">{matchCell}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>
    </>
  );
}

// ─── Reports ──────────────────────────────────────────
function ReportsPage({ predictionHistory, improvementSnapshot, improvementPlan, darkMode, onNavigate }) {
  const [pdfBusy, setPdfBusy] = useState(false);

  const exportJson = async () => {
    let artifactMeta = null;
    try {
      artifactMeta = await getMetaVersion();
    } catch {
      artifactMeta = null;
    }
    const payload = {
      exportedAt: new Date().toISOString(),
      artifactMeta,
      runs: predictionHistory,
    };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `edupredict-log-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  const exportPdf = async () => {
    setPdfBusy(true);
    try {
      await downloadEduPredictPdf({
        predictionHistory,
        improvementSnapshot,
        improvementPlan,
        darkMode: Boolean(darkMode),
      });
    } catch (e) {
      console.error(e);
      window.alert(
        e?.message
          ? `Could not build PDF: ${e.message}\n\nStart the API from the project root:\npython -m uvicorn backend.main:app --port 8000`
          : 'Could not build PDF. Start the API (see Settings) and try again.',
      );
    } finally {
      setPdfBusy(false);
    }
  };

  return (
    <>
      <div className="page-hero">
        <div className="page-title-row page-title-row--flush">
          <div>
            <h1>Reports</h1>
            <p>Export and summary tools for your session data</p>
          </div>
        </div>
      </div>

      <div className="panel panel--elevated" style={{ marginBottom: 24 }}>
        <div className="panel-header">
          <div>
            <h3>PDF report — what&apos;s inside</h3>
            <p>Multipage export built on the server (ReportLab). Outline only; download from the card below.</p>
          </div>
        </div>
        <div className="panel-body">
          <ul className="reports-pdf-outline">
            <li><strong>1</strong> Session title, timestamp, and theme snapshot</li>
            <li><strong>2</strong> Run summary table (when, outcome, confidence, subject average)</li>
            <li><strong>3</strong> Per-run inputs and model outputs (recent runs)</li>
            <li><strong>4</strong> Improvement coach section when a snapshot exists</li>
            <li><strong>5</strong> Appendix — app areas and export metadata</li>
          </ul>
        </div>
      </div>

      <div className="reports-grid">
        <div className="feature-card">
          <div className="feature-card-icon">
            <Download size={22} />
          </div>
          <h3>Export prediction log</h3>
          <p>Download all logged runs as JSON for your report or viva.</p>
          <button type="button" className="btn btn-primary" onClick={exportJson} disabled={predictionHistory.length === 0}>
            Download JSON
          </button>
        </div>
        <div className="feature-card">
          <div className="feature-card-icon">
            <FileDown size={22} />
          </div>
          <h3>Multipage PDF report</h3>
          <p>
            Built on the server with ReportLab: session summary, input-performance narrative, Improvement coach (if any), run index, and per-run{' '}
            <strong>full inputs + predictions</strong> (new runs). Appendix lists app areas. Matches light/dark header style. Requires the FastAPI service running (
            <code>uvicorn backend.main:app</code>).
          </p>
          <button type="button" className="btn btn-primary" onClick={exportPdf} disabled={predictionHistory.length === 0 || pdfBusy}>
            {pdfBusy ? 'Building PDF…' : 'Download PDF report'}
          </button>
        </div>
        <div className="feature-card">
          <div className="feature-card-icon feature-card-icon--violet">
            <FileText size={22} />
          </div>
          <h3>Summary</h3>
          <p>
            Total runs: <strong>{predictionHistory.length}</strong>
            <br />
            Pass: <strong>{predictionHistory.filter((r) => r.outcome === 'Pass').length}</strong> · Fail:{' '}
            <strong>{predictionHistory.filter((r) => r.outcome === 'Fail').length}</strong>
          </p>
          <button type="button" className="btn btn-outline" onClick={() => onNavigate('analytics')}>
            Open analytics
          </button>
        </div>
        <div className="feature-card">
          <div className="feature-card-icon feature-card-icon--teal">
            <GraduationCap size={22} />
          </div>
          <h3>Model bundle</h3>
          <p>Trained artifacts live in the project: <code>model.h5</code>, <code>scaler.pkl</code>.</p>
          <span className="feature-card-muted">Train via Jupyter notebook when you update data.</span>
        </div>
      </div>
    </>
  );
}

// ─── Analytics (animated charts) ────────────────────
const DONUT_R = 52;
const DONUT_C = 2 * Math.PI * DONUT_R;

function useChartAnimate(depsKey) {
  const [on, setOn] = useState(false);
  useEffect(() => {
    let id2;
    const id1 = requestAnimationFrame(() => {
      setOn(false);
      id2 = requestAnimationFrame(() => setOn(true));
    });
    return () => {
      cancelAnimationFrame(id1);
      if (id2 != null) cancelAnimationFrame(id2);
    };
  }, [depsKey]);
  return on;
}

/**
 * Seeds the Analytics charts by running the real BPNN against a handful of varied profiles,
 * so a fresh session isn't met with empty cards.
 */
const SAMPLE_PROFILES = [
  { label: 'Balanced', math: 78, sci: 76, eng: 74, hist: 72, comp: 80, att: 0.88, study: 4.5, sleep: 7, stress: 4, rev: 1.5, gpa: 3.2, quiz: 74, asg: 78, proj: 80, exam: 72 },
  { label: 'Academic star', math: 94, sci: 92, eng: 90, hist: 88, comp: 96, att: 0.97, study: 6, sleep: 7, stress: 3, rev: 2.5, gpa: 3.9, quiz: 92, asg: 94, proj: 95, exam: 93 },
  { label: 'Sleep-deprived', math: 72, sci: 68, eng: 70, hist: 66, comp: 74, att: 0.82, study: 5.5, sleep: 4.5, stress: 8, rev: 1, gpa: 2.9, quiz: 70, asg: 68, proj: 66, exam: 65 },
  { label: 'Low attendance', math: 68, sci: 64, eng: 62, hist: 60, comp: 66, att: 0.52, study: 3, sleep: 7, stress: 5, rev: 1, gpa: 2.5, quiz: 60, asg: 58, proj: 62, exam: 55 },
  { label: 'Consistent B+', math: 82, sci: 80, eng: 78, hist: 76, comp: 82, att: 0.9, study: 4, sleep: 7.5, stress: 4, rev: 1.5, gpa: 3.4, quiz: 80, asg: 82, proj: 84, exam: 78 },
  { label: 'Burning out', math: 74, sci: 70, eng: 72, hist: 68, comp: 76, att: 0.78, study: 7, sleep: 5, stress: 9, rev: 2, gpa: 2.8, quiz: 68, asg: 70, proj: 66, exam: 62 },
  { label: 'Weak math', math: 52, sci: 74, eng: 78, hist: 76, comp: 70, att: 0.86, study: 4, sleep: 7, stress: 5, rev: 1.5, gpa: 3.0, quiz: 68, asg: 72, proj: 74, exam: 66 },
  { label: 'Science crush', math: 88, sci: 96, eng: 84, hist: 80, comp: 92, att: 0.95, study: 5.5, sleep: 7, stress: 4, rev: 2, gpa: 3.8, quiz: 90, asg: 88, proj: 92, exam: 90 },
  { label: 'Under pressure', math: 62, sci: 60, eng: 58, hist: 56, comp: 64, att: 0.75, study: 3.5, sleep: 6, stress: 8, rev: 1, gpa: 2.6, quiz: 58, asg: 62, proj: 60, exam: 54 },
  { label: 'Return runner', math: 70, sci: 72, eng: 74, hist: 68, comp: 72, att: 0.83, study: 4.5, sleep: 7, stress: 5, rev: 1.25, gpa: 3.0, quiz: 72, asg: 74, proj: 70, exam: 66 },
];

function sampleProfileToForm(p) {
  return {
    ...INITIAL_FORM,
    math_score: String(p.math),
    science_score: String(p.sci),
    english_score: String(p.eng),
    history_score: String(p.hist),
    computer_score: String(p.comp),
    attendance_rate: String(p.att),
    study_hours_daily: String(p.study),
    sleep_hours: String(p.sleep),
    mental_stress: String(p.stress),
    revision_hours: String(p.rev),
    previous_gpa: String(p.gpa),
    quiz_avg: String(p.quiz),
    assignment_avg: String(p.asg),
    project_score: String(p.proj),
    standardized_exam_score: String(p.exam),
    parent_involvement: '6',
  };
}

async function seedSampleRuns(onRecordPrediction, { spreadHours = 7 } = {}) {
  const total = SAMPLE_PROFILES.length;
  let ok = 0;
  for (let i = 0; i < total; i += 1) {
    const profile = SAMPLE_PROFILES[i];
    const formData = sampleProfileToForm(profile);
    const numeric = formStringsToNumbers(formData);
    let res;
    try {
      res = await axios.post(predictUrl(), numeric, { timeout: 15000 });
    } catch {
      continue;
    }
    const marksSnap = subjectMarksAverageFromFormData(formData);
    const subjectAvgLabel = subjectAverageLabelFromFormData(formData);
    const atDate = new Date(Date.now() - ((total - i) / total) * spreadHours * 60 * 60 * 1000);
    onRecordPrediction({
      id: globalThis.crypto?.randomUUID?.() ?? `demo-${Date.now()}-${i}`,
      atLabel: atDate.toLocaleString(),
      outcome: res.data.prediction,
      confidence: Math.round((res.data.pass_probability ?? res.data.probability ?? 0) * 100),
      composite: marksSnap.allEmpty ? null : marksSnap.composite,
      compositeLabel: subjectAvgLabel,
      formData,
      numeric,
      passProb: typeof res.data.pass_probability === 'number' ? res.data.pass_probability : res.data.probability,
      failProb: typeof res.data.fail_probability === 'number' ? res.data.fail_probability : undefined,
      source: 'sample',
      sampleLabel: profile.label,
    });
    ok += 1;
  }
  return ok;
}

function SampleRunsButton({ onRecordPrediction, hasRuns }) {
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState(null);

  const seed = async () => {
    if (!onRecordPrediction || busy) return;
    setBusy(true);
    setMsg(null);
    try {
      const ok = await seedSampleRuns(onRecordPrediction);
      setMsg(`Seeded ${ok} demo runs. Charts below will populate.`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="analytics-sample-wrap">
      <button
        type="button"
        className="btn btn-outline btn-sm"
        onClick={seed}
        disabled={busy}
        title={hasRuns ? 'Add more demo runs to the session' : 'Populate charts with sample runs scored by the live model'}
      >
        {busy ? 'Seeding…' : hasRuns ? 'Add more sample runs' : 'Load sample runs'}
      </button>
      {msg ? <span className="analytics-sample-msg">{msg}</span> : null}
    </div>
  );
}

function AnalyticsPage({ predictionHistory, lastPredictPayload, onRecordPrediction }) {
  const lineGradId = `lg-${useId().replace(/:/g, '')}`;
  const n = predictionHistory.length;
  const chronological = useMemo(() => [...predictionHistory].reverse(), [predictionHistory]);

  // Auto-seed sample runs whenever we land on Analytics with nothing to chart.
  const seededRef = useRef(false);
  useEffect(() => {
    if (seededRef.current) return;
    if (n > 0) {
      seededRef.current = true;
      return;
    }
    if (!onRecordPrediction) return;
    seededRef.current = true;
    seedSampleRuns(onRecordPrediction).catch(() => {
      seededRef.current = false;
    });
  }, [n, onRecordPrediction]);

  const [calib, setCalib] = useState(undefined);
  useEffect(() => {
    let cancelled = false;
    getCalibration()
      .then((d) => {
        if (!cancelled) setCalib(d);
      })
      .catch(() => {
        if (!cancelled) setCalib({ available: false, reason: 'Could not load calibration (is the API running?)' });
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const pass = predictionHistory.filter((r) => r.outcome === 'Pass').length;
  const fail = n - pass;
  const passPct = n ? Math.round((pass / n) * 100) : 0;

  const confidences = useMemo(
    () =>
      chronological.map((r) =>
        typeof r.confidence === 'number' && Number.isFinite(r.confidence) ? r.confidence : 0,
      ),
    [chronological],
  );
  const composites = useMemo(
    () => chronological.map((r) => (typeof r.composite === 'number' ? r.composite : 0)),
    [chronological],
  );

  const meanConfRaw = n ? confidences.reduce((s, v) => s + v, 0) / n : 0;
  const avgConf = n ? Math.round(meanConfRaw) : 0;
  const minConf = n ? Math.min(...confidences) : 0;
  const maxConf = n ? Math.max(...confidences) : 0;
  const sortedC = useMemo(() => [...confidences].sort((a, b) => a - b), [confidences]);
  const medianConf = n ? sortedC[Math.floor((n - 1) / 2)] : 0;
  const stdConf =
    n > 1
      ? Math.round(
          Math.sqrt(
            confidences.reduce((s, v) => s + (v - meanConfRaw) ** 2, 0) / (n - 1),
          ) * 10,
        ) / 10
      : 0;

  const avgComposite = n
    ? Math.round((composites.reduce((s, v) => s + v, 0) / n) * 10) / 10
    : 0;

  const confBuckets = useMemo(() => {
    const b = [0, 0, 0, 0, 0];
    const labels = ['0–20%', '21–40%', '41–60%', '61–80%', '81–100%'];
    confidences.forEach((c) => {
      const i = Math.min(4, Math.floor(c / 20));
      b[i]++;
    });
    const max = Math.max(1, ...b);
    return b.map((count, i) => ({ label: labels[i], count, pct: (count / max) * 100 }));
  }, [confidences]);

  const linePathRef = useRef(null);
  const animCharts = useChartAnimate(`${n}-${passPct}`);

  const lineGeom = useMemo(() => {
    if (n < 2) return { d: '', points: [] };
    const w = 360;
    const h = 140;
    const pad = { t: 16, r: 12, b: 28, l: 36 };
    const innerW = w - pad.l - pad.r;
    const innerH = h - pad.t - pad.b;
    const pts = confidences.map((c, i) => {
      const x = pad.l + (i / Math.max(1, n - 1)) * innerW;
      const y = pad.t + innerH - (c / 100) * innerH;
      return { x, y, c };
    });
    const d = pts.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x.toFixed(1)} ${p.y.toFixed(1)}`).join(' ');
    return { d, pts, w, h, pad };
  }, [confidences, n]);

  useEffect(() => {
    const el = linePathRef.current;
    if (!el || !lineGeom.d) return;
    const len = el.getTotalLength();
    el.style.strokeDasharray = `${len}`;
    el.style.strokeDashoffset = animCharts ? '0' : `${len}`;
    el.style.transition = animCharts ? 'stroke-dashoffset 1.4s cubic-bezier(0.4, 0, 0.2, 1)' : 'none';
  }, [lineGeom.d, animCharts, n]);

  const donutOffset = animCharts ? DONUT_C * (1 - passPct / 100) : DONUT_C;

  const lastRuns = useMemo(() => {
    const take = Math.min(12, chronological.length);
    return chronological.slice(-take).map((r, i) => {
      const idx = chronological.length - take + i + 1;
      const nm = r.csvMeta?.displayName?.trim();
      const short = nm && nm.length > 20 ? `${nm.slice(0, 20)}…` : nm;
      return {
        key: r.id ?? i,
        label: short || `#${idx}`,
        outcome: r.outcome,
        composite: typeof r.composite === 'number' ? r.composite : 0,
      };
    });
  }, [chronological]);

  let streak = 0;
  if (n > 0) {
    const last = predictionHistory[0]?.outcome;
    for (const r of predictionHistory) {
      if (r.outcome === last) streak++;
      else break;
    }
  }

  return (
    <>
      <div className="page-hero">
        <div className="page-title-row page-title-row--flush">
          <div>
            <h1>Analytics</h1>
            <p>Animated charts and detailed metrics from your session runs (newest listed first in records, charts use chronological order).</p>
          </div>
          <SampleRunsButton onRecordPrediction={onRecordPrediction} hasRuns={n > 0} />
        </div>
      </div>

      <div className={`analytics-dashboard ${animCharts ? 'analytics-dashboard--live' : ''}`}>
        <div className="analytics-kpi-strip">
          <div className="analytics-kpi-tile">
            <span className="analytics-kpi-label">Runs</span>
            <strong className="analytics-kpi-val">{n}</strong>
          </div>
          <div className="analytics-kpi-tile">
            <span className="analytics-kpi-label">Avg confidence</span>
            <strong className="analytics-kpi-val">{n ? `${avgConf}%` : '—'}</strong>
          </div>
          <div className="analytics-kpi-tile">
            <span className="analytics-kpi-label">Median</span>
            <strong className="analytics-kpi-val">{n ? `${medianConf}%` : '—'}</strong>
          </div>
          <div className="analytics-kpi-tile">
            <span className="analytics-kpi-label">σ confidence</span>
            <strong className="analytics-kpi-val">{n > 1 ? stdConf : '—'}</strong>
          </div>
          <div className="analytics-kpi-tile">
            <span className="analytics-kpi-label">Avg subject marks</span>
            <strong className="analytics-kpi-val">{n ? `${avgComposite}` : '—'}</strong>
          </div>
          <div className="analytics-kpi-tile">
            <span className="analytics-kpi-label">Current streak</span>
            <strong className="analytics-kpi-val">{n ? `${streak}× ${predictionHistory[0]?.outcome ?? ''}` : '—'}</strong>
          </div>
        </div>

        <div className="panel panel--elevated analytics-panel-wide" style={{ marginBottom: 16 }}>
          <div className="panel-header">
            <div>
              <h3>Model calibration (training sample)</h3>
              <p>
                Empirical pass rate by predicted-probability decile on <code>student_data.csv</code> (in-sample; first load may take a few seconds).
              </p>
            </div>
          </div>
          <div className="panel-body">
            {calib === undefined && <p className="analytics-empty">Loading calibration…</p>}
            {calib?.available && (
              <>
                <p style={{ marginBottom: 12, fontSize: 13, opacity: 0.9 }}>
                  Scored rows: <strong>{calib.n_scored}</strong>. {calib.note}
                </p>
                <div style={{ overflowX: 'auto' }}>
                  <table className="analytics-calib-table">
                    <thead>
                      <tr>
                        <th>Decile</th>
                        <th>p range</th>
                        <th>n</th>
                        <th>Empirical pass</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(calib.deciles || []).map((row) => (
                        <tr key={row.decile}>
                          <td>{row.decile}</td>
                          <td>
                            {row.p_range?.[0]}–{row.p_range?.[1]}
                          </td>
                          <td>{row.n}</td>
                          <td>
                            {row.empirical_pass_rate != null ? `${(row.empirical_pass_rate * 100).toFixed(1)}%` : '—'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            )}
            {calib && !calib.available && (
              <p className="analytics-empty">{calib.reason || 'Calibration unavailable.'}</p>
            )}
          </div>
        </div>

        <div className="analytics-charts-grid">
          <div className="panel panel--elevated analytics-panel-wide">
            <div className="panel-header">
              <div>
                <h3>Outcome balance</h3>
                <p>Animated share of Pass vs Fail (session total)</p>
              </div>
            </div>
            <div className="panel-body analytics-panel-body">
              {n === 0 ? (
                <p className="analytics-empty">Run predictions to populate charts.</p>
              ) : (
                <div className="analytics-split-charts">
                  <div className="donut-chart donut-chart--lg">
                    <svg viewBox="0 0 120 120" className="donut-svg">
                      <circle className="donut-track" cx="60" cy="60" r={DONUT_R} fill="none" strokeWidth="14" />
                      <circle
                        className="donut-fill donut-fill--animated"
                        cx="60"
                        cy="60"
                        r={DONUT_R}
                        fill="none"
                        strokeWidth="14"
                        strokeLinecap="round"
                        strokeDasharray={`${DONUT_C} ${DONUT_C}`}
                        strokeDashoffset={donutOffset}
                        transform="rotate(-90 60 60)"
                        style={{
                          transition: 'stroke-dashoffset 1.25s cubic-bezier(0.34, 1.2, 0.64, 1)',
                        }}
                      />
                    </svg>
                    <div className="donut-label">
                      <strong>{passPct}%</strong>
                      <span>pass rate</span>
                    </div>
                  </div>
                  <div className="outcome-bars-animated">
                    <div className="outcome-bar-block">
                      <span className="outcome-bar-title">Pass</span>
                      <div className="outcome-bar-track">
                        <div
                          className="outcome-bar-fill outcome-bar-fill--pass"
                          style={{
                            transform: animCharts ? 'scaleY(1)' : 'scaleY(0)',
                            transformOrigin: 'bottom',
                            transition: 'transform 0.9s cubic-bezier(0.34, 1.3, 0.64, 1) 0.15s',
                            height: `${n ? (pass / n) * 100 : 0}%`,
                          }}
                        />
                      </div>
                      <span className="outcome-bar-count">{pass}</span>
                    </div>
                    <div className="outcome-bar-block">
                      <span className="outcome-bar-title">Fail</span>
                      <div className="outcome-bar-track">
                        <div
                          className="outcome-bar-fill outcome-bar-fill--fail"
                          style={{
                            transform: animCharts ? 'scaleY(1)' : 'scaleY(0)',
                            transformOrigin: 'bottom',
                            transition: 'transform 0.9s cubic-bezier(0.34, 1.3, 0.64, 1) 0.28s',
                            height: `${n ? (fail / n) * 100 : 0}%`,
                          }}
                        />
                      </div>
                      <span className="outcome-bar-count">{fail}</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>

          <div className="panel panel--elevated analytics-panel-wide">
            <div className="panel-header">
              <div>
                <h3>Confidence trajectory</h3>
                <p>Model confidence (%) across runs in time order — line draws on load</p>
              </div>
            </div>
            <div className="panel-body analytics-panel-body">
              {n < 2 ? (
                <p className="analytics-empty">Need at least two predictions to plot a trend.</p>
              ) : (
                <div className="line-chart-wrap">
                  <svg
                    viewBox={`0 0 ${lineGeom.w} ${lineGeom.h}`}
                    className="line-chart-svg"
                    preserveAspectRatio="xMidYMid meet"
                  >
                    <defs>
                      <linearGradient id={lineGradId} x1="0%" y1="0%" x2="0%" y2="100%">
                        <stop offset="0%" stopColor="#3b82f6" stopOpacity="0.35" />
                        <stop offset="100%" stopColor="#3b82f6" stopOpacity="0.02" />
                      </linearGradient>
                    </defs>
                    {[0, 25, 50, 75, 100].map((t) => {
                      const y =
                        lineGeom.pad.t +
                        (lineGeom.h - lineGeom.pad.t - lineGeom.pad.b) * (1 - t / 100);
                      return (
                        <g key={t}>
                          <line
                            x1={lineGeom.pad.l}
                            y1={y}
                            x2={lineGeom.w - lineGeom.pad.r}
                            y2={y}
                            stroke="var(--border-light)"
                            strokeWidth="1"
                          />
                          <text x={4} y={y + 4} fontSize="9" fill="#94a3b8">
                            {t}
                          </text>
                        </g>
                      );
                    })}
                    <path
                      d={`${lineGeom.d} L ${lineGeom.pts[lineGeom.pts.length - 1]?.x.toFixed(1)} ${lineGeom.pad.t + (lineGeom.h - lineGeom.pad.t - lineGeom.pad.b)} L ${lineGeom.pts[0]?.x.toFixed(1)} ${lineGeom.pad.t + (lineGeom.h - lineGeom.pad.t - lineGeom.pad.b)} Z`}
                      fill={`url(#${lineGradId})`}
                      className="line-area-path"
                      opacity={animCharts ? 0.9 : 0}
                      style={{ transition: 'opacity 0.6s ease 0.5s' }}
                    />
                    <path
                      ref={linePathRef}
                      d={lineGeom.d}
                      fill="none"
                      stroke="#2563eb"
                      strokeWidth="2.5"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      className="line-stroke-path"
                    />
                    {lineGeom.pts.map((p, i) => (
                      <circle
                        key={i}
                        cx={p.x}
                        cy={p.y}
                        r={4}
                        className="line-dot"
                        fill="#fff"
                        stroke="#2563eb"
                        strokeWidth="2"
                        opacity={animCharts ? 1 : 0}
                        style={{
                          transition: `opacity 0.35s ease ${0.6 + i * 0.04}s`,
                        }}
                      />
                    ))}
                  </svg>
                  <div className="line-chart-legend">
                    <span>
                      Min <strong>{minConf}%</strong>
                    </span>
                    <span>
                      Max <strong>{maxConf}%</strong>
                    </span>
                  </div>
                </div>
              )}
            </div>
          </div>

          <div className="panel panel--elevated">
            <div className="panel-header">
              <div>
                <h3>Confidence distribution</h3>
                <p>How often the model sat in each band</p>
              </div>
            </div>
            <div className="panel-body analytics-hist-body">
              {n === 0 ? (
                <p className="analytics-empty">No data.</p>
              ) : (
                <div className="hist-chart">
                  {confBuckets.map((bucket, i) => (
                    <div key={bucket.label} className="hist-col">
                      <div className="hist-col-track">
                        <div
                          className="hist-col-fill"
                          style={{
                            height: animCharts ? `${bucket.pct}%` : '0%',
                            transition: `height 0.85s cubic-bezier(0.34, 1.2, 0.64, 1) ${0.06 * i}s`,
                          }}
                          title={`${bucket.count} run(s)`}
                        />
                      </div>
                      <span className="hist-col-count">{bucket.count}</span>
                      <span className="hist-col-label">{bucket.label}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          <div className="panel panel--elevated">
            <div className="panel-header">
              <div>
                <h3>Avg subject marks by run</h3>
                <p>Last {lastRuns.length} runs — avg subject marks % (0–100)</p>
              </div>
            </div>
            <div className="panel-body analytics-composite-body">
              {lastRuns.length === 0 ? (
                <p className="analytics-empty">No runs.</p>
              ) : (
                <div className="composite-bars">
                  {lastRuns.map((run, i) => (
                    <div key={run.key} className="composite-row">
                      <span className="composite-row-label">{run.label}</span>
                      <div className="composite-row-track">
                        <div
                          className={`composite-row-fill ${
                            run.outcome === 'Pass'
                              ? 'composite-row-fill--pass'
                              : run.outcome === 'Fail'
                                ? 'composite-row-fill--fail'
                                : ''
                          }`}
                          style={{
                            width: animCharts ? `${Math.min(100, run.composite)}%` : '0%',
                            transition: `width 0.7s cubic-bezier(0.34, 1.2, 0.64, 1) ${0.05 * i}s`,
                            ...(run.outcome !== 'Pass' && run.outcome !== 'Fail'
                              ? { background: 'var(--border)', opacity: 0.85 }
                              : {}),
                          }}
                        />
                      </div>
                      <span className="composite-row-val">{run.composite.toFixed(1)}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          <div className="panel panel--elevated analytics-insights-panel">
            <div className="panel-header">
              <div>
                <h3>Session insights</h3>
                <p>Auto-generated from your numbers</p>
              </div>
            </div>
            <div className="panel-body">
              <ul className="insights-list">
                {n === 0 && <li>Run the Performance Predictor to unlock trajectory, histogram, and subject-average views.</li>}
                {n > 0 && passPct >= 70 && (
                  <li>
                    <strong>Skew Pass-heavy:</strong> {passPct}% of runs predicted Pass — check inputs if you expect more balance.
                  </li>
                )}
                {n > 0 && passPct <= 30 && (
                  <li>
                    <strong>Skew Fail-heavy:</strong> only {passPct}% Pass — low attendance or subject averages often correlate.
                  </li>
                )}
                {n > 1 && stdConf < 8 && (
                  <li>
                    <strong>Stable confidence:</strong> low spread (σ ≈ {stdConf}) — the model is similarly sure each time.
                  </li>
                )}
                {n > 1 && stdConf >= 15 && (
                  <li>
                    <strong>Variable confidence:</strong> σ ≈ {stdConf} — inputs span easier vs harder cases.
                  </li>
                )}
                {n >= 2 && confidences[n - 1] > confidences[n - 2] && (
                  <li>
                    <strong>Latest run:</strong> confidence rose vs previous ({confidences[n - 2]}% → {confidences[n - 1]}%).
                  </li>
                )}
                {n >= 2 && confidences[n - 1] < confidences[n - 2] && (
                  <li>
                    <strong>Latest run:</strong> confidence fell vs previous ({confidences[n - 2]}% → {confidences[n - 1]}%).
                  </li>
                )}
                {streak >= 3 && (
                  <li>
                    <strong>Streak:</strong> last {streak} predictions were all <strong>{predictionHistory[0]?.outcome}</strong>.
                  </li>
                )}
              </ul>
            </div>
          </div>
        </div>

        <NetworkDiagram lastPredictPayload={lastPredictPayload} />

        <AnalyticsExtraGraphs predictionHistory={predictionHistory} />
      </div>
    </>
  );
}

// ─── Settings ─────────────────────────────────────────
function SettingsPage({ onReplayLanding }) {
  const [apiStatus, setApiStatus] = useState(null);
  const [checking, setChecking] = useState(false);

  const checkApi = async () => {
    setChecking(true);
    setApiStatus(null);
    try {
      const { data } = await axios.get(`${apiBaseUrl()}/health`, { timeout: 8000 });
      setApiStatus({ ok: true, msg: data?.status === 'ok' ? 'API reachable' : JSON.stringify(data) });
    } catch (e) {
      setApiStatus({ ok: false, msg: e.message || 'Unreachable' });
    } finally {
      setChecking(false);
    }
  };

  return (
    <>
      <div className="page-hero">
        <div className="page-title-row page-title-row--flush">
          <div>
            <h1>Settings</h1>
            <p>Connectivity and workspace preferences</p>
          </div>
        </div>
      </div>

      <div className="settings-stack">
        {onReplayLanding && (
          <div className="panel panel--elevated landing-replay-panel">
            <div className="panel-header">
              <div>
                <h3>
                  <Sparkles size={16} style={{ display: 'inline', verticalAlign: '-2px', marginRight: 6 }} />
                  Animated intro
                </h3>
                <p>Replay the full-screen landing with scroll animations and project story.</p>
              </div>
              <button type="button" className="btn btn-outline btn-sm" onClick={() => onReplayLanding()}>
                Replay landing
              </button>
            </div>
            <div className="panel-body">
              <p className="settings-hint">Does not change your theme or prediction history.</p>
            </div>
          </div>
        )}

        <div className="panel panel--elevated">
          <div className="panel-header">
            <div>
              <h3>
                <Shield size={16} style={{ display: 'inline', verticalAlign: '-2px', marginRight: 6 }} />
                Backend API
              </h3>
              <p>FastAPI default: {apiBaseUrl()}</p>
            </div>
            <button type="button" className="btn btn-outline btn-sm" onClick={checkApi} disabled={checking}>
              <RefreshCw size={14} className={checking ? 'spin' : ''} /> {checking ? 'Checking…' : 'Test connection'}
            </button>
          </div>
          <div className="panel-body">
            {apiStatus && (
              <div className={`settings-banner ${apiStatus.ok ? 'settings-banner--ok' : 'settings-banner--err'}`}>
                {apiStatus.ok ? <CheckCircle2 size={18} /> : <AlertCircle size={18} />}
                {apiStatus.msg}
              </div>
            )}
            <p className="settings-hint">
              Start the server from the project root: <code>python -m uvicorn backend.main:app --port 8000</code>
            </p>
          </div>
        </div>

        <div className="panel panel--elevated">
          <div className="panel-header">
            <div>
              <h3>Frontend</h3>
              <p>Vite dev proxy maps <code>/api</code> → backend.</p>
            </div>
          </div>
          <div className="panel-body">
            <p className="settings-hint">
              Production: set <code>VITE_API_URL</code> to your API origin. Theme toggle lives in the header.
            </p>
            <p className="settings-hint" style={{ marginTop: 12 }}>
              Docker: from the project root, <code>docker compose up --build</code> serves the UI on port <strong>8080</strong> and proxies{' '}
              <code>/api</code> to the API container (CORS includes <code>http://localhost:8080</code>).
            </p>
          </div>
        </div>
      </div>
    </>
  );
}

// ─── Help ─────────────────────────────────────────────
function HelpPage({ onNavigate }) {
  const items = [
    {
      q: 'How do I run a prediction?',
      a: 'Open Performance Predictor, fill academic and behavioral fields, then Run prediction. Results log to Student Records and the dashboard.',
    },
    {
      q: 'Where does overall marks come from?',
      a: 'The academic bar chart is a visual breakdown only. Average subject marks (mean of entered subject scores) is for display. The API uses student_marks_percent = 0 so previous_marks comes from the mean of subject scores, not from the chart.',
    },
    {
      q: 'Why did my prediction fail?',
      a: 'Ensure the API is running (Settings → Test connection) and model.h5 / scaler.pkl exist in the project folder.',
    },
    {
      q: 'How does Improvement coach stay accurate?',
      a: 'It loads your last successful prediction snapshot (the same form values you submitted). Rules reference those numbers and the features the BPNN actually uses, so fill fields honestly and re-run after you change habits.',
    },
    {
      q: 'What is Study timetable?',
      a: 'Sidebar → Study timetable builds a sample week from that same last snapshot: it ranks subject scores (weaker subjects first), uses your study, revision, and sleep hours, and adds notes for stress and attendance. Run Predictor again after you change habits to refresh the plan.',
    },
    {
      q: 'Where can I read how EduPredict is built?',
      a: 'Open Settings → Replay landing for the animated intro (stack, pipeline, model). You can skip it anytime or choose not to show it again on future visits.',
    },
    {
      q: 'What is Innovation Labs?',
      a: 'Sidebar → Innovation Labs: local sensitivity (model autopsy), adversary challenge copy, counterfactual “what-if” mirrors, peer-shadow percentiles vs training data, export manifest nonce, and session narrative on the Dashboard. Run at least one prediction first so Labs can reuse your last payload.',
    },
  ];

  return (
    <>
      <div className="page-hero">
        <div className="page-title-row page-title-row--flush">
          <div>
            <h1>Help & Support</h1>
            <p>Quick answers for EduPredict + BPNN</p>
          </div>
          <button type="button" className="btn btn-primary" onClick={() => onNavigate('predict')}>
            Open predictor <ExternalLink size={14} />
          </button>
        </div>
      </div>

      <div className="help-faq">
        {items.map((item) => (
          <details key={item.q} className="help-faq-item">
            <summary>{item.q}</summary>
            <p>{item.a}</p>
          </details>
        ))}
      </div>
    </>
  );
}

// ─── Animated landing (full-screen, before main app) ──
const LANDING_TITLE = 'EduPredict';

function LandingExperience({ onEnterApp, darkMode, onToggleTheme }) {
  const [dontShowAgain, setDontShowAgain] = useState(false);

  useEffect(() => {
    const panels = document.querySelectorAll('.landing-panel:not(.landing-hero)');
    const io = new IntersectionObserver(
      (entries) => {
        entries.forEach((en) => {
          if (en.isIntersecting) en.target.classList.add('landing-section--in');
        });
      },
      { threshold: 0.14, rootMargin: '0px 0px -6% 0px' },
    );
    panels.forEach((p) => io.observe(p));
    return () => io.disconnect();
  }, []);

  const enter = (persistSkip) => {
    if (persistSkip) {
      try {
        localStorage.setItem(LANDING_SKIP_STORAGE_KEY, '1');
      } catch {
        /* ignore quota / private mode */
      }
    }
    onEnterApp();
  };

  return (
    <div className={`landing-experience${darkMode ? '' : ' landing-experience--light'}`}>
      <div className="landing-bg-grid" aria-hidden />
      <div className="landing-orb landing-orb--1" aria-hidden />
      <div className="landing-orb landing-orb--2" aria-hidden />
      <div className="landing-orb landing-orb--3" aria-hidden />
      <div className="landing-noise" aria-hidden />

      <header className="landing-topbar">
        <div className="landing-brand">
          <span className="landing-brand-icon">
            <Sparkles size={18} strokeWidth={2} />
          </span>
          <span className="landing-brand-text">EduPredict</span>
        </div>
        <div className="landing-topbar-actions">
          <button type="button" className="landing-skip" onClick={() => enter(false)}>
            Skip intro
          </button>
          <button
            type="button"
            className="landing-theme-btn"
            onClick={onToggleTheme}
            title={darkMode ? 'Light mode' : 'Dark mode'}
            aria-label={darkMode ? 'Light mode' : 'Dark mode'}
          >
            {darkMode ? <Sun size={18} /> : <Moon size={18} />}
          </button>
        </div>
      </header>

      <main className="landing-scroll">
        <section className="landing-panel landing-hero landing-section--in">
          <div className="landing-hero-inner">
            <p className="landing-eyebrow landing-reveal-child" style={{ '--lr': 0 }}>
              <Zap size={14} strokeWidth={2} className="landing-eyebrow-icon" aria-hidden />
              Backpropagation neural network · Pass / fail insight
            </p>
            <h1 className="landing-hero-title" aria-label={LANDING_TITLE}>
              {LANDING_TITLE.split('').map((ch, i) => (
                <span key={`${ch}-${i}`} className="landing-char" style={{ '--ci': i }}>
                  {ch}
                </span>
              ))}
            </h1>
            <p className="landing-hero-sub landing-reveal-child" style={{ '--lr': 1 }}>
              A cinematic student-success workspace: rich forms, live charts, session analytics, and an improvement coach — powered by{' '}
              <strong>TensorFlow</strong> + <strong>FastAPI</strong> + artifacts from your Jupyter pipeline.
            </p>
            <div className="landing-hero-stats landing-reveal-child" style={{ '--lr': 2 }}>
              <div className="landing-stat">
                <strong>BPNN</strong>
                <span>Keras inference</span>
              </div>
              <div className="landing-stat">
                <strong>7 → N</strong>
                <span>Features + index</span>
              </div>
              <div className="landing-stat">
                <strong>React</strong>
                <span>Vite + motion UI</span>
              </div>
            </div>
          </div>
          <div className="landing-scroll-cue landing-reveal-child" style={{ '--lr': 3 }} aria-hidden>
            <span>Scroll</span>
            <ChevronDown size={22} className="landing-scroll-cue-icon" strokeWidth={2} />
          </div>
        </section>

        <section className="landing-panel landing-crafted">
          <div className="landing-section-head landing-reveal-child" style={{ '--lr': 0 }}>
            <span className="landing-kicker">How it’s made</span>
            <h2>Three layers, one story</h2>
            <p>From notebook to browser — every step stays aligned with <code>backend/inference.py</code>.</p>
          </div>
          <div className="landing-cards-3">
            <article className="landing-card landing-reveal-child" style={{ '--lr': 1 }}>
              <div className="landing-card-icon landing-card-icon--blue">
                <Code2 size={24} strokeWidth={1.75} />
              </div>
              <h3>Interface</h3>
              <p>
                React 18 &amp; Vite. Dev proxy sends <code>/api</code> to Uvicorn. Production uses <code>VITE_API_URL</code> when you deploy the API
                separately.
              </p>
            </article>
            <article className="landing-card landing-reveal-child" style={{ '--lr': 2 }}>
              <div className="landing-card-icon landing-card-icon--violet">
                <Server size={24} strokeWidth={1.75} />
              </div>
              <h3>Service</h3>
              <p>
                FastAPI <code>POST /predict</code> maps your JSON into training columns, engineers engagement &amp; stress×sleep, builds the
                performance index, scales, and runs the network.
              </p>
            </article>
            <article className="landing-card landing-reveal-child" style={{ '--lr': 3 }}>
              <div className="landing-card-icon landing-card-icon--amber">
                <Database size={24} strokeWidth={1.75} />
              </div>
              <h3>Artifacts</h3>
              <p>
                <code>model.h5</code> + <code>scaler.pkl</code> (scaler, column order, min/max bounds, index weights). Train and export from{' '}
                <code>student_performance_bpnn.ipynb</code>.
              </p>
            </article>
          </div>
        </section>

        <section className="landing-panel landing-pipeline">
          <div className="landing-section-head landing-reveal-child" style={{ '--lr': 0 }}>
            <span className="landing-kicker">Model pipeline</span>
            <h2>Data flows like this</h2>
          </div>
          <div className="landing-pipeline-track landing-reveal-child" style={{ '--lr': 1 }}>
            <div className="landing-pipeline-node">
              <span className="landing-pipeline-dot" />
              Form JSON
            </div>
            <div className="landing-pipeline-line" aria-hidden />
            <div className="landing-pipeline-node">
              <span className="landing-pipeline-dot" />
              7 base fields
            </div>
            <div className="landing-pipeline-line" aria-hidden />
            <div className="landing-pipeline-node">
              <span className="landing-pipeline-dot" />
              + engagement · stress_sleep
            </div>
            <div className="landing-pipeline-line" aria-hidden />
            <div className="landing-pipeline-node">
              <span className="landing-pipeline-dot" />
              performance_index
            </div>
            <div className="landing-pipeline-line" aria-hidden />
            <div className="landing-pipeline-node landing-pipeline-node--out">
              <Brain size={20} strokeWidth={1.75} />
              BPNN → p(Pass)
            </div>
          </div>
          <p className="landing-pipeline-note landing-reveal-child" style={{ '--lr': 2 }}>
            <code>previous_marks</code> uses the mean of subject scores (composite display does not override this). Threshold 0.5 labels Pass vs
            Fail.
          </p>
        </section>

        <section className="landing-panel landing-suite">
          <div className="landing-section-head landing-reveal-child" style={{ '--lr': 0 }}>
            <span className="landing-kicker">Inside the app</span>
            <h2>Everything after this screen</h2>
          </div>
          <div className="landing-feature-grid">
            {[
              { Icon: Brain, t: 'Predictor', d: 'Subject avg % (display), behavioral budget, BPNN run.' },
              { Icon: BarChart3, t: 'Analytics', d: 'Animated charts from your session history.' },
              { Icon: Lightbulb, t: 'Coach', d: 'Personalized levers from your last snapshot.' },
              { Icon: Users, t: 'Records', d: 'Log, search, export JSON for reports.' },
            ].map((row, i) => {
              const TileIcon = row.Icon;
              return (
              <div key={row.t} className="landing-feature-tile landing-reveal-child" style={{ '--lr': i + 1 }}>
                <TileIcon size={22} strokeWidth={1.75} className="landing-feature-ic" />
                <strong>{row.t}</strong>
                <span>{row.d}</span>
              </div>
              );
            })}
          </div>
        </section>

        <section className="landing-panel landing-cta">
          <div className="landing-cta-card landing-reveal-child" style={{ '--lr': 0 }}>
            <Workflow size={28} strokeWidth={1.5} className="landing-cta-icon" aria-hidden />
            <h2>Enter the workspace</h2>
            <p>Open the dashboard, run predictions, and explore analytics. You can replay this intro anytime from Settings.</p>
            <label className="landing-checkbox">
              <input type="checkbox" checked={dontShowAgain} onChange={(e) => setDontShowAgain(e.target.checked)} />
              <span>Don’t show this intro on future visits</span>
            </label>
            <button type="button" className="landing-cta-btn" onClick={() => enter(dontShowAgain)}>
              Launch EduPredict
              <ArrowRight size={20} strokeWidth={2} />
            </button>
          </div>
        </section>
      </main>
    </div>
  );
}

// Text fields so decimals like "0." can be typed freely (controlled number inputs fight the keyboard).
const DECIMAL_TYPING = /^-?\d*\.?\d*$/;

const INITIAL_FORM = {
  attendance_rate: '', assignment_avg: '', quiz_avg: '',
  project_score: '', previous_gpa: '',
  study_hours_daily: '',
  revision_hours: '', sleep_hours: '', mental_stress: '',
  sleep_quality: '', standardized_exam_score: '',
  learning_efficiency: '', stress_index: '',
  physical_activity: '', lms_login_frequency: '',
  coding_practice_hours: '', digital_literacy: '',
  parent_involvement: '6',
  math_score: '', science_score: '', english_score: '',
  history_score: '',   computer_score: '',
};

/** Quick-fill scenarios — merges into INITIAL_FORM (empty fields reset). */
const SCENARIO_PRESETS = {
  balanced: {
    label: 'Balanced',
    patch: {
      attendance_rate: '0.85',
      previous_gpa: '3.2',
      assignment_avg: '75',
      quiz_avg: '72',
      project_score: '78',
      standardized_exam_score: '70',
      study_hours_daily: '3',
      sleep_hours: '7',
      mental_stress: '5',
      math_score: '74',
      science_score: '72',
      english_score: '76',
      history_score: '70',
      computer_score: '75',
      parent_involvement: '7',
    },
  },
  atRisk: {
    label: 'At-risk',
    patch: {
      attendance_rate: '0.55',
      previous_gpa: '2.1',
      assignment_avg: '58',
      quiz_avg: '55',
      project_score: '52',
      standardized_exam_score: '50',
      study_hours_daily: '1.5',
      sleep_hours: '5.5',
      mental_stress: '8',
      math_score: '52',
      science_score: '50',
      english_score: '55',
      history_score: '48',
      computer_score: '50',
      parent_involvement: '4',
    },
  },
  strong: {
    label: 'Strong',
    patch: {
      attendance_rate: '0.94',
      previous_gpa: '3.7',
      assignment_avg: '88',
      quiz_avg: '86',
      project_score: '90',
      standardized_exam_score: '85',
      study_hours_daily: '4',
      sleep_hours: '8',
      mental_stress: '3',
      math_score: '88',
      science_score: '86',
      english_score: '87',
      history_score: '84',
      computer_score: '89',
      parent_involvement: '9',
    },
  },
  grind: {
    label: 'Grind',
    patch: {
      attendance_rate: '0.78',
      previous_gpa: '3.0',
      assignment_avg: '70',
      quiz_avg: '68',
      project_score: '72',
      standardized_exam_score: '65',
      study_hours_daily: '6',
      sleep_hours: '5.5',
      mental_stress: '7',
      math_score: '70',
      science_score: '68',
      english_score: '72',
      history_score: '66',
      computer_score: '71',
      parent_involvement: '6',
    },
  },
};

const GOAL_SEEK_UI_FIELDS = [
  { value: 'study_hours_daily', label: 'Study hours / day' },
  { value: 'sleep_hours', label: 'Sleep hours' },
  { value: 'mental_stress', label: 'Mental stress' },
  { value: 'attendance_rate', label: 'Attendance (0–1)' },
  { value: 'assignment_avg', label: 'Assignment avg' },
  { value: 'parent_involvement', label: 'Parent involvement' },
];

/**
 * Sparse JSON for POST /predict: only keys the user filled are sent.
 * Omitted features use training medians from scaler.pkl (v2 tabular model).
 */
function formStringsToNumbers(data) {
  const out = {};
  const isProvided = (k) => {
    const v = data[k];
    return v !== '' && v !== undefined && v !== null && v !== '-' && v !== '.';
  };
  for (const key of Object.keys(INITIAL_FORM)) {
    if (!isProvided(key)) continue;
    const raw = data[key];
    const n = typeof raw === 'string' ? parseFloat(raw) : Number(raw);
    if (!Number.isFinite(n)) continue;
    out[key] = n;
  }
  if (out.attendance_rate !== undefined) {
    if (out.attendance_rate > 1 && out.attendance_rate <= 100) {
      out.attendance_rate = out.attendance_rate / 100;
    }
    out.attendance_rate = Math.min(1, Math.max(0, out.attendance_rate));
  }
  if (out.sleep_hours !== undefined) out.sleep_hours = Math.min(24, Math.max(0, out.sleep_hours));
  if (out.study_hours_daily !== undefined) out.study_hours_daily = Math.min(24, Math.max(0, out.study_hours_daily));
  if (out.revision_hours !== undefined) out.revision_hours = Math.min(24, Math.max(0, out.revision_hours));
  if (out.mental_stress !== undefined) out.mental_stress = Math.min(10, Math.max(0, out.mental_stress));
  if (out.sleep_quality !== undefined) out.sleep_quality = Math.min(10, Math.max(0, out.sleep_quality));
  if (out.digital_literacy !== undefined) out.digital_literacy = Math.min(10, Math.max(0, out.digital_literacy));
  if (out.parent_involvement !== undefined) out.parent_involvement = Math.min(10, Math.max(0, out.parent_involvement));
  if (out.learning_efficiency !== undefined) out.learning_efficiency = Math.min(1, Math.max(0, out.learning_efficiency));
  if (out.physical_activity !== undefined) out.physical_activity = Math.min(40, Math.max(0, out.physical_activity));
  if (out.lms_login_frequency !== undefined) out.lms_login_frequency = Math.min(50, Math.max(0, out.lms_login_frequency));
  if (out.coding_practice_hours !== undefined) out.coding_practice_hours = Math.min(60, Math.max(0, out.coding_practice_hours));
  const subjectKeys = ['math_score', 'science_score', 'english_score', 'history_score', 'computer_score'];
  const proxyKeys = ['assignment_avg', 'quiz_avg', 'project_score', 'standardized_exam_score'];
  const markCandidates = [];
  for (const k of subjectKeys) {
    if (isProvided(k)) {
      const raw = data[k];
      const v = typeof raw === 'string' ? parseFloat(raw) : Number(raw);
      if (Number.isFinite(v)) markCandidates.push(Math.min(100, Math.max(0, v)));
    }
  }
  if (markCandidates.length === 0) {
    for (const k of proxyKeys) {
      if (!isProvided(k)) continue;
      const raw = data[k];
      const v = typeof raw === 'string' ? parseFloat(raw) : Number(raw);
      if (Number.isFinite(v)) markCandidates.push(Math.min(100, Math.max(0, v)));
    }
    if (isProvided('previous_gpa')) {
      const g = typeof data.previous_gpa === 'string' ? parseFloat(data.previous_gpa) : Number(data.previous_gpa);
      if (Number.isFinite(g)) markCandidates.push(Math.min(100, Math.max(0, (g / 4) * 100)));
    }
  }
  if (markCandidates.length > 0) {
    out.student_marks_percent = markCandidates.reduce((a, b) => a + b, 0) / markCandidates.length;
  }
  return out;
}

const TIMETABLE_DAY_NAMES = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];

/** Build a printable weekly rhythm from the last predictor snapshot (weak subjects + study/sleep/revision). */
function buildStudyTimetable(snapshot) {
  if (!snapshot?.formData) return null;
  const n = formStringsToNumbers(snapshot.formData);
  const subjectDefs = [
    { key: 'math_score', label: 'Mathematics' },
    { key: 'science_score', label: 'Science' },
    { key: 'english_score', label: 'English' },
    { key: 'history_score', label: 'History' },
    { key: 'computer_score', label: 'Computer Science' },
  ];
  let subjects = subjectDefs
    .map((s) => ({ label: s.label, score: n[s.key] || 0 }))
    .filter((s) => s.score > 0);
  if (subjects.length === 0) {
    subjects = subjectDefs.map((s) => ({ label: s.label, score: 55 }));
  }
  subjects.sort((a, b) => a.score - b.score);
  const focusOrder = subjects.map((s) => s.label);

  let study = n.study_hours_daily > 0 ? n.study_hours_daily : 2;
  let revision = n.revision_hours >= 0 ? n.revision_hours : 1;
  let sleep = n.sleep_hours > 0 ? n.sleep_hours : 7;
  study = Math.min(10, Math.max(0.5, study));
  revision = Math.min(8, Math.max(0, revision));
  sleep = Math.min(12, Math.max(5, sleep));
  const total = study + revision + sleep;
  if (total > 23.5) {
    const k = 23 / total;
    study *= k;
    revision *= k;
    sleep *= k;
  }

  const stress = n.mental_stress;
  const stressNote =
    stress >= 8
      ? 'High stress in your profile — keep sleep protected; shorten blocks if you feel drained.'
      : stress >= 6
        ? 'Moderate stress — add short breaks between study blocks.'
        : 'Use breaks to stay fresh; adjust times to match your real school schedule.';

  const fmtEnd = (hStart, mStart, durMins) => {
    let t = hStart * 60 + mStart + durMins;
    if (t >= 24 * 60) t = 24 * 60 - 1;
    const eh = Math.floor(t / 60);
    const em = Math.round(t % 60);
    return `${String(eh).padStart(2, '0')}:${String(em).padStart(2, '0')}`;
  };
  const range = (h1, m1, h2, m2) =>
    `${String(h1).padStart(2, '0')}:${String(m1).padStart(2, '0')} – ${String(h2).padStart(2, '0')}:${String(m2).padStart(2, '0')}`;

  const bedM = 22 * 60 + 30;
  const wakeTotalM = (bedM + Math.round(sleep * 60)) % (24 * 60);
  const wakeHh = Math.floor(wakeTotalM / 60);
  const wakeMm = wakeTotalM % 60;
  const sleepStartH = 22;
  const sleepStartM = 30;

  const days = TIMETABLE_DAY_NAMES.map((name, di) => {
    const isWeekend = di >= 5;
    const dayStudyH = isWeekend ? Math.max(1, study * 0.55) : study;
    const dayRevH = isWeekend ? Math.max(0.5, revision * 0.45) : revision * (di % 2 === 0 ? 0.55 : 0.45);
    const focus = focusOrder[di % focusOrder.length];
    const secondary = focusOrder[(di + 1) % focusOrder.length];

    const slots = [];
    slots.push({
      range: range(wakeHh, wakeMm, 8, 0),
      title: 'Wake & breakfast',
      detail: 'Light prep; avoid screens for the first few minutes if possible.',
      kind: 'routine',
    });

    if (!isWeekend) {
      slots.push({
        range: '08:00 – 15:30',
        title: 'School / classes',
        detail: `Attendance and class focus tie directly to your model inputs — prioritize ${focus} if you have gaps there.`,
        kind: 'school',
      });
      const studyMins = Math.round(dayStudyH * 60);
      const revMins = Math.round(dayRevH * 60);
      const evStartH = 16;
      const evStartM = 0;
      slots.push({
        range: `${String(evStartH).padStart(2, '0')}:00 – ${fmtEnd(evStartH, evStartM, studyMins)}`,
        title: 'Core study block',
        detail: `${dayStudyH.toFixed(1)} h — ${focus}: notes, textbook, LMS tasks (${Math.round(n.lms_login_frequency || 0)} logins/wk target in your profile).`,
        kind: 'study',
      });
      if (revMins > 0) {
        const r1 = fmtEnd(evStartH, evStartM, studyMins);
        const [rh, rm] = r1.split(':').map(Number);
        slots.push({
          range: `${r1} – ${fmtEnd(rh, rm, revMins)}`,
          title: 'Revision & recall',
          detail: `${dayRevH.toFixed(1)} h — flashcards, quiz mistakes, ${secondary} review.`,
          kind: 'revision',
        });
      }
    } else {
      const block1 = Math.round(dayStudyH * 0.55 * 60);
      const block2 = Math.round(dayStudyH * 0.45 * 60);
      const revMins = Math.round(dayRevH * 60);
      slots.push({
        range: '09:30 – ' + fmtEnd(9, 30, block1),
        title: 'Weekend deep work',
        detail: `${(block1 / 60).toFixed(1)} h — weakest area first: ${focus}.`,
        kind: 'study',
      });
      slots.push({
        range: '11:00 – ' + fmtEnd(11, 0, block2),
        title: 'Second subject block',
        detail: `${(block2 / 60).toFixed(1)} h — ${secondary}; mix problems with reading.`,
        kind: 'study',
      });
      if (revMins > 0) {
        slots.push({
          range: '15:00 – ' + fmtEnd(15, 0, revMins),
          title: 'Light revision',
          detail: `${dayRevH.toFixed(1)} h — week recap, planner for Monday.`,
          kind: 'revision',
        });
      }
    }

    const bedH = Math.min(23, sleepStartH);
    slots.push({
      range: `${String(bedH).padStart(2, '0')}:${String(sleepStartM).padStart(2, '0')} – (next day)`,
      title: 'Wind-down & sleep',
      detail: `${sleep.toFixed(1)} h target — screens down 30–45 min before sleep${stress >= 7 ? '; try a fixed bedtime' : ''}.`,
      kind: 'sleep',
    });

    return { name, slots };
  });

  return {
    days,
    summary: {
      focusOrder,
      study,
      revision,
      sleep,
      stressNote,
      atLabel: snapshot.atLabel ?? '',
      outcome: snapshot.outcome ?? '',
      attendance: n.attendance_rate,
    },
  };
}

const ACADEMIC_GRAPH_KEYS = [
  'attendance_rate',
  'previous_gpa',
  'assignment_avg',
  'quiz_avg',
  'project_score',
  'standardized_exam_score',
];

function parseFormNum(data, key) {
  const raw = data[key];
  const n = typeof raw === 'string' ? parseFloat(raw) : Number(raw);
  return Number.isFinite(n) ? n : 0;
}

function academicBarsNumeric(att, gpa, assign, quiz, project, exam) {
  const bars = [
    { key: 'att', label: 'Attendance', short: 'Att', value: Math.min(100, Math.max(0, att * 100)) },
    { key: 'gpa', label: 'GPA', short: 'GPA', value: Math.min(100, Math.max(0, (gpa / 4) * 100)) },
    { key: 'asn', label: 'Assignments', short: 'Asn', value: Math.min(100, Math.max(0, assign)) },
    { key: 'quiz', label: 'Quiz', short: 'Quiz', value: Math.min(100, Math.max(0, quiz)) },
    { key: 'proj', label: 'Project', short: 'Prj', value: Math.min(100, Math.max(0, project)) },
    { key: 'exam', label: 'Exam', short: 'Exam', value: Math.min(100, Math.max(0, exam)) },
  ];
  const sum = bars.reduce((s, b) => s + b.value, 0);
  const composite = bars.length ? Math.round((sum / bars.length) * 10) / 10 : 0;
  return { bars, composite };
}

/** Map form strings to bars + composite; `allEmpty` uses raw field emptiness for UX. */
function academicBarsFromFormData(fd) {
  const att = parseFormNum(fd, 'attendance_rate');
  const gpa = parseFormNum(fd, 'previous_gpa');
  const assign = parseFormNum(fd, 'assignment_avg');
  const quiz = parseFormNum(fd, 'quiz_avg');
  const project = parseFormNum(fd, 'project_score');
  const exam = parseFormNum(fd, 'standardized_exam_score');
  const { bars, composite } = academicBarsNumeric(att, gpa, assign, quiz, project, exam);
  const allEmpty = ACADEMIC_GRAPH_KEYS.every(
    (k) => fd[k] === '' || fd[k] === undefined || fd[k] === '-' || fd[k] === '.',
  );
  return { bars, composite, allEmpty };
}

function fieldProvided(fd, key) {
  const v = fd[key];
  return v !== '' && v !== undefined && v !== null && v !== '-' && v !== '.';
}

const SUBJECT_MARK_KEYS = ['math_score', 'science_score', 'english_score', 'history_score', 'computer_score'];

/** Average % of entered subject marks — display only; not sent as student_marks_percent to the API. */
function subjectMarksAverageFromFormData(fd) {
  const vals = [];
  for (const k of SUBJECT_MARK_KEYS) {
    if (fieldProvided(fd, k)) {
      vals.push(parseFormNum(fd, k));
    }
  }
  if (!vals.length) {
    return { composite: null, allEmpty: true };
  }
  const sum = vals.reduce((a, b) => a + b, 0);
  const composite = Math.round((sum / vals.length) * 10) / 10;
  return { composite, allEmpty: false };
}

function subjectAverageLabelFromFormData(fd) {
  const marks = subjectMarksAverageFromFormData(fd);
  if (!marks.allEmpty) return `${marks.composite}%`;

  const fallbackKeys = ['student_marks_percent', 'previous_marks', 'assignment_avg', 'quiz_avg', 'project_score'];
  const vals = [];
  for (const k of fallbackKeys) {
    if (fieldProvided(fd, k)) {
      const v = parseFormNum(fd, k);
      if (Number.isFinite(v)) vals.push(Math.min(100, Math.max(0, v)));
    }
  }
  if (!vals.length) return '—';
  const avg = Math.round((vals.reduce((a, b) => a + b, 0) / vals.length) * 10) / 10;
  return `${avg}%`;
}

/** Training data uses attendance as 0–1; users may type percent — normalize for messaging. */
function attendanceAsPercent(fd, numeric) {
  if (!fieldProvided(fd, 'attendance_rate')) return null;
  const a = numeric.attendance_rate;
  if (a <= 1 && a >= 0) return Math.round(a * 100);
  return Math.min(100, Math.max(0, Math.round(a)));
}

/**
 * Evidence-aligned coaching from the last saved form snapshot + BPNN-visible drivers
 * (attendance, previous_marks, assignments, study_hours, sleep_hours, mental_stress, parent_involvement).
 */
function buildImprovementPlan(snapshot) {
  if (!snapshot?.formData || !snapshot.numeric) {
    return { items: [], missingModelFields: [], meta: null };
  }
  const { formData: fd, numeric: n, outcome, confidence, atLabel, composite, compositeLabel } = snapshot;
  const items = [];
  const push = (item) => items.push({ id: `${items.length}`, ...item });

  const attPct = attendanceAsPercent(fd, n);
  const subjectAvg = subjectMarksAverageFromFormData(fd);
  const comp = typeof composite === 'number' ? composite : (subjectAvg.composite ?? 0);
  const allEmptySubjectMarks = subjectAvg.allEmpty;

  const studyH = fieldProvided(fd, 'study_hours_daily') ? n.study_hours_daily : null;
  const sleepH = fieldProvided(fd, 'sleep_hours') ? n.sleep_hours : null;
  const revH = fieldProvided(fd, 'revision_hours') ? n.revision_hours : null;
  const dayHours =
    studyH != null || sleepH != null || revH != null
      ? (studyH ?? 0) + (sleepH ?? 0) + (revH ?? 0)
      : null;

  const engagement =
    attPct != null && studyH != null ? (attPct / 100) * studyH : null;

  const modelKeys = [
    ['attendance_rate', 'Attendance'],
    ['assignment_avg', 'Assignment average'],
    ['study_hours_daily', 'Daily study hours'],
    ['sleep_hours', 'Sleep hours'],
    ['mental_stress', 'Mental stress'],
  ];
  const missingModelFields = modelKeys.filter(([k]) => !fieldProvided(fd, k)).map(([, label]) => label);

  if (attPct != null) {
    if (attPct < 65) {
      push({
        priority: 1,
        category: 'Academic',
        title: 'Increase class attendance',
        why: `You reported attendance at about ${attPct}% (model uses this directly). Below ~65% it is hard to recover concepts and assignment quality.`,
        actions: [
          'Treat attendance as non‑negotiable for 3 weeks — aim for ≥90% in that window.',
          'For each missed hour, schedule a 25‑minute catch‑up the same day (notes + one practice problem).',
          'Pair with a peer for accountability check‑ins twice per week.',
        ],
      });
    } else if (attPct < 82) {
      push({
        priority: 2,
        category: 'Academic',
        title: 'Push attendance into the “safe” band',
        why: `At ~${attPct}% you are above critical risk but still below the range many successful cohorts cluster in (roughly mid‑80s+ as a fraction of sessions).`,
        actions: [
          'Identify the one recurring slot you miss most and resolve the clash (transport, timing, or workload).',
          'Ask instructors for the three key outcomes per week so partial absences hurt less.',
        ],
      });
    }
  }

  if (!allEmptySubjectMarks && comp < 58) {
    push({
      priority: 1,
      category: 'Academic',
      title: 'Lift overall subject performance (avg ~' + comp + '%)',
      why: `Your average across entered subject marks is about ${comp}% (display only). The model still uses each subject score and other fields separately; strengthen weak subjects and core academics.`,
      actions: [
        'Pick the weakest subject bar and add one measurable upgrade this week.',
        'Use spaced repetition: same topic on day 0, 2, and 7 instead of one long cram block.',
        'Cross‑check assignment and quiz averages — they feed the model independently of this display average.',
      ],
    });
  } else if (!allEmptySubjectMarks && comp >= 58 && comp < 72) {
    push({
      priority: 2,
      category: 'Academic',
      title: 'Solidify mid‑level subject performance',
      why: `Average subject marks near ${comp}% usually mean one or two subjects drag the mean.`,
      actions: [
        'Export scores by subject and target only the bottom two for two weeks.',
        'Add one weekly “exam‑style” timed practice even if the next exam is far away.',
      ],
    });
  }

  if (fieldProvided(fd, 'assignment_avg')) {
    if (n.assignment_avg < 55) {
      push({
        priority: 1,
        category: 'Academic',
        title: 'Raise assignment performance',
        why: `Assignment average is about ${Math.round(n.assignment_avg)}%. The model ingests this as assignments — sustained lows correlate with weaker previous_marks.`,
        actions: [
          'Start each task with a rubric highlight: mark mandatory vs stretch criteria.',
          'Submit a rough draft 24h early when allowed; use feedback loops instead of one‑shot attempts.',
        ],
      });
    } else if (n.assignment_avg < 70) {
      push({
        priority: 2,
        category: 'Academic',
        title: 'Turn consistent B work into A‑range work',
        why: `Near ${Math.round(n.assignment_avg)}% suggests understanding is partial, not missing.`,
        actions: [
          'After each return, rewrite one missed question from scratch without notes, then compare to the solution.',
        ],
      });
    }
  }

  if (studyH != null) {
    if (studyH < 1.25) {
      push({
        priority: 1,
        category: 'Study habits',
        title: 'Increase focused study time gradually',
        why: `About ${formatHour(studyH)} h/day study is very low relative to typical training ranges for this model’s study_hours feature.`,
        actions: [
          'Add +25 minutes of distraction‑free study daily for two weeks, then reassess with a new prediction.',
          'Use Pomodoro (25/5) and log sessions so the number you enter reflects reality.',
        ],
      });
    } else if (studyH < 2.5) {
      push({
        priority: 2,
        category: 'Study habits',
        title: 'Build depth, not just more hours',
        why: `At ~${formatHour(studyH)} h/day you are in a workable range but may lack depth for harder material.`,
        actions: [
          'Reserve 40% of study time for active recall (flashcards, closed‑book questions) vs passive reading.',
        ],
      });
    }
  }

  if (dayHours != null && dayHours > 24.01) {
    push({
      priority: 1,
      category: 'Study habits',
      title: 'Fix the 24‑hour day plan',
      why: `Sleep (${formatHour(sleepH ?? 0)} h) + study (${formatHour(studyH ?? 0)} h) + revision (${formatHour(revH ?? 0)} h) exceeds 24 — inputs are inconsistent.`,
      actions: [
        'Re‑enter hours so they reflect a single typical weekday; trim the least realistic bucket first.',
      ],
    });
  }

  if (sleepH != null) {
    if (sleepH < 6) {
      push({
        priority: 1,
        category: 'Wellbeing',
        title: 'Prioritize sleep duration',
        why: `About ${formatHour(sleepH)} h/night sleep is below common recommendations for learning and mood (often ~7–9h for students). The model also multiplies stress × sleep.`,
        actions: [
          'Move bedtime earlier by 15 minutes every three nights until you reach a stable 7h minimum trial.',
          'Cut late caffeine and replace last hour of screen with light review or planning only.',
        ],
      });
    } else if (sleepH >= 6 && sleepH < 7) {
      push({
        priority: 2,
        category: 'Wellbeing',
        title: 'Small sleep gains, large payoff',
        why: `Near ${formatHour(sleepH)} h is close but still under many students’ optimal range.`,
        actions: [
          'Anchor wake time 7 days/week; adjust bedtime instead of sleeping in wildly on weekends.',
        ],
      });
    }
  }

  if (fieldProvided(fd, 'mental_stress')) {
    if (n.mental_stress >= 7) {
      push({
        priority: 1,
        category: 'Wellbeing',
        title: 'Reduce sustained high stress',
        why: `Mental stress is ${n.mental_stress}/10. The engineered feature stress_sleep = stress × sleep links both to model behavior.`,
        actions: [
          'Book one non‑negotiable recovery block weekly (walk, sport, or counseling).',
          'Split large deadlines into 45‑minute tasks with visible checkboxes to lower perceived load.',
        ],
      });
    } else if (n.mental_stress >= 5) {
      push({
        priority: 2,
        category: 'Wellbeing',
        title: 'Manage stress before it spikes',
        why: `Stress at ${n.mental_stress}/10 is moderate — good window for habits before exams.`,
        actions: [
          'Try brief daily breathing or box breathing before study blocks.',
        ],
      });
    }
  }

  if (fieldProvided(fd, 'stress_index')) {
    const si = Math.min(10, Math.max(0, n.stress_index));
    if (si >= 7) {
      push({
        priority: 2,
        category: 'Wellbeing',
        title: 'Stress index is elevated',
        why: `You reported stress_index ${si}/10 — align workload and recovery even though the BPNN uses mental_stress specifically.`,
        actions: [
          'Map stress spikes to specific courses or deadlines and address the top one with your instructor or advisor.',
        ],
      });
    }
  }

  if (fieldProvided(fd, 'parent_involvement') && n.parent_involvement <= 3) {
    push({
      priority: 2,
      category: 'Support',
      title: 'Increase structured support',
      why: `Parent involvement is ${n.parent_involvement}/10 in your inputs — the model includes parent_involvement as a feature.`,
      actions: [
        'Schedule a short weekly check‑in with a parent/guardian or mentor: goals, blockers, one win.',
      ],
    });
  }

  if (engagement != null && engagement < 0.9 && attPct != null && attPct >= 50) {
    push({
      priority: 2,
      category: 'Study habits',
      title: 'Raise “engagement” (attendance × study)',
      why: `The training pipeline builds engagement = attendance × study_hours. With your numbers, that product is still modest (~${engagement.toFixed(2)} in model units).`,
      actions: [
        'If attendance is already OK, the lever is quality study time blocks, not more passive hours.',
      ],
    });
  }

  const subj = [
    ['math_score', 'Math'],
    ['science_score', 'Science'],
    ['english_score', 'English'],
    ['history_score', 'History'],
    ['computer_score', 'Computer'],
  ];
  for (const [key, label] of subj) {
    if (!fieldProvided(fd, key)) continue;
    const v = n[key];
    if (v < 60) {
      push({
        priority: 2,
        category: 'Subjects',
        title: `Strengthen ${label}`,
        why: `${label} score is about ${Math.round(v)}% — weak subjects pull down confidence and exam readiness alongside your other inputs.`,
        actions: [
          `Spend 3 short sessions/week only on ${label} fundamentals before new topics.`,
          'Use error log: rewrite every missed item after 48h without looking at the first attempt.',
        ],
      });
    }
  }

  if (fieldProvided(fd, 'physical_activity') && n.physical_activity < 3) {
    push({
      priority: 3,
      category: 'Wellbeing',
      title: 'Add light physical activity',
      why: `Physical activity is low (${n.physical_activity} in your scale) — movement supports sleep and stress regulation.`,
      actions: ['Add 20–30 minutes of brisk walking or sport 3×/week.'],
    });
  }

  if (fieldProvided(fd, 'lms_login_frequency') && n.lms_login_frequency < 8) {
    push({
      priority: 3,
      category: 'Engagement',
      title: 'Increase LMS engagement',
      why: `LMS logins reported at ${n.lms_login_frequency} — low visibility into courses often tracks missed deadlines.`,
      actions: ['Check the LMS daily at a fixed time; enable deadline reminders.'],
    });
  }

  if (outcome === 'Fail' && confidence >= 65) {
    push({
      priority: 1,
      category: 'Model readout',
      title: 'Model is fairly confident about risk — act on drivers above',
      why: `Latest prediction: Fail with ~${confidence}% estimated pass probability (so the model leans Fail). Treat the ranked academic and habit items as the main levers.`,
      actions: [
        'Re‑run Prediction after 2–3 concrete changes so this coach updates with fresh numbers.',
      ],
    });
  } else if (outcome === 'Pass' && !allEmptySubjectMarks && comp < 68) {
    push({
      priority: 2,
      category: 'Model readout',
      title: 'Pass prediction, but subject average is middling',
      why: `Pass with ~${confidence}% pass probability, yet your average subject marks (~${comp}%) suggest a thin margin — strengthen weak inputs before high‑stakes assessments.`,
      actions: [
        'Keep monitoring assignment and attendance trends; one bad stretch can flip the pattern you entered.',
      ],
    });
  }

  items.sort((a, b) => {
    if (a.priority !== b.priority) return a.priority - b.priority;
    const order = { Academic: 0, 'Study habits': 1, Wellbeing: 2, Subjects: 3, Engagement: 4, Support: 5, 'Model readout': 6 };
    return (order[a.category] ?? 9) - (order[b.category] ?? 9);
  });

  return {
    items,
    missingModelFields,
    meta: {
      outcome,
      confidence,
      atLabel,
      compositeLabel: compositeLabel ?? (allEmptySubjectMarks ? '—' : `${comp}%`),
      composite: allEmptySubjectMarks ? null : comp,
      attPct,
    },
  };
}

function AcademicMarksBarChart({ bars, allEmpty, gradientId }) {
  if (allEmpty) {
    return (
      <div className="academic-chart-empty">
        Enter attendance, GPA, and score fields above — the chart updates automatically (each metric scaled 0–100).
      </div>
    );
  }

  const vbW = 360;
  const vbH = 200;
  const padL = 32;
  const padR = 16;
  const padB = 40;
  const padT = 28;
  const chartW = vbW - padL - padR;
  const chartH = vbH - padT - padB;
  const n = bars.length;
  const gap = 6;
  const barW = (chartW - gap * (n - 1)) / n;

  return (
    <svg className="academic-marks-svg" viewBox={`0 0 ${vbW} ${vbH}`} role="img" aria-label="Academic inputs as percent bars">
      <defs>
        <linearGradient id={gradientId} x1="0%" y1="0%" x2="0%" y2="100%">
          <stop offset="0%" stopColor="#3b82f6" />
          <stop offset="100%" stopColor="#1d4ed8" />
        </linearGradient>
      </defs>
      {[0, 25, 50, 75, 100].map((t) => {
        const y = padT + chartH - (t / 100) * chartH;
        return (
          <g key={t}>
            <line x1={padL} y1={y} x2={vbW - padR} y2={y} stroke="#e2e8f0" strokeWidth="1" />
            <text x={4} y={y + 4} fontSize="9" fill="#94a3b8">
              {t}
            </text>
          </g>
        );
      })}
      {bars.map((b, i) => {
        const h = (b.value / 100) * chartH;
        const x = padL + i * (barW + gap);
        const y = padT + chartH - h;
        return (
          <g key={b.key}>
            <rect
              x={x}
              y={y}
              width={barW}
              height={Math.max(h, 2)}
              rx={4}
              fill={`url(#${gradientId})`}
              opacity={0.92}
            />
            <text x={x + barW / 2} y={vbH - 10} textAnchor="middle" fontSize="10" fill="#64748b" fontWeight="500">
              {b.short}
            </text>
            {h >= 14 ? (
              <text x={x + barW / 2} y={y + h / 2 + 4} textAnchor="middle" fontSize="10" fill="#fff" fontWeight="700">
                {Math.round(b.value)}
              </text>
            ) : (
              <text x={x + barW / 2} y={y - 4} textAnchor="middle" fontSize="9" fill="#0f172a" fontWeight="600">
                {Math.round(b.value)}
              </text>
            )}
          </g>
            );
          })}
    </svg>
  );
}

function parseHourField(raw) {
  const n = parseFloat(raw);
  return Number.isFinite(n) ? n : 0;
}

function formatHour(n) {
  return (Math.round(n * 10) / 10).toFixed(1).replace(/\.0$/, '');
}

/** Suggested split of “remaining” hours: slightly more study than revision (evidence-based rough heuristic). */
const STUDY_SHARE_OF_REMAINING = 0.55;

const IMPROVEMENT_CATEGORY_ICONS = {
  Academic: GraduationCap,
  'Study habits': Clock,
  Wellbeing: Activity,
  Subjects: BookOpen,
  Engagement: Target,
  Support: Users,
  'Model readout': Brain,
};

function ImprovementCategoryIcon({ category }) {
  const Icon = IMPROVEMENT_CATEGORY_ICONS[category] || Sparkles;
  return <Icon className="improvement-card-icon-svg" strokeWidth={1.75} aria-hidden />;
}

// ─── Study timetable (from last predictor inputs) ─────────────────
function TimetablePage({ improvementSnapshot, onNavigate }) {
  const plan = useMemo(() => buildStudyTimetable(improvementSnapshot), [improvementSnapshot]);
  const hasSnapshot = Boolean(improvementSnapshot?.formData);

  return (
    <div className="timetable-page">
      <div className="page-hero">
        <div className="page-title-row page-title-row--flush">
          <div>
            <h1>Study timetable</h1>
            <p>
              A suggested week built from <strong>your last Performance Predictor inputs</strong>: sleep, daily study &amp; revision hours, subject scores
              (weaker subjects get more focus in the plan), attendance, and stress. Adjust all times to match your real school or college schedule.
            </p>
          </div>
        </div>
      </div>

      {!hasSnapshot || !plan ? (
        <div className="panel panel--elevated timetable-empty">
          <div className="panel-body timetable-empty-body">
            <CalendarDays size={44} className="timetable-empty-icon" strokeWidth={1.25} aria-hidden />
            <h2>No profile captured yet</h2>
            <p>
              Run <strong>Performance Predictor</strong> once with your numbers. We reuse that snapshot to prioritize weak subjects and respect your
              study, revision, and sleep targets.
            </p>
            <button type="button" className="btn btn-primary" onClick={() => onNavigate('predict')}>
              <Brain size={16} /> Open Performance Predictor
            </button>
          </div>
        </div>
      ) : (
        <>
          <div className="panel panel--elevated timetable-summary">
            <div className="panel-header">
              <div>
                <h3>Plan summary</h3>
                <p>
                  Based on inputs from <strong>{plan.summary.atLabel || 'your last run'}</strong>
                  {plan.summary.outcome ? (
                    <>
                      {' '}
                      · Model outcome: <strong>{plan.summary.outcome}</strong>
                    </>
                  ) : null}
                </p>
              </div>
              <button type="button" className="btn btn-outline btn-sm" onClick={() => onNavigate('predict')}>
                Update inputs
              </button>
            </div>
            <div className="panel-body timetable-summary-body">
              <ul className="timetable-summary-list">
                <li>
                  <span>Priority subjects</span>
                  <strong>{plan.summary.focusOrder.join(' → ')}</strong>
                  <small>Lowest scores first — rotate through the week.</small>
                </li>
                <li>
                  <span>Daily study (weekday baseline)</span>
                  <strong className="tabnums">{plan.summary.study.toFixed(1)} h</strong>
                </li>
                <li>
                  <span>Daily revision (split across week)</span>
                  <strong className="tabnums">{plan.summary.revision.toFixed(1)} h</strong>
                </li>
                <li>
                  <span>Sleep target</span>
                  <strong className="tabnums">{plan.summary.sleep.toFixed(1)} h</strong>
                </li>
                <li>
                  <span>Wellbeing note</span>
                  <strong>{plan.summary.stressNote}</strong>
                </li>
                {plan.summary.attendance > 0 && plan.summary.attendance < 0.75 ? (
                  <li>
                    <span>Attendance</span>
                    <strong>Below ~75% in your profile — protect school block; add short reviews so gaps do not compound.</strong>
                  </li>
                ) : null}
                {improvementSnapshot?.numeric && improvementSnapshot.numeric.physical_activity > 0 ? (
                  <li>
                    <span>Activity</span>
                    <strong>
                      Aim for ~{improvementSnapshot.numeric.physical_activity} h/week movement — slot 20–45 min on 2–3 days between study blocks.
                    </strong>
                  </li>
                ) : null}
              </ul>
            </div>
          </div>

          <div className="timetable-week panel panel--elevated">
            <div className="panel-header">
              <div>
                <h3>Weekly rhythm</h3>
                <p>Example times — shift earlier or later to fit your commute and meal times.</p>
              </div>
            </div>
            <div className="panel-body panel-body--flush">
              <div className="timetable-day-list" role="list">
                {plan.days.map((d) => (
                  <section key={d.name} className="timetable-day" aria-labelledby={`tt-${d.name}`}>
                    <h4 id={`tt-${d.name}`} className="timetable-day-title">
                      {d.name}
                    </h4>
                    <ul className="timetable-slot-list">
                      {d.slots.map((s, i) => (
                        <li key={`${d.name}-${i}`} className={`timetable-slot timetable-slot--${s.kind}`}>
                          <span className="timetable-slot-time tabnums">{s.range}</span>
                          <div className="timetable-slot-body">
                            <span className="timetable-slot-title">{s.title}</span>
                            <span className="timetable-slot-detail">{s.detail}</span>
                          </div>
                        </li>
                      ))}
                    </ul>
                  </section>
                ))}
              </div>
            </div>
          </div>

          <p className="timetable-disclaimer">
            This timetable is a <strong>study aid</strong> derived from your form inputs, not medical or legal advice. If you are overwhelmed, talk to a
            teacher, counsellor, or guardian — and scale hours down.
          </p>
        </>
      )}
    </div>
  );
}

// ─── Improvement coach (uses last successful prediction snapshot) ──
function ImprovementPage({ improvementSnapshot, onNavigate }) {
  const plan = useMemo(() => buildImprovementPlan(improvementSnapshot), [improvementSnapshot]);
  const hasSnapshot = Boolean(improvementSnapshot?.formData);
  const actionCount = plan.items.length;

  return (
    <div className="improvement-page">
      <section className="improvement-hero-band" aria-labelledby="improvement-heading">
        <div className="improvement-hero-glow" aria-hidden />
        <div className="improvement-hero-grid">
          <div className="improvement-hero-copy">
            <span className="improvement-eyebrow">
              <Sparkles size={14} strokeWidth={2} aria-hidden />
              Personalized coaching
            </span>
            <h1 id="improvement-heading">Improvement coach</h1>
            <p className="improvement-hero-lede">
              Action steps built from <strong>your last inputs</strong> and the same signals the BPNN reads: attendance, subject scores, assignments →
              marks, assignments, study hours, sleep, stress, and parent involvement.
            </p>
          </div>
          <div className="improvement-hero-cta">
            <button type="button" className="btn btn-primary improvement-cta-primary" onClick={() => onNavigate('predict')}>
              {hasSnapshot ? (
                <>
                  Update profile <ArrowRight size={17} strokeWidth={2} />
                </>
              ) : (
                <>
                  Start in Predictor <ArrowRight size={17} strokeWidth={2} />
                </>
              )}
            </button>
            {hasSnapshot && (
              <button type="button" className="btn btn-outline improvement-cta-secondary" onClick={() => onNavigate('analytics')}>
                <PieChart size={16} strokeWidth={1.75} />
                Session analytics
              </button>
            )}
          </div>
        </div>
      </section>

      {!hasSnapshot ? (
        <div className="improvement-empty-layout panel panel--elevated">
          <div className="improvement-empty-visual" aria-hidden>
            <div className="improvement-empty-orbit improvement-empty-orbit--1" />
            <div className="improvement-empty-orbit improvement-empty-orbit--2" />
            <div className="improvement-empty-orbit improvement-empty-orbit--3" />
            <div className="improvement-empty-icon-wrap">
              <Lightbulb size={40} strokeWidth={1.4} />
            </div>
          </div>
          <div className="improvement-empty-copy">
            <h2>Capture your profile first</h2>
            <p className="improvement-empty-desc">
              Run <strong>Performance Predictor</strong> once with your real numbers. We keep that snapshot (including fields you skip) so every tip
              references <em>your</em> data — not generic checklists.
            </p>
            <ol className="improvement-empty-steps">
              <li>
                <span className="improvement-step-num">1</span>
                Open Performance Predictor and enter academics + habits.
              </li>
              <li>
                <span className="improvement-step-num">2</span>
                Click <strong>Run prediction</strong> and wait for a successful result.
              </li>
              <li>
                <span className="improvement-step-num">3</span>
                Return here for a ranked improvement plan.
              </li>
            </ol>
            <button type="button" className="btn btn-primary btn-lg improvement-empty-btn" onClick={() => onNavigate('predict')}>
              Open Performance Predictor
            </button>
          </div>
        </div>
      ) : (
        <div className="improvement-body">
          <div className="improvement-summary-card">
            <div className="improvement-summary-top">
              <div className="improvement-outcome-block">
                <div
                  className={`improvement-outcome-ring ${plan.meta?.outcome === 'Pass' ? 'improvement-outcome-ring--pass' : 'improvement-outcome-ring--fail'}`}
                  aria-hidden
                />
                <div className="improvement-outcome-text">
                  <span className="improvement-outcome-label">Model outcome</span>
                  <span className={`improvement-outcome-value ${plan.meta?.outcome === 'Pass' ? 'improvement-outcome-value--pass' : 'improvement-outcome-value--fail'}`}>
                    {plan.meta?.outcome ?? '—'}
                  </span>
                </div>
              </div>
              <div className="improvement-summary-intro">
                <h2 className="improvement-summary-heading">Snapshot in use</h2>
                <p className="improvement-summary-time">
                  <Clock size={15} strokeWidth={2} aria-hidden />
                  <time dateTime={plan.meta?.atLabel}>{plan.meta?.atLabel}</time>
                </p>
              </div>
            </div>

            <div className="improvement-metric-tiles">
              <div className="improvement-tile">
                <span className="improvement-tile-label">Pass probability</span>
                <strong className="improvement-tile-value improvement-tile-value--blue">{plan.meta?.confidence ?? '—'}%</strong>
                <span className="improvement-tile-hint">BPNN output</span>
              </div>
              <div className="improvement-tile">
                <span className="improvement-tile-label">Avg subject marks</span>
                <strong className="improvement-tile-value">{plan.meta?.compositeLabel ?? '—'}</strong>
                <span className="improvement-tile-hint">From your bars</span>
              </div>
              <div className="improvement-tile">
                <span className="improvement-tile-label">Attendance</span>
                <strong className="improvement-tile-value">{plan.meta?.attPct != null ? `${plan.meta.attPct}%` : '—'}</strong>
                <span className="improvement-tile-hint">If provided</span>
              </div>
              <div className="improvement-tile">
                <span className="improvement-tile-label">Action items</span>
                <strong className="improvement-tile-value improvement-tile-value--accent">{actionCount}</strong>
                <span className="improvement-tile-hint">This plan</span>
              </div>
            </div>

            <div className="improvement-disclosure">
              <AlertCircle size={16} strokeWidth={2} className="improvement-disclosure-icon" aria-hidden />
              <p>
                Tips use thresholds on <em>your</em> fields only. They are educational — not medical advice and not a substitute for instructors or
                counselors.
              </p>
            </div>
          </div>

          {plan.missingModelFields.length > 0 && (
            <div className="improvement-missing-banner" role="status">
              <div className="improvement-missing-icon-wrap">
                <ListChecks size={22} strokeWidth={1.75} />
              </div>
              <div className="improvement-missing-text">
                <span className="improvement-missing-title">Sharpen the next run</span>
                <p>
                  Add <strong>{plan.missingModelFields.join(', ')}</strong> when you can — the API passes them straight into the model for fuller
                  context.
                </p>
              </div>
            </div>
          )}

          <div className="improvement-plan-header">
            <h2 className="improvement-plan-title">Your action plan</h2>
            <span className="improvement-plan-badge">
              {actionCount === 0 ? 'All clear' : `${actionCount} focus ${actionCount === 1 ? 'area' : 'areas'}`}
            </span>
          </div>

          <div className="improvement-list">
            {plan.items.length === 0 ? (
              <div className="improvement-all-clear">
                <div className="improvement-all-clear-icon">
                  <CheckCircle2 size={32} strokeWidth={1.5} />
                </div>
                <h3>Nothing critical flagged</h3>
                <p>
                  Your inputs did not hit our high‑priority rules. You can still tighten habits or fill more fields on the next prediction for
                  deeper guidance.
                </p>
              </div>
            ) : (
              plan.items.map((item, index) => (
                <article
                  key={item.id}
                  className={`improvement-card improvement-card--p${item.priority}`}
                  style={{ '--improvement-i': index }}
                >
                  <div className={`improvement-card-icon improvement-card-icon--p${item.priority}`}>
                    <ImprovementCategoryIcon category={item.category} />
                  </div>
                  <div className="improvement-card-body">
                    <header className="improvement-card-head">
                      <span className="improvement-pill">{item.category}</span>
                      <span className={`improvement-impact improvement-impact--p${item.priority}`}>
                        {item.priority === 1 ? 'High impact' : item.priority === 2 ? 'Medium' : 'Supporting'}
                      </span>
                    </header>
                    <h3>{item.title}</h3>
                    <p className="improvement-why">{item.why}</p>
                    <p className="improvement-actions-label">Suggested steps</p>
                    <ol className="improvement-actions">
                      {item.actions.map((a, i) => (
                        <li key={i}>{a}</li>
                      ))}
                    </ol>
                  </div>
                </article>
              ))
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// ─── Prediction Page ──────────────────────────────────
function PredictPage({ onRecordPrediction, onImprovementSnapshot, onLastPayload }) {
  const [formData, setFormData] = useState(() => ({ ...INITIAL_FORM }));

  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [toast, setToast] = useState(null);
  const [artifactInfo, setArtifactInfo] = useState(null);
  const resultStripRef = useRef(null);

  useEffect(() => {
    let cancelled = false;
    getMetaVersion()
      .then((d) => {
        if (!cancelled) setArtifactInfo(d);
      })
      .catch(() => {
        if (!cancelled) setArtifactInfo(null);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Real-time payload feed for the Analytics network diagram (no API call here; diagram debounces).
  useEffect(() => {
    const id = setTimeout(() => {
      try {
        onLastPayload?.(formStringsToNumbers(formData));
      } catch {
        /* ignore typing errors */
      }
    }, 150);
    return () => clearTimeout(id);
  }, [formData, onLastPayload]);

  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => setToast(null), 4200);
    return () => clearTimeout(t);
  }, [toast]);

  const handleChange = (e) => {
    const { name, value } = e.target;
    if (value === '' || DECIMAL_TYPING.test(value)) {
      setFormData((prev) => ({ ...prev, [name]: value }));
    }
  };

  const handleBlurClamp = (e, min, max) => {
    const { name, value } = e.target;
    if (value === '' || value === '-' || value === '.') return;
    const n = parseFloat(value);
    if (!Number.isFinite(n)) {
      setFormData((prev) => ({ ...prev, [name]: '' }));
      return;
    }
    const c = Math.min(max, Math.max(min, n));
    setFormData((prev) => ({ ...prev, [name]: String(Math.round(c * 10) / 10) }));
  };

  const handleBlurScale1to10 = (e) => {
    const { name, value } = e.target;
    if (value === '' || value === '-' || value === '.') return;
    const n = parseFloat(value);
    if (!Number.isFinite(n)) {
      setFormData((prev) => ({ ...prev, [name]: '' }));
      return;
    }
    const c = Math.min(10, Math.max(1, Math.round(n)));
    setFormData((prev) => ({ ...prev, [name]: String(c) }));
  };

  const dayBudget = useMemo(() => {
    const raw = {
      sleep: parseHourField(formData.sleep_hours),
      study: parseHourField(formData.study_hours_daily),
      revision: parseHourField(formData.revision_hours),
    };
    const sleep = Math.min(24, Math.max(0, raw.sleep));
    const study = Math.min(24, Math.max(0, raw.study));
    const revision = Math.min(24, Math.max(0, raw.revision));
    const used = sleep + study + revision;
    const remaining = 24 - used;
    const invalidSleep = formData.sleep_hours !== '' && raw.sleep > 24;
    const invalidStudy = formData.study_hours_daily !== '' && raw.study > 24;
    const invalidRevision = formData.revision_hours !== '' && raw.revision > 24;
    return {
      sleep,
      study,
      revision,
      used,
      remaining,
      over24: used > 24,
      invalidSleep,
      invalidStudy,
      invalidRevision,
    };
  }, [formData]);

  const [budgetPhysics, setBudgetPhysics] = useState(null);
  useEffect(() => {
    let cancelled = false;
    postStudyBudget({
      sleep_hours: dayBudget.sleep,
      study_hours_daily: dayBudget.study,
      revision_hours: dayBudget.revision,
    })
      .then((d) => {
        if (!cancelled) setBudgetPhysics(d);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, [dayBudget.sleep, dayBudget.study, dayBudget.revision]);

  const academicSectionRef = useRef(null);
  const behavioralSectionRef = useRef(null);
  const subjectSectionRef = useRef(null);
  const [guidedNav, setGuidedNav] = useState(false);

  const [completeness, setCompleteness] = useState(null);
  const [goalSeekOpen, setGoalSeekOpen] = useState(false);
  const [gsField, setGsField] = useState('study_hours_daily');
  const [gsTargetPct, setGsTargetPct] = useState('75');
  const [gsLo, setGsLo] = useState('0');
  const [gsHi, setGsHi] = useState('12');
  const [gsSteps, setGsSteps] = useState('24');
  const [gsResult, setGsResult] = useState(null);
  const [gsBusy, setGsBusy] = useState(false);

  const chartGradId = `academicBarGrad-${useId().replace(/:/g, '')}`;
  const academicChart = useMemo(() => academicBarsFromFormData(formData), [formData]);
  const subjectMarksComposite = useMemo(() => subjectMarksAverageFromFormData(formData), [formData]);

  useEffect(() => {
    const t = setTimeout(() => {
      const payload = formStringsToNumbers(formData);
      postCompleteness(payload)
        .then(setCompleteness)
        .catch(() => setCompleteness(null));
    }, 450);
    return () => clearTimeout(t);
  }, [formData]);

  const applySuggestedDaySplit = () => {
    setFormData((prev) => {
      let s = parseFloat(prev.sleep_hours);
      if (!Number.isFinite(s)) s = 0;
      s = Math.min(24, Math.max(0, s));
      const rem = Math.max(0, 24 - s);
      const studyS = Math.round(rem * STUDY_SHARE_OF_REMAINING * 10) / 10;
      const revS = Math.round((rem - studyS) * 10) / 10;
      const next = {
        ...prev,
        study_hours_daily: String(studyS),
        revision_hours: String(revS),
      };
      if (prev.sleep_hours !== '') {
        next.sleep_hours = String(Math.round(s * 10) / 10);
      }
      return next;
    });
  };

  const handlePredict = async () => {
    setLoading(true);
    setError(null);
    const payload = formStringsToNumbers(formData);
    const marksSnap = subjectMarksAverageFromFormData(formData);
    const subjectAvgLabel = subjectAverageLabelFromFormData(formData);
    try {
      const res = await axios.post(predictUrl(), payload);
      setPrediction(res.data);
      onLastPayload?.(payload);
      onRecordPrediction?.({
        id: globalThis.crypto?.randomUUID?.() ?? `p-${Date.now()}`,
        atLabel: new Date().toLocaleString(),
        outcome: res.data.prediction,
        confidence: Math.round((res.data.probability ?? 0) * 100),
        composite: marksSnap.allEmpty ? null : marksSnap.composite,
        compositeLabel: subjectAvgLabel,
        formData: { ...formData },
        numeric: formStringsToNumbers(formData),
        passProb: typeof res.data.pass_probability === 'number' ? res.data.pass_probability : res.data.probability,
        failProb: typeof res.data.fail_probability === 'number' ? res.data.fail_probability : undefined,
      });
      onImprovementSnapshot?.({
        formData: { ...formData },
        numeric: formStringsToNumbers(formData),
        atLabel: new Date().toLocaleString(),
        outcome: res.data.prediction,
        confidence: Math.round((res.data.probability ?? 0) * 100),
        composite: marksSnap.allEmpty ? null : marksSnap.composite,
        compositeLabel: subjectAvgLabel,
      });
      setToast('Prediction saved. Result below; session log updated.');
      const reduceMotion =
        typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
      requestAnimationFrame(() => {
        resultStripRef.current?.scrollIntoView({ behavior: reduceMotion ? 'auto' : 'smooth', block: 'nearest' });
      });
    } catch (err) {
      const detail = err.response?.data?.detail;
      const msg = Array.isArray(detail)
        ? detail.map((d) => d.msg || d).join('; ')
        : detail || err.message;
      setError(msg || 'Prediction failed. Start API: python -m uvicorn backend.main:app --port 8000 (from project root).');
    } finally {
      setLoading(false);
    }
  };

  const runInlineGoalSeek = async () => {
    setGsBusy(true);
    setGsResult(null);
    try {
      const payload = formStringsToNumbers(formData);
      const rawT = parseFloat(gsTargetPct);
      const target = Math.min(0.99, Math.max(0.01, (Number.isFinite(rawT) ? rawT : 75) / 100));
      const out = await postGoalSeek({
        payload,
        field: gsField,
        target_pass_probability: target,
        range_lo: parseFloat(gsLo) || 0,
        range_hi: parseFloat(gsHi) || 12,
        steps: Math.min(64, Math.max(8, parseInt(gsSteps, 10) || 24)),
      });
      setGsResult(out);
    } catch (err) {
      const detail = err.response?.data?.detail;
      const msg = Array.isArray(detail) ? detail.map((d) => d.msg || d).join('; ') : detail || err.message;
      setGsResult({ error: msg || String(err) });
    } finally {
      setGsBusy(false);
    }
  };

  const inputField = (label, name, placeholder = '0', onBlur) => (
    <div className="form-group">
      <label htmlFor={name}>{label}</label>
      <input
        id={name}
        type="text"
        inputMode="decimal"
        name={name}
        value={formData[name]}
        onChange={handleChange}
        onBlur={onBlur}
        placeholder={placeholder}
        autoComplete="off"
        spellCheck={false}
      />
    </div>
  );

  const isPass = prediction?.prediction === 'Pass';
  const attendanceNum = parseFloat(formData.attendance_rate);
  const attendanceStrong = Number.isFinite(attendanceNum) && attendanceNum > 0.7;

  const compositePercentDisplay = subjectMarksComposite.allEmpty ? '—' : `${subjectMarksComposite.composite}%`;
  const canRunPredict = !dayBudget.over24;

  return (
    <>
      <div className="page-hero">
        <div className="page-title-row page-title-row--flush">
          <div>
            <h1>Performance Predictor</h1>
            <p>
              BPNN pass/fail from your inputs. <strong>Tab</strong> between fields, <strong>Enter</strong> to predict. Successful runs appear on Dashboard &amp; Student Records.
            </p>
          </div>
        </div>
      </div>

      <div
        className="choreography-strip"
        style={{
          marginBottom: 16,
          padding: '12px 16px',
          borderRadius: 12,
          border: '1px solid var(--border)',
          background: 'var(--surface)',
          fontSize: 13,
          lineHeight: 1.45,
        }}
      >
        <div style={{ display: 'flex', alignItems: 'flex-start', gap: 10 }}>
          <Server size={18} strokeWidth={1.75} style={{ flexShrink: 0, marginTop: 2, opacity: 0.85 }} aria-hidden />
          <div>
            <strong style={{ display: 'block', marginBottom: 4 }}>Live model (FastAPI)</strong>
            {artifactInfo?.artifacts_present ? (
              <span style={{ color: 'var(--text-light)' }}>
                Pipeline v{artifactInfo.pipeline_version ?? '—'} · {artifactInfo.feature_column_count ?? '—'} input features · Pass if{' '}
                <code>P(pass) ≥ {artifactInfo.decision_threshold != null && Number.isFinite(Number(artifactInfo.decision_threshold)) ? Number(artifactInfo.decision_threshold).toFixed(3) : '—'}</code>
                {artifactInfo.threshold_metric ? ` (threshold: ${artifactInfo.threshold_metric})` : ''}. Fields you leave empty are filled with training
                medians from <code>scaler.pkl</code> (same as the notebook).
              </span>
            ) : (
              <span style={{ color: 'var(--text-light)' }}>
                Could not confirm artifacts. From the <code>ML_Project2</code> folder run{' '}
                <code>python -m uvicorn backend.main:app --reload --port 8000</code> and keep <code>model.h5</code> + <code>scaler.pkl</code> beside{' '}
                <code>backend/</code>. Dev UI proxies <code>/api</code> → port 8000 (see <code>vite.config.js</code>).
              </span>
            )}
          </div>
        </div>
      </div>

      {/* Result Strip */}
      {prediction && (
        <div ref={resultStripRef} className="result-strip" style={{ marginBottom: 24 }}>
          <div className={`result-indicator ${isPass ? 'pass' : 'fail'}`} />
          <div className="result-content">
            <div className={`result-icon-wrap ${isPass ? 'pass' : 'fail'}`}>
              {isPass ? <CheckCircle2 size={28} /> : <AlertCircle size={28} />}
            </div>
            <div className="result-details">
              <h3 className={isPass ? 'pass' : 'fail'}>{isPass ? 'PASS — Likely to Succeed' : 'FAIL — At Risk'}</h3>
              <p>Prediction generated by BPNN model with {attendanceStrong ? 'strong' : 'weak'} attendance signal</p>
              {prediction.pass_probability != null || prediction.probability != null ? (
                <p style={{ fontSize: 13, marginTop: 8, opacity: 0.88 }}>
                  Raw <code>P(pass)</code> ≈{' '}
                  {(
                    Math.round((prediction.pass_probability ?? prediction.probability ?? 0) * 1000) / 1000
                  ).toFixed(3)}
                  {prediction.decision_threshold != null && Number.isFinite(Number(prediction.decision_threshold)) ? (
                    <>
                      {' '}
                      · labeled <strong>Pass</strong> when ≥ {Number(prediction.decision_threshold).toFixed(3)} (bundle threshold)
                    </>
                  ) : null}
                </p>
              ) : null}
            </div>
            <div className="result-stats">
              <div className="result-stat">
                <div className="result-stat-label">Confidence</div>
                <div className="result-stat-value tabnums">{Math.round(prediction.probability * 100)}%</div>
              </div>
              <div className="result-stat">
                <div className="result-stat-label">Outcome</div>
                <div className="result-stat-value">{prediction.prediction}</div>
              </div>
              <div className="result-stat confidence-bar-wrap">
                <div className="result-stat-label">Score Bar</div>
                <div className="confidence-bar-bg">
                  <div className={`confidence-bar-fill ${isPass ? 'pass' : 'fail'}`} style={{ width: `${prediction.probability * 100}%` }} />
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {error && (
        <div className="error-banner" style={{ marginBottom: 24 }}>
          <AlertCircle size={18} /> {error}
        </div>
      )}

      {loading && (
        <div className="predict-loading-panel" aria-busy="true" aria-live="polite">
          <div className="predict-skeleton-line predict-skeleton-line--lg" />
          <div className="predict-skeleton-line" />
          <div className="predict-skeleton-line predict-skeleton-line--short" />
          <p className="predict-loading-hint">Calling the BPNN — TensorFlow may take a few seconds on first load.</p>
        </div>
      )}

      <div className="choreography-strip" style={{ marginBottom: 16, padding: '12px 16px', borderRadius: 12, border: '1px solid var(--border)', background: 'var(--surface)' }}>
        <label style={{ display: 'inline-flex', alignItems: 'center', gap: 8, cursor: 'pointer', fontSize: 14 }}>
          <input type="checkbox" checked={guidedNav} onChange={(e) => setGuidedNav(e.target.checked)} />
          Guided entry — scroll choreography (24h conservation + fields in interview order)
        </label>
        {guidedNav && (
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginTop: 10 }}>
            <button type="button" className="btn btn-outline btn-sm" onClick={() => academicSectionRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })}>
              1 · Academics &amp; chart
            </button>
            <button type="button" className="btn btn-outline btn-sm" onClick={() => behavioralSectionRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })}>
              2 · Day budget &amp; habits
            </button>
            <button type="button" className="btn btn-outline btn-sm" onClick={() => subjectSectionRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })}>
              3 · Subject marks
            </button>
            <button
              type="button"
              className="btn btn-primary btn-sm"
              disabled={loading || !canRunPredict}
              title={!canRunPredict ? 'Sleep + study + revision must total ≤ 24 hours.' : undefined}
              onClick={() => handlePredict()}
            >
              4 · Run prediction
            </button>
          </div>
        )}
      </div>

      <div
        className="panel panel--elevated"
        style={{ marginBottom: 16, padding: '14px 16px' }}
      >
        <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'center', gap: 8, marginBottom: 12 }}>
          <span style={{ fontSize: 13, fontWeight: 600 }}>Presets</span>
          {Object.entries(SCENARIO_PRESETS).map(([key, { label, patch }]) => (
            <button
              key={key}
              type="button"
              className="btn btn-outline btn-sm"
              onClick={() => setFormData({ ...INITIAL_FORM, ...patch })}
            >
              {label}
            </button>
          ))}
          <button type="button" className="btn btn-outline btn-sm" onClick={() => setFormData({ ...INITIAL_FORM })}>
            Clear
          </button>
        </div>
        {completeness && (
          <div style={{ marginBottom: 12 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13, marginBottom: 4 }}>
              <span>Input completeness (weighted)</span>
              <strong>{completeness.score}%</strong>
            </div>
            <div style={{ height: 8, borderRadius: 4, background: 'var(--border-light)', overflow: 'hidden' }}>
              <div
                style={{
                  width: `${Math.min(100, completeness.score)}%`,
                  height: '100%',
                  background: 'var(--primary)',
                  transition: 'width 0.3s ease',
                }}
              />
            </div>
          </div>
        )}
        <div>
          <button type="button" className="btn btn-outline btn-sm" onClick={() => setGoalSeekOpen((v) => !v)}>
            <Target size={14} style={{ marginRight: 6, verticalAlign: '-2px' }} />
            {goalSeekOpen ? 'Hide' : 'Show'} goal seek
          </button>
          {goalSeekOpen && (
            <div style={{ marginTop: 12, display: 'grid', gap: 10, maxWidth: 640 }}>
              <p style={{ fontSize: 12, margin: 0, opacity: 0.9 }}>
                Grid-search one lever on your <strong>current form values</strong> toward a target pass probability (see Innovation Labs for more tools).
              </p>
              <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, alignItems: 'flex-end' }}>
                <label style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: 12 }}>
                  Field
                  <select
                    value={gsField}
                    onChange={(e) => setGsField(e.target.value)}
                    style={{
                      padding: '8px 10px',
                      borderRadius: 8,
                      border: '1px solid var(--border-color, #e2e8f0)',
                      background: 'var(--input-bg, #fff)',
                      color: 'inherit',
                      minWidth: 160,
                    }}
                  >
                    {GOAL_SEEK_UI_FIELDS.map((f) => (
                      <option key={f.value} value={f.value}>
                        {f.label}
                      </option>
                    ))}
                  </select>
                </label>
                <label style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: 12 }}>
                  Target pass %
                  <input
                    type="text"
                    inputMode="decimal"
                    value={gsTargetPct}
                    onChange={(e) => setGsTargetPct(e.target.value)}
                    style={{ width: 72, padding: '8px 10px', borderRadius: 8, border: '1px solid var(--border-color, #e2e8f0)' }}
                  />
                </label>
                <label style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: 12 }}>
                  Range lo
                  <input
                    type="text"
                    value={gsLo}
                    onChange={(e) => setGsLo(e.target.value)}
                    style={{ width: 64, padding: '8px 10px', borderRadius: 8, border: '1px solid var(--border-color, #e2e8f0)' }}
                  />
                </label>
                <label style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: 12 }}>
                  Range hi
                  <input
                    type="text"
                    value={gsHi}
                    onChange={(e) => setGsHi(e.target.value)}
                    style={{ width: 64, padding: '8px 10px', borderRadius: 8, border: '1px solid var(--border-color, #e2e8f0)' }}
                  />
                </label>
                <label style={{ display: 'flex', flexDirection: 'column', gap: 4, fontSize: 12 }}>
                  Steps
                  <input
                    type="text"
                    value={gsSteps}
                    onChange={(e) => setGsSteps(e.target.value)}
                    style={{ width: 56, padding: '8px 10px', borderRadius: 8, border: '1px solid var(--border-color, #e2e8f0)' }}
                  />
                </label>
                <button type="button" className="btn btn-primary btn-sm" disabled={gsBusy || loading} onClick={runInlineGoalSeek}>
                  {gsBusy ? 'Searching…' : 'Run goal seek'}
                </button>
              </div>
              {gsResult && !gsResult.error && (
                <div style={{ fontSize: 13, background: 'var(--card-inner-bg, rgba(0,0,0,0.04))', padding: 10, borderRadius: 8 }}>
                  Best <code>{gsResult.field}</code> ≈ <strong>{gsResult.best_value}</strong> → p ≈{' '}
                  {(gsResult.best_pass_probability * 100).toFixed(1)}% (target {(gsResult.target_pass_probability * 100).toFixed(0)}%)
                </div>
              )}
              {gsResult?.error && (
                <div className="error-banner" style={{ margin: 0 }}>
                  {String(gsResult.error)}
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Input Form: native <form> so Enter submits from any field */}
      <form
        className="predict-form"
        aria-busy={loading}
        onSubmit={(e) => {
          e.preventDefault();
          if (!loading && canRunPredict) handlePredict();
        }}
      >
      <div className="predict-form-grid-main">
        <div>
          <div className="form-section-head">Academic</div>
          <div className="panel" ref={academicSectionRef}>
          <div className="panel-header">
            <div>
              <h3><Activity size={16} style={{ display: 'inline', verticalAlign: '-2px', marginRight: 6 }} />Academic Metrics</h3>
              <p>
                Core performance indicators. The chart is a <strong>visual breakdown</strong> only (each bar 0–100). It does <strong>not</strong> feed a single “composite” into the model — the BPNN uses these fields alongside behavioral inputs and subject scores.
              </p>
            </div>
          </div>
          <div className="panel-body">
            <div className="form-grid-2">
              {inputField('Attendance Rate (0–1)', 'attendance_rate', 'e.g. 0.85')}
              {inputField('Previous GPA (0–4)', 'previous_gpa', 'e.g. 3.2')}
              {inputField('Assignment Average', 'assignment_avg')}
              {inputField('Quiz Average', 'quiz_avg')}
              {inputField('Project Score', 'project_score')}
              {inputField('Standardized Exam', 'standardized_exam_score')}
            </div>

            <div className="academic-marks-graph-wrap">
              <div className="academic-marks-graph-head">
                <span className="academic-marks-graph-title">Performance from academic inputs</span>
                <span className="academic-marks-graph-composite" role="status" aria-live="polite">
                  Avg subject marks: {compositePercentDisplay}
                </span>
              </div>
              <AcademicMarksBarChart bars={academicChart.bars} allEmpty={academicChart.allEmpty} gradientId={chartGradId} />
              <p className="academic-marks-graph-note">
                Bars: attendance as % (rate × 100); GPA as % of 4.0; other fields capped at 100. <strong>Avg subject marks</strong> is the mean of entered subject scores (display only). The API sets <code>student_marks_percent</code> to 0 so <code>previous_marks</code> comes from the mean of subject scores, not from this chart.
              </p>
            </div>
          </div>
        </div>
        </div>

        <div>
          <div className="form-section-head">Behavior &amp; wellness</div>
          <div className="panel" ref={behavioralSectionRef}>
          <div className="panel-header">
            <div>
              <h3><Clock size={16} style={{ display: 'inline', verticalAlign: '-2px', marginRight: 6 }} />Behavioral Factors</h3>
              <p>Learning habits and wellness. Average subject marks % (mean of subject scores) is shown read-only next to Parent involvement.</p>
            </div>
          </div>
          <div className="panel-body">
            <div className="day-budget-card">
              <div className="day-budget-card-title">
                <Hourglass size={18} />
                <span>24-hour day balance</span>
              </div>
              <p className="day-budget-card-desc">
                Enter realistic hours (each 0–24). Remaining time is what is left in the day after sleep, study, and revision.
              </p>
              <div className="day-budget-stats">
                <div>
                  <span className="day-budget-label">Sleep</span>
                  <strong>{formatHour(dayBudget.sleep)} h</strong>
                </div>
                <div>
                  <span className="day-budget-label">Study / day</span>
                  <strong>{formatHour(dayBudget.study)} h</strong>
                </div>
                <div>
                  <span className="day-budget-label">Revision</span>
                  <strong>{formatHour(dayBudget.revision)} h</strong>
                </div>
                <div className="day-budget-total">
                  <span className="day-budget-label">Used</span>
                  <strong>{formatHour(dayBudget.used)} / 24 h</strong>
                </div>
                <div className={`day-budget-remaining ${dayBudget.over24 ? 'is-over' : ''}`}>
                  <span className="day-budget-label">Remaining in day</span>
                  <strong>{dayBudget.over24 ? '—' : `${formatHour(Math.max(0, dayBudget.remaining))} h`}</strong>
                </div>
              </div>
              {!dayBudget.over24 && dayBudget.remaining > 0 && (
                <p className="day-budget-tip">
                  Example: if sleep is <strong>7 h</strong>, you have <strong>17 h</strong> left for everything else (meals, school, study, revision, transport).
                  Use the button below to split the <em>full</em> leftover day into study and revision only (55% / 45% — a simple academic balance); adjust afterward to match your real schedule.
                </p>
              )}
              {budgetPhysics && !dayBudget.over24 && (
                <p className="day-budget-tip" style={{ marginTop: 10 }}>
                  <strong>24h physics (hand-crafted):</strong> recovery score {budgetPhysics.recovery_score}/100 · fatigue risk {budgetPhysics.fatigue_risk_score}/100 · day balance {budgetPhysics.day_balance_score}/100 — interpret alongside the BPNN, not as a second model.
                </p>
              )}
              {dayBudget.over24 && (
                <p className="day-budget-warn">
                  These three fields add up to more than 24 hours. Tab out of a field or blur to clamp values (0–24), or lower them manually.
                </p>
              )}
              {(dayBudget.invalidSleep || dayBudget.invalidStudy || dayBudget.invalidRevision) && (
                <p className="day-budget-warn">
                  One or more values exceed 24 h — not realistic for a single day. They are shown capped in the summary until you fix the input.
                </p>
              )}
              <button
                type="button"
                className="btn btn-outline day-budget-btn"
                onClick={applySuggestedDaySplit}
              >
                Apply suggested study / revision split (uses sleep, fills ~55% / 45% of remaining time)
              </button>
            </div>

            <div className="form-grid-2" style={{ marginTop: 20 }}>
              {inputField('Study Hours / Day (0–24)', 'study_hours_daily', 'e.g. 3.5', (e) => handleBlurClamp(e, 0, 24))}
              {inputField('Revision Hours / Day (0–24)', 'revision_hours', 'e.g. 1.5', (e) => handleBlurClamp(e, 0, 24))}
              {inputField('Sleep Hours / Day (0–24)', 'sleep_hours', 'e.g. 7–9', (e) => handleBlurClamp(e, 0, 24))}
              {inputField('Mental Stress (1–10)', 'mental_stress', 'e.g. 5', (e) => handleBlurScale1to10(e))}
              {inputField('Sleep Quality (1–10)', 'sleep_quality', 'e.g. 7', (e) => handleBlurScale1to10(e))}
              {inputField('Learning Efficiency (0–1)', 'learning_efficiency', 'e.g. 0.5', (e) => handleBlurClamp(e, 0, 1))}
              {inputField('Physical Activity (Hrs/W, 0–40)', 'physical_activity', 'e.g. 5', (e) => handleBlurClamp(e, 0, 40))}
              {inputField('LMS logins / Week (0–50)', 'lms_login_frequency', 'e.g. 12', (e) => handleBlurClamp(e, 0, 50))}
              {inputField('Coding hours / Week (0–60)', 'coding_practice_hours', 'e.g. 8', (e) => handleBlurClamp(e, 0, 60))}
              {inputField('Digital Literacy (1–10)', 'digital_literacy', 'e.g. 7', (e) => handleBlurScale1to10(e))}
            </div>
            <div className="form-grid-2 behavioral-marks-row">
              {inputField('Parent Involvement (1–10)', 'parent_involvement', '6', (e) => handleBlurScale1to10(e))}
              <div className="form-group form-group-readout" aria-live="polite">
                <span className="form-readout-label" id="behavioral-marks-readout-label">
                  Avg subject marks (% of 100)
                </span>
                <div
                  id="behavioral-marks-readout"
                  className="form-readout-box"
                  role="status"
                  aria-labelledby="behavioral-marks-readout-label"
                  title="Mean of entered subject scores — display only"
                >
                  {compositePercentDisplay}
                </div>
                <span className="form-readout-hint">Mean of Mathematics–Computer fields you filled in</span>
              </div>
            </div>
          </div>
        </div>
        </div>
      </div>

      <div className="form-section-head" style={{ marginTop: 20 }}>
        Subject marks
      </div>
      <div className="panel" style={{ marginTop: 8 }} ref={subjectSectionRef}>
        <div className="panel-header">
          <div>
            <h3><BookOpen size={16} style={{ display: 'inline', verticalAlign: '-2px', marginRight: 6 }} />Subject-wise Marks</h3>
            <p>
              Each score is sent to the API individually. The model derives <code>previous_marks</code> from the <strong>mean of these subject scores</strong> (not from the academic bar chart). The <strong>average %</strong> shown above is for your reference only.
            </p>
          </div>
        </div>
        <div className="panel-body">
          <div className="form-grid-5">
            {inputField('Mathematics', 'math_score')}
            {inputField('Science', 'science_score')}
            {inputField('English', 'english_score')}
            {inputField('History', 'history_score')}
            {inputField('Computer Science', 'computer_score')}
          </div>

          <div style={{ marginTop: 8 }}>
            {inputField('Stress Index (0–1)', 'stress_index', 'e.g. 0.35')}
          </div>
        </div>
      </div>

      <div style={{ marginTop: 24, display: 'flex', gap: 12, flexWrap: 'wrap', alignItems: 'center' }}>
        <button
          type="submit"
          className="btn btn-primary btn-lg"
          disabled={loading || !canRunPredict}
          title={!canRunPredict ? 'Sleep + study + revision must total ≤ 24 hours in a day.' : undefined}
        >
          {loading ? (
            <>Analyzing…</>
          ) : (
            <><Zap size={18} /> Run Prediction</>
          )}
        </button>
        {!canRunPredict && (
          <span style={{ fontSize: 12, color: 'var(--text-light)', maxWidth: '42ch' }}>
            Fix the 24h day total before running — the model still accepts the request, but totals over 24h are not realistic.
          </span>
        )}
        <button
          type="button"
          className="btn btn-outline btn-lg"
          onClick={() => {
            setFormData({ ...INITIAL_FORM });
            setPrediction(null);
            setError(null);
          }}
        >
          Reset Form
        </button>
      </div>
      </form>

      {toast ? (
        <div className="app-toast" role="status" aria-live="polite">
          {toast}
        </div>
      ) : null}
    </>
  );
}

// ─── App Layout ───────────────────────────────────────
function App() {
  const [page, setPage] = useState('dashboard');
  const [showLanding, setShowLanding] = useState(readShowLandingOnStartup);
  const [searchQuery, setSearchQuery] = useState('');
  const [darkMode, setDarkMode] = useState(() => localStorage.getItem('edupredict-theme') === 'dark');
  const [predictionHistory, setPredictionHistory] = useState([]);
  const [improvementSnapshot, setImprovementSnapshot] = useState(null);
  const [lastPredictPayload, setLastPredictPayload] = useState(null);
  const [csvImportState, setCsvImportState] = useState({ loading: false, message: '', error: false });

  useEffect(() => {
    document.documentElement.classList.toggle('dark', darkMode);
    localStorage.setItem('edupredict-theme', darkMode ? 'dark' : 'light');
  }, [darkMode]);

  useEffect(() => {
    if (showLanding) {
      document.title = 'EduPredict · Welcome';
      return;
    }
    const crumb = PAGE_BREADCRUMB[page] ?? 'App';
    document.title = `EduPredict · ${crumb}`;
  }, [page, showLanding]);

  const addPrediction = (entry) => {
    setPredictionHistory((prev) => [entry, ...prev].slice(0, 2500));
  };

  const handleCsvFileSelect = async (file) => {
    setCsvImportState({ loading: true, message: 'Uploading CSV to server…', error: false });
    try {
      const makeFd = () => {
        const fd = new FormData();
        fd.append('file', file);
        return fd;
      };
      const csvPostUrls = () => {
        const out = [];
        const add = (u) => {
          if (!u || out.includes(u)) return;
          out.push(u);
        };
        add(importCsvUrl());
        // Always try local Uvicorn too: Vite proxy can 404, or VITE_API_URL may point at an old/wrong host.
        for (const u of [
          'http://127.0.0.1:8000/import/csv',
          'http://127.0.0.1:8000/api/import/csv',
          'http://localhost:8000/import/csv',
          'http://localhost:8000/api/import/csv',
        ]) {
          add(u);
        }
        return out;
      };
      let res;
      const urls = csvPostUrls();
      for (let i = 0; i < urls.length; i += 1) {
        const url = urls[i];
        setCsvImportState({
          loading: true,
          message:
            i === 0
              ? `Analyzing ${file.name} on server (parse + BPNN)…`
              : `CSV upload retry (${i + 1}/${urls.length}): ${url}`,
          error: false,
        });
        try {
          res = await axios.post(url, makeFd(), { timeout: 180000 });
          break;
        } catch (err) {
          const st = err?.response?.status;
          if (st === 404 && i < urls.length - 1) continue;
          throw err;
        }
      }
      const data = res.data;
      const rows = Array.isArray(data?.results) ? data.results : [];
      if (rows.length === 0) {
        throw new Error('Server returned no scored rows.');
      }

      const polarityMode = data.polarity?.mode ?? 'one_is_pass';
      const polarityNote = data.polarity?.note ?? '';
      const displayName = data.filename || file.name;

      const importedEntries = rows.map((r, idx) => {
        const formData = { ...INITIAL_FORM, ...(r.form_data || {}) };
        for (const k of Object.keys(formData)) {
          const v = formData[k];
          if (v === undefined || v === null) {
            formData[k] = INITIAL_FORM[k] ?? '';
          } else {
            formData[k] = String(v);
          }
        }
        const marksSnap = subjectMarksAverageFromFormData(formData);
        const subjectAvgLabel = subjectAverageLabelFromFormData(formData);
        const passProb = typeof r.pass_probability === 'number' ? r.pass_probability : Number.NaN;
        const confPct = Number.isFinite(passProb) ? Math.round(passProb * 100) : null;
        const ft = r.file_truth || {};
        const disp = r.display || {};
        return {
          id: globalThis.crypto?.randomUUID?.() ?? `csv-${Date.now()}-${idx}`,
          atLabel: new Date().toLocaleString(),
          outcome: r.prediction ?? 'Fail',
          confidence: confPct,
          composite: marksSnap.allEmpty ? null : marksSnap.composite,
          compositeLabel: subjectAvgLabel,
          formData,
          numeric: r.numeric && typeof r.numeric === 'object' ? r.numeric : {},
          passProb: Number.isFinite(passProb) ? passProb : undefined,
          failProb: typeof r.fail_probability === 'number' ? r.fail_probability : undefined,
          source: 'csv',
          csvMeta: {
            displayName: disp.displayName ?? '',
            displayId: disp.displayId ?? '',
            extras: Array.isArray(disp.extras) ? disp.extras : [],
            rowIndex: r.row_index ?? idx + 1,
            fileName: displayName,
            batchError: null,
            resultPolarity: polarityMode,
            resultPolarityNote: polarityNote,
            fileTruthRaw: ft.raw_display ?? '',
            fileExpectPass: ft.expect_pass ?? null,
            fileTruthLabel: ft.label ?? '',
          },
        };
      });

      const agreePct = typeof data.agreement?.percent === 'number' ? data.agreement.percent : null;
      const compare = data.agreement?.compared_rows ?? 0;

      setPredictionHistory((prev) => [...importedEntries.reverse(), ...prev].slice(0, 2500));
      setPage('students');
      const summary =
        agreePct != null && compare > 0
          ? `${importedEntries.length} imported · ${agreePct}% match`
          : `${importedEntries.length} student(s) imported`;
      const details = [
        `File: ${displayName}`,
        `Server parse + BPNN`,
        agreePct != null && compare > 0
          ? `Model vs CSV label: ${agreePct}% match on ${compare} row(s). ${polarityNote}`
          : null,
        'Open Student Records for Name / ID / details.',
      ]
        .filter(Boolean)
        .join('\n');
      try {
        appendAudit({
          at: new Date().toLocaleString(),
          file: displayName,
          rows: importedEntries.length,
          matchPct: agreePct,
          threshold: typeof data.results?.[0]?.decision_threshold === 'number'
            ? Number(data.results[0].decision_threshold.toFixed(3))
            : null,
        });
      } catch { /* ignore localStorage issues */ }
      setCsvImportState({
        loading: false,
        summary,
        message: details,
        error: false,
      });
    } catch (error) {
      const status = error?.response?.status;
      const detail = error?.response?.data?.detail;
      let msg = Array.isArray(detail) ? detail.map((d) => d.msg || d).join('; ') : detail || error.message;
      if (status === 404 || msg === 'Not Found') {
        msg =
          'API 404 after trying /api/import/csv and direct :8000 URLs: nothing on port 8000 exposes POST /import/csv. From the inner ML_Project2 folder run scripts/run_api.ps1 (or: python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000). Open http://127.0.0.1:8000/docs — you should see import/csv. Wrong folder if uvicorn fails to import backend. Clear VITE_API_URL unless it is exactly your API origin (no /api suffix).';
      }
      setCsvImportState({
        loading: false,
        summary: 'Import failed',
        message: msg || 'CSV import failed.',
        error: true,
      });
    }
  };

  const reportImprovementPlan = useMemo(() => buildImprovementPlan(improvementSnapshot), [improvementSnapshot]);

  const breadcrumb = PAGE_BREADCRUMB[page] ?? 'EduPredict';

  const renderPage = () => {
    switch (page) {
      case 'dashboard':
        return <DashboardPage onNavigate={setPage} predictionHistory={predictionHistory} />;
      case 'predict':
        return (
          <PredictPage
            onRecordPrediction={addPrediction}
            onImprovementSnapshot={setImprovementSnapshot}
            onLastPayload={setLastPredictPayload}
          />
        );
      case 'labs':
        return <LabsPage lastPredictPayload={lastPredictPayload} onNavigate={setPage} />;
      case 'improvement':
        return <ImprovementPage improvementSnapshot={improvementSnapshot} onNavigate={setPage} />;
      case 'timetable':
        return <TimetablePage improvementSnapshot={improvementSnapshot} onNavigate={setPage} />;
      case 'students':
        return <StudentRecordsPage predictionHistory={predictionHistory} searchQuery={searchQuery} onNavigate={setPage} />;
      case 'imports':
        return <ImportHubPage predictionHistory={predictionHistory} />;
      case 'reports':
        return (
          <ReportsPage
            predictionHistory={predictionHistory}
            improvementSnapshot={improvementSnapshot}
            improvementPlan={reportImprovementPlan}
            darkMode={darkMode}
            onNavigate={setPage}
          />
        );
      case 'analytics':
        return (
          <AnalyticsPage
            predictionHistory={predictionHistory}
            lastPredictPayload={lastPredictPayload}
            onRecordPrediction={addPrediction}
          />
        );
      case 'settings':
        return (
          <SettingsPage
            onReplayLanding={() => {
              try {
                localStorage.removeItem(LANDING_SKIP_STORAGE_KEY);
                localStorage.removeItem(LANDING_SKIP_LEGACY_KEY);
              } catch {
                /* ignore */
              }
              setShowLanding(true);
            }}
          />
        );
      case 'help':
        return <HelpPage onNavigate={setPage} />;
      default:
        return <DashboardPage onNavigate={setPage} predictionHistory={predictionHistory} />;
    }
  };

  if (showLanding) {
    return (
      <LandingExperience
        onEnterApp={() => setShowLanding(false)}
        darkMode={darkMode}
        onToggleTheme={() => setDarkMode((d) => !d)}
      />
    );
  }

  return (
    <div className="erp-layout">
      <Sidebar
        activePage={page}
        onNavigate={setPage}
        onCsvFileSelect={handleCsvFileSelect}
        csvImportState={csvImportState}
      />
      <div className="main-area">
        <Header
          breadcrumb={breadcrumb}
          searchQuery={searchQuery}
          onSearchChange={setSearchQuery}
          darkMode={darkMode}
          onToggleTheme={() => setDarkMode((d) => !d)}
        />
        <div className="content content--mesh">{renderPage()}</div>
      </div>
    </div>
  );
}

export default App;
