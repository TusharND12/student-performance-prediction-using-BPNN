import React, { useEffect, useMemo, useState } from 'react';
import {
  FlaskConical,
  Scale,
  Skull,
  Users,
  Fingerprint,
  Gauge,
  ListOrdered,
  Loader2,
  Target,
  Table2,
  BarChart3,
  Package,
  Percent,
  ArrowRight,
  Sparkles,
  Hourglass,
} from 'lucide-react';
import {
  postSensitivity,
  postCounterfactual,
  postAdversary,
  getCohortSummary,
  postCohortCompare,
  getManifest,
  postStudyBudget,
  postShadowModel,
} from './labsApi.js';
import {
  postGoalSeek,
  postBatchPredict,
  getCalibration,
  getMetaVersion,
  postCompleteness,
} from './metaApi.js';

/** Demo profile so Labs tools work before any Predictor run (session-only). */
const DEMO_LABS_PAYLOAD = {
  attendance_rate: 0.82,
  assignment_avg: 72,
  quiz_avg: 70,
  project_score: 74,
  previous_gpa: 3.1,
  student_marks_percent: 0,
  study_hours_daily: 3.5,
  revision_hours: 1,
  sleep_hours: 7,
  mental_stress: 5,
  sleep_quality: 6,
  standardized_exam_score: 68,
  learning_efficiency: 0.5,
  stress_index: 4,
  physical_activity: 10,
  lms_login_frequency: 12,
  coding_practice_hours: 4,
  digital_literacy: 6,
  parent_involvement: 7,
  math_score: 72,
  science_score: 70,
  english_score: 74,
  history_score: 69,
  computer_score: 73,
};

const CF_FIELDS = [
  { key: 'parent_involvement', label: 'Parent involvement', def: 7, min: 1, max: 10 },
  { key: 'mental_stress', label: 'Mental stress', def: 5, min: 1, max: 10 },
  { key: 'sleep_hours', label: 'Sleep hours', def: 7, min: 0, max: 24 },
  { key: 'attendance_rate', label: 'Attendance (0–1)', def: 0.85, min: 0, max: 1 },
  { key: 'study_hours_daily', label: 'Study hours / day', def: 3.5, min: 0, max: 24 },
  { key: 'assignment_avg', label: 'Assignment avg', def: 72, min: 0, max: 100 },
  { key: 'quiz_avg', label: 'Quiz avg', def: 70, min: 0, max: 100 },
  { key: 'project_score', label: 'Project score', def: 74, min: 0, max: 100 },
  { key: 'previous_gpa', label: 'GPA (0–4)', def: 3.1, min: 0, max: 4 },
  { key: 'standardized_exam_score', label: 'Exam score', def: 68, min: 0, max: 100 },
  { key: 'math_score', label: 'Math %', def: 72, min: 0, max: 100 },
];

const FIELD_LABELS = {
  attendance_rate: 'Attendance rate',
  assignment_avg: 'Assignment avg',
  quiz_avg: 'Quiz avg',
  project_score: 'Project score',
  previous_gpa: 'GPA',
  student_marks_percent: 'Student marks %',
  study_hours_daily: 'Study hours / day',
  sleep_hours: 'Sleep hours',
  mental_stress: 'Mental stress',
  parent_involvement: 'Parent involvement',
  math_score: 'Math',
  science_score: 'Science',
  english_score: 'English',
  history_score: 'History',
  computer_score: 'Computer',
  standardized_exam_score: 'Exam score',
};

/** Used by `App.jsx` for dashboard copy; not a React component. */
// eslint-disable-next-line react-refresh/only-export-components -- shared helper, not a route component
export function buildSessionNarrativeArc(predictionHistory) {
  const runs = [...predictionHistory].reverse();
  if (!runs.length) {
    return { title: 'Empty story', beats: ['Run the predictor to start your session arc.'], mood: 'neutral' };
  }
  const beats = [];
  const first = runs[0];
  const last = runs[runs.length - 1];
  beats.push(`Opening: first logged run was ${first.outcome ?? '—'} at ~${first.confidence ?? '—'}% pass probability.`);
  if (runs.length >= 2) {
    const trend = (last.confidence ?? 0) - (first.confidence ?? 0);
    beats.push(
      trend > 3
        ? `Rising action: probability climbs by ~${Math.round(trend)} pts across the session — inputs may be improving or becoming more optimistic.`
        : trend < -3
          ? `Tension: probability drifts down ~${Math.round(Math.abs(trend))} pts — stress, sleep, or academics may be moving against you in the log.`
          : `Steady state: probabilities stay within a narrow band — consistent scenario framing.`,
    );
  }
  const passStreak = runs.filter((r) => r.outcome === 'Pass').length;
  beats.push(
    `Finale snapshot: ${runs.length} run(s), ${passStreak} Pass — treat the arc as narrative context, not fate.`,
  );
  const mood = passStreak / runs.length >= 0.7 ? 'hopeful' : passStreak / runs.length <= 0.3 ? 'caution' : 'mixed';
  return { title: 'Session narrative arc', beats, mood };
}

export default function LabsPage({ lastPredictPayload, onNavigate }) {
  const [useDemo, setUseDemo] = useState(false);

  const effectivePayload = useMemo(() => {
    if (lastPredictPayload && Object.keys(lastPredictPayload).length > 0) {
      return lastPredictPayload;
    }
    if (useDemo) {
      return DEMO_LABS_PAYLOAD;
    }
    return null;
  }, [lastPredictPayload, useDemo]);

  const [sens, setSens] = useState(null);
  const [adv, setAdv] = useState(null);
  const [cf, setCf] = useState(null);
  const [cohortSum, setCohortSum] = useState(null);
  const [cohortCmp, setCohortCmp] = useState(null);
  const [manifest, setManifest] = useState(null);
  const [busy, setBusy] = useState(null);
  const [err, setErr] = useState(null);
  const [cfField, setCfField] = useState('parent_involvement');
  const [cfVal, setCfVal] = useState(7);

  const [gsField, setGsField] = useState('study_hours_daily');
  const [gsTargetPct, setGsTargetPct] = useState('75');
  const [gsLo, setGsLo] = useState('0');
  const [gsHi, setGsHi] = useState('12');
  const [gsSteps, setGsSteps] = useState('24');
  const [gsOut, setGsOut] = useState(null);

  const [batchText, setBatchText] = useState('[{"attendance_rate":0.85,"study_hours_daily":3}]');
  const [batchOut, setBatchOut] = useState(null);

  const [shadow, setShadow] = useState(null);
  const [calib, setCalib] = useState(undefined);
  const [metaVer, setMetaVer] = useState(null);
  const [completeness, setCompleteness] = useState(null);
  const [budgetLab, setBudgetLab] = useState(null);

  useEffect(() => {
    getCohortSummary().then(setCohortSum).catch(() => setCohortSum({ available: false }));
    getManifest().then(setManifest).catch(() => setManifest(null));
    getCalibration()
      .then(setCalib)
      .catch(() => setCalib({ available: false, reason: 'unavailable' }));
    getMetaVersion()
      .then(setMetaVer)
      .catch(() => setMetaVer(null));
  }, []);

  useEffect(() => {
    const t = setTimeout(() => {
      if (!effectivePayload) {
        setCompleteness(null);
        return;
      }
      postCompleteness(effectivePayload)
        .then(setCompleteness)
        .catch(() => setCompleteness(null));
    }, 400);
    return () => clearTimeout(t);
  }, [effectivePayload]);

  useEffect(() => {
    if (!effectivePayload) {
      setBudgetLab(null);
      return;
    }
    postStudyBudget({
      sleep_hours: Number(effectivePayload.sleep_hours) || 0,
      study_hours_daily: Number(effectivePayload.study_hours_daily) || 0,
      revision_hours: Number(effectivePayload.revision_hours) || 0,
    })
      .then(setBudgetLab)
      .catch(() => setBudgetLab(null));
  }, [effectivePayload]);

  useEffect(() => {
    const ranges = {
      study_hours_daily: ['0', '12'],
      sleep_hours: ['0', '12'],
      mental_stress: ['1', '10'],
      attendance_rate: ['0', '1'],
      assignment_avg: ['40', '100'],
      parent_involvement: ['1', '10'],
    };
    const [lo, hi] = ranges[gsField] || ['0', '12'];
    setGsLo(lo);
    setGsHi(hi);
  }, [gsField]);

  const canRun = Boolean(effectivePayload);

  const cfMeta = useMemo(() => CF_FIELDS.find((x) => x.key === cfField) || CF_FIELDS[0], [cfField]);

  const GS_FIELDS = [
    { key: 'study_hours_daily', label: 'Study hours / day' },
    { key: 'sleep_hours', label: 'Sleep hours' },
    { key: 'mental_stress', label: 'Mental stress' },
    { key: 'attendance_rate', label: 'Attendance (0–1)' },
    { key: 'assignment_avg', label: 'Assignment avg' },
    { key: 'parent_involvement', label: 'Parent involvement' },
  ];

  useEffect(() => {
    const m = CF_FIELDS.find((x) => x.key === cfField) || CF_FIELDS[0];
    setCfVal(m.def);
  }, [cfField]);

  const runSens = async () => {
    if (!canRun) return;
    setBusy('sens');
    setErr(null);
    try {
      const [s, a] = await Promise.all([postSensitivity(effectivePayload), postAdversary(effectivePayload)]);
      setSens(s);
      setAdv(a);
    } catch (e) {
      setErr(e?.response?.data?.detail || e.message);
    } finally {
      setBusy(null);
    }
  };

  const runAdvOnly = async () => {
    if (!canRun) return;
    setBusy('adv');
    setErr(null);
    try {
      const a = await postAdversary(effectivePayload);
      setAdv(a);
    } catch (e) {
      setErr(e?.response?.data?.detail || e.message);
    } finally {
      setBusy(null);
    }
  };

  const runCf = async () => {
    if (!canRun) return;
    setBusy('cf');
    setErr(null);
    try {
      const raw = Number(cfVal);
      const v = Math.min(cfMeta.max, Math.max(cfMeta.min, Number.isFinite(raw) ? raw : cfMeta.def));
      const out = await postCounterfactual(effectivePayload, cfField, v);
      setCf(out);
    } catch (e) {
      setErr(e?.response?.data?.detail || e.message);
    } finally {
      setBusy(null);
    }
  };

  const runCohort = async () => {
    if (!canRun) return;
    setBusy('coh');
    setErr(null);
    try {
      const out = await postCohortCompare(effectivePayload);
      setCohortCmp(out);
    } catch (e) {
      setErr(e?.response?.data?.detail || e.message);
    } finally {
      setBusy(null);
    }
  };

  const runGoalSeek = async () => {
    if (!canRun) return;
    setBusy('gs');
    setErr(null);
    setGsOut(null);
    try {
      const rawT = parseFloat(gsTargetPct);
      const target = Math.min(0.99, Math.max(0.01, (Number.isFinite(rawT) ? rawT : 75) / 100));
      const out = await postGoalSeek({
        payload: effectivePayload,
        field: gsField,
        target_pass_probability: target,
        range_lo: parseFloat(gsLo) || 0,
        range_hi: parseFloat(gsHi) || 12,
        steps: Math.min(64, Math.max(8, parseInt(gsSteps, 10) || 24)),
      });
      setGsOut(out);
    } catch (e) {
      setErr(e?.response?.data?.detail || e.message);
    } finally {
      setBusy(null);
    }
  };

  const runShadow = async () => {
    if (!canRun) return;
    setBusy('shadow');
    setErr(null);
    setShadow(null);
    try {
      const out = await postShadowModel(effectivePayload);
      setShadow(out);
    } catch (e) {
      setErr(e?.response?.data?.detail || e.message);
    } finally {
      setBusy(null);
    }
  };

  const runBatch = async () => {
    setBusy('batch');
    setErr(null);
    setBatchOut(null);
    try {
      const parsed = JSON.parse(batchText);
      if (!Array.isArray(parsed)) throw new Error('JSON must be an array of row objects');
      const out = await postBatchPredict(parsed);
      setBatchOut(out);
    } catch (e) {
      setErr(e?.response?.data?.detail || e.message);
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="labs-page">
      <div className="page-hero">
        <div className="page-title-row page-title-row--flush">
          <div>
            <h1>
              <FlaskConical size={28} style={{ verticalAlign: '-4px', marginRight: 8 }} />
              Innovation Labs
            </h1>
            <p>
              Sensitivity autopsy, adversary challenge, fairness-style counterfactuals, peer cohort shadows, and export manifest — powered by the same BPNN pipeline.
            </p>
          </div>
        </div>
      </div>

      {!canRun && (
        <div className="feature-card" style={{ marginBottom: 20, borderColor: 'var(--color-warning, #d97706)' }}>
          <p>
            <strong>No analysis payload yet.</strong> Check <strong>Use demo profile</strong> below to run every tool with sample inputs, or submit <strong>Performance Predictor</strong> once so Labs uses your last mapped row (session-only, in-memory).
          </p>
          <label style={{ display: 'inline-flex', alignItems: 'center', gap: 10, marginTop: 12, cursor: 'pointer', fontSize: 14 }}>
            <input type="checkbox" checked={useDemo} onChange={(e) => setUseDemo(e.target.checked)} />
            Use demo profile (sample student)
          </label>
        </div>
      )}
      {canRun && useDemo && !lastPredictPayload && (
        <p className="labs-muted" style={{ marginBottom: 16 }}>
          Demo profile is active. Run Performance Predictor to replace this with your real last payload.
        </p>
      )}

      {err && (
        <div className="error-banner" style={{ marginBottom: 16 }}>
          {String(err)}
        </div>
      )}

      <div className="labs-grid">
        <div className="feature-card" style={{ gridColumn: '1 / -1' }}>
          <div className="feature-card-icon"><BarChart3 size={22} /></div>
          <h3>Calibration (training deciles)</h3>
          <p>Empirical pass rate by model probability bin — same data as Analytics.</p>
          {calib === undefined && <p className="labs-muted">Loading…</p>}
          {calib?.available && (
            <div style={{ overflowX: 'auto' }}>
              <table className="labs-calib-table">
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
                      <td>{row.empirical_pass_rate != null ? `${(row.empirical_pass_rate * 100).toFixed(1)}%` : '—'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {calib && !calib.available && <p className="labs-muted">{calib.reason || 'Unavailable (start API and ensure student_data.csv exists).'}</p>}
        </div>

        <div className="feature-card">
          <div className="feature-card-icon"><Package size={22} /></div>
          <h3>Artifact version</h3>
          <p>Model / scaler fingerprints for exports and reports.</p>
          {metaVer ? (
            <div className="labs-pre labs-mono">
              model: {metaVer.model_h5_sha256_prefix || '—'}
              <br />
              scaler: {metaVer.scaler_pkl_sha256_prefix || '—'}
              <br />
              schema: {metaVer.export_schema || '—'}
            </div>
          ) : (
            <p className="labs-muted">Could not load (API running?)</p>
          )}
        </div>

        <div className="feature-card">
          <div className="feature-card-icon"><Percent size={22} /></div>
          <h3>Input completeness</h3>
          <p>Weighted coverage of key fields for the active payload (demo or your last run).</p>
          {completeness && (
            <>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13, marginBottom: 6 }}>
                <span>Score</span>
                <strong>{completeness.score}%</strong>
              </div>
              <div style={{ height: 8, borderRadius: 4, background: 'var(--border-light)', overflow: 'hidden' }}>
                <div
                  style={{
                    width: `${Math.min(100, completeness.score)}%`,
                    height: '100%',
                    background: 'var(--primary)',
                  }}
                />
              </div>
            </>
          )}
          {!completeness && canRun && <p className="labs-muted">Computing…</p>}
          {!canRun && <p className="labs-muted">Enable demo or run Predictor.</p>}
        </div>

        <div className="feature-card">
          <div className="feature-card-icon"><Hourglass size={22} /></div>
          <h3>24h study budget (physics)</h3>
          <p>Recovery / fatigue scores from sleep + study + revision hours on the active payload.</p>
          {budgetLab && (
            <div className="labs-pre">
              Used {budgetLab.used_hours}h · Remaining {budgetLab.remaining_hours}h
              <br />
              Recovery {budgetLab.recovery_score} · Fatigue risk {budgetLab.fatigue_risk_score} · Balance {budgetLab.day_balance_score}
            </div>
          )}
          {!budgetLab && canRun && <p className="labs-muted">Computing…</p>}
          {!canRun && <p className="labs-muted">Enable demo or run Predictor.</p>}
        </div>

        <div className="feature-card">
          <div className="feature-card-icon"><Gauge size={22} /></div>
          <h3>Model autopsy (local sensitivity)</h3>
          <p>Finite-difference Δp per input — see what moves pass probability fastest around your current point.</p>
          <button type="button" className="btn btn-primary" disabled={!canRun || busy} onClick={runSens}>
            {busy === 'sens' ? <Loader2 className="spin" size={18} /> : 'Run sensitivity + adversary'}
          </button>
          {sens && (
            <div className="labs-pre">
              <div>Baseline p(pass): <strong>{sens.baseline_pass_probability}</strong></div>
              <ul className="labs-list">
                {(sens.items || []).slice(0, 8).map((it) => (
                  <li key={it.field}>
                    <strong>{FIELD_LABELS[it.field] || it.field}</strong>: Δp ≈ {it.delta_pass_probability} (step {it.epsilon})
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>

        <div className="feature-card">
          <div className="feature-card-icon feature-card-icon--violet"><Skull size={22} /></div>
          <h3>Viva adversary</h3>
          <p>Rule-based “challenge” bullets — generated with sensitivity signals. Not a second model; a structured skeptic.</p>
          <button type="button" className="btn btn-outline btn-sm" style={{ marginBottom: 8 }} disabled={!canRun || busy} onClick={runAdvOnly}>
            {busy === 'adv' ? <Loader2 className="spin" size={18} /> : 'Run adversary only'}
          </button>
          {adv && (
            <ul className="labs-list">
              {(adv.bullets || []).map((b, i) => (
                <li key={i}>{b}</li>
              ))}
            </ul>
          )}
          {!adv && <p className="text-muted">Run sensitivity above to generate adversary copy.</p>}
        </div>

        <div className="feature-card">
          <div className="feature-card-icon"><Scale size={22} /></div>
          <h3>Counterfactual mirror</h3>
          <p>Hold all inputs fixed except one alternate value — compare pass probabilities (fairness / what-if).</p>
          <div className="labs-cf-row">
            <label>
              Field
              <select className="input-like" value={cfField} onChange={(e) => setCfField(e.target.value)}>
                {CF_FIELDS.map((f) => (
                  <option key={f.key} value={f.key}>{f.label}</option>
                ))}
              </select>
            </label>
            <label>
              Alternate value
              <input
                type="number"
                className="input-like"
                value={cfVal}
                min={cfMeta.min}
                max={cfMeta.max}
                step="any"
                onChange={(e) => setCfVal(e.target.value)}
              />
            </label>
            <button type="button" className="btn btn-outline" disabled={!canRun || busy} onClick={runCf}>
              {busy === 'cf' ? <Loader2 className="spin" size={18} /> : 'Compare'}
            </button>
          </div>
          {cf && (
            <div className="labs-pre">
              Baseline: <strong>{cf.baseline_pass_probability}</strong> → Counterfactual: <strong>{cf.counterfactual_pass_probability}</strong>
              <br />
              Δ: <strong>{cf.delta}</strong> (only <code>{cf.field}</code> = {cf.alternate_value})
            </div>
          )}
        </div>

        <div className="feature-card">
          <div className="feature-card-icon"><Users size={22} /></div>
          <h3>Peer-shadow cohort</h3>
          <p>Percentile vs <code>student_data.csv</code> training distribution — no row-level leakage.</p>
          <button type="button" className="btn btn-primary" disabled={!canRun || busy} onClick={runCohort}>
            {busy === 'coh' ? <Loader2 className="spin" size={18} /> : 'Compare my mapped row'}
          </button>
          {cohortSum?.available && (
            <p className="labs-muted">Reference N = {cohortSum.n_rows} · training pass rate ≈ {cohortSum.pass_rate_label_1 != null ? `${(cohortSum.pass_rate_label_1 * 100).toFixed(1)}%` : '—'}</p>
          )}
          {cohortCmp?.available && (
            <ul className="labs-list">
              {Object.entries(cohortCmp.percentiles || {}).map(([k, v]) => (
                <li key={k}>
                  <strong>{k}</strong>: ~{v}th percentile vs training
                </li>
              ))}
            </ul>
          )}
        </div>

        <div className="feature-card">
          <div className="feature-card-icon feature-card-icon--violet"><Target size={22} /></div>
          <h3>Goal seek (last payload)</h3>
          <p>Same grid search as Predictor — here it uses your <strong>last successful</strong> mapped payload.</p>
          <div className="labs-cf-row">
            <label>
              Field
              <select className="input-like" value={gsField} onChange={(e) => setGsField(e.target.value)}>
                {GS_FIELDS.map((f) => (
                  <option key={f.key} value={f.key}>{f.label}</option>
                ))}
              </select>
            </label>
            <label>
              Target pass %
              <input type="number" className="input-like" value={gsTargetPct} min={1} max={99} step={1} onChange={(e) => setGsTargetPct(e.target.value)} />
            </label>
            <label>
              Lo
              <input type="text" className="input-like" value={gsLo} onChange={(e) => setGsLo(e.target.value)} style={{ width: 64 }} />
            </label>
            <label>
              Hi
              <input type="text" className="input-like" value={gsHi} onChange={(e) => setGsHi(e.target.value)} style={{ width: 64 }} />
            </label>
            <label>
              Steps
              <input type="text" className="input-like" value={gsSteps} onChange={(e) => setGsSteps(e.target.value)} style={{ width: 56 }} />
            </label>
            <button type="button" className="btn btn-primary" disabled={!canRun || busy} onClick={runGoalSeek}>
              {busy === 'gs' ? <Loader2 className="spin" size={18} /> : 'Run'}
            </button>
          </div>
          {gsOut && (
            <div className="labs-pre">
              Best <code>{gsOut.field}</code> ≈ <strong>{gsOut.best_value}</strong> → p ≈ {(gsOut.best_pass_probability * 100).toFixed(1)}%
            </div>
          )}
        </div>

        <div className="feature-card">
          <div className="feature-card-icon"><Table2 size={22} /></div>
          <h3>Batch predict</h3>
          <p>POST <code>/predict/batch</code> — paste a JSON <strong>array</strong> of row objects (max 200). Omitted fields default to 0.</p>
          <textarea
            className="labs-batch-textarea"
            value={batchText}
            onChange={(e) => setBatchText(e.target.value)}
            rows={5}
            spellCheck={false}
          />
          <button type="button" className="btn btn-outline" style={{ marginTop: 8 }} disabled={busy} onClick={runBatch}>
            {busy === 'batch' ? <Loader2 className="spin" size={18} /> : 'Run batch'}
          </button>
          {batchOut?.results && (
            <div className="labs-pre labs-mono" style={{ marginTop: 10 }}>
              {batchOut.results.slice(0, 8).map((r, i) => (
                <div key={i}>
                  #{i + 1} {r.prediction} p={(r.pass_probability ?? r.probability ?? 0).toFixed(4)}
                </div>
              ))}
              {batchOut.results.length > 8 && <div>… {batchOut.results.length} total</div>}
            </div>
          )}
        </div>

        <div className="feature-card">
          <div className="feature-card-icon"><Fingerprint size={22} /></div>
          <h3>Paper-trail manifest</h3>
          <p>Nonce + model fingerprint for coursework audit (pair with PDF export).</p>
          {manifest && (
            <div className="labs-pre labs-mono">
              nonce: {manifest.export_nonce}
              <br />
              model_sha256_prefix: {manifest.model_sha256_prefix}
            </div>
          )}
        </div>

        <div className="feature-card">
          <div className="feature-card-icon feature-card-icon--violet"><Sparkles size={22} /></div>
          <h3>Shadow model vs BPNN</h3>
          <p>
            A simple logistic on z-scores vs <code>student_data.csv</code> — a second numeric opinion for disagreement checks (not a trained ensemble).
          </p>
          <button type="button" className="btn btn-primary" disabled={!canRun || busy} onClick={runShadow}>
            {busy === 'shadow' ? <Loader2 className="spin" size={18} /> : 'Compare shadow vs BPNN'}
          </button>
          {shadow?.available && (
            <div className="labs-pre">
              BPNN p(pass): <strong>{shadow.bpnn_pass_probability}</strong>
              <br />
              Shadow p: <strong>{shadow.shadow_pass_probability}</strong>
              <br />
              |Δ|: <strong>{shadow.disagreement}</strong>
              <br />
              <span className="labs-muted">{shadow.note}</span>
            </div>
          )}
          {shadow && !shadow.available && <p className="labs-muted">{shadow.reason || 'Unavailable'}</p>}
        </div>

        <div className="feature-card">
          <div className="feature-card-icon"><ListOrdered size={22} /></div>
          <h3>Input choreography (on Predictor)</h3>
          <p>
            Enable <strong>Guided entry</strong> on Performance Predictor — stepwise panels (sleep → stress → academics → subjects) so inputs feel like an interview, not a wall of fields.
          </p>
          {typeof onNavigate === 'function' && (
            <button type="button" className="btn btn-outline" onClick={() => onNavigate('predict')}>
              Open Performance Predictor <ArrowRight size={16} style={{ verticalAlign: '-2px', marginLeft: 4 }} />
            </button>
          )}
        </div>
      </div>

      <style>{`
        .labs-page .labs-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 16px; }
        .labs-page .labs-pre { margin-top: 12px; font-size: 13px; line-height: 1.5; background: var(--card-inner-bg, rgba(0,0,0,0.04)); padding: 10px; border-radius: 8px; }
        .labs-page .labs-mono { font-family: ui-monospace, monospace; font-size: 12px; word-break: break-all; }
        .labs-page .labs-list { margin: 8px 0 0 16px; padding: 0; }
        .labs-page .labs-list li { margin-bottom: 6px; }
        .labs-page .labs-muted { font-size: 12px; opacity: 0.85; margin-top: 8px; }
        .labs-page .labs-cf-row { display: flex; flex-wrap: wrap; gap: 12px; align-items: flex-end; margin-top: 8px; }
        .labs-page .labs-cf-row label { display: flex; flex-direction: column; gap: 4px; font-size: 12px; }
        .labs-page .input-like { padding: 8px 10px; border-radius: 8px; border: 1px solid var(--border-color, #e2e8f0); background: var(--input-bg, #fff); color: inherit; min-width: 120px; }
        .labs-page .labs-batch-textarea { width: 100%; margin-top: 8px; padding: 10px; border-radius: 8px; border: 1px solid var(--border-color, #e2e8f0); background: var(--input-bg, #fff); color: inherit; font-family: ui-monospace, monospace; font-size: 12px; resize: vertical; box-sizing: border-box; }
        .labs-page .labs-calib-table { width: 100%; border-collapse: collapse; font-size: 12px; margin-top: 8px; }
        .labs-page .labs-calib-table th, .labs-page .labs-calib-table td { border: 1px solid var(--border-color, #e2e8f0); padding: 6px 8px; text-align: left; }
        .labs-page .labs-calib-table th { background: var(--card-inner-bg, rgba(0,0,0,0.04)); font-weight: 600; }
        .labs-page .spin { animation: labs-spin 0.8s linear infinite; }
        @keyframes labs-spin { to { transform: rotate(360deg); } }
      `}</style>
    </div>
  );
}
