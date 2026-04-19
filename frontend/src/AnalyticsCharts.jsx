/**
 * Extra analytics charts for the Analytics page.
 *
 * All derived from `predictionHistory` (runs produced by the Predictor + CSV import).
 * Pure SVG, no new deps. Charts degrade gracefully when sample size is too small.
 *
 * Includes:
 *   1. Threshold slider (what-if) — recount Pass/Fail as threshold varies.
 *   2. Pass probability histogram (stacked Pass/Fail).
 *   3. Confidence vs subject avg scatter (+ linear fit).
 *   4. Source split donut (Predictor vs CSV).
 *   5. Run-over-run confidence delta bars.
 *   6. Recent-risk gauge (fail rate over last 10).
 *   7. Hour-of-day avg confidence strip.
 *   8. Weekday pass-rate heat strip.
 *   9. Sleep vs stress quadrant scatter.
 *  10. Study hours vs confidence scatter.
 *  11. Attendance bucket pass rate bars.
 *  12. Feature completeness stacked bar.
 */
import React, { useMemo, useState } from 'react';

const PASS = '#10b981';
const FAIL = '#ef4444';
const NEUTRAL = '#64748b';
const ACCENT = '#3b82f6';

function safeNumber(v) {
  if (v == null) return null;
  const n = typeof v === 'number' ? v : Number(v);
  return Number.isFinite(n) ? n : null;
}

function parseAtDate(run) {
  if (!run?.atLabel) return null;
  const d = new Date(run.atLabel);
  return Number.isFinite(d.getTime()) ? d : null;
}

function runPassProb(run) {
  if (typeof run.passProb === 'number' && Number.isFinite(run.passProb)) return run.passProb;
  if (typeof run.confidence === 'number' && Number.isFinite(run.confidence)) return run.confidence / 100;
  return null;
}

function linearFit(points) {
  const n = points.length;
  if (n < 2) return null;
  const mx = points.reduce((s, p) => s + p.x, 0) / n;
  const my = points.reduce((s, p) => s + p.y, 0) / n;
  let num = 0;
  let den = 0;
  for (const p of points) {
    num += (p.x - mx) * (p.y - my);
    den += (p.x - mx) ** 2;
  }
  if (den === 0) return null;
  const slope = num / den;
  const intercept = my - slope * mx;
  return { slope, intercept };
}

function ChartCard({ title, subtitle, empty, children, wide }) {
  return (
    <div className={`ax-card ${wide ? 'ax-card--wide' : ''}`}>
      <div className="ax-card-head">
        <strong>{title}</strong>
        {subtitle ? <span>{subtitle}</span> : null}
      </div>
      {empty ? <div className="ax-empty">{empty}</div> : <div className="ax-card-body">{children}</div>}
    </div>
  );
}

/* ── 1. Threshold slider ── */
function ThresholdSlider({ runs }) {
  const [thr, setThr] = useState(0.5);
  const probs = useMemo(
    () => runs.map(runPassProb).filter((x) => typeof x === 'number'),
    [runs],
  );
  if (!probs.length) return <div className="ax-empty">Run some predictions first.</div>;
  const passNow = probs.filter((p) => p >= thr).length;
  const failNow = probs.length - passNow;
  const passPct = Math.round((100 * passNow) / probs.length);

  return (
    <div className="ax-slider-wrap">
      <div className="ax-slider-row">
        <input
          type="range"
          min={0.02}
          max={0.98}
          step={0.01}
          value={thr}
          onChange={(e) => setThr(parseFloat(e.target.value))}
        />
        <span className="ax-slider-val">thr = {thr.toFixed(2)}</span>
      </div>
      <div className="ax-bar ax-bar--split">
        <span className="ax-bar-fill ax-bar-fill--pass" style={{ width: `${passPct}%` }}>
          {passNow > 0 ? `${passNow}` : ''}
        </span>
        <span className="ax-bar-fill ax-bar-fill--fail" style={{ width: `${100 - passPct}%` }}>
          {failNow > 0 ? `${failNow}` : ''}
        </span>
      </div>
      <div className="ax-meta">
        {passNow} Pass / {failNow} Fail over {probs.length} runs ({passPct}% Pass).
      </div>
    </div>
  );
}

/* ── 2. Pass probability histogram stacked ── */
function ProbabilityHistogram({ runs }) {
  const bins = useMemo(() => {
    const out = Array.from({ length: 10 }, () => ({ pass: 0, fail: 0 }));
    for (const r of runs) {
      const p = runPassProb(r);
      if (p == null) continue;
      const idx = Math.min(9, Math.max(0, Math.floor(p * 10)));
      if (r.outcome === 'Pass') out[idx].pass += 1;
      else if (r.outcome === 'Fail') out[idx].fail += 1;
    }
    return out;
  }, [runs]);
  const max = Math.max(1, ...bins.map((b) => b.pass + b.fail));
  const total = bins.reduce((s, b) => s + b.pass + b.fail, 0);
  if (total === 0) return <div className="ax-empty">No probability data.</div>;

  const W = 320;
  const H = 110;
  const barW = (W - 20) / 10;

  return (
    <svg viewBox={`0 0 ${W} ${H + 24}`} className="ax-svg">
      {bins.map((b, i) => {
        const x = 10 + i * barW + 2;
        const passH = (b.pass / max) * H;
        const failH = (b.fail / max) * H;
        const top = H - passH - failH;
        return (
          <g key={i}>
            <rect x={x} y={top} width={barW - 4} height={passH} fill={PASS} rx={2} />
            <rect x={x} y={top + passH} width={barW - 4} height={failH} fill={FAIL} rx={2} />
            <text x={x + (barW - 4) / 2} y={H + 14} textAnchor="middle" fontSize="9" fill="#64748b">
              {(i / 10).toFixed(1)}
            </text>
          </g>
        );
      })}
      <text x={W - 6} y={H + 14} textAnchor="end" fontSize="9" fill="#64748b">
        1.0
      </text>
    </svg>
  );
}

/* ── 3. Confidence vs subject avg scatter + linear fit ── */
function ScatterConfVsComposite({ runs }) {
  const pts = useMemo(() => {
    return runs
      .map((r) => {
        let x = safeNumber(r.composite);
        if (x == null) {
          const mathVal = safeNumber(r.numeric?.math_score);
          const sciVal = safeNumber(r.numeric?.science_score);
          const engVal = safeNumber(r.numeric?.english_score);
          const histVal = safeNumber(r.numeric?.history_score);
          const compVal = safeNumber(r.numeric?.computer_score);
          const arr = [mathVal, sciVal, engVal, histVal, compVal].filter(
            (v) => typeof v === 'number' && v > 0,
          );
          if (arr.length) x = arr.reduce((s, v) => s + v, 0) / arr.length;
          else x = safeNumber(r.numeric?.student_marks_percent);
        }
        let y = safeNumber(r.confidence);
        if (y == null) {
          const pp = safeNumber(r.passProb);
          if (pp != null) y = pp * 100;
        }
        if (x == null || y == null) return null;
        return { x, y, pass: r.outcome === 'Pass' };
      })
      .filter(Boolean);
  }, [runs]);
  if (pts.length === 0) return <div className="ax-empty">Need more labeled runs.</div>;
  const W = 320;
  const H = 150;
  const maxX = 100;
  const sx = (x) => 24 + (Math.min(maxX, Math.max(0, x)) / maxX) * (W - 34);
  const sy = (y) => H - 18 - (Math.min(100, Math.max(0, y)) / 100) * (H - 28);
  const fit = linearFit(pts);

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="ax-svg">
      <line x1={24} y1={H - 18} x2={W - 10} y2={H - 18} stroke="#94a3b8" strokeWidth={0.5} />
      <line x1={24} y1={10} x2={24} y2={H - 18} stroke="#94a3b8" strokeWidth={0.5} />
      {fit ? (
        <line
          x1={sx(0)}
          y1={sy(Math.min(100, Math.max(0, fit.intercept)))}
          x2={sx(100)}
          y2={sy(Math.min(100, Math.max(0, fit.slope * 100 + fit.intercept)))}
          stroke={ACCENT}
          strokeDasharray="4 3"
          strokeWidth={1.2}
        />
      ) : null}
      {pts.map((p, i) => (
        <circle key={i} cx={sx(p.x)} cy={sy(p.y)} r={3} fill={p.pass ? PASS : FAIL} fillOpacity={0.8} />
      ))}
      <text x={W - 10} y={H - 4} textAnchor="end" fontSize="9" fill="#64748b">subj avg %</text>
      <text x={26} y={14} fontSize="9" fill="#64748b">confidence %</text>
    </svg>
  );
}

/* ── 4. Source split donut ── */
function SourceSplitDonut({ runs }) {
  const counts = useMemo(() => {
    const csv = runs.filter((r) => r.source === 'csv').length;
    const manual = runs.length - csv;
    return { csv, manual };
  }, [runs]);
  if (!runs.length) return <div className="ax-empty">No runs.</div>;
  const total = counts.csv + counts.manual;
  const R = 36;
  const C = 2 * Math.PI * R;
  const csvFrac = counts.csv / total;
  return (
    <div className="ax-inline-center">
      <svg viewBox="0 0 110 110" className="ax-svg" style={{ maxHeight: 140 }}>
        <circle cx="55" cy="55" r={R} fill="none" stroke="rgba(100,116,139,0.15)" strokeWidth={12} />
        <circle
          cx="55"
          cy="55"
          r={R}
          fill="none"
          stroke={ACCENT}
          strokeWidth={12}
          strokeDasharray={`${C * csvFrac} ${C}`}
          transform="rotate(-90 55 55)"
          strokeLinecap="round"
        />
        <text x="55" y="50" textAnchor="middle" fontSize="14" fontWeight="700" fill="#0f172a">
          {Math.round(csvFrac * 100)}%
        </text>
        <text x="55" y="65" textAnchor="middle" fontSize="9" fill="#64748b">CSV</text>
      </svg>
      <ul className="ax-legend-list">
        <li><i className="ax-dot" style={{ background: ACCENT }} />CSV import: <strong>{counts.csv}</strong></li>
        <li><i className="ax-dot" style={{ background: NEUTRAL }} />Predictor: <strong>{counts.manual}</strong></li>
      </ul>
    </div>
  );
}

/* ── 5. Confidence delta bars ── */
function ConfidenceDelta({ runs }) {
  const series = useMemo(() => {
    const ordered = [...runs].reverse();
    const deltas = [];
    for (let i = 1; i < ordered.length; i += 1) {
      const a = safeNumber(ordered[i - 1].confidence);
      const b = safeNumber(ordered[i].confidence);
      if (a == null || b == null) continue;
      deltas.push(b - a);
    }
    return deltas.slice(-30);
  }, [runs]);
  if (series.length < 1) return <div className="ax-empty">Need ≥2 predictions.</div>;
  const W = 320;
  const H = 100;
  const barW = Math.max(4, (W - 20) / series.length - 2);
  const maxAbs = Math.max(5, ...series.map((v) => Math.abs(v)));
  const zero = H / 2;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="ax-svg">
      <line x1={10} y1={zero} x2={W - 10} y2={zero} stroke="#94a3b8" strokeWidth={0.5} />
      {series.map((v, i) => {
        const x = 10 + i * (barW + 2);
        const h = (Math.abs(v) / maxAbs) * (H / 2 - 6);
        const y = v >= 0 ? zero - h : zero;
        const color = v >= 0 ? PASS : FAIL;
        return <rect key={i} x={x} y={y} width={barW} height={h} fill={color} opacity={0.8} rx={1.5} />;
      })}
    </svg>
  );
}

/* ── 6. Recent risk gauge (fail rate in last 10) ── */
function RiskGauge({ runs }) {
  const last10 = runs.slice(0, 10);
  if (!last10.length) return <div className="ax-empty">No runs.</div>;
  const fails = last10.filter((r) => r.outcome === 'Fail').length;
  const frac = fails / last10.length;
  const angle = Math.PI * (1 - frac);
  const cx = 80;
  const cy = 74;
  const r = 54;
  const x = cx + r * Math.cos(Math.PI - angle);
  const y = cy - r * Math.sin(Math.PI - angle);
  return (
    <svg viewBox="0 0 160 96" className="ax-svg">
      <path d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`} stroke="rgba(100,116,139,0.2)" strokeWidth={10} fill="none" />
      <path
        d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${x} ${y}`}
        stroke={frac > 0.5 ? FAIL : frac > 0.25 ? '#f59e0b' : PASS}
        strokeWidth={10}
        fill="none"
        strokeLinecap="round"
      />
      <text x={cx} y={cy - 6} textAnchor="middle" fontSize="20" fontWeight="700" fill="#0f172a">
        {Math.round(frac * 100)}%
      </text>
      <text x={cx} y={cy + 10} textAnchor="middle" fontSize="9" fill="#64748b">
        Fail rate · last {last10.length}
      </text>
    </svg>
  );
}

/* ── 7. Hour-of-day avg confidence strip ── */
function HourHeatStrip({ runs }) {
  const byHour = useMemo(() => {
    const sums = Array.from({ length: 24 }, () => ({ sum: 0, n: 0 }));
    for (const r of runs) {
      const d = parseAtDate(r);
      const c = safeNumber(r.confidence);
      if (!d || c == null) continue;
      const h = d.getHours();
      sums[h].sum += c;
      sums[h].n += 1;
    }
    return sums.map((s) => (s.n ? s.sum / s.n : null));
  }, [runs]);
  const some = byHour.some((v) => v != null);
  if (!some) return <div className="ax-empty">Need recent timestamps.</div>;
  const W = 320;
  const H = 46;
  const cellW = (W - 20) / 24;
  return (
    <svg viewBox={`0 0 ${W} ${H + 24}`} className="ax-svg">
      {byHour.map((v, i) => {
        const x = 10 + i * cellW;
        if (v == null) {
          return (
            <rect key={i} x={x + 1} y={4} width={cellW - 2} height={H} fill="rgba(100,116,139,0.15)" rx={2} />
          );
        }
        const t = Math.min(1, Math.max(0, v / 100));
        const color = `rgba(37, 99, 235, ${0.15 + t * 0.75})`;
        return (
          <g key={i}>
            <rect x={x + 1} y={4} width={cellW - 2} height={H} fill={color} rx={2} />
            <text x={x + cellW / 2} y={H + 16} textAnchor="middle" fontSize="8" fill="#64748b">
              {i % 3 === 0 ? `${i}h` : ''}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

/* ── 8. Weekday pass-rate strip ── */
const WEEKDAY_LABELS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

function WeekdayPassStrip({ runs }) {
  const data = useMemo(() => {
    const bins = Array.from({ length: 7 }, () => ({ pass: 0, fail: 0 }));
    for (const r of runs) {
      const d = parseAtDate(r);
      if (!d) continue;
      const w = d.getDay();
      if (r.outcome === 'Pass') bins[w].pass += 1;
      else if (r.outcome === 'Fail') bins[w].fail += 1;
    }
    return bins.map((b, i) => ({ label: WEEKDAY_LABELS[i], pass: b.pass, fail: b.fail }));
  }, [runs]);
  const any = data.some((d) => d.pass + d.fail);
  if (!any) return <div className="ax-empty">Need timestamped runs.</div>;
  const W = 320;
  const H = 80;
  const colW = (W - 20) / 7;
  return (
    <svg viewBox={`0 0 ${W} ${H + 20}`} className="ax-svg">
      {data.map((d, i) => {
        const total = d.pass + d.fail;
        const passFrac = total ? d.pass / total : 0;
        const x = 10 + i * colW + 4;
        const tint = total ? `rgba(16, 185, 129, ${0.2 + passFrac * 0.65})` : 'rgba(100,116,139,0.12)';
        return (
          <g key={d.label}>
            <rect x={x} y={4} width={colW - 8} height={H - 20} fill={tint} rx={3} />
            <text x={x + (colW - 8) / 2} y={H - 20 / 2} textAnchor="middle" fontSize="10" fontWeight="600" fill="#0f172a">
              {total ? `${Math.round(passFrac * 100)}%` : '—'}
            </text>
            <text x={x + (colW - 8) / 2} y={H + 10} textAnchor="middle" fontSize="9" fill="#64748b">{d.label}</text>
          </g>
        );
      })}
    </svg>
  );
}

/* ── 9. Sleep vs stress quadrant scatter ── */
function SleepStressQuadrant({ runs }) {
  const pts = useMemo(() => {
    return runs
      .map((r) => {
        const sleep = safeNumber(r.numeric?.sleep_hours);
        const stress = safeNumber(r.numeric?.mental_stress);
        if (sleep == null || stress == null || (sleep === 0 && stress === 0)) return null;
        return { sleep, stress, pass: r.outcome === 'Pass' };
      })
      .filter(Boolean);
  }, [runs]);
  if (!pts.length) return <div className="ax-empty">Add sleep + stress values to fill this.</div>;
  const W = 320;
  const H = 150;
  const sx = (v) => 24 + (Math.min(10, Math.max(0, v)) / 10) * (W - 34);
  const sy = (v) => H - 18 - (Math.min(10, Math.max(0, v)) / 10) * (H - 28);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="ax-svg">
      <line x1={24} y1={H - 18} x2={W - 10} y2={H - 18} stroke="#94a3b8" strokeWidth={0.5} />
      <line x1={24} y1={10} x2={24} y2={H - 18} stroke="#94a3b8" strokeWidth={0.5} />
      <line x1={sx(5)} y1={10} x2={sx(5)} y2={H - 18} stroke="rgba(100,116,139,0.3)" strokeDasharray="3 3" />
      <line x1={24} y1={sy(5)} x2={W - 10} y2={sy(5)} stroke="rgba(100,116,139,0.3)" strokeDasharray="3 3" />
      {pts.map((p, i) => (
        <circle key={i} cx={sx(p.sleep)} cy={sy(p.stress)} r={3.5} fill={p.pass ? PASS : FAIL} fillOpacity={0.75} />
      ))}
      <text x={W - 10} y={H - 4} textAnchor="end" fontSize="9" fill="#64748b">sleep (h)</text>
      <text x={26} y={14} fontSize="9" fill="#64748b">stress</text>
    </svg>
  );
}

/* ── 10. Study hours vs confidence scatter ── */
function StudyVsConf({ runs }) {
  const pts = useMemo(() => {
    return runs
      .map((r) => {
        const x = safeNumber(r.numeric?.study_hours_daily);
        const y = safeNumber(r.confidence);
        if (x == null || y == null || x === 0) return null;
        return { x, y, pass: r.outcome === 'Pass' };
      })
      .filter(Boolean);
  }, [runs]);
  if (!pts.length) return <div className="ax-empty">Need study_hours_daily values.</div>;
  const W = 320;
  const H = 150;
  const maxX = Math.max(12, ...pts.map((p) => p.x));
  const sx = (x) => 28 + (x / maxX) * (W - 38);
  const sy = (y) => H - 18 - (Math.min(100, Math.max(0, y)) / 100) * (H - 28);
  const fit = linearFit(pts);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="ax-svg">
      <line x1={28} y1={H - 18} x2={W - 10} y2={H - 18} stroke="#94a3b8" strokeWidth={0.5} />
      <line x1={28} y1={10} x2={28} y2={H - 18} stroke="#94a3b8" strokeWidth={0.5} />
      {fit ? (
        <line
          x1={sx(0)}
          y1={sy(Math.min(100, Math.max(0, fit.intercept)))}
          x2={sx(maxX)}
          y2={sy(Math.min(100, Math.max(0, fit.slope * maxX + fit.intercept)))}
          stroke={ACCENT}
          strokeDasharray="4 3"
          strokeWidth={1.2}
        />
      ) : null}
      {pts.map((p, i) => (
        <circle key={i} cx={sx(p.x)} cy={sy(p.y)} r={3} fill={p.pass ? PASS : FAIL} fillOpacity={0.8} />
      ))}
      <text x={W - 10} y={H - 4} textAnchor="end" fontSize="9" fill="#64748b">study h / day</text>
      <text x={30} y={14} fontSize="9" fill="#64748b">confidence %</text>
    </svg>
  );
}

/* ── 11. Attendance bucket pass rate ── */
function AttendanceBuckets({ runs }) {
  const bins = useMemo(() => {
    const out = [
      { label: '<60%', min: 0, max: 0.6, pass: 0, n: 0 },
      { label: '60–75%', min: 0.6, max: 0.75, pass: 0, n: 0 },
      { label: '75–85%', min: 0.75, max: 0.85, pass: 0, n: 0 },
      { label: '85–95%', min: 0.85, max: 0.95, pass: 0, n: 0 },
      { label: '≥95%', min: 0.95, max: 10, pass: 0, n: 0 },
    ];
    for (const r of runs) {
      let att = safeNumber(r.numeric?.attendance_rate);
      if (att == null) continue;
      if (att > 1) att /= 100;
      for (const b of out) {
        if (att >= b.min && att < b.max) {
          b.n += 1;
          if (r.outcome === 'Pass') b.pass += 1;
          break;
        }
      }
    }
    return out;
  }, [runs]);
  const any = bins.some((b) => b.n);
  if (!any) return <div className="ax-empty">Need attendance values.</div>;
  const W = 320;
  const H = 110;
  const colW = (W - 20) / bins.length;
  return (
    <svg viewBox={`0 0 ${W} ${H + 18}`} className="ax-svg">
      {bins.map((b, i) => {
        const frac = b.n ? b.pass / b.n : 0;
        const barH = frac * (H - 20);
        const x = 10 + i * colW + 4;
        return (
          <g key={b.label}>
            <rect x={x} y={H - 20 - barH} width={colW - 8} height={barH} fill={PASS} rx={2} />
            <rect
              x={x}
              y={H - 20 - (H - 20)}
              width={colW - 8}
              height={H - 20 - barH}
              fill="rgba(100,116,139,0.1)"
              rx={2}
            />
            <text x={x + (colW - 8) / 2} y={H + 12} textAnchor="middle" fontSize="9" fill="#64748b">{b.label}</text>
            <text x={x + (colW - 8) / 2} y={H - 24 - barH} textAnchor="middle" fontSize="10" fill="#0f172a" fontWeight="600">
              {b.n ? `${Math.round(frac * 100)}%` : '—'}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

/* ── 12. Feature completeness stacked bars ── */
const COMPLETENESS_KEYS = [
  ['attendance_rate', 'attendance'],
  ['study_hours_daily', 'study h'],
  ['sleep_hours', 'sleep h'],
  ['mental_stress', 'stress'],
  ['previous_gpa', 'prev GPA'],
  ['assignment_avg', 'assignments'],
  ['quiz_avg', 'quizzes'],
  ['project_score', 'projects'],
  ['standardized_exam_score', 'exam'],
  ['math_score', 'math'],
];

function FeatureCompleteness({ runs }) {
  const rows = useMemo(() => {
    return COMPLETENESS_KEYS.map(([key, label]) => {
      let filled = 0;
      for (const r of runs) {
        const raw = r.formData?.[key];
        if (raw != null && String(raw).trim() !== '' && String(raw) !== '0') filled += 1;
      }
      return { key, label, filled, total: runs.length };
    });
  }, [runs]);
  if (!runs.length) return <div className="ax-empty">No runs.</div>;
  return (
    <ul className="ax-complete-list">
      {rows.map((r) => {
        const pct = r.total ? Math.round((100 * r.filled) / r.total) : 0;
        return (
          <li key={r.key}>
            <span className="ax-complete-label">{r.label}</span>
            <span className="ax-bar">
              <span className="ax-bar-fill" style={{ width: `${pct}%`, background: ACCENT }} />
            </span>
            <span className="ax-complete-val">
              {r.filled}/{r.total} ({pct}%)
            </span>
          </li>
        );
      })}
    </ul>
  );
}

export default function AnalyticsExtraGraphs({ predictionHistory }) {
  const runs = predictionHistory ?? [];
  if (!runs.length) {
    return (
      <div className="ax-extra">
        <h3>More analytics</h3>
        <div className="ax-empty">Run the predictor or import a CSV to populate these charts.</div>
      </div>
    );
  }
  return (
    <div className="ax-extra">
      <h3>More analytics</h3>
      <div className="ax-grid">
        <ChartCard title="What-if threshold" subtitle="Recount Pass/Fail as the decision threshold moves" wide>
          <ThresholdSlider runs={runs} />
        </ChartCard>
        <ChartCard title="Pass probability histogram" subtitle="Stacked Pass / Fail per 0.1 bin">
          <ProbabilityHistogram runs={runs} />
        </ChartCard>
        <ChartCard title="Confidence vs subject avg" subtitle="Dashed line = linear fit across runs">
          <ScatterConfVsComposite runs={runs} />
        </ChartCard>
        <ChartCard title="Source split" subtitle="Predictor vs CSV-imported rows">
          <SourceSplitDonut runs={runs} />
        </ChartCard>
        <ChartCard title="Confidence delta" subtitle="Run-over-run change in confidence %">
          <ConfidenceDelta runs={runs} />
        </ChartCard>
        <ChartCard title="Recent risk gauge" subtitle="Fail rate across the last 10 runs">
          <RiskGauge runs={runs} />
        </ChartCard>
        <ChartCard title="Hour of day" subtitle="Average confidence per 24h bucket" wide>
          <HourHeatStrip runs={runs} />
        </ChartCard>
        <ChartCard title="Weekday pass rate" subtitle="Share of Pass per weekday" wide>
          <WeekdayPassStrip runs={runs} />
        </ChartCard>
        <ChartCard title="Sleep vs stress" subtitle="Dots are runs; colored by outcome">
          <SleepStressQuadrant runs={runs} />
        </ChartCard>
        <ChartCard title="Study hours vs confidence" subtitle="Dashed line = linear fit">
          <StudyVsConf runs={runs} />
        </ChartCard>
        <ChartCard title="Attendance → pass rate" subtitle="Pass rate inside attendance buckets">
          <AttendanceBuckets runs={runs} />
        </ChartCard>
        <ChartCard title="Feature completeness" subtitle="How often each input is supplied vs blank">
          <FeatureCompleteness runs={runs} />
        </ChartCard>
      </div>
    </div>
  );
}
