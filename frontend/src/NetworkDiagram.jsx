/**
 * Animated BPNN architecture viewer for the Analytics page.
 *
 * - `/meta/network` returns subsampled weights/biases per Dense layer (10 nodes/layer max).
 * - `/predict/trace` (optional) returns per-layer activations for a given payload so the
 *   diagram lights up neurons and sends pulses along edges in intensity order.
 *
 * Colors follow the user's reference image: blue = positive weight, red = negative,
 * opacity/width = |weight|. A sequential CSS pulse animates signal flow L0 → L1 → L2 → L3.
 */
import React, { useEffect, useMemo, useRef, useState } from 'react';
import axios from 'axios';
import { apiUrl } from './apiRoutes.js';

const COLS_PAD_X = 90;
const COL_GAP = 160;
const ROW_GAP = 36;
const NODE_R = 10;
const OUTPUT_R = 14;

function shortLabel(name, max = 14) {
  if (!name) return '';
  if (name.length <= max) return name;
  return `${name.slice(0, max - 1)}…`;
}

function formatLayerName(name, i, total) {
  if (i === total - 1) return 'Output';
  return `Hidden ${i + 1}`;
}

/** Normalize |w| across all edges so stroke width / opacity is comparable. */
function collectWeightStats(layers) {
  let max = 1e-6;
  for (const L of layers) {
    for (const row of L.weights) {
      for (const w of row) {
        const a = Math.abs(Number(w));
        if (a > max) max = a;
      }
    }
  }
  return { maxAbs: max };
}

function edgeStyle(w, maxAbs) {
  const a = Math.abs(Number(w));
  const t = Math.min(1, a / (maxAbs || 1));
  const color = Number(w) >= 0 ? '#2563eb' : '#dc2626';
  return {
    stroke: color,
    strokeWidth: 0.4 + t * 2.6,
    opacity: 0.08 + t * 0.85,
  };
}

function sanitizePayload(payload) {
  if (!payload || typeof payload !== 'object') return null;
  const out = {};
  for (const [k, v] of Object.entries(payload)) {
    if (v == null || v === '') continue;
    const n = typeof v === 'number' ? v : Number(v);
    if (!Number.isFinite(n)) continue;
    out[k] = n;
  }
  return Object.keys(out).length ? out : null;
}

export default function NetworkDiagram({ lastPredictPayload }) {
  const [net, setNet] = useState(null);
  const [err, setErr] = useState(null);
  const [loading, setLoading] = useState(true);
  const [trace, setTrace] = useState(null);
  const [tracing, setTracing] = useState(false);
  const [pulse, setPulse] = useState(0);
  const [autoLive, setAutoLive] = useState(true);
  const timerRef = useRef(null);
  const liveDebounceRef = useRef(null);
  const cancelTokenRef = useRef(null);
  const lastPayloadKeyRef = useRef('');

  useEffect(() => {
    let cancelled = false;
    axios
      .get(apiUrl('meta/network'))
      .then((res) => {
        if (!cancelled) setNet(res.data);
      })
      .catch((e) => {
        if (!cancelled) setErr(e?.response?.data?.detail || e?.message || 'Failed to load network');
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!net) return undefined;
    if (timerRef.current) clearInterval(timerRef.current);
    const steps = net.layers.length;
    timerRef.current = setInterval(() => {
      setPulse((p) => (p + 1) % steps);
    }, 900);
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [net]);

  const runTraceFor = async (payload) => {
    const clean = sanitizePayload(payload);
    if (!clean) return;
    if (cancelTokenRef.current) cancelTokenRef.current.cancel('superseded');
    const source = axios.CancelToken.source();
    cancelTokenRef.current = source;
    setTracing(true);
    try {
      const res = await axios.post(apiUrl('predict/trace'), clean, { cancelToken: source.token });
      setTrace(res.data);
      setErr(null);
    } catch (e) {
      if (axios.isCancel?.(e) || e?.message === 'superseded') return;
      setErr(e?.response?.data?.detail || e?.message || 'Trace failed');
    } finally {
      setTracing(false);
    }
  };

  const runTrace = () => runTraceFor(lastPredictPayload);
  const clearTrace = () => {
    setTrace(null);
    setAutoLive(false);
  };

  // Auto-trace on payload change when live mode is on (debounced).
  useEffect(() => {
    if (!autoLive) return undefined;
    const clean = sanitizePayload(lastPredictPayload);
    const key = clean ? JSON.stringify(clean) : '';
    if (!key) return undefined;
    if (key === lastPayloadKeyRef.current) return undefined;
    lastPayloadKeyRef.current = key;
    if (liveDebounceRef.current) clearTimeout(liveDebounceRef.current);
    liveDebounceRef.current = setTimeout(() => runTraceFor(clean), 350);
    return () => {
      if (liveDebounceRef.current) clearTimeout(liveDebounceRef.current);
    };
  }, [lastPredictPayload, autoLive]);

  const geom = useMemo(() => {
    if (!net) return null;
    const columns = [
      {
        kind: 'input',
        count: net.feature_names_shown.length,
        labels: net.feature_names_shown,
      },
      ...net.layers.map((L, i) => ({
        kind: i === net.layers.length - 1 ? 'output' : 'hidden',
        count: L.units_shown,
        title: formatLayerName(L.name, i, net.layers.length),
        unitsShown: L.units_shown,
        unitsTotal: L.units_total,
      })),
    ];
    const height = Math.max(...columns.map((c) => c.count)) * ROW_GAP + 80;
    const width = COLS_PAD_X * 2 + (columns.length - 1) * COL_GAP;

    const positions = columns.map((col, ci) => {
      const x = COLS_PAD_X + ci * COL_GAP;
      const colHeight = (col.count - 1) * ROW_GAP;
      const yStart = (height - colHeight) / 2;
      const nodes = Array.from({ length: col.count }, (_, i) => ({
        x,
        y: yStart + i * ROW_GAP,
      }));
      return { ...col, nodes };
    });

    return { columns: positions, width, height };
  }, [net]);

  if (loading) return <div className="nn-panel nn-panel--empty">Loading network…</div>;
  if (err && !net) {
    return (
      <div className="nn-panel nn-panel--empty">
        <strong>Network diagram unavailable.</strong>
        <div style={{ fontSize: 12, marginTop: 6 }}>{String(err)}</div>
      </div>
    );
  }
  if (!net || !geom) return null;

  const stats = collectWeightStats(net.layers);

  const inputActs = trace?.input_values_scaled || null;
  const layerActs = trace?.layers_activations || null;

  const nodeFill = (columnIndex, row) => {
    if (columnIndex === 0 && inputActs) {
      const v = inputActs[row];
      const mag = Math.min(1, Math.abs(v));
      return `rgba(37, 99, 235, ${0.25 + mag * 0.6})`;
    }
    if (columnIndex > 0 && layerActs) {
      const layer = layerActs[columnIndex - 1];
      if (layer == null) return columnIndex === geom.columns.length - 1 ? '#10b981' : '#38bdf8';
      const raw = layer[row] ?? 0;
      const bound = Math.min(1, Math.abs(raw) / (columnIndex === geom.columns.length - 1 ? 1 : 4));
      if (columnIndex === geom.columns.length - 1) {
        return `rgba(16, 185, 129, ${0.35 + bound * 0.55})`;
      }
      return `rgba(56, 189, 248, ${0.3 + bound * 0.6})`;
    }
    if (columnIndex === 0) return '#64748b';
    if (columnIndex === geom.columns.length - 1) return '#10b981';
    return '#38bdf8';
  };

  return (
    <div className="nn-panel">
      <div className="nn-panel-header">
        <div>
          <h3>BPNN architecture &amp; weights</h3>
          <p>
            Blue = positive weight, red = negative. Line width and opacity track |weight|. Showing up to{' '}
            {net.max_nodes_per_layer} nodes per layer ({net.feature_names_shown.length}/{net.total_features} inputs).
          </p>
        </div>
        <div className="nn-panel-actions">
          <label className="nn-live-toggle" title="Auto-trace whenever the predictor inputs change">
            <input
              type="checkbox"
              checked={autoLive}
              onChange={(e) => setAutoLive(e.target.checked)}
            />
            <span>Live from inputs</span>
            {tracing ? <span className="nn-live-dot" /> : null}
          </label>
          <button
            type="button"
            className="btn btn-sm btn-outline"
            onClick={runTrace}
            disabled={!lastPredictPayload || tracing}
            title={lastPredictPayload ? 'Force a trace now' : 'Type any input in Predictor first'}
          >
            {tracing ? 'Tracing…' : 'Trace now'}
          </button>
          {trace ? (
            <button type="button" className="btn btn-sm btn-ghost" onClick={clearTrace}>
              Clear
            </button>
          ) : null}
        </div>
      </div>

      <div className="nn-panel-body">
        <svg
          role="img"
          aria-label="BPNN network diagram"
          viewBox={`0 0 ${geom.width} ${geom.height}`}
          preserveAspectRatio="xMidYMid meet"
          className="nn-svg"
        >
          {geom.columns.map((col, ci) => {
            if (ci === geom.columns.length - 1) return null;
            const L = net.layers[ci];
            const next = geom.columns[ci + 1];
            const active = ci === pulse;
            return (
              <g key={`edges-${ci}`} className={`nn-edges ${active ? 'nn-edges--active' : ''}`}>
                {col.nodes.map((a, i) =>
                  next.nodes.map((b, j) => {
                    const w = L.weights?.[i]?.[j];
                    if (w == null) return null;
                    const es = edgeStyle(w, stats.maxAbs);
                    return (
                      <line
                        key={`e-${ci}-${i}-${j}`}
                        x1={a.x + NODE_R}
                        y1={a.y}
                        x2={b.x - NODE_R}
                        y2={b.y}
                        stroke={es.stroke}
                        strokeWidth={es.strokeWidth}
                        strokeOpacity={es.opacity}
                      />
                    );
                  }),
                )}
              </g>
            );
          })}

          {geom.columns.map((col, ci) => {
            if (ci === geom.columns.length - 1) return null;
            const L = net.layers[ci];
            const next = geom.columns[ci + 1];
            const active = ci === pulse;
            if (!active) return null;
            return (
              <g key={`pulse-${ci}`} className="nn-pulses">
                {col.nodes.map((a, i) =>
                  next.nodes.map((b, j) => {
                    const w = L.weights?.[i]?.[j];
                    if (w == null) return null;
                    const mag = Math.abs(Number(w));
                    if (mag < stats.maxAbs * 0.35) return null;
                    const color = Number(w) >= 0 ? '#2563eb' : '#dc2626';
                    const dur = 0.9 + (1 - mag / stats.maxAbs) * 0.4;
                    return (
                      <circle key={`p-${ci}-${i}-${j}`} r={2.6} fill={color} opacity={0.95}>
                        <animate
                          attributeName="cx"
                          from={a.x + NODE_R}
                          to={b.x - NODE_R}
                          dur={`${dur}s`}
                          repeatCount="indefinite"
                        />
                        <animate
                          attributeName="cy"
                          from={a.y}
                          to={b.y}
                          dur={`${dur}s`}
                          repeatCount="indefinite"
                        />
                      </circle>
                    );
                  }),
                )}
              </g>
            );
          })}

          {geom.columns.map((col, ci) => {
            const isOutput = ci === geom.columns.length - 1;
            const isInput = ci === 0;
            const title = isInput
              ? `Input · ${net.feature_names_shown.length}/${net.total_features} shown`
              : `${col.title} · ${col.unitsShown}/${col.unitsTotal} shown`;
            return (
              <g key={`col-${ci}`}>
                <text x={col.nodes[0].x} y={22} textAnchor="middle" className="nn-col-title">
                  {isInput ? 'Input' : col.title}
                </text>
                <text x={col.nodes[0].x} y={38} textAnchor="middle" className="nn-col-subtitle">
                  {title.split('·')[1]?.trim() || ''}
                </text>
                {col.nodes.map((p, i) => (
                  <g key={`n-${ci}-${i}`}>
                    <circle
                      cx={p.x}
                      cy={p.y}
                      r={isOutput ? OUTPUT_R : NODE_R}
                      fill={nodeFill(ci, i)}
                      stroke="#0f172a"
                      strokeOpacity={0.35}
                      strokeWidth={1}
                    />
                    {isInput ? (
                      <text x={p.x - NODE_R - 6} y={p.y + 4} textAnchor="end" className="nn-input-label">
                        {shortLabel(net.feature_names_shown[i])}
                      </text>
                    ) : null}
                  </g>
                ))}
              </g>
            );
          })}

          {trace ? (
            <g>
              <text
                x={geom.width - 8}
                y={geom.height - 10}
                textAnchor="end"
                className="nn-trace-readout"
              >
                P(Pass) = {Math.round(trace.pass_probability * 1000) / 10}%
              </text>
            </g>
          ) : null}
        </svg>

        <div className="nn-legend">
          <span>
            <i className="nn-legend-swatch" style={{ background: '#2563eb' }} />
            positive weight
          </span>
          <span>
            <i className="nn-legend-swatch" style={{ background: '#dc2626' }} />
            negative weight
          </span>
          <span>
            Pulse: layer <strong>{pulse + 1}</strong> / {net.layers.length}
          </span>
          {trace ? <span className="nn-legend-dot">activations from trace</span> : null}
        </div>

        {trace?.explanation ? <SolveTrace trace={trace} /> : null}
      </div>
    </div>
  );
}

function fmt(x, d = 3) {
  if (x == null || !Number.isFinite(Number(x))) return '—';
  const n = Number(x);
  if (Math.abs(n) >= 100) return n.toFixed(0);
  if (Math.abs(n) >= 10) return n.toFixed(1);
  return n.toFixed(d);
}

function pct(x, d = 1) {
  if (x == null || !Number.isFinite(Number(x))) return '—';
  return `${(Number(x) * 100).toFixed(d)}%`;
}

function SolveTrace({ trace }) {
  const { explanation, pass_probability: p } = trace;
  const { top_inputs, layer_stats, worked_example, decision, total_features } = explanation;
  const labelClass = decision.label === 'Pass' ? 'solve-badge solve-badge--pass' : 'solve-badge solve-badge--fail';
  const worked = worked_example;

  const sigTerm = Number(worked.z);
  const sigFormula = `relu(${fmt(worked.z, 3)}) = ${fmt(worked.activation, 3)}`;
  const prodSum = worked.top_terms.reduce((s, t) => s + t.product, 0);
  const others = Number(worked.z) - worked.bias - prodSum;

  return (
    <div className="solve-trace">
      <div className="solve-trace-head">
        <h4>How the model solved it</h4>
        <p>
          Forward pass over {total_features} features · threshold {fmt(decision.decision_threshold, 2)} · verdict{' '}
          <span className={labelClass}>{decision.label}</span>
        </p>
      </div>

      <div className="solve-grid">
        <section className="solve-card">
          <header>
            <strong>Step 1 · Top input contributions to Hidden 1</strong>
            <span>|weight| × |scaled value|, summed across all 64 H1 neurons</span>
          </header>
          <table>
            <thead>
              <tr>
                <th>Feature</th>
                <th>Raw</th>
                <th>Scaled</th>
                <th>Contribution</th>
              </tr>
            </thead>
            <tbody>
              {top_inputs.map((t) => (
                <tr key={t.name}>
                  <td>
                    <code>{t.name}</code>
                    {t.was_imputed ? <span className="solve-pill"> median</span> : null}
                  </td>
                  <td>{fmt(t.value_raw)}</td>
                  <td>{fmt(t.value_scaled)}</td>
                  <td>
                    <div className="solve-bar">
                      <div
                        className="solve-bar-fill"
                        style={{ width: `${Math.min(100, (t.contribution / (top_inputs[0]?.contribution || 1)) * 100)}%` }}
                      />
                      <span>{fmt(t.contribution)}</span>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>

        <section className="solve-card">
          <header>
            <strong>Step 2 · Hidden layers (ReLU)</strong>
            <span>How many of each layer&apos;s neurons fired</span>
          </header>
          <table>
            <thead>
              <tr>
                <th>Layer</th>
                <th>Active / total</th>
                <th>Mean a</th>
                <th>Max a</th>
              </tr>
            </thead>
            <tbody>
              {layer_stats.map((L, i) => (
                <tr key={L.name}>
                  <td>
                    Hidden {i + 1} <code>({L.name})</code>
                  </td>
                  <td>
                    <strong>{L.active_units}</strong> / {L.units_total}{' '}
                    <span style={{ color: 'var(--text-light)' }}>({pct(L.active_pct, 0)})</span>
                  </td>
                  <td>{fmt(L.mean_activation)}</td>
                  <td>{fmt(L.max_activation)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>

        <section className="solve-card solve-card--wide">
          <header>
            <strong>Step 3 · Worked example — H1 neuron #{worked.neuron_index}</strong>
            <span>z = Σ wᵢ·xᵢ + b, then a = ReLU(z)</span>
          </header>
          <div className="solve-worked">
            <div className="solve-worked-terms">
              <span className="solve-worked-line">
                <code>z</code> = (top terms)
                {worked.top_terms.map((t) => (
                  <span key={t.feature} className="solve-term">
                    <code>{t.feature}</code>: {fmt(t.value_scaled)} × {fmt(t.weight)} = {fmt(t.product)}
                  </span>
                ))}
                <span className="solve-term">+ other terms: {fmt(others)}</span>
                <span className="solve-term">+ bias: {fmt(worked.bias)}</span>
              </span>
              <span className="solve-worked-line">
                <code>z = {fmt(sigTerm, 3)}</code>
              </span>
              <span className="solve-worked-line">
                <code>a = {sigFormula}</code>
              </span>
            </div>
          </div>
        </section>

        <section className="solve-card solve-card--wide">
          <header>
            <strong>Step 4 · Output neuron → probability → label</strong>
            <span>Pre-sigmoid logit z, then sigmoid, then threshold compare</span>
          </header>
          <div className="solve-decision">
            <code>z_out = {fmt(decision.logit, 3)}</code>
            <span className="solve-arrow">→</span>
            <code>P(Pass) = 1 / (1 + e^-z_out) = {pct(p, 2)}</code>
            <span className="solve-arrow">→</span>
            <code>
              {pct(p, 1)} {p >= decision.decision_threshold ? '≥' : '&lt;'} threshold {fmt(decision.decision_threshold, 2)}
            </code>
            <span className="solve-arrow">→</span>
            <span className={labelClass}>{decision.label}</span>
          </div>
          <p className="solve-note">
            Margin above/below threshold:{' '}
            <strong>{decision.margin >= 0 ? '+' : ''}{fmt(decision.margin * 100, 1)}%</strong> · fail_probability ={' '}
            {pct(decision.fail_probability, 2)}.
          </p>
        </section>
      </div>
    </div>
  );
}
