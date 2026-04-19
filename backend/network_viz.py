"""
BPNN network visualization data for the Analytics dashboard.

Exposes a lightweight, subsampled view of the trained Dense layers so the frontend
can render an animated graph (blue = positive weight, red = negative, line width =
|weight|). Also supports a forward-pass "trace" returning per-layer activations for
a payload so the UI can animate which neurons lit up for a given student.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from tensorflow import keras

from backend.inference import _features_to_matrix, load_artifacts, map_frontend_payload

MAX_NODES_PER_LAYER = 10


def _dense_layers(model: keras.Model) -> list[keras.layers.Layer]:
    return [layer for layer in model.layers if isinstance(layer, keras.layers.Dense)]


def _top_input_indices(w: np.ndarray, k: int) -> list[int]:
    """Pick input features by total absolute outgoing weight (how loudly they speak into H1)."""
    score = np.abs(w).sum(axis=1)
    k = min(k, w.shape[0])
    return sorted(np.argsort(-score)[:k].tolist())


def _top_neuron_indices(w: np.ndarray, k: int) -> list[int]:
    """Pick a layer's neurons by L2 norm of incoming weights (most 'responsive')."""
    if w.shape[1] <= 1:
        return list(range(w.shape[1]))
    score = np.linalg.norm(w, axis=0)
    k = min(k, w.shape[1])
    return sorted(np.argsort(-score)[:k].tolist())


def network_summary() -> dict[str, Any]:
    """Subsampled weights/biases for each Dense layer, plus shown feature names."""
    model, art = load_artifacts()
    feats = list(art["features_for_model"])
    dense = _dense_layers(model)
    if len(dense) < 2:
        raise RuntimeError("Model has fewer than 2 Dense layers; cannot build a diagram.")

    w0, b0 = dense[0].get_weights()
    input_idx = _top_input_indices(w0, MAX_NODES_PER_LAYER)
    h1_idx = _top_neuron_indices(w0, MAX_NODES_PER_LAYER)

    layers_out: list[dict[str, Any]] = []
    w_sub = w0[np.ix_(input_idx, h1_idx)]
    b_sub = b0[h1_idx]
    layers_out.append(
        {
            "name": dense[0].name,
            "units_total": int(dense[0].units),
            "units_shown": len(h1_idx),
            "indices_shown": h1_idx,
            "weights": w_sub.tolist(),
            "biases": b_sub.tolist(),
        }
    )

    prev_idx = h1_idx
    for layer in dense[1:]:
        wi, bi = layer.get_weights()
        units = int(layer.units)
        if units == 1:
            cur_idx: list[int] = [0]
        else:
            cur_idx = _top_neuron_indices(wi, MAX_NODES_PER_LAYER)
        w_sub = wi[np.ix_(prev_idx, cur_idx)]
        b_sub = bi[cur_idx]
        layers_out.append(
            {
                "name": layer.name,
                "units_total": units,
                "units_shown": len(cur_idx),
                "indices_shown": cur_idx,
                "weights": w_sub.tolist(),
                "biases": b_sub.tolist(),
            }
        )
        prev_idx = cur_idx

    return {
        "feature_names_shown": [feats[i] for i in input_idx],
        "feature_indices_shown": input_idx,
        "total_features": len(feats),
        "layers": layers_out,
        "max_nodes_per_layer": MAX_NODES_PER_LAYER,
    }


def _numpy_forward(model: keras.Model, x: np.ndarray) -> list[np.ndarray]:
    """
    Numpy replay of the model: runs through Dense -> BatchNorm (inference) -> Dropout (identity).
    Returns the post-activation output of each Dense layer, in order.
    """
    h = x.astype(np.float64, copy=True)
    outs: list[np.ndarray] = []
    for layer in model.layers:
        if isinstance(layer, keras.layers.Dense):
            w, b = layer.get_weights()
            z = h @ w + b
            cfg = layer.get_config()
            act = cfg.get("activation")
            if act == "relu":
                h = np.maximum(0.0, z)
            elif act == "sigmoid":
                h = 1.0 / (1.0 + np.exp(-z))
            elif act in (None, "linear"):
                h = z
            else:
                raise RuntimeError(f"Unsupported Dense activation: {act!r}")
            outs.append(h.copy())
        elif isinstance(layer, keras.layers.BatchNormalization):
            gamma, beta, mean, var = layer.get_weights()
            h = gamma * (h - mean) / np.sqrt(var + layer.epsilon) + beta
        # Dropout / GaussianNoise / Input: identity at inference.
    return outs


def _numpy_forward_with_logit(
    model: keras.Model, x: np.ndarray
) -> tuple[list[np.ndarray], float]:
    """Same as `_numpy_forward` but also returns the pre-sigmoid output logit."""
    h = x.astype(np.float64, copy=True)
    outs: list[np.ndarray] = []
    logit: float | None = None
    for layer in model.layers:
        if isinstance(layer, keras.layers.Dense):
            w, b = layer.get_weights()
            z = h @ w + b
            cfg = layer.get_config()
            act = cfg.get("activation")
            if act == "relu":
                h = np.maximum(0.0, z)
            elif act == "sigmoid":
                logit = float(z.reshape(-1)[0])
                h = 1.0 / (1.0 + np.exp(-z))
            elif act in (None, "linear"):
                h = z
            else:
                raise RuntimeError(f"Unsupported Dense activation: {act!r}")
            outs.append(h.copy())
        elif isinstance(layer, keras.layers.BatchNormalization):
            gamma, beta, mean, var = layer.get_weights()
            h = gamma * (h - mean) / np.sqrt(var + layer.epsilon) + beta
    if logit is None:
        logit = float(outs[-1].reshape(-1)[0])
    return outs, logit


def predict_trace(payload: dict[str, Any]) -> dict[str, Any]:
    """Return per-Dense-layer activations (subsampled) for the given payload."""
    model, art = load_artifacts()
    dense = _dense_layers(model)
    student = map_frontend_payload(payload)
    x_scaled = _features_to_matrix(student, art)

    acts, logit = _numpy_forward_with_logit(model, x_scaled)
    summary = network_summary()
    feats_all: list[str] = list(art["features_for_model"])
    impute = dict(art.get("impute_values") or {})
    decision_threshold = float(art.get("decision_threshold", 0.5))

    input_vals_shown = [float(x_scaled[0, i]) for i in summary["feature_indices_shown"]]

    layers_activations: list[list[float]] = []
    for layer_info, a in zip(summary["layers"], acts):
        idx = layer_info["indices_shown"]
        vec = np.asarray(a).reshape(-1)
        layers_activations.append([float(vec[i]) for i in idx])

    p_pass = float(acts[-1].reshape(-1)[0])

    # ── Solving / explanation payload ─────────────────────────────────────
    x0 = x_scaled[0]
    w_all, b_all = dense[0].get_weights()
    # Per-input contribution into the full H1 layer: sum_j |w_ij| * |x_i|
    contrib_full = (np.abs(w_all).sum(axis=1) * np.abs(x0)).astype(np.float64)
    top_n = min(8, int(contrib_full.size))
    top_input_idx = np.argsort(-contrib_full)[:top_n].tolist()
    top_inputs = [
        {
            "name": feats_all[i],
            "value_scaled": float(x0[i]),
            "value_raw": float(student.get(feats_all[i], impute.get(feats_all[i], 0.0))),
            "contribution": float(contrib_full[i]),
            "was_imputed": feats_all[i] not in payload or payload.get(feats_all[i]) in (None, "", 0, 0.0),
        }
        for i in top_input_idx
    ]

    layer_stats: list[dict[str, Any]] = []
    for i, a in enumerate(acts[:-1]):
        vec = np.asarray(a).reshape(-1)
        active_mask = vec > 1e-6
        layer_stats.append(
            {
                "name": dense[i].name,
                "units_total": int(dense[i].units),
                "active_units": int(active_mask.sum()),
                "active_pct": float(active_mask.mean()),
                "mean_activation": float(vec.mean()),
                "max_activation": float(vec.max()) if vec.size else 0.0,
                "activation_fn": str(dense[i].get_config().get("activation") or "linear"),
            }
        )

    # Worked example for Hidden 1's strongest neuron
    h1_vec = np.asarray(acts[0]).reshape(-1)
    strongest_h1 = int(np.argmax(h1_vec))
    w_col = w_all[:, strongest_h1]
    b_col = float(b_all[strongest_h1])
    term_vals = (x0 * w_col).astype(np.float64)
    term_idx = np.argsort(-np.abs(term_vals))[: min(6, term_vals.size)].tolist()
    worked_example = {
        "layer_name": dense[0].name,
        "neuron_index": strongest_h1,
        "bias": b_col,
        "z": float(x0 @ w_col + b_col),
        "activation": float(h1_vec[strongest_h1]),
        "top_terms": [
            {
                "feature": feats_all[i],
                "value_scaled": float(x0[i]),
                "weight": float(w_col[i]),
                "product": float(term_vals[i]),
            }
            for i in term_idx
        ],
    }

    # Output decision steps
    output_weights, output_bias = dense[-1].get_weights()
    output_weights = output_weights.reshape(-1)
    output_bias = float(output_bias.reshape(-1)[0])
    decision = {
        "output_bias": output_bias,
        "logit": float(logit),
        "sigmoid_formula": "P(Pass) = 1 / (1 + exp(-z))",
        "pass_probability": p_pass,
        "fail_probability": 1.0 - p_pass,
        "decision_threshold": decision_threshold,
        "label": "Pass" if p_pass >= decision_threshold else "Fail",
        "margin": float(p_pass - decision_threshold),
    }

    return {
        "pass_probability": p_pass,
        "fail_probability": 1.0 - p_pass,
        "input_values_scaled": input_vals_shown,
        "layers_activations": layers_activations,
        "explanation": {
            "total_features": int(x0.size),
            "top_inputs": top_inputs,
            "layer_stats": layer_stats,
            "worked_example": worked_example,
            "decision": decision,
        },
    }
