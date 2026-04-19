"""
Inference pipeline aligned with training artifacts:
  v1: engineered performance_index + legacy 7 raw fields
  v2: full tabular features from student_academic_performance_1M.csv (+ MinMaxScaler, medians for missing API fields)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "model.h5"
SCALER_PATH = ROOT_DIR / "scaler.pkl"

_model: keras.Model | None = None
_artifact: dict[str, Any] | None = None


def artifacts_exist() -> bool:
    return MODEL_PATH.is_file() and SCALER_PATH.is_file()


def _ensure_artifact_dict() -> dict[str, Any]:
    """Load scaler metadata from disk (cheap); avoids importing the Keras model for mapping only."""
    global _artifact
    if _artifact is None:
        if not SCALER_PATH.is_file():
            raise FileNotFoundError(
                f"Missing artifact bundle {SCALER_PATH.name} under {ROOT_DIR}."
            )
        _artifact = joblib.load(SCALER_PATH)
    return _artifact


def load_artifacts() -> tuple[keras.Model, dict[str, Any]]:
    global _model
    art = _ensure_artifact_dict()
    if _model is None:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Missing model file {MODEL_PATH.name} under {ROOT_DIR}."
            )
        _model = keras.models.load_model(MODEL_PATH)
    return _model, art


def _uses_legacy_engineering(art: dict[str, Any]) -> bool:
    w = art.get("performance_index_weights") or {}
    return isinstance(w, dict) and len(w) > 0


def _decision_threshold(art: dict[str, Any], override: float | None) -> float:
    if override is not None:
        return float(override)
    return float(art.get("decision_threshold", 0.5))


def map_frontend_payload(payload: dict[str, Any]) -> dict[str, float]:
    """
    Map API / form payload to the training feature vector.
    v2: one key per CSV feature; unknown keys ignored; missing keys use training medians from artifact.
    v1: legacy 7-field mapping (used only when artifacts use performance_index).
    """
    if not artifacts_exist():
        raise FileNotFoundError("Model artifacts missing; cannot map payload.")
    art = _ensure_artifact_dict()

    if _uses_legacy_engineering(art):
        return _map_frontend_payload_legacy(payload)

    feats: list[str] = list(art["feature_columns"])
    impute: dict[str, float] = dict(art.get("impute_values") or {})
    raw = dict(payload)

    out: dict[str, float] = {}
    for c in feats:
        v = raw.get(c)
        if v is None or (isinstance(v, str) and str(v).strip() == ""):
            out[c] = float(impute.get(c, 0.0))
            continue
        try:
            out[c] = float(v)
        except (TypeError, ValueError):
            out[c] = float(impute.get(c, 0.0))
    return out


def _map_frontend_payload_legacy(payload: dict[str, Any]) -> dict[str, float]:
    g = lambda k, default=0.0: float(payload.get(k, default) or 0)

    subj = ["math_score", "science_score", "english_score", "history_score", "computer_score"]
    subj_mean = float(np.mean([g(s) for s in subj]))
    percent = g("student_marks_percent")
    if percent > 0:
        previous_marks = float(np.clip(percent, 0.0, 100.0))
    else:
        previous_marks = subj_mean

    parent = payload.get("parent_involvement")
    if parent is None or parent == "":
        parent = 6.0
    else:
        parent = float(parent)

    return {
        "study_hours": g("study_hours_daily"),
        "attendance": g("attendance_rate"),
        "previous_marks": previous_marks,
        "assignments": g("assignment_avg"),
        "sleep_hours": g("sleep_hours"),
        "mental_stress": g("mental_stress"),
        "parent_involvement": parent,
    }


def predict_pass_probability_from_payload(payload: dict[str, Any]) -> float:
    """Scalar pass probability for analysis endpoints (sensitivity, counterfactuals)."""
    student = map_frontend_payload(payload)
    out = predict_from_student_row(student)
    return float(out["pass_probability"])


def _features_to_matrix(student: dict[str, float], art: dict[str, Any]) -> np.ndarray:
    feats = list(art["feature_columns"])
    feats_model = list(art["features_for_model"])
    row = {k: float(student[k]) for k in feats}
    X0 = pd.DataFrame([row])

    if _uses_legacy_engineering(art):
        X0["engagement"] = X0["attendance"] * X0["study_hours"]
        X0["stress_sleep"] = X0["mental_stress"] * X0["sleep_hours"]

        mins_ = art["performance_index_mins"]
        maxs_ = art["performance_index_maxs"]
        wts = art["performance_index_weights"]
        eps = 1e-8
        norm: dict[str, pd.Series] = {}
        for c in wts:
            norm[c] = (X0[c] - float(mins_[c])) / (float(maxs_[c] - mins_[c]) + eps)
        X0["performance_index"] = float(sum(wts[c] * float(norm[c].iloc[0]) for c in wts))

    X0 = X0.reindex(columns=list(feats_model), fill_value=0.0)
    scaler = art["scaler"]
    return scaler.transform(X0.to_numpy(dtype=np.float64, copy=False))


def predict_from_student_row(student: dict[str, float], threshold: float | None = None) -> dict[str, Any]:
    model, art = load_artifacts()
    thr = _decision_threshold(art, threshold)
    Xs = _features_to_matrix(student, art)
    p_pass = float(model.predict(Xs, verbose=0).reshape(-1)[0])
    label = "Pass" if p_pass >= thr else "Fail"

    return {
        "prediction": label,
        "probability": p_pass,
        "pass_probability": p_pass,
        "fail_probability": 1.0 - p_pass,
        "decision_threshold": float(thr),
    }


def predict_from_student_rows(
    students: list[dict[str, float]], threshold: float | None = None
) -> list[dict[str, Any]]:
    """Vectorized batch inference for multiple students."""
    if not students:
        return []

    model, art = load_artifacts()
    thr = _decision_threshold(art, threshold)
    feats = list(art["feature_columns"])
    feats_model = list(art["features_for_model"])
    scaler = art["scaler"]

    rows = [{k: float(s[k]) for k in feats} for s in students]
    X0 = pd.DataFrame(rows)

    if _uses_legacy_engineering(art):
        X0["engagement"] = X0["attendance"] * X0["study_hours"]
        X0["stress_sleep"] = X0["mental_stress"] * X0["sleep_hours"]

        mins_ = art["performance_index_mins"]
        maxs_ = art["performance_index_maxs"]
        wts = art["performance_index_weights"]
        eps = 1e-8
        X0["performance_index"] = 0.0
        for c, wt in wts.items():
            denom = float(maxs_[c] - mins_[c]) + eps
            comp = (X0[c] - float(mins_[c])) / denom
            X0["performance_index"] += float(wt) * comp

    X0 = X0.reindex(columns=list(feats_model), fill_value=0.0)
    Xs = scaler.transform(X0.to_numpy(dtype=np.float64, copy=False))
    probs = model.predict(Xs, verbose=0).reshape(-1)

    out: list[dict[str, Any]] = []
    thr_f = float(thr)
    for p in probs:
        p_pass = float(p)
        out.append(
            {
                "prediction": "Pass" if p_pass >= thr_f else "Fail",
                "probability": p_pass,
                "pass_probability": p_pass,
                "fail_probability": 1.0 - p_pass,
                "decision_threshold": thr_f,
            }
        )
    return out
