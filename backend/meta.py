"""
Artifact versioning, input completeness, calibration bins (training CSV), goal-seek helper.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import joblib

from backend.inference import (
    ROOT_DIR,
    artifacts_exist,
    map_frontend_payload,
    predict_from_student_row,
    predict_from_student_rows,
    predict_pass_probability_from_payload,
)

COMPLETENESS_WEIGHTS: dict[str, float] = {
    "attendance_rate": 1.2,
    "previous_gpa": 1.0,
    "assignment_avg": 1.1,
    "study_hours_daily": 1.3,
    "sleep_hours": 1.2,
    "mental_stress": 1.0,
    "math_score": 0.9,
    "science_score": 0.9,
    "english_score": 0.9,
    "history_score": 0.9,
    "computer_score": 0.9,
    "parent_involvement": 0.7,
}


def _file_fp(path: Path, nbytes: int = 65536) -> str:
    if not path.is_file():
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(nbytes))
    return h.hexdigest()


def artifact_version() -> dict[str, Any]:
    model_p = ROOT_DIR / "model.h5"
    scaler_p = ROOT_DIR / "scaler.pkl"
    out: dict[str, Any] = {
        "artifacts_present": artifacts_exist(),
        "model_h5_sha256_prefix": _file_fp(model_p)[:32] if model_p.is_file() else "",
        "scaler_pkl_sha256_prefix": _file_fp(scaler_p)[:32] if scaler_p.is_file() else "",
        "model_h5_bytes": model_p.stat().st_size if model_p.is_file() else 0,
        "scaler_pkl_bytes": scaler_p.stat().st_size if scaler_p.is_file() else 0,
        "export_schema": "edupredict-meta-v1",
    }
    if scaler_p.is_file():
        try:
            art = joblib.load(scaler_p)
            feats = art.get("feature_columns") or []
            out["pipeline_version"] = art.get("pipeline_version")
            out["decision_threshold"] = art.get("decision_threshold")
            out["threshold_metric"] = art.get("threshold_metric")
            out["feature_column_count"] = len(feats) if isinstance(feats, list) else 0
        except OSError:
            pass
    return out


def completeness_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """0–100 weighted completeness for trust / UX (not a model metric)."""
    total_w = sum(COMPLETENESS_WEIGHTS.values())
    earned = 0.0
    detail: list[dict[str, Any]] = []
    for key, w in COMPLETENESS_WEIGHTS.items():
        raw = payload.get(key)
        try:
            v = float(raw) if raw is not None and str(raw).strip() != "" else 0.0
        except (TypeError, ValueError):
            v = 0.0
        filled = v != 0.0
        if key == "parent_involvement" and raw is not None and str(raw).strip() != "":
            filled = True
        if filled:
            earned += w
        detail.append({"field": key, "weight": w, "filled": bool(filled)})
    score = round(100.0 * earned / total_w, 1) if total_w > 0 else 0.0
    return {"score": score, "max_score": 100.0, "weights_version": "v1", "fields": detail}


@lru_cache(maxsize=1)
def calibration_bins() -> dict[str, Any]:
    """
    Empirical pass rate by predicted probability decile on a reference CSV sample.
    Supports v2 tabular artifacts (student_academic_performance_1M.csv) or legacy student_data.csv.
    """
    if not artifacts_exist():
        return {"available": False, "reason": "artifacts missing"}
    try:
        art = joblib.load(ROOT_DIR / "scaler.pkl")
    except OSError:
        return {"available": False, "reason": "scaler.pkl unreadable"}

    feats = art.get("feature_columns") or []
    w = art.get("performance_index_weights") or {}
    path_1m = ROOT_DIR / "student_academic_performance_1M.csv"
    path_legacy = ROOT_DIR / "student_data.csv"

    ps: list[float] = []
    ys: list[float] = []

    if isinstance(feats, list) and len(feats) > 0 and not w and path_1m.is_file():
        df = pd.read_csv(path_1m, nrows=8000)
        tgt = "pass_fail"
        if tgt not in df.columns:
            return {"available": False, "reason": "pass_fail missing in 1M CSV"}
        miss = [c for c in feats if c not in df.columns]
        if miss:
            return {"available": False, "reason": f"CSV missing model columns (e.g. {miss[:3]})"}
        rows = df[list(feats)].astype(float).to_dict("records")
        ys = df[tgt].astype(float).to_numpy().tolist()
        try:
            outs = predict_from_student_rows(rows)
            ps = [float(o["pass_probability"]) for o in outs]
        except Exception as e:
            return {"available": False, "reason": f"batch scoring failed: {e}"}
    else:
        path = path_legacy
        if not path.is_file():
            return {"available": False, "reason": "student_data.csv not found"}
        df = pd.read_csv(path)
        if len(df) > 4000:
            df = df.sample(n=4000, random_state=42).reset_index(drop=True)
        required = [
            "study_hours",
            "attendance",
            "previous_marks",
            "assignments",
            "sleep_hours",
            "mental_stress",
            "parent_involvement",
            "result",
        ]
        if not all(c in df.columns for c in required):
            return {"available": False, "reason": "unexpected CSV schema"}
        for _, row in df.iterrows():
            student = {c: float(row[c]) for c in required[:-1]}
            try:
                out = predict_from_student_row(student)
                ps.append(float(out["pass_probability"]))
                ys.append(float(row["result"]))
            except Exception:
                continue

    if len(ps) < 100:
        return {"available": False, "reason": "too few valid rows"}
    p_arr = np.array(ps)
    y_arr = np.array(ys)
    deciles = []
    for d in range(10):
        lo = d / 10.0
        hi = (d + 1) / 10.0
        if d < 9:
            mask = (p_arr >= lo) & (p_arr < hi)
        else:
            mask = (p_arr >= lo) & (p_arr <= 1.0)
        n = int(mask.sum())
        if n == 0:
            deciles.append(
                {
                    "decile": d + 1,
                    "p_range": [round(lo, 2), round(hi, 2)],
                    "n": 0,
                    "empirical_pass_rate": None,
                }
            )
        else:
            deciles.append(
                {
                    "decile": d + 1,
                    "p_range": [round(lo, 2), round(hi, 2)],
                    "n": n,
                    "empirical_pass_rate": round(float(y_arr[mask].mean()), 4),
                }
            )
    return {
        "available": True,
        "n_scored": len(ps),
        "deciles": deciles,
        "note": "Empirical fraction of label=1 in each predicted-probability decile (in-sample; use held-out data for rigorous calibration).",
    }


GOAL_SEEK_FIELDS: frozenset[str] = frozenset(
    {
        "study_hours_daily",
        "sleep_hours",
        "mental_stress",
        "attendance_rate",
        "assignment_avg",
        "parent_involvement",
    }
)


def goal_seek_pass_probability(
    payload: dict[str, Any],
    field: str,
    target_p: float,
    lo: float,
    hi: float,
    steps: int = 24,
) -> dict[str, Any]:
    """Grid search for a single field to get closest pass probability to target (no monotonicity guarantee)."""
    if field not in GOAL_SEEK_FIELDS:
        raise ValueError(f"field must be one of {sorted(GOAL_SEEK_FIELDS)}")
    t = float(np.clip(target_p, 0.01, 0.99))
    lo_f = float(lo)
    hi_f = float(hi)
    n = max(8, min(64, int(steps)))
    best_v = lo_f
    best_err = 1.0
    best_p = 0.0
    curve: list[dict[str, float]] = []
    for i in range(n):
        alpha = i / max(1, n - 1)
        v = lo_f + (hi_f - lo_f) * alpha
        bumped = {**payload, field: v}
        try:
            p = predict_pass_probability_from_payload(bumped)
        except Exception:
            continue
        curve.append({"value": round(v, 4), "pass_probability": round(p, 6)})
        err = abs(p - t)
        if err < best_err:
            best_err = err
            best_v = v
            best_p = p
    return {
        "field": field,
        "target_pass_probability": t,
        "search_range": [lo_f, hi_f],
        "best_value": round(best_v, 6),
        "best_pass_probability": round(best_p, 6),
        "error": round(best_err, 6),
        "curve_sample": curve[:12],
    }
