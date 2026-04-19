"""
Advanced analysis: local sensitivity, counterfactuals, cohort benchmarks, adversary copy.
"""
from __future__ import annotations

import hashlib
import uuid
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from backend.inference import ROOT_DIR, map_frontend_payload, predict_pass_probability_from_payload

# Frontend payload keys we perturb for finite-difference sensitivity (aligned with PredictRequest).
SENSITIVITY_KEYS: tuple[str, ...] = (
    "attendance_rate",
    "assignment_avg",
    "quiz_avg",
    "project_score",
    "previous_gpa",
    "student_marks_percent",
    "study_hours_daily",
    "sleep_hours",
    "mental_stress",
    "parent_involvement",
    "math_score",
    "science_score",
    "english_score",
    "history_score",
    "computer_score",
    "standardized_exam_score",
)


def _epsilon_for_key(key: str, current: float) -> float:
    """Small step for finite differences; scale-aware."""
    c = float(current) if current is not None else 0.0
    floors: dict[str, float] = {
        "attendance_rate": 0.03,
        "study_hours_daily": 0.35,
        "sleep_hours": 0.35,
        "mental_stress": 0.4,
        "parent_involvement": 0.5,
        "previous_gpa": 0.15,
        "student_marks_percent": 2.0,
    }
    floor = floors.get(key, 2.0)
    if key == "attendance_rate":
        return min(0.08, max(floor * 0.01, 0.02))
    return max(floor, abs(c) * 0.04 + 0.01)


def _payload_as_dict(body: Any) -> dict[str, Any]:
    if hasattr(body, "model_dump"):
        return body.model_dump(exclude_unset=True, exclude_none=True)
    return dict(body)


def _effective_scalar_for_perturb(payload: dict[str, Any], key: str) -> float:
    """Match inference defaults (e.g. parent_involvement → 6) when perturbing."""
    raw = payload.get(key)
    if key == "parent_involvement":
        if raw is None or (isinstance(raw, str) and raw.strip() == ""):
            return 6.0
        try:
            return float(raw)
        except (TypeError, ValueError):
            return 6.0
    if raw is None or (isinstance(raw, str) and str(raw).strip() == ""):
        return 0.0
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def compute_sensitivity(body: Any) -> dict[str, Any]:
    """One-sided finite-difference sensitivity per input field."""
    payload = _payload_as_dict(body)
    p0 = predict_pass_probability_from_payload(payload)
    items: list[dict[str, Any]] = []
    for key in SENSITIVITY_KEYS:
        cur = _effective_scalar_for_perturb(payload, key)
        eps = _epsilon_for_key(key, cur)
        bumped = {**payload, key: cur + eps}
        try:
            p1 = predict_pass_probability_from_payload(bumped)
        except Exception:
            continue
        items.append(
            {
                "field": key,
                "delta_pass_probability": round(p1 - p0, 6),
                "epsilon": eps,
                "direction_hint": "raising this input" if (p1 - p0) >= 0 else "lowering this input",
            }
        )
    items.sort(key=lambda x: abs(x["delta_pass_probability"]), reverse=True)
    return {
        "baseline_pass_probability": round(p0, 6),
        "items": items[:12],
    }


def compute_counterfactual(body: Any, field: str, value: float) -> dict[str, Any]:
    """Baseline vs alternate single-field value (fairness / what-if mirror)."""
    payload = _payload_as_dict(body)
    if field not in payload and field not in SENSITIVITY_KEYS:
        raise ValueError(f"Unknown field: {field}")
    p0 = predict_pass_probability_from_payload(payload)
    alt = {**payload, field: float(value)}
    p1 = predict_pass_probability_from_payload(alt)
    return {
        "field": field,
        "baseline_pass_probability": round(p0, 6),
        "counterfactual_pass_probability": round(p1, 6),
        "delta": round(p1 - p0, 6),
        "alternate_value": float(value),
    }


def adversary_narrative(sensitivity: dict[str, Any], baseline_p: float) -> dict[str, Any]:
    """Rule-based 'devil's advocate' bullets + worst drivers from sensitivity."""
    bullets: list[str] = []
    if baseline_p >= 0.5:
        bullets.append(
            "The calibrated threshold still allows Fail if measurement error or distribution shift applies — treat the label as probabilistic."
        )
    else:
        bullets.append(
            "Below 0.5 pass probability, the BPNN leans Fail under training assumptions — the burden of proof for 'safe' is on stronger academics and wellbeing signals."
        )

    items = sensitivity.get("items") or []
    negative = [x for x in items if x.get("delta_pass_probability", 0) < -1e-6][:3]
    for x in negative:
        bullets.append(
            f"Raising {x['field']} slightly was associated with lower pass probability in the local sensitivity probe — check for coupling with sleep/stress or data entry scale."
        )
    if not negative and items:
        top = items[0]
        bullets.append(
            f"Largest local sensitivity is on {top['field']} (Δp ≈ {top['delta_pass_probability']:+.4f}) — small changes there move the needle fastest."
        )

    return {"bullets": bullets[:5], "stance": "challenge"}


@lru_cache(maxsize=1)
def _training_frame() -> pd.DataFrame:
    big = ROOT_DIR / "student_academic_performance_1M.csv"
    small = ROOT_DIR / "student_data.csv"
    if big.is_file():
        return pd.read_csv(big, nrows=50000)
    if small.is_file():
        return pd.read_csv(small)
    return pd.DataFrame()


def cohort_training_summary() -> dict[str, Any]:
    """Aggregate stats from the reference training CSV for peer-shadow benchmarking."""
    df = _training_frame()
    if df.empty:
        return {"available": False, "n_rows": 0, "columns": {}}
    if "study_hours_daily" in df.columns:
        cols = [
            "study_hours_daily",
            "attendance_rate",
            "previous_gpa",
            "assignment_avg",
            "sleep_hours",
            "mental_stress",
            "parent_involvement",
        ]
    else:
        cols = [
            "study_hours",
            "attendance",
            "previous_marks",
            "assignments",
            "sleep_hours",
            "mental_stress",
            "parent_involvement",
        ]
    out: dict[str, Any] = {"available": True, "n_rows": int(len(df)), "columns": {}}
    for c in cols:
        if c not in df.columns:
            continue
        s = df[c].dropna()
        if s.empty:
            continue
        out["columns"][c] = {
            "mean": float(s.mean()),
            "std": float(s.std()),
            "p10": float(np.percentile(s, 10)),
            "p50": float(np.percentile(s, 50)),
            "p90": float(np.percentile(s, 90)),
        }
    if "pass_fail" in df.columns:
        out["pass_rate_label_1"] = float(df["pass_fail"].mean())
    elif "result" in df.columns:
        out["pass_rate_label_1"] = float(df["result"].mean())
    else:
        out["pass_rate_label_1"] = None
    return out


def cohort_compare_mapped_student(student: dict[str, float]) -> dict[str, Any]:
    """Percentile ranks vs training CSV (privacy-safe: only aggregates exposed client-side)."""
    df = _training_frame()
    if df.empty:
        return {"available": False}
    percentiles: dict[str, float] = {}
    for col in student:
        if col not in df.columns:
            continue
        v = float(student[col])
        s = df[col].dropna()
        if len(s) == 0:
            continue
        pct = float((s < v).mean() * 100.0)
        percentiles[col] = round(pct, 1)
    return {"available": True, "percentiles": percentiles, "n_reference": int(len(df))}


def export_manifest() -> dict[str, Any]:
    """Paper-trail metadata for PDF / audits."""
    nonce = uuid.uuid4().hex
    fp = ""
    try:
        p = ROOT_DIR / "model.h5"
        if p.is_file():
            h = hashlib.sha256()
            with open(p, "rb") as f:
                h.update(f.read(65536))
            fp = h.hexdigest()[:24]
    except OSError:
        fp = "unavailable"
    return {
        "export_nonce": nonce,
        "model_sha256_prefix": fp,
        "api": "EduPredict Labs manifest v1",
    }


def shadow_model_compare(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Lightweight second opinion: logistic on z-scores vs training marginals (not a trained classifier).
    Exposes BPNN vs shadow disagreement for Innovation Labs demos.
    """
    df = _training_frame()
    if df.empty or len(df) < 50:
        return {"available": False, "reason": "training CSV missing or too small"}
    student = map_frontend_payload(payload)
    if "study_hours_daily" in df.columns:
        keys = [
            "study_hours_daily",
            "attendance_rate",
            "previous_gpa",
            "assignment_avg",
            "sleep_hours",
            "mental_stress",
            "parent_involvement",
        ]
        w = np.array([0.12, 0.22, 0.18, 0.14, 0.08, -0.18, 0.08], dtype=float)
        ref = "student_academic_performance_1M.csv (sample)"
    else:
        keys = [
            "study_hours",
            "attendance",
            "previous_marks",
            "assignments",
            "sleep_hours",
            "mental_stress",
            "parent_involvement",
        ]
        w = np.array([0.12, 0.22, 0.18, 0.14, 0.08, -0.18, 0.08], dtype=float)
        ref = "student_data.csv"
    for k in keys:
        if k not in df.columns:
            return {"available": False, "reason": f"missing column {k}"}
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for k in keys:
        s = df[k].dropna()
        if len(s) < 2:
            return {"available": False, "reason": f"no variance in {k}"}
        means[k] = float(s.mean())
        stds[k] = float(s.std()) or 1.0
    z = np.array([(student.get(k, 0.0) - means[k]) / stds[k] for k in keys], dtype=float)
    logit = float(np.dot(w, z))
    p_shadow = float(1.0 / (1.0 + np.exp(-logit)))
    p_shadow = max(0.01, min(0.99, p_shadow))
    p_bpnn = predict_pass_probability_from_payload(payload)
    return {
        "available": True,
        "bpnn_pass_probability": round(p_bpnn, 6),
        "shadow_pass_probability": round(p_shadow, 6),
        "disagreement": round(abs(p_bpnn - p_shadow), 6),
        "note": f"Shadow is a hand-weighted logistic on z-scores vs {ref} — not a second trained model.",
    }


def study_budget_physics(sleep_h: float, study_h: float, revision_h: float) -> dict[str, Any]:
    """Conservation-style scores for the 24h day (hand-crafted, interpretable)."""
    used = max(0.0, sleep_h) + max(0.0, study_h) + max(0.0, revision_h)
    remaining = max(0.0, 24.0 - used)
    # "Recovery adequacy" vs sleep (target band 7–9)
    sleep_score = 100.0 * (1.0 - min(1.0, abs(max(0.0, sleep_h) - 8.0) / 4.0))
    # Fatigue proxy: high study+revision with low sleep
    fatigue_risk = min(
        100.0,
        max(0.0, (study_h + revision_h) * 8.0 + max(0.0, 7.5 - sleep_h) * 10.0),
    )
    consistency = max(0.0, 100.0 - abs(used - 24.0) * 4.0) if used <= 24.0 else 0.0
    return {
        "used_hours": round(used, 2),
        "remaining_hours": round(remaining, 2),
        "recovery_score": round(sleep_score, 1),
        "fatigue_risk_score": round(fatigue_risk, 1),
        "day_balance_score": round(consistency, 1),
    }
