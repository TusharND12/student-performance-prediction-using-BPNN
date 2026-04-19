"""
FastAPI service for student pass/fail inference (BPNN + scaler artifact).
"""
from __future__ import annotations

import time
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field

from backend.inference import (
    artifacts_exist,
    map_frontend_payload,
    predict_from_student_row,
    predict_from_student_rows,
)
from backend.meta import (
    artifact_version,
    calibration_bins,
    completeness_from_payload,
    goal_seek_pass_probability,
)
from backend.labs import (
    adversary_narrative,
    cohort_compare_mapped_student,
    cohort_training_summary,
    compute_counterfactual,
    compute_sensitivity,
    export_manifest,
    shadow_model_compare,
    study_budget_physics,
)
from backend.report_pdf import ReportPdfRequest, build_session_report_pdf
from backend.csv_import import analyze_csv_text
from backend.network_viz import network_summary, predict_trace

app = FastAPI(title="Student Performance BPNN API", version="1.0.0")


def _sparse_payload(body: PredictRequest) -> dict[str, Any]:
    """Only keys the client sent; omitted fields map to training medians in inference (v2)."""
    return body.model_dump(exclude_unset=True, exclude_none=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictRequest(BaseModel):
    """All fields optional; missing numeric fields default to 0 (parent_involvement defaults in mapper)."""

    model_config = ConfigDict(extra="ignore")

    attendance_rate: float = 0.0
    assignment_avg: float = 0.0
    quiz_avg: float = 0.0
    project_score: float = 0.0
    previous_gpa: float = 0.0
    student_marks_percent: float = 0.0
    study_hours_daily: float = 0.0
    revision_hours: float = 0.0
    sleep_hours: float = 0.0
    mental_stress: float = 0.0
    sleep_quality: float = 0.0
    standardized_exam_score: float = 0.0
    learning_efficiency: float = 0.0
    stress_index: float = 0.0
    physical_activity: float = 0.0
    lms_login_frequency: float = 0.0
    coding_practice_hours: float = 0.0
    digital_literacy: float = 0.0
    math_score: float = 0.0
    science_score: float = 0.0
    english_score: float = 0.0
    history_score: float = 0.0
    computer_score: float = 0.0
    parent_involvement: float | None = Field(
        default=None,
        description="Optional; defaults to 6.0 if omitted (matches training artifact).",
    )
    # Extended tabular features (student_academic_performance_1M.csv); omitted → imputed at inference.
    age: float = 0.0
    gender: float = 0.0
    urban_flag: float = 0.0
    family_size: float = 0.0
    parent_education: float = 0.0
    family_income: float = 0.0
    screen_time: float = 0.0
    junk_food_freq: float = 0.0
    bmi: float = 0.0
    illness_days: float = 0.0
    internet_access: float = 0.0
    private_tuition: float = 0.0
    tuition_hours: float = 0.0
    study_room: float = 0.0
    scholarship_flag: float = 0.0
    part_time_job_hours: float = 0.0
    financial_stress: float = 0.0
    online_course_hours: float = 0.0
    ai_tool_usage: float = 0.0
    video_watch_hours: float = 0.0
    forum_participation: float = 0.0
    device_availability: float = 0.0
    final_gpa: float = 0.0
    improvement_next_term: float = 0.0
    dropout_risk_score: float = 0.0
    honors_flag: float = 0.0
    at_risk_flag: float = 0.0
    top_performer_flag: float = 0.0


class PredictResponse(BaseModel):
    prediction: str
    probability: float
    pass_probability: float
    fail_probability: float
    decision_threshold: float | None = None


class CounterfactualRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    payload: PredictRequest
    field: str
    value: float


class StudyBudgetRequest(BaseModel):
    sleep_hours: float = 0.0
    study_hours_daily: float = 0.0
    revision_hours: float = 0.0


class GoalSeekRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    payload: PredictRequest
    field: str = "study_hours_daily"
    target_pass_probability: float = 0.75
    range_lo: float = 0.0
    range_hi: float = 12.0
    steps: int = 24


class BatchPredictRequest(BaseModel):
    rows: list[PredictRequest] = Field(default_factory=list)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "artifacts_present": artifacts_exist()}


@app.get("/meta/version")
def meta_version() -> dict[str, Any]:
    """Artifact fingerprints for versioned exports."""
    return {**artifact_version(), "artifacts_present": artifacts_exist()}


@app.post("/meta/completeness")
def meta_completeness(body: PredictRequest) -> dict[str, Any]:
    """Weighted input completeness score (0–100)."""
    return completeness_from_payload(_sparse_payload(body))


@app.get("/meta/calibration")
def meta_calibration() -> dict[str, Any]:
    """Empirical pass rate by model probability decile (training CSV sample)."""
    if not artifacts_exist():
        raise HTTPException(status_code=503, detail="Model artifacts missing.")
    try:
        return calibration_bins()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/meta/network")
def meta_network() -> dict[str, Any]:
    """Subsampled BPNN weights + biases for the Analytics network diagram (blue/red = sign)."""
    if not artifacts_exist():
        raise HTTPException(status_code=503, detail="Model artifacts missing.")
    try:
        return network_summary()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/predict/trace")
def predict_trace_route(body: PredictRequest) -> dict[str, Any]:
    """Forward-pass trace: per-Dense-layer activations for the given payload (subsampled)."""
    if not artifacts_exist():
        raise HTTPException(status_code=503, detail="Model artifacts missing.")
    try:
        return predict_trace(_sparse_payload(body))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/meta/ensemble")
def meta_ensemble() -> dict[str, Any]:
    """Placeholder for a second model (logistic/XGB) — train separately to enable."""
    return {
        "available": False,
        "message": "Train a sklearn logistic or XGBoost baseline on the same engineered row; deploy joblib next to model.h5 to enable disagreement checks.",
    }


@app.post("/labs/goal-seek")
def labs_goal_seek(body: GoalSeekRequest) -> dict[str, Any]:
    """Grid search one lever toward a target pass probability."""
    if not artifacts_exist():
        raise HTTPException(status_code=503, detail="Model artifacts missing.")
    try:
        return goal_seek_pass_probability(
            _sparse_payload(body.payload),
            body.field,
            body.target_pass_probability,
            body.range_lo,
            body.range_hi,
            body.steps,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/import/csv")
@app.post("/api/import/csv")
async def import_csv(file: UploadFile = File(...)) -> dict[str, Any]:
    """
    Upload a UTF-8 CSV: server parses columns (aliases, BOM), runs BPNN batch inference,
    and returns per-row predictions plus optional agreement vs label columns (result/pass_fail).

    Registered at both ``/import/csv`` and ``/api/import/csv`` so proxies that forward the
    ``/api`` prefix to Uvicorn (without stripping it) still reach this handler.
    """
    if not artifacts_exist():
        raise HTTPException(status_code=503, detail="Model artifacts missing.")
    try:
        raw = await file.read()
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise HTTPException(status_code=400, detail="CSV must be UTF-8 encoded.") from e
    try:
        return analyze_csv_text(text, filename=file.filename or "upload.csv")
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/predict/batch")
def predict_batch(body: BatchPredictRequest) -> dict[str, Any]:
    """Batch inference (max 2000 rows) for CSV-style workflows."""
    if not artifacts_exist():
        raise HTTPException(status_code=503, detail="Model artifacts missing.")
    if not body.rows:
        raise HTTPException(status_code=400, detail="rows must be non-empty")
    if len(body.rows) > 2000:
        raise HTTPException(status_code=400, detail="max 2000 rows per batch")
    try:
        students = [map_frontend_payload(_sparse_payload(r)) for r in body.rows]
        out_rows = predict_from_student_rows(students)
        results = [
            {
                "prediction": out["prediction"],
                "pass_probability": out["pass_probability"],
                "fail_probability": out["fail_probability"],
            }
            for out in out_rows
        ]
        return {"count": len(results), "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/predict", response_model=PredictResponse)
def predict(body: PredictRequest) -> PredictResponse:
    if not artifacts_exist():
        raise HTTPException(
            status_code=503,
            detail="Model files missing. Run `student_performance_bpnn.ipynb` to export model.h5 and scaler.pkl.",
        )
    try:
        student = map_frontend_payload(_sparse_payload(body))
        out = predict_from_student_row(student)
        return PredictResponse(**out)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/labs/sensitivity")
def labs_sensitivity(body: PredictRequest) -> dict[str, Any]:
    """Local finite-difference sensitivity (model autopsy)."""
    if not artifacts_exist():
        raise HTTPException(status_code=503, detail="Model artifacts missing.")
    try:
        return compute_sensitivity(body)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/labs/counterfactual")
def labs_counterfactual(body: CounterfactualRequest) -> dict[str, Any]:
    """Single-field alternate value vs baseline (fairness / what-if mirror)."""
    if not artifacts_exist():
        raise HTTPException(status_code=503, detail="Model artifacts missing.")
    try:
        return compute_counterfactual(_sparse_payload(body.payload), body.field, body.value)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/labs/adversary")
def labs_adversary(body: PredictRequest) -> dict[str, Any]:
    """Devil's advocate bullets from sensitivity + rules."""
    if not artifacts_exist():
        raise HTTPException(status_code=503, detail="Model artifacts missing.")
    try:
        sens = compute_sensitivity(body)
        p0 = float(sens["baseline_pass_probability"])
        return adversary_narrative(sens, p0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/labs/cohort/summary")
def labs_cohort_summary() -> dict[str, Any]:
    """Training-cohort aggregates (peer-shadow, no row-level data)."""
    return cohort_training_summary()


@app.post("/labs/cohort/compare")
def labs_cohort_compare(body: PredictRequest) -> dict[str, Any]:
    """Percentile ranks vs student_data.csv for the mapped student row."""
    if not artifacts_exist():
        raise HTTPException(status_code=503, detail="Model artifacts missing.")
    try:
        student = map_frontend_payload(_sparse_payload(body))
        return cohort_compare_mapped_student(student)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get("/labs/manifest")
def labs_manifest() -> dict[str, Any]:
    """Paper-trail nonce + model fingerprint for exports."""
    return export_manifest()


@app.post("/labs/study-budget")
def labs_study_budget(body: StudyBudgetRequest) -> dict[str, Any]:
    """24h conservation physics scores."""
    return study_budget_physics(body.sleep_hours, body.study_hours_daily, body.revision_hours)


@app.post("/labs/shadow-model")
def labs_shadow_model(body: PredictRequest) -> dict[str, Any]:
    """BPNN vs training-z logistic shadow (Innovation Labs second opinion)."""
    if not artifacts_exist():
        raise HTTPException(status_code=503, detail="Model artifacts missing.")
    try:
        return shadow_model_compare(_sparse_payload(body))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/report/pdf")
def report_pdf(body: ReportPdfRequest) -> Response:
    """Multipage session PDF (ReportLab) — same payload shape as the frontend Reports export."""
    try:
        pdf_bytes = build_session_report_pdf(body)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF build failed: {e}") from e
    fname = f"edupredict-session-report-{int(time.time() * 1000)}.pdf"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )
