"""
Server-side CSV import: parse, alias columns to training names, batch BPNN inference, optional label analysis.
Mirrors frontend extended CSV mapping so uploads are scored consistently with map_frontend_payload + artifacts.
"""
from __future__ import annotations

import csv
import io
import re
from typing import Any

from backend.inference import artifacts_exist, map_frontend_payload, predict_from_student_rows

# Max rows per upload (aligned with /predict/batch).
CSV_IMPORT_MAX_ROWS = 2000

# Keys and defaults matching frontend `INITIAL_FORM` + `csvRowToFormData` extras.
INITIAL_FORM_DEFAULTS: dict[str, str] = {
    "attendance_rate": "",
    "assignment_avg": "",
    "quiz_avg": "",
    "project_score": "",
    "previous_gpa": "",
    "study_hours_daily": "",
    "revision_hours": "",
    "sleep_hours": "",
    "mental_stress": "",
    "sleep_quality": "",
    "standardized_exam_score": "",
    "learning_efficiency": "",
    "stress_index": "",
    "physical_activity": "",
    "lms_login_frequency": "",
    "coding_practice_hours": "",
    "digital_literacy": "",
    "parent_involvement": "6",
    "math_score": "",
    "science_score": "",
    "english_score": "",
    "history_score": "",
    "computer_score": "",
    "student_marks_percent": "",
}

# Alternate normalized headers -> canonical API / model field names (keep in sync with frontend App.jsx).
EXTENDED_CSV_FIELD_ALIASES: dict[str, str] = {
    "assignment_average": "assignment_avg",
    "assignment_avg_pct": "assignment_avg",
    "quiz_average": "quiz_avg",
    "quiz_avg_pct": "quiz_avg",
    "standardized_exam": "standardized_exam_score",
    "standardized_test_score": "standardized_exam_score",
    "exam_score": "standardized_exam_score",
    "study_hours_day": "study_hours_daily",
    "study_hours_per_day": "study_hours_daily",
    "daily_study_hours": "study_hours_daily",
    "revision_hours_day": "revision_hours",
    "revision_hours_per_day": "revision_hours",
    "sleep_hours_day": "sleep_hours",
    "sleep_hours_per_day": "sleep_hours",
    "physical_activity_hrs_week": "physical_activity",
    "physical_activity_hours_week": "physical_activity",
    "weekly_physical_activity": "physical_activity",
    "lms_logins_week": "lms_login_frequency",
    "lms_logins_per_week": "lms_login_frequency",
    "coding_hours_week": "coding_practice_hours",
    "coding_hours_per_week": "coding_practice_hours",
    "mathematics": "math_score",
    "math": "math_score",
    "maths_score": "math_score",
    "science": "science_score",
    "english": "english_score",
    "history": "history_score",
    "computer_science": "computer_score",
    "cs_score": "computer_score",
    "avg_subject_marks": "student_marks_percent",
    "average_subject_marks": "student_marks_percent",
    "subject_marks_avg": "student_marks_percent",
    "mean_subject_score": "student_marks_percent",
}

CSV_COLUMNS_NEVER_IN_MODEL_PAYLOAD: frozenset[str] = frozenset(
    {
        "pass_fail",
        "result",
        "label",
        "target",
        "y",
        "final_gpa",
        "improvement_next_term",
        "dropout_risk_score",
        "honors_flag",
        "at_risk_flag",
        "top_performer_flag",
    }
)

# All numeric keys accepted by PredictRequest (sparse subset sent to map_frontend_payload).
PREDICT_NUMERIC_KEYS: tuple[str, ...] = (
    "attendance_rate",
    "assignment_avg",
    "quiz_avg",
    "project_score",
    "previous_gpa",
    "student_marks_percent",
    "study_hours_daily",
    "revision_hours",
    "sleep_hours",
    "mental_stress",
    "sleep_quality",
    "standardized_exam_score",
    "learning_efficiency",
    "stress_index",
    "physical_activity",
    "lms_login_frequency",
    "coding_practice_hours",
    "digital_literacy",
    "math_score",
    "science_score",
    "english_score",
    "history_score",
    "computer_score",
    "parent_involvement",
    "age",
    "gender",
    "urban_flag",
    "family_size",
    "parent_education",
    "family_income",
    "screen_time",
    "junk_food_freq",
    "bmi",
    "illness_days",
    "internet_access",
    "private_tuition",
    "tuition_hours",
    "study_room",
    "scholarship_flag",
    "part_time_job_hours",
    "financial_stress",
    "online_course_hours",
    "ai_tool_usage",
    "video_watch_hours",
    "forum_participation",
    "device_availability",
)

NAME_KEYS = (
    "student_name",
    "full_name",
    "name",
    "learner_name",
    "student",
    "display_name",
    "pupil_name",
)
ID_KEYS = (
    "student_id",
    "id",
    "roll_no",
    "roll_number",
    "enrollment_id",
    "uid",
    "learners_id",
    "srn",
)


def normalize_csv_header(header: str) -> str:
    s = str(header or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return re.sub(r"^_+|_+$", "", s)


def dedupe_headers(headers: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    out: list[str] = []
    for h in headers:
        base = h if h else "column"
        counts[base] = counts.get(base, 0) + 1
        n = counts[base]
        out.append(base if n == 1 else f"{base}_{n}")
    return out


def strip_bom(text: str) -> str:
    return text.lstrip("\ufeff") if text else text


def parse_csv_rows(text: str) -> tuple[list[str], list[dict[str, str]]]:
    src = strip_bom(text)
    reader = csv.reader(io.StringIO(src))
    rows_list = list(reader)
    if not rows_list:
        return [], []
    headers_raw = [normalize_csv_header(h) for h in rows_list[0]]
    headers = dedupe_headers(headers_raw)
    data: list[dict[str, str]] = []
    for cells in rows_list[1:]:
        if not any((c or "").strip() for c in cells):
            continue
        row: dict[str, str] = {}
        for i, h in enumerate(headers):
            row[h] = (cells[i] if i < len(cells) else "").strip()
        if any(v for v in row.values()):
            data.append(row)
    return headers, data


def expand_row_to_canonical(row: dict[str, str]) -> dict[str, str]:
    out = dict(row)
    for alias, canonical in EXTENDED_CSV_FIELD_ALIASES.items():
        raw_a = out.get(alias, "")
        if raw_a is None or str(raw_a).strip() == "":
            continue
        raw_c = out.get(canonical, "")
        if raw_c is not None and str(raw_c).strip() != "":
            continue
        out[canonical] = raw_a
    return out


def _pick(row: dict[str, str], *keys: str) -> str | None:
    for k in keys:
        v = row.get(k)
        if v is not None and str(v).strip() != "":
            return str(v).strip()
    return None


def expanded_row_to_form_data(expanded: dict[str, str]) -> dict[str, str]:
    """Mirror frontend `csvRowToFormData` for Student Records / composite labels."""

    def pick(*keys: str) -> str:
        for key in keys:
            v = expanded.get(key)
            if v is None:
                continue
            s = str(v).strip()
            if s != "":
                return s
        return ""

    out = dict(INITIAL_FORM_DEFAULTS)
    for key in list(INITIAL_FORM_DEFAULTS.keys()):
        if key == "student_marks_percent":
            continue
        raw = pick(key)
        if raw:
            out[key] = raw
    out["study_hours_daily"] = pick("study_hours_daily", "study_hours") or out.get("study_hours_daily", "")
    out["attendance_rate"] = pick("attendance_rate", "attendance") or out.get("attendance_rate", "")
    out["assignment_avg"] = pick("assignment_avg", "assignments") or out.get("assignment_avg", "")
    out["student_marks_percent"] = pick("student_marks_percent", "previous_marks") or ""
    pi = pick("parent_involvement")
    out["parent_involvement"] = pi if pi else (out.get("parent_involvement") or "6")
    return out


def row_to_sparse_api_payload(expanded: dict[str, str]) -> dict[str, Any]:
    """Sparse body for map_frontend_payload (omit missing; no label columns)."""
    sparse: dict[str, Any] = {}
    for key in PREDICT_NUMERIC_KEYS:
        if key in CSV_COLUMNS_NEVER_IN_MODEL_PAYLOAD:
            continue
        raw = expanded.get(key, "")
        if raw is None or str(raw).strip() == "":
            continue
        try:
            sparse[key] = float(raw)
        except ValueError:
            continue
    sh = _pick(expanded, "study_hours_daily", "study_hours")
    if sh is not None:
        try:
            sparse["study_hours_daily"] = float(sh)
        except ValueError:
            pass
    att = _pick(expanded, "attendance_rate", "attendance")
    if att is not None:
        try:
            sparse["attendance_rate"] = float(att)
        except ValueError:
            pass
    asn = _pick(expanded, "assignment_avg", "assignments")
    if asn is not None:
        try:
            sparse["assignment_avg"] = float(asn)
        except ValueError:
            pass
    pm = _pick(expanded, "student_marks_percent", "previous_marks")
    if pm is not None:
        try:
            sparse["student_marks_percent"] = float(pm)
        except ValueError:
            pass
    par = _pick(expanded, "parent_involvement")
    if par is not None:
        try:
            sparse["parent_involvement"] = float(par)
        except ValueError:
            pass
    return sparse


def infer_result_polarity(rows: list[dict[str, str]]) -> tuple[str, str]:
    """Fixed to training convention; previous auto-flip heuristic was removed."""
    _ = rows
    return "one_is_pass", "Using training convention: 1 = Pass, 0 = Fail."


def parse_file_truth(row: dict[str, str], polarity: str) -> dict[str, Any]:
    """Parse an optional CSV ground-truth cell using training convention 1 = Pass, 0 = Fail."""
    _ = polarity
    raw_val = _pick(row, "result", "pass_fail", "target")
    if raw_val is None:
        return {"raw_display": "", "expect_pass": None, "label": ""}
    low = raw_val.lower()
    if low in ("pass", "p", "yes", "true"):
        return {"raw_display": raw_val, "expect_pass": True, "label": "Pass"}
    if low in ("fail", "f", "no", "false"):
        return {"raw_display": raw_val, "expect_pass": False, "label": "Fail"}
    try:
        n = int(low, 10)
    except ValueError:
        return {"raw_display": raw_val, "expect_pass": None, "label": ""}
    if n not in (0, 1):
        return {"raw_display": raw_val, "expect_pass": None, "label": ""}
    ok = n == 1
    return {"raw_display": str(n), "expect_pass": ok, "label": "Pass" if ok else "Fail"}


def extract_display(row: dict[str, str]) -> dict[str, Any]:
    """Keys aligned with frontend `extractCsvStudentDisplay` / csvMeta."""
    def pick_first(keys: tuple[str, ...]) -> str:
        for k in keys:
            v = row.get(k, "")
            if v is not None and str(v).strip() != "":
                return str(v).strip()
        fn = str(row.get("first_name", row.get("firstname", "")) or "").strip()
        ln = str(row.get("last_name", row.get("lastname", "")) or "").strip()
        if fn or ln:
            return f"{fn} {ln}".strip()
        return ""

    name = pick_first(NAME_KEYS)
    sid = pick_first(ID_KEYS)
    reserved = set(PREDICT_NUMERIC_KEYS) | set(CSV_COLUMNS_NEVER_IN_MODEL_PAYLOAD)
    reserved |= set(EXTENDED_CSV_FIELD_ALIASES.keys())
    reserved |= set(NAME_KEYS) | set(ID_KEYS)
    reserved |= {
        "first_name",
        "firstname",
        "last_name",
        "lastname",
        "study_hours",
        "attendance",
        "assignments",
        "previous_marks",
        "result",
    }
    extras: list[dict[str, str]] = []
    for k, v in row.items():
        if k in reserved:
            continue
        if v is None or str(v).strip() == "":
            continue
        if len(extras) >= 8:
            break
        label = k.replace("_", " ").title()
        val = str(v).strip()
        if len(val) > 120:
            val = val[:117] + "..."
        extras.append({"key": k, "label": label, "value": val})
    return {"displayName": name, "displayId": sid, "extras": extras}


def analyze_csv_text(text: str, filename: str = "upload.csv") -> dict[str, Any]:
    if not artifacts_exist():
        raise FileNotFoundError("Model artifacts missing.")

    headers, rows = parse_csv_rows(text)
    if not rows:
        raise ValueError("No data rows in CSV (empty file or header only).")
    if len(rows) > CSV_IMPORT_MAX_ROWS:
        raise ValueError(f"Too many rows ({len(rows)}); maximum is {CSV_IMPORT_MAX_ROWS}.")

    polarity_mode, polarity_note = infer_result_polarity(rows)

    students_full: list[dict[str, float]] = []
    meta_rows: list[dict[str, Any]] = []

    for idx, row in enumerate(rows):
        expanded = expand_row_to_canonical(row)
        sparse = row_to_sparse_api_payload(expanded)
        student = map_frontend_payload(sparse)
        students_full.append(student)
        truth = parse_file_truth(row, polarity_mode)
        disp = extract_display(row)
        meta_rows.append(
            {
                "row_index": idx + 1,
                "truth": truth,
                "display": disp,
                "sparse_input_keys": sorted(sparse.keys()),
                "form_data": expanded_row_to_form_data(expanded),
                "numeric": {k: float(v) for k, v in sparse.items()},
            }
        )

    try:
        outs = predict_from_student_rows(students_full)
    except Exception as e:
        raise RuntimeError(f"Batch inference failed: {e}") from e

    if len(outs) != len(meta_rows):
        raise RuntimeError("Prediction count mismatch.")

    agree = compare = 0
    results: list[dict[str, Any]] = []
    for meta, out in zip(meta_rows, outs):
        pred = str(out.get("prediction", "Fail"))
        ft = meta["truth"].get("expect_pass")
        if ft is not None:
            compare += 1
            if (pred == "Pass") == bool(ft):
                agree += 1
        results.append(
            {
                "row_index": meta["row_index"],
                "prediction": pred,
                "pass_probability": float(out.get("pass_probability", 0.0)),
                "fail_probability": float(out.get("fail_probability", 0.0)),
                "decision_threshold": out.get("decision_threshold"),
                "file_truth": meta["truth"],
                "display": meta["display"],
                "sparse_input_keys": meta["sparse_input_keys"],
                "form_data": meta["form_data"],
                "numeric": meta["numeric"],
            }
        )

    agreement_pct = round(100.0 * agree / compare, 1) if compare else None

    return {
        "filename": filename,
        "row_count": len(results),
        "headers": headers,
        "polarity": {"mode": polarity_mode, "note": polarity_note},
        "agreement": {
            "compared_rows": compare,
            "matched_rows": agree,
            "percent": agreement_pct,
        },
        "results": results,
    }
