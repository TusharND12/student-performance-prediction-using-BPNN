"""
Professional multipage session report (ReportLab) — EduPredict dashboard theme.
Includes quantitative analytics, cohort summaries, and vector performance charts.
"""
from __future__ import annotations

import io
import re
import statistics
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from backend.labs import export_manifest
from backend.meta import artifact_version
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.shapes import Drawing, String
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

# ── Brand palette (matches frontend :root / dashboard) ─────────────────
PRIMARY = HexColor("#2563eb")
PRIMARY_DARK = HexColor("#1d4ed8")
PRIMARY_FAINT = HexColor("#eff6ff")
SLATE_50 = HexColor("#f8fafc")
SLATE_100 = HexColor("#f1f5f9")
SLATE_200 = HexColor("#e2e8f0")
SLATE_400 = HexColor("#94a3b8")
SLATE_600 = HexColor("#475569")
SLATE_700 = HexColor("#334155")
SLATE_800 = HexColor("#1e293b")
SLATE_900 = HexColor("#0f172a")
SUCCESS = HexColor("#059669")
SUCCESS_BG = HexColor("#ecfdf5")
DANGER = HexColor("#dc2626")
DANGER_BG = HexColor("#fef2f2")
WARNING = HexColor("#d97706")
WHITE = colors.white
ACCENT = HexColor("#6366f1")
ACCENT_SOFT = HexColor("#eef2ff")
MUTED = HexColor("#64748b")
CHART_GRID = HexColor("#e2e8f0")

# Chart dimensions (points, ~170mm wide)
CHART_W = 480
CHART_H = 195


class ReportPdfRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    generated_at: str = ""
    dark_mode: bool = False
    predictions: list[dict[str, Any]] = Field(default_factory=list)
    improvement_plan: dict[str, Any] | None = None
    improvement_snapshot: dict[str, Any] | None = None


INPUT_FIELDS: list[tuple[str, str, str]] = [
    ("attendance_rate", "Attendance rate", "Academic"),
    ("assignment_avg", "Assignment average", "Academic"),
    ("quiz_avg", "Quiz average", "Academic"),
    ("project_score", "Project score", "Academic"),
    ("previous_gpa", "Previous GPA", "Academic"),
    ("standardized_exam_score", "Standardized exam score", "Academic"),
    ("study_hours_daily", "Study hours / day", "Behavioral"),
    ("revision_hours", "Revision hours / day", "Behavioral"),
    ("sleep_hours", "Sleep hours / day", "Behavioral"),
    ("mental_stress", "Mental stress (1–10)", "Behavioral"),
    ("sleep_quality", "Sleep quality (1–10)", "Behavioral"),
    ("learning_efficiency", "Learning efficiency (0–1)", "Behavioral"),
    ("stress_index", "Stress index", "Behavioral"),
    ("physical_activity", "Physical activity (hrs/wk)", "Behavioral"),
    ("lms_login_frequency", "LMS logins / week", "Behavioral"),
    ("coding_practice_hours", "Coding practice hrs / wk", "Behavioral"),
    ("digital_literacy", "Digital literacy (1–10)", "Behavioral"),
    ("parent_involvement", "Parent involvement (1–10)", "Behavioral"),
    ("math_score", "Mathematics", "Subject scores"),
    ("science_score", "Science", "Subject scores"),
    ("english_score", "English", "Subject scores"),
    ("history_score", "History", "Subject scores"),
    ("computer_score", "Computer science", "Subject scores"),
]


def _display_cell(fd: dict[str, Any] | None, key: str) -> str:
    if not fd:
        return "—"
    v = fd.get(key)
    if v is None:
        return "—"
    s = str(v).strip()
    if s in ("", "-", "."):
        return "—"
    return s


def _analytics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(predictions)
    if not n:
        return {"n": 0, "pass": 0, "fail": 0, "avg": None, "min_c": None, "max_c": None}
    confs = [float(p.get("confidence") or 0) for p in predictions]
    pss = sum(1 for p in predictions if p.get("outcome") == "Pass")
    return {
        "n": n,
        "pass": pss,
        "fail": n - pss,
        "avg": sum(confs) / n,
        "min_c": min(confs),
        "max_c": max(confs),
    }


def _parse_composite_label(label: Any) -> float | None:
    if label is None:
        return None
    s = str(label).strip()
    if s in ("", "—", "-", "."):
        return None
    m = re.search(r"([\d.]+)", s)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def _extended_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate stats + chronological series for charts (oldest → newest)."""
    n = len(predictions)
    if not n:
        return {"empty": True}
    confs = [float(p.get("confidence") or 0) for p in predictions]
    chrono = list(reversed(predictions))
    chrono_confs = [float(p.get("confidence") or 0) for p in chrono]

    composites: list[float] = []
    for p in predictions:
        c = _parse_composite_label(p.get("compositeLabel"))
        if c is not None:
            composites.append(c)

    pass_confs = [float(p.get("confidence") or 0) for p in predictions if p.get("outcome") == "Pass"]
    fail_confs = [float(p.get("confidence") or 0) for p in predictions if p.get("outcome") == "Fail"]

    bins = [0] * 10
    for c in confs:
        idx = min(9, int(c // 10))
        bins[idx] += 1

    stdev = statistics.stdev(confs) if n > 1 else 0.0
    cv = (stdev / statistics.mean(confs) * 100.0) if n > 1 and statistics.mean(confs) > 0 else 0.0

    return {
        "empty": False,
        "n": n,
        "mean": statistics.mean(confs),
        "median": statistics.median(confs),
        "stdev": stdev,
        "cv_pct": cv,
        "min_c": min(confs),
        "max_c": max(confs),
        "range_c": max(confs) - min(confs),
        "pass_n": len(pass_confs),
        "fail_n": len(fail_confs),
        "pass_rate": 100.0 * len(pass_confs) / n,
        "pass_mean_conf": statistics.mean(pass_confs) if pass_confs else None,
        "fail_mean_conf": statistics.mean(fail_confs) if fail_confs else None,
        "chrono_confs": chrono_confs,
        "composites": composites,
        "composite_mean": statistics.mean(composites) if composites else None,
        "composite_stdev": statistics.stdev(composites) if len(composites) > 1 else 0.0,
        "histogram_bins": bins,
    }


def _cohort_numeric_averages(predictions: list[dict[str, Any]]) -> dict[str, float | None]:
    keys = (
        "attendance_rate",
        "study_hours_daily",
        "sleep_hours",
        "mental_stress",
    )
    acc: dict[str, list[float]] = {k: [] for k in keys}
    disp: list[float] = []
    for p in predictions:
        n = p.get("numeric") or {}
        for k in keys:
            v = n.get(k)
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if k == "attendance_rate" and fv <= 1.0:
                fv = fv * 100.0
            acc[k].append(fv)
        c = p.get("composite")
        if isinstance(c, (int, float)):
            disp.append(float(c))
    out: dict[str, float | None] = {k: (statistics.mean(v) if v else None) for k, v in acc.items()}
    out["display_subject_avg"] = statistics.mean(disp) if disp else None
    return out


def _interpretation_text(em: dict[str, Any]) -> list[str]:
    """Auto-generated narrative from session metrics."""
    if em.get("empty"):
        return []
    lines: list[str] = []
    n = int(em["n"])
    stdev = float(em["stdev"])
    cv = float(em["cv_pct"])
    pr = float(em["pass_rate"])
    pm, fm = em.get("pass_mean_conf"), em.get("fail_mean_conf")

    if n > 1:
        if stdev >= 18:
            lines.append(
                "<b>Volatility:</b> pass-probability spread is wide across runs — inputs or scenarios differ materially. "
                "Treat the latest snapshot as directional, not a single fixed point estimate."
            )
        elif stdev < 8:
            lines.append(
                "<b>Stability:</b> model pass probabilities cluster tightly — your recent inputs produce a consistent signal."
            )

    if cv > 25 and n > 2:
        lines.append(
            "<b>Relative dispersion:</b> coefficient of variation is elevated; document what changed between runs (sleep, stress, composite) when reporting."
        )

    if pr >= 75:
        lines.append(
            "<b>Session skew:</b> most logged outcomes are Pass — the session may reflect favorable inputs or improvement vs. earlier baselines."
        )
    elif pr <= 35 and n >= 3:
        lines.append(
            "<b>Session skew:</b> Fail outcomes dominate this log — prioritize the Improvement coach levers and re-measure after concrete changes."
        )

    if pm is not None and fm is not None:
        gap = float(pm) - float(fm)
        lines.append(
            f"<b>Outcome separation:</b> mean pass probability is <b>{pm:.0f}%</b> on Pass-labeled runs vs "
            f"<b>{fm:.0f}%</b> on Fail-labeled runs (Δ ≈ {gap:.0f} pts). This aligns with the classifier margin on your session data."
        )

    if not lines:
        lines.append(
            "<b>Summary:</b> metrics are within a typical single-session range; add more diverse runs for richer trend and distribution charts."
        )
    return lines


def _drawing_trajectory(chrono_confs: list[float]) -> Drawing:
    """Line chart: pass probability (%) vs run order (oldest → newest)."""
    n = len(chrono_confs)
    d = Drawing(CHART_W, CHART_H)
    d.add(
        String(
            20,
            CHART_H - 22,
            "Pass probability trajectory (chronological)",
            fontName="Helvetica-Bold",
            fontSize=10,
            fillColor=SLATE_800,
        )
    )
    if n < 2:
        d.add(
            String(
                40,
                CHART_H // 2,
                "Add at least two prediction runs to plot a trend line.",
                fontName="Helvetica",
                fontSize=9,
                fillColor=MUTED,
            )
        )
        return d

    pts = [(float(i + 1), chrono_confs[i]) for i in range(n)]
    lp = LinePlot()
    lp.x = 45
    lp.y = 36
    lp.height = 118
    lp.width = CHART_W - 90
    lp.data = [pts]
    lp.lines[0].strokeColor = ACCENT
    lp.lines[0].strokeWidth = 2.5
    lp.xValueAxis.valueMin = 1
    lp.xValueAxis.valueMax = float(n)
    lp.xValueAxis.valueSteps = [1, max(2, n // 4), max(3, n // 2), n] if n > 4 else [1, float(n)]
    lp.yValueAxis.valueMin = 0
    lp.yValueAxis.valueMax = 100
    lp.yValueAxis.valueSteps = [0, 25, 50, 75, 100]
    lp.yValueAxis.labels.fontName = "Helvetica"
    lp.xValueAxis.labels.fontName = "Helvetica"
    lp.yValueAxis.labels.fontSize = 7
    lp.xValueAxis.labels.fontSize = 7
    d.add(lp)
    d.add(
        String(
            45,
            18,
            "Run order (oldest → newest) · vertical axis: model pass probability %",
            fontName="Helvetica",
            fontSize=7,
            fillColor=MUTED,
        )
    )
    return d


def _drawing_outcome_pie(pass_n: int, fail_n: int) -> Drawing:
    d = Drawing(CHART_W, CHART_H)
    d.add(
        String(
            20,
            CHART_H - 22,
            "Outcome distribution (session)",
            fontName="Helvetica-Bold",
            fontSize=10,
            fillColor=SLATE_800,
        )
    )
    total = pass_n + fail_n
    if total == 0:
        d.add(String(40, CHART_H // 2, "No outcomes", fontName="Helvetica", fontSize=9, fillColor=MUTED))
        return d
    pc = Pie()
    pc.x = CHART_W // 2 - 70
    pc.y = 28
    pc.width = 130
    pc.height = 130
    if fail_n == 0:
        pc.data = [float(pass_n)]
        pc.labels = ["Pass"]
        pc.slices[0].fillColor = SUCCESS
    elif pass_n == 0:
        pc.data = [float(fail_n)]
        pc.labels = ["Fail"]
        pc.slices[0].fillColor = DANGER
    else:
        pc.data = [float(pass_n), float(fail_n)]
        pc.labels = ["Pass", "Fail"]
        pc.slices[0].fillColor = SUCCESS
        pc.slices[1].fillColor = DANGER
    d.add(pc)
    pct_p = 100.0 * pass_n / total
    d.add(
        String(
            45,
            18,
            f"Pass {pass_n} ({pct_p:.0f}%) · Fail {fail_n} ({100 - pct_p:.0f}%)",
            fontName="Helvetica",
            fontSize=7,
            fillColor=MUTED,
        )
    )
    return d


def _drawing_confidence_histogram(bins: list[int]) -> Drawing:
    d = Drawing(CHART_W, CHART_H)
    d.add(
        String(
            20,
            CHART_H - 22,
            "Pass probability histogram (10% bins)",
            fontName="Helvetica-Bold",
            fontSize=10,
            fillColor=SLATE_800,
        )
    )
    mx = max(bins) if bins else 1
    mx = max(mx, 1)
    names = [f"{i * 10}-{i * 10 + 9}" for i in range(10)]
    bc = VerticalBarChart()
    bc.x = 42
    bc.y = 38
    bc.height = 115
    bc.width = CHART_W - 84
    bc.data = [bins]
    bc.categoryAxis.categoryNames = names
    bc.categoryAxis.labels.fontSize = 6
    bc.categoryAxis.labels.angle = 35
    bc.valueAxis.valueMin = 0
    bc.valueAxis.valueMax = float(mx * 1.15) if mx else 5
    bc.valueAxis.labels.fontSize = 7
    bc.bars[0].fillColor = PRIMARY
    bc.bars[0].strokeColor = PRIMARY_DARK
    d.add(bc)
    d.add(
        String(
            45,
            16,
            "X: pass probability % bins · Y: count of prediction runs in this session",
            fontName="Helvetica",
            fontSize=7,
            fillColor=MUTED,
        )
    )
    return d


def _performance_narrative(latest: dict[str, Any] | None) -> list[str]:
    """Short input-performance bullets from the newest run with numeric payload."""
    if not latest:
        return ["No prediction with stored numeric payload was found in this export."]
    num = latest.get("numeric") or {}
    fd = latest.get("formData") or {}
    lines: list[str] = []

    att = num.get("attendance_rate")
    if att is not None:
        ap = float(att) * 100 if float(att) <= 1 else float(att)
        if ap < 65:
            lines.append(f"<b>Attendance signal is weak</b> (~{ap:.0f}%): the model uses attendance directly; sustained low presence usually correlates with weaker outcomes.")
        elif ap < 82:
            lines.append(f"Attendance is moderate (~{ap:.0f}%): consider pushing toward the mid‑80s+ band for a stronger engagement profile.")
        else:
            lines.append(f"Attendance looks solid (~{ap:.0f}%): keep consistency through assessment periods.")

    comp_disp: float | None = None
    if latest.get("composite") is not None:
        try:
            comp_disp = float(latest.get("composite"))
        except (TypeError, ValueError):
            comp_disp = None
    if comp_disp is None:
        comp_disp = _parse_composite_label(latest.get("compositeLabel"))
    if comp_disp is not None:
        c = float(comp_disp)
        if c < 58:
            lines.append(
                f"<b>Average subject marks (display)</b> are low ({c:.1f}%): for reference only — the model uses individual fields; <i>previous_marks</i> follows the mean of subject scores."
            )
        elif c < 72:
            lines.append(
                f"Average subject marks (display) are mid‑range ({c:.1f}%): tighten weaker subjects over 2–3 weeks."
            )
        else:
            lines.append(
                f"Average subject marks (display) look healthy ({c:.1f}%): keep breadth across subjects and core academics."
            )

    study = num.get("study_hours_daily")
    if study is not None and _display_cell(fd, "study_hours_daily") != "—":
        s = float(study)
        if s < 1.25:
                lines.append(f"<b>Study hours are very low</b> ({s:.1f} h/day): gradually add focused blocks; the model includes <i>study_hours_daily</i>.")
        elif s < 2.5:
            lines.append(f"Study time ({s:.1f} h/day) is workable; add active recall and timed practice for depth.")

    sleep = num.get("sleep_hours")
    if sleep is not None and _display_cell(fd, "sleep_hours") != "—":
        sl = float(sleep)
        if sl < 6:
            lines.append(f"<b>Sleep duration is short</b> ({sl:.1f} h): recovery supports stress regulation; stress×sleep is an engineered feature.")
        elif sl < 7:
            lines.append(f"Sleep ({sl:.1f} h) is close to minimum; a stable wake time often helps learning consistency.")

    stress = num.get("mental_stress")
    if stress is not None and _display_cell(fd, "mental_stress") != "—":
        st = float(stress)
        if st >= 7:
            lines.append(f"<b>Mental stress is high</b> ({st:.0f}/10): pair workload cuts with recovery; stress interacts with sleep in the pipeline.")
        elif st >= 5:
            lines.append(f"Mental stress is elevated ({st:.0f}/10): use short planning windows to prevent spikes before deadlines.")

    if not lines:
        lines.append(
            "Run additional predictions with fields filled in to unlock attendance, subject-average display, study, sleep, and stress commentary."
        )
    return lines


def _make_styles() -> tuple[dict[str, ParagraphStyle], ParagraphStyle]:
    base = getSampleStyleSheet()
    styles: dict[str, ParagraphStyle] = {}

    styles["title"] = ParagraphStyle(
        name="Title",
        parent=base["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=24,
        textColor=SLATE_900,
        spaceAfter=6,
        leading=28,
    )
    styles["subtitle"] = ParagraphStyle(
        name="Subtitle",
        parent=base["Normal"],
        fontSize=10,
        textColor=SLATE_600,
        spaceAfter=6,
        leading=14,
    )
    styles["tagline"] = ParagraphStyle(
        name="Tagline",
        parent=base["Normal"],
        fontSize=8.5,
        textColor=ACCENT,
        fontName="Helvetica-Bold",
        spaceAfter=14,
        leading=11,
    )
    styles["h1"] = ParagraphStyle(
        name="H1",
        parent=base["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13.5,
        textColor=SLATE_900,
        spaceBefore=12,
        spaceAfter=8,
        borderPadding=0,
    )
    styles["h2"] = ParagraphStyle(
        name="H2",
        parent=base["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=11,
        textColor=PRIMARY_DARK,
        spaceBefore=10,
        spaceAfter=6,
    )
    styles["h3"] = ParagraphStyle(
        name="H3",
        parent=base["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9.5,
        textColor=SLATE_700,
        spaceBefore=8,
        spaceAfter=4,
    )
    styles["body"] = ParagraphStyle(
        name="Body",
        parent=base["Normal"],
        fontSize=9,
        textColor=SLATE_700,
        leading=13.5,
        alignment=TA_JUSTIFY,
        spaceAfter=6,
    )
    styles["body_tight"] = ParagraphStyle(
        name="BodyTight",
        parent=styles["body"],
        spaceAfter=4,
    )
    styles["caption"] = ParagraphStyle(
        name="Caption",
        parent=base["Normal"],
        fontSize=8,
        textColor=SLATE_400,
        leading=11,
        spaceAfter=10,
    )
    styles["figure_cap"] = ParagraphStyle(
        name="FigureCap",
        parent=base["Normal"],
        fontSize=8,
        textColor=MUTED,
        leading=11,
        spaceAfter=14,
        leftIndent=0,
    )
    styles["metric_cell"] = ParagraphStyle(
        name="MetricCell",
        parent=base["Normal"],
        fontSize=9,
        textColor=SLATE_800,
        leading=12,
    )
    styles["kpi_val"] = ParagraphStyle(
        name="KpiVal",
        parent=base["Normal"],
        fontName="Helvetica-Bold",
        fontSize=14,
        textColor=SLATE_900,
        alignment=TA_CENTER,
        leading=17,
    )
    styles["kpi_lab"] = ParagraphStyle(
        name="KpiLab",
        parent=base["Normal"],
        fontSize=7.5,
        textColor=SLATE_600,
        alignment=TA_CENTER,
        leading=10,
    )
    return styles, base["Normal"]


def _page_canvas_factory(dark_band: bool):
    """Top brand bar, accent rule, footer — editorial / dashboard style."""

    def _draw(canv: Any, doc: Any) -> None:
        w, h = A4
        canv.saveState()
        band = SLATE_900 if dark_band else PRIMARY
        canv.setFillColor(band)
        canv.rect(0, h - 11 * mm, w, 11 * mm, stroke=0, fill=1)
        canv.setStrokeColor(ACCENT if not dark_band else SLATE_700)
        canv.setLineWidth(1.2)
        canv.line(14 * mm, h - 11 * mm, 52 * mm, h - 11 * mm)
        canv.setFillColor(WHITE)
        canv.setFont("Helvetica-Bold", 10.5)
        canv.drawString(14 * mm, h - 7.2 * mm, "EduPredict")
        canv.setFont("Helvetica", 7.5)
        canv.drawRightString(w - 14 * mm, h - 7.2 * mm, "Analytics & performance intelligence · BPNN session export")
        canv.setStrokeColor(SLATE_200)
        canv.setLineWidth(0.35)
        canv.line(14 * mm, 9.2 * mm, w - 14 * mm, 9.2 * mm)
        canv.setFillColor(SLATE_400)
        canv.setFont("Helvetica", 7.5)
        canv.drawString(14 * mm, 6 * mm, f"Page {canv.getPageNumber()}")
        canv.drawRightString(w - 14 * mm, 6 * mm, "Confidential · for academic / advisory use")
        canv.restoreState()

    return _draw


def build_session_report_pdf(req: ReportPdfRequest) -> bytes:
    buf = io.BytesIO()
    preds = req.predictions or []
    dark = bool(req.dark_mode)

    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=22 * mm,
        bottomMargin=16 * mm,
        title="EduPredict Session Report",
        onFirstPage=_page_canvas_factory(dark),
        onLaterPages=_page_canvas_factory(dark),
    )

    styles, _ = _make_styles()
    story: list[Any] = []

    # ── Cover (extra top space; header still draws)
    story.append(Spacer(1, 8 * mm))
    story.append(Paragraph("Session performance report", styles["title"]))
    story.append(
        Paragraph(
            "ADVANCED ANALYTICS · VECTOR VISUALIZATIONS · COHORT METRICS",
            styles["tagline"],
        )
    )
    story.append(
        Paragraph(
            f"<font color='#64748b'>Generated {req.generated_at or '—'}</font><br/>"
            f"<font size=9>Workspace theme: <b>{'Dark' if dark else 'Light'}</b> (PDF charts use a print-optimized light palette)</font>",
            styles["subtitle"],
        )
    )

    ax = _analytics(preds)
    kpi_data = [
        [
            Paragraph(f"{ax['n']}", styles["kpi_val"]),
            Paragraph(f"{ax['pass']}", styles["kpi_val"]),
            Paragraph(f"{ax['fail']}", styles["kpi_val"]),
            Paragraph(
                f"{ax['avg']:.0f}%" if ax["avg"] is not None else "—",
                styles["kpi_val"],
            ),
        ],
        [
            Paragraph("Total runs", styles["kpi_lab"]),
            Paragraph("Pass", styles["kpi_lab"]),
            Paragraph("Fail", styles["kpi_lab"]),
            Paragraph("Avg pass prob.", styles["kpi_lab"]),
        ],
    ]
    kt = Table(kpi_data, colWidths=[42 * mm, 42 * mm, 42 * mm, 42 * mm])
    kt.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), PRIMARY_FAINT),
                ("BACKGROUND", (0, 1), (-1, 1), SLATE_50),
                ("BOX", (0, 0), (-1, -1), 0.75, SLATE_200),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, SLATE_200),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(kt)
    story.append(Spacer(1, 6 * mm))

    story.append(Paragraph("Executive summary", styles["h1"]))
    story.append(
        Paragraph(
            "This document consolidates your <b>browser session</b>: logged BPNN predictions, captured form inputs (when available), "
            "aggregate performance indicators, and structured improvement guidance aligned with the same levers the FastAPI mapper sends "
            "to <i>model.h5</i>. It is suitable for coursework, viva, or stakeholder review — not a substitute for instructors or clinicians.",
            styles["body"],
        )
    )

    if ax["n"]:
        pass_pct = 100.0 * ax["pass"] / ax["n"]
        story.append(
            Paragraph(
                f"Over <b>{ax['n']}</b> run(s), <b>{ax['pass']}</b> were labeled Pass and <b>{ax['fail']}</b> Fail "
                f"({pass_pct:.0f}% pass rate in this session). "
                f"Reported model pass probability ranged from <b>{ax['min_c']:.0f}%</b> to <b>{ax['max_c']:.0f}%</b> "
                f"(mean <b>{ax['avg']:.0f}%</b>).",
                styles["body"],
            )
        )
    else:
        story.append(Paragraph("No predictions were included in this export.", styles["body"]))

    em = _extended_metrics(preds)
    if not em.get("empty"):
        story.append(PageBreak())
        story.append(Paragraph("Quantitative session analytics", styles["h1"]))
        story.append(
            Paragraph(
                "Derived from every logged run in this export: pass-probability distribution, outcome mix, and mean subject-marks display. "
                "Figure axes use the same chronological ordering as the web Analytics view (oldest run → newest run).",
                styles["body"],
            )
        )
        pm_txt = f"{em['pass_mean_conf']:.1f}%" if em.get("pass_mean_conf") is not None else "—"
        fm_txt = f"{em['fail_mean_conf']:.1f}%" if em.get("fail_mean_conf") is not None else "—"
        comp_txt = f"{em['composite_mean']:.1f}%" if em.get("composite_mean") is not None else "—"
        mrows = [
            ["Metric", "Value"],
            ["Mean pass probability", f"{em['mean']:.1f}%"],
            ["Median", f"{em['median']:.1f}%"],
            ["Std. deviation", f"{em['stdev']:.1f} pts"],
            ["Range (max − min)", f"{em['range_c']:.1f} pts"],
            ["Coeff. of variation (σ/μ)", f"{em['cv_pct']:.1f}%"],
            ["Pass rate (session)", f"{em['pass_rate']:.1f}%"],
            ["Mean P(pass) | Pass label", pm_txt],
            ["Mean P(pass) | Fail label", fm_txt],
            ["Mean avg subject marks % (display, parsed)", comp_txt],
        ]
        mt = Table(mrows, colWidths=[78 * mm, 92 * mm], repeatRows=1)
        mt.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), SLATE_800),
                    ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, ACCENT_SOFT]),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("BOX", (0, 0), (-1, -1), 0.65, SLATE_200),
                    ("INNERGRID", (0, 0), (-1, -1), 0.25, SLATE_200),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 7),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        story.append(mt)
        story.append(Spacer(1, 5 * mm))

        story.append(Paragraph("Interpretation engine", styles["h2"]))
        for line in _interpretation_text(em):
            story.append(Paragraph(line, styles["body"]))

        cohort = _cohort_numeric_averages(preds)
        if any(v is not None for v in cohort.values()):
            story.append(Paragraph("Cohort averages (numeric payloads)", styles["h2"]))
            labels = {
                "display_subject_avg": "Avg subject marks % (display, session mean)",
                "attendance_rate": "Attendance (%)",
                "study_hours_daily": "Study hours / day",
                "sleep_hours": "Sleep hours / day",
                "mental_stress": "Mental stress (1–10)",
            }
            crows = [["Signal", "Session average"]]
            for k, lab in labels.items():
                v = cohort.get(k)
                crows.append([lab, f"{v:.2f}" if v is not None else "—"])
            ct = Table(crows, colWidths=[78 * mm, 92 * mm], repeatRows=1)
            ct.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
                        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, SLATE_50]),
                        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                        ("BOX", (0, 0), (-1, -1), 0.65, SLATE_200),
                        ("INNERGRID", (0, 0), (-1, -1), 0.25, SLATE_200),
                        ("LEFTPADDING", (0, 0), (-1, -1), 6),
                        ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ]
                )
            )
            story.append(ct)
            story.append(
                Paragraph(
                    "Averages include only runs where each field exists in the mapped numeric payload — aligned with dashboard cohort summaries.",
                    styles["caption"],
                )
            )

        story.append(PageBreak())
        story.append(Paragraph("Performance visualizations", styles["h1"]))
        story.append(
            Paragraph(
                "Vector figures (ReportLab Graphics) — no raster screenshots. They summarize the session without a live browser and stay sharp when printed.",
                styles["body"],
            )
        )
        story.append(Spacer(1, 2 * mm))
        story.append(_drawing_trajectory(em["chrono_confs"]))
        story.append(
            Paragraph(
                "<i>Figure 1.</i> Pass probability (%) vs run order (oldest → newest). "
                "Upward slopes often track improved attendance, subject averages, sleep, or reduced stress in later runs.",
                styles["figure_cap"],
            )
        )
        story.append(_drawing_outcome_pie(em["pass_n"], em["fail_n"]))
        story.append(
            Paragraph(
                "<i>Figure 2.</i> Outcome counts (Pass vs Fail). Compare with Figure 3 to see whether probabilities cluster near the 50% decision boundary.",
                styles["figure_cap"],
            )
        )
        story.append(PageBreak())
        story.append(_drawing_confidence_histogram(em["histogram_bins"]))
        story.append(
            Paragraph(
                "<i>Figure 3.</i> Histogram of pass-probability bins (10% wide). "
                "Mass below 50% indicates repeated fail-leaning scenarios within this session log.",
                styles["figure_cap"],
            )
        )

    story.append(PageBreak())

    # ── Input performance (newest run with data)
    story.append(Paragraph("Input performance analysis", styles["h1"]))
    latest = next((p for p in preds if p.get("numeric")), None)
    for line in _performance_narrative(latest):
        story.append(Paragraph(line, styles["body"]))

    if len(preds) > 1 and latest and not em.get("empty"):
        lc = float(latest.get("confidence") or 0)
        diff = lc - float(em["mean"])
        story.append(
            Paragraph(
                f"<b>Latest run vs session mean:</b> the newest prediction with numeric payload is at <b>{lc:.0f}%</b> pass probability, "
                f"compared with a session mean of <b>{em['mean']:.1f}%</b> "
                f"(<font color='#6366f1'><b>{diff:+.1f}</b> points</font> vs mean).",
                styles["body"],
            )
        )

    story.append(Spacer(1, 4 * mm))
    story.append(Paragraph("Model-facing signals (reference)", styles["h2"]))
    story.append(
        Paragraph(
            "The API merges numeric form fields with the trained feature vector (training-set medians fill omitted columns), "
            "applies the saved <b>MinMaxScaler</b>, and runs the Keras BPNN (threshold 0.5). "
            "Older deployments used engineered <b>performance_index</b> features; the current bundle follows the training script saved next to <b>model.h5</b>.",
            styles["body_tight"],
        )
    )

    story.append(PageBreak())

    # ── Improvements
    story.append(Paragraph("Recommended improvements", styles["h1"]))
    plan = req.improvement_plan or {}
    items = plan.get("items") or []
    meta = plan.get("meta") or {}

    if req.improvement_snapshot and meta:
        story.append(Paragraph("Profile used for coaching (last captured snapshot)", styles["h2"]))
        meta_rows = [
            ["Captured", str(meta.get("atLabel") or "—")],
            ["Outcome", str(meta.get("outcome") or "—")],
            ["Pass probability (%)", str(meta.get("confidence") if meta.get("confidence") is not None else "—")],
            ["Avg subject marks (display)", str(meta.get("compositeLabel") or "—")],
            ["Attendance (approx. %)", str(meta.get("attPct")) if meta.get("attPct") is not None else "—"],
        ]
        mt = Table(meta_rows, colWidths=[52 * mm, 118 * mm])
        mt.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, -1), SLATE_100),
                    ("TEXTCOLOR", (0, 0), (-1, -1), SLATE_800),
                    ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                    ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("BOX", (0, 0), (-1, -1), 0.5, SLATE_200),
                    ("INNERGRID", (0, 0), (-1, -1), 0.25, SLATE_200),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        story.append(mt)
        story.append(Spacer(1, 4 * mm))

        missing = plan.get("missingModelFields") or []
        if missing:
            story.append(
                Paragraph(
                    f"<b>Fields to strengthen next run:</b> {', '.join(missing)}.",
                    styles["body"],
                )
            )

    if not items:
        story.append(
            Paragraph(
                "No improvement items were computed for this export. Run a successful prediction and open <b>Improvement coach</b> "
                "in the app, or ensure the snapshot is included when generating this PDF.",
                styles["body"],
            )
        )
    else:
        story.append(
            Paragraph(
                f"The following <b>{len(items)}</b> prioritized items mirror the in-app Improvement coach (impact, rationale, and actions).",
                styles["body"],
            )
        )
        for i, it in enumerate(items[:18], 1):
            pr = it.get("priority", 3)
            impact = "High impact" if pr == 1 else "Medium" if pr == 2 else "Supporting"
            cat = str(it.get("category") or "—")
            title = str(it.get("title") or "—")
            why = str(it.get("why") or "")
            actions = it.get("actions") or []
            act_txt = " ".join(str(a) for a in actions) if actions else ""
            band_hex = "#dc2626" if pr == 1 else "#d97706" if pr == 2 else "#64748b"
            story.append(Spacer(1, 2 * mm))
            story.append(
                Paragraph(
                    f"<font color='{band_hex}'><b>{i}. [{impact}]</b></font> "
                    f"<b>{title}</b> <font color='#64748b'>· {cat}</font>",
                    styles["h2"],
                )
            )
            story.append(Paragraph(why, styles["body_tight"]))
            if act_txt:
                story.append(Paragraph(f"<b>Suggested steps:</b> {act_txt}", styles["body_tight"]))
            story.append(HRFlowable(width="100%", thickness=0.5, color=SLATE_200, spaceBefore=4, spaceAfter=2))

    story.append(PageBreak())

    # ── Runs index
    story.append(Paragraph("Prediction log — overview", styles["h1"]))
    idx_body: list[list[Any]] = []
    for j, p in enumerate(preds[:45], 1):
        has_in = "Yes" if p.get("formData") else "No"
        idx_body.append(
            [
                str(j),
                str(p.get("atLabel") or "—"),
                str(p.get("outcome") or "—"),
                f"{p.get('confidence', '—')}%",
                str(p.get("compositeLabel") or "—"),
                has_in,
            ]
        )
    if not idx_body:
        idx_body = [["—", "—", "—", "—", "—", "—"]]
    idx_table = Table(
        [["#", "When", "Outcome", "Pass prob.", "Composite", "Inputs stored"], *idx_body],
        colWidths=[10 * mm, 38 * mm, 22 * mm, 24 * mm, 28 * mm, 28 * mm],
        repeatRows=1,
    )
    idx_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
                ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 8),
                ("FONTSIZE", (0, 1), (-1, -1), 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, SLATE_50]),
                ("BOX", (0, 0), (-1, -1), 0.75, SLATE_200),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, SLATE_200),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(idx_table)

    # ── Per-run detail
    detailed = [p for p in preds if p.get("formData")]
    for idx, run in enumerate(detailed[:22], 1):
        story.append(PageBreak())
        story.append(Paragraph(f"Detailed run #{idx}", styles["h1"]))
        oc = run.get("outcome") or "—"
        badge_bg = SUCCESS_BG if oc == "Pass" else DANGER_BG
        badge_hex = "#059669" if oc == "Pass" else "#dc2626"
        head_data = [
            [
                Paragraph(f"<b><font color='{badge_hex}'>{oc}</font></b>", styles["body"]),
                Paragraph(
                    f"<b>When:</b> {run.get('atLabel') or '—'}<br/>"
                    f"<b>Pass probability:</b> {run.get('confidence', '—')}%<br/>"
                    f"<b>Avg subject marks:</b> {run.get('compositeLabel') or '—'}",
                    styles["body_tight"],
                ),
            ]
        ]
        ht = Table(head_data, colWidths=[28 * mm, 142 * mm])
        ht.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (0, 0), badge_bg),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("TOPPADDING", (0, 0), (-1, -1), 8),
                    ("BOX", (0, 0), (-1, -1), 0.75, SLATE_200),
                ]
            )
        )
        story.append(ht)
        story.append(Spacer(1, 4 * mm))

        fd = run.get("formData") or {}
        rows = [["Section", "Field", "Value entered"]]
        for key, label, group in INPUT_FIELDS:
            rows.append([group, label, _display_cell(fd, key)])
        num = run.get("numeric") or {}
        if run.get("composite") is not None:
            rows.append(["Display", "Avg subject marks % (mean of entered subjects)", f"{run.get('composite')}"])

        t = Table(rows, colWidths=[32 * mm, 58 * mm, 80 * mm], repeatRows=1)
        t.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), SLATE_800),
                    ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, SLATE_50]),
                    ("BOX", (0, 0), (-1, -1), 0.75, SLATE_200),
                    ("INNERGRID", (0, 0), (-1, -1), 0.25, SLATE_200),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(t)

    if len(detailed) > 22:
        story.append(PageBreak())
        story.append(Paragraph("Note", styles["h1"]))
        story.append(
            Paragraph(
                f"{len(detailed) - 22} additional run(s) with stored inputs are listed in the overview table only. "
                "Use JSON export in Reports for the full machine-readable session.",
                styles["body"],
            )
        )

    story.append(PageBreak())
    story.append(Paragraph("Appendix — EduPredict application map", styles["h1"]))
    app_rows = [
        ["Dashboard", "KPIs and recent activity."],
        ["Performance Predictor", "Academic + behavioral form; avg subject marks for display; BPNN inference."],
        ["Student Records", "Searchable session log."],
        ["Reports", "JSON + this PDF export."],
        ["Analytics", "Pass mix, confidence trends, histograms, subject marks average by run."],
        ["Improvement coach", "Prioritized actions from your last snapshot."],
        ["Settings", "API health, theme, replay landing."],
        ["Help", "FAQs."],
    ]
    at = Table([["Area", "Description"], *app_rows], colWidths=[45 * mm, 125 * mm])
    at.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
                ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, PRIMARY_FAINT]),
                ("BOX", (0, 0), (-1, -1), 0.75, SLATE_200),
                ("INNERGRID", (0, 0), (-1, -1), 0.25, SLATE_200),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(at)
    story.append(Spacer(1, 8 * mm))
    story.append(
        Paragraph(
            "<i>Artifacts:</i> model.h5, scaler.pkl · <i>Stack:</i> TensorFlow Keras, FastAPI, React, Vite.",
            styles["caption"],
        )
    )

    man = export_manifest()
    ver = artifact_version()
    story.append(PageBreak())
    story.append(Paragraph("Provenance & integrity (paper trail)", styles["h1"]))
    story.append(
        Paragraph(
            f"This export can be cross-checked with the <b>Innovation Labs</b> manifest endpoint. "
            f"<b>Nonce:</b> {man['export_nonce']} · "
            f"<b>Model fingerprint (prefix):</b> {man['model_sha256_prefix']} — "
            f"SHA-256 over the first 64KB of <i>model.h5</i>. "
            f"<b>scaler.pkl prefix:</b> {ver.get('scaler_pkl_sha256_prefix', '—')} · "
            f"<b>Schema:</b> {ver.get('export_schema', '—')}.",
            styles["body"],
        )
    )

    doc.build(story)
    return buf.getvalue()
