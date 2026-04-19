"""
Recalibrate the saved BPNN's decision threshold without retraining.

Why:
  The deployed artifact had `decision_threshold=0.89` (accuracy-target calibration),
  which rejects most rows from sparse CSV imports (marks-only files). This script
  loads the exact artifacts served by the backend, rebuilds the same validation
  split (seed + leakage exclusions from training_hist.json), scores val probabilities,
  picks a new threshold with the chosen metric, and overwrites `scaler.pkl` in place.

Usage (from inner ML_Project2 folder):
  python scripts/recalibrate_threshold.py --metric balanced_accuracy
  python scripts/recalibrate_threshold.py --metric accuracy --target-val-acc 0.80
  python scripts/recalibrate_threshold.py --threshold 0.5        # fixed override
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow import keras

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "model.h5"
SCALER_PATH = ROOT / "scaler.pkl"
DEFAULT_CSV = ROOT / "student_academic_performance_1M.csv"
TRAIN_HIST = ROOT / "training_hist.json"
TARGET = "pass_fail"


def _threshold_grid() -> np.ndarray:
    return np.linspace(0.02, 0.98, 97)


def _calibrate(y_true: np.ndarray, proba: np.ndarray, metric: str, target_acc: float) -> float:
    y_true = y_true.astype(int).reshape(-1)
    proba = np.clip(proba.reshape(-1).astype(np.float64), 1e-6, 1.0 - 1e-6)
    grid = _threshold_grid()
    if metric == "balanced_accuracy":
        best_t, best_score = 0.5, -1.0
        for t in grid:
            pred = (proba >= t).astype(int)
            score = balanced_accuracy_score(y_true, pred)
            if score > best_score:
                best_score, best_t = score, float(t)
        return best_t
    if metric == "accuracy":
        best_t, best_err = 0.5, 1.0
        for t in grid:
            pred = (proba >= t).astype(int)
            acc = float((pred == y_true).mean())
            err = abs(acc - target_acc)
            if err < best_err:
                best_err, best_t = err, float(t)
        return best_t
    raise ValueError(f"Unknown metric: {metric!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.15)
    ap.add_argument("--val-size", type=float, default=0.15)
    ap.add_argument(
        "--metric",
        choices=("accuracy", "balanced_accuracy"),
        default="balanced_accuracy",
        help="How to pick the new threshold.",
    )
    ap.add_argument(
        "--target-val-acc",
        type=float,
        default=0.80,
        help="Only used with --metric accuracy.",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override: set this literal threshold instead of calibrating.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print new threshold but do not write scaler.pkl.",
    )
    args = ap.parse_args()

    if not MODEL_PATH.is_file() or not SCALER_PATH.is_file():
        raise SystemExit("model.h5 / scaler.pkl not found in project root.")
    if not args.csv.is_file():
        raise SystemExit(f"CSV not found: {args.csv}")

    print("Loading artifacts...")
    artifact = joblib.load(SCALER_PATH)
    feat_cols = list(artifact["feature_columns"])
    scaler = artifact["scaler"]
    current_thr = float(artifact.get("decision_threshold", 0.5))
    current_metric = artifact.get("threshold_metric")
    print(f"Current decision_threshold={current_thr} (metric={current_metric!r})")

    print("Loading model.h5...")
    model = keras.models.load_model(MODEL_PATH)

    print(f"Loading CSV ({args.csv.name})...")
    df = pd.read_csv(args.csv, nrows=args.max_rows)
    if TARGET not in df.columns:
        raise SystemExit(f"Missing target column {TARGET!r} in CSV.")

    missing_feats = [c for c in feat_cols if c not in df.columns]
    if missing_feats:
        raise SystemExit(
            "CSV is missing feature columns required by the artifact: " + ", ".join(missing_feats)
        )
    for c in feat_cols + [TARGET]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    before = len(df)
    df = df.dropna(subset=feat_cols + [TARGET]).reset_index(drop=True)
    print(f"Rows: {before} -> {len(df)} after NA drop")

    X = df[feat_cols].astype(np.float32).to_numpy()
    y = df[TARGET].astype(np.float32).to_numpy().reshape(-1, 1)

    X_train, X_temp, _y_train, y_temp = train_test_split(
        X, y, test_size=args.test_size + args.val_size, random_state=args.seed, stratify=y
    )
    _ = X_train
    rel_val = args.val_size / (args.test_size + args.val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1.0 - rel_val, random_state=args.seed, stratify=y_temp
    )
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    print("Predicting val probabilities...")
    val_proba = model.predict(X_val_s, verbose=0).reshape(-1)
    y_val_flat = y_val.reshape(-1).astype(int)

    if args.threshold is not None:
        new_thr = float(args.threshold)
        new_metric = None
        print(f"Using --threshold override: {new_thr}")
    else:
        new_thr = _calibrate(
            y_val_flat,
            val_proba,
            metric=args.metric,
            target_acc=args.target_val_acc,
        )
        new_metric = args.metric
        print(f"New decision_threshold={new_thr:.4f} (metric={new_metric!r})")

    val_pred = (val_proba >= new_thr).astype(int)
    print("Val accuracy:", round(accuracy_score(y_val_flat, val_pred), 4))
    print("Val balanced accuracy:", round(balanced_accuracy_score(y_val_flat, val_pred), 4))
    print("Val confusion (rows=true [0,1], cols=pred [0,1]):\n", confusion_matrix(y_val_flat, val_pred))

    test_proba = model.predict(X_test_s, verbose=0).reshape(-1)
    y_test_flat = y_test.reshape(-1).astype(int)
    test_pred = (test_proba >= new_thr).astype(int)
    print("\n=== Holdout test metrics ===")
    print("Accuracy:", round(accuracy_score(y_test_flat, test_pred), 4))
    print("Balanced accuracy:", round(balanced_accuracy_score(y_test_flat, test_pred), 4))
    print("Confusion:\n", confusion_matrix(y_test_flat, test_pred))

    if args.dry_run:
        print("\n--dry-run set: scaler.pkl NOT updated.")
        return

    artifact["decision_threshold"] = float(new_thr)
    artifact["threshold_metric"] = new_metric
    tc = dict(artifact.get("training_config") or {})
    tc["decision_threshold"] = float(new_thr)
    tc["threshold_metric"] = new_metric
    tc["recalibrated_without_retrain"] = True
    artifact["training_config"] = tc

    joblib.dump(artifact, SCALER_PATH)
    try:
        TRAIN_HIST.write_text(json.dumps(tc, indent=2), encoding="utf-8")
    except OSError:
        pass
    print(f"\nUpdated {SCALER_PATH.name}: decision_threshold={new_thr:.4f}, metric={new_metric!r}")
    print("Restart the API (or rely on --reload) so inference picks up the new artifact.")


if __name__ == "__main__":
    main()
