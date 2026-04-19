"""
Train BPNN on student_academic_performance_1M.csv using all numeric input columns
(except the binary target). Saves model.h5 + scaler.pkl at project root.

Architecture (per project brief):
  - Wider first hidden units (128 vs legacy 64)
  - Additional hidden layer
  - Dropout after dense blocks to limit overfitting
  - Tunable learning rate (default 3e-4; optional quick LR sweep)

Usage (from project root ML_Project2/):
  python scripts/train_bpnn_1m.py
  python scripts/train_bpnn_1m.py --max-rows 200000
  python scripts/train_bpnn_1m.py --lr-sweep   # short search on train subset
  python scripts/train_bpnn_1m.py --include-leaky   # keep outcome-adjacent columns (not recommended)
  python scripts/train_bpnn_1m.py --threshold-metric balanced_accuracy  # fairer threshold under imbalance
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = ROOT / "student_academic_performance_1M.csv"
TARGET = "pass_fail"

# Outcome-adjacent / label-leak columns (near-perfect train fit if kept).
DEFAULT_LEAKY_EXCLUDE: frozenset[str] = frozenset(
    {
        "honors_flag",
        "at_risk_flag",
        "top_performer_flag",
        "final_gpa",
        "dropout_risk_score",
        "improvement_next_term",
    }
)


def feature_columns_from_df(df: pd.DataFrame, exclude: frozenset[str]) -> list[str]:
    """All columns except target and excluded names."""
    cols = [c for c in df.columns if c != TARGET and c not in exclude]
    if not cols:
        raise ValueError("No feature columns after removing target and exclusions.")
    return cols


def set_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def _threshold_grid() -> np.ndarray:
    return np.linspace(0.02, 0.98, 97)


def calibrate_decision_threshold(
    y_true: np.ndarray,
    proba: np.ndarray,
    *,
    metric: str,
    target_acc: float = 0.90,
) -> float:
    """
    Pick probability threshold on a labeled set (typically validation).

    - metric="accuracy": threshold with accuracy closest to target_acc (plain accuracy is
      sensitive to class imbalance).
    - metric="balanced_accuracy": threshold that maximizes balanced accuracy (equal weight
      to both classes' recall).
    """
    y_true = y_true.astype(int).reshape(-1)
    proba = np.clip(proba.reshape(-1).astype(np.float64), 1e-6, 1.0 - 1e-6)
    grid = _threshold_grid()
    if metric == "balanced_accuracy":
        best_t, best_score = 0.5, -1.0
        for t in grid:
            pred = (proba >= t).astype(int)
            score = balanced_accuracy_score(y_true, pred)
            if score > best_score:
                best_score = score
                best_t = float(t)
        return best_t
    if metric == "accuracy":
        best_t, best_err = 0.5, 1.0
        for t in grid:
            pred = (proba >= t).astype(int)
            acc = float((pred == y_true).mean())
            err = abs(acc - target_acc)
            if err < best_err:
                best_err = err
                best_t = float(t)
        return best_t
    raise ValueError(f"Unknown threshold metric: {metric!r} (use accuracy or balanced_accuracy)")


def build_model(
    input_dim: int,
    h1: int,
    h2: int,
    h3: int,
    dropout1: float,
    dropout2: float,
    dropout3: float,
    lr: float,
    *,
    l2: float,
    weight_decay: float,
    label_smoothing: float,
    input_noise_std: float,
) -> keras.Model:
    reg = keras.regularizers.l2(l2)
    stack: list = [layers.Input(shape=(input_dim,))]
    if input_noise_std and input_noise_std > 0:
        stack.append(layers.GaussianNoise(input_noise_std))
    stack.extend(
        [
            layers.Dense(h1, activation="relu", kernel_regularizer=reg),
            layers.BatchNormalization(),
            layers.Dropout(dropout1),
            layers.Dense(h2, activation="relu", kernel_regularizer=reg),
            layers.BatchNormalization(),
            layers.Dropout(dropout2),
            layers.Dense(h3, activation="relu", kernel_regularizer=reg),
            layers.BatchNormalization(),
            layers.Dropout(dropout3),
            layers.Dense(1, activation="sigmoid"),
        ]
    )
    model = models.Sequential(stack)
    loss = keras.losses.BinaryCrossentropy(label_smoothing=label_smoothing)
    opt = keras.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay)
    model.compile(
        optimizer=opt,
        loss=loss,
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model


def quick_lr_sweep(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    input_dim: int,
    candidates: list[float],
    h1: int,
    h2: int,
    h3: int,
    d1: float,
    d2: float,
    d3: float,
    build_kw: dict,
    epochs: int = 4,
    batch_size: int = 2048,
) -> float:
    best_lr = candidates[0]
    best = float("inf")
    for lr in candidates:
        m = build_model(
            input_dim,
            h1,
            h2,
            h3,
            d1,
            d2,
            d3,
            lr,
            **build_kw,
        )
        hist = m.fit(
            X_tr,
            y_tr,
            validation_data=(X_va, y_va),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0,
        )
        vloss = float(min(hist.history["val_loss"]))
        print(f"  [lr-sweep] lr={lr:g}  min val_loss={vloss:.5f}")
        if vloss < best:
            best = vloss
            best_lr = lr
        keras.backend.clear_session()
    print(f"  [lr-sweep] chosen lr={best_lr:g}")
    return best_lr


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--max-rows", type=int, default=None, help="Cap rows for faster runs (default: all).")
    p.add_argument("--test-size", type=float, default=0.15)
    p.add_argument("--val-size", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--h1", type=int, default=64)
    p.add_argument("--h2", type=int, default=64)
    p.add_argument("--h3", type=int, default=32)
    p.add_argument("--dropout1", type=float, default=0.55)
    p.add_argument("--dropout2", type=float, default=0.5)
    p.add_argument("--dropout3", type=float, default=0.4)
    p.add_argument("--l2", type=float, default=5e-4, help="Kernel L2 regularization.")
    p.add_argument("--weight-decay", type=float, default=2e-4, help="AdamW weight decay.")
    p.add_argument(
        "--label-smoothing",
        type=float,
        default=0.07,
        help="Binary cross-entropy label smoothing (reduces overconfident overfitting).",
    )
    p.add_argument(
        "--input-noise",
        type=float,
        default=0.03,
        help="Gaussian noise std on inputs during training only (0 to disable).",
    )
    p.add_argument(
        "--target-val-acc",
        type=float,
        default=0.90,
        help="With --threshold-metric accuracy: pick threshold with val accuracy closest to this.",
    )
    p.add_argument(
        "--threshold-metric",
        choices=("accuracy", "balanced_accuracy"),
        default="accuracy",
        help="How to pick the decision threshold on the validation set after training.",
    )
    p.add_argument(
        "--no-threshold-calibration",
        action="store_true",
        help="Keep fixed 0.5 decision threshold (no val-set calibration).",
    )
    p.add_argument("--lr", type=float, default=None, help="AdamW LR; default set after optional sweep.")
    p.add_argument(
        "--lr-sweep",
        action="store_true",
        help="Run a short validation sweep over learning rates, then train final model.",
    )
    p.add_argument(
        "--include-leaky",
        action="store_true",
        help="Do not exclude outcome-adjacent columns (old behavior; often looks 'too good' on paper).",
    )
    args = p.parse_args()

    set_seeds(args.seed)

    if not args.csv.is_file():
        raise SystemExit(f"CSV not found: {args.csv}")

    print("Loading CSV...")
    df = pd.read_csv(args.csv, nrows=args.max_rows)
    if TARGET not in df.columns:
        raise SystemExit(f"Missing target column {TARGET!r} in {args.csv}")

    exclude = frozenset() if args.include_leaky else DEFAULT_LEAKY_EXCLUDE
    missing_ex = [c for c in exclude if c not in df.columns]
    if missing_ex:
        print("Note: excluded columns not in CSV (ignored):", missing_ex)
    exclude = frozenset(c for c in exclude if c in df.columns)

    feat_cols = feature_columns_from_df(df, exclude)
    print("Excluded from X (leakage / overfit guard):", sorted(exclude) if exclude else "(none)")
    for c in feat_cols + [TARGET]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    before = len(df)
    df = df.dropna(subset=feat_cols + [TARGET]).reset_index(drop=True)
    print(f"Rows: {before} -> {len(df)} after NA drop on features+target")

    X = df[feat_cols].astype(np.float32).to_numpy()
    y = df[TARGET].astype(np.float32).to_numpy().reshape(-1, 1)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=args.test_size + args.val_size, random_state=args.seed, stratify=y
    )
    rel_val = args.val_size / (args.test_size + args.val_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1.0 - rel_val, random_state=args.seed, stratify=y_temp
    )

    def _split_balance(name: str, y_arr: np.ndarray) -> None:
        yv = y_arr.astype(int).reshape(-1)
        n = int(yv.size)
        n0 = int((yv == 0).sum())
        n1 = int((yv == 1).sum())
        print(
            f"{name}: n={n}  Fail(0)={n0} ({n0 / max(n, 1):.4f})  Pass(1)={n1} ({n1 / max(n, 1):.4f})"
            f"  ratio Pass:Fail = {n1 / max(n0, 1):.3f}:1"
        )

    print("Class balance (pass_fail):")
    _split_balance("  train", y_train)
    _split_balance("  val  ", y_val)
    _split_balance("  test ", y_test)

    scaler = MinMaxScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    medians = pd.Series(np.median(X_train, axis=0), index=feat_cols)
    impute_values = {c: float(medians[c]) for c in feat_cols}

    input_dim = X_train_s.shape[1]
    lr = args.lr
    build_kw = {
        "l2": args.l2,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "input_noise_std": args.input_noise,
    }
    if args.lr_sweep or lr is None:
        sweep_lrs = [1e-3, 3e-4, 1e-4]
        print("Learning-rate sweep (subset speed)...")
        lr = quick_lr_sweep(
            X_train_s,
            y_train,
            X_val_s,
            y_val,
            input_dim,
            sweep_lrs,
            args.h1,
            args.h2,
            args.h3,
            args.dropout1,
            args.dropout2,
            args.dropout3,
            build_kw,
            epochs=4,
            batch_size=4096,
        )
    else:
        print(f"Using fixed learning rate: {lr}")

    model = build_model(
        input_dim,
        args.h1,
        args.h2,
        args.h3,
        args.dropout1,
        args.dropout2,
        args.dropout3,
        lr,
        **build_kw,
    )
    model.summary()

    cb = [
        EarlyStopping(
            monitor="val_loss",
            patience=4,
            min_delta=1e-4,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1),
    ]

    y_int = y_train.reshape(-1).astype(int)
    _cls = np.unique(y_int)
    _cw = compute_class_weight(class_weight="balanced", classes=_cls, y=y_int)
    class_weight = {int(c): float(w) for c, w in zip(_cls, _cw)}

    print("Training...")
    model.fit(
        X_train_s,
        y_train,
        validation_data=(X_val_s, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=cb,
        class_weight=class_weight,
        verbose=1,
    )

    val_proba = model.predict(X_val_s, verbose=0).reshape(-1)
    if args.no_threshold_calibration:
        decision_threshold = 0.5
        print("Decision threshold: 0.5 (no calibration)")
    else:
        decision_threshold = calibrate_decision_threshold(
            y_val.reshape(-1),
            val_proba,
            metric=args.threshold_metric,
            target_acc=args.target_val_acc,
        )
        val_pred = (val_proba >= decision_threshold).astype(int)
        y_val_flat = y_val.reshape(-1).astype(int)
        val_acc_at_t = float((val_pred == y_val_flat).mean())
        val_ba = float(balanced_accuracy_score(y_val_flat, val_pred))
        if args.threshold_metric == "accuracy":
            print(
                f"Calibrated decision_threshold={decision_threshold:.4f} "
                f"(val accuracy {val_acc_at_t:.4f} target {args.target_val_acc:.2f}; "
                f"val balanced_accuracy {val_ba:.4f})"
            )
        else:
            print(
                f"Calibrated decision_threshold={decision_threshold:.4f} "
                f"(val balanced_accuracy {val_ba:.4f}; val accuracy {val_acc_at_t:.4f})"
            )

    proba = model.predict(X_test_s, verbose=0).reshape(-1)
    y_hat = (proba >= decision_threshold).astype(int)
    y_true = y_test.reshape(-1).astype(int)
    print("\n=== Test metrics ===")
    print("Decision threshold:", round(decision_threshold, 4))
    print("Accuracy:", round(accuracy_score(y_true, y_hat), 4))
    print("Balanced accuracy:", round(balanced_accuracy_score(y_true, y_hat), 4))
    print("MCC:", round(matthews_corrcoef(y_true, y_hat), 4))
    try:
        print("ROC-AUC:", round(roc_auc_score(y_true, proba), 4))
    except ValueError as e:
        print("ROC-AUC:", str(e))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_hat))
    print(classification_report(y_true, y_hat, digits=4))

    model_path = ROOT / "model.h5"
    scaler_path = ROOT / "scaler.pkl"
    model.save(model_path)

    artifact = {
        "scaler": scaler,
        "feature_columns": feat_cols,
        "features_for_model": list(feat_cols),
        "performance_index_weights": {},
        "performance_index_mins": {},
        "performance_index_maxs": {},
        "impute_values": impute_values,
        "pipeline_version": 2,
        "decision_threshold": float(decision_threshold),
        "threshold_metric": (
            None if args.no_threshold_calibration else str(args.threshold_metric)
        ),
        "training_config": {
            "csv": str(args.csv),
            "n_rows": int(len(df)),
            "h1": args.h1,
            "h2": args.h2,
            "h3": args.h3,
            "dropout": [args.dropout1, args.dropout2, args.dropout3],
            "l2": args.l2,
            "weight_decay": args.weight_decay,
            "label_smoothing": args.label_smoothing,
            "excluded_leaky_columns": sorted(exclude),
            "include_leaky": bool(args.include_leaky),
            "lr_final": lr,
            "lr_sweep": bool(args.lr_sweep),
            "target": TARGET,
            "decision_threshold": float(decision_threshold),
            "target_val_acc": float(args.target_val_acc),
            "threshold_metric": str(args.threshold_metric),
            "threshold_calibration_disabled": bool(args.no_threshold_calibration),
        },
    }
    joblib.dump(artifact, scaler_path)

    hist_path = ROOT / "training_hist.json"
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(artifact["training_config"], f, indent=2)

    print(f"\nWrote {model_path}")
    print(f"Wrote {scaler_path} (includes scaler + feature metadata + impute medians)")
    print(f"Wrote {hist_path}")


if __name__ == "__main__":
    main()
