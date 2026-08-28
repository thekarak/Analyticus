#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
optimize_threshold.py - Penalty-aware, fold-level dynamic threshold search.
Part of the Analyticus submission (PlayHack 2026 ML Track, IIT Guwahati).

Why this exists
--------------
The competition penalises a false negative (predicting healthy when the
athlete is injured) with +30 days on *both* onset and recovery MAE.  A
single global threshold tuned on the pooled OOF can overfit the validation
distribution.  This script proves production-level rigour by:

1. Running 5-fold GroupKFold (grouped by athlete_id).
2. For each fold, searching t in 0.05..0.50 (step 0.01) for the threshold
   that minimises  cost = 5*FN + FP  subject to recall >= 0.90.
3. Averaging the 5 fold thresholds -> final threshold.
4. Reporting per-fold and pooled F1 / Precision / Recall.

Usage
-----
    python src/optimize_threshold.py               # uses train data auto-discovered
    python src/optimize_threshold.py --data-dir data/train

Output
------
- Prints a fold table + pooled report to stdout.
- Writes output/threshold_search.csv  (t, F1, recall, FN, FP, cost per candidate).
"""
import os
import sys
import argparse

import numpy as np
import pandas as pd

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)

from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from preprocess import (  # noqa: E402
    N_FOLDS, RECALL_FLOOR, build_features, add_advanced_features,
    Preprocessor, make_classifiers, proba_or_pred, optimize_threshold,
)

OUTPUT = os.path.join(ROOT, "output")


def resolve_train_dir(cli_dir=None):
    if cli_dir and os.path.isdir(cli_dir):
        return cli_dir
    for cand in [os.path.join(ROOT, "data", "train"), ROOT]:
        if os.path.exists(os.path.join(cand, "train_labels.csv")):
            return cand
    return ROOT


def _grid_report(y_true, proba, out_csv=None):
    rows = []
    for t in np.arange(0.05, 0.50 + 1e-9, 0.01):
        pred = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        rows.append({
            "threshold": round(float(t), 2),
            "F1": round(float(f1_score(y_true, pred)), 4),
            "precision": round(float(precision_score(y_true, pred)), 4),
            "recall": round(float(recall_score(y_true, pred)), 4),
            "FN": int(fn), "FP": int(fp),
            "cost_5FN_FP": int(5 * fn + fp),
        })
    df = pd.DataFrame(rows)
    if out_csv:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"[threshold] wrote grid report -> {out_csv}")
    return df


def main():
    ap = argparse.ArgumentParser(description="Fold-level penalty-aware threshold search.")
    ap.add_argument("--data-dir", default=None, help="Directory with train_labels.csv")
    args = ap.parse_args()

    tdir = resolve_train_dir(args.data_dir)
    print(f"[threshold] train dir: {tdir}")

    labels = pd.read_csv(os.path.join(tdir, "train_labels.csv"))
    train_ids = labels["athlete_id"].tolist()
    y = labels["injured_in_risk_window"].astype(int).values
    groups = np.array(train_ids)

    print(f"[threshold] athletes={len(train_ids)}  injured={int(y.sum())} ({y.mean():.3f})")
    print("[threshold] building features ...")
    feat = add_advanced_features(build_features(train_ids, tdir))
    CAT = [c for c in ["sport", "gender", "dominant_side", "position", "team_id"] if c in feat.columns]
    NUM = [c for c in feat.columns if c not in CAT]
    X = feat.copy()

    gkf = GroupKFold(n_splits=N_FOLDS)
    fold_thresholds, fold_reports = [], []
    oof = np.zeros(len(y))

    for fold, (tr, va) in enumerate(gkf.split(X, y, groups)):
        prep = Preprocessor(CAT, NUM).fit(X.iloc[tr], y[tr])
        Xtr, Xva = prep.transform(X.iloc[tr]), prep.transform(X.iloc[va])
        probas = []
        for m in make_classifiers().values():
            m.fit(Xtr, y[tr])
            probas.append(proba_or_pred(m, Xva))
        proba = np.mean(probas, axis=0)
        oof[va] = proba
        t, f1, rec = optimize_threshold(y[va], proba)
        fold_thresholds.append(t)
        tn, fp, fn, tp = confusion_matrix(y[va], (proba >= t).astype(int), labels=[0, 1]).ravel()
        fold_reports.append({
            "fold": fold + 1, "threshold": round(t, 3),
            "F1": round(float(f1), 4), "recall": round(float(rec), 4),
            "FN": int(fn), "FP": int(fp),
        })
        print(f"  fold {fold+1}: t={t:.2f}  F1={f1:.4f}  recall={rec:.4f}  FN={fn} FP={fp}")

    pooled_t = float(np.mean(fold_thresholds))
    pooled_pred = (oof >= pooled_t).astype(int)
    pooled_f1 = f1_score(y, pooled_pred)
    pooled_rec = recall_score(y, pooled_pred)
    tn, fp, fn, tp = confusion_matrix(y, pooled_pred, labels=[0, 1]).ravel()

    print("\n" + "=" * 74)
    print("THRESHOLD SEARCH REPORT  (penalty-aware, per-fold averaged)")
    print("=" * 74)
    print(pd.DataFrame(fold_reports).to_string(index=False))
    print(f"\nAveraged threshold : {pooled_t:.3f}  (mean of {len(fold_thresholds)} folds)")
    print(f"Pooled OOF (t={pooled_t:.3f}) : F1={pooled_f1:.4f}  recall={pooled_rec:.4f}  FN={fn} FP={fp}")
    print(f"Recall floor       : {RECALL_FLOOR}  |  cost = 5*FN + FP  |  grid 0.05..0.50 step 0.01")
    print("=" * 74)

    # full grid report on pooled OOF for the appendix
    _grid_report(y, oof, out_csv=os.path.join(OUTPUT, "threshold_search.csv"))


if __name__ == "__main__":
    main()
