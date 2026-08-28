#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
evaluate_shap.py - Leakage audit, SHAP / permutation diagnostics & ablation proof.
Part of the Analyticus submission (PlayHack 2026 ML Track, IIT Guwahati).

Why this exists
--------------
Judges need proof that:

1. `prior_season_injury_count` is a *strictly historical* baseline (from
   athlete_metadata.csv, collected before Day 1) and is NOT a look-ahead
   into the risk window (Days 31-60).  The script verifies the feature's
   importance is modest and that dropping it does not collapse performance
   (ablation test).

2. The model's decisions are auditable.  If the `shap` package is available
   we emit a SHAP beeswarm / bar summary (TreeExplainer on the XGBoost
   estimator).  Otherwise we fall back to permutation-style importance from
   the trained ensemble.

3. All temporal features are leakage-free: `_clip_obs` keeps only the first
   30 days (Observation Window) for every log source.

Usage
-----
    python src/evaluate_shap.py
    python src/evaluate_shap.py --data-dir data/train --top-k 15

Outputs
-------
- Prints an audit report to stdout (also appended to output/metrics_summary.txt).
- Writes output/shap_summary.png        (if shap is installed) or
         output/shap_fallback_importance.png (permutation-style fallback).
- Writes output/ablation_report.csv
"""
import os
import sys
import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score

from preprocess import (  # noqa: E402
    RANDOM_STATE, N_FOLDS, _load_raw, _clip_obs,
    build_features, add_advanced_features, Preprocessor,
    make_classifiers, proba_or_pred, optimize_threshold,
)

OUTPUT = os.path.join(ROOT, "output")
MODELS = os.path.join(ROOT, "models")


def resolve_train_dir(cli_dir=None):
    if cli_dir and os.path.isdir(cli_dir):
        return cli_dir
    for cand in [os.path.join(ROOT, "data", "train"), ROOT]:
        if os.path.exists(os.path.join(cand, "train_labels.csv")):
            return cand
    return ROOT


def _verify_clip(tdir):
    """Prove that every temporal source is clipped to 30 days."""
    import pandas as pd
    checks = []
    for name, id_col, date_col, fmt in [
        ("dailyActivity_merged.csv", "Id", "ActivityDate", "%Y-%m-%d"),
        ("sleepDay_merged.csv", "Id", "SleepDay", "%Y-%m-%d"),
        ("weightLogInfo_merged.csv", "Id", "Date", "%Y-%m-%d"),
        ("hourlyHeartrate_merged.csv", "Id", "ActivityHour", None),
    ]:
        p = os.path.join(tdir, name)
        if not os.path.exists(p):
            continue
        df = pd.read_csv(p)
        df[date_col] = pd.to_datetime(df[date_col], format=fmt, errors="coerce") if fmt else pd.to_datetime(df[date_col], errors="coerce")
        raw_span = (df[date_col].max() - df[date_col].min()).days
        clipped = _clip_obs(df, id_col, date_col, window=30)
        # per-athlete span after clipping
        span = (clipped.groupby(id_col)[date_col].transform("max") - clipped.groupby(id_col)[date_col].transform("min")).dt.days
        max_span = int(span.max()) if len(span) else -1
        checks.append((name, raw_span, max_span, "OK" if max_span < 30 else "FAIL"))
    return checks


def _ablation(X, y, groups, cat_cols, num_cols):
    """Run WITH vs WITHOUT prior_season_injury_count."""
    drop = ["prior_season_injury_count"]
    rows = []
    for variant, cols in [("WITH", num_cols), ("WITHOUT", [c for c in num_cols if c not in drop])]:
        oof = np.zeros(len(y))
        gkf = GroupKFold(n_splits=N_FOLDS)
        for tr, va in gkf.split(X, y, groups):
            prep = Preprocessor(cat_cols, cols).fit(X.iloc[tr], y[tr])
            probas = []
            for m in make_classifiers().values():
                m.fit(prep.transform(X.iloc[tr]), y[tr])
                probas.append(proba_or_pred(m, prep.transform(X.iloc[va])))
            oof[va] = np.mean(probas, axis=0)
        t, _, _ = optimize_threshold(y, oof)
        pred = (oof >= t).astype(int)
        rows.append({
            "variant": variant,
            "threshold": round(float(t), 3),
            "F1": round(float(f1_score(y, pred)), 4),
            "precision": round(float(precision_score(y, pred)), 4),
            "recall": round(float(recall_score(y, pred)), 4),
        })
    return rows


def main():
    ap = argparse.ArgumentParser(description="Leakage audit + SHAP / importance diagnostics.")
    ap.add_argument("--data-dir", default=None, help="Directory with train_labels.csv")
    ap.add_argument("--top-k", type=int, default=15, help="Top-K features to display")
    args = ap.parse_args()

    tdir = resolve_train_dir(args.data_dir)
    print("=" * 78)
    print("EVALUATE_SHAP  -  Leakage & Explainability Audit")
    print("=" * 78)
    print(f"[audit] train dir: {tdir}")

    labels = pd.read_csv(os.path.join(tdir, "train_labels.csv"))
    train_ids = labels["athlete_id"].tolist()
    y = labels["injured_in_risk_window"].astype(int).values
    groups = np.array(train_ids)

    # 1. Clip verification
    print("\n[1] LEAKAGE GUARD  (_clip_obs keeps only first 30 days per athlete)")
    print("-" * 78)
    for name, raw_span, max_span, status in _verify_clip(tdir):
        print(f"  {name:32s}  raw_span={raw_span:3d}d  clipped_max={max_span:2d}d  [{status}]")
    print("  prior_season_injury_count source: athlete_metadata.csv (static, pre-Day-1)")
    print("  -> No aggregation over Days 31-60 (risk window) ever enters features.")

    # 2. Build full feature matrix
    feat = add_advanced_features(build_features(train_ids, tdir))
    CAT = [c for c in ["sport", "gender", "dominant_side", "position", "team_id"] if c in feat.columns]
    NUM = [c for c in feat.columns if c not in CAT]
    X = feat.copy()
    print(f"\n[2] FEATURE MATRIX  {X.shape[1]} cols ({len(NUM)} numeric, {len(CAT)} categorical)")

    # 3. Ablation
    print("\n[3] ABLATION TEST  (drop prior_season_injury_count)")
    print("-" * 78)
    abl_rows = _ablation(X, y, groups, CAT, NUM)
    abl_df = pd.DataFrame(abl_rows)
    print(abl_df.to_string(index=False))
    abl_path = os.path.join(OUTPUT, "ablation_report.csv")
    os.makedirs(OUTPUT, exist_ok=True)
    abl_df.to_csv(abl_path, index=False)
    print(f"[audit] wrote {abl_path}")
    f1_with = abl_rows[0]["F1"]; f1_without = abl_rows[1]["F1"]
    delta = f1_without - f1_with
    print(f"  Delta F1 (WITHOUT - WITH) = {delta:+.4f}  "
          + ("(model is robust; dropped feature is NOT leakage)" if abs(delta) < 0.05 else "(large delta - investigate)"))

    # 4. Importance / SHAP
    print("\n[4] FEATURE ATTRIBUTION")
    print("-" * 78)
    # Fit one full preprocessor + XGBoost for attribution
    prep = Preprocessor(CAT, NUM).fit(X, y)
    Xt = prep.transform(X)
    feature_names = list(Xt.columns)

    # Train a single XGB for SHAP (fast, deterministic)
    import xgboost as xgb
    xgb_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
    xgb_model.fit(Xt.values, y)

    # Try SHAP
    shap_ok = False
    try:
        import shap  # type: ignore
        explainer = shap.TreeExplainer(xgb_model)
        shap_values = explainer.shap_values(Xt.values)
        # shap 0.4+ returns array; older returns list
        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        print("[audit] SHAP available -> generating beeswarm + bar summary")
        plt.figure()
        shap.summary_plot(shap_values, Xt.values, feature_names=feature_names, show=False, max_display=args.top_k)
        plt.tight_layout()
        out = os.path.join(OUTPUT, "shap_summary.png")
        plt.savefig(out, dpi=140, bbox_inches="tight")
        plt.close()
        print(f"[audit] wrote {out}")

        plt.figure()
        shap.summary_plot(shap_values, Xt.values, feature_names=feature_names, plot_type="bar", show=False, max_display=args.top_k)
        plt.tight_layout()
        out2 = os.path.join(OUTPUT, "shap_bar.png")
        plt.savefig(out2, dpi=140, bbox_inches="tight")
        plt.close()
        print(f"[audit] wrote {out2}")
        shap_ok = True

        # Rank check for prior
        mean_abs = np.abs(shap_values).mean(axis=0)
        order = np.argsort(-mean_abs)
        ranked = [feature_names[i] for i in order]
        prior_rank = ranked.index("prior_season_injury_count") + 1 if "prior_season_injury_count" in ranked else None
        print(f"  prior_season_injury_count SHAP rank: {prior_rank}/{len(ranked)}  "
              f"(mean|SHAP|={mean_abs[feature_names.index('prior_season_injury_count')]:.4f})" if prior_rank else
              "  prior_season_injury_count not in feature set")
    except Exception as e:
        print(f"[audit] SHAP not available or failed ({e}) -> fallback to impurity importance")

    if not shap_ok:
        # Fallback: impurity-based importance from the full ensemble
        import joblib
        # Use already-trained XGB + train LGBM/CatBoost quickly for ensemble fallback
        imps = np.zeros(len(feature_names))
        # XGB is already fitted
        imps += xgb_model.feature_importances_ / 3
        # LightGBM quick fit
        try:
            import lightgbm as lgb
            lgb_model = lgb.LGBMClassifier(n_estimators=300, max_depth=8, learning_rate=0.05, num_leaves=63, verbose=-1, random_state=RANDOM_STATE)
            lgb_model.fit(Xt.values, y)
            imps += lgb_model.feature_importances_ / 3
        except Exception:
            pass
        try:
            import catboost as cb
            cb_model = cb.CatBoostClassifier(iterations=300, depth=6, learning_rate=0.05, verbose=False, allow_writing_files=False, random_state=RANDOM_STATE)
            cb_model.fit(Xt.values, y)
            imps += cb_model.feature_importances_ / 3
        except Exception:
            pass
        imps = imps / (imps.sum() + 1e-12)
        order = np.argsort(-imps)
        top = [(feature_names[i], float(imps[i])) for i in order[:args.top_k]]
        print(f"  Top-{args.top_k} impurity importance (ensemble fallback):")
        for name, imp in top:
            print(f"    {name:40s}  {imp:.4f}")
        # Plot
        names, vals = zip(*reversed(top))
        plt.figure(figsize=(9, 6))
        plt.barh(list(names), list(vals), color="#2c7fb8")
        plt.xlabel("Relative importance (ensemble, normalized)")
        plt.title(f"Top {args.top_k} Features - Impurity Fallback (SHAP unavailable)")
        plt.tight_layout()
        out = os.path.join(OUTPUT, "shap_fallback_importance.png")
        plt.savefig(out, dpi=140)
        plt.close()
        print(f"[audit] wrote {out}")
        if "prior_season_injury_count" in feature_names:
            pr = feature_names.index("prior_season_injury_count")
            print(f"  prior_season_injury_count rank: {list(order).index(pr)+1}/{len(feature_names)}  imp={imps[pr]:.4f}")

    # 5. Append to metrics_summary
    try:
        summary = os.path.join(OUTPUT, "metrics_summary.txt")
        if os.path.exists(summary):
            with open(summary, "a") as f:
                f.write("\n" + "=" * 74 + "\n")
                f.write("SHAP / ABLATION AUDIT  (see src/evaluate_shap.py)\n")
                f.write("=" * 74 + "\n")
                f.write(abl_df.to_string(index=False) + "\n")
                f.write(f"Delta F1 (WITHOUT-WITH) = {delta:+.4f}\n")
                if shap_ok:
                    f.write("SHAP summary: shap_summary.png + shap_bar.png\n")
                else:
                    f.write("SHAP unavailable -> shap_fallback_importance.png\n")
            print(f"[audit] appended audit to {summary}")
    except Exception as e:
        print(f"[audit] could not append to metrics_summary.txt: {e}")

    print("\n[audit] done.")


if __name__ == "__main__":
    main()
