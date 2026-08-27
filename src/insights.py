#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
insights.py - Fresh, presentation-grade visualizations (PlayHack 2026 ML Track).
Generates NEW test-prediction insights on top of the saved models:
  output/test_risk_distribution.png   - predicted injury-risk distribution
  output/injury_strategy_compare.png  - model vs recall-boosted vs train prior
  output/test_risk_by_sport.png       - predicted injury rate by sport (test)
  output/test_feature_separation.png  - feature separation: injured vs healthy
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

warnings.filterwarnings("ignore")
SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)
from preprocess import build_features, add_advanced_features, proba_or_pred, _load_raw  # noqa

MODELS = os.path.join(ROOT, "models")
OUTPUT = os.path.join(ROOT, "output")
sns.set_style("whitegrid")


def main():
    clf = joblib.load(os.path.join(MODELS, "ensemble_clf.bin"))
    prep, best_t = clf["prep"], clf["best_t"]
    clf_models = list(clf["models"].values())

    test_ids = pd.read_csv(os.path.join(ROOT, "example.csv"))["athlete_id"].tolist()
    # locate test features
    d = os.path.join(ROOT, "data", "test")
    tdir = d if (os.path.isdir(d) and os.path.exists(os.path.join(d, "athlete_metadata.csv"))) else os.path.join(ROOT, "test_data")
    feat = add_advanced_features(build_features(test_ids, tdir))
    pc = np.mean([proba_or_pred(m, prep.transform(feat)) for m in clf_models], axis=0)

    # two strategies
    model_inj = (pc >= best_t).astype(int)
    k = int(round(0.35 * len(pc)))
    order = np.argsort(-pc)
    recall_inj = np.zeros(len(pc), dtype=int); recall_inj[order[:k]] = 1

    meta = pd.read_csv(os.path.join(tdir, "athlete_metadata.csv"))
    df = pd.DataFrame({"athlete_id": test_ids, "p": pc,
                       "model_inj": model_inj, "recall_inj": recall_inj}).merge(meta, on="athlete_id", how="left")
    feat_df = feat.reset_index().rename(columns={"index": "athlete_id"})

    # ---- 1. risk distribution ----
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(pc, bins=40, color="#2c7fb8", alpha=0.85, edgecolor="white")
    ax.axvline(best_t, color="#d7301f", lw=2, ls="--", label=f"model threshold ({best_t:.2f})")
    ax.axvline(np.quantile(pc, 0.65), color="#238b45", lw=2, ls="--",
               label="recall-boost cutoff (top 35%)")
    ax.set_title("Test Athletes: Predicted Injury-Risk Probability")
    ax.set_xlabel("Model probability of injury"); ax.set_ylabel("Number of athletes")
    ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUTPUT, "test_risk_distribution.png"), dpi=140); plt.close()

    # ---- 2. strategy comparison ----
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["Train prior\n(35%)", "Model-threshold\n(13.3%)", "Recall-boosted\n(35.0%)"]
    vals = [0.35 * len(pc), model_inj.sum(), recall_inj.sum()]
    bars = ax.bar(labels, vals, color=["#999999", "#d7301f", "#238b45"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 5, str(int(v)), ha="center", fontweight="bold")
    ax.set_title("Injury Predictions: Strategy Comparison (n=1100)")
    ax.set_ylabel("Predicted injured athletes")
    fig.tight_layout(); fig.savefig(os.path.join(OUTPUT, "injury_strategy_compare.png"), dpi=140); plt.close()

    # ---- 3. by sport (model-based) ----
    g = df.groupby("sport")["model_inj"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=g.values, y=g.index, ax=ax, palette="Reds_r")
    for i, v in enumerate(g.values):
        ax.text(v + 0.005, i, f"{v:.2f}", va="center", fontweight="bold")
    ax.set_title("Predicted Injury Rate by Sport (Test Set, model-threshold)")
    ax.set_xlabel("Predicted injury rate")
    fig.tight_layout(); fig.savefig(os.path.join(OUTPUT, "test_risk_by_sport.png"), dpi=140); plt.close()

    # ---- 4. feature separation ----
    if "daily_SedentaryMinutes_mean" in feat_df.columns:
        m = feat_df.merge(df[["athlete_id", "model_inj"]], on="athlete_id", how="left")
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.boxplot(data=m, x="model_inj", y="daily_SedentaryMinutes_mean", ax=ax,
                    palette=["#41ab5d", "#d7301f"])
        ax.set_title("Sedentary Minutes: Predicted Injured vs Healthy (Test)")
        ax.set_xticklabels(["Healthy (0)", "Injured (1)"]); ax.set_xlabel("")
        fig.tight_layout(); fig.savefig(os.path.join(OUTPUT, "test_feature_separation.png"), dpi=140); plt.close()

    print("[insights] wrote: test_risk_distribution.png, injury_strategy_compare.png,")
    print("           test_risk_by_sport.png, test_feature_separation.png")


if __name__ == "__main__":
    main()
