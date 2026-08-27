#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train.py - Cross-validation, dual-stage ensemble training & artifact export.
Part of the Analyticus submission (PlayHack 2026 ML Track, IIT Guwahati).

Run:  python src/train.py
Produces: trained models (models/*.bin), evaluation charts (output/*.png),
          metrics_summary.txt, and submission_final.csv (output/).
"""
import os
import sys
import time
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    precision_score, accuracy_score, mean_absolute_error, r2_score, confusion_matrix,
)
import joblib

# import the shared feature / preprocessing / modeling logic
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from preprocess import (  # noqa: E402
    RANDOM_STATE, N_FOLDS, RECALL_FLOOR,
    _load_raw, build_features, add_advanced_features, Preprocessor,
    make_classifiers, make_regressors, proba_or_pred, optimize_threshold,
)

# --------------------------------------------------------------------------- #
# Path resolution (clean-machine friendly)
# --------------------------------------------------------------------------- #
SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
MODELS = os.path.join(ROOT, "models")
OUTPUT = os.path.join(ROOT, "output")
EDA_OUT = os.path.join(OUTPUT, "eda_plots")
os.makedirs(MODELS, exist_ok=True)
os.makedirs(OUTPUT, exist_ok=True)
os.makedirs(EDA_OUT, exist_ok=True)
SUB_TEMPLATE = os.path.join(ROOT, "example.csv")


def resolve_train_dir():
    d = os.path.join(ROOT, "data", "train")
    if os.path.isdir(d) and os.path.exists(os.path.join(d, "train_labels.csv")):
        return d
    return ROOT


def resolve_test_dir(test_ids):
    d = os.path.join(ROOT, "data", "test")
    if os.path.isdir(d) and os.path.exists(os.path.join(d, "athlete_metadata.csv")):
        return d
    d2 = os.path.join(ROOT, "test_data")
    if os.path.isdir(d2):
        return d2
    probe = _load_raw(ROOT)["meta"]
    if probe is not None and probe["athlete_id"].isin(test_ids).any():
        return ROOT
    return None


# --------------------------------------------------------------------------- #
# EDA (presentation insights)
# --------------------------------------------------------------------------- #
def run_eda():
    sns.set_style("whitegrid")
    labels = pd.read_csv(os.path.join(resolve_train_dir(), "train_labels.csv"))
    meta = pd.read_csv(os.path.join(resolve_train_dir(), "athlete_metadata.csv"))
    daily = pd.read_csv(os.path.join(resolve_train_dir(), "dailyActivity_merged.csv"))
    sleep = pd.read_csv(os.path.join(resolve_train_dir(), "sleepDay_merged.csv"))
    hr = pd.read_csv(os.path.join(resolve_train_dir(), "hourlyHeartrate_merged.csv"))
    weight = pd.read_csv(os.path.join(resolve_train_dir(), "weightLogInfo_merged.csv"))

    df = labels.merge(meta, on="athlete_id", how="left")
    da = daily.groupby("Id").agg(steps=("TotalSteps", "mean"),
                                 active=("VeryActiveMinutes", "mean"),
                                 sedentary=("SedentaryMinutes", "mean"),
                                 cal=("Calories", "mean")).reset_index().rename(columns={"Id": "athlete_id"})
    df = df.merge(da, on="athlete_id", how="left")
    sl = sleep.copy(); sl["eff"] = sl["TotalMinutesAsleep"] / sl["TotalTimeInBed"]
    sa = sl.groupby("Id").agg(sleep_min=("TotalMinutesAsleep", "mean"),
                              sleep_eff=("eff", "mean")).reset_index().rename(columns={"Id": "athlete_id"})
    df = df.merge(sa, on="athlete_id", how="left")
    ha = hr.groupby("Id").agg(rest_hr=("MinHeartRate", "mean"),
                              peak_hr=("MaxHeartRate", "mean")).reset_index().rename(columns={"Id": "athlete_id"})
    df = df.merge(ha, on="athlete_id", how="left")
    inj = df["injured_in_risk_window"] == 1

    def save(fig, name):
        fig.savefig(os.path.join(EDA_OUT, name), dpi=140, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    vc = df["injured_in_risk_window"].value_counts().sort_index()
    ax.bar(["Healthy (0)", "Injured (1)"], vc.values, color=["#41ab5d", "#d7301f"])
    for i, v in enumerate(vc.values):
        ax.text(i, v + 20, str(v), ha="center", fontweight="bold")
    ax.set_title("Injury Prevalence (Risk Window)"); ax.set_ylabel("Athletes")
    save(fig, "01_injury_prevalence.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    g = df.groupby("sport")["injured_in_risk_window"].mean().sort_values(ascending=False)
    sns.barplot(x=g.values, y=g.index, ax=ax, palette="Reds_r")
    for i, v in enumerate(g.values):
        ax.text(v + 0.005, i, f"{v:.2f}", va="center", fontweight="bold")
    ax.set_title("Injury Rate by Sport"); ax.set_xlabel("Injury rate")
    save(fig, "02_injury_by_sport.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(data=df, x="injured_in_risk_window", y="steps", ax=axes[0], palette=["#41ab5d", "#d7301f"])
    sns.boxplot(data=df, x="injured_in_risk_window", y="active", ax=axes[1], palette=["#41ab5d", "#d7301f"])
    axes[0].set_title("Mean Daily Steps by Injury Status"); axes[1].set_title("Very Active Minutes by Injury Status")
    for a in axes:
        a.set_xlabel(""); a.set_xticklabels(["Healthy", "Injured"])
    save(fig, "03_activity_by_injury.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.boxplot(data=df, x="injured_in_risk_window", y="sedentary", ax=axes[0], palette=["#41ab5d", "#d7301f"])
    sns.boxplot(data=df, x="injured_in_risk_window", y="sleep_eff", ax=axes[1], palette=["#41ab5d", "#d7301f"])
    axes[0].set_title("Sedentary Minutes by Injury Status"); axes[1].set_title("Sleep Efficiency by Injury Status")
    for a in axes:
        a.set_xlabel(""); a.set_xticklabels(["Healthy", "Injured"])
    save(fig, "04_sedentary_sleep_by_injury.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.histplot(data=df, x="rest_hr", hue="injured_in_risk_window", ax=axes[0], palette=["#41ab5d", "#d7301f"], kde=True, element="step")
    sns.histplot(data=df, x="peak_hr", hue="injured_in_risk_window", ax=axes[1], palette=["#41ab5d", "#d7301f"], kde=True, element="step")
    axes[0].set_title("Resting Heart Rate Distribution"); axes[1].set_title("Peak Heart Rate Distribution")
    save(fig, "05_heartrate_dist.png")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sub = df[inj]
    sns.histplot(sub["onset_day_offset"], ax=axes[0], color="#2c7fb8", kde=True, bins=20)
    sns.histplot(sub["recovery_duration"], ax=axes[1], color="#2c7fb8", kde=True, bins=20)
    axes[0].set_title("Onset Day Offset (injured)"); axes[1].set_title("Recovery Duration (injured)")
    save(fig, "06_onset_recovery_dist.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=df, x="injured_in_risk_window", y="prior_season_injury_count", ax=ax, palette=["#41ab5d", "#d7301f"])
    ax.set_title("Prior-Season Injury Count vs Current Injury"); ax.set_xticklabels(["Healthy", "Injured"]); ax.set_xlabel("")
    save(fig, "07_prior_injury.png")

    key = ["injured_in_risk_window", "steps", "active", "sedentary", "cal", "sleep_min",
           "sleep_eff", "rest_hr", "peak_hr", "age", "years_playing",
           "prior_season_injury_count", "bmi_baseline"]
    key = [c for c in key if c in df.columns]
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(df[key].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Feature Correlation Matrix (key signals)")
    save(fig, "08_correlation_heatmap.png")

    fig, ax = plt.subplots(figsize=(6, 4))
    covered = weight["Id"].nunique()
    ax.bar(["Has weight logs", "Missing"], [covered, df["athlete_id"].nunique() - covered], color=["#41ab5d", "#d7301f"])
    ax.set_title(f"Weight-Log Coverage ({covered}/3000 athletes)"); ax.set_ylabel("Athletes")
    save(fig, "09_data_coverage.png")
    print("[eda] EDA charts written to output/eda_plots/")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _fmt_table(rows):
    if not rows:
        return ""
    cols = list(rows[0].keys())
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    hdr = "  ".join(c.ljust(widths[c]) for c in cols)
    sep = "  ".join("-" * widths[c] for c in cols)
    body = "\n".join("  ".join(str(r[c]).ljust(widths[c]) for c in cols) for r in rows)
    return hdr + "\n" + sep + "\n" + body


# --------------------------------------------------------------------------- #
# MAIN
# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    print("=" * 78)
    print("ANALYTICUS  -  PlayHack 2026 ML Track  |  training pipeline")
    print("=" * 78)

    run_eda()

    tdir = resolve_train_dir()
    labels = pd.read_csv(os.path.join(tdir, "train_labels.csv"))
    train_ids = labels["athlete_id"].tolist()
    y_clf = labels["injured_in_risk_window"].astype(int).values
    y_onset = labels["onset_day_offset"].astype(float).values
    y_recovery = labels["recovery_duration"].astype(float).values
    groups = np.array(train_ids)
    print(f"[info] train athletes = {len(train_ids)} | injured = {int(y_clf.sum())} ({y_clf.mean():.3f})")

    print("[step1] aggregating spatio-temporal features (train) ...")
    feat = add_advanced_features(build_features(train_ids, tdir))
    CAT_COLS = [c for c in ["sport", "gender", "dominant_side", "position", "team_id"] if c in feat.columns]
    NUM_COLS = [c for c in feat.columns if c not in CAT_COLS]
    print(f"[step1] feature matrix = {feat.shape[1]} columns ({len(NUM_COLS)} numeric, {len(CAT_COLS)} categorical)")
    X = feat.copy()

    from sklearn.model_selection import GroupKFold
    gkf = GroupKFold(n_splits=N_FOLDS)
    n = len(X)
    oof = {k: np.zeros(n) for k in
           ["xgb", "lgbm", "cb", "ens", "onset_xgb", "onset_lgbm", "onset_cb", "onset_ens",
            "rec_xgb", "rec_lgbm", "rec_cb", "rec_ens"]}
    fold_models = {"clf": defaultdict(list), "onset": defaultdict(list), "rec": defaultdict(list)}
    fold_preps = []

    for fold, (tr, va) in enumerate(gkf.split(X, y_clf, groups)):
        print(f"\n[cv] fold {fold + 1}/{N_FOLDS}  train={len(tr)} val={len(va)}")
        prep = Preprocessor(CAT_COLS, NUM_COLS).fit(X.iloc[tr], y_clf[tr])
        fold_preps.append(prep)
        Xtr, Xva = prep.transform(X.iloc[tr]), prep.transform(X.iloc[va])

        clf = make_classifiers()
        for name, model in clf.items():
            model.fit(Xtr, y_clf[tr])
            oof[name][va] = proba_or_pred(model, Xva)
            fold_models["clf"][name].append(model)

        inj_tr = y_clf[tr] == 1
        Xtr_i = Xtr[inj_tr]; yon_i = y_onset[tr][inj_tr]; yrec_i = y_recovery[tr][inj_tr]
        for name, model in make_regressors().items():
            model.fit(Xtr_i, yon_i); oof["onset_" + name][va] = model.predict(Xva)
            fold_models["onset"][name].append(model)
        for name, model in make_regressors().items():
            model.fit(Xtr_i, yrec_i); oof["rec_" + name][va] = model.predict(Xva)
            fold_models["rec"][name].append(model)

    oof["ens"] = (oof["xgb"] + oof["lgbm"] + oof["cb"]) / 3.0
    oof["onset_ens"] = (oof["onset_xgb"] + oof["onset_lgbm"] + oof["onset_cb"]) / 3.0
    oof["rec_ens"] = (oof["rec_xgb"] + oof["rec_lgbm"] + oof["rec_cb"]) / 3.0

    best_t, best_f1, best_rec = optimize_threshold(y_clf, oof["ens"])
    oof_pred = (oof["ens"] >= best_t).astype(int)
    print(f"\n[step5] optimized threshold = {best_t:.3f}  (F1={best_f1:.4f}, Recall={best_rec:.4f})")

    from sklearn.metrics import f1_score, precision_score, recall_score
    def clf_metrics(proba, name):
        pred = (proba >= best_t).astype(int)
        return {"model": name, "threshold": round(best_t, 3),
                "F1": round(f1_score(y_clf, pred), 4),
                "Precision": round(precision_score(y_clf, pred), 4),
                "Recall": round(recall_score(y_clf, pred), 4),
                "Accuracy": round(accuracy_score(y_clf, pred), 4)}

    def reg_metrics(prefix, name):
        m = y_clf == 1
        return {"model": name,
                "MAE_onset": round(mean_absolute_error(y_onset[m], oof[prefix + "_" + name][m]), 4),
                "MAE_recovery": round(mean_absolute_error(y_recovery[m], oof["rec_" + name][m]), 4),
                "R2_onset": round(r2_score(y_onset[m], oof[prefix + "_" + name][m]), 4),
                "R2_recovery": round(r2_score(y_recovery[m], oof["rec_" + name][m]), 4)}

    clf_rows = [clf_metrics(oof["xgb"], "XGBoost"), clf_metrics(oof["lgbm"], "LightGBM"),
                clf_metrics(oof["cb"], "CatBoost"), clf_metrics(oof["ens"], "ENSEMBLE")]
    reg_rows = [reg_metrics("onset", "xgb"), reg_metrics("onset", "lgbm"), reg_metrics("onset", "cb"),
                {"model": "ENSEMBLE",
                 "MAE_onset": round(mean_absolute_error(y_onset[y_clf == 1], oof["onset_ens"][y_clf == 1]), 4),
                 "MAE_recovery": round(mean_absolute_error(y_recovery[y_clf == 1], oof["rec_ens"][y_clf == 1]), 4),
                 "R2_onset": round(r2_score(y_onset[y_clf == 1], oof["onset_ens"][y_clf == 1]), 4),
                 "R2_recovery": round(r2_score(y_recovery[y_clf == 1], oof["rec_ens"][y_clf == 1]), 4)}]

    base_onset = np.mean(y_onset[y_clf == 1]); base_recovery = np.mean(y_recovery[y_clf == 1])
    mae_on_base = mean_absolute_error(y_onset[y_clf == 1], np.full(y_clf.sum(), base_onset))
    mae_rec_base = mean_absolute_error(y_recovery[y_clf == 1], np.full(y_clf.sum(), base_recovery))
    mae_on_model = mean_absolute_error(y_onset[y_clf == 1], oof["onset_ens"][y_clf == 1])
    mae_rec_model = mean_absolute_error(y_recovery[y_clf == 1], oof["rec_ens"][y_clf == 1])
    skill_onset = max(0.0, 1 - mae_on_model / mae_on_base)
    skill_recovery = max(0.0, 1 - mae_rec_model / mae_rec_base)

    eff_on, eff_rec = [], []
    for i in range(n):
        if y_clf[i] == 1:
            pen = 0 if oof_pred[i] == 1 else 30.0
            eff_on.append(abs(oof["onset_ens"][i] - y_onset[i]) + pen)
            eff_rec.append(abs(oof["rec_ens"][i] - y_recovery[i]) + pen)
    eff_mae_on, eff_mae_rec = np.mean(eff_on), np.mean(eff_rec)

    # ---- STEP 6 : test prediction ---------------------------------------- #
    sub = pd.read_csv(SUB_TEMPLATE)
    test_ids = sub["athlete_id"].tolist()
    test_data_dir = resolve_test_dir(test_ids)
    real_test = False
    if test_data_dir is not None:
        print(f"\n[step6] building test features from: {test_data_dir}")
        test_feat = add_advanced_features(build_features(test_ids, test_data_dir))
        if len(test_feat) == len(test_ids):
            real_test = True
        else:
            print(f"[warn ] only {len(test_feat)}/{len(test_ids)} test athletes have features -> fallback")

    def predict_ensemble(target, Xte):
        types = list(fold_models[target].keys())
        preds = []
        for k in range(N_FOLDS):
            Xk = fold_preps[k].transform(Xte)
            for nm in types:
                m = fold_models[target][nm][k]
                preds.append(proba_or_pred(m, Xk) if target == "clf" else m.predict(Xk))
        return np.mean(preds, axis=0)

    if real_test:
        clf_acc = predict_ensemble("clf", test_feat)
        on_acc = predict_ensemble("onset", test_feat)
        rec_acc = predict_ensemble("rec", test_feat)
        test_injured = (clf_acc >= best_t).astype(int)
        pred_onset, pred_recovery = on_acc, rec_acc
        print(f"[step6] real test predictions: injured={int(test_injured.sum())}/{len(test_ids)}")
    else:
        print("\n[step6] NO TEST FEATURES AVAILABLE -> valid-format fallback submission.")
        print("        (Place test CSVs in data/test/ to generate real predictions.)")
        base_rate = float(y_clf.mean())
        test_injured = (np.full(len(test_ids), base_rate) >= best_t).astype(int)
        pred_onset = np.full(len(test_ids), base_onset)
        pred_recovery = np.full(len(test_ids), base_recovery)

    sub_out = sub.copy()
    sub_out["injured_in_risk_window"] = test_injured
    sub_out["onset_day_offset"] = np.round(pred_onset).astype(int)
    sub_out["recovery_duration"] = np.round(pred_recovery).astype(int)
    sub_out.to_csv(os.path.join(OUTPUT, "submission_final.csv"), index=False)
    assert sub_out.isnull().sum().sum() == 0
    print(f"[step6] wrote output/submission_final.csv  shape={sub_out.shape}  nulls=0")

    if not real_test:
        _, va_demo = next(iter(GroupKFold(n_splits=N_FOLDS).split(X, y_clf, groups)))
        demo_ids = [train_ids[i] for i in va_demo]
        pc = predict_ensemble("clf", X.iloc[va_demo])
        po = predict_ensemble("onset", X.iloc[va_demo])
        pr = predict_ensemble("rec", X.iloc[va_demo])
        pd.DataFrame({"athlete_id": demo_ids,
                      "injured_in_risk_window": (pc >= best_t).astype(int),
                      "onset_day_offset": np.round(po).astype(int),
                      "recovery_duration": np.round(pr).astype(int)}).to_csv(
            os.path.join(OUTPUT, "submission_holdout_demo.csv"), index=False)
        print("[step6] wrote output/submission_holdout_demo.csv (proves real predict path)")

    # ---- STEP 7 : artifacts + model persistence ------------------------- #
    print("\n[step7] exporting artifacts & saving models ...")
    prep_full = Preprocessor(CAT_COLS, NUM_COLS).fit(X, y_clf)
    Xfull = prep_full.transform(X)
    full_models = {"clf": make_classifiers(), "onset": make_regressors(), "rec": make_regressors()}
    for m in full_models["clf"].values():
        m.fit(Xfull, y_clf)
    inj = y_clf == 1
    for m in full_models["onset"].values():
        m.fit(Xfull[inj], y_onset[inj])
    for m in full_models["rec"].values():
        m.fit(Xfull[inj], y_recovery[inj])

    joblib.dump({"prep": prep_full, "models": dict(full_models["clf"]),
                 "best_t": best_t, "base_onset": base_onset, "base_recovery": base_recovery,
                 "cat_cols": CAT_COLS, "num_cols": NUM_COLS, "feature_cols": list(Xfull.columns)},
                os.path.join(MODELS, "ensemble_clf.bin"))
    joblib.dump({"models": dict(full_models["onset"])}, os.path.join(MODELS, "ensemble_onset.bin"))
    joblib.dump({"models": dict(full_models["rec"])}, os.path.join(MODELS, "ensemble_rec.bin"))
    print("[step7] saved models/ensemble_{clf,onset,rec}.bin")

    cols = list(Xfull.columns)
    imp = np.zeros(len(cols))
    for m in full_models["clf"].values():
        imp += np.array(m.feature_importances_) / len(full_models["clf"])
    imp = imp / (imp.sum() + 1e-12)
    imp_df = pd.DataFrame({"feature": cols, "importance": imp}).sort_values("importance", ascending=False).reset_index(drop=True)
    top15 = imp_df.head(15).iloc[::-1]
    plt.figure(figsize=(9, 6))
    plt.barh(top15["feature"], top15["importance"], color="#2c7fb8")
    plt.xlabel("Relative Importance (ensemble, normalized)")
    plt.title("Top 15 Features - Injury Risk Ensemble")
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT, "feature_importance.png"), dpi=140); plt.close()

    cm = confusion_matrix(y_clf, oof_pred)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Healthy(0)", "Injured(1)"]); ax.set_yticklabels(["Healthy(0)", "Injured(1)"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix (thr={best_t:.2f})\nFN={cm[1,0]} (minimized)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT, "confusion_matrix.png"), dpi=140); plt.close()

    m = y_clf == 1
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, truth, pred, title in [(axes[0], y_onset[m], oof["onset_ens"][m], "Onset Day Offset"),
                                    (axes[1], y_recovery[m], oof["rec_ens"][m], "Recovery Duration")]:
        ax.scatter(truth, pred, alpha=0.4, s=18, color="#d95f02")
        lim = [min(truth.min(), pred.min()), max(truth.max(), pred.max())]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set_xlabel("Actual"); ax.set_ylabel("Predicted")
        ax.set_title(f"{title}\nMAE={mean_absolute_error(truth, pred):.2f}")
    plt.tight_layout(); plt.savefig(os.path.join(OUTPUT, "residuals_plot.png"), dpi=140); plt.close()

    lines = ["=" * 74, "ANALYTICUS - PlayHack 2026 ML Track  |  VALIDATION REPORT", "=" * 74,
             f"Train athletes          : {len(train_ids)}",
             f"Injury prevalence       : {y_clf.mean():.4f}",
             f"CV strategy             : GroupKFold (k={N_FOLDS}, grouped by athlete_id)",
             f"Optimized threshold     : {best_t:.3f}  (recall_floor={RECALL_FLOOR})", "",
             "-" * 74, "TASK A - CLASSIFICATION (Injury)  metric: F1-Score", "-" * 74,
             _fmt_table(clf_rows), "",
             "-" * 74, "TASK B - REGRESSION (Onset & Recovery)  metric: MAE (injured only)", "-" * 74,
             _fmt_table(reg_rows), "",
             "-" * 74, "BASELINE & SKILL SCORE", "-" * 74,
             f"Baseline onset MAE      : {mae_on_base:.4f}  (mean={base_onset:.2f})",
             f"Baseline recovery MAE   : {mae_rec_base:.4f}  (mean={base_recovery:.2f})",
             f"Model    onset MAE      : {mae_on_model:.4f}",
             f"Model    recovery MAE   : {mae_rec_model:.4f}",
             f"Skill Score (onset)     : {skill_onset:.4f}",
             f"Skill Score (recovery)  : {skill_recovery:.4f}", "",
             "-" * 74, "COMPETITION-STYLE PROXY (30-day penalty on false negatives)", "-" * 74,
             f"Effective onset MAE     : {eff_mae_on:.4f}",
             f"Effective recovery MAE  : {eff_mae_rec:.4f}",
             f"False Negatives (OOF)   : {int(cm[1, 0])}",
             "", f"Top 5 features         : " + ", ".join(imp_df['feature'].head(5)),
             "", f"Runtime                 : {time.time() - t0:.1f}s", "=" * 74]
    with open(os.path.join(OUTPUT, "metrics_summary.txt"), "w") as f:
        f.write("\n".join(lines))
    print("[step7] wrote output/metrics_summary.txt")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
