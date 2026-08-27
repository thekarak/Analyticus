#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
preprocess.py - Data merging, feature engineering & leakage-free preprocessing.
Part of the Analyticus submission (PlayHack 2026 ML Track, IIT Guwahati).

This module is imported by both train.py and predict.py so that the exact
same feature pipeline is applied to training and test data (no leakage).
"""
import os
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, recall_score

import xgboost as xgb
import lightgbm as lgb
import catboost as cb

# --------------------------------------------------------------------------- #
# Global configuration (fixed seeds for full reproducibility)
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
N_FOLDS = 5
RECALL_FLOOR = 0.90          # threshold search keeps recall >= this (beat FN penalty)
np.random.seed(RANDOM_STATE)

_RAW_CACHE = {}


# --------------------------------------------------------------------------- #
# STEP 1 - DATA LOADING (cached per source directory)
# --------------------------------------------------------------------------- #
def _load_raw(data_dir):
    """Load all relational CSVs from `data_dir`, caching by directory."""
    if data_dir in _RAW_CACHE:
        return _RAW_CACHE[data_dir]

    def _rd(name):
        p = os.path.join(data_dir, name)
        return pd.read_csv(p) if os.path.exists(p) else None

    raw = {
        "meta":     _rd("athlete_metadata.csv"),
        "daily":    _rd("dailyActivity_merged.csv"),
        "sleep":    _rd("sleepDay_merged.csv"),
        "weight":   _rd("weightLogInfo_merged.csv"),
        "hr":       _rd("hourlyHeartrate_merged.csv"),
        "steps":    _rd("hourlySteps_merged.csv"),
        "cal":      _rd("hourlyCalories_merged.csv"),
        "inten":    _rd("hourlyIntensities_merged.csv"),
        "sessions": _rd("training_sessions.csv"),
    }
    _RAW_CACHE[data_dir] = raw
    return raw


def _parse_dates(series, fmt=None):
    if fmt:
        return pd.to_datetime(series, format=fmt, errors="coerce")
    return pd.to_datetime(series, errors="coerce")


def build_features(ids, data_dir):
    """
    Build ONE master feature row per athlete_id (given by `ids`) from the
    relational CSVs located in `data_dir`. Returns a DataFrame indexed by
    athlete_id with numeric + a few reserved categorical columns.
    """
    raw = _load_raw(data_dir)
    ids = set(ids)

    feat = pd.DataFrame(index=pd.Index(sorted(ids), name="athlete_id"))

    # ---- athlete_metadata (static profile) -------------------------------- #
    if raw["meta"] is not None:
        m = raw["meta"].copy()
        m = m[m["athlete_id"].isin(ids)]
        m["bmi_baseline"] = m["weight_kg_baseline"] / ((m["height_cm"] / 100.0) ** 2)
        m = m.set_index("athlete_id")
        for c in ["age", "height_cm", "weight_kg_baseline", "years_playing",
                  "prior_season_injury_count", "bmi_baseline"]:
            if c in m.columns:
                feat[c] = m[c]
        for c in ["sport", "gender", "dominant_side", "position", "team_id"]:
            if c in m.columns:
                feat[c] = m[c].astype(str).fillna("Unknown")

    # ---- dailyActivity_merged --------------------------------------------- #
    if raw["daily"] is not None:
        d = raw["daily"].copy()
        d = d[d["Id"].isin(ids)].rename(columns={"Id": "athlete_id"})
        d["ActivityDate"] = _parse_dates(d["ActivityDate"], "%Y-%m-%d")
        d["active_minutes"] = (d["VeryActiveMinutes"].fillna(0)
                               + d["FairlyActiveMinutes"].fillna(0)
                               + d["LightlyActiveMinutes"].fillna(0))
        d["cal_per_active_min"] = d["Calories"] / d["active_minutes"].replace(0, np.nan)
        d["active_sed_ratio"] = d["active_minutes"] / d["SedentaryMinutes"].replace(0, np.nan)

        agg_map = {
            "TotalSteps": ["mean", "std", "max", "sum"],
            "TotalDistance": ["mean", "max"],
            "active_minutes": ["mean", "std", "max", "sum"],
            "VeryActiveMinutes": ["mean", "sum", "max"],
            "FairlyActiveMinutes": ["mean", "sum"],
            "LightlyActiveMinutes": ["mean", "sum"],
            "SedentaryMinutes": ["mean", "std", "max", "sum"],
            "Calories": ["mean", "std", "max", "sum"],
            "cal_per_active_min": ["mean"],
            "active_sed_ratio": ["mean"],
        }
        g = d.groupby("athlete_id").agg(agg_map)
        g.columns = [f"daily_{a}_{s}" for a, s in g.columns]
        feat = feat.join(g)

    # ---- sleepDay_merged --------------------------------------------------- #
    if raw["sleep"] is not None:
        s = raw["sleep"].copy()
        s = s[s["Id"].isin(ids)].rename(columns={"Id": "athlete_id"})
        s["SleepDay"] = _parse_dates(s["SleepDay"], "%Y-%m-%d")
        s["sleep_eff"] = s["TotalMinutesAsleep"] / s["TotalTimeInBed"].replace(0, np.nan)
        g = s.groupby("athlete_id").agg({
            "TotalSleepRecords": ["mean", "sum"],
            "TotalMinutesAsleep": ["mean", "std", "sum"],
            "TotalTimeInBed": ["mean"],
            "sleep_eff": ["mean"],
        })
        g.columns = [f"sleep_{a}_{s}" for a, s in g.columns]
        feat = feat.join(g)

    # ---- weightLogInfo_merged --------------------------------------------- #
    if raw["weight"] is not None:
        w = raw["weight"].copy()
        w = w[w["Id"].isin(ids)].rename(columns={"Id": "athlete_id"})
        w["Date"] = _parse_dates(w["Date"], "%Y-%m-%d")
        g = w.groupby("athlete_id").agg({
            "WeightKg": ["mean", "std"],
            "Fat": ["mean"],
            "BMI": ["mean", "std"],
        })
        g.columns = [f"weight_{a}_{s}" for a, s in g.columns]
        last = w.sort_values("Date").groupby("athlete_id").last()[["BMI", "WeightKg"]]
        first = w.sort_values("Date").groupby("athlete_id").first()[["BMI", "WeightKg"]]
        g["weight_bmi_trend"] = last["BMI"] - first["BMI"]
        g["weight_kg_trend"] = last["WeightKg"] - first["WeightKg"]
        if "weight_IsManualReport" in w.columns or "IsManualReport" in w.columns:
            col = "IsManualReport" if "IsManualReport" in w.columns else "weight_IsManualReport"
            frac = w[col].astype(str).eq("True").groupby(w["athlete_id"]).mean() \
                       .rename("weight_manual_frac")
            g = g.join(frac)
        feat = feat.join(g)

    # ---- hourlyHeartrate_merged ------------------------------------------- #
    if raw["hr"] is not None:
        h = raw["hr"].copy()
        h = h[h["Id"].isin(ids)].rename(columns={"Id": "athlete_id"})
        h["ActivityHour"] = _parse_dates(h["ActivityHour"])   # ISO 8601
        g = h.groupby("athlete_id").agg({
            "AvgHeartRate": ["mean", "std", "max"],
            "MinHeartRate": ["mean", "min"],     # resting HR proxy
            "MaxHeartRate": ["mean", "max"],     # peak exertion HR
        })
        g.columns = [f"hr_{a}_{s}" for a, s in g.columns]
        feat = feat.join(g)

    # ---- hourlySteps_merged ----------------------------------------------- #
    if raw["steps"] is not None:
        st = raw["steps"].copy()
        st = st[st["Id"].isin(ids)].rename(columns={"Id": "athlete_id"})
        st["ActivityHour"] = _parse_dates(st["ActivityHour"], "%m/%d/%Y %I:%M:%S %p")
        g = st.groupby("athlete_id")["StepTotal"].agg(["mean", "std", "max"])
        g.columns = [f"steps_hr_{s}" for s in g.columns]
        feat = feat.join(g)

    # ---- hourlyCalories_merged -------------------------------------------- #
    if raw["cal"] is not None:
        c = raw["cal"].copy()
        c = c[c["Id"].isin(ids)].rename(columns={"Id": "athlete_id"})
        c["ActivityHour"] = _parse_dates(c["ActivityHour"], "%m/%d/%Y %I:%M:%S %p")
        g = c.groupby("athlete_id")["Calories"].agg(["mean", "max"])
        g.columns = [f"cal_hr_{s}" for s in g.columns]
        feat = feat.join(g)

    # ---- hourlyIntensities_merged ----------------------------------------- #
    if raw["inten"] is not None:
        it = raw["inten"].copy()
        it = it[it["Id"].isin(ids)].rename(columns={"Id": "athlete_id"})
        it["ActivityHour"] = _parse_dates(it["ActivityHour"], "%m/%d/%Y %I:%M:%S %p")
        g = it.groupby("athlete_id")[["TotalIntensity", "AverageIntensity"]].mean()
        g.columns = [f"inten_hr_{s}" for s in g.columns]
        feat = feat.join(g)

    # ---- training_sessions.csv (optional) --------------------------------- #
    if raw["sessions"] is not None:
        ss = raw["sessions"].copy()
        ss = ss[ss["athlete_id"].isin(ids)]
        ss["total_hours"] = (ss["end_hour"] - ss["start_hour"]).clip(lower=0)
        g = ss.groupby("athlete_id").agg({
            "session_id": ["count"],
            "total_hours": ["sum"],
        })
        g.columns = ["sess_count", "sess_total_hours"]
        if "sport_session_type" in ss.columns:
            typ = pd.get_dummies(ss["sport_session_type"], prefix="sess").groupby(ss["athlete_id"]).mean()
            g = g.join(typ)
        feat = feat.join(g)

    return feat


# --------------------------------------------------------------------------- #
# STEP 3 - ADVANCED FEATURE ENGINEERING (strain / recovery interactions)
# --------------------------------------------------------------------------- #
def add_advanced_features(df):
    """Adds domain-driven strain & recovery features."""
    df = df.copy()

    resting = df.get("hr_MinHeartRate_min", df.get("hr_MinHeartRate_mean"))
    peak = df.get("hr_MaxHeartRate_max", df.get("hr_MaxHeartRate_mean"))
    peak_mean = df.get("hr_MaxHeartRate_mean")
    resting_mean = df.get("hr_MinHeartRate_mean")

    if peak is not None and resting is not None:
        df["hr_reserve"] = peak - resting
        df["hr_ratio"] = peak_mean / resting_mean.replace(0, np.nan)

    if "daily_active_minutes_mean" in df.columns and "daily_SedentaryMinutes_mean" in df.columns:
        df["active_vs_sedentary"] = (
            df["daily_active_minutes_mean"] / df["daily_SedentaryMinutes_mean"].replace(0, np.nan))

    if "daily_Calories_mean" in df.columns and "daily_active_minutes_mean" in df.columns:
        df["cal_per_active_min2"] = (
            df["daily_Calories_mean"] / df["daily_active_minutes_mean"].replace(0, np.nan))

    if "sleep_TotalMinutesAsleep_std" in df.columns:
        df["sleep_consistency"] = df["sleep_TotalMinutesAsleep_std"].fillna(
            df["sleep_TotalMinutesAsleep_std"].median())

    if "sleep_sleep_eff_mean" in df.columns and resting_mean is not None:
        df["recovery_index"] = df["sleep_sleep_eff_mean"] * (1.0 / resting_mean.replace(0, np.nan))

    if "daily_TotalSteps_mean" in df.columns and peak_mean is not None:
        df["steps_x_peakhr"] = df["daily_TotalSteps_mean"] * peak_mean
    if "daily_active_minutes_mean" in df.columns and "daily_Calories_mean" in df.columns:
        df["active_x_calories"] = df["daily_active_minutes_mean"] * df["daily_Calories_mean"]
    if "bmi_baseline" in df.columns and "age" in df.columns:
        df["bmi_x_age"] = df["bmi_baseline"] * df["age"]
    if "hr_reserve" in df.columns and "daily_VeryActiveMinutes_mean" in df.columns:
        df["hrreserve_x_va"] = df["hr_reserve"] * df["daily_VeryActiveMinutes_mean"]

    df = df.replace([np.inf, -np.inf], np.nan)
    return df


# --------------------------------------------------------------------------- #
# STEP 2 - LEAKAGE-FREE PREPROCESSOR
# --------------------------------------------------------------------------- #
class Preprocessor:
    """Median imputation + smoothed target encoding + RobustScaler.
    Fit on train split only; transform val/test identically."""

    def __init__(self, cat_cols, num_cols, smooth=10):
        self.cat_cols = list(cat_cols)
        self.num_cols = list(num_cols)
        self.smooth = smooth
        self.num_medians = {}
        self.cat_modes = {}
        self.te_maps = {}
        self.scaler = RobustScaler()

    def fit(self, X, y):
        X = X.copy()
        self.num_medians = X[self.num_cols].median(numeric_only=True).to_dict()
        Xn = X[self.num_cols].fillna(self.num_medians)
        y = np.asarray(y)
        gmean = y.mean()
        for c in self.cat_cols:
            col = X[c].astype(str).fillna("Unknown")
            self.cat_modes[c] = col.mode().iloc[0] if len(col.mode()) else "Unknown"
            grp = pd.DataFrame({"c": col, "y": y}).groupby("c")["y"].agg(["mean", "count"])
            enc = (grp["mean"] * grp["count"] + gmean * self.smooth) / (grp["count"] + self.smooth)
            self.te_maps[c] = (enc.to_dict(), float(gmean))
        self.scaler.fit(Xn.values)
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.num_cols:
            if c not in X.columns:
                X[c] = np.nan
        for c in self.cat_cols:
            if c not in X.columns:
                X[c] = "Unknown"
        Xn = X[self.num_cols].fillna(self.num_medians)
        Xs = self.scaler.transform(Xn.values)
        out = pd.DataFrame(Xs, columns=self.num_cols, index=X.index)
        for c in self.cat_cols:
            enc_map, gmean = self.te_maps[c]
            col = X[c].astype(str).fillna(self.cat_modes.get(c, "Unknown"))
            out[c + "_te"] = col.map(enc_map).fillna(gmean).astype(float).values
        return out


# --------------------------------------------------------------------------- #
# STEP 5 - MODEL FACTORIES
# --------------------------------------------------------------------------- #
def make_classifiers():
    return {
        "xgb": xgb.XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0.0,
            eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1,
            verbosity=0),
        "lgbm": lgb.LGBMClassifier(
            n_estimators=400, max_depth=8, learning_rate=0.05,
            num_leaves=63, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
        "cb": cb.CatBoostClassifier(
            iterations=400, depth=6, learning_rate=0.05, l2_leaf_reg=3.0,
            random_state=RANDOM_STATE, verbose=False, allow_writing_files=False),
    }


def make_regressors():
    return {
        "xgb": xgb.XGBRegressor(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0.0,
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0),
        "lgbm": lgb.LGBMRegressor(
            n_estimators=400, max_depth=8, learning_rate=0.05,
            num_leaves=63, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=RANDOM_STATE, n_jobs=-1, verbose=-1),
        "cb": cb.CatBoostRegressor(
            iterations=400, depth=6, learning_rate=0.05, l2_leaf_reg=3.0,
            random_state=RANDOM_STATE, verbose=False, allow_writing_files=False),
    }


def proba_or_pred(model, X):
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(X)
        return p[:, 1] if p.shape[1] > 1 else p[:, 0]
    return model.predict(X)


def optimize_threshold(y_true, oof_proba, recall_floor=RECALL_FLOOR):
    """Pick threshold maximizing F1 while keeping recall >= recall_floor
    (defends against the 30-day false-negative penalty)."""
    best_t, best_f1, best_rec = 0.5, -1, 0
    for t in np.linspace(0.05, 0.6, 56):
        pred = (oof_proba >= t).astype(int)
        rec = recall_score(y_true, pred)
        f1 = f1_score(y_true, pred)
        if rec >= recall_floor and f1 > best_f1:
            best_t, best_f1, best_rec = t, f1, rec
    if best_f1 < 0:
        best_rec, best_t = -1, 0.05
        for t in np.linspace(0.05, 0.6, 56):
            pred = (oof_proba >= t).astype(int)
            rec = recall_score(y_true, pred)
            if rec > best_rec:
                best_rec, best_t = rec, t
        best_f1 = f1_score(y_true, (oof_proba >= best_t).astype(int))
    return float(best_t), float(best_f1), float(best_rec)
