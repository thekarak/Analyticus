#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
predict.py - Inference using the pre-trained ensemble.
Part of the Analyticus submission (PlayHack 2026 ML Track, IIT Guwahati).

Run:
  python src/predict.py                        # model-threshold mode (optimized t)
  python src/predict.py --recall-mode         # recall-boosted mode (top `prior`%)
  python src/predict.py --recall-mode --prior 0.35

Two files are always written so you can A/B:
  output/submission_final.csv        (whichever mode was last run)
  output/submission_modelbased.csv   (model-threshold predictions)
  output/submission_recallboost.csv  (recall-boosted predictions)

The recall-boosted mode directly defends against the competition's 30-day
false-negative penalty: it predicts "injured" for the top `prior` fraction of
athletes ranked by model probability, guaranteeing high recall.
"""
import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")
SRC = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SRC)
sys.path.insert(0, SRC)

from preprocess import build_features, add_advanced_features, proba_or_pred, _load_raw  # noqa

MODELS = os.path.join(ROOT, "models")
OUTPUT = os.path.join(ROOT, "output")
SUB_TEMPLATE = os.path.join(ROOT, "example.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=SUB_TEMPLATE)
    ap.add_argument("--out", default=os.path.join(OUTPUT, "submission_final.csv"))
    ap.add_argument("--recall-mode", action="store_true",
                    help="Predict injured for the top `prior` fraction by probability.")
    ap.add_argument("--prior", type=float, default=0.35,
                    help="Target injured fraction for recall mode (default 0.35).")
    ap.add_argument("--data-dir", default=None,
                    help="Directory containing test CSV files (e.g. TEst_data or data/test).")
    args = ap.parse_args()

    clf_bundle = joblib.load(os.path.join(MODELS, "ensemble_clf.bin"))
    onset_bundle = joblib.load(os.path.join(MODELS, "ensemble_onset.bin"))
    rec_bundle = joblib.load(os.path.join(MODELS, "ensemble_rec.bin"))

    prep = clf_bundle["prep"]
    best_t = clf_bundle["best_t"]
    base_onset = clf_bundle["base_onset"]
    base_recovery = clf_bundle["base_recovery"]
    clf_models = list(clf_bundle["models"].values())
    onset_models = list(onset_bundle["models"].values())
    rec_models = list(rec_bundle["models"].values())

    sub = pd.read_csv(args.input)
    test_ids = sub["athlete_id"].tolist()

    # locate test features
    test_data_dir = None
    if args.data_dir and os.path.isdir(args.data_dir):
        test_data_dir = args.data_dir
    else:
        candidates = [
            os.path.join(ROOT, "data", "test"),
            os.path.join(ROOT, "TEst_data"),
            os.path.join(ROOT, "test_data"),
            os.path.join(ROOT, "Test_data"),
            os.path.join(ROOT, "data", "TEst_data"),
        ]
        for c in candidates:
            if os.path.isdir(c) and os.path.exists(os.path.join(c, "athlete_metadata.csv")):
                test_data_dir = c
                break

        if test_data_dir is None:
            # search any directory with matching athlete_ids in athlete_metadata.csv
            for item in os.listdir(ROOT):
                full_p = os.path.join(ROOT, item)
                if os.path.isdir(full_p) and os.path.exists(os.path.join(full_p, "athlete_metadata.csv")):
                    meta = pd.read_csv(os.path.join(full_p, "athlete_metadata.csv"))
                    if meta["athlete_id"].isin(test_ids).any():
                        test_data_dir = full_p
                        break

        if test_data_dir is None:
            probe = _load_raw(ROOT)["meta"]
            if probe is not None and probe["athlete_id"].isin(test_ids).any():
                test_data_dir = ROOT

    if test_data_dir is None:
        print("[predict] WARNING: no test features found -> fallback predictions.")
        out = sub.copy()
        out["injured_in_risk_window"] = (np.full(len(test_ids), 0.35) >= best_t).astype(int)
        out["onset_day_offset"] = int(round(base_onset))
        out["recovery_duration"] = int(round(base_recovery))
        out.to_csv(args.out, index=False)
        print(f"[predict] wrote fallback -> {args.out}")
        return

    print(f"[predict] building test features from {test_data_dir}")
    feat = add_advanced_features(build_features(test_ids, test_data_dir))
    Xte = prep.transform(feat)

    pc = np.mean([proba_or_pred(m, Xte) for m in clf_models], axis=0)
    po = np.mean([m.predict(Xte) for m in onset_models], axis=0)
    pr = np.mean([m.predict(Xte) for m in rec_models], axis=0)

    # ---- decide injury labels for this mode ----------------------------- #
    if args.recall_mode:
        prior = min(max(args.prior, 0.0), 1.0)
        k = int(round(prior * len(pc)))
        k = min(max(k, 1), len(pc))
        order = np.argsort(-pc)                 # rank by probability, descending
        injured = np.zeros(len(pc), dtype=int)
        injured[order[:k]] = 1
        mode_name = f"RECALL-BOOSTED (top {prior*100:.0f}% by probability)"
        copy_name = "submission_recallboost.csv"
    else:
        injured = (pc >= best_t).astype(int)
        mode_name = f"MODEL-THRESHOLD (t={best_t:.3f})"
        copy_name = "submission_modelbased.csv"

    out = sub.copy()
    out["injured_in_risk_window"] = injured
    out["onset_day_offset"] = np.round(po).astype(int)
    out["recovery_duration"] = np.round(pr).astype(int)
    out.to_csv(args.out, index=False)
    out.to_csv(os.path.join(OUTPUT, copy_name), index=False)
    assert out.isnull().sum().sum() == 0

    n_inj = int(injured.sum())
    print(f"[predict] MODE = {mode_name}")
    print(f"[predict] predicted injured = {n_inj}/{len(test_ids)} "
          f"({100*n_inj/len(test_ids):.1f}%)")
    print(f"[predict] wrote {args.out}")
    print(f"[predict] wrote output/{copy_name}  (for A/B comparison)")

    if args.recall_mode:
        print("\n[recommend] Given the 30-day false-negative penalty, the "
              "recall-boosted file is the safer primary submission. Swap "
              "submission_recallboost.csv -> submission_final.csv if desired.")


if __name__ == "__main__":
    main()
