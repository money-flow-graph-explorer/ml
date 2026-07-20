"""
Train an XGBoost re-score model on rule-candidate features.

Input CSV (produced by the backend collect mode) columns:
  ts, label, <feature columns...>
where `ts` is the simulation timestamp (used ONLY for a temporal train/test split,
not a model feature) and `label` is 1 if the rule candidate involved a ground-truth
fraud edge, else 0.

The model learns to separate true-positive candidates from false-positive ones, so
the streaming pipeline can suppress the candidates it scores as normal (cutting the
rule detector's false positives while keeping recall bound by the rules).

Usage:
  python train.py [--data ../data/training_candidates.csv] [--out .] [--test-frac 0.3]
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from xgboost import XGBClassifier

META_COLS = ["ts", "label"]  # not features


def temporal_split(df: pd.DataFrame, test_frac: float):
    """Split by timestamp so the model is tested on later events (no leakage)."""
    df = df.sort_values("ts", kind="stable").reset_index(drop=True)
    cut = df["ts"].quantile(1 - test_frac)
    train = df[df["ts"] <= cut]
    test = df[df["ts"] > cut]
    # Fallback if the temporal cut leaves an empty/degenerate side (e.g. all same ts).
    if len(test) == 0 or train["label"].nunique() < 2:
        n = len(df)
        split = int(n * (1 - test_frac))
        train, test = df.iloc[:split], df.iloc[split:]
    return train, test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data/training_candidates.csv")
    ap.add_argument("--out", default=".")
    ap.add_argument("--test-frac", type=float, default=0.3)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    feature_cols = [c for c in df.columns if c not in META_COLS]
    print(f"Loaded {len(df)} candidates, {len(feature_cols)} features: {feature_cols}")
    pos = int(df["label"].sum())
    print(f"Label balance: positives={pos} ({pos/len(df):.2%}), negatives={len(df)-pos}")

    train, test = temporal_split(df, args.test_frac)
    Xtr, ytr = train[feature_cols].fillna(0.0), train["label"].astype(int)
    Xte, yte = test[feature_cols].fillna(0.0), test["label"].astype(int)
    print(f"Train={len(Xtr)} (pos={int(ytr.sum())})  Test={len(Xte)} (pos={int(yte.sum())})")

    # Handle extreme imbalance among candidates.
    n_pos = max(1, int(ytr.sum()))
    n_neg = max(1, len(ytr) - n_pos)
    spw = n_neg / n_pos

    model = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=spw,
        eval_metric="aucpr",
        n_jobs=4,
    )
    model.fit(Xtr, ytr)

    proba = model.predict_proba(Xte)[:, 1]
    if yte.nunique() >= 2:
        pr_auc = average_precision_score(yte, proba)
        roc_auc = roc_auc_score(yte, proba)
    else:
        pr_auc = roc_auc = float("nan")
    print(f"\n=== Holdout ===\nPR-AUC={pr_auc:.4f}  ROC-AUC={roc_auc:.4f}")

    print("\nthreshold  precision  recall   f1     alerts")
    best = None
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        pred = (proba >= thr).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(
            yte, pred, average="binary", zero_division=0
        )
        print(f"  {thr:0.2f}      {p:0.3f}     {r:0.3f}   {f1:0.3f}   {int(pred.sum())}")
        if best is None or f1 > best[1]:
            best = (thr, f1)
    print(f"\nBest-F1 threshold ~ {best[0]:.2f}")
    thr = 0.5
    print(f"\nConfusion matrix @ {thr}:\n{confusion_matrix(yte, (proba>=thr).astype(int))}")

    # Feature importances
    imp = sorted(zip(feature_cols, model.feature_importances_), key=lambda x: -x[1])
    print("\nTop features:")
    for name, val in imp[:10]:
        print(f"  {name}: {val:.3f}")

    os.makedirs(args.out, exist_ok=True)
    model.save_model(os.path.join(args.out, "model.json"))
    with open(os.path.join(args.out, "feature_names.json"), "w") as f:
        json.dump(feature_cols, f)
    print(f"\nSaved model.json + feature_names.json to {args.out}")


if __name__ == "__main__":
    main()
