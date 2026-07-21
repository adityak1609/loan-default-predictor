"""Does the model hold up out-of-time, and can we even tell?

Part 1 -- train on 2007-2015, test on 2016-2018, plus per-year AUC.

Part 2 -- the confound. This snapshot ends 2018Q4, so no loan issued from 2016
onward has reached the end of its term. Every 2016+ loan with a terminal
status resolved *early*. Restricting to terminal statuses therefore samples a
different population in the test window than in the training window, and any
change in AUC mixes genuine distribution shift with that selection effect.

We cannot remove the confound with this snapshot, but we can measure it: 2014
and 2015 contain both matured and un-matured resolved loans, so scoring the
same model on each subset isolates the maturity effect from the time effect.

Run: python scripts/temporal_validation.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import lightgbm as lgb
import pandas as pd

from loanguard import config as C
from loanguard import evaluate as E
from loanguard.features import FeatureSpec
from loanguard.model import LGB_PARAMS

TRAIN_END = 2015
VAL_YEAR = 2015


def main() -> None:
    C.REPORTS.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(C.PROCESSED / "loans.parquet")

    train = df[df["issue_year"] <= TRAIN_END - 1]
    val = df[df["issue_year"] == VAL_YEAR]
    test = df[df["issue_year"] > TRAIN_END]
    print(f"train <=2014: {len(train):,} | val 2015: {len(val):,} "
          f"| test 2016-18: {len(test):,}")

    spec = FeatureSpec.build("full", train)
    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(
        spec.transform(train), train["target"],
        eval_set=[(spec.transform(val), val["target"])],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    def score(frame):
        if len(frame) < 500 or frame["target"].nunique() < 2:
            return None
        p = model.predict_proba(spec.transform(frame))[:, 1]
        out = {"n": len(frame)}
        out.update(E.ranking_metrics(frame["target"].values, p))
        out.update(E.calibration_metrics(frame["target"].values, p))
        return out

    # ── per-year ─────────────────────────────────────────────────────────
    per_year = []
    for year, g in df.groupby("issue_year"):
        s = score(g)
        if s:
            per_year.append({"issue_year": int(year),
                             "matured_share": float(g["matured"].mean()), **s})
    per_year = pd.DataFrame(per_year)
    per_year.to_csv(C.REPORTS / "temporal_per_year.csv", index=False)
    print("\nper-year performance (model trained on <=2014):")
    print(per_year[["issue_year", "n", "matured_share", "base_rate",
                    "auc", "brier", "ece"]].round(4).to_string(index=False))

    in_time = score(val)
    out_time = score(test)
    print(f"\nin-time  (2015):      AUC {in_time['auc']:.4f}  ECE {in_time['ece']:.4f}")
    print(f"out-of-time (2016-18): AUC {out_time['auc']:.4f}  ECE {out_time['ece']:.4f}")

    # ── the maturity confound, measured ──────────────────────────────────
    mixed = df[df["issue_year"].isin([2014, 2015])]
    rows = []
    for (year, matured), g in mixed.groupby(["issue_year", "matured"]):
        s = score(g)
        if s:
            rows.append({"issue_year": int(year), "matured": bool(matured), **s})
    maturity = pd.DataFrame(rows)
    maturity.to_csv(C.REPORTS / "maturity_confound.csv", index=False)
    print("\nmaturity effect within 2014-2015 (same years, same model):")
    print(maturity[["issue_year", "matured", "n", "base_rate", "auc"]]
          .round(4).to_string(index=False))

    gap = (
        maturity[~maturity["matured"]]["base_rate"].mean()
        - maturity[maturity["matured"]]["base_rate"].mean()
    )
    auc_gap = (
        maturity[~maturity["matured"]]["auc"].mean()
        - maturity[maturity["matured"]]["auc"].mean()
    )

    (C.REPORTS / "temporal.json").write_text(json.dumps({
        "in_time_2015": in_time,
        "out_of_time_2016_2018": out_time,
        "auc_drop": in_time["auc"] - out_time["auc"],
        "maturity_base_rate_gap": float(gap),
        "maturity_auc_gap": float(auc_gap),
        "note": ("No 2016+ loan reached term before the 2018Q4 snapshot, so the "
                 "out-of-time test set contains only early-resolving loans. The "
                 "maturity gaps quantify how much of the AUC change that alone "
                 "could explain."),
    }, indent=2), encoding="utf-8")

    print(f"\nAUC drop in-time -> out-of-time: {in_time['auc'] - out_time['auc']:+.4f}")
    print(f"Default-rate gap attributable to maturity alone: {gap:+.4f}")
    print(f"AUC gap attributable to maturity alone:          {auc_gap:+.4f}")


if __name__ == "__main__":
    main()
