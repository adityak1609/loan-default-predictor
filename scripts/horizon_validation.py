"""Re-test for temporal drift once survivorship is removed.

`temporal_validation.py` found a 0.024 out-of-time AUC drop and showed it was
roughly the size of the maturity penalty measured within single origination
years -- i.e. probably an artefact. That argument was indirect, because the
terminal-status target could not be observed cleanly for late vintages at all.

The 12-month observation window removes the confound outright: every vintage
is watched for exactly the same 12 months, and loans still performing at the
snapshot count as negatives instead of being dropped. If the earlier drop was
survivorship, it should now largely disappear. If genuine drift exists, it
should survive.

Run: python scripts/horizon_validation.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from loanguard import config as C
from loanguard import evaluate as E
from loanguard.features import FeatureSpec
from loanguard.model import fit_lgbm

TRAIN_END = 2014
VAL_YEAR = 2015


def main() -> None:
    C.REPORTS.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(C.PROCESSED / "loans_12m.parquet")

    train = df[df["issue_year"] <= TRAIN_END]
    val = df[df["issue_year"] == VAL_YEAR]
    test = df[df["issue_year"] > VAL_YEAR]
    print(f"train <={TRAIN_END}: {len(train):,} | val {VAL_YEAR}: {len(val):,} "
          f"| test {VAL_YEAR + 1}-2017: {len(test):,}")
    print(f"base rates -- train {train['target'].mean():.2%} | "
          f"val {val['target'].mean():.2%} | test {test['target'].mean():.2%}")

    spec = FeatureSpec.build("full", train)
    model = fit_lgbm(spec, train, val)

    def score(frame):
        if len(frame) < 500 or frame["target"].nunique() < 2:
            return None
        p = model.predict_proba(spec.transform(frame))[:, 1]
        out = {"n": len(frame)}
        out.update(E.ranking_metrics(frame["target"].values, p))
        out.update(E.calibration_metrics(frame["target"].values, p))
        return out

    per_year = []
    for year, g in df.groupby("issue_year"):
        s = score(g)
        if s:
            per_year.append({"issue_year": int(year), **s})
    per_year = pd.DataFrame(per_year)
    per_year.to_csv(C.REPORTS / "horizon_per_year.csv", index=False)
    print("\nper-year performance (trained on <=2014, 12-month window target):")
    print(per_year[["issue_year", "n", "base_rate", "auc",
                    "average_precision", "brier", "ece"]]
          .round(4).to_string(index=False))

    in_time, out_time = score(val), score(test)
    drop = in_time["auc"] - out_time["auc"]

    # The comparable figure from the resolved-only target.
    prev = json.loads((C.REPORTS / "temporal.json").read_text())

    result = {
        "target": f"default within {C.HORIZON_MONTHS} months of origination",
        "dpd_lag_months": C.DPD_LAG_MONTHS,
        "n_total": int(len(df)),
        "in_time_2015": in_time,
        "out_of_time_2016_2017": out_time,
        "auc_drop": drop,
        "auc_drop_resolved_only_target": prev["auc_drop"],
        "maturity_auc_gap_resolved_only": prev["maturity_auc_gap"],
    }
    (C.REPORTS / "horizon_validation.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")

    print(f"\nin-time  ({VAL_YEAR}):        AUC {in_time['auc']:.4f}  "
          f"AP {in_time['average_precision']:.4f}  ECE {in_time['ece']:.4f}")
    print(f"out-of-time (2016-17):   AUC {out_time['auc']:.4f}  "
          f"AP {out_time['average_precision']:.4f}  ECE {out_time['ece']:.4f}")
    print(f"\nAUC drop, fixed 12-month window:  {drop:+.4f}")
    print(f"AUC drop, resolved-only target:   {prev['auc_drop']:+.4f}")
    print(f"  (maturity penalty measured earlier: "
          f"{prev['maturity_auc_gap']:+.4f})")


if __name__ == "__main__":
    main()
