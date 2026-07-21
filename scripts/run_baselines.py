"""How much does the model add over the lender's existing scorecard?

`grade` and `int_rate` are LendingClub's own risk-model output, not borrower
attributes. A model trained on them is partly re-learning an existing
scorecard, so a headline AUC is uninterpretable without knowing what those
features alone already deliver.

Fits, on one shared split:
  grade_only      logistic regression on a single feature
  incumbent_only  grade + sub_grade + int_rate
  applicant_only  borrower attributes only, no incumbent signal
  full            everything
  full_logreg     properly scaled linear baseline

Run: python scripts/run_baselines.py
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
from loanguard.model import fit_lgbm, fit_logreg, split


def main() -> None:
    C.REPORTS.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(C.PROCESSED / "loans.parquet")
    train, val, test = split(df)
    print(f"train {len(train):,} | val {len(val):,} | test {len(test):,}")

    results = []

    for name in ["grade_only", "incumbent_only", "applicant_only", "full"]:
        spec = FeatureSpec.build(name, train)
        model = (
            fit_logreg(spec, train)
            if name == "grade_only"
            else fit_lgbm(spec, train, val)
        )
        p = model.predict_proba(spec.transform(test))[:, 1]
        row = {"model": name, "n_features": len(spec.columns)}
        row.update(E.ranking_metrics(test["target"].values, p))
        results.append(row)
        print(f"  {name:16s} AUC {row['auc']:.4f}  AP {row['average_precision']:.4f}")

    spec_full = FeatureSpec.build("full", train)
    lr = fit_logreg(spec_full, train)
    p = lr.predict_proba(spec_full.transform(test))[:, 1]
    row = {"model": "full_logreg", "n_features": len(spec_full.columns)}
    row.update(E.ranking_metrics(test["target"].values, p))
    results.append(row)
    print(f"  {'full_logreg':16s} AUC {row['auc']:.4f}  AP {row['average_precision']:.4f}")

    out = pd.DataFrame(results)
    base = out.loc[out["model"] == "grade_only", "auc"].iloc[0]
    out["auc_over_grade_only"] = (out["auc"] - base).round(4)
    out.to_csv(C.REPORTS / "baselines.csv", index=False)
    (C.REPORTS / "baselines.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )

    print("\n" + out.to_string(index=False))


if __name__ == "__main__":
    main()
