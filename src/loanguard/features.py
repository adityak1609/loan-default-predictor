"""Feature construction, shared by training and serving.

The Streamlit app imports `FeatureSpec.load()` and calls `transform()` on a
single-row frame, so the column set and ordering can never drift from what the
model was fitted on. Previously the one-hot column list was retyped by hand in
the app, which happened to work only because unknown columns were silently
filled with zeros.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C

FEATURE_SETS = {
    # A single feature: the lender's own grade. The reference point for
    # judging whether the full model adds anything.
    "grade_only": (["grade"], []),
    # Everything LendingClub's existing scorecard already encodes.
    "incumbent_only": (["grade", "sub_grade", "int_rate"], []),
    # Borrower attributes only -- can we underwrite without the incumbent?
    "applicant_only": (
        C.APPLICANT_NUMERIC + C.ENGINEERED,
        C.APPLICANT_CATEGORICAL,
    ),
    # Everything.
    "full": (
        C.APPLICANT_NUMERIC + C.ENGINEERED + C.INCUMBENT,
        C.APPLICANT_CATEGORICAL,
    ),
    # NOT A VALID MODEL. Adds post-origination columns that cannot exist when
    # the lending decision is made. Used only by scripts/leakage_demo.py to
    # show what they do to the reported metric.
    "full_leaky": (
        C.APPLICANT_NUMERIC + C.ENGINEERED + C.INCUMBENT + C.LEAKY,
        C.APPLICANT_CATEGORICAL,
    ),
}


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Ratios and transforms that the original pipeline never built."""
    df = df.copy()
    inc = df["annual_inc"].replace(0, np.nan)

    df["loan_to_income"] = df["loan_amnt"] / inc
    df["installment_to_monthly_income"] = df["installment"] / (inc / 12)
    df["revol_bal_to_income"] = df["revol_bal"] / inc
    # Self-reported income is heavy-tailed with implausible extremes; the log
    # keeps a linear model usable without discarding the tail.
    df["log_annual_inc"] = np.log1p(df["annual_inc"].clip(lower=0))

    if "earliest_cr_line" in df.columns:
        df["credit_history_years"] = (
            (df["issue_d"] - df["earliest_cr_line"]).dt.days / 365.25
        )
    else:
        df["credit_history_years"] = np.nan

    return df


@dataclass
class FeatureSpec:
    """Frozen record of the exact columns a model was trained on."""

    name: str
    numeric: list[str]
    categorical: list[str]
    columns: list[str] = field(default_factory=list)

    @classmethod
    def build(cls, name: str, df: pd.DataFrame) -> "FeatureSpec":
        numeric, categorical = FEATURE_SETS[name]
        spec = cls(name=name, numeric=list(numeric), categorical=list(categorical))
        spec.columns = list(spec._encode(df).columns)
        return spec

    def _encode(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df[self.numeric].copy()
        if self.categorical:
            dummies = pd.get_dummies(
                df[self.categorical].astype("string"),
                columns=self.categorical,
                dummy_na=False,
                dtype=float,
            )
            out = pd.concat([out, dummies], axis=1)
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode and align to the fitted column set."""
        if "loan_to_income" not in df.columns:
            df = engineer(df)
        out = self._encode(df)
        # reindex both adds missing dummy levels as 0 and enforces ordering
        return out.reindex(columns=self.columns, fill_value=0.0).astype(float)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.__dict__, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "FeatureSpec":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))
