"""Load and clean the raw LendingClub extract into a modelling table.

Two deliberate departures from the original notebooks:

1. Rows with missing values are kept. LightGBM handles NaN natively, and
   missingness here is not random -- `emp_length` is absent more often for
   applicants without steady employment, which correlates with the target.
   Dropping those rows silently removes the riskiest slice of the book.
2. `issue_d` and `term` are retained so that temporal validation and the
   maturity filter are possible downstream.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def _to_float_pct(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.replace("%", "", regex=False).str.strip(),
        errors="coerce",
    )


def load_raw(path=None, nrows=None) -> pd.DataFrame:
    """Read only the columns we need; the raw file is ~1.6 GB."""
    path = path or C.RAW_CSV
    return pd.read_csv(path, usecols=C.USECOLS, low_memory=False, nrows=nrows)


def build(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to resolved loans and attach target, dates, and economics."""
    df = df[df["loan_status"].isin(C.TERMINAL_STATUSES)].copy()
    df["target"] = df["loan_status"].isin(C.DEFAULT_STATUSES).astype(int)

    # ── dates ────────────────────────────────────────────────────────────
    df["issue_d"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    df["earliest_cr_line"] = pd.to_datetime(
        df["earliest_cr_line"], format="%b-%Y", errors="coerce"
    )
    df = df[df["issue_d"].notna()]
    df["issue_year"] = df["issue_d"].dt.year

    # ── numeric coercion ─────────────────────────────────────────────────
    df["int_rate"] = _to_float_pct(df["int_rate"])
    df["revol_util"] = _to_float_pct(df["revol_util"])
    df["emp_length"] = df["emp_length"].map(C.EMP_LENGTH_MAP)
    df["grade"] = df["grade"].map(C.GRADE_MAP)
    df["sub_grade"] = (
        df["sub_grade"].str[0].map(C.GRADE_MAP) * 5
        + pd.to_numeric(df["sub_grade"].str[1], errors="coerce")
    )
    df["term_months"] = pd.to_numeric(
        df["term"].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
    )

    # ── maturity ─────────────────────────────────────────────────────────
    # A loan is "matured" if its full term elapsed before the snapshot date.
    # Un-matured loans that reached a terminal status did so early, which for
    # this dataset means disproportionately by default -- including them in a
    # late test window confounds distribution shift with survivorship.
    snapshot = pd.Timestamp(C.SNAPSHOT_END)
    df["maturity_date"] = df["issue_d"] + pd.to_timedelta(
        df["term_months"] * 30.44, unit="D"
    )
    df["matured"] = df["maturity_date"] <= snapshot

    df = _attach_economics(df)
    return df.reset_index(drop=True)


def estimate_lgd(df: pd.DataFrame) -> float:
    """Empirical loss given default from charged-off loans.

    Loss = principal never recovered, as a share of the amount lent. Uses only
    realised outcomes, so it is an input to the cost model rather than a
    feature -- it is never available at decision time for a live application.
    """
    d = df[df["target"] == 1]
    lent = d["loan_amnt"]
    recovered = d["total_rec_prncp"].fillna(0) + d["recoveries"].fillna(0)
    lgd = ((lent - recovered) / lent).clip(0, 1)
    value = float(lgd.mean())
    return value if np.isfinite(value) else C.FALLBACK_LGD


def _attach_economics(df: pd.DataFrame) -> pd.DataFrame:
    """Per-loan cost of each error type, using only ex-ante quantities.

    cost_fn -- approving a loan that defaults: principal lost.
    cost_fp -- declining a loan that would have repaid: interest forgone over
               the full scheduled term.

    Both are computed from the loan amount, instalment, and term, all known at
    origination. Realised payment columns are deliberately *not* used here.
    """
    scheduled_total = df["installment"] * df["term_months"]
    df["forgone_interest"] = (scheduled_total - df["loan_amnt"]).clip(lower=0)
    return df


def add_costs(df: pd.DataFrame, lgd: float) -> pd.DataFrame:
    df = df.copy()
    df["cost_fn"] = df["loan_amnt"] * lgd
    df["cost_fp"] = df["forgone_interest"]
    return df
