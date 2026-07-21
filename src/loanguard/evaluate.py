"""Evaluation: ranking, calibration, and cost-based threshold selection.

The original pipeline reported AUC and an F1 score taken at p = 0.5. F1 at an
arbitrary threshold implicitly assumes a false negative and a false positive
cost the same amount, which for lending is wrong by roughly an order of
magnitude. Everything here is built around the decision instead of the score.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)


def ranking_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "base_rate": float(np.mean(y)),
    }


def expected_calibration_error(y, p, n_bins: int = 20) -> float:
    """Bin-weighted mean gap between predicted and observed default rate."""
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        total += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(total)


def calibration_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    return {
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p),
        "mean_predicted": float(np.mean(p)),
        "observed_rate": float(np.mean(y)),
    }


def reliability_curve(y, p, n_bins: int = 20) -> pd.DataFrame:
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        rows.append(
            {
                "bin": b,
                "n": int(mask.sum()),
                "mean_predicted": float(p[mask].mean()),
                "observed_rate": float(y[mask].mean()),
            }
        )
    return pd.DataFrame(rows)


def cost_curve(
    y: np.ndarray,
    p: np.ndarray,
    cost_fn: np.ndarray,
    cost_fp: np.ndarray,
    grid: np.ndarray | None = None,
) -> pd.DataFrame:
    """Total expected loss across candidate approval thresholds.

    Decision rule: approve when predicted default probability < threshold.
      - approved and defaults  -> principal lost      (cost_fn)
      - declined and repays    -> interest forgone    (cost_fp)

    A threshold of 1.0 approves everyone, which is the do-nothing baseline.
    """
    if grid is None:
        grid = np.linspace(0.01, 1.0, 100)

    y = np.asarray(y).astype(bool)
    p = np.asarray(p, dtype=float)
    cost_fn = np.asarray(cost_fn, dtype=float)
    cost_fp = np.asarray(cost_fp, dtype=float)

    order = np.argsort(p)
    p_s, y_s = p[order], y[order]
    fn_s, fp_s = cost_fn[order], cost_fp[order]

    # cum_fn[i] = cost of approving the i lowest-risk loans that default
    cum_fn = np.concatenate([[0.0], np.cumsum(np.where(y_s, fn_s, 0.0))])
    # cum_fp_tail[i] = cost of declining everything from i onward that repays
    rev = np.cumsum(np.where(~y_s, fp_s, 0.0)[::-1])[::-1]
    cum_fp_tail = np.concatenate([rev, [0.0]])

    cut = np.searchsorted(p_s, grid, side="left")
    total = cum_fn[cut] + cum_fp_tail[cut]

    n = len(p_s)
    return pd.DataFrame(
        {
            "threshold": grid,
            "total_cost": total,
            "approval_rate": cut / n,
            "cost_per_application": total / n,
        }
    )


def select_threshold(curve: pd.DataFrame, tolerance: float = 0.01) -> dict:
    """Pick the cost-minimising threshold, and report how flat the optimum is.

    Reporting a single argmin here would be false precision: the cost curve on
    this data is flat to within a fraction of a percent across a wide band, so
    the argmin moves between validation and test purely on sampling noise. The
    honest summary is the band of thresholds whose cost is within `tolerance`
    of the minimum -- anything inside it is operationally equivalent.
    """
    best = curve.loc[curve["total_cost"].idxmin()]
    approve_all = float(curve["total_cost"].iloc[-1])
    near = curve[curve["total_cost"] <= best["total_cost"] * (1 + tolerance)]
    return {
        "threshold": float(best["threshold"]),
        "total_cost": float(best["total_cost"]),
        "approval_rate": float(best["approval_rate"]),
        "cost_per_application": float(best["cost_per_application"]),
        "approve_all_cost": approve_all,
        "savings_vs_approve_all": approve_all - float(best["total_cost"]),
        "flat_region": [float(near["threshold"].min()), float(near["threshold"].max())],
        "flat_region_tolerance": tolerance,
        "flat_region_cost_spread": float(
            near["total_cost"].max() - near["total_cost"].min()
        ),
    }


def lgd_sensitivity(
    y, p, loan_amnt, forgone_interest, lgds=(0.4, 0.5, 0.622, 0.7, 0.8)
) -> pd.DataFrame:
    """How much does the operating point move with the loss-given-default
    assumption? LGD is the least certain input to the cost model, so the
    threshold's sensitivity to it bounds how much the decision can be trusted.
    """
    rows = []
    for lgd in lgds:
        curve = cost_curve(y, p, np.asarray(loan_amnt) * lgd, forgone_interest)
        sel = select_threshold(curve)
        rows.append(
            {
                "lgd": lgd,
                "threshold": sel["threshold"],
                "approval_rate": sel["approval_rate"],
                "flat_region_low": sel["flat_region"][0],
                "flat_region_high": sel["flat_region"][1],
                "savings_vs_approve_all": sel["savings_vs_approve_all"],
            }
        )
    return pd.DataFrame(rows)


def confusion_at(y, p, threshold: float) -> dict:
    y = np.asarray(y).astype(bool)
    approved = np.asarray(p) < threshold
    return {
        "approved": int(approved.sum()),
        "declined": int((~approved).sum()),
        "approved_defaults": int((approved & y).sum()),
        "declined_goods": int((~approved & ~y).sum()),
        "default_rate_in_book": float(y[approved].mean()) if approved.any() else float("nan"),
    }
