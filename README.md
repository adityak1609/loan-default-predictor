# LoanGuard

Credit default risk modelling on the LendingClub 2007–2018 extract (1.35M resolved loans).

The interesting results here are not the AUC. They are what happens when you ask
three questions the headline metric hides: *how much does the model add over the
lender's existing scorecard, are the probabilities it emits real probabilities,
and does the out-of-time drop mean what it appears to mean.*

---

## Findings

### 1. A single feature gets most of the way there

`grade` and `int_rate` are LendingClub's own risk-model output, not borrower
attributes. Any model trained on them is partly re-learning an existing
scorecard, so the headline number is uninterpretable on its own.

| Feature set | Features | Test AUC | Δ vs grade-only |
|---|---|---|---|
| `grade_only` (logistic regression) | 1 | 0.6794 | — |
| `incumbent_only` (grade + sub_grade + int_rate) | 3 | 0.6968 | +0.0174 |
| **`applicant_only` (no incumbent signal)** | 43 | **0.7182** | **+0.0388** |
| `full` | 46 | 0.7243 | +0.0449 |
| `full` — scaled logistic regression | 46 | 0.7101 | +0.0307 |

Two things fall out. Borrower attributes alone (0.7182) beat LendingClub's entire
grading system (0.6968), so the model is not merely echoing the incumbent — it
substitutes for it. And adding grade on top of borrower attributes buys only
+0.006, meaning the incumbent scorecard carries almost no information the raw
application does not already contain.

The gradient-boosted model beats a *properly scaled* linear baseline by 0.014 AUC.
The original notebook reported a much larger gap, but its logistic regression was
fed unscaled features and never converged.

Reproduce: `python scripts/run_baselines.py` → `reports/baselines.csv`

### 2. The probabilities were inflated ~3x by class weighting

The earlier model set `scale_pos_weight` to the class ratio (4.01), which is a
ranking device: it deliberately biases predicted probabilities upward. The
Streamlit app then rendered that output as "probability of default".

| Model | AUC | Brier | ECE | Mean predicted | Observed |
|---|---|---|---|---|---|
| weighted (previous) | 0.7077 | 0.1557 | 0.0759 | 25.1% | 20.0% |
| unweighted | 0.7243 | 0.1427 | 0.0020 | 20.0% | 20.0% |
| unweighted + isotonic | 0.7242 | 0.1428 | 0.0019 | 20.0% | 20.0% |

The honest result is that isotonic regression was **not needed**. LightGBM
optimising log-loss is already well calibrated (ECE 0.0020); the entire
miscalibration was self-inflicted by the class weighting, which also cost 0.017
AUC. The calibrator is retained in the serving path because it is near-identity
here and provides a hook if the base rate shifts, but claiming it fixed the
problem would be wrong — removing `scale_pos_weight` did.

Reproduce: `python scripts/train_calibrate.py` → `reports/calibration.csv`

### 3. The cost-optimal threshold is flat, and dominated by an assumption

Loss given default, estimated empirically from charged-off loans, is **0.622**.
Against a mean forgone-interest cost of $4,274, that puts the false-negative to
false-positive ratio at only **2.1:1** — not the 10:1 that lending intuition
suggests, because these are high-interest unsecured loans where rejecting a good
borrower is genuinely expensive.

At the selected threshold (0.51, the midpoint of the near-optimal band) on the
test set: 96.4% approval rate, $1,893 expected loss per application, **$9.37M
saved** versus approving everyone across 201,803 applications.

But the curve is flat to within 1% across thresholds **0.41–0.61**. Reporting a
single argmin would be false precision — the validation argmin (0.44) and the
test argmin (0.48) differ by pure sampling noise. The threshold is reported as a
band for that reason.

What actually moves the decision is the LGD assumption, not the model:

| LGD | Threshold | Approval rate | Savings vs approve-all |
|---|---|---|---|
| 0.40 | 0.62 | 99.5% | $0.6M |
| 0.50 | 0.53 | 98.4% | $2.9M |
| **0.622 (empirical)** | **0.48** | **96.4%** | **$9.4M** |
| 0.70 | 0.41 | 92.0% | $16.8M |
| 0.80 | 0.36 | 87.4% | $34.3M |

Any effort spent tightening the model is worth less than getting LGD right.

Reproduce: `reports/decision.json`, `reports/lgd_sensitivity.csv`

### 4. The out-of-time drop is a sampling artefact, not distribution shift

Training on ≤2014 and testing on 2016–2018 gives AUC 0.7359 → 0.7121, a drop of
**0.0238**. The tempting conclusion is model decay.

It isn't. The snapshot ends 2018Q4, so **no loan issued from 2016 onward has
reached the end of its term** — `matured_share` is exactly 0.0000 for 2016, 2017
and 2018. Filtering to resolved statuses therefore selects only loans that
terminated *early*, and early termination is heavily skewed toward default.

2014 and 2015 contain both matured and un-matured resolved loans, so the same
model scored on each subset isolates the effect:

| Year | Matured | n | Default rate | AUC |
|---|---|---|---|---|
| 2014 | no | 60,533 | 31.1% | 0.6950 |
| 2014 | yes | 162,570 | 13.7% | 0.7230 |
| 2015 | no | 92,520 | 36.4% | 0.6751 |
| 2015 | yes | 283,026 | 14.9% | 0.6970 |

Un-matured loans default at 2.3x the rate of matured ones, and the model scores
**0.0250 AUC worse** on them — within the same origination years, where there is
no distribution shift at all.

That 0.0250 maturity penalty accounts for essentially the entire 0.0238
out-of-time drop. **There is no evidence of genuine temporal degradation in this
data; the apparent decay is survivorship.**

*Limitation:* this snapshot cannot fully resolve the question. The correct fix is
a fixed observation window — relabel the target as "defaulted within 12 months of
origination", which makes every vintage comparable and allows currently-`Current`
loans to be included as negatives rather than dropped. That requires
`last_pymnt_d` to date resolutions and is the intended next step.

Reproduce: `python scripts/temporal_validation.py` → `reports/maturity_confound.csv`

---

## What changed from the original pipeline

| | Before | After |
|---|---|---|
| Rows | 1,220,127 | 1,345,350 (`dropna` removed — LightGBM handles NaN, and `emp_length` missingness correlates with the target) |
| Baseline | none | grade-only, incumbent-only, applicant-only ablations |
| Linear baseline | unscaled, non-converged | imputed + scaled pipeline |
| Threshold | 0.5, implicit in `predict()` | expected-loss minimisation with per-loan costs |
| Headline metric | F1 = 0.427 at p=0.5 | cost per application, approval rate, calibration |
| Calibration | unmeasured, ~3x inflated | ECE 0.0020, reported before/after |
| Validation | random split only | random + out-of-time + maturity decomposition |
| Serving | column list retyped by hand in the app | shared `FeatureSpec` artefact |
| Features | encoding only | leverage, payment burden, credit history, log income |

Two live bugs in the Streamlit app were fixed along the way: it displayed
uncalibrated scores as probabilities, and its SHAP waterfall used
`expected_value[0]`, which explains the **repayment** class — every contribution
was shown with the wrong sign.

---

## Layout

```
src/loanguard/     config, data prep, features, evaluation
scripts/           prepare_data · run_baselines · train_calibrate · temporal_validation
reports/           all generated results (tracked in git)
notebooks/legacy/  original exploratory notebooks, superseded
streamlit_app.py   serving UI
```

## Running it

```bash
pip install -r requirements.txt

# Download accepted_2007_to_2018Q4.csv (~1.6 GB) into the repo root:
# https://www.kaggle.com/datasets/wordsforthewise/lending-club

python scripts/prepare_data.py        # ~3 min, writes processed/loans.parquet
python scripts/run_baselines.py
python scripts/train_calibrate.py     # writes serving artefacts
python scripts/temporal_validation.py

streamlit run streamlit_app.py
```

Seed is fixed at 42 throughout (`src/loanguard/config.py`).

## Known limitations

- **Maturity bias is measured, not removed.** See finding 4. The 12-month
  observation window is the fix.
- **LGD is a portfolio average.** It almost certainly varies by grade and term;
  a per-segment estimate would sharpen the cost model more than any modelling
  change (see the sensitivity table).
- **`cost_fp` assumes a declined applicant is a total loss of interest.** In
  practice they may be re-priced rather than declined outright.
- **No fairness audit.** `addr_state` and `zip_code` are available and the model
  is not tested for disparate impact — required before this could be taken
  seriously as an underwriting tool.
