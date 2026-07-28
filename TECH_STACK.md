# LoanGuard — Tech Stack & Feature Guide (Beginner Edition)

This is the friendly version. It explains, in plain English:

1. **What each tool (library) in this project does, and why we picked it.** (Part A)
2. **What information the model looks at to decide if a loan is risky, and why each piece matters.** (Part B)

No prior knowledge assumed. Where a technical word shows up, it's explained the
first time.

**The 30-second summary of the whole project:** we take a big spreadsheet of past
loans, clean it up, teach a computer program to spot which loans went bad, check
carefully that it's actually any good, and then wrap it in a simple website where
you can type in a borrower's details and get a risk score.

---

# Part A — The Tools (Tech Stack)

## A.0 The big picture

Think of building this project like running a kitchen. Raw ingredients come in,
get prepped, get cooked, and get served on a plate. Each tool is one station in
that kitchen:

```
   RAW INGREDIENTS   →   PREP   →   COOK   →   PLATE UP
   (the loan data)      (clean)    (model)   (the website)
```

Here's the same idea with the actual tools:

```
              A giant spreadsheet of past loans (1.6 GB)
                              │
                    ┌─────────┴──────────┐
                    │  pandas + numpy    │   clean it up, do the math
                    │  pyarrow           │   save it in a fast-to-reload format
                    └─────────┬──────────┘
                              │
                    ┌─────────┴──────────┐
                    │  lightgbm          │   the "brain" that learns patterns
                    │  scikit-learn      │   splits data, scores results, helpers
                    └─────────┬──────────┘
                              │
                    ┌─────────┴──────────┐
                    │  streamlit         │   the website you click on
                    │  shap + matplotlib │   explains WHY it gave that score
                    └────────────────────┘
```

**One important habit:** every tool is "pinned" to an exact version in the
`requirements.txt` file (e.g. `pandas==2.2.2`). "Pinned" just means we froze the
version number. Why? Because our documentation quotes precise numbers (like
"AUC 0.7243"). If a tool silently updated itself, those numbers might shift
slightly and the docs would look wrong. Freezing versions means **anyone who runs
this project gets the exact same results we did.**

## A.1 The tools at a glance

| Tool | Think of it as… | What it does here |
|---|---|---|
| `pandas` | A super-powered Excel | Loads, cleans, and reshapes the loan data |
| `numpy` | A fast calculator for big lists of numbers | Does the heavy math (costs, scoring) |
| `pyarrow` | A fast "save" button | Stores the cleaned data so it reloads in seconds |
| `scikit-learn` | A toolbox of standard ML helpers | Splits data, scores the model, runs the simple baseline |
| `lightgbm` | The learning brain | The actual model that predicts loan risk |
| `shap` | An "explain yourself" tool | Shows why the model gave a particular score |
| `matplotlib` | A drawing program | Draws the explanation chart |
| `joblib` | A "save the trained brain to a file" tool | Stores the finished model so the website can use it |
| `streamlit` | A website builder for Python people | The clickable app |
| `mlflow` | An old lab notebook | Leftover from earlier work; not used anymore |

The rest of Part A explains each one a bit more, with the "why did we pick this
one?" answered simply.

---

## A.2 pandas — the super-powered spreadsheet

**What it does.** Everything before the model is about tidying data, and pandas is
the tool for that. Real-world data is messy. For example, in this dataset:

- Interest rates are stored as text like `"13.56%"` — pandas strips the `%` and
  turns it into the number `13.56`.
- Employment length is written as `"10+ years"` — pandas converts it to the
  number `10`.
- Dates are written like `"Aug-2001"` — pandas turns them into real dates it can
  do math with.

**Why pandas?** The data is big (over a million rows) but still small enough to
fit in a normal computer's memory. That's exactly what pandas is built for. Bigger
"cluster" tools (like Spark) would be overkill — like renting a moving truck to
carry a single suitcase.

**A neat trick:** the file is huge (1.6 GB) with 151 columns, but we only need
about 30 of them. pandas lets us read *just* those 30. That's the difference
between the program finishing in about 4 minutes versus crashing because it ran
out of memory.

**A deliberate choice — we keep the blanks.** Lots of projects delete any row with
missing information. We don't, for two reasons: (1) our model (LightGBM) can handle
blanks just fine, and (2) the blanks themselves are a *clue*. People without steady
jobs are more likely to leave "employment length" blank — and they're also more
likely to default. Deleting those rows would throw away useful signal.

## A.3 numpy — the fast number-cruncher

**What it does.** When we need to do the same math on millions of numbers at once
— like calculating how much money each wrong decision would cost — numpy does it
extremely fast.

**Why it matters here.** The star example is the "cost curve." We want to know: if
we set our approval bar at different strictness levels, how much money do we lose
each time? The slow way would be to recalculate everything from scratch for every
setting — that could take ages. numpy uses a clever shortcut (sorting the loans
once and adding up running totals) to answer for *all* settings in one fast pass.

Think of it as the difference between counting a jar of coins one at a time versus
running them all through a coin-counting machine.

## A.4 pyarrow — the fast save-and-reload

**What it does.** After we clean the data, we save it. pyarrow saves it in a format
called **Parquet** instead of a plain spreadsheet (CSV) file.

**Why Parquet instead of a regular spreadsheet?**
- **It remembers the cleanup.** A plain CSV forgets that `"13.56%"` was turned into
  `13.56` — you'd have to redo all that cleaning every time. Parquet saves the
  cleaned-up version, types and all.
- **It's fast.** Reloading over a million rows takes seconds. Since six different
  scripts each reload this data, that speed adds up.
- **It keeps everyone honest.** Because every script reads the exact same saved
  file, we know for certain they're all working with identical data — so comparing
  their results is fair.

## A.5 scikit-learn — the box of standard ML tools

scikit-learn (often written "sklearn") is a famous, trusted toolkit. We use it for
the supporting jobs *around* the main model — not the model itself. Four jobs:

1. **Splitting the data.** Before teaching a model, you split your data into three
   piles: one to *learn* from (train), one to *tune* on (validation), and one to
   *test* on that the model never saw. sklearn does this split, and we always use
   the same random "seed" (the number 42) so the split is identical every time.
   This is like always shuffling a deck the exact same way so experiments are
   repeatable.

2. **Scoring the results.** sklearn provides the trusted, standard measuring sticks
   (like "AUC" — a score from 0.5 to 1.0 for how well the model separates good
   loans from bad ones). We use its versions rather than writing our own, because
   these are the ones everyone trusts.

3. **The simple comparison model.** To prove our fancy model is worth it, we build
   a plain, simple model too (called logistic regression) and see if the fancy one
   actually beats it. sklearn builds that simple one.

4. **The "calibrator."** More on this below — it's a small adjuster that makes sure
   the model's percentages mean what they say.

**Why sklearn?** It's the industry-standard toolbox. Any reviewer instantly
recognizes what every piece does — no surprises.

## A.6 lightgbm — the brain of the operation

**What it does.** This is the actual model — the thing that learns "loans that look
like *this* tend to go bad." It's a type called a **gradient-boosted decision
tree**. Don't worry about the name; here's the intuition:

Imagine a huge game of "20 questions" — "Is the income below $30k? Is the credit
utilization above 80%?" Each question narrows down the risk. Now imagine hundreds
of these little question-trees, each one fixing the mistakes of the last, all
voting together. That's gradient boosting.

**Why this kind of model?**
- **It's the best fit for spreadsheet-style data.** For tables of mixed numbers and
  categories, this family of models is the reigning champion. A neural network
  (the tech behind image and language AI) would be more work for no benefit here.
- **It handles blanks by itself.** Remember we kept the missing values? This model
  learns what to do with them automatically. That's *why* we could keep them.
- **It's fast.** It trains on over a million rows in seconds.

**Why we didn't fine-tune it.** You *can* fiddle with a model's settings to squeeze
out a tiny bit more accuracy. We deliberately didn't, because we found that one
business assumption (how much money is lost when a loan defaults) affects the final
decision far more than any fine-tuning ever could. So we spent our energy there
instead. The point of this project is *careful evaluation*, not chasing a bigger
number.

## A.7 shap — the "why did you decide that?" tool

**What it does.** A risk score on its own isn't very useful — a loan officer needs
to know *why*. SHAP breaks down a single prediction and shows which details pushed
the risk **up** and which pushed it **down**. For example: "low income pushed risk
up, long credit history pushed it down."

**Why SHAP?** It's the well-respected, mathematically fair way to hand out
"credit" (or blame) to each piece of information for a single decision. And it has
a version made specifically for tree models like ours that's both exact and fast.

**Two tricky bits the code gets right** (both were bugs in an earlier version):
1. **Explaining the right thing.** The model actually tracks two mirror-image
   outcomes: "will repay" and "will default." An earlier version accidentally
   explained the "will repay" side, so every arrow pointed the wrong way — things
   that *increased* risk were drawn as *decreasing* it. Now it correctly explains
   the "will default" side.
2. **Being honest about the scale.** The explanation is on the model's internal
   scale, not the final tidied-up percentage. The app says so plainly rather than
   pretending they're identical.

## A.8 matplotlib — the drawing tool

**What it does.** It draws the SHAP explanation as a chart on the website.

**One setting worth knowing.** The code sets matplotlib to a "headless" mode
(`Agg`). A website runs on a server that has no screen. Normally a drawing tool
tries to pop open a window to show the picture — but on a screenless server that
would crash. Headless mode means "just draw the image to a file, don't try to
display it." Essential for a web app.

## A.9 joblib — the "save the trained brain" tool

**What it does.** Once the model has finished learning, we need to save it to a file
so the website can load it later without retraining. joblib does that saving.

**Why joblib?** A trained model is basically a big pile of numbers, and joblib is
built to save big piles of numbers efficiently — better than Python's general-
purpose save tool. Since the training program and the website use the exact same
tool versions (remember, everything is pinned), this simple approach is perfectly
safe.

## A.10 streamlit — the website

**What it does.** Streamlit turns a plain Python script into an interactive
website. `streamlit_app.py` is the whole app: type in a borrower's details on the
left, see the risk score and explanation on the right.

**Why Streamlit?** Normally, making a website means learning a second set of
web-only languages and tools. Streamlit lets us build the whole thing in Python —
the same language as the rest of the project. Fastest path from "working model" to
"something you can click on."

**Two clever details:**
- It **loads the model only once** and reuses it, instead of reloading the file
  every time you move a slider. (Streamlit re-runs the script on every click, so
  this shortcut matters.)
- It **calculates the monthly payment for you** from the loan amount, rate, and
  term, using the standard loan-payment formula — so you can't accidentally type a
  payment that doesn't match the loan.

## A.11 How the pieces snap together (the "contract")

This is the most important integration idea, so here it is simply.

When we finish training, we save **four files**:

| File | What's inside | Saved by |
|---|---|---|
| `lgbm_model.pkl` | the trained model | joblib |
| `calibrator.pkl` | the percentage-adjuster | joblib |
| `feature_spec.json` | **the exact list of inputs, in the exact order** | plain text (JSON) |
| `serving_config.json` | the approve/decline cutoff | plain text (JSON) |

Why does that third file matter so much? Imagine a machine that only works if you
feed it ingredients in a precise order. If the website hands the model its inputs
in the wrong order, or misspells one, the model quietly gives bad answers — no
error, just wrong. The old version of the app typed this input list out by hand,
which was exactly this kind of silent-bug risk. Now the website reads the input
list straight from the file the model was trained with, so the training side and
the website side can **never** disagree. That's why plain-text JSON files are a
real, important part of the toolkit — they carry the "instruction manual" that
keeps both sides in sync.

## A.12 mlflow — the leftover

You'll see an `mlflow.db` file in the project. mlflow is a tool for tracking
experiments — like a lab notebook. It's left over from the project's earlier days.
We replaced it with something simpler: **every result is just saved as a file in
the `reports/` folder** and tracked in the project's history. So mlflow is history
now; nothing active uses it.

---

# Part B — What the Model Looks At (Features)

A "feature" is just one piece of information about a borrower that the model uses
to make its guess — like income, or credit score. This model uses **46 features**.

**The golden rule for every feature:** it must be something you could know *before*
handing over the money — from the application form or a credit check. Anything you'd
only learn *after* the loan is running is banned (more on that in B.5).

The features come in four groups.

## B.1 The lender's own scorecard (3 features)

These three are special because they're **not really facts about the borrower** —
they're LendingClub's *own opinion* about the borrower, already baked in.

| Feature | Plain meaning |
|---|---|
| `grade` | LendingClub's letter rating, A (safest) to G (riskiest) |
| `sub_grade` | A finer version of the same rating (35 levels) |
| `int_rate` | The interest rate they charged — basically the price of that risk |

**Why flag these separately?** If our model uses LendingClub's grade, it's partly
just copying LendingClub's homework, not doing its own. So we always test the model
both *with* and *without* these. The cool finding: the model does *better* using
only real borrower facts than using LendingClub's grades — meaning it can stand on
its own, not just parrot the existing system.

(Small note: `grade` A–G is turned into numbers 1–7 in order, because A really is
safer than B is safer than C. The ordering carries meaning, so we keep it.)

## B.2 Facts from the application (14 features)

These are genuine borrower details, all known at application time. Grouped by what
they tell us:

**Can they afford it?**
- `loan_amnt` — how much they're asking for.
- `annual_inc` — their yearly income.
- `dti` — debt-to-income ratio: how much of their income already goes to debt.
- `installment` — the monthly payment this loan would require.

**How deep and healthy is their credit history?**
- `fico_range_low` — their credit score at application. One of the strongest signals.
- `emp_length` — how many years they've been employed (0 to 10+).
- `open_acc` / `total_acc` — how many credit accounts they have open / have ever had.
- `mort_acc` — how many mortgages they have (a sign of stability).

**How are they using their credit cards?**
- `revol_util` — what percent of their available credit they're using. Maxed-out
  cards are a warning sign.
- `revol_bal` — the dollar balance they're carrying.

**Any recent trouble?**
- `delinq_2yrs` — missed payments in the last 2 years.
- `inq_last_6mths` — how many times they applied for credit recently (a flurry can
  signal desperation).
- `pub_rec` — serious public records like bankruptcies.

## B.3 Categories (3 features)

These are labels rather than numbers, so we handle them differently:

| Feature | What it captures |
|---|---|
| `home_ownership` | Rent / own / mortgage — housing stability |
| `purpose` | Why they want the loan (debt consolidation, car, small business…) — a small-business loan behaves very differently from a car loan |
| `verification_status` | Whether their income was actually verified — tells us how much to trust the income number |

**Why handle these differently?** Unlike the A–G grade, these have no natural order
— "rent" isn't higher or lower than "own," they're just different. So instead of
numbering them, we make a yes/no column for each option (this is called
"one-hot encoding"). It's like a checklist with one box ticked.

## B.4 Features we built ourselves (6)

The raw data doesn't include everything useful, so we calculated six extra ones.
Combining two facts often says more than either alone:

| Feature | How it's calculated | Why it helps |
|---|---|---|
| `loan_to_income` | loan ÷ income | How big is the ask compared to what they earn? |
| `installment_to_monthly_income` | monthly payment ÷ monthly income | **How much of their paycheck this loan eats — the single most useful one we built.** |
| `revol_bal_to_income` | card balance ÷ income | Existing card debt, sized to income |
| `log_annual_inc` | a "squished" version of income | Some people report absurd incomes (millions). This squishing keeps the *simple* comparison model from being thrown off by them. Our main model doesn't need it, but it makes the comparison fair. |
| `credit_history_years` | how long they've had credit | Longer history usually means lower risk |
| `term_months` | 36 or 60 | Longer loans are riskier |

## B.5 The banned features (6) — and why they're banned

Some columns in the data are poison, and we deliberately keep them *out* of the
model. Why? Because they're things you only find out *after* the loan is already
running:

| Banned column | Why it's cheating |
|---|---|
| `last_fico_range_low/high` | Their credit score from a *recent* check — for a defaulted loan, that's measured *after* they defaulted |
| `last_pymnt_amnt` | The size of their final payment |
| `out_prncp` | How much they still owe |
| `total_pymnt`, `total_rec_int` | How much they ended up paying in total — this basically *is* the answer |

If you sneak these in, the model looks almost perfect (a score of 0.9995 instead of
0.7243) — but it's fake, like acing a test because you saw the answer key. That's
why some people publish suspiciously high scores on this dataset. We keep these
columns around only to *demonstrate* the trap, in a script called `leakage_demo.py`.

**One subtle point:** we *do* use one "after the loan" column, `last_pymnt_d` (the
date of the last payment) — but only to figure out *whether the loan went bad in
the first place* (to create the correct answer to learn from). We never feed it to
the model as a clue. Using the future to write down the correct answer is fine;
using it as a hint is cheating.

## B.6 The five "input recipes"

Finally, we bundle these features into five named sets so we can ask a specific
question with each:

| Recipe | What's in it | The question it answers |
|---|---|---|
| `grade_only` | just the grade | The bare minimum — what does the lender already know? |
| `incumbent_only` | grade + sub-grade + rate | LendingClub's whole scorecard |
| `applicant_only` | real borrower facts only (43) | **Can we judge risk without LendingClub's opinion?** |
| `full` | everything allowed (46) | The real, deployed model |
| `full_leaky` | full + the banned columns (52) | **Not a real model** — just to show the cheating trap |

By switching feature groups on and off, we can measure exactly how much each group
actually adds.

---

## Where to change things

| I want to change… | Look in this file |
|---|---|
| A tool's version | `requirements.txt` |
| Which columns get read, or the feature groups | `src/loanguard/config.py` |
| How a built feature is calculated | `src/loanguard/features.py` |
| The model's settings or the data split | `src/loanguard/model.py` |
| How results are scored | `src/loanguard/evaluate.py` |
| The website / explanations | `streamlit_app.py` |

For the deeper *reasoning* behind the design choices, see `TECHNICAL.md`. This
file is the beginner-friendly map; that one goes into the details.
