"""Model hyperparameters and the shared split, so every script trains and
evaluates on exactly the same partition."""
from __future__ import annotations

import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import config as C

LGB_PARAMS = dict(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.03,
    num_leaves=50,
    min_child_samples=20,
    colsample_bytree=0.8,
    subsample=0.8,
    subsample_freq=1,
    random_state=C.RANDOM_SEED,
    n_jobs=-1,
    verbose=-1,
)


def split(df):
    """70/15/15 stratified split, fixed seed."""
    train, temp = train_test_split(
        df, test_size=0.30, random_state=C.RANDOM_SEED, stratify=df["target"]
    )
    val, test = train_test_split(
        temp, test_size=0.50, random_state=C.RANDOM_SEED, stratify=temp["target"]
    )
    return train, val, test


def fit_lgbm(spec, train, val, **overrides):
    model = lgb.LGBMClassifier(**{**LGB_PARAMS, **overrides})
    model.fit(
        spec.transform(train), train["target"],
        eval_set=[(spec.transform(val), val["target"])],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return model


def fit_logreg(spec, train):
    """Median imputation + scaling.

    The original notebook fed raw unscaled features to lbfgs, which did not
    converge -- that made the linear baseline artificially weak and the tree
    model look better by comparison than it actually is.
    """
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=C.RANDOM_SEED),
    )
    model.fit(spec.transform(train), train["target"])
    return model
