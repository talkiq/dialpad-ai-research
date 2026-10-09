"""Initial-response classifier (H78 in the paper's supplement) and the difference-of-vectors check.

Input: the embedding of the agent's initial response (agent's next ~two turns after the first upset
turn; see `detector.response_window`), a 0/1 bad-rating label and company ids.

Classifier: StandardScaler + L2 logistic regression (C=1, class_weight="balanced").
- `cv_auc`: company-grouped K-fold AUC (no company in both train and test).
- `permutation_null`: the same CV AUC after permuting labels *within* company (keeps company base
  rates), giving a null mean/SD, a z-score and a one-sided p.
- `company_disjoint_test`: fit on one set of companies, apply unchanged to a disjoint set, report AUC
  and the observed bad-rating % in fifths of predicted risk.

Difference-of-vectors check ("how much is the agent's?"), optional:
- `projection_residual`: agent vector minus its projection on the customer's upset-turn vector.
- `residual_check`: company-grouped CV AUC of customer-only, customer+agent, agent-only, projection
  residual, and a full linear residual (agent minus a ridge prediction from the customer vector, ridge
  refit inside each training fold so the residualizer never sees test companies).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge, RidgeCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RIDGE_ALPHAS = np.logspace(-2, 3, 11)


def make_classifier(C: float = 1.0):
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=3000, class_weight="balanced", C=C))


def cv_auc(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int = 5, C: float = 1.0) -> dict:
    """Company-grouped CV. Returns per-fold AUCs, their mean, and the pooled out-of-fold AUC."""
    y = np.asarray(y)
    oof = np.full(len(y), np.nan)
    folds = []
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        p = make_classifier(C).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        oof[te] = p
        folds.append(roc_auc_score(y[te], p))
    ok = ~np.isnan(oof)
    return {"fold_aucs": folds, "mean_fold_auc": float(np.mean(folds)) if folds else np.nan,
            "pooled_auc": float(roc_auc_score(y[ok], oof[ok])) if ok.any() else np.nan, "oof": oof}


def permutation_null(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_perm: int = 50,
                     seed: int = 0, n_splits: int = 5) -> dict:
    """Within-company label permutation null for the mean fold AUC."""
    rng = np.random.default_rng(seed)
    observed = cv_auc(X, y, groups, n_splits)["mean_fold_auc"]
    d = pd.DataFrame({"y": np.asarray(y), "g": np.asarray(groups)})
    null = []
    for _ in range(n_perm):
        yp = d.groupby("g")["y"].transform(lambda s: rng.permutation(s.values)).to_numpy()
        m = cv_auc(X, yp, groups, n_splits)["mean_fold_auc"]
        if not np.isnan(m):
            null.append(m)
    null = np.array(null)
    sd = null.std()
    return {"observed": observed, "null_mean": float(null.mean()), "null_sd": float(sd),
            "z": float((observed - null.mean()) / sd) if sd > 0 else np.nan,
            "p_one_sided": float((1 + np.sum(null >= observed)) / (1 + len(null)))}


def risk_fifths(pred: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    """Observed bad-rating % in fifths of predicted risk (fifth 1 = lowest risk)."""
    q = pd.qcut(pd.Series(pred).rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    t = pd.DataFrame({"fifth": q, "y": np.asarray(y, float)})
    return t.groupby("fifth", observed=True)["y"].agg(n="size", bad_pct=lambda s: 100 * s.mean()).reset_index()


def company_disjoint_test(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray,
                          C: float = 1.0) -> dict:
    """Train on one company set, apply unchanged to another. Caller guarantees the sets are disjoint."""
    model = make_classifier(C).fit(X_train, y_train)
    pred = model.predict_proba(X_test)[:, 1]
    return {"auc": float(roc_auc_score(y_test, pred)), "base_pct": float(100 * np.mean(y_test)),
            "fifths": risk_fifths(pred, y_test), "model": model, "pred": pred}


# ---------------------------------------------------------------------------- residual check
def projection_residual(e_agent: np.ndarray, e_cust: np.ndarray) -> np.ndarray:
    """Row-wise: e_a - (<e_a, e_c> / <e_c, e_c>) e_c."""
    coef = (e_agent * e_cust).sum(1, keepdims=True) / (e_cust * e_cust).sum(1, keepdims=True)
    return e_agent - coef * e_cust


def _foldwise_ridge_residual_oof(e_agent, e_cust, y, groups, n_splits, alpha):
    oof = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=n_splits).split(e_cust, y, groups):
        ridge = (Ridge(alpha=alpha) if alpha else RidgeCV(alphas=RIDGE_ALPHAS)).fit(e_cust[tr], e_agent[tr])
        m = make_classifier().fit(e_agent[tr] - ridge.predict(e_cust[tr]), y[tr])
        oof[te] = m.predict_proba(e_agent[te] - ridge.predict(e_cust[te]))[:, 1]
    return oof


def residual_check(e_agent: np.ndarray, e_cust: np.ndarray, y: np.ndarray, groups: np.ndarray,
                   n_splits: int = 5, ridge_alpha: float | None = None) -> dict:
    """Pooled out-of-fold AUCs for the customer/agent/residual feature sets (see module docstring)."""
    y = np.asarray(y)
    r1 = projection_residual(e_agent, e_cust)
    sets = {"customer": e_cust, "customer+agent": np.hstack([e_cust, e_agent]), "agent": e_agent,
            "projection_residual": r1}
    out = {k: cv_auc(X, y, groups, n_splits)["pooled_auc"] for k, X in sets.items()}
    oof = _foldwise_ridge_residual_oof(e_agent, e_cust, y, groups, n_splits, ridge_alpha)
    out["ridge_residual_foldwise"] = float(roc_auc_score(y, oof))
    out["frac_sq_norm_removed_by_projection"] = float(1 - (r1 ** 2).sum(1).mean() / (e_agent ** 2).sum(1).mean())
    out["increment_over_customer"] = out["customer+agent"] - out["customer"]
    return out
