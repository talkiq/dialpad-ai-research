"""Within-company (Mundlak) cluster-robust logistic regression with per-within-SD odds ratios.

For a feature x and cluster (company), x_between = company mean of x and x_within = x - x_between are
entered as separate regressors. The within coefficient approximates a company-fixed-effects estimate
(not confounded by differences between companies); the between coefficient is the company-level
association. The feature is z-scored, then both Mundlak terms are z-scored again, so exp(coef) is an
odds ratio per within-company SD. Numeric controls are z-scored; non-numeric controls enter as
categoricals. Standard errors are clustered by company.

This is a frequentist approximation to a crossed-random-effects GLMM; it is what the paper reports.

An effect "clears the SESOI" when its whole 95% CI lies outside [1/1.10, 1.10] (the pre-specified
smallest effect size of interest, OR 1.10 per within-company SD).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

SESOI_OR = 1.10


def zscore(s: pd.Series) -> pd.Series:
    mu, sd = s.mean(), s.std(ddof=1)
    if sd == 0 or np.isnan(sd):
        return s * 0.0
    return (s - mu) / sd


def mundlak_decompose(df: pd.DataFrame, feature: str, cluster_col: str = "company_id") -> pd.DataFrame:
    """Adds `{feature}_between` (cluster mean) and `{feature}_within` (deviation) columns."""
    cluster_mean = df.groupby(cluster_col)[feature].transform("mean")
    out = df.copy()
    out[f"{feature}_between"] = cluster_mean
    out[f"{feature}_within"] = df[feature] - cluster_mean
    return out


def _formula(outcome: str, terms: list[str], work: pd.DataFrame) -> str:
    rhs = [t if work[t].dtype.kind in "iufb" else f"C({t})" for t in terms]
    return f"{outcome} ~ " + " + ".join(rhs)


def _term_summary(result, term: str) -> dict:
    coef, se = result.params[term], result.bse[term]
    return {"coef": float(coef), "or": float(np.exp(coef)),
            "or_ci_lo": float(np.exp(coef - 1.96 * se)), "or_ci_hi": float(np.exp(coef + 1.96 * se)),
            "p": float(result.pvalues[term])}


def _prepare(work: pd.DataFrame, features: list[str], controls: list[str], cluster_col: str):
    terms = []
    for f in features:
        work[f] = zscore(work[f].astype(float))
        work = mundlak_decompose(work, f, cluster_col)
        work[f"{f}_between"] = zscore(work[f"{f}_between"])
        work[f"{f}_within"] = zscore(work[f"{f}_within"].fillna(0.0))
        terms += [f"{f}_within", f"{f}_between"]
    for c in controls:
        if work[c].dtype.kind in "iuf":
            work[c] = zscore(work[c].astype(float))
    return work, terms + list(controls)


def fit_cluster_robust_logit(df: pd.DataFrame, outcome: str, feature: str, controls: list[str],
                             cluster_col: str = "company_id") -> dict:
    """logit(outcome) ~ feature_within + feature_between + controls, company-clustered SEs.

    Returns {"n", "n_clusters", "within": {coef, or, or_ci_lo, or_ci_hi, p}, "between": {...}} or
    {"error", "n"}.
    """
    work = df.dropna(subset=[outcome, feature, cluster_col] + list(controls)).copy()
    if work[cluster_col].nunique() < 2 or work[outcome].nunique() < 2:
        return {"error": "insufficient variation", "n": len(work)}
    work, terms = _prepare(work, [feature], controls, cluster_col)
    try:
        result = smf.logit(_formula(outcome, terms, work), data=work).fit(
            disp=0, cov_type="cluster", cov_kwds={"groups": work[cluster_col]})
    except Exception as e:  # singular fits, perfect separation
        return {"error": str(e), "n": len(work)}
    out = {"n": len(work), "n_clusters": int(work[cluster_col].nunique())}
    for label in ("within", "between"):
        term = f"{feature}_{label}"
        if term in result.params.index:
            out[label] = _term_summary(result, term)
    return out


def fit_cluster_robust_logit_multi(df: pd.DataFrame, outcome: str, features: list[str], controls: list[str],
                                   cluster_col: str = "company_id") -> dict:
    """Several features in one model (each Mundlak-decomposed). Keys: f"{feature}__within"/"__between"."""
    work = df.dropna(subset=[outcome, cluster_col] + list(features) + list(controls)).copy()
    if work[cluster_col].nunique() < 2 or work[outcome].nunique() < 2:
        return {"error": "insufficient variation", "n": len(work)}
    work, terms = _prepare(work, list(features), controls, cluster_col)
    try:
        result = smf.logit(_formula(outcome, terms, work), data=work).fit(
            disp=0, cov_type="cluster", cov_kwds={"groups": work[cluster_col]})
    except Exception as e:
        return {"error": str(e), "n": len(work)}
    out = {"n": len(work), "n_clusters": int(work[cluster_col].nunique())}
    for f in features:
        for label in ("within", "between"):
            term = f"{f}_{label}"
            if term in result.params.index:
                out[f"{f}__{label}"] = _term_summary(result, term)
    return out


def clears_sesoi(result: dict, sesoi_or: float = SESOI_OR) -> dict:
    """{"within": bool, "between": bool}: True when the 95% CI lies entirely outside [1/sesoi, sesoi]."""
    verdict = {}
    for label in ("within", "between"):
        if label in result:
            lo, hi = result[label]["or_ci_lo"], result[label]["or_ci_hi"]
            verdict[label] = bool((lo > sesoi_or) or (hi < 1.0 / sesoi_or))
    return verdict
