"""Percentage-point translations of within-company effects.

- `logit_to_pp_marginal`: coef * p(1-p) * 100, the derivative of the logistic at the base rate
  (points of bad-rating probability per within-company SD; used for the common-scale figure).
- `logit_to_pp_shift`: 100 * (logistic(logit(p) + coef) - p), the change in probability when the base
  rate is moved by one within-company SD (used for the overlap table).
- `within_company_lpm`: company-demeaned linear probability model with company-clustered SEs.
- `position_translation`: re-expresses blow-up position/severity effects as bad-rating percentages
  (raw rates by fifth, per-SD effect, last-vs-first fifth adjusted for severity and controls).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm


def logit_to_pp_marginal(coef: float, base: float) -> float:
    return float(coef * base * (1 - base) * 100)


def logit_to_pp_shift(coef: float, base: float) -> float:
    lo = np.log(base / (1 - base)) + coef
    return float(100 * (1 / (1 + np.exp(-lo)) - base))


def within_company_lpm(df: pd.DataFrame, outcome: str, regressors: list[str],
                       cluster_col: str = "company_id"):
    """OLS on company-demeaned outcome and regressors; returns the statsmodels result (clustered SEs)."""
    cols = list(dict.fromkeys(regressors))
    d = df[[outcome, cluster_col] + cols].dropna().copy()
    dm = d[[outcome] + cols] - d.groupby(cluster_col)[[outcome] + cols].transform("mean")
    return sm.OLS(dm[outcome].astype(float), dm[cols].astype(float)).fit(
        cov_type="cluster", cov_kwds={"groups": d[cluster_col]})


def _pp(res, term: str) -> list[float]:
    b = res.params[term]
    lo, hi = res.conf_int().loc[term]
    return [round(float(100 * v), 1) for v in (b, lo, hi)]


def position_translation(df: pd.DataFrame, controls: list[str], outcome: str = "bottom",
                         cluster_col: str = "company_id") -> dict:
    """df: calls with a blow-up; needs rupture_position_fraction, n_ruptures, rupture_severity_sum,
    the outcome, company and controls. Returns bad-rating percentages and points per 100 calls."""
    df = df.copy()
    y = df[outcome].astype(float)
    out = {"n": len(df), "n_companies": int(df[cluster_col].nunique()), "base_pct": round(float(100 * y.mean()), 1)}
    pos = df["rupture_position_fraction"]
    df["pos5"] = np.minimum((pos * 5).astype(int), 4)
    out["pct_by_position_fifth"] = (100 * y.groupby(df["pos5"]).mean()).round(1).tolist()
    out["pct_by_n_blowups_1_2_3plus"] = (100 * y.groupby(np.minimum(df["n_ruptures"], 3)).mean()).round(1).tolist()
    sev5 = pd.qcut(df["rupture_severity_sum"].rank(method="first"), 5, labels=False)
    out["pct_by_severity_fifth"] = (100 * y.groupby(sev5).mean()).round(1).tolist()

    base_x = list(controls) + ["rupture_severity_sum"]
    df["pos_sd"] = (pos - pos.mean()) / pos.std()
    out["pp_per_sd_position"] = _pp(within_company_lpm(df, outcome, ["pos_sd"] + base_x, cluster_col), "pos_sd")
    df["last_fifth"] = (df["pos5"] == 4).astype(float)
    df["mid_fifths"] = df["pos5"].between(1, 3).astype(float)
    res = within_company_lpm(df, outcome, ["last_fifth", "mid_fifths"] + base_x, cluster_col)
    out["pp_last_vs_first_fifth"] = _pp(res, "last_fifth")
    return out
