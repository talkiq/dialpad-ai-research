"""Synthetic-data recovery tests for the within-company logit."""
import numpy as np
import pandas as pd

from csat_blowup.within_company import clears_sesoi, fit_cluster_robust_logit


def _synthetic_df(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for cid in range(50):
        cluster_effect = rng.normal(0, 1.0)
        n = rng.integers(20, 200)
        x_between = rng.normal(0, 1)
        for _ in range(n):
            x_within = rng.normal(0, 1)
            logit = -1.5 + 0.5 * x_within + 0.1 * x_between + 0.8 * cluster_effect
            y = rng.binomial(1, 1 / (1 + np.exp(-logit)))
            rows.append(dict(company_id=cid, x=x_between + x_within, y=y, dur=rng.normal(300, 50)))
    return pd.DataFrame(rows)


def test_within_coefficient_recovers_true_effect():
    res = fit_cluster_robust_logit(_synthetic_df(), outcome="y", feature="x", controls=["dur"])
    assert abs(res["within"]["coef"] - 0.5) < 0.1
    assert res["within"]["p"] < 0.001
    assert clears_sesoi(res)["within"] is True


def test_between_coefficient_not_spuriously_significant():
    # true between effect (0.1) is small relative to company-level noise (0.8 * cluster effect)
    res = fit_cluster_robust_logit(_synthetic_df(), outcome="y", feature="x", controls=["dur"])
    assert clears_sesoi(res)["between"] is False
