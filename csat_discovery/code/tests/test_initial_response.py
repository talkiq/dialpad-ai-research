import numpy as np

from csat_blowup.effects import logit_to_pp_marginal, logit_to_pp_shift
from csat_blowup.initial_response import company_disjoint_test, cv_auc, projection_residual, risk_fifths


def test_projection_residual_is_orthogonal():
    rng = np.random.default_rng(0)
    a, c = rng.normal(size=(50, 8)), rng.normal(size=(50, 8))
    r = projection_residual(a, c)
    assert np.allclose((r * c).sum(1), 0, atol=1e-10)


def test_classifier_finds_planted_signal_on_disjoint_companies():
    rng = np.random.default_rng(0)
    n, dim = 1200, 20
    groups = rng.integers(0, 40, n)
    y = rng.binomial(1, 0.25, n)
    X = rng.normal(size=(n, dim))
    X[:, 0] += 1.0 * y
    assert cv_auc(X, y, groups)["pooled_auc"] > 0.65
    tr, te = groups < 20, groups >= 20
    res = company_disjoint_test(X[tr], y[tr], X[te], y[te])
    f = res["fifths"]
    assert res["auc"] > 0.65 and len(f) == 5 and f.bad_pct.iloc[-1] > f.bad_pct.iloc[0]
    assert risk_fifths(np.arange(10.0), np.r_[np.zeros(8), np.ones(2)]).bad_pct.tolist() == [0, 0, 0, 0, 100]


def test_pp_translations():
    assert abs(logit_to_pp_marginal(0.1, 0.2) - 1.6) < 1e-9
    assert abs(logit_to_pp_shift(0.0, 0.3)) < 1e-9
    assert 1.5 < logit_to_pp_shift(0.1, 0.2) < 1.7
