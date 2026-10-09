import numpy as np
import pandas as pd

from csat_blowup.matching import fe_lpm, knn_within_vertical


def test_knn_stays_within_vertical_and_can_exclude_company():
    rng = np.random.default_rng(0)
    d = pd.DataFrame({"company_id": rng.integers(0, 8, 200), "vertical": rng.choice(["v1", "v2"], 200)})
    emb = rng.normal(size=(200, 16))
    nb = knn_within_vertical(d, emb, k=5, exclude_same_company=True)
    assert len(nb) == 200
    for a, (nbrs, sims) in nb.items():
        assert len(nbrs) == 5 and a not in nbrs
        assert (d.vertical.values[nbrs] == d.vertical.values[a]).all()
        assert (d.company_id.values[nbrs] != d.company_id.values[a]).all()
        assert np.all(np.diff(sims) <= 1e-9)


def test_fe_lpm_recovers_slope_net_of_neighbourhood_effects():
    rng = np.random.default_rng(1)
    n_nb, size = 300, 11
    nid = np.repeat(np.arange(n_nb), size)
    level = np.repeat(rng.normal(0, 2, n_nb), size)       # confounding neighbourhood level
    x = rng.normal(size=n_nb * size) + level
    y = 0.05 * x - 0.05 * level + rng.normal(0, 0.1, n_nb * size)
    rows = pd.DataFrame({"nid": nid, "comp": nid % 40, "row": np.arange(len(x)), "y": y, "x": x})
    res = fe_lpm(rows, ["x"], np.random.default_rng(2), n_boot=200)
    assert abs(res["x"]["beta"] - 0.05) < 0.01
    assert res["x"]["lo"] < 0.05 < res["x"]["hi"]
