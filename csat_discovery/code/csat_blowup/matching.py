"""Matched look-alike comparison on the customer's upset turn.

1. Every call is an anchor; its neighbourhood is the anchor plus its k=10 nearest other calls *in the
   same vertical* by cosine similarity of the customer's upset-turn embedding (all-MiniLM-L6-v2 in the
   paper). Optionally neighbours from the anchor's own company are excluded (sensitivity).
2. Rows (anchor, member) are stacked; a call can appear in several neighbourhoods.
3. Linear probability model with neighbourhood fixed effects (within-neighbourhood demeaning) of
   bad rating on the z-scored exposure, plus a residual-customer adjustment: `cust_risk_z`, an
   out-of-fold (company-grouped) logistic risk score from the customer's own upset-turn embedding.
   Only neighbourhoods with outcome variation contribute.
4. 95% CIs and p-values from a cluster bootstrap over anchor companies, resampling per-company
   cross-product sums (so overlapping neighbourhoods inside a company move together).

Coefficients x 100 are points of bad rating per 100 calls per SD of the exposure.
`within_neighbourhood_auc` is the matching diagnostic: how well the customer risk score still separates
outcomes inside neighbourhoods (0.5 would mean matching fully held the customer state fixed).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

K = 10
N_BOOT = 2000


def zscore(x) -> np.ndarray:
    x = np.asarray(x, float)
    return (x - np.nanmean(x)) / np.nanstd(x, ddof=1)


def oof_risk_score(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int = 5) -> np.ndarray:
    """Out-of-fold P(bad) from embeddings; folds grouped by company."""
    out = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
        clf.fit(sc.transform(X[tr]), y[tr])
        out[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return out


def knn_within_vertical(d: pd.DataFrame, emb: np.ndarray, k: int = K, exclude_same_company: bool = False,
                        anchors: np.ndarray | None = None, block: int = 2048) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """{anchor_row: (neighbour_rows[k], cosine_sims[k])}. `d` row positions index `emb`.
    `anchors` (row positions) defaults to all rows."""
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True).clip(1e-12)
    comp = d["company_id"].to_numpy()
    vert = d["vertical"].to_numpy()
    is_anchor = np.ones(len(d), bool) if anchors is None else np.isin(np.arange(len(d)), anchors)
    res = {}
    for v in np.unique(vert):
        idx = np.where(vert == v)[0]
        a_all = idx[is_anchor[idx]]
        if len(idx) <= k or len(a_all) == 0:
            continue
        for st in range(0, len(a_all), block):
            a = a_all[st:st + block]
            S = emb[a] @ emb[idx].T
            for r, ai in enumerate(a):
                s = S[r].copy()
                s[idx == ai] = -np.inf
                if exclude_same_company:
                    s[comp[idx] == comp[ai]] = -np.inf
                if np.isfinite(s).sum() < k:
                    continue
                top = np.argpartition(-s, k)[:k]
                top = top[np.argsort(-s[top])]
                res[int(ai)] = (idx[top], s[top])
    return res


def stack(d: pd.DataFrame, nb: dict, y_col: str = "bottom", include_anchor: bool = True) -> pd.DataFrame:
    """Long table: nid (anchor row), comp (anchor company), row (member row), y."""
    comp = d["company_id"].to_numpy()
    recs = [(a, comp[a], m) for a, (nbrs, _) in nb.items()
            for m in (np.append(nbrs, a) if include_anchor else nbrs)]
    rows = pd.DataFrame(recs, columns=["nid", "comp", "row"])
    rows["y"] = d[y_col].to_numpy()[rows["row"].to_numpy()]
    return rows


def _summ(est: float, draws: np.ndarray) -> dict:
    bd = draws[~np.isnan(draws)]
    return {"beta": float(est), "lo": float(np.percentile(bd, 2.5)), "hi": float(np.percentile(bd, 97.5)),
            "p": float(max(1 / len(bd), np.mean(np.abs(bd - est) >= abs(est)))), "se": float(bd.std())}


def fe_lpm(rows: pd.DataFrame, xcols: list[str], rng: np.random.Generator, n_boot: int = N_BOOT,
           min_neigh: int = 20) -> dict:
    """Neighbourhood-FE linear probability model with anchor-company cluster bootstrap."""
    r = rows.dropna(subset=xcols + ["y"])
    r = r[r.groupby("nid")["y"].transform("std") > 0]
    if r["nid"].nunique() < min_neigh:
        return {"error": "too few neighbourhoods with outcome variation", "n_rows": len(r)}
    cols = ["y"] + xcols
    dm = (r[cols] - r.groupby("nid")[cols].transform("mean")).to_numpy(float)
    y, X = dm[:, 0], dm[:, 1:]
    comps, inv = np.unique(r["comp"].to_numpy(), return_inverse=True)
    p = len(xcols)
    XX = np.zeros((len(comps), p, p))
    Xy = np.zeros((len(comps), p))
    np.add.at(XX, inv, X[:, :, None] * X[:, None, :])
    np.add.at(Xy, inv, X * y[:, None])
    if np.linalg.matrix_rank(XX.sum(0)) < p:
        return {"error": "a regressor has no within-neighbourhood variation", "n_rows": len(r)}
    beta = np.linalg.solve(XX.sum(0), Xy.sum(0))
    B = np.full((n_boot, p), np.nan)
    for b in range(n_boot):
        pick = rng.integers(0, len(comps), len(comps))
        try:
            B[b] = np.linalg.solve(XX[pick].sum(0), Xy[pick].sum(0))
        except np.linalg.LinAlgError:
            pass
    out = {"n_rows": int(len(r)), "n_neigh": int(r["nid"].nunique()), "n_anchor_companies": int(len(comps)),
           "n_calls": int(r["row"].nunique())}
    for j, c in enumerate(xcols):
        out[c] = _summ(beta[j], B[:, j])
    return out


def within_neighbourhood_auc(rows: pd.DataFrame, score_col: str) -> float:
    r = rows[rows.groupby("nid")["y"].transform("std") > 0]
    s = r[score_col] - r.groupby("nid")[score_col].transform("mean")
    return float(roc_auc_score(r["y"], s))


def matched_effects(d: pd.DataFrame, cust_emb: np.ndarray, exposures: list[str], k: int = K,
                    exclude_same_company: bool = False, extra_controls: list[str] | None = None,
                    n_boot: int = N_BOOT, seed: int = 0) -> dict:
    """End-to-end matched comparison.

    d: one row per call with a blow-up; columns company_id, vertical, bottom (0/1) and the exposure
       columns. cust_emb: customer upset-turn embeddings aligned with d's rows.
    For each exposure x: M0 = company-FE LPM (unmatched reference), M1 = neighbourhood FE on z(x),
    M2 = M1 + cust_risk_z, [M3 = M2 + extra_controls]. Betas are in points per 100 calls per SD of x.
    """
    rng = np.random.default_rng(seed)
    d = d.reset_index(drop=True).copy()
    d["cust_risk_z"] = zscore(oof_risk_score(cust_emb, d["bottom"].to_numpy(), d["company_id"].to_numpy()))
    nb = knn_within_vertical(d, cust_emb, k=k, exclude_same_company=exclude_same_company)
    rows = stack(d, nb)
    ref = pd.DataFrame({"nid": d["company_id"], "comp": d["company_id"], "row": np.arange(len(d)), "y": d["bottom"]})
    sims = np.array([v[1] for v in nb.values()])
    out = {"n_calls": len(d), "n_neighbourhoods": len(nb), "mean_cosine_distance": float((1 - sims).mean()),
           "exposures": {}}
    controls = ["cust_risk_z"] + list(extra_controls or [])
    for c in controls[1:]:
        d[c] = zscore(d[c])
    rows = rows.assign(**{c: d[c].to_numpy()[rows["row"]] for c in controls})
    out["within_neighbourhood_auc_cust_risk"] = within_neighbourhood_auc(rows, "cust_risk_z")
    for x in exposures:
        zx = zscore(d[x])
        rr = rows.assign(zx=zx[rows["row"]])
        specs = {"M0_company_FE": (ref.assign(zx=zx), ["zx"]), "M1_matched": (rr, ["zx"]),
                 "M2_matched_plus_cust_risk": (rr, ["zx", "cust_risk_z"])}
        if extra_controls:
            specs["M3_plus_controls"] = (rr, ["zx"] + controls)
        res = {}
        for name, (frame, cols) in specs.items():
            f = fe_lpm(frame, cols, rng, n_boot=n_boot)
            res[name] = ({kk: 100 * v if kk in ("beta", "lo", "hi", "se") else v for kk, v in f["zx"].items()}
                         if "zx" in f else f)
        out["exposures"][x] = res
    return out
