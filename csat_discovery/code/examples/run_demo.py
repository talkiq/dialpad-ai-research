"""End-to-end demo of every method on synthetic data (see make_synthetic.py; no real data involved).

    OMP_NUM_THREADS=1 python examples/run_demo.py [--encoder auto|minilm|hashing] [--n-calls 1600]

Company set A plays the role of the paper's Explore split and set B of the disjoint Confirm split.
Numbers printed here describe the synthetic generator only and say nothing about real calls.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

from make_synthetic import make_synthetic  # noqa: E402

from csat_blowup import detector, effects, initial_response, matching, overlaps  # noqa: E402
from csat_blowup.embeddings import get_encoder  # noqa: E402
from csat_blowup.schema import add_outcomes  # noqa: E402
from csat_blowup.within_company import clears_sesoi, fit_cluster_robust_logit  # noqa: E402

CONTROLS = ["log_duration", "n_legs", "hour", "dow"]
DCONTROLS = CONTROLS + ["n_ruptures", "rupture_severity_sum"]


def show_logit(label, res):
    if "within" not in res:
        print(f"  {label}: {res.get('error')}")
        return
    w = res["within"]
    print(f"  {label}: within OR/SD={w['or']:.2f} [{w['or_ci_lo']:.2f}, {w['or_ci_hi']:.2f}] p={w['p']:.2g} "
          f"clears SESOI={clears_sesoi(res)['within']}  (n={res['n']}, companies={res['n_clusters']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="auto", choices=["auto", "minilm", "hashing"])
    ap.add_argument("--n-calls", type=int, default=1600)
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--n-perm", type=int, default=5)
    a = ap.parse_args()

    turns, calls = make_synthetic(a.n_calls)
    calls = add_outcomes(calls)
    print(f"synthetic data: {len(turns)} turns, {len(calls)} calls, {calls.company_id.nunique()} companies")

    # 1. blow-up detector -------------------------------------------------------------------
    print("\n[1] Blow-up detector (v3 lexicon)")
    blow = detector.detect_blowups(turns)
    d = calls.merge(blow, on="call_id")
    d["has_rupture_int"] = d["has_rupture"].astype(int)
    for s in ("A", "B"):
        print(f"  set {s}:", detector.blowup_summary(calls[calls.company_set == s], blow))

    # 2. within-company logit -----------------------------------------------------------------
    print("\n[2] Within-company (Mundlak) cluster-robust logit, set A")
    A = d[d.company_set == "A"]
    show_logit("has blow-up -> bad rating", fit_cluster_robust_logit(A, "bottom", "has_rupture_int", CONTROLS))
    Ar = A[A.has_rupture]
    for f in ("n_ruptures", "rupture_severity_sum", "rupture_position_fraction"):
        show_logit(f"{f} (blow-up calls)", fit_cluster_robust_logit(Ar, "bottom", f, CONTROLS))

    # 3. percentage translation ---------------------------------------------------------------
    print("\n[3] Position in percentage points (blow-up calls, set A)")
    print(" ", effects.position_translation(Ar, CONTROLS))

    # 4. initial-response classifier ---------------------------------------------------------
    print("\n[4] Initial-response classifier")
    win = detector.response_window(turns, blow)
    win = win[win.agent_text.str.strip().str.len() > 0].merge(d, on="call_id")
    enc_name, enc = get_encoder(a.encoder)
    print(f"  encoder: {enc_name}; {len(win)} blow-up calls with an agent reply")
    e_agent = enc(win.agent_text.tolist())
    e_cust = enc(win.customer_text.tolist())
    y, g = win.bottom.to_numpy(), win.company_id.to_numpy()
    iA, iB = (win.company_set == "A").to_numpy(), (win.company_set == "B").to_numpy()
    cv = initial_response.cv_auc(e_agent[iA], y[iA], g[iA])
    print(f"  set A company-grouped CV AUC: mean fold {cv['mean_fold_auc']:.3f}, pooled {cv['pooled_auc']:.3f}")
    pn = initial_response.permutation_null(e_agent[iA], y[iA], g[iA], n_perm=a.n_perm)
    print(f"  within-company permutation null (n={a.n_perm}): mean {pn['null_mean']:.3f}, z={pn['z']:.1f}")
    tr = initial_response.company_disjoint_test(e_agent[iA], y[iA], e_agent[iB], y[iB])
    print(f"  train A -> test B: AUC {tr['auc']:.3f}, base {tr['base_pct']:.1f}%, bad % by risk fifth: "
          f"{tr['fifths'].bad_pct.round(1).tolist()}")
    rc = initial_response.residual_check(e_agent[iA], e_cust[iA], y[iA], g[iA])
    print("  difference-of-vectors check (set A):", {k: round(v, 3) for k, v in rc.items()})

    # 5. overlaps ------------------------------------------------------------------------------
    print("\n[5] Overlaps after the blow-up (set A), points of bad rating per SD (within-company logit)")
    ev = overlaps.overlap_events(turns)
    print("  event classes:", ev.groupby(["interrupter", "cls"]).size().to_dict())
    post = overlaps.post_anchor_rates(turns, win[["call_id", "anchor_turn_index", "anchor_start_s"]], ev)
    pa = post.merge(d, on="call_id")
    pa = pa[pa.company_set == "A"]
    base = pa.bottom.mean()
    for f in ("agent_coop_rate", "agent_coop_backchannel_rate", "agent_comp_rate", "agent_comp_share"):
        r = fit_cluster_robust_logit(pa, "bottom", f, DCONTROLS)
        if "within" in r:
            print(f"  {f}: {effects.logit_to_pp_shift(r['within']['coef'], base):+.1f} pts/SD "
                  f"(OR {r['within']['or']:.2f}, n={r['n']})")

    # 6. matched look-alike comparison ------------------------------------------------------------
    print("\n[6] Matched look-alikes (10-NN within vertical on the customer's upset turn), set A")
    expo = overlaps.window_listening_exposures(turns, win, ev)
    m = win.merge(expo[["call_id", "bc_rate", "u_bc"]], on="call_id")
    m["agent_reply_risk"] = matching.oof_risk_score(e_agent, y, g)
    keep = (m.company_set == "A").to_numpy()
    res = matching.matched_effects(m[keep], e_cust[keep], ["bc_rate", "u_bc", "agent_reply_risk"],
                                   n_boot=a.n_boot, seed=1)
    print(f"  {res['n_calls']} calls, {res['n_neighbourhoods']} neighbourhoods, mean cosine distance "
          f"{res['mean_cosine_distance']:.3f}; customer-risk AUC inside neighbourhoods "
          f"{res['within_neighbourhood_auc_cust_risk']:.3f}")
    for x, specs in res["exposures"].items():
        txt = "; ".join(f"{k}: {v['beta']:+.1f} [{v['lo']:+.1f}, {v['hi']:+.1f}]" if "beta" in v else f"{k}: {v.get('error')}"
                        for k, v in specs.items())
        print(f"  {x} (pts per SD): {txt}")
    print("\ndone")


if __name__ == "__main__":
    np.seterr(all="ignore")
    warnings.filterwarnings("ignore", message="Maximum Likelihood optimization failed")
    main()
