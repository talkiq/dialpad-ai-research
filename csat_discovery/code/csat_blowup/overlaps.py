"""Overlap rules: backchannels, cooperative overlaps and floor-taking (competitive) cut-ins.

An overlap event is a turn B that starts before the previous turn A (by turn_index) has ended. The
speaker of B is the "interrupter". Each event gets exactly one class (rules applied in order):

    x_resumption      A was itself a short affirmation, or A had started in overlap (B is A's
                      original speaker resuming, not an interruption)
    coop_backchannel  B is a short affirmation: <=4 words, entirely from the backchannel lexicon,
                      shorter than 3 s ("mm-hmm", "got it", "okay sure")
    coop_terminal     B starts within the last 1 s of A (end-of-turn anticipation)
    coop_nonfloor     A keeps the floor, B <=4 words, B does not open with a floor-claim word or an
                      affirmation
    x_ack_then_talk   B opens with an affirmation and continues
    x_other           A had <5 words
    comp_attempt      A keeps the floor (failed cut-in)
    x_other           B is a single non-claim word
    comp_floor        everything else: a floor-taking cut-in

"A keeps the floor" = A ends at/after B's end, or the turn after B starts before B ends. Events in the
final 10 s of the call are excluded from all rates (goodbye overlaps). `redirect` flags floor-taking
cut-ins that open with a redirection ("no", "wait", "hold on", "let me", ...).

Rates are events per minute of conversation (or per minute after the blow-up anchor), winsorized at
the 1st/99th percentile. comp_share = floor-taking cut-ins / (cooperative overlaps + cut-ins), defined
when the denominator is >= 3: the "ratio of cut-ins to listening signals".
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .schema import check_turns

BACKCHANNEL_WORDS = [
    "okay", "ok", "okey", "mm-hmm", "mhm", "mm", "mmhmm", "hmm", "uh-huh", "yeah", "yes", "yep",
    "yup", "yea", "oh", "alright", "alrighty", "right", "sure", "correct", "perfect", "gotcha",
    "good", "great", "awesome", "cool", "absolutely", "exactly", "definitely", "fine", "wow",
    "really", "thanks", "true", "totally", "nice", "excellent", "wonderful", "understood",
    "ma'am", "sir", "si", "sí", "ajá", "bye", "bye-bye"]
BACKCHANNEL_PHRASES = [
    ("thank", "you"), ("i", "see"), ("got", "it"), ("of", "course"), ("no", "problem"),
    ("no", "worries"), ("that's", "fine"), ("that's", "okay"), ("that's", "correct"),
    ("that's", "right"), ("sounds", "good"), ("i", "understand"), ("so", "much"),
    ("very", "much"), ("go", "ahead"), ("you're", "welcome"), ("you", "too")]
CLAIM_WORDS = ["no", "nope", "but", "wait", "actually", "sorry", "excuse", "hold", "hang", "listen", "well",
               "um", "uh", "so", "and", "what", "huh", "hey", "hello"]
REDIRECT_RE = re.compile(r"^(no|nope|but|wait|actually|sorry|excuse me|hold on|hang on|listen|well|let me|"
                         r"i already|that's not)\b|^\S+ (no|but|wait|actually|sorry|hold on|hang on)\b")
CLASSES = ["coop_backchannel", "coop_terminal", "coop_nonfloor", "comp_attempt", "comp_floor",
           "x_ack_then_talk", "x_resumption", "x_other"]
COOP = ["coop_backchannel", "coop_terminal", "coop_nonfloor"]

_LEX = set(BACKCHANNEL_WORDS)
_PH = set(BACKCHANNEL_PHRASES)
_STRIP = '.,!?;:"`'
_FIRST_WORD = re.compile(r"^[^a-z']*([a-z']+)")


def affirmation_prefix(text: str) -> tuple[int, int]:
    """(k, n): the longest prefix of the n words that can be tiled by backchannel words/phrases."""
    if not isinstance(text, str):
        return 0, 0
    w = [x.strip(_STRIP) for x in text.lower().split()]
    w = [x for x in w if x]
    n = len(w)
    dp = [True] + [False] * n
    for i in range(1, n + 1):
        if dp[i - 1] and w[i - 1] in _LEX:
            dp[i] = True
        if not dp[i] and i >= 2 and dp[i - 2] and (w[i - 2], w[i - 1]) in _PH:
            dp[i] = True
    k = max(j for j in range(n + 1) if dp[j])
    return k, n


def is_backchannel(text: str, duration_s: float | None = None) -> bool:
    k, n = affirmation_prefix(text)
    return n > 0 and n <= 4 and k == n and (duration_s is None or duration_s < 3.0)


def overlap_events(turns: pd.DataFrame) -> pd.DataFrame:
    """One row per overlap event, with its class. Columns: call_id, b_turn_index, interrupter, ..., cls."""
    check_turns(turns)
    t = turns.sort_values(["call_id", "turn_index"]).reset_index(drop=True).copy()
    t["n_words"] = t["text"].fillna("").str.split().str.len()
    ap = t["text"].map(affirmation_prefix)
    t["ap_k"], t["ap_n"] = ap.str[0], ap.str[1]
    g = t.groupby("call_id")
    for col in ["end_s", "start_s", "n_words", "text", "ap_k", "ap_n"]:
        t["a_" + col] = g[col].shift(1)
    t["a_gap_before"] = g["start_s"].shift(1) - g["end_s"].shift(2)
    t["next_start"] = g["start_s"].shift(-1)
    t["conv_end"] = g["end_s"].transform("max")
    e = t[t["a_end_s"].notna() & (t["start_s"] < t["a_end_s"])].copy()

    low = e["text"].fillna("").str.lower()
    e["rem_s"] = e["a_end_s"] - e["start_s"]
    e["b_affirm"] = (e["n_words"] <= 4) & (e["ap_k"] == e["ap_n"]) & (e["ap_n"] > 0) & (e["end_s"] - e["start_s"] < 3)
    e["b_affirm_start"] = e["ap_k"] >= 1
    e["a_affirm"] = (e["a_n_words"] <= 4) & (e["a_ap_k"] == e["a_ap_n"]) & (e["a_ap_n"] > 0)
    e["a_was_overlap"] = (e["a_gap_before"] < 0).fillna(False)
    e["a_holds"] = (e["a_end_s"] >= e["end_s"]) | (e["next_start"].notna() & (e["next_start"] < e["end_s"]))
    e["claim_start"] = low.map(lambda s: (m := _FIRST_WORD.match(s)) is not None and m.group(1) in CLAIM_WORDS)
    e["redirect"] = low.str.replace(r"[.,!?;:]", "", regex=True).map(lambda s: REDIRECT_RE.search(s) is not None)
    e["in_final_10s"] = e["conv_end"] - e["start_s"] < 10

    conds = [
        e["a_affirm"] | e["a_was_overlap"],
        e["b_affirm"],
        e["rem_s"] <= 1.0,
        e["a_holds"] & (e["n_words"] <= 4) & ~e["claim_start"] & ~e["b_affirm_start"],
        e["b_affirm_start"],
        e["a_n_words"] < 5,
        e["a_holds"],
        (e["n_words"] <= 1) & ~e["claim_start"],
    ]
    choices = ["x_resumption", "coop_backchannel", "coop_terminal", "coop_nonfloor", "x_ack_then_talk",
               "x_other", "comp_attempt", "x_other"]
    e["cls"] = np.select(conds, choices, default="comp_floor")
    e = e.rename(columns={"turn_index": "b_turn_index", "speaker": "interrupter", "n_words": "b_words",
                          "a_n_words": "a_words"})
    return e[["call_id", "b_turn_index", "interrupter", "start_s", "end_s", "b_words", "a_words", "rem_s",
              "a_holds", "claim_start", "redirect", "in_final_10s", "cls", "text"]].reset_index(drop=True)


def _counts(ev: pd.DataFrame) -> pd.DataFrame:
    ev = ev[~ev["in_final_10s"].astype(bool)]
    full = [f"{w}_{k}" for w in ("agen", "cust") for k in CLASSES + ["comp_floor_redirect"]]
    if ev.empty:
        return pd.DataFrame(columns=["n_" + c for c in full], dtype=float)
    c = ev.assign(key=ev["interrupter"].str[:4] + "_" + ev["cls"]).pivot_table(
        index="call_id", columns="key", values="cls", aggfunc="size", fill_value=0)
    red = ev[(ev["cls"] == "comp_floor") & ev["redirect"]].groupby(["call_id", "interrupter"]).size().unstack(fill_value=0)
    for who in ("agent", "customer"):
        c[f"{who[:4]}_comp_floor_redirect"] = red[who] if who in red else 0
    return c.reindex(columns=full, fill_value=0).fillna(0).add_prefix("n_")


def winsor(s: pd.Series) -> pd.Series:
    lo, hi = s.quantile([0.01, 0.99])
    return s.clip(lo, hi)


def add_rates(df: pd.DataFrame, minutes_col: str) -> pd.DataFrame:
    """Adds {agent,cust}_{coop,comp}_rate, per-class rates and comp_share (see module docstring)."""
    df = df.copy()
    for who, pre in [("agent", "n_agen_"), ("cust", "n_cust_")]:
        coop = sum(df[pre + c] for c in COOP)
        df[f"{who}_coop_rate"] = winsor(coop / df[minutes_col])
        df[f"{who}_comp_rate"] = winsor(df[pre + "comp_floor"] / df[minutes_col])
        for sub in COOP + ["comp_attempt", "comp_floor_redirect"]:
            df[f"{who}_{sub}_rate"] = winsor(df[pre + sub] / df[minutes_col])
        den = coop + df[pre + "comp_floor"]
        df[f"{who}_comp_share"] = winsor(pd.Series(
            np.where(den >= 3, df[pre + "comp_floor"] / den.where(den > 0), np.nan), index=df.index))
    return df


def call_overlap_rates(turns: pd.DataFrame, events: pd.DataFrame | None = None) -> pd.DataFrame:
    """Whole-call counts and rates per minute of conversation, one row per call."""
    ev = overlap_events(turns) if events is None else events
    g = turns.groupby("call_id")
    base = pd.DataFrame({"conv_min": (g["end_s"].max() - g["start_s"].min()) / 60})
    out = base.join(_counts(ev), how="left").fillna(0)
    return add_rates(out, "conv_min").reset_index()


def post_anchor_rates(turns: pd.DataFrame, windows: pd.DataFrame, events: pd.DataFrame | None = None,
                      min_minutes: float = 0.5) -> pd.DataFrame:
    """Rates from the blow-up anchor to the end of the call (events with turn index > anchor), per minute
    after the anchor start. `windows` = output of detector.response_window. Calls with < min_minutes
    after the anchor are dropped."""
    ev = overlap_events(turns) if events is None else events
    w = windows.set_index("call_id")
    ev = ev[ev["call_id"].isin(w.index)]
    ev = ev[ev["b_turn_index"].to_numpy() > w.loc[ev["call_id"], "anchor_turn_index"].to_numpy()]
    conv_end = turns.groupby("call_id")["end_s"].max()
    base = pd.DataFrame({"post_min": (conv_end.reindex(w.index) - w["anchor_start_s"]) / 60})
    base = base[base["post_min"] >= min_minutes]
    out = base.join(_counts(ev), how="left").fillna(0)
    return add_rates(out, "post_min").reset_index()


def window_listening_exposures(turns: pd.DataFrame, windows: pd.DataFrame,
                               events: pd.DataFrame | None = None) -> pd.DataFrame:
    """Agent listening exposures in the initial-response window (anchor .. end_turn_index), per call:
    w_bc (agent backchannels), w_coop (all agent cooperative overlaps), u_bc (backchannel *during* the
    upset turn, i.e. at turn anchor+1), p_bc (before the anchor; placebo), w_cust_min (customer talk
    minutes in the window), bc_rate = w_bc / max(w_cust_min, 0.05) winsorized, and pre_bc_rate
    (p_bc per minute before the anchor; NaN if < 0.5 min)."""
    ev = overlap_events(turns) if events is None else events
    ev = ev[(ev["interrupter"] == "agent") & ev["call_id"].isin(windows["call_id"])]
    w = windows.set_index("call_id")
    a = w.loc[ev["call_id"], "anchor_turn_index"].to_numpy()
    e_ = w.loc[ev["call_id"], "end_turn_index"].to_numpy()
    b = ev["b_turn_index"].to_numpy()
    fin = ev["in_final_10s"].to_numpy()
    cls = ev["cls"].to_numpy()
    in_w = (b > a) & (b <= e_) & ~fin
    flags = pd.DataFrame({
        "call_id": ev["call_id"].to_numpy(),
        "w_bc": in_w & (cls == "coop_backchannel"),
        "w_coop": in_w & np.isin(cls, COOP),
        "u_bc": (b == a + 1) & ~fin & (cls == "coop_backchannel"),
        "p_bc": (b < a) & (cls == "coop_backchannel"),
    }).groupby("call_id").sum()
    t = turns[turns["call_id"].isin(w.index)].copy()
    t = t.join(w[["anchor_turn_index", "end_turn_index"]], on="call_id")
    cw = t[(t["speaker"] == "customer") & t["turn_index"].between(t["anchor_turn_index"], t["end_turn_index"])]
    cust_min = ((cw["end_s"] - cw["start_s"]).groupby(cw["call_id"]).sum() / 60).rename("w_cust_min")
    pre_min = ((w["anchor_start_s"] - turns.groupby("call_id")["start_s"].min().reindex(w.index)) / 60).rename("pre_min")
    out = pd.DataFrame(index=w.index).join(flags).fillna(0).join(cust_min).join(pre_min)
    out["w_cust_min"] = out["w_cust_min"].fillna(0)
    out["bc_rate"] = winsor(out["w_bc"] / out["w_cust_min"].clip(lower=0.05))
    out["pre_bc_rate"] = np.where(out["pre_min"] >= 0.5, out["p_bc"] / out["pre_min"].clip(lower=0.5), np.nan)
    out["pre_bc_rate"] = winsor(out["pre_bc_rate"])
    return out.reset_index()
