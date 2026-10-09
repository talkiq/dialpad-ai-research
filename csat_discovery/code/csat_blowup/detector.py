"""Upset-moment ("blow-up" / "rupture") detector, final (v3) lexicon.

A customer turn is flagged when it contains at least one escalation/frustration marker (case-insensitive,
whole-word/phrase). Markers are tone/escalation words plus a manager/supervisor *request* pattern
(a request verb followed by up to four filler words and "manager"/"supervisor"), so that a bare mention
of "my manager" is not flagged. Topic/intent words that were in earlier versions ("cancel my",
"cancel this", "corporate", "lawsuit") are dropped: in calm, transactional speech they are common and
were the largest false-positive source against independent labels. Only customer turns are scanned.

Per call:
    has_rupture                bool
    n_ruptures                 number of blow-up *episodes*: consecutive flagged customer turns (no
                               unflagged customer turn in between; agent turns are ignored) count once
    rupture_severity_sum       total number of marker matches across flagged turns
    first_rupture_turn_index   turn_index of the first flagged customer turn (the "anchor")
    last_rupture_turn_index
    total_turns                max(turn_index) + 1
    rupture_position_fraction  first_rupture_turn_index / total_turns  (share of the call elapsed)

`response_window` then extracts, per flagged call, the customer's first upset turn and the agent's
next (up to) two turns -- the "initial response" used by the matching and initial-response modules.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from .schema import check_turns

TONE_PHRASES = [
    "escalate", "unacceptable", "ridiculous",
    "worst", "furious", "terrible", "awful", "pathetic", "sue you",
    "never again", "done with this", "speak to someone else", "speak to somebody else",
    "this is a joke", "waste of my time", "waste of time", "fed up", "sick of", r"can.?t believe",
    "horrible", "disgusted", "incompetent", "useless", "scam", "rip off", "ripoff", "shut up",
    "damn", "crap",
]
DROPPED_PHRASES = ["cancel my", "cancel this", "corporate", "lawsuit"]  # documented, not used

MANAGER_REQUEST_VERBS = [
    "speak to", "speak with", "talk to", "talk with", "get me", "transfer me to",
    "connect me to", "put me through to", "ask for a", "ask for the",
]
MANAGER_REQUEST_TARGETS = ["manager", "supervisor"]
MANAGER_REQUEST_GAP = 4  # filler words allowed between verb and target

MANAGER_REQUEST_PHRASES = [
    rf"{verb}(?:\s+\w+){{0,{MANAGER_REQUEST_GAP}}}\s+{target}"
    for verb in MANAGER_REQUEST_VERBS
    for target in MANAGER_REQUEST_TARGETS
]

RUPTURE_PHRASES = TONE_PHRASES + MANAGER_REQUEST_PHRASES
# Valid both as a Python `re` pattern and as an RE2 (SQL engine) pattern: no lookaround/backrefs.
RUPTURE_REGEX = r"\b(?:" + "|".join(RUPTURE_PHRASES) + r")\b"
RUPTURE_RE = re.compile(RUPTURE_REGEX, re.IGNORECASE)


def find_markers(text: str) -> list[str]:
    """All non-overlapping marker matches in one turn (lower-cased)."""
    if not isinstance(text, str):
        return []
    return [m.group(0).lower() for m in RUPTURE_RE.finditer(text)]


def flag_turns(turns: pd.DataFrame) -> pd.DataFrame:
    """Returns the customer turns with columns n_matches, is_rupture and episode (1-based per call,
    NaN for unflagged turns)."""
    check_turns(turns, timing=False)
    cust = turns[turns["speaker"] == "customer"].sort_values(["call_id", "turn_index"]).copy()
    cust["n_matches"] = cust["text"].map(lambda t: len(find_markers(t)))
    cust["is_rupture"] = cust["n_matches"] > 0
    cust["customer_seq"] = cust.groupby("call_id").cumcount()
    # gaps-and-islands: consecutive flagged customer turns share customer_seq - rank-among-flagged
    flagged = cust[cust["is_rupture"]].copy()
    flagged["grp"] = flagged["customer_seq"] - flagged.groupby("call_id").cumcount()
    flagged["episode"] = flagged.groupby("call_id")["grp"].rank(method="dense").astype(int)
    cust["episode"] = flagged["episode"]
    return cust.drop(columns=["customer_seq"])


def detect_blowups(turns: pd.DataFrame) -> pd.DataFrame:
    """One row per call in `turns` with the per-call blow-up features listed in the module docstring."""
    flagged = flag_turns(turns)
    totals = turns.groupby("call_id")["turn_index"].max().add(1).rename("total_turns")
    f = flagged[flagged["is_rupture"]]
    agg = f.groupby("call_id").agg(
        n_ruptures=("episode", "nunique"),
        rupture_severity_sum=("n_matches", "sum"),
        first_rupture_turn_index=("turn_index", "min"),
        last_rupture_turn_index=("turn_index", "max"),
    )
    out = totals.to_frame().join(agg, how="left").reset_index()
    out["n_ruptures"] = out["n_ruptures"].fillna(0).astype(int)
    out["rupture_severity_sum"] = out["rupture_severity_sum"].fillna(0).astype(int)
    out["has_rupture"] = out["n_ruptures"] > 0
    out["rupture_position_fraction"] = out["first_rupture_turn_index"] / out["total_turns"]
    return out


def response_window(turns: pd.DataFrame, blowups: pd.DataFrame, n_agent_turns: int = 2) -> pd.DataFrame:
    """Initial-response window for each call with a blow-up.

    anchor_turn_index   = first_rupture_turn_index
    end_turn_index      = index of the agent's n-th turn after the anchor (falls back to the last
                          available agent turn, then to the anchor itself)
    customer_text       = text of the anchor turn
    agent_text          = agent turns with anchor < turn_index <= end_turn_index, joined by spaces
    response_gap_s      = start of turn anchor+1 minus end of the anchor turn (negative = overlap)
    """
    check_turns(turns)
    b = blowups.loc[blowups["has_rupture"], ["call_id", "first_rupture_turn_index"]]
    b = b.rename(columns={"first_rupture_turn_index": "anchor_turn_index"})
    t = turns.merge(b, on="call_id").sort_values(["call_id", "turn_index"])
    after = t[(t["speaker"] == "agent") & (t["turn_index"] > t["anchor_turn_index"])].copy()
    after["agent_rank"] = after.groupby("call_id").cumcount() + 1
    after = after[after["agent_rank"] <= n_agent_turns]
    end = after.groupby("call_id")["turn_index"].max().rename("end_turn_index")
    agent_text = after.groupby("call_id")["text"].agg(lambda s: " ".join(s.astype(str))).rename("agent_text")

    anchor = t[t["turn_index"] == t["anchor_turn_index"]].set_index("call_id")
    nxt = t[t["turn_index"] == t["anchor_turn_index"] + 1].set_index("call_id")
    out = b.set_index("call_id").join(end).join(agent_text)
    out["end_turn_index"] = out["end_turn_index"].fillna(out["anchor_turn_index"]).astype(int)
    out["agent_text"] = out["agent_text"].fillna("")
    out["customer_text"] = anchor["text"]
    out["anchor_start_s"] = anchor["start_s"]
    out["anchor_end_s"] = anchor["end_s"]
    out["response_gap_s"] = nxt["start_s"] - anchor["end_s"]
    return out.reset_index()


def blowup_summary(calls: pd.DataFrame, blowups: pd.DataFrame) -> dict:
    """Base rate plus bad-rating % by presence, count (1, 2, 3+) and position fifth (descriptive)."""
    d = calls.merge(blowups, on="call_id")
    bottom = (d["rating"] <= 2).astype(float)
    out = {"n_calls": len(d), "blowup_rate": float(d["has_rupture"].mean()),
           "bad_pct_with": float(100 * bottom[d["has_rupture"]].mean()),
           "bad_pct_without": float(100 * bottom[~d["has_rupture"]].mean())}
    r = d[d["has_rupture"]]
    nr = np.minimum(r["n_ruptures"], 3)
    out["bad_pct_by_n_1_2_3plus"] = (100 * bottom[r.index].groupby(nr).mean()).round(1).tolist()
    pos5 = np.minimum((r["rupture_position_fraction"] * 5).astype(int), 4)
    out["bad_pct_by_position_fifth"] = (100 * bottom[r.index].groupby(pos5).mean()).round(1).tolist()
    return out
