"""Input schemas shared by every module.

turns : one row per transcribed speaker turn
    call_id     str/int   call identifier
    turn_index  int       0-based order of the turn within the call
    speaker     str       "agent" or "customer"
    text        str       ASR text of the turn
    start_s     float     turn start, seconds from call start
    end_s       float     turn end, seconds from call start
    (company_id is optional on turns; it is read from `calls`.)

calls : one row per rated call
    call_id     str/int
    company_id  str/int   cluster for within-company models and bootstraps
    vertical    str       industry; look-alike matching happens within vertical
    rating      int 1-5   post-call satisfaction survey
    + any numeric controls (e.g. log_duration, n_legs, hour, dow)

Derived outcome used throughout: bottom = rating <= 2 ("bad rating"), success = rating >= 4.
"""
from __future__ import annotations

import pandas as pd

TURN_COLS = ["call_id", "turn_index", "speaker", "text", "start_s", "end_s"]
CALL_COLS = ["call_id", "company_id", "vertical", "rating"]
SPEAKERS = {"agent", "customer"}


def check_columns(df: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def check_turns(turns: pd.DataFrame, timing: bool = True) -> None:
    cols = TURN_COLS if timing else TURN_COLS[:4]
    check_columns(turns, cols, "turns")
    bad = set(turns["speaker"].unique()) - SPEAKERS
    if bad:
        raise ValueError(f"turns.speaker must be in {SPEAKERS}; found {bad}")


def add_outcomes(calls: pd.DataFrame) -> pd.DataFrame:
    """Adds bottom (rating 1-2) and success (rating 4-5) 0/1 columns."""
    check_columns(calls, ["rating"], "calls")
    out = calls.copy()
    out["bottom"] = (out["rating"] <= 2).astype(int)
    out["success"] = (out["rating"] >= 4).astype(int)
    return out
