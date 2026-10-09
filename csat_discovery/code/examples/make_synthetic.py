"""Fully synthetic support-call data for the demo and tests.

Nothing here comes from real calls: companies, verticals, sentences and ratings are invented and
generated from fixed random seeds. The generator plants the paper's qualitative patterns (blow-ups and
late blow-ups go with worse ratings, listening/acknowledging replies with better ones, deflecting
replies and manager requests with worse ones) so every method has something to find.

    python examples/make_synthetic.py --out /tmp/synthetic   # writes turns.csv and calls.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

VERTICALS = ["vertical_a", "vertical_b", "vertical_c"]

GREETING = ["thanks for calling how can i help you today", "hello you are speaking with support what can i do for you"]
CUSTOMER_NEUTRAL = [
    "i am calling about my account", "the package has not arrived yet", "can you check my order status",
    "i need to update my mailing address", "the app keeps logging me out", "i was charged twice this month",
    "my appointment time needs to change", "the device will not turn on", "i have a question about my bill",
    "the form on the website gives an error", "i never got the confirmation email", "it says my card was declined",
]
CUSTOMER_UPSET = [
    "this is ridiculous i have called three times about this", "honestly this is unacceptable",
    "what a waste of my time this has been", "i am fed up with this whole thing",
    "this is the worst service i have had", "i can't believe nobody has fixed this yet",
    "this is terrible and nobody calls me back", "you people are useless",
]
CUSTOMER_ESCALATE = ["i want to speak to a manager right now", "can you transfer me to your supervisor please"]
AGENT_NEUTRAL = [
    "let me pull up the account", "can you confirm the email on file", "one moment while i check that",
    "thanks for waiting i see the order here", "i can help with that", "let me look at the history",
]
AGENT_LISTEN = [
    "i understand how frustrating that is let me fix this for you now",
    "i am so sorry about that i will take care of it right away",
    "got it thank you for explaining i will sort this out today",
    "that makes sense i see exactly what happened let me correct it",
]
AGENT_DEFLECT = [
    "that is our policy and there is nothing i can do",
    "you will have to call back another day", "the system does not allow that",
    "i cannot change that you need to go through the website",
]
BACKCHANNELS = ["mm-hmm", "okay", "got it", "i see", "right", "yeah"]
CUT_INS = ["no wait let me explain how this works", "but that is not what the account shows here"]
CLOSING = ["is there anything else i can help with", "thanks for calling have a good day"]


def _turn(rows, call_id, speaker, text, start, rng, dur=None):
    if dur is None:
        dur = max(0.8, 0.55 * len(text.split()) + rng.normal(0.5, 0.3))
    rows.append([call_id, len(rows), speaker, text, start, start + dur])
    return start + dur


def make_synthetic(n_calls: int = 1600, n_companies: int = 60, seed: int = 7) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (turns, calls) with the schemas documented in csat_blowup/schema.py.
    calls has an extra column company_set ("A" for the first half of companies, "B" for the rest)."""
    rng = np.random.default_rng(seed)
    companies = [f"co_{i:02d}" for i in range(n_companies)]
    comp_vertical = {c: VERTICALS[i % len(VERTICALS)] for i, c in enumerate(companies)}
    comp_effect = {c: rng.normal(0, 0.5) for c in companies}
    comp_listen = {c: rng.normal(0, 0.6) for c in companies}
    all_rows, calls = [], []
    for k in range(n_calls):
        rows = []
        call_id = f"call_{k:05d}"
        comp = companies[rng.integers(n_companies)]
        n_pairs = int(rng.integers(6, 12))
        has_blowup = rng.random() < 0.45
        anchor_pair = int(rng.integers(1, n_pairs)) if has_blowup else -1
        escalate = has_blowup and rng.random() < 0.3
        listen = rng.random() < 1 / (1 + np.exp(-comp_listen[comp]))
        second = has_blowup and rng.random() < 0.25
        t = 0.0
        t = _turn(rows, call_id, "agent", rng.choice(GREETING), t, rng)
        for p in range(n_pairs):
            t += rng.normal(0.5, 0.3)
            if p == anchor_pair or (second and p == anchor_pair + 1 and rng.random() < 0.5):
                text = rng.choice(CUSTOMER_ESCALATE) if escalate and p == anchor_pair else (
                    rng.choice(CUSTOMER_UPSET) + " " + rng.choice(CUSTOMER_NEUTRAL))
            elif second and p == n_pairs - 1:
                text = rng.choice(CUSTOMER_UPSET)
            else:
                text = rng.choice(CUSTOMER_NEUTRAL)
            c_start = t
            t = _turn(rows, call_id, "customer", text, c_start, rng)
            cut_in = rng.random() < (0.05 if listen else 0.2)
            # agent backchannel overlapping the customer's turn (more likely for listening agents)
            if not cut_in and rng.random() < (0.55 if listen else 0.12) and t - c_start > 2.0:
                _turn(rows, call_id, "agent", rng.choice(BACKCHANNELS), c_start + 0.4 * (t - c_start), rng, dur=0.5)
            if p >= anchor_pair >= 0:
                if p == anchor_pair:
                    reply = rng.choice(AGENT_LISTEN) if listen else rng.choice(AGENT_DEFLECT)
                else:
                    reply = rng.choice(AGENT_LISTEN + AGENT_NEUTRAL) if listen else rng.choice(AGENT_DEFLECT + AGENT_NEUTRAL)
            else:
                reply = rng.choice(AGENT_NEUTRAL)
            if cut_in:  # floor-taking cut-in
                start = t - 1.8
                reply = rng.choice(CUT_INS)
            else:
                start = t + max(0.1, rng.normal(0.6, 0.3))
            t = _turn(rows, call_id, "agent", reply, start, rng)
        t = _turn(rows, call_id, "customer", "okay thanks", t + 0.4, rng)
        t = _turn(rows, call_id, "agent", rng.choice(CLOSING), t + 0.4, rng)
        all_rows.extend(rows)

        pos = (anchor_pair + 1) / (n_pairs + 1) if has_blowup else 0.0
        logit = -2.0 + comp_effect[comp] - 0.4 * listen
        if has_blowup:
            logit += 0.4 + 2.0 * pos + 1.5 * escalate + 0.5 * second - 1.0 * listen
        bottom = rng.random() < 1 / (1 + np.exp(-logit))
        rating = int(rng.choice([1, 2], p=[0.7, 0.3])) if bottom else int(rng.choice([3, 4, 5], p=[0.1, 0.2, 0.7]))
        calls.append(dict(call_id=call_id, company_id=comp, vertical=comp_vertical[comp], rating=rating,
                          log_duration=np.log(max(t, 1.0)), n_legs=int(rng.integers(1, 3)),
                          hour=int(rng.integers(8, 20)), dow=int(rng.integers(0, 7)),
                          company_set="A" if companies.index(comp) < n_companies // 2 else "B"))
    turns = pd.DataFrame(all_rows, columns=["call_id", "turn_index", "speaker", "text", "start_s", "end_s"])
    return turns, pd.DataFrame(calls)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="directory to write turns.csv / calls.csv")
    ap.add_argument("--n-calls", type=int, default=1600)
    a = ap.parse_args()
    turns, calls = make_synthetic(a.n_calls)
    print(turns.head(12).to_string(index=False))
    print(calls.head().to_string(index=False))
    if a.out:
        Path(a.out).mkdir(parents=True, exist_ok=True)
        turns.to_csv(Path(a.out) / "turns.csv", index=False)
        calls.to_csv(Path(a.out) / "calls.csv", index=False)
