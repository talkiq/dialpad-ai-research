import pandas as pd

from csat_blowup.overlaps import affirmation_prefix, is_backchannel, overlap_events


def test_affirmation_prefix():
    assert affirmation_prefix("Okay, got it.") == (3, 3)
    assert affirmation_prefix("yeah but that is wrong") == (1, 5)
    assert affirmation_prefix("") == (0, 0)
    assert is_backchannel("mm-hmm", 0.4) and not is_backchannel("mm-hmm", 4.0)


def test_overlap_classes():
    # A = long customer turn; B overlaps it in a different way in each call
    long_a = "so the order never arrived and i have been waiting two weeks for it"
    cases = {
        "bc": ("mm-hmm", 2.0, 2.4),                                    # short affirmation inside A
        "term": ("let me check that for you now", 7.6, 10.0),         # starts within last 1 s of A
        "floor": ("no wait let me explain how this works", 4.0, 9.0),  # takes the floor 4 s early
    }
    rows = []
    for cid, (b_text, b_start, b_end) in cases.items():
        rows += [dict(call_id=cid, turn_index=0, speaker="customer", text=long_a, start_s=0.0, end_s=8.0),
                 dict(call_id=cid, turn_index=1, speaker="agent", text=b_text, start_s=b_start, end_s=b_end),
                 dict(call_id=cid, turn_index=2, speaker="customer", text="okay", start_s=30.0, end_s=31.0)]
    ev = overlap_events(pd.DataFrame(rows)).set_index("call_id")
    assert ev.loc["bc", "cls"] == "coop_backchannel"
    assert ev.loc["term", "cls"] == "coop_terminal"
    assert ev.loc["floor", "cls"] == "comp_floor" and bool(ev.loc["floor", "redirect"])
    assert set(ev["interrupter"]) == {"agent"}
