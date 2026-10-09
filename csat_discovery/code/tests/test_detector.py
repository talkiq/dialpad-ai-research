import pandas as pd

from csat_blowup.detector import detect_blowups, find_markers, response_window


def _turns(rows):
    out = []
    for i, (spk, txt) in enumerate(rows):
        out.append(dict(call_id="c1", turn_index=i, speaker=spk, text=txt, start_s=3.0 * i, end_s=3.0 * i + 2.5))
    return pd.DataFrame(out)


def test_lexicon_rules():
    assert find_markers("Honestly this is RIDICULOUS") == ["ridiculous"]
    assert find_markers("can i speak to your manager please") != []
    assert find_markers("let me talk with the floor supervisor") != []
    assert find_markers("my manager told me to call") == []          # mention, not a request
    assert find_markers("i want to cancel my subscription") == []    # topic word dropped in v3
    assert find_markers("i can't believe it, what a waste of time") == ["can't believe", "waste of time"]


def test_episodes_severity_position_and_window():
    t = _turns([
        ("agent", "hello how can i help"),
        ("customer", "this is ridiculous and terrible"),   # 2 matches, episode 1
        ("agent", "mm-hmm"),
        ("customer", "honestly unacceptable"),             # next customer turn also flagged -> same episode
        ("agent", "i understand let me fix it"),
        ("customer", "okay"),                              # unflagged customer turn ends the episode
        ("agent", "one moment"),
        ("customer", "this is the worst"),                 # episode 2
        ("agent", "bye"),
    ])
    blow = detect_blowups(t)
    b = blow.iloc[0]
    assert b.has_rupture and b.n_ruptures == 2 and b.rupture_severity_sum == 4
    assert b.first_rupture_turn_index == 1 and b.total_turns == 9
    assert abs(b.rupture_position_fraction - 1 / 9) < 1e-12
    w = response_window(t, blow).iloc[0]
    assert w.end_turn_index == 4
    assert w.agent_text == "mm-hmm i understand let me fix it"
    assert w.customer_text == "this is ridiculous and terrible"
