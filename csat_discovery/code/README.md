# Code: What to Say After a Blow-Up

Reference implementation of the methods in *What to Say After a Blow-Up: Satisfaction and Agent Behavior
in Transcribed Support Calls*. The modules take pandas DataFrames. No call data is included. The
examples run on **fully synthetic** data generated in code (`examples/make_synthetic.py`). The numbers
the demo prints describe that generator, not real calls.

## Quickstart

```bash
pip install -r requirements.txt
OMP_NUM_THREADS=1 python examples/run_demo.py                     # MiniLM if available, else hashing fallback
OMP_NUM_THREADS=1 python examples/run_demo.py --encoder hashing   # no model download
python -m pytest tests
```

The demo runs in under a minute on a laptop (1,600 synthetic calls, about 36k turns).

## Input schemas

**turns**: one row per transcribed speaker turn

| column | type | notes |
|---|---|---|
| `call_id` | str/int | |
| `turn_index` | int | 0-based order within the call |
| `speaker` | str | `agent` or `customer` |
| `text` | str | ASR text |
| `start_s`, `end_s` | float | seconds from call start. Overlaps are turns that start before the previous turn ends |

**calls**: one row per rated call

| column | type | notes |
|---|---|---|
| `call_id` | str/int | |
| `company_id` | str/int | cluster for within-company models and bootstraps |
| `vertical` | str | industry; look-alike matching stays within a vertical |
| `rating` | int 1-5 | post-call survey. `schema.add_outcomes` adds `bottom` (1-2 stars) and `success` (4-5) |
| controls | numeric | e.g. `log_duration`, `n_legs`, hour, day of week |

## Modules (`csat_blowup/`)

| module | what it does |
|---|---|
| `detector.py` | Upset-moment ("blow-up") detector with the final (v3) lexicon. It flags a customer turn when it contains a tone/escalation marker or a manager/supervisor *request* (a request verb, up to 4 filler words, then "manager"/"supervisor"). Topic words such as "cancel my" are not used. Per call it returns the episode count (consecutive flagged customer turns count once), severity (total marker matches), the first/last flagged turn and position (first flagged turn / total turns). `response_window` returns the first upset turn and the agent's next two turns. |
| `within_company.py` | Mundlak within/between cluster-robust logit. The feature and its within/between parts are z-scored, so the odds ratios are per within-company SD. SEs are clustered by company. `clears_sesoi` checks whether the 95% CI lies outside [1/1.10, 1.10]. |
| `effects.py` | Turns logit coefficients into percentage points, fits a within-company linear probability model, and reports blow-up position/severity as bad-rating percentages (raw rates by fifth, last vs. first fifth adjusted for severity). |
| `initial_response.py` | Initial-response classifier (H78 in the paper's supplement): StandardScaler + L2 logistic regression on the embedding of the agent's reply. Reports company-grouped CV AUC, a within-company label-permutation null, and a company-disjoint train/apply test with bad-rating % in fifths of predicted risk. Optional difference-of-vectors check: projection removal and a ridge residual refit inside each training fold. |
| `matching.py` | Matched look-alike comparison. Each call is matched to its 10 nearest neighbours in the same vertical, using the embedding of the customer's upset turn. A linear probability model with neighbourhood fixed effects is fit, plus an out-of-fold customer-risk score. CIs come from a company-cluster bootstrap. Results are in points per 100 calls per SD of exposure. It also reports the customer-risk AUC inside neighbourhoods as a matching diagnostic. |
| `overlaps.py` | Overlap rules. Backchannels (≤4 words, all from the backchannel lexicon, <3 s), terminal and non-floor cooperative overlaps, failed attempts and floor-taking cut-ins, resumptions excluded. Computes per-call and post-blow-up rates per minute and the cut-in share, plus listening exposures in the initial-response window (with a pre-blow-up placebo rate). |
| `embeddings.py` | Pluggable encoders: `minilm` (all-MiniLM-L6-v2, as in the paper), `hashing` (offline fallback), or any callable `list[str] -> ndarray`. |
| `schema.py` | Column checks and outcome columns. |

## Differences from the paper's runs

- The demo uses small synthetic data and a reduced number of bootstrap/permutation draws (`--n-boot`, `--n-perm`). The paper used 1,000-2,000 bootstrap reps and 50 permutations.
- The paper also matched on LLM-coded strategy labels and used audio measures. Neither is included here because they depend on an external labeler or audio.
