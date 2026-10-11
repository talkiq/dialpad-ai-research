# Pre-registration: held-out ("Sealed") confirmation of the headline findings

**Paper:** *What to Say After a Blow-Up: Satisfaction and Agent Behavior in Transcribed Support Calls* (manuscript in preparation).
**Status:** DRAFT, not yet frozen. This file is frozen by the commit and release tag that remove this status line. No Sealed outcome has been computed at any point before that.

## 1. Background and design

We study what predicts a customer's post-call satisfaction (CSAT) rating after the customer becomes upset in a support call (a *blow-up*). The data are transcribed calls from a commercial contact-center platform. Each call has a post-call survey rating on a 1–5 scale; the outcome is a rating of 1–2 stars. The calls cannot be shared.

Companies, not calls, were randomly assigned to three disjoint splits before analysis:

- **Explore** (40% of companies): all method development.
- **Confirm** (40%): each hypothesis tested once, with the method frozen.
- **Sealed** (20%): untouched until the tests below.

All findings in the manuscript come from Explore and Confirm: 345,609 and 557,412 calls from 215 and 205 companies over three years, after a data-retention compliance filter. Before this registration, the only operation run on Sealed data was an outcome-blind count of eligible calls: 316,412 calls from 108 companies after the same compliance filter. No ratings were read.

## 2. Tests

Each test is run **once** on Sealed, with the code, model weights, lexicons, label rubrics and specifications frozen as used on Explore/Confirm. Code for the methods is in [`code/`](code/). "Matched design" means:
- 10 nearest neighbors within vertical on the customer's upset-turn embedding;
- neighborhood fixed effects and an out-of-fold customer-risk score;
- company-cluster bootstrap 95% CIs.

"Initial response" means the agent's next two turns after the customer's first blow-up.

### Tests (11, equal standing)

| # | Claim | Estimate | Confirmed if |
|---|---|---|---|
| S1 | A late blow-up predicts a low rating | Last vs first fifth of the call in which the first blow-up occurs. Within-company linear probability model with duration, number of legs, hour, weekday and blow-up severity; company-clustered SEs. Within-company logit per SD of position reported alongside | Difference > 0, CI excludes 0 |
| S2a | The initial response predicts the ending in unseen companies | Logistic model trained on Explore initial-response embeddings, applied unchanged; fifths of predicted risk; null distribution from permuting outcomes within company with scores fixed | AUC above the permutation null, p < 0.05 |
| S2b | The agent's wording adds signal beyond the customer's words | Explore-trained models on the customer's upset turn alone vs customer + agent initial response, both applied unchanged; difference in AUC, company-cluster bootstrap | ΔAUC > 0, CI excludes 0 |
| S3 | Agent listening signals go with fewer low ratings | Agent backchannel rate after the blow-up, per SD, matched design | Coefficient < 0, CI excludes 0 |
| S4 | Arguing with a manager request goes with more low ratings | Agent argues with or refuses an explicit request for a manager, vs does not address it; within company, controlling for duration and blow-up count and severity | Coefficient > 0, CI excludes 0 |
| S5 | A concrete fix in the initial response goes with fewer low ratings | LLM-coded concrete fix in the initial response, net of coded understanding, matched design | Coefficient < 0, CI excludes 0 |
| S6 | Customers who warm up give fewer low ratings when nothing can be fixed | Customer "sweetened" (turned genuinely warm) vs not, among emotional calls coded as not fixable, matched design | Coefficient < 0, CI excludes 0 |
| S7 | Reassurance goes with more acceptance of the outcome | Reassurance phrases → LLM-coded acceptance, matched design | Coefficient > 0, CI excludes 0 |
| S8 | Validation that first appears only after a refusal goes with less acceptance | Validation-timing phrases → acceptance, matched design | Coefficient < 0, CI excludes 0 |
| S9 | At least one helpful action goes with fewer low ratings | Helpful-action phrases; within-company model of low rating on ≥1 action vs none, percentage points, company-clustered CI | Coefficient < 0, CI excludes 0 |
| S10 | Among upset customers, a deflated voice goes with worse ratings | Change at the upset turn relative to the speaker's own baseline turns, low (1–2 star) minus 5-star raters, in units of the Explore+Confirm calm-call SD (fixed), company-cluster bootstrap CI. Measures: speaking (articulation) rate, voiced fraction, breathiness (H1–H2) | Articulation rate lower in low raters, CI excludes 0 (the other two reported) |

Outcome categories:
- **confirmed:** sign as stated and CI excludes 0;
- **weakened:** sign as stated, CI includes 0;
- **not confirmed:** opposite sign.

**Multiplicity:** all 11 tests have equal standing and each is judged on its own criterion. As a sensitivity analysis we also report which of the 11 survive Holm–Bonferroni at familywise α = 0.05. The p-values used are two-sided bootstrap p for bootstrap-based tests, cluster-robust Wald p for S1 and S9, and the permutation p for S2a. Both the per-test verdicts and the Holm results are reported.

**Runner:** a single script computes all 11 estimates with fixed seeds per test. Run on the Confirm split, it reproduces every published Confirm estimate; it refuses to run on Sealed unless this document's frozen hash is supplied.

### Frozen artifacts

- **Text labels (S5–S8):** produced by the same LLM (`claude-sonnet-5`) and frozen rubrics as on Explore/Confirm, called through the API. Before any Sealed label existed, a bridge check re-labeled 500 Confirm calls this way: agreement with the original labels matched the original labeler's own re-label reliability (κ 0.67–0.83 across fields). The rubrics will be published with the paper; their SHA-256 hashes are: distress rubric `89b95db4d35af6b101a192a79c3f1e6bd4d867921917c4f65eea92e1bc00ad84`, trajectory/acceptance rubric `6ab67196a4e0f6b5d20faaa4c9011bedf6d7612681784b8380bd5b1c2ded14c0`.
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2`.
- **Explore-trained classifiers (S2a, S2b):** refit on Explore, reproducing the published Explore→Confirm AUCs exactly (0.730; 0.780 and 0.798). SHA-256 of the frozen model files:
  - S2a: `aaad441b3211d4a0274c76a0cd24e0d59c620271b8b52b744df2d323d7e8634a`
  - S2b customer-only: `c5d239842d6a61b2222540a2ca39c73311ca45fa6aa9722b42d43ec391a76bbb`
  - S2b customer + agent: `5e1dfb96f958e114086e0b4fd9f88ae9a587c6539e4c7ec87f5d1091a769e5f8`
- **Random seeds:** fixed for all samples, bootstraps and permutations.

## 3. Reporting commitments

- All 11 results are reported, whatever they show: in the manuscript's abstract ("k of 11 confirmed on held-out companies") and in a Sealed column of the main or supplementary tables.
- No test is re-run, re-specified or dropped after Sealed data are touched. If a test cannot be run (for example, too few eligible calls or too little audio), it is reported as "not run" with the reason.
- Any deviation from this document is listed explicitly in the manuscript.
