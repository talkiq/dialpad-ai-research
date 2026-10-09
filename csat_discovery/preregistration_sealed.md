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

Each test is run **once** on Sealed, with the code, model weights, lexicons, label rubrics and specifications frozen as used on Explore/Confirm. Code for the methods is in [`code/`](code/).

| # | Claim | Estimate | Confirmed if |
|---|---|---|---|
| S1 | A late blow-up predicts a low rating | Last vs first fifth of the call (by share of turns elapsed) in which the first blow-up occurs. Within-company linear probability model with duration, number of legs, hour, weekday and blow-up severity; company-clustered SEs. The within-company logit per SD of position is reported alongside | Difference > 0 and its 95% CI excludes 0 |
| S2 | The agent's initial response predicts the ending in unseen companies | The logistic model trained on Explore initial-response embeddings (the agent's next two turns after the first blow-up), applied unchanged; calls sorted into fifths of predicted risk | AUC > 0.5 beyond a permutation null (p < 0.05); fifths reported |
| S3 | Agent listening signals go with fewer low ratings | Agent backchannel rate after the blow-up, per SD. Matched design: 10 nearest neighbors within vertical on the customer's upset-turn embedding, neighborhood fixed effects, out-of-fold customer-risk score, company-cluster bootstrap CI | Coefficient < 0 and CI excludes 0 |
| S4 | Arguing with a manager request goes with more low ratings | Agent argues with or refuses a customer's explicit request for a manager, vs does not address it. Within company, controlling for duration and blow-up count and severity | Coefficient > 0 and CI excludes 0 |
| S5 | Customers who warm up give fewer low ratings when nothing can be fixed | Customer "sweetened" (turned genuinely warm) vs not, among emotional calls judged not fixable (frozen LLM rubric); matched design as in S3 | Coefficient < 0 and CI excludes 0 |

Secondary, reported but not headline:
- the agent's increment in AUC over a customer-only classifier;
- a concrete fix in the initial response (matched design).

Outcome categories: **confirmed** (sign as stated, CI excludes 0); **weakened** (sign as stated, CI includes 0); **not confirmed** (opposite sign).

## 3. Reporting commitments

- All five results are reported in the manuscript's abstract and main tables, whatever they show.
- No test is re-run, re-specified or dropped after Sealed data are touched. If a test cannot be run (for example, too few eligible calls for S5), it is reported as "not run" with the reason.
- Any deviation from this document is listed explicitly in the manuscript.
