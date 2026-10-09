# What to Say After a Blow-Up: Satisfaction and Agent Behavior in Transcribed Support Calls

**Paper:** manuscript in preparation (link to follow).

## Overview

What happens after a customer gets upset in a support call? Using transcribed calls with post-call satisfaction ratings from two disjoint sets of companies, we find:
- a blow-up late in a call is the strongest predictor of a low rating;
- the agent's initial response predicts how the call ends, mostly because it reflects the customer's situation;
- at matched customer state, a few agent behaviors go with better endings: listening signals, early validation, a concrete first action, and not arguing with a manager request.

All evidence is observational. Methods use a company-level Explore / Confirm / Sealed split, within-company models and matched look-alike comparisons.

## Contents

- [`preregistration_sealed.md`](preregistration_sealed.md): pre-registered one-time confirmation of the headline findings on the held-out Sealed split of companies.
- [`code/`](code/): implementations of the main methods (blow-up detector, within-company models, matched comparisons, initial-response classifier, overlap and backchannel rules). The demo runs on synthetic data.

## Data

The call recordings and transcripts are customer data and are not released. No transcripts, call identifiers or personal information are included in this folder.
