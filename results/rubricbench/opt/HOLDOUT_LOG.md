# Holdout usage log

The holdout half is the 547 cases listed under `holdout` in
`results/rubricbench/split.json` (frozen, seed 20260831, stratified by domain
group). It may be scored **at most three times** for the whole exercise. Every
spend is recorded here before the number is read, with the candidate, the reason
the spend was justified, and the outcome.

A spend means: computing or reading any holdout score for any source. The
verdicts for `none`, `baseline`, `framed`, `agentic` and `expert` on all 1,147
cases already exist on disk from the pre-split runs, so holdout numbers for them
are *computable* at zero cost — which makes the discipline entirely a matter of
not looking. They have not been looked at.

## Budget

| # | status | date | candidate | outcome |
|---|---|---|---|---|
| 1 | **unspent** | — | — | — |
| 2 | **unspent** | — | — | — |
| 3 | **unspent** | — | — | — |

**Used: 0 of 3.**

## Why nothing has been spent

A holdout spend is only worth making for a candidate that beat `framed` on dev
by enough to be worth confirming. No candidate did, in either round.

**Round 1** tested `framed_noweight`, `framed_bare`, `qform`, `balanced` and
`contrastive`; they landed between −0.0618 and +0.0083 against `framed` on the
full dev reading. **Round 2** tested `fs_sim5` and `fs_fix5` — few-shot on the
dev half's expert rubrics — at −0.0436 and −0.0251. Dev's minimum detectable
difference is 0.038–0.042, so the best of the seven is not distinguishable from
`framed` and two are confirmed regressions. Confirming "no change" on the holdout
would spend a third of the budget to learn nothing dev has not already said.

Round 2's other direction never reached a candidate at all: per-domain routing
was closed on an offline bound computed from verdicts already on disk (oracle
routing with ground-truth labels: +0.0233 ex-SAFETY, threshold +0.038), so there
was nothing to confirm. That bound cost zero LLM calls and zero holdout.

The one holdout use that would have been defensible — the mid-point check for
"dev jumped, is it overfitting?" — never triggered, because dev never jumped.

**A note on round 2 and the holdout, because the two interact.** `fs_sim5` and
`fs_fix5` are trained on labelled data: their few-shot examples are expert
rubrics from the dev half. The pool is dev-only by construction, the TF-IDF
featuriser is fitted on dev text alone, and a query case is excluded from its own
retrieval (audited on 200 cases for both self-inclusion and holdout
contamination). So a holdout score for either would still have been honest. It
was not taken, because neither earned one.

Consequently **every number in `OPTIMIZATION_LOG.md`, `BEST.md` and the round-2
artefacts (`route_bound*.md`, `rubric_content.md`, `dev_round2.md`) is dev-only**
and is labelled as such. The current best method (`framed`) has a published
full-benchmark score, but that score is not a holdout estimate either: `framed`
was written after reading failures drawn from the whole benchmark. The nearest
thing to an unbiased estimate of it that exists is the verification's
quasi-holdout — the four domains its prompt was not written from — at **+0.0132
over `baseline`, p=0.33**.

## If a later run does spend one

Record, before reading the result: which of the three this is, the exact
candidate and its dev numbers, why dev was not sufficient, and the command. Then
the result, whatever it is. A spend that is logged after the number is known is
not a spend, it is a rationalisation.
