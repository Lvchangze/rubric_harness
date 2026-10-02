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

## GLM-5.3 re-run (2026-09-30): no spend

All six sources were re-generated and re-judged with GLM-5.3 on the dev half
only. The rule, fixed before the run, was to spend one holdout score only if
`agentic-tools` beat `baseline` on dev by at least the dev detection limit
(about 0.037). It did not: it came out 0.0134 *below* `baseline` (p=0.50).
Nothing on the holdout was computed or read. Still **0 of 3** used.

## Distill variant (2026-10-01): no spend

`agentic-tools-distill` adds one call after the lint that rewrites the verified
checklist into 3-7 evaluative criteria. Same rule as above, fixed before the
run. On dev it came out +0.0101 over `baseline` (p=0.64) against a detection
limit of 0.037. Nothing on the holdout was computed or read. Still **0 of 3**.

## Distill v2 (2026-10-01): no spend

`agentic-tools-distill2` keeps verified concrete correctness checks for code
and other checkable outputs. Same rule, fixed before the run: −0.0067 against
`baseline` on dev (p=0.77). Two distill variants were tried on dev in total.
Nothing on the holdout was computed or read. Still **0 of 3**.

## Self-evolution (2026-10-01): not spent

`scripts/rubricbench_evolve.py` evolved the writer prompt for 5 rounds, 15
candidates, on a 400-case train slice of dev with a 200-case val slice held
back. The plan announced before the run was to price the final candidate on
the holdout. The final candidate (`r1c2`) then scored 0.5850 on val, below
both the seed (0.6050) and `baseline` (0.6000), so the spend was put to the
user rather than made automatically. Nothing on the holdout was computed or
read. Still **0 of 3**.

## Evolution loop (2026-10-01 to 10-02): not spent

`scripts/rubricbench_evolve_loop.py` ran 17 rounds (67 candidates) before the
user stopped it at the start of round 18. Its gate, fixed before the run, sent a
beam member to the holdout only if it beat `baseline` by 0.02 on the train slice
and 0.03 on the val slice. Four cleared the train margin; on val they scored
0.5750, 0.5600, 0.5750 and 0.5900 against `baseline`'s 0.6000. None reached the
holdout. Still **0 of 3**.

## Contrastive stage (2026-10-02): rule fixed before the dev run

`agentic-tools-contrast` swaps the distill rewrite for an empirical test. After
the lint it writes one GOOD response and two TEMPTING ones, which look better at
a glance but are worse in a way that matters (`SIMULATE_SYSTEM`, sha256
`6b53f9c0b216`). It drafts criteria meant to prefer GOOD (`CONTRAST_DRAFT_SYSTEM`,
`4f99d2ac4f47`), then runs every draft criterion on all three responses with the
bench's literal-truth judge prompt. Criteria GOOD satisfies and at least one
TEMPTING response fails are kept first, weighted at least 4. If fewer than 3
survive, criteria GOOD satisfies that no TEMPTING response fails fill the gap,
weighted at most 3. At most 7 are kept. If nothing separates the responses, the
finalised rubric is kept unchanged.

Before this entry it had only been run on the three dev cases in
`/tmp/rb_smoke3.json`. That run exposed a format bug: the simulator returned one
TEMPTING response instead of two in 2 of 3 cases, and the output instruction was
fixed. The first smoke run printed a 3-case accuracy (2/3). At n=3 that number
carries no information, and no decision used it.

Rule, fixed now: one run on the 600 dev cases, with the same judge and protocol
as every other GLM-5.3 dev number. Forward accuracy is compared with `baseline`
(0.6117), paired, using an exact McNemar test. Holdout spend #1 happens only if
Δ ≥ +0.037 and p < 0.05. If it happens, the entry is written here first. Then
contrast rubrics are generated for the 547 holdout cases (generation computes no
score), and they are scored together with the `baseline` rubrics that already
exist. The test is an exact McNemar at α/3 = 0.0167. Otherwise nothing is spent
and the dev number is reported as it is. Any prompt change after the dev number
is a new variant, and the number of variants tried is reported with it.

**Outcome (2026-10-02 20:34): not spent.** Dev forward accuracy 0.6300 against
`baseline` 0.6117. On the 596 cases both answered, 68 were right only under
contrast and 55 only under `baseline`: Δ +0.0218, exact McNemar p = 0.279. The
detection limit for this pair is 0.0386. The rule needed Δ ≥ +0.037 and p < 0.05,
and neither condition held. Nothing on the holdout was computed or read. Still
**0 of 3**.

## Contrastive variants 2 and 3 (2026-10-02): rule fixed before either run

Both were chosen after reading the dev result above, so they are the second and
third tries of this family, and their dev numbers are optimistic by construction.
Prompts are unchanged (`SIMULATE_SYSTEM` `6b53f9c0b216`, `CONTRAST_DRAFT_SYSTEM`
`4f99d2ac4f47`).

- **`agentic-tools-contrast2`** answers the STEM/CODE shortfall. Its rubric is the
  `agentic-tools-contrast` rubric from the 2026-10-02 run. For cases whose domain
  group is `stem` or `code` (273 of 600), it adds the two highest-weight
  positive criteria of the same run's upstream rubric (`pre_final`). Any
  upstream criterion whose word-set Jaccard overlap with an already kept
  criterion is ≥ 0.5 is skipped. It is built from the saved artefacts with the
  same function the generator uses (`keep_upstream`), so the upstream is
  identical to the first variant's, and then judged. The domain group is an
  input field of each case, not a label.
- **`baseline-contrast`** answers the upstream gap. The contrastive stage runs on
  the GLM-5.3 `baseline` rubric (the cached call behind the `baseline` row,
  byte-identical) instead of the agentic upstream. Nothing else changes. If the
  stage fails, the case keeps the `baseline` rubric text unchanged.

Rule: each variant is compared with `baseline` on dev, paired, exact McNemar. A
variant qualifies only if Δ ≥ +0.037 and p < 0.05. This family spends at most
one holdout. If both qualify, the one with the larger dev Δ goes. The holdout
test is an exact McNemar against `baseline` at α/3 = 0.0167, logged here before
anything is scored.

**Outcome (2026-10-02 23:34): not spent.** Neither variant qualified.

| variant | dev ACC | only it right / only `baseline` right | Δ | p |
|---|--:|--:|--:|--:|
| `agentic-tools-contrast2` | 0.6400 | 74 / 56 | +0.0302 | 0.136 |
| `baseline-contrast` | 0.6317 | 53 / 40 | +0.0218 | 0.213 |

Three variants of the contrastive stage have now been tried on dev. Nothing on
the holdout was computed or read. Still **0 of 3**.

## If a later run does spend one

Record, before reading the result: which of the three this is, the exact
candidate and its dev numbers, why dev was not sufficient, and the command. Then
the result, whatever it is. A spend that is logged after the number is known is
not a spend, it is a rationalisation.
