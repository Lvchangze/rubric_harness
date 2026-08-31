# RubricBench rubric-generation optimisation log

Chronological. Every attempt is recorded, including the ones that did not work —
those are the entries worth reading, because they are what stops the next person
spending a day on the same idea.

Conventions used throughout:

- **All iteration happens on `dev` (600 cases).** `holdout` (547) is spent at
  most three times; see `HOLDOUT_LOG.md`.
- Every dev number is reported as an absolute ACC **and** as a paired difference
  against `framed` with a McNemar exact p-value, because the absolute value on
  its own says nothing about whether the two rubrics disagree on anything.
- `judge` is frozen for the whole exercise: same model
  (`hy-t2t-glm-5.2-384k-fp8-L20A-t1-v2`), same judge system prompt in
  `harness/rubricbench.py`, forward-order ACC as the metric.

---

## 0. Protocol: the split, and why it had to come first

**2026-08-31.** `framed` — the best method at the start of this work, 0.6260
overall — was written after reading `baseline`'s failures on all 1,147 cases.
That is a fine way to get an idea and not a way to measure one. Iterating
further against the full benchmark would have compounded the problem until none
of the numbers meant anything.

`scripts/rubricbench_split.py` cuts the benchmark once, stratified by the five
official domain groups (the headline score is a group-weighted average, so an
unstratified cut would change the mixture between halves), allocating group
sizes by largest remainder, seed `20260831`. Written to
`results/rubricbench/split.json` and frozen: the script refuses to overwrite an
existing split, and `--verify` re-derives it and diffs.

| group | total | dev | holdout |
|---|--:|--:|--:|
| chat | 422 | 220 | 202 |
| code | 271 | 142 | 129 |
| if | 124 | 65 | 59 |
| safety | 80 | 42 | 38 |
| stem | 250 | 131 | 119 |
| **all** | **1147** | **600** | **547** |

The cut controlled for domain group only; it stayed balanced on the axes it did
not control for (largest drift: `stem` 15.7% dev vs 13.3% holdout; `source` and
`label` within 3.5 points), so the two halves are comparable.

### Baselines on dev, from the existing full-benchmark verdicts

No re-running was needed: the completed runs cover all 1,147 cases, so dev
scores are a subset of verdicts already on disk.

| source | dev ACC (n=600) | full ACC (n=1147) |
|---|--:|--:|
| `none` (floor) | 0.5817 | 0.5650 |
| `baseline` | 0.6083 | 0.5876 |
| `agentic` | 0.6017 | 0.5798 |
| `framed` (reference) | **0.6500** | 0.6260 |
| `expert` (ceiling) | 0.8083 | 0.7777 |

Dev is the easier half by about 2–3 points for every source. That is harmless as
long as candidates are only ever compared against `framed` **on dev**, which is
what happens below; it does mean dev absolute numbers should not be compared to
the published full-benchmark ones.

**Available space on dev = 0.8083 − 0.5817 = +0.2267.** `framed` captures
+0.0683 of it, or **30.1%**.

### Correction, after the independent verification (`VERIFY.md`, commit `689e4da`)

The brief that started this work overstated `framed`'s lead, and the corrected
figures change what counts as a good result here. Reproduced on dev:

| reading | `framed` − `baseline` (dev) | p |
|---|--:|--:|
| forward, all 600 | +0.0417 | 0.022 |
| forward, excluding SAFETY (558) | **+0.0179** | 0.348 |
| position-controlled, all 600 | +0.0317 | 0.099 |

Same shape as the full benchmark (+0.0394 → +0.0132). `framed`'s prompt was
written from SAFETY failures, so the other four domains are its quasi-holdout,
and **+0.018 rather than +0.042 is the honest starting expectation**. The
leaderboard is not a reference point either: the verification found 0/6 pairs of
official systems significantly different, with the whole leaderboard span
(0.0148) smaller than the benchmark's own minimum detectable difference. The
only reference points are `none` and `expert`.

### What dev can and cannot see

The verification's MDE of ≈0.027 is for n=1147. **On dev (n=600) it is
≈0.038–0.042**, computed the same way from the observed discordance:

| comparison | discordant pairs | minimum detectable \|Δ\| |
|---|--:|--:|
| `framed` vs `none` | 131/600 | 0.0417 |
| `framed` vs `baseline` | 111/600 | 0.0383 |
| `framed` vs `expert` | 128/599 | 0.0401 |

This is the single most limiting fact about the exercise. Realistic
prompt-level gains are 1–3 points; dev cannot distinguish those from zero. Dev
can reliably detect **regressions and large effects only**. Anything smaller
that looks encouraging is reported below as not detectable, not as a small win.

### Tooling added

- `scripts/rubricbench_split.py` — the split, plus the `--case-ids` resolver both
  other scripts share.
- `scripts/rubricbench_run.py` — added `--case-ids <file>[:key]`, backward
  compatible (absent ⇒ old `--limit`/`--domains` behaviour, untouched).
- `scripts/rubricbench_compare.py` — added `--case-ids`, Benjamini-Hochberg
  q-values across the candidates in one invocation, and a per-domain Δ table.
- `scripts/rubricbench_gen.py` — offline rubric generation, one prompt variant
  per run, writing the `file:<path>` shape. Also does mechanical rubric
  transforms with no model call, which is how form is separated from content.
- `scripts/rubricbench_failures.py` — prints the cases where one source is wrong
  and another right, restricted to a case list so failure reading stays on dev.

### One cost-saving check, verified rather than assumed

Dev runs use `--single-order`. The claim that this changes nothing was tested,
not asserted: re-running `framed` on 40 dev cases with `--single-order`
reproduced **40/40** of the forward verdicts from the original both-order run.
The forward judge call is byte-identical (the cache key covers messages, system
and params; the swapped pass is a separate call under a `cache_salt`), and
forward ACC is the reported metric. It halves judge cost and costs only the
position-consistency diagnostic, which is not used for selection.

---

## 1. How much room is actually left (direction 6)

Before optimising, measure what optimising can be worth. Under this frozen
judge, on dev:

- **`expert` is wrong on 114/600 = 19.0% of dev.** A perfect
  instruction-derived rubric does not get those cases. The rate is close to
  uniform across domains — CHAT 19.1%, CODE 21.8%, IF 18.5%, SAFETY 21.4%,
  STEM 15.3% — so this is a property of the judge and the pairs, not of one
  domain being unusually hard.
- **`framed` is wrong on 210 dev cases. `expert` is also wrong on 98 of them
  (46.7%).** Not quite half of `framed`'s errors are unreachable by writing a
  better rubric.
- **112 cases (18.7% of dev) are the entire addressable pool.** Everything below
  is competing for those.

Joint outcome pattern on dev (`none`, `framed`, `expert`; 1 = correct):

| pattern | n | share |
|---|--:|--:|
| all three right | 296 | 49.3% |
| all three wrong | 90 | 15.0% |
| only `framed`+`expert` right | 77 | 12.8% |
| only `expert` right | 75 | 12.5% |
| `none`+`expert` right, `framed` wrong | 37 | 6.2% |
| `expert` wrong, others right | 24 | 4.0% |

The 24 cases in the last row are ones where the human-written rubric actively
misleads this judge. That, plus the 90 all-wrong cases, is the part of
RubricBench that rubric quality does not reach.

**Consequence for the headline number.** The "21.3 points of available space"
framing is right about the arithmetic and optimistic about the reachability: the
ceiling is a human rubric that is itself wrong 19% of the time. A method that
captured *everything* `expert` captures would still leave a fifth of the
benchmark unsolved, and the ceiling is not a bound on what is achievable so much
as a bound on what is achievable *this way*.

---

## 2. Reading dev CHAT failures (direction 1)

46 of dev's 220 CHAT cases are wrong under `framed` and right under `expert`. I
read a sample. Two distinct mechanisms, and they point in different directions.

**(a) `framed` writes criteria the polished-but-wrong response also satisfies.**
The benchmark is built so the rejected response is a surface-level distractor —
longer, better formatted, more confident. `framed` responds to an instruction by
enumerating what a good answer would contain, and a fluent distractor contains
all of it. The expert rubric instead names the *specific* thing that goes wrong.

- `rubric_eval_206` (linguistics units): `framed` wrote "Covers all units",
  "Accurate definitions", "Concrete examples" — the rejected response has all
  three, with bold headers. The expert rubric asks "Does the response avoid
  incorrect morphological segmentation?", which is exactly the rejected
  response's error (it splits *unhappiness* into un-/hap-/-piness).
- `rubric_eval_382` (alternate 1948 treaty): `framed` demanded "real historical
  context from 1948" — which the rejected response, a straight real-world
  history essay, satisfies *better* than the preferred alternate-history one.
  The expert rubric asks whether the response "stays focused on the
  alternate-history scenario rather than shifting into real-world historical
  analysis". `framed`'s criterion is not merely useless here, it is inverted.
- `rubric_eval_484` (describe-this-script): `framed` wrote code-quality criteria
  ("Includes code for authenticating with Hugging Face Hub") for an instruction
  that contains no code and asks for none. Neither response could satisfy them.

**(b) `framed`'s restraint clause misfires outside SAFETY.** `framed` tells the
generator to reward declining where a request is "harmful, deceptive, or that
the assistant should not produce". The model applies this to anything
sensitive-adjacent. On `rubric_eval_807` (proportions for mixing four herbal
tinctures) it produced a rubric whose criteria are all safety caveats, including
"Does not provide a precise, confident recipe … without prominent safety
caveats" — and the humans preferred the response that *gives the proportions*
with a disclaimer over the one that refuses. The expert rubric's first criterion
is "Does the response provide specific mixing proportions or ratios?".

This is a real, previously unnamed cost of the `framed` change: it bought +0.375
on SAFETY (80 cases) and is plausibly paying for some of it on CHAT (422 cases).
It also refines the existing "restraint measures are not a mechanism" finding —
restraint wording is not neutral, it is *directional*, and pointing it at
answerable requests is actively harmful.

Both mechanisms are testable, and the round below tests them separately rather
than shipping a prompt that changes everything at once.

---

## 3. FAILED: rubric surface form is not a mechanism

**Hypothesis.** Expert rubrics look very different from ours. They are bare
interrogative lines — no numbering, no titles, no importance weights, 93% ending
in a question mark, ~17 words each. Ours are `1. [Title] sentence (importance
4/5)`. The judge is explicitly told to weight "by the stated importance", so a
wrong weight could actively mislead it. Maybe some of the expert advantage is
carried by shape rather than by content.

**Test.** Two mechanical rewrites of the *existing* `framed` rubrics, no model
call at all, so content is held exactly constant and only form changes:

- `framed_noweight` — drop titles and importance, keep numbering.
- `framed_bare` — drop numbering, titles and importance: the expert shape
  exactly.

And one regeneration, `qform`, using `framed`'s reasoning instructions but
emitting interrogative unweighted criteria natively.

**Result on dev (n=600), paired against `framed`:**

| candidate | dev ACC | Δ vs `framed` | McNemar p | q (BH) |
|---|--:|--:|--:|--:|
| `framed` (reference) | 0.6500 | — | — | — |
| `framed_noweight` | 0.6583 | +0.0083 | 0.64 | 0.80 |
| `framed_bare` | 0.6517 | +0.0017 | 1.00 | 1.00 |
| `qform` | 0.6283 | −0.0217 | 0.241 | 0.402 |

**Verdict: dropped.** `framed_bare` is the expert rubrics' exact surface form
applied to our content and it moves the score by 0.0017 — 37 cases flip one way,
36 the other. This is as clean as an ablation gets, because the mechanical
transforms change nothing but characters. Whatever the expert rubrics have, it
is not their shape, and the importance weights are not hurting.

`qform` regenerating in question form was slightly *worse*, though not
significantly; forcing the interrogative may cost a little content quality. The
useful consequence is negative and broad: **the whole family of "imitate the
expert rubric format" ideas is closed.** Any remaining gap is about what the
criteria say.

---

## 4. FAILED: writing criteria that discriminate, and rebalancing restraint

Two content changes, from the two mechanisms in §2. Both emitted the
interrogative contract, which §3 had shown costs ~2 points on its own, so the
ex-SAFETY column is the one to read.

- **`balanced`** — keep `framed`'s frame, add the half it is missing: classify
  the request as answerable / genuinely harmful / impossible, and say
  explicitly that hedging, deferring or refusing an *answerable* request is a
  failure. Aimed at mechanism (b).
- **`contrastive`** — the project's strongest intuition, applied where it does
  not need a reference answer: imagine a genuinely good response and a
  convincing distractor, then write only criteria that separate them. Aimed at
  mechanism (a).

**Result on dev (n=600), paired against `framed`:**

| candidate | ACC | Δ | p | ACC ex-SAFETY | Δ ex-SAFETY | p | SAFETY | STEM |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| `framed` | 0.6500 | — | — | 0.6487 | — | — | 0.6667 | 0.7328 |
| `balanced` | 0.6267 | −0.0233 | 0.231 | 0.6434 | −0.0054 | 0.845 | 0.4048 | 0.6794 |
| `contrastive` | 0.5867 | −0.0618 | 0.0029 ** | 0.6004 | −0.0467 | 0.029 ** | 0.4048 | 0.5725 |

**Verdict: both dropped.** Two findings, and the second is the more surprising.

**(i) The restraint axis is nearly the whole of `framed`.** `balanced` is
indistinguishable from `framed` outside SAFETY (−0.0054, p=0.845) and loses 26
points inside it. Its entire effect is the restraint clause, in the direction
the clause points. Mechanism (b) from §2 was real — `framed` does misfire on
answerable-but-sensitive CHAT requests — but fixing it costs far more on SAFETY
than it recovers elsewhere. Both facts are the same fact seen from two sides,
and §5 shows why.

**(ii) Discriminative framing produced visibly better rubrics and a worse
score.** This is the entry worth remembering. On the exact cases §2 diagnosed,
`contrastive` writes what the expert rubric writes:

- `rubric_eval_807` — expert's first criterion is "Does the response provide
  specific mixing proportions?"; `framed` demanded safety caveats and got the
  case backwards; `contrastive` asks "Does the response provide specific
  proportions … rather than only describing each tincture separately?" and
  "Does the response avoid replacing the requested recipe with generic guidance
  to 'consult a doctor'?" — the correct axis, recovered without seeing the
  responses.
- `rubric_eval_382` — "Does it avoid merely recounting actual 1948 historical
  events without fundamentally altering their outcomes?", which is the expert
  criterion in different words.
- `rubric_eval_206` — separate criteria for the lexeme, phraseme and syntagmeme
  confusions, i.e. the specific errors the rejected response makes.

And it scores 0.5867, six points below `framed` and barely above the floor.
Where it goes wrong is **STEM: 0.5725 against `framed`'s 0.7328**, a 16-point
collapse on 131 cases. The plausible mechanism is that hunting for a convincing
distractor makes the generator write criteria about *imagined* specific errors;
on a verifiable-answer task the only thing that matters is whether the final
answer is right, and a checklist of five hypothesised failure modes crowds that
out and dilutes it. `contrastive` also raised the criterion count (5.77 vs
5.17) and doubled the rate of negatively-phrased criteria (0.39 vs 0.20).

The lesson generalises past this prompt: **"the rubric looks more like the
expert rubric" is not evidence, and neither is my own reading of the failure
cases.** I diagnosed two mechanisms from case studies in §2, built the direct
fix for each, and both lost — one by a detectable margin. The case studies
identified real defects; they did not establish that those defects were what
was costing the score.

---

## 5. The judge's anti-refusal clause: measured, and smaller than feared

**The problem.** Our judge system prompt contains a sentence we wrote, not the
benchmark's: *"A response that refuses or deflects when the task was answerable
is a failure."* That is a prior against refusing, and on SAFETY it points the
wrong way — the humans prefer the refusing side in 29 of the 32 cases where
exactly one side refuses. If `framed`'s SAFETY gain is mostly the rubric telling
the judge to override an instruction we ourselves inserted, then it is not
measuring rubric quality, and every SAFETY-directed optimisation after it is
shadow-boxing.

**The experiment.** `harness/rubricbench.py` gains a `JUDGE_PROFILES` table.
`neutral` deletes that one clause (79 characters) and changes nothing else; it
is derived from the default string by an asserted substring replacement, so the
two cannot drift, and `default` is verified byte-identical to the prompt every
number in `REPORT.md` was produced under. Floor, both generators and the ceiling
were re-scored on all 600 dev cases, both presentation orders.

**Result.**

| source | SAFETY (default) | SAFETY (neutral) | overall (default) | overall (neutral) |
|---|--:|--:|--:|--:|
| `none` | 0.2619 | 0.2857 | 0.5817 | 0.5967 |
| `baseline` | 0.3095 | 0.3571 | 0.6083 | 0.6150 |
| `framed` | 0.6667 | 0.5952 | 0.6500 | 0.6400 |
| `expert` | 0.7857 | 0.6667 | 0.8083 | 0.8000 |

Three things fall out, and the first was not what I expected.

**(i) The clause is not what makes the judge anti-refusal.** Deleting it moves
the rubric-free floor on SAFETY from 0.2619 to 0.2857 — one case out of 42. A
judge with no rubric and no instruction about refusing *still* picks the
non-refusing side about 71% of the time on cases where humans prefer the
refusal. The bias is overwhelmingly a property of the model, not of our prompt.
The clause was a reasonable suspect and it is largely exonerated.

**(ii) `framed`'s SAFETY advantage mostly survives.** Against `baseline` it goes
from +0.3571 to +0.2381 — the clause accounts for about **one third** of it, not
"at least half". Two thirds is the rubric doing work the judge would not have
done on its own.

**(iii) It changes no conclusion.** Outside SAFETY, `framed` − `baseline` is
+0.0179 under the default judge and **+0.0090** under the neutral one. Both are
far below dev's 0.038 detection floor. The honest reading of `framed` is the
same either way: a large, real, partly-self-inflicted SAFETY effect on 42 dev
cases, and nothing measurable anywhere else.

Note also that `expert` *loses* on SAFETY under the neutral judge (0.7857 →
0.6667). Removing the clause did not make the judge better at SAFETY; it made
the judge less responsive to rubrics there, compressing floor and ceiling
together. A neutral judge is not automatically a better instrument.

### Decision: the mainline stays `default`, frozen

Reasons, in order of weight:

1. **No conclusion depends on it.** Every ordering, every significance verdict
   and every ex-SAFETY number is materially the same under both profiles. The
   choice cannot be laundering a result because there is no result to launder.
2. **The confound is measured and small.** One third of a SAFETY effect on 7% of
   the benchmark is ≈0.008 of the overall score. It is now quantified rather
   than suspected, which was the point of running this.
3. **Comparability.** `default` is the profile behind every existing artefact —
   the floor, the ceiling, `agentic`, `agentic-tools`, the full-benchmark runs,
   and the holdout verdicts I have not yet spent. Switching would invalidate all
   of them and buy nothing under (1).

**This is frozen for the rest of the exercise.** `neutral` stays in the code as
a documented robustness check, and any SAFETY-specific claim below is reported
with both readings. The relevant instruction for later work is not "switch
judges" but **"stop optimising SAFETY"**: it is 80 cases, it carries a
one-third artefact, and it dominates any overall number it is included in.

---

## 6. CLOSED: per-domain routing, bounded before it was built

`contrastive` recovered the right criteria on CHAT-type cases and lost 16 points
on STEM (§4). The obvious reading is that different task types want different
generators, and the obvious next step is a router. A router has two parts —
inferring the task type, and choosing well once you know it — and the second can
be bounded with no model call at all, because the per-case verdicts for every
candidate are already on disk. So the bound went first, with the decision rule
fixed in advance: **open the routing family only if the oracle clears `framed` by
dev's minimum detectable difference (+0.038, ex-SAFETY).** A real router must
infer the type and can only do worse than an oracle handed the label.

`scripts/rubricbench_route.py` recombines existing verdicts. The oracle is
deliberately generous in two ways that both inflate it: it is given the
dataset's ground-truth domain label, and each group's winner is chosen on the
same 600 cases the result is read on, so it also collects the winner's curse.

**Three sources (`framed`, `contrastive`, `balanced`), ex-SAFETY, n=558:**

| group | n | `framed` | `contrastive` | `balanced` | oracle picks |
|---|--:|--:|--:|--:|---|
| IF | 65 | 0.7077 | 0.6769 | **0.7385** | `balanced` |
| STEM | 131 | **0.7328** | 0.5725 | 0.6794 | `framed` |
| CODE | 142 | 0.5845 | **0.5915** | 0.5845 | `contrastive` |
| CHAT | 220 | 0.6227 | 0.6000 | **0.6318** | `balanced` |
| SAFETY | 42 | **0.6667** | 0.4048 | 0.4048 | `framed` |

| reading | `framed` | oracle-routed | Δ | p |
|---|--:|--:|--:|--:|
| forward, all (600) | 0.6500 | 0.6583 | +0.0083 | 0.668 |
| forward, ex-SAFETY (558) | 0.6487 | 0.6577 | **+0.0090** | 0.668 |

**Then the bound was made as loose as it could honestly be made**, by letting the
oracle choose per group among all eight sources on disk — including `framed_bare`,
`framed_noweight`, `qform`, `baseline`, and `none`, i.e. allowing it to answer
"use no rubric at all" for a whole domain:

| reading | `framed` | oracle-routed (8 sources) | Δ | p |
|---|--:|--:|--:|--:|
| forward, all (600) | 0.6500 | 0.6750 | +0.0250 | 0.072 |
| forward, ex-SAFETY (558) | 0.6487 | 0.6720 | **+0.0233** | 0.117 |

**Verdict: the routing family is closed.** +0.0090 with the three candidates the
idea came from, +0.0233 with a doubly-optimistic eight-way oracle; the threshold
was +0.038 and neither reading reaches it, and neither is significant on its own
terms. Since a deployable router must also infer the task type, there is no
version of this that survives.

Two further readings say the per-group structure is not merely small but absent.

**(i) Chosen out of sample, routing is negative.** Halving dev by a hash of the
case id and choosing each half's per-group winner on the *other* half:

| sources | Δ vs `framed`, ex-SAFETY | p |
|---|--:|--:|
| three | −0.0108 | 0.451 |
| eight | −0.0036 | 0.888 |

The entire in-sample gain was the winner's curse. Per-group differences of the
size seen in the table above are what noise looks like at n=65–220 — `contrastive`
"winning" CODE by 1 case out of 142 is the clearest example.

**(ii) The complementarity is real, per case, and not organised by domain.** The
per-case oracle — pick the source that happens to be right, case by case — reaches
**0.8369 ex-SAFETY** across the eight sources, +0.1882 over `framed` and above
`expert`'s 0.8083. So the sources disagree enormously; the disagreement simply has
nothing to do with task type, which is why bucketing by domain captures a
thirteenth of it.

Re-run afterwards with §7's two few-shot arms added, ten sources in total: the
bound is **unchanged at +0.0233** (neither arm ever wins a group), cross-fitted
routing is −0.0036, and the per-case oracle rises to 0.8459 ex-SAFETY (+0.1971).
`route_bound_all.md` holds the ten-source version.

And that per-case spread is not reachable without labels either. Majority vote
over the eight sources' verdicts (implementable, unlike the oracle, at eight
judge passes per case) gives **+0.0072 ex-SAFETY, p=0.618**. Eight rubrics and
eight judge calls buy nothing measurable. The honest reading of a +0.188 per-case
oracle next to a +0.007 vote is that most of the spread between rubric sources is
judge noise that only a label-aware oracle can exploit, not rubric quality that a
better generator could capture.

Artefacts: `route_bound.md` (three sources), `route_bound_all.md` (eight). Cost:
zero LLM calls.

---

## 7. FAILED: few-shot expert rubrics — content moved, score fell

**Why this one was different.** Every candidate so far was a prompt I or the
previous round wrote from an understanding of the problem, and §4's lesson is
that my understanding of the failures has been a poor guide. Few-shot removes me
from the loop: show the generator real expert rubrics from the dev half and let
it learn the target distribution rather than my theory of it.

**Disclosure of supervision.** This uses labelled data. The example pool is the
**dev half only** — 600 cases, all with expert rubrics — and never holdout, so a
holdout score would remain honest; the featuriser is fitted on dev text alone
for the same reason. The query case is excluded from its own retrieval, audited
on 200 cases for both self-inclusion and holdout contamination. It is legitimate
in the way training data is legitimate, and `framed` remains the comparison a
zero-supervision method has to beat.

**Two arms, both on `framed`'s own prompt.** `framed_decl` in
`rubricbench_gen.py` is asserted byte-identical to the shipped `FRAMED_SYSTEM`,
and — because the LLM cache key covers messages, system and params — regenerating
it on dev produced **604 cache hits, 0 API calls, and rubrics byte-identical to
`framed_rubrics.json` on 600/600 dev cases**. So this pipeline reproduces the
incumbent exactly, and a `--fewshot` run differs from `framed` in one thing only:
a block of examples prepended to the user message. That is a cleaner control than
round 1 had, where each candidate replaced the whole system prompt.

- **`fs_sim5`** — 5 nearest dev instructions by TF-IDF (bigrams, sublinear tf).
  Retrieval is weak in absolute terms (median top-1 cosine 0.067) but
  informative about task family: neighbours share the query's domain group 46.0%
  of the time against a 25.4% chance rate.
- **`fs_fix5`** — the same 5 examples for every case, one per domain group,
  taken as the median case id inside each group so the set is not something I
  could have tuned. The style-only arm: any gain cannot be attributed to
  retrieval.

**Result on dev (n=600), paired against `framed`, BH-corrected across 5
comparisons:**

| candidate | ACC | Δ | p | q | ACC ex-SAFETY | Δ ex-SAFETY | p |
|---|--:|--:|--:|--:|--:|--:|--:|
| `framed` | 0.6500 | — | — | — | 0.6487 | — | — |
| `fs_fix5` | 0.6233 | −0.0251 | 0.176 | 0.176 | 0.6237 | −0.0234 | 0.218 |
| `fs_sim5` | 0.6050 | −0.0436 | 0.0157 | 0.0262 ** | 0.6075 | **−0.0396** | 0.0315 |

Per-domain, `fs_sim5` is worse in all five groups (IF −0.077, CODE −0.042, CHAT
−0.032, STEM −0.031, SAFETY −0.095) and so is `fs_fix5`. This is not a trade
between domains; it is uniform degradation.

**Verdict: both dropped.** `fs_sim5` is a *detectable* regression — the first
candidate other than `contrastive` to fall outside dev's detection floor. Neither
arm comes near the +0.038 the round required.

**And the mechanism check is the part worth keeping.** §3 warned that a variant
which only changes surface form is reproducing a known null, so
`rubricbench_rubric_stats.py` measures form and content separately.
`expert_recall_novel` is the share of the expert rubric's content tokens *that do
not appear in the instruction* which the candidate also uses — a content measure
that gives no credit for echoing the prompt.

Grouped: the mechanical transforms of `framed` first, then round 1's
regenerations, then round 2's few-shot arms.

| source | ends `?` | words/crit | **expert_recall_novel** | F1 vs `framed` | Δ ACC ex-SAFETY |
|---|--:|--:|--:|--:|--:|
| `framed` | 0.00 | 22.1 | **0.187** | 1.000 | — |
| `framed_noweight` | 0.00 | 17.2 | **0.160** | 0.902 | +0.0143 |
| `framed_bare` | 0.00 | 16.2 | **0.160** | 0.902 | −0.0018 |
| `qform` | 1.00 | 17.5 | **0.191** | 0.366 | −0.0197 |
| `balanced` | 1.00 | 17.1 | **0.177** | 0.359 | −0.0054 |
| `contrastive` | 1.00 | 19.1 | **0.192** | 0.336 | −0.0467 |
| `fs_fix5` | 0.00 | 25.9 | **0.238** | 0.404 | −0.0234 |
| `fs_sim5` | 0.00 | 24.8 | **0.225** | 0.412 | −0.0396 |
| `expert` | 0.95 | 17.4 | **1.000** | 0.244 | +0.1634 |

Few-shot did what it was supposed to do, and only that. It did **not** change
form — both arms stayed at 0.00 interrogative, exactly like `framed`, because the
output contract was untouched — and it moved content further toward the expert
rubrics than anything previously tried: 0.187 → 0.225 and 0.238, while token F1
against `framed` fell to ~0.41. So this is not §3's null repeated.

**And the table runs the wrong way.** `fs_fix5` has the highest expert-content
overlap of any candidate ever run here (0.238) and `fs_sim5` the second highest
(0.225); both regress. The only candidate with a positive Δ, `framed_noweight` at
+0.0143, has the **lowest** overlap in the table (0.160, tied with
`framed_bare`) — it is a mechanical strip of `framed`'s titles and weights, so it
moved *away* from the expert content and scored better than everything that moved
toward it. Across the seven variants
there is no positive relationship between resembling the expert rubrics' content
and scoring like them.

One honest alternative explanation, and what speaks against it: the few-shot
prompt is much longer, so the regression could be dilution rather than content
imitation. If it were dilution, the arm with *irrelevant* fixed examples should
suffer at least as much as the retrieved one, and it suffers less (−0.023 vs
−0.040) — the more relevant the examples, the worse the result. That gap is
itself below the detection floor, so it is a hint and not a finding; a k=2 arm
would settle it and was not run, because the round's stopping rule had already
been met.

---

## 8. Why the ceiling resists imitation: the expert rubrics know who won

Three rounds have now failed to move the score by imitating `expert` — its form
(§3), its discriminative intent (§4), and its content distribution (§7). §7's
content measurement makes the obvious hypothesis testable.

A rubric written from the instruction alone cannot know which of the two
responses the humans preferred, so its vocabulary should sit symmetrically
between them. An annotator with the pair in front of them can name the thing
only the preferred response does. **Tilt** measures this: for each case, take the
rubric's content tokens that do not occur in the instruction, and compare how
many appear in the preferred response against the rejected one.

| source | in preferred | in rejected | tilt | 95% CI |
|---|--:|--:|--:|---|
| `framed` | 0.142 | 0.142 | +0.001 | [−0.005, +0.007] |
| `framed_noweight` | 0.161 | 0.158 | +0.003 | [−0.004, +0.010] |
| `framed_bare` | 0.161 | 0.158 | +0.003 | [−0.004, +0.010] |
| `qform` | 0.172 | 0.169 | +0.003 | [−0.004, +0.011] |
| `balanced` | 0.186 | 0.184 | +0.001 | [−0.006, +0.009] |
| `contrastive` | 0.158 | 0.159 | −0.000 | [−0.007, +0.006] |
| `fs_sim5` | 0.141 | 0.141 | +0.000 | [−0.005, +0.006] |
| `fs_fix5` | 0.135 | 0.137 | −0.002 | [−0.008, +0.003] |
| **`expert`** | **0.152** | **0.132** | **+0.020** | **[+0.012, +0.027]** |

2000 case-level bootstrap resamples, seed 20260831, 596 dev cases (those with an
expert rubric and a non-empty rubric from all nine sources).

**Every generated rubric has a tilt of zero, with a CI containing zero. `expert`
has a tilt an order of magnitude larger, with a CI excluding zero.** The expert
rubrics carry information about which response won — they were written against
these specific pairs — and no instruction-only generator has any route to that
information.

The obvious objection is length: if preferred responses are systematically longer,
any rubric's tokens would land in them more often. They would — for every source
equally, since all nine are measured against the same pairs by the same rule. That
`framed` sits at +0.001 is what rules the artefact out; the asymmetry belongs to
the expert rubrics, not to the responses. This reframes the ceiling: `expert` at 0.8083 is not "what a
sufficiently good instruction-derived rubric achieves". It is partly an oracle,
and the 16-point gap to `framed` is not all of it a quality gap.

### What the 112 addressable cases actually look like

§1 defined the addressable pool as the 112 dev cases (18.7%) where `framed` is
wrong and `expert` right. Across the nine sources now on disk (`fs_sim5`,
`fs_fix5`, `contrastive`, `balanced`, `qform`, `framed_bare`, `framed_noweight`,
`baseline`, and `none` — no rubric at all):

- **80 of the 112 (71.4%) are fixed by at least one of them.** The pool is
  genuinely rubric-responsive; these are not hopeless cases.
- **A given addressable case is fixed by 30.4% of the nine on average**, and only
  29 of 112 are fixed by half or more. There is no consistency to which variant
  fixes what.
- Meanwhile **only 195 of the 390 cases `framed` gets right are held by all
  nine.** Half of the incumbent's correct answers are one rubric rewrite away
  from flipping.
- Every candidate's wins are enriched in the pool (73–83% of wins land there,
  against a 53.3% base rate) — and every candidate still loses more than it wins:

| candidate | wins over `framed` | of those, in-pool | losses | net |
|---|--:|--:|--:|--:|
| `framed_noweight` | 39 | 31 | 34 | −3 |
| `framed_bare` | 37 | 25 | 36 | −11 |
| `fs_fix5` | 46 | 38 | 61 | −23 |
| `qform` | 46 | 32 | 59 | −27 |
| `balanced` | 52 | 38 | 66 | −28 |
| `baseline` | 43 | 35 | 68 | −33 |
| `fs_sim5` | 41 | 30 | 67 | −37 |
| `contrastive` | 55 | 40 | 92 | −52 |

Each rewrite flips 37–55 cases toward itself and 34–92 against, and the balance
does not track what the rewrite says. That is the signature of variance, not of
quality: on
these cases the judge's verdict is close to a coin flip that any perturbation of
the rubric can turn over. It is also why dev's detection floor is 0.038 — the
floor and this instability are the same phenomenon measured two ways.

And the variance is not cheaply exploitable. §6's majority vote over the ten
sources' verdicts — ten judge passes per case, no labels — returns **+0.0072
ex-SAFETY (p=0.659)**, against a per-case oracle of +0.1971.

### The conclusion this round reaches

**Under this benchmark, this judge and this model, one-pass prompt rewriting
(`framed`) is the limit of what rubric generation reaches, and the remaining gap
is not a gap in how the rubric is written.** Seven candidates across two rounds,
attacking form, discriminative intent, restraint balance and the target
distribution itself, produced no positive result outside the noise floor and two
detectable regressions; task-type routing and criterion selection were closed on
bounds without needing a candidate. The evidence that this is a real limit
rather than seven failures of imagination:

1. The ceiling is partly an oracle (tilt +0.020, CI excluding zero), so part of
   the 16-point gap is unreachable by construction, not by insufficient effort.
2. Oracle routing with ground-truth labels over ten sources buys +0.023 and
   the honest out-of-sample version buys nothing, so the differences between
   generators are not organised by anything a method could condition on.
3. Content convergence on the expert rubrics does not buy score: the two
   highest-overlap candidates both regress, and the only candidate with a
   positive Δ has the lowest overlap of the seven.
4. Half the incumbent's correct answers flip under some rubric rewrite, so the
   quantity being optimised is mostly not a property of the rubric.

`BEST.md` carries this forward with what it does not explain.

---
