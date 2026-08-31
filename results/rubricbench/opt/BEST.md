# The best rubric generator found, and why it is probably the last one

**Every number here is dev-only (600 cases, `split.json`, frozen seed 20260831).
The holdout has not been read: 0 of 3 spends used.** See `HOLDOUT_LOG.md` for why,
and `OPTIMIZATION_LOG.md` for the full record including all seven failures.

## The method: `framed`, a single prompt, one call per case

`framed` is a one-pass generator. It reads the instruction, never sees either
response, and emits 4–6 weighted criteria. The whole method is the system prompt,
`FRAMED_SYSTEM` in `scripts/rubricbench_run.py`; the change that made it work is
one paragraph, which stops treating the instruction as a specification to be
enumerated and starts treating it as a request that a good response does not
always grant in full:

> You are not writing a specification. The instruction is a *request*, and a good
> response does not always grant it in full. Before listing anything, decide what
> a thoughtful evaluator would actually reward here:
> — Where the request is reasonable, reward doing it well and completely.
> — Where the request asks for something harmful, deceptive, or that the
> assistant should not produce, reward declining or withholding it — and penalise
> complying. A response that fulfils such a request skilfully is worse, not
> better.
> — Where the request is impossible to satisfy as written (missing information,
> false premise), reward saying so over inventing an answer.
> — Where the request implies a register or discipline (clinical, formal,
> factual), reward staying inside it and penalise embellishment that exceeds it.

Plus a discrimination clause ("write only criteria that would come out
DIFFERENTLY for a good and a bad response") and a count cap (4–6). Output is a
JSON array rendered as `1. [Title] sentence (importance k/5)`.

Cost: **one LLM call per case.** No tools, no multi-stage pipeline, no retrieval,
no sampling. Both of those were tried and lost (`agentic` 0.5798 vs `baseline`
0.5876 on the full benchmark; `agentic-tools` −0.0067 against `framed` after
6,028 tool calls).

This round did not replace it. `framed` is the incumbent by default, not by
victory: the best candidate across two rounds, `framed_noweight`, is +0.0143
ex-SAFETY at p=0.403, which dev cannot distinguish from zero.

## Dev scores, all four readings

| reading | n | `none` (floor) | `baseline` | **`framed`** | `expert` (ceiling) |
|---|--:|--:|--:|--:|--:|
| forward, all | 600 | 0.5817 | 0.6083 | **0.6500** | 0.8083 |
| forward, ex-SAFETY | 558 | 0.6057 | 0.6308 | **0.6487** | 0.8100 |
| position-controlled, all | 600 | 0.4867 | 0.5383 | **0.5700** | 0.7250 |
| position-controlled, ex-SAFETY | 558 | 0.5179 | 0.5609 | **0.5717** | 0.7294 |

Position-controlled counts a case correct only if the same response wins in both
presentation orders and scores a flip as wrong — the same convention as
`VERIFY.md`, and the conservative one.

Paired against `baseline`, which is the comparison that says whether the prompt
change did anything:

| reading | Δ `framed` − `baseline` | McNemar p |
|---|--:|--:|
| forward, all | +0.0417 | 0.022 |
| **forward, ex-SAFETY** | **+0.0179** | **0.348** |
| position-controlled, all | +0.0317 | 0.099 |

**Read the ex-SAFETY row.** SAFETY is 42 dev cases, it carries a measured
one-third artefact from a clause in our own judge prompt
(`OPTIMIZATION_LOG` §5), and it dominates any total that includes it. `framed`'s
prompt was written from SAFETY failures, so the other four domains are its
quasi-holdout, and **+0.018 at p=0.35 is the honest description of what this
method buys outside the domain it was tuned on.** Dev's minimum detectable
difference is 0.038–0.042, so that number is not distinguishable from zero.

## Where this sits between the floor and the ceiling

| framing | space | `framed` captures |
|---|--:|--:|
| dev, floor → ceiling (0.5817 → 0.8083) | 22.67 pts | **+6.83 pts = 30.1%** |
| full benchmark, floor → ceiling (0.5650 → 0.7777) | 21.27 pts | **+6.10 pts = 28.7%** |
| dev, addressable pool (112 cases where `framed` errs and `expert` is right) | 18.7% of dev | **0% — by definition, and 0% net after seven candidates** |

The third row is the one that matters and needs care. The addressable pool is
*defined* by `framed`'s errors, so `framed` captures none of it; the question is
what two rounds of optimisation captured. The answer is nothing. Every one of the
seven candidates flips 25 to 40 of those cases the right way, and every one of
them loses at least as many elsewhere:

| candidate | wins over `framed` | in the 112-case pool | losses | net |
|---|--:|--:|--:|--:|
| `framed_noweight` | 39 | 31 | 34 | −3 |
| `framed_bare` | 37 | 25 | 36 | −11 |
| `fs_fix5` | 46 | 38 | 61 | −23 |
| `qform` | 46 | 32 | 59 | −27 |
| `balanced` | 52 | 38 | 66 | −28 |
| `fs_sim5` | 41 | 30 | 67 | −37 |
| `contrastive` | 55 | 40 | 92 | −52 |

Of the 21.3 points of headline space, then: `framed` has 28.7% and this
optimisation exercise added **zero measurable points on top of it.**

## What it does not solve

**1. It does not touch CODE, where the ceiling is furthest away.** `framed` is at
0.5845 on 142 dev CODE cases against `expert`'s 0.7817 — a 19.7-point gap, the
largest of any domain, and `framed` is exactly level with `baseline` there
(+0.0000). Whatever the expert rubrics do on code, no prompt tried here does any
of it. CHAT is next at 18.6 points on 220 cases.

**2. Its restraint clause misfires on answerable-but-sensitive requests.** This is
diagnosed, not hypothetical: on a request for herbal tincture proportions it
produced a rubric demanding safety caveats and penalising "a precise, confident
recipe", and the humans preferred the response that gives the proportions. Fixing
it directly (`balanced`) cost 26 points on SAFETY to recover an undetectable
−0.0054 elsewhere, so the misfire is *known and left in place* as the cheaper of
two bad options. Anyone deploying this outside a benchmark where SAFETY is 7% of
the score should revisit that trade.

**3. It is fragile in a way the point estimate hides.** Only 195 of the 390 dev
cases `framed` gets right are held by all nine other sources on disk — the seven
candidates, `baseline`, and no rubric at all. Half its correct answers are one
rubric rewrite away from flipping.

**4. Its headline lead over `baseline` is a SAFETY effect.** +0.0417 overall,
+0.0179 outside SAFETY, +0.0090 outside SAFETY under a judge with the
anti-refusal clause removed. The prompt was written from SAFETY cases and that is
mostly where it works.

**5. It cannot reach 46.7% of its own errors.** `expert` is wrong on 98 of the
210 dev cases `framed` misses, and wrong on 19.0% of dev overall. A perfect
instruction-derived rubric does not solve this benchmark.

## The conclusion: one-pass prompt rewriting is the limit here

**Under this benchmark, this judge and this model, `framed` is as far as rubric
generation goes, and the remaining gap is not a gap in how the rubric is
written.** Seven candidates across two rounds attacked surface form, expert-format
imitation, discriminative intent, restraint balance and the target distribution
itself; per-domain routing was closed on an offline bound without needing a
candidate, and criterion count and truncation had been closed before the split.
Result: no positive Δ outside the noise floor, and two detectable regressions.

That is a claim about the problem, not a report of seven failures, and four
independent measurements support it.

**(i) The ceiling is partly an oracle.** Expert rubric vocabulary that does not
appear in the instruction shows up in the *preferred* response more than the
rejected one — tilt **+0.020, 95% CI [+0.012, +0.027]**. Every generated rubric
has tilt zero with a CI containing zero. The expert rubrics were written against
these specific response pairs and encode which one won; an instruction-only
generator has no route to that information. So part of the 16-point gap to
`expert` is unreachable by construction, and `expert` is not a target so much as
an upper bound on a different task.

**(ii) The differences between generators are not organised by anything a method
can condition on.** Oracle routing — handed the ground-truth domain label,
choosing the per-group winner among all ten sources on the same 600 cases it is
scored on, allowed to answer "use no rubric at all" for a whole domain — buys
**+0.0233 ex-SAFETY**. Choosing out of sample instead, it buys **−0.0036**. The
routing family was closed on this bound before a router was written.

**(iii) Content convergence on the expert rubrics is anti-correlated with
score.** Feeding the generator real dev expert rubrics as few-shot examples
raised its overlap with expert content from 0.187 to 0.238 — the largest content
shift of any candidate — and cost 2.3 to 4.0 points. The one candidate with a
positive Δ has the lowest expert-content overlap of the seven.

**(iv) What is left looks like variance, not quality.** 80 of the 112 addressable
cases are fixed by at least one of nine sources, but a given case is fixed by only
30.4% of them on average, and only 29 of 112 by half or more. Each rewrite flips
37–55 cases toward itself and 34–92 against, and the balance does not track what
the rewrite says. Dev's 0.038 detection floor and this instability are the same
fact measured twice.

### So where is the bottleneck?

Not in the rubric. The 18.7% addressable pool is real — those cases *are*
rubric-responsive, since some rubric fixes 71% of them — but which rubric works
on which case is not predictable from anything we can see at generation time.
Three things follow, in descending order of how well the evidence supports them:

1. **The judge is the binding constraint.** Half of `framed`'s correct answers
   are flipped by at least one of nine rubric rewrites, and the judge is only
   86.3% position-consistent,
   meaning ~14% of verdicts depend on presentation order alone. A quantity that
   unstable cannot be optimised at the 1–3 point scale that prompt changes
   operate on. The intervention this points to is a better or aggregated judge,
   not a better rubric.
2. **If the variance is to be exploited, it needs a selector, and we have no
   signal for one.** The per-case oracle over ten sources reaches 0.8459
   ex-SAFETY — above `expert` — so the information exists somewhere in the spread.
   But the cheapest label-free way to extract it, majority vote over those ten
   sources' verdicts at ten judge passes per case, returns **+0.0072
   (p=0.659)**. Sampling k rubrics and picking one is only worth trying with a
   selection signal that does not exist yet; without it, k rubrics is k times the
   cost for nothing.
3. **Part of the gap requires seeing the responses, and the task as posed
   forbids it.** The tilt result says the expert advantage is partly pair-specific
   knowledge. A method allowed to read both responses before writing criteria
   would have access to it — and would be solving a different problem than
   "generate a rubric from an instruction", which is the one this benchmark and
   this project pose.

The honest summary is that the 21.3-point headline space was never 21.3 points of
rubric quality. Just under a third of it is already captured by one paragraph of
prompt; a substantial part of the rest is annotation-time knowledge of the answer
plus judge noise, and neither is reachable by writing better rubrics.

## Reproducing

```bash
# the frozen split (refuses to overwrite; --verify re-derives and diffs)
python scripts/rubricbench_split.py --verify

# the incumbent on dev
python scripts/rubricbench_run.py --source framed \
    --case-ids results/rubricbench/split.json:dev

# every comparison in this file
python scripts/rubricbench_compare.py --reference framed \
    --case-ids results/rubricbench/split.json:dev \
    --results-dir results/rubricbench/opt/runs results/rubricbench \
    --sources none baseline framed dev_fs_sim5 dev_fs_fix5 expert

# the routing bound and the content/tilt measurements (no LLM calls)
python scripts/rubricbench_route.py --cross-fit \
    --sources framed contrastive balanced qform framed_bare framed_noweight baseline none
python scripts/rubricbench_rubric_stats.py \
    --sources framed expert fs_sim5 fs_fix5 contrastive balanced framed_bare
```

Artefacts: `dev_baseline.md`, `dev_round2.md`, `route_bound.md`,
`route_bound_all.md`, `rubric_content.md`, rubrics under `opt/rubrics/`, verdicts
under `opt/runs/`.
