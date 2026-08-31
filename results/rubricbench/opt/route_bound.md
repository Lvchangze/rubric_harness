# Oracle routing bound on 600 cases (results/rubricbench/split.json:dev)

Sources: `framed`, `contrastive`, `balanced`. Reference `framed`. No model calls: every number below is a recombination of verdicts already on disk.

## Per-group ACC, and the oracle's pick

| group | n | `framed` | `contrastive` | `balanced` | oracle picks |
|---|--:|--:|--:|--:|---|
| IF | 65 | 0.7077 | 0.6769 | 0.7385* | `balanced` |
| STEM | 131 | 0.7328* | 0.5725 | 0.6794 | `framed` |
| CODE | 142 | 0.5845 | 0.5915* | 0.5845 | `contrastive` |
| SAFETY | 42 | 0.6667* | 0.4048 | 0.4048 | `framed` |
| CHAT | 220 | 0.6227 | 0.6000 | 0.6318* | `balanced` |

`*` marks the group winner. Ties resolve to the reference.

## The bound

| reading | n | `framed` | oracle-routed | Δ | McNemar p | clears +0.038? |
|---|--:|--:|--:|--:|--:|---|
| forward, all | 600 | 0.6500 | 0.6583 | +0.0083 | 0.668 | no |
| forward, ex-SAFETY | 558 | 0.6487 | 0.6577 | +0.0090 | 0.668 | no |

## The same selection, made out of sample

Dev is halved by a hash of the case id (stable, no seed to pick). Each half's per-group winner is chosen on the *other* half, so the reported score no longer contains the selection it came from. The difference between this and the row above is the winner's curse.

| reading | n | `framed` | cross-fitted routing | Δ | McNemar p |
|---|--:|--:|--:|--:|--:|
| forward, all | 600 | 0.6500 | 0.6400 | -0.0100 | 0.451 |
| forward, ex-SAFETY | 558 | 0.6487 | 0.6380 | -0.0108 | 0.451 |

## For scale: the per-case oracle

Not a routing bound — no router sees per-case outcomes — but it says whether the candidates disagree enough for selection to be worth anything at all.

| reading | n | union of correct | Δ vs reference |
|---|--:|--:|--:|
| forward, all | 600 | 0.7733 | +0.1233 |
| forward, ex-SAFETY | 558 | 0.7778 | +0.1290 |

## Diagnostic: is the per-case spread reachable without labels?

Majority vote over the sources' forward verdicts. Not a candidate (it judges k times per case, and it is not a rubric-generation method) — reported only because it distinguishes usable complementarity from noise.

| reading | n | majority vote | Δ vs reference | McNemar p |
|---|--:|--:|--:|--:|
| forward, all | 600 | 0.6400 | -0.0100 | 0.556 |
| forward, ex-SAFETY | 558 | 0.6505 | +0.0018 | 1 |

Oracle routing and `framed` disagree on 87/558 ex-SAFETY cases; the smallest |Δ| an exact McNemar could call significant there is 0.0376.

## Is the group structure real?

| source (fixed, no routing) | ex-SAFETY ACC |
|---|--:|
| `framed` | 0.6487 |
| `contrastive` | 0.6004 |
| `balanced` | 0.6434 |
| **oracle-routed** | **0.6577** |

Best subset to route between (ex-SAFETY): `framed`, `contrastive`, `balanced` at 0.6577.

## Verdict

Oracle routing with ground-truth labels moves ex-SAFETY ACC by +0.0090. The pre-registered rule was to open the routing family only if this exceeded +0.038. It does not, so the family is **closed**.

