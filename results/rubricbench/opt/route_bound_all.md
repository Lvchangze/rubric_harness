# Oracle routing bound on 600 cases (results/rubricbench/split.json:dev)

Sources: `framed`, `contrastive`, `balanced`, `qform`, `framed_bare`, `framed_noweight`, `baseline`, `none`, `fs_sim5`, `fs_fix5`. Reference `framed`. No model calls: every number below is a recombination of verdicts already on disk.

## Per-group ACC, and the oracle's pick

| group | n | `framed` | `contrastive` | `balanced` | `qform` | `framed_bare` | `framed_noweight` | `baseline` | `none` | `fs_sim5` | `fs_fix5` | oracle picks |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|---|
| IF | 65 | 0.7077 | 0.6769 | 0.7385 | 0.6769 | 0.6923 | 0.7692* | 0.6769 | 0.6308 | 0.6308 | 0.6769 | `framed_noweight` |
| STEM | 131 | 0.7328* | 0.5725 | 0.6794 | 0.6870 | 0.6870 | 0.7328 | 0.7023 | 0.6794 | 0.6947 | 0.7099 | `framed` |
| CODE | 142 | 0.5845 | 0.5915 | 0.5845 | 0.5986 | 0.5915 | 0.6127* | 0.5845 | 0.5634 | 0.5423 | 0.5634 | `framed_noweight` |
| SAFETY | 42 | 0.6667 | 0.4048 | 0.4048 | 0.6190 | 0.7143* | 0.5952 | 0.3095 | 0.2619 | 0.5714 | 0.6190 | `framed_bare` |
| CHAT | 220 | 0.6227 | 0.6000 | 0.6318 | 0.6000 | 0.6455* | 0.6227 | 0.6045 | 0.5818 | 0.5909 | 0.5955 | `framed_bare` |

`*` marks the group winner. Ties resolve to the reference.

## The bound

| reading | n | `framed` | oracle-routed | Δ | McNemar p | clears +0.038? |
|---|--:|--:|--:|--:|--:|---|
| forward, all | 600 | 0.6500 | 0.6750 | +0.0250 | 0.0722 | no |
| forward, ex-SAFETY | 558 | 0.6487 | 0.6720 | +0.0233 | 0.117 | no |

## The same selection, made out of sample

Dev is halved by a hash of the case id (stable, no seed to pick). Each half's per-group winner is chosen on the *other* half, so the reported score no longer contains the selection it came from. The difference between this and the row above is the winner's curse.

| reading | n | `framed` | cross-fitted routing | Δ | McNemar p |
|---|--:|--:|--:|--:|--:|
| forward, all | 600 | 0.6500 | 0.6383 | -0.0117 | 0.427 |
| forward, ex-SAFETY | 558 | 0.6487 | 0.6452 | -0.0036 | 0.888 |

## For scale: the per-case oracle

Not a routing bound — no router sees per-case outcomes — but it says whether the candidates disagree enough for selection to be worth anything at all.

| reading | n | union of correct | Δ vs reference |
|---|--:|--:|--:|
| forward, all | 600 | 0.8467 | +0.1967 |
| forward, ex-SAFETY | 558 | 0.8459 | +0.1971 |

## Diagnostic: is the per-case spread reachable without labels?

Majority vote over the sources' forward verdicts. Not a candidate (it judges k times per case, and it is not a rubric-generation method) — reported only because it distinguishes usable complementarity from noise.

| reading | n | majority vote | Δ vs reference | McNemar p |
|---|--:|--:|--:|--:|
| forward, all | 600 | 0.6533 | +0.0033 | 0.885 |
| forward, ex-SAFETY | 558 | 0.6559 | +0.0072 | 0.659 |

Oracle routing and `framed` disagree on 59/558 ex-SAFETY cases; the smallest |Δ| an exact McNemar could call significant there is 0.0305.

## Is the group structure real?

| source (fixed, no routing) | ex-SAFETY ACC |
|---|--:|
| `framed` | 0.6487 |
| `contrastive` | 0.6004 |
| `balanced` | 0.6434 |
| `qform` | 0.6290 |
| `framed_bare` | 0.6470 |
| `framed_noweight` | 0.6631 |
| `baseline` | 0.6308 |
| `none` | 0.6057 |
| `fs_sim5` | 0.6075 |
| `fs_fix5` | 0.6237 |
| **oracle-routed** | **0.6720** |

Best subset to route between (ex-SAFETY): `qform`, `framed_bare`, `framed_noweight`, `none`, `fs_sim5`, `fs_fix5` at 0.6720.

## Verdict

Oracle routing with ground-truth labels moves ex-SAFETY ACC by +0.0233. The pre-registered rule was to open the routing family only if this exceeded +0.038. It does not, so the family is **closed**.

