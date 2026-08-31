# Rubric form and content, 596 dev cases with an expert rubric

| source | criteria | words/crit | ends `?` | negated | expert_recall | **expert_recall_novel** | framed_f1 |
|---|--:|--:|--:|--:|--:|--:|--:|
| `framed` | 5.16 | 22.1 | 0.00 | 0.36 | 0.270 | **0.187** | 1.000 |
| `framed_noweight` | 5.16 | 17.2 | 0.00 | 0.35 | 0.240 | **0.160** | 0.902 |
| `qform` | 5.09 | 17.5 | 1.00 | 0.30 | 0.275 | **0.191** | 0.366 |
| `expert` | 5.73 | 17.4 | 0.95 | 0.25 | 1.000 | **1.000** | 0.244 |
| `fs_sim5` | 5.35 | 24.8 | 0.00 | 0.36 | 0.314 | **0.225** | 0.412 |
| `fs_fix5` | 5.29 | 25.9 | 0.00 | 0.37 | 0.323 | **0.238** | 0.404 |
| `contrastive` | 5.76 | 19.1 | 1.00 | 0.45 | 0.278 | **0.192** | 0.336 |
| `balanced` | 5.21 | 17.1 | 1.00 | 0.26 | 0.264 | **0.177** | 0.359 |
| `framed_bare` | 5.16 | 16.2 | 0.00 | 0.35 | 0.240 | **0.160** | 0.902 |

`expert_recall_novel` excludes expert tokens that already occur in the instruction, so it does not reward echoing the prompt. `framed_f1` is token F1 against `framed`; 1.000 means the content did not move.

## Is the rubric's vocabulary tilted toward the winning response?

For each case, rubric tokens that do not occur in the instruction are looked up in each response. A generator that never sees the responses has no way to tilt; an annotator who saw them does.

| source | in preferred | in rejected | **tilt** | 95% CI (bootstrap) |
|---|--:|--:|--:|---|
| `framed` | 0.142 | 0.142 | **+0.001** | [-0.005, +0.007] |
| `framed_noweight` | 0.161 | 0.158 | **+0.003** | [-0.004, +0.010] |
| `qform` | 0.172 | 0.169 | **+0.003** | [-0.004, +0.011] |
| `expert` | 0.152 | 0.132 | **+0.020** | [+0.012, +0.027] |
| `fs_sim5` | 0.141 | 0.141 | **+0.000** | [-0.005, +0.006] |
| `fs_fix5` | 0.135 | 0.137 | **-0.002** | [-0.008, +0.003] |
| `contrastive` | 0.158 | 0.159 | **-0.000** | [-0.007, +0.006] |
| `balanced` | 0.186 | 0.184 | **+0.001** | [-0.006, +0.009] |
| `framed_bare` | 0.161 | 0.158 | **+0.003** | [-0.004, +0.010] |

2000 case-level bootstrap resamples, seed 20260831.

