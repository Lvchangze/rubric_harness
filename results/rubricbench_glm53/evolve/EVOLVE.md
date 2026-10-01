# Self-evolution of the rubric writer (RubricBench dev, GLM-5.3)

Frozen upstream: `results/rubricbench_glm53/agentic-tools-distill_predistill.json`. Train 400 / val 200 (seed 20261001).
Baseline: train 0.6175, val 0.6000.

| incumbent | train | val |
|---|--:|--:|
| `seed` | 0.6225 | 0.6050 |
| `r1c2` | 0.6350 | 0.5850 |

Selection used the train split only; val is reported, never selected on. The holdout was not read.
