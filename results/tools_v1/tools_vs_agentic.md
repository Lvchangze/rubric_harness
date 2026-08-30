# `agentic-tools` vs `agentic`

来源：`results/tools_v1/vs_agentic/tools_v1/summary.json`；参照列 = `agentic`；粗体 = BH-FDR 校正后 q<0.05。

### 两域合并

| metric | dir | n | `agentic` | `agentic-tools` | Δ | 95% CI | p | q |
|---|:--:|--:|--:|--:|--:|---|--:|--:|
| grounding/generic_criterion_rate | ↓ | 128 | 0.079 | 0.078 | -0.001 | [-0.022, +0.019] | 0.934 | 1 |
| grounding/generic_rate_permuted | ↓ | 128 | 0.967 | 0.966 | -0.001 | [-0.013, +0.010] | 0.855 | 1 |
| grounding/weight_mass_on_generic_paper | ↓ | 128 | 0.073 | 0.078 | +0.005 | [-0.016, +0.026] | 0.645 | 1 |
| grounding/mean_anchors_per_criterion | ↑ | 128 | 3.034 | 2.946 | -0.088 | [-0.221, +0.040] | 0.196 | 0.441 |
| grounding/weak_grounding_rate | ↓ | 128 | 0.264 | 0.255 | -0.010 | [-0.037, +0.017] | 0.498 | 0.901 |
| grounding/recyclable_rate | ↓ | 128 | 0.000 | 0.000 | +0.000 | [+0.000, +0.000] | 1 | 1 |
| grounding/subjective_or_style_rate | ↓ | 128 | 0.073 | 0.060 | -0.013 | [-0.028, +0.002] | 0.0797 | 0.287 |
| grounding/weight_mass_on_subjective | ↓ | 128 | 0.074 | 0.058 | -0.016 | [-0.031, -0.001] | 0.0442 | 0.199 |
| grounding/specificity_lift | ↑ | 128 | 0.888 | 0.888 | -0.000 | [-0.025, +0.024] | 0.984 | 1 |
| **lint/compliance_rate** | ↑ | 128 | 0.675 | 0.735 | **+0.061** | [+0.029, +0.095] | 0.0004 | 0.0072 |
| lint/polarity_single_rate | ↑ | 128 | 1.000 | 1.000 | +0.000 | [+0.000, +0.000] | 1 | 1 |
| lint/unclassified_polarity_rate | ↓ | 128 | 0.000 | 0.000 | +0.000 | [+0.000, +0.000] | 1 | 1 |
| lint/negative_weight_rate | ↓ | 128 | 0.175 | 0.205 | +0.029 | [+0.002, +0.058] | 0.0405 | 0.199 |
| lint/weight_order_violation_rate | ↓ | 128 | 0.391 | 0.258 | -0.133 | [-0.242, -0.023] | 0.0286 | 0.199 |
| lint/pitfall_mirror_rate | ↓ | 128 | 0.188 | 0.227 | +0.039 | [-0.055, +0.133] | 0.5 | 0.901 |
| adaptivity/n_items | ↓ | 128 | 10.117 | 9.602 | -0.516 | [-1.164, +0.164] | 0.139 | 0.358 |
| adaptivity/items_per_subpart | ↓ | 128 | 10.004 | 9.459 | -0.545 | [-1.195, +0.127] | 0.115 | 0.344 |
| adaptivity/orphan_subpart_rate | ↓ | 2 | 0.000 | 0.000 | +0.000 | [+0.000, +0.000] | 1 | 1 |

### 判读

- 通过 FDR 的改善：1 项 — lint/compliance_rate
- 通过 FDR 的退步：0 项
- 无显著差异：17 项

