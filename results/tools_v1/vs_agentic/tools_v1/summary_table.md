## 合并（两域 pooled）

| Metric | 方向 | n | `agentic` | `agentic-tools` | Δ `agentic-tools` − `agentic` (95% CI, p) |
|---|---|---|---|---|---|
| **grounding** | | |  |  |  | |
| Generic criterion rate | ↓ | 128 | 0.079 | 0.078 | -0.001 [-0.022, 0.019], p=0.93 |
| …permutation control | · | 128 | 0.967 | 0.966 | -0.001 [-0.013, 0.010], p=0.85 |
| Reward mass on generic criteria | ↓ | 128 | 0.073 | 0.078 | +0.005 [-0.016, 0.026], p=0.65 |
| Anchors per criterion | ↑ | 128 | 3.034 | 2.946 | -0.088 [-0.221, 0.040], p=0.2 |
| Weakly grounded (<2 anchors) | ↓ | 128 | 0.264 | 0.255 | -0.010 [-0.037, 0.017], p=0.5 |
| Recyclable criterion rate | ↓ | 128 | 0.000 | 0.000 | +0.000 [0.000, 0.000], p=1 |
| Subjective/style criterion rate | ↓ | 128 | 0.073 | 0.060 | -0.013 [-0.028, 0.002], p=0.08 |
| Reward mass on subjective criteria | ↓ | 128 | 0.074 | 0.058 | -0.016 [-0.031, -0.001], p=0.044 |
| Specificity lift vs permuted | ↑ | 128 | 0.888 | 0.888 | -0.000 [-0.025, 0.024], p=0.98 |
| **lint** | | |  |  |  | |
| RaR prompt compliance rate | ↑ | 128 | 0.675 | 0.735 | **+0.061 [0.029, 0.095], p=0.0004** |
| Single-polarity rubric rate (F6) | ↑ | 128 | 1.000 | 1.000 | +0.000 [0.000, 0.000], p=1 |
| Unclassified polarity rate | ↓ | 128 | 0.000 | 0.000 | +0.000 [0.000, 0.000], p=1 |
| Negative-weight criterion rate | ↓ | 128 | 0.175 | 0.205 | +0.029 [0.002, 0.058], p=0.04 |
| Weight-order violations | ↓ | 128 | 0.391 | 0.258 | -0.133 [-0.242, -0.023], p=0.029 |
| Pitfall mirrors a positive item (F7) | ↓ | 128 | 0.188 | 0.227 | +0.039 [-0.055, 0.133], p=0.5 |
| **adaptivity** | | |  |  |  | |
| Criteria per rubric | · | 128 | 10.117 | 9.602 | -0.516 [-1.164, 0.164], p=0.14 |
| Criteria per sub-question | · | 128 | 10.004 | 9.459 | -0.545 [-1.195, 0.127], p=0.11 |
| Orphan sub-questions | ↓ | **2** ⚠ | 0.000 | 0.000 | +0.000 [0.000, 0.000], p=1 |

方向: ↑ 越大越好, ↓ 越小越好, · 描述性指标。粗体表示 BH-FDR 校正后 q<0.05。括号内为 bootstrap 95% 置信区间；Δ 列为与 `agentic` 的配对差值。各来源列为**配对均值**（在同一批 n 题上），因此 Δ 恰为两列之差。n 列标 ⚠ 表示配对题数 < 30，该行的置信区间不可用于下结论。

## 分域：`rar_medicine`

| Metric | 方向 | n | `agentic` | `agentic-tools` | Δ `agentic-tools` − `agentic` (95% CI, p) |
|---|---|---|---|---|---|
| **grounding** | | |  |  |  | |
| Generic criterion rate | ↓ | 65 | 0.039 | 0.037 | -0.002 [-0.023, 0.019], p=0.84 |
| …permutation control | · | 65 | 0.974 | 0.978 | +0.004 [-0.007, 0.016], p=0.59 |
| Reward mass on generic criteria | ↓ | 65 | 0.036 | 0.041 | +0.005 [-0.015, 0.026], p=0.65 |
| Anchors per criterion | ↑ | 65 | 3.343 | 3.246 | -0.097 [-0.287, 0.083], p=0.31 |
| Weakly grounded (<2 anchors) | ↓ | 65 | 0.178 | 0.168 | -0.010 [-0.046, 0.026], p=0.6 |
| Recyclable criterion rate | ↓ | 65 | 0.000 | 0.000 | +0.000 [0.000, 0.000], p=1 |
| Subjective/style criterion rate | ↓ | 65 | 0.076 | 0.074 | -0.002 [-0.018, 0.016], p=0.82 |
| Reward mass on subjective criteria | ↓ | 65 | 0.080 | 0.075 | -0.005 [-0.024, 0.014], p=0.59 |
| Specificity lift vs permuted | ↑ | 65 | 0.935 | 0.941 | +0.006 [-0.018, 0.031], p=0.66 |
| **lint** | | |  |  |  | |
| RaR prompt compliance rate | ↑ | 65 | 0.529 | 0.578 | +0.049 [-0.004, 0.103], p=0.083 |
| Single-polarity rubric rate (F6) | ↑ | 65 | 1.000 | 1.000 | +0.000 [0.000, 0.000], p=1 |
| Unclassified polarity rate | ↓ | 65 | 0.000 | 0.000 | +0.000 [0.000, 0.000], p=1 |
| Negative-weight criterion rate | ↓ | 65 | 0.184 | 0.198 | +0.014 [-0.012, 0.042], p=0.32 |
| Weight-order violations | ↓ | 65 | 0.385 | 0.354 | -0.031 [-0.200, 0.123], p=0.85 |
| Pitfall mirrors a positive item (F7) | ↓ | 65 | 0.262 | 0.277 | +0.015 [-0.138, 0.169], p=1 |
| **adaptivity** | | |  |  |  | |
| Criteria per rubric | · | 65 | 10.323 | 9.462 | -0.862 [-1.477, -0.262], p=0.011 |
| Criteria per sub-question | · | 65 | 10.323 | 9.462 | -0.862 [-1.477, -0.262], p=0.011 |

方向: ↑ 越大越好, ↓ 越小越好, · 描述性指标。粗体表示 BH-FDR 校正后 q<0.05。括号内为 bootstrap 95% 置信区间；Δ 列为与 `agentic` 的配对差值。各来源列为**配对均值**（在同一批 n 题上），因此 Δ 恰为两列之差。n 列标 ⚠ 表示配对题数 < 30，该行的置信区间不可用于下结论。

## 分域：`rar_science`

| Metric | 方向 | n | `agentic` | `agentic-tools` | Δ `agentic-tools` − `agentic` (95% CI, p) |
|---|---|---|---|---|---|
| **grounding** | | |  |  |  | |
| Generic criterion rate | ↓ | 63 | 0.121 | 0.121 | +0.001 [-0.037, 0.035], p=0.98 |
| …permutation control | · | 63 | 0.960 | 0.954 | -0.006 [-0.027, 0.014], p=0.6 |
| Reward mass on generic criteria | ↓ | 63 | 0.112 | 0.117 | +0.005 [-0.033, 0.040], p=0.79 |
| Anchors per criterion | ↑ | 63 | 2.715 | 2.636 | -0.079 [-0.271, 0.109], p=0.43 |
| Weakly grounded (<2 anchors) | ↓ | 63 | 0.353 | 0.343 | -0.010 [-0.052, 0.032], p=0.65 |
| Recyclable criterion rate | ↓ | 63 | 0.000 | 0.000 | +0.000 [0.000, 0.000], p=1 |
| Subjective/style criterion rate | ↓ | 63 | 0.070 | 0.046 | -0.025 [-0.050, -0.001], p=0.045 |
| Reward mass on subjective criteria | ↓ | 63 | 0.068 | 0.042 | -0.026 [-0.051, -0.004], p=0.032 |
| Specificity lift vs permuted | ↑ | 63 | 0.839 | 0.833 | -0.006 [-0.050, 0.038], p=0.78 |
| **lint** | | |  |  |  | |
| RaR prompt compliance rate | ↑ | 63 | 0.826 | 0.898 | **+0.073 [0.038, 0.108], p=0.0003** |
| Single-polarity rubric rate (F6) | ↑ | 63 | 1.000 | 1.000 | +0.000 [0.000, 0.000], p=1 |
| Unclassified polarity rate | ↓ | 63 | 0.000 | 0.000 | +0.000 [0.000, 0.000], p=1 |
| Negative-weight criterion rate | ↓ | 63 | 0.167 | 0.212 | +0.045 [-0.005, 0.095], p=0.081 |
| Weight-order violations | ↓ | 63 | 0.397 | 0.159 | **-0.238 [-0.381, -0.095], p=0.0045** |
| Pitfall mirrors a positive item (F7) | ↓ | 63 | 0.111 | 0.175 | +0.063 [-0.048, 0.175], p=0.38 |
| **adaptivity** | | |  |  |  | |
| Criteria per rubric | · | 63 | 9.905 | 9.746 | -0.159 [-1.365, 1.048], p=0.82 |
| Criteria per sub-question | · | 63 | 9.675 | 9.456 | -0.218 [-1.413, 0.988], p=0.74 |
| Orphan sub-questions | ↓ | **2** ⚠ | 0.000 | 0.000 | +0.000 [0.000, 0.000], p=1 |

方向: ↑ 越大越好, ↓ 越小越好, · 描述性指标。粗体表示 BH-FDR 校正后 q<0.05。括号内为 bootstrap 95% 置信区间；Δ 列为与 `agentic` 的配对差值。各来源列为**配对均值**（在同一批 n 题上），因此 Δ 恰为两列之差。n 列标 ⚠ 表示配对题数 < 30，该行的置信区间不可用于下结论。

### 分布层面指标（无逐题对应值，故不报置信区间）

| Metric | 方向 | `agentic` | `agentic-tools` |
|---|---|---|---|
| Item-count CV (shipped science 0.107) | ↑ | 0.331 | 0.393 |
| Modal item-count share (shipped science 0.646) | ↓ | 0.219 | 0.172 |
| r(items, question length) (shipped science 0.19) | ↑ | 0.139 | 0.230 |
| r(items, #sub-questions) | ↑ | 0.043 | 0.160 |
| Weight vs uniform correlation (shipped ~0.94) | ↓ | — | — |
| Mean \|weighted − uniform\| score | ↑ | — | — |
