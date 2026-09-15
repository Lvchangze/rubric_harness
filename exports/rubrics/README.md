# RaR 语料上的 rubric 导出

266 道题（rar_science 132 / rar_medicine 134），三个来源，按「域 × 来源」切成 6 个 JSONL。

由 `scripts/export_rubrics.py` 生成，可重复运行：

```bash
python3 scripts/export_rubrics.py --out-dir exports/rubrics
```

## 三个来源的区别

| 来源 | 是什么 | 生成者 |
|---|---|---|
| `shipped` | **RaR 数据集自带**的 rubric | 论文的单次流水线（science 用 o3-mini，medicine 用 GPT-4o）。**不是本仓库产出** |
| `baseline` | 论文原版 prompt（逐字、按域分派）+ **我们的模型**单次合成 | 本仓库。控制了模型差异，所以它与 `agentic` 的差别是**方法**差别 |
| `agentic` | 本仓库的多阶段流水线 | 本仓库 |

**做对照时用 `baseline` 而不是 `shipped`。** `shipped` 与 `agentic` 之间同时差了模型和方法，两者无法分离；`baseline` 把模型固定住。`shipped` 的价值是它是论文的原始产物。

## 文件

| 文件 | 题数 | 空 rubric | 平均条数 |
|---|--:|--:|--:|
| `rar_science_shipped.jsonl` | 132 | 0 | 7.46 |
| `rar_science_baseline.jsonl` | 132 | **2** | 10.39 |
| `rar_science_agentic.jsonl` | 132 | 0 | 10.33 |
| `rar_medicine_shipped.jsonl` | 134 | 0 | 9.57 |
| `rar_medicine_baseline.jsonl` | 134 | 0 | 8.19 |
| `rar_medicine_agentic.jsonl` | 134 | 0 | 9.80 |

三个来源覆盖的 uid 完全相同，可逐题配对。

## 记录格式

```json
{
  "uid": "sci-0285c4aecb7a",
  "domain": "rar_science",
  "question_source": "Meta/natural_reasoning",
  "question": "...",
  "reference_answer": "...",
  "rubric_source": "agentic",
  "n_criteria": 6,
  "rubric": [
    {"title": "Correct integral formula",
     "description": "States the expectation value formula as ⟨A⟩ = ∫ dV ψ*(Âψ) ...",
     "weight": 5,
     "category": "Essential",
     "polarity": "positive"}
  ],
  "rubric_list": ["Essential Criteria: States the expectation value formula as ..."]
}
```

`rubric_list` 是数据集原生的「带类别前缀的单句」形式，方便直接喂给 RaR 的现成工具。

## 两个使用时必须知道的点

**1. 按 `|weight|` 计分，方向取 `polarity`，不要看权重符号。**
两个语料对「负权重的 criterion」含义是**相反的**：RaR-Science 有 80.3% 写成「避免式」（满足 = 好），
RaR-Medicine 有 88.1% 写成「失败式」（满足 = 坏），但两边权重都是 −1/−2
（`docs/01_data_forensics.md` F6）。所以导出里显式带了 `polarity` 字段。
任选一种符号约定会让其中一个域被训反。

**2. `rar_science_baseline.jsonl` 有 2 条空 rubric**（`sci-67a441205027`、`sci-79900a0d65f6`）。
这是 baseline 在这两题上 JSON 解析彻底失败的真实结果，不是导出 bug，
**刻意保留**而不是悄悄丢掉——丢掉会把对照组唯一的两次完全失败从统计里抹去
（见 `results/REPORT.md` 附录 B 第 7 条的 intent-to-treat 讨论）。

## 来源与完整性

rubric 来自三个 run 目录（`runs/pilot_v2` 200 题、`runs/rollout_v2` 与 `runs/tools_v1` 各 128 题，
互相重叠 62 题），按 uid 合并。导出脚本**断言**共享 uid 在各 run 中的 rubric 完全一致，
不一致会直接报错退出而不是任选一份。

`shipped` 额外做了交叉校验：导出的条数与 `data/` 下原始 parquet 的 `rubric_count`
**266/266 完全一致**。

## 这些 rubric 的质量结论

不要默认 `agentic` 更好。本仓库的测量结果是：它在结构性指标上大幅领先，
但**没有证据表明这转化成了更好的 reward signal**；在唯一一次外部直接测量
（RubricBench，1147 组人类偏好）上它反而**输给了**单次合成。
详见 [`../../HANDOFF.md`](../../HANDOFF.md) §2 的证据分级表。
