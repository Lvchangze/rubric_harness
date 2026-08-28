# Agentic Rubric Generation：本地初步验证报告

**运行**：`pilot_v2` · **日期**：2026-08-28 · **数据**：`runs/pilot_v2/`, `results/pilot_v2/`

> **一句话结论**：**在 rar_science 上**，agentic rubric 相对单次生成的 baseline 有稳健但有限的优势——
> 它把好坏答案的分数拉得更开（science 上 z_separation +0.325, q=0.009，删掉泄漏正例后仍成立；medicine 上无效应），
> 并显著降低"长而错的答案拿高分"这个 RL 漏洞（长度相关性 +0.272 → −0.040, p=0.044）。
> **在 rar_medicine 上，所有这些效应都是零，个别指标还反向。**
> 结构性指标（通用条目率 21.0%→9.5%、主观率 18.1%→7.8%、锚点 2.19→2.82）是全篇最干净的证据，
> 不经过 judge、不经过 reference，两域一致。
> **"排序更准"（AUC、ranking accuracy、best-of-n 选择准确率）没有证据**：
> 在生成器从未见过的真实模型 rollout 上（§4.4），没有任何来源在 FDR 之后优于 baseline，
> best-of-n 甚至是 baseline 最高。
> Stage 6 的哪一半在起作用，**两个评测协议给出相反答案**（§4.4.8），本次实验无定论。

> ⚠️ **三条阅读警告**
> 1. **§3 的数字未做泄漏校正**，必须连同 §4.3 一起读；只引用 §3 会高估 agentic。
> 2. **所有主结论都必须带域的限定**。pooled 显著性主要由 science 拉动（§4.3.5b、§4.4.5）。
> 3. **§4.4 的 n 只有 61 题**（策略在 83.5% 的 rollout 上答对，题内标签不混合的题无法进入指标）。
>    该节"不显著"主要反映检验力不足，不等于"无效应"。

---

## 0. 阅读指引

| 我想知道… | 看这里 |
|---|---|
| 主结果表（三/四来源 × 各指标 + 配对置信区间） | §3、`results/pilot_v2/summary_table.md` |
| 验证在环到底有没有用 | §3.2（消融）→ **必须接着读 §4.3.4，那里推翻了 §3.2 的归因** |
| 赢是不是靠堆条目 | §4.1（条数受控） |
| 赢是不是靠算错对照组的分 | §4.2（极性敏感性） |
| **赢是不是循环论证（最严重的混淆）** | **§4.3（必读）**、`results/pilot_v2/confound_audit.txt` |
| 扣掉泄漏之后还剩多少 | §4.3.3、§4.3.5（修订后的主结论） |
| **结论在两个域上一样吗（不一样）** | **§4.3.5b、§4.4.5（必读）** |
| **在真实模型 rollout 上还成立吗** | **§4.4（低功效）→ §4.6（加大功效后的定论，必读）** |
| 换一个 judge 还成立吗 | §4.5、`results/judgeswap/SUMMARY.md` |
| 两个评测协议冲突怎么办 | §4.4.8 → §4.5.7（三方交叉）→ §4.6 |
| agentic 哪里没赢 | §5（**必读**） |
| 具体 rubric 长什么样 | §6、`results/pilot_v2/case_study_*.md` |
| F1–F10 逐条对照 | §7 |
| 后续同学怎么做真验证 | §8 |

---

## 1. 动机与定位

RaR（Rubrics as Rewards）的做法是：给定 question + reference answer，让一个大模型**一次性合成**一份带权重的
rubric，作为 RL 的 reward signal。本研究的假设是：**一次性合成的 rubric 质量不够，应该用 agentic 的方式
"搜索 + 验证"出 criterion**。

论文自己的消融给了这个假设最好的支持（`docs/00_paper_notes.md` §6）：

- **Table 3**：把单次生成器换成更强的模型，下游收益已经**见顶**。这说明瓶颈不在生成器的能力，而在
  "一次性生成"这个**范式**——要换方法，不是换模型。
- **Table 2**：去掉 category label 反而更好（38.8% > 37.2%），去掉 Pitfall 完全没影响（37.2% = 37.2%）。
  也就是说论文自己引入的结构（类别、Pitfall）**并没有承载信息**。
- **Table 1**：人工 rubric 优于合成 rubric，说明合成 rubric 与"好 rubric"之间确实有差距可填。
- **§6.4** 列出论文没做的消融——那正是本研究的空间。

取证（`docs/01_data_forensics.md`，全量 45,325 题 / 386,675 条 criteria）进一步给出了可量化的靶子：
shipped rubric 有 22–29% 的通用条目、条数近乎常数（science 64.6% 恰好 7 条）、权重与等权平均相关性高达 0.94、
两个 domain 的 Pitfall 极性约定相反。

### 1.1 三（四）个对比对象：解耦"模型差异"与"方法差异"

| 来源 | 生成方式 | 生成模型 | 作用 |
|---|---|---|---|
| `shipped` | 数据集自带（论文原版产物） | o3-mini / GPT-4o（论文所述） | 论文实际交付物 |
| `baseline` | 单次合成，**论文原版 prompt 逐字复现** | 与 agentic 同一个模型 | **解耦关键**：与 agentic 只差方法 |
| `agentic` | 本研究的多阶段 pipeline | 同上 | 本方法 |
| `agentic-noval` | 同上，但**关掉 Stage 6 验证环** | 同上 | 消融：验证在环是否是有效成分 |

`baseline` 这一列是全篇最重要的对照：它与 `agentic` 用同一个模型、同一批题、同一个 judge，
**唯一变量是生成方法**。`shipped` 与 `baseline` 的差值则反映"模型 + 实现"的差异。

---

## 2. 实验设置

| 项 | 值 |
|---|---|
| 数据 | `rar_science` / `rar_medicine`，`val` split |
| 抽样 | 每域 100 题，seed=1234，**按题面去重** |
| 实际配对样本 | **n=198**（2 题因 baseline JSON 解析失败被丢弃，见 §9） |
| 生成 / judge 模型 | `hy-t2t-glm-5.2-384k-fp8-L20A-t1-v2`，`reasoning_effort=high` |
| 聚合方式 | **`paper_explicit`**（RaR Eq.(1)，类别重映射 `{Essential 1.0, Important 0.7, Optional 0.3, Pitfall 0.9}`）；`raw_weight` / `uniform` 作稳健性检查 |
| 极性 | **逐条检测**（`detected`），所有来源同口径；另按整份 rubric 扫描 `all_avoidance` / `all_failure`，给对照组取最有利者（§4.2） |
| Judge prompt | `docs/00_paper_notes.md` §4.3 原文 |
| 公平性 | 同一批降级 response 复用于所有来源；criterion 顺序随机化（seed=7）；judge 对 rubric 来源盲 |
| 统计 | 配对 bootstrap 95% CI（10,000 次）+ 配对置换检验；跨表 BH-FDR 校正 |

### 2.1 质量分层的 response 梯子

对每题生成一次、所有 rubric 来源复用（共 1,359 条 response）：

`gold`（= reference answer）、`missing_step`（删关键推理步）、`numeric_error`（注入数值/符号错）、
`right_method_wrong_answer`（框架对、答案错）、`verbose_empty`（冗长空洞——reward-hack 探针）、
`terse_correct`（简短正确无过程）、`off_topic`（完全跑题）。

### 2.2 花钱之前的 sanity gate

在跑任何 LLM 评测之前，先用**零 LLM 调用**的结构性指标算 `shipped` 那一列，与取证 n=45k 基线对齐。
全部 12 项都落在抽样噪声内（最大偏离 3.3pp）：

| 指标 | domain | 本次 (n=100) | 取证 (n=45k) | 差 |
|---|---|---|---|---|
| 通用条目率 | science | 26.28 | 28.87 | −2.59 |
| 通用条目率 | medicine | 19.08 | 21.97 | −2.89 |
| 置换对照 | science | 97.28 | 96.39 | +0.89 |
| 置换对照 | medicine | 98.93 | 98.54 | +0.39 |
| 加权后无关条目占比 | science | 21.29 | 24.56 | −3.27 |
| 加权后无关条目占比 | medicine | 14.80 | 15.20 | −0.40 |
| 弱接地条目率 | science | 50.93 | 52.61 | −1.68 |
| 弱接地条目率 | medicine | 42.05 | 43.53 | −1.48 |
| Pitfall 镜像率 (F7) | science | 1.02 | 0.95 | +0.07 |
| Pitfall 镜像率 (F7) | medicine | 14.00 | 16.21 | −2.21 |
| 权重序违规 | science | 38.78 | 41.30 | −2.52 |
| 权重序违规 | medicine | 0.00 | 0.00 | +0.00 |

极性分布也复现了取证：science 79.79% 避免式（取证 80.33%），medicine 83.80% 失败式（取证 88.11%）。
条数分布同样复现：science mode=7 占 63.0%（取证 64.6%），7 或 8 条占 90.0%（取证 88.9%）。

**这一步的意义**：它同时验证了两件事——抽样是有代表性的，以及指标实现确实复现了取证的定义。
在此之前不花任何 LLM 预算。

---

## 3. 主结果

完整表见 `results/pilot_v2/summary_table.md`（50 项指标 × 4 来源，含 BH-FDR 校正）。
下表为核心指标，Δ 为**与 `baseline` 的配对差值**，括号内 95% bootstrap CI。n=198。

### 3.1 判别效度（主结果）

| 指标 | 方向 | `shipped` | `baseline` | `agentic-noval` | `agentic` | Δ agentic − baseline |
|---|---|---|---|---|---|---|
| **平均 margin**（gold − 降级） | ↑ | 0.239 | 0.272 | 0.274 | **0.385** | **+0.113 [0.089, 0.136], p=1e-4** |
| 最小 margin | ↑ | −0.107 | −0.045 | −0.024 | **0.035** | **+0.080 [0.052, 0.108], p=1e-4** |
| 配对排序准确率 | ↑ | 0.737 | 0.771 | 0.762 | **0.801** | **+0.030 [0.013, 0.047], p=4e-4** |
| AUC (gold vs 全部降级) | ↑ | 0.775 | 0.823 | 0.827 | **0.869** | **+0.046 [0.022, 0.071], p=2e-4** |
| AUC (good vs bad) | ↑ | 0.737 | 0.787 | 0.793 | **0.842** | **+0.055 [0.032, 0.079], p=1e-4** |
| 分数动态范围 | ↑ | 0.737 | 0.687 | 0.639 | **0.769** | **+0.081 [0.055, 0.106], p=1e-4** |
| margin vs 跑题 | ↑ | 0.556 | 0.560 | 0.532 | **0.694** | **+0.134 [0.104, 0.163], p=1e-4** |
| margin vs 冗长空洞 | ↑ | 0.282 | 0.354 | 0.352 | **0.523** | **+0.168 [0.132, 0.204], p=1e-4** |
| margin vs 自信但错 | ↑ | 0.124 | 0.184 | 0.193 | **0.310** | **+0.127 [0.090, 0.165], p=1e-4** |
| margin vs 数值错 | ↑ | 0.216 | 0.271 | 0.283 | **0.380** | **+0.109 [0.080, 0.140], p=1e-4** |
| margin vs 漏步骤 | ↑ | 0.028 | 0.067 | 0.070 | **0.113** | **+0.045 [0.018, 0.073], p=0.0017** |
| 饱和率（降级仍 ≥0.8） | ↓ | 0.108 | 0.095 | **0.076** | 0.151 | **+0.055 [0.031, 0.081]** ⚠️ **变差** |

**值得单独说的两点**：

1. **最小 margin 由负转正**。`shipped`（−0.107）和 `baseline`（−0.045）的最小 margin 是**负数**——
   意味着平均而言，**存在一档降级 response 得分高于 gold**。这是 reward signal 的严重缺陷：
   RL 会朝这个方向爬。`agentic` 是四者中唯一把最小 margin 拉到正数（+0.035）的。
2. **`shipped` 全面差于 `baseline`**。用同一个现代模型 + 论文原版 prompt 复现，判别力比论文交付的数据
   更好（margin 0.272 vs 0.239, p=0.004）。这说明 shipped 与 baseline 的差距是"模型 + 实现"造成的，
   与方法无关——**这正是设立 `baseline` 这一列的目的**。所有方法层面的结论都应以 `baseline` 为准。

### 3.2 消融：验证在环（Stage 6）是唯一的有效成分

这是本次实验**最重要、也最出乎意料**的发现。

| 指标 | `baseline` | `agentic-noval`（无 Stage 6） | Δ vs baseline | `agentic`（有 Stage 6） | Δ vs baseline |
|---|---|---|---|---|---|
| 平均 margin | 0.272 | 0.274 | +0.001, **p=0.89** | 0.385 | +0.113, p=1e-4 |
| 排序准确率 | 0.771 | 0.762 | −0.008, **p=0.35** | 0.801 | +0.030, p=4e-4 |
| AUC (gold vs 降级) | 0.823 | 0.827 | +0.003, **p=0.79** | 0.869 | +0.046, p=2e-4 |
| AUC (good vs bad) | 0.787 | 0.793 | +0.006, **p=0.59** | 0.842 | +0.055, p=1e-4 |
| specificity gap | 0.549 | 0.510 | −0.039, p=0.003 ↓ | 0.693 | +0.143, p=1e-4 |

**解读**：`agentic-noval` 仍然保留了 Stage 1–5（分解、独立求解、参考对齐、pitfall 挖掘、起草）和
Stage 7（去重、标定）。它在**结构性**指标上依然大幅胜出（通用条目率 8.9% vs 21.0%，锚点数 2.74 vs 2.19，
主观条目率 7.3% vs 18.1%——见 §3.3），说明前面的阶段确实让 criterion 更接地、更具体。

**但这些结构性改进没有转化为任何判别力提升**（四项判别指标全部 p>0.3）。只有加上 Stage 6——
用 gold 和构造的 negative response 去**实际执行**每条 criterion、丢掉 gold 不过的、降权所有 negative 都过的——
判别力才出现。

这条结论对本研究的 framing 影响很大：**卖点不是"用 agent 写 rubric"，而是"把 rubric 当作可执行的判别器来筛选"**。
如果后续同学要压缩 pipeline 成本，Stage 6 是唯一不能砍的。

反过来，`agentic-noval` 也不是没有价值：它在饱和率（0.076，四者最低）和 reward-hack 探针上表现最好
（见 §4.1 条数受控后的结果），说明它写的 criterion 更严格但**区分不出好坏**——严格和有判别力是两回事。

> ⚠️ **本节的归因随后被 §4.3 推翻了一半，不要单独引用。** Stage 6 确实是唯一起作用的阶段，
> 但它同时是唯一**泄漏评测正例**的阶段：它用 `reference_answer` 筛选条目，而上表里的 `gold`
> 就是同一个 `reference_answer`。把 Stage 6 拆成 gold 侧与 negative 侧分别消融之后（§4.3.4）：
> 增益几乎全部来自**泄漏的 gold 侧**，而构造上干净的 negative 侧（"用独立反例筛选"）
> **没有任何指标通过 FDR 校正**。因此"把 rubric 当作可执行的判别器来筛选"这个说法，
> 本次实验只支持其中"拿正确答案核对条目是否写对"的那一半。修订后的表述见 §4.3.5。

### 3.3 Query-specificity / 可迁移性

方法直接复用取证的检测器（`analysis/rubric_grounding.py` 的定义在 `harness/eval/grounding.py` 中按同样定义
重实现，零 LLM 调用；`shipped` 那一列与 n=45k 基线对齐见 §2.2）。另有一套 LLM 判定的 transfer 指标作交叉验证。

| 指标 | 方向 | `shipped` | `baseline` | `agentic-noval` | `agentic` | Δ agentic − baseline |
|---|---|---|---|---|---|---|
| 通用（可回收）条目率 | ↓ | 0.226 | 0.210 | **0.089** | 0.095 | **−0.115 [−0.139, −0.089]** |
| 加权后无关条目占比 | ↓ | 0.180 | 0.185 | **0.086** | 0.090 | **−0.095 [−0.120, −0.069]** |
| 主观/文体条目率 | ↓ | 0.320 | 0.181 | **0.073** | 0.078 | **−0.103 [−0.128, −0.079]** |
| 弱接地条目率（<2 锚点） | ↓ | 0.464 | 0.410 | 0.285 | **0.271** | **−0.139 [−0.171, −0.106]** |
| 每条 criterion 锚点数 | ↑ | 2.03 | 2.19 | 2.74 | **2.82** | **+0.630 [0.509, 0.754]** |
| **specificity gap**（LLM） | ↑ | 0.550 | 0.549 | 0.510 | **0.693** | **+0.143 [0.116, 0.171]** |
| 同题 gold pass rate（LLM） | · | 0.621 | 0.657 | 0.616 | **0.789** | **+0.132 [0.102, 0.164]** |
| 跨题 pass rate（LLM） | ↓ | **0.071** | 0.108 | 0.106 | 0.097 | −0.011 [−0.031, 0.009], p=0.29 |
| grounded criterion 比例（LLM） | ↑ | 0.589 | 0.647 | 0.617 | **0.765** | **+0.118 [0.089, 0.148]** |
| generic criterion 比例（LLM） | ↓ | 0.220 | 0.100 | 0.021 | **0.009** | **−0.090 [−0.105, −0.076]** |

**一个重要的细节**：`agentic` 的 specificity gap 提升（+0.143）**几乎全部来自"同题 gold pass rate"上升
（+0.132），而不是"跨题 pass rate"下降（−0.011, p=0.29）**。也就是说 agentic 的 criterion 并不是
"更难在别的题上通过"，而是**在本题的正确答案上更容易通过**——它写的是本题真正该检查的东西，
而不是 shipped 那种"gold 自己都只有 62% 通过率"的条目。

`shipped` 的跨题 pass rate 最低（0.071），但它的同题 pass rate 也最低（0.621）——它的条目只是**更难通过**，
不是**更对题**。这两个数字必须一起看。

### 3.4 对参考答案的覆盖度

| 指标 | 方向 | `shipped` | `baseline` | `agentic-noval` | `agentic` | Δ agentic − baseline |
|---|---|---|---|---|---|---|
| gold claim 召回率 | ↑ | 0.967 | 0.959 | 0.970 | 0.957 | −0.002 [−0.020, 0.016], p=0.85 |
| 核心 claim 召回率 | ↑ | 0.977 | 0.972 | 0.980 | 0.965 | −0.007 [−0.023, 0.010], p=0.43 |
| 无依据 criterion 比例 | ↓ | **0.190** | 0.254 | 0.362 | 0.226 | −0.028 [−0.057, 0.001], p=0.059 |
| 子问题覆盖率（**n=72**，仅多小问题目） | ↑ | 1.000 | 1.000 | 1.000 | 0.974 | −0.026, p=0.032, q=0.044 ⚠️ **变差** |

**覆盖度上 agentic 没有优势**，这与它在判别力上的优势形成对照：所有来源的 claim 召回率都在 96% 上下，
天花板效应明显，这个指标在本设置下**区分不出来源**。子问题覆盖率 agentic 反而略低（见 §5）。

### 3.5 内在质量

| 指标 | 方向 | `shipped` | `baseline` | `agentic-noval` | `agentic` | Δ agentic − baseline |
|---|---|---|---|---|---|---|
| 客观可判定比例 | ↑ | 0.705 | 0.927 | 0.989 | **0.989** | **+0.062 [0.050, 0.075]** |
| judge 自一致率 | ↑ | 0.929 | 0.966 | 0.963 | 0.967 | +0.001, p=0.86 |
| judge 自一致 kappa（配对 n=81） | ↑ | 0.723 | **0.847** | 0.865 | 0.767 | −0.079, p=0.040, **q=0.053（未过 FDR）**；且为 base-rate 假象，见 §5.3.2 |
| 原子性（每条检查点数） | ↓ | 1.406 | 1.248 | 1.289 | **1.231** | −0.017, p=0.45 |
| 单一极性 rubric 比例 (F6) | ↑ | 0.520 | 0.167 | **1.000** | **1.000** | **+0.833 [0.778, 0.884]** |
| 未分类极性条目率 | ↓ | 0.019 | 0.067 | **0.000** | **0.000** | **−0.067 [−0.079, −0.056]** |

**冗余度指标已按 P1-5 移除**：取证证明"句内冗余严重"这个假设不成立（朴素 TF-IDF 看似 22%/58%，
加入实体/数字判别过滤后真实冗余率只有 0.8%/7.9%）。省下的预算加到了主结果的样本量上。
唯一真实的冗余形式——Pitfall 镜像 Essential 导致同一事实双重计分（F7）——保留在 `lint` 中测量。

---

## 4. 两个必须做的混淆控制

### 4.1 条数受控（P1-6）：赢不是靠堆条目

`agentic` 平均 10.05 条，`baseline` 9.27 条，`agentic-noval` 12.41 条。条数不同会污染覆盖率类指标，
也可能污染判别力。因此把**每题所有来源截断到相同条数**后重跑（总条数 shipped/baseline/agentic/noval
从 1696/1854/1983/2453 一律截到 **1456**），完整表见 `results/pilot_v2/cc_summary_table.md`：

| 指标 | 未受控 Δ(agentic−baseline) | **条数受控 Δ** | 结论 |
|---|---|---|---|
| 平均 margin | +0.113 | **+0.138 [0.114, 0.162]** | 优势**变大** |
| 排序准确率 | +0.030 | **+0.047 [0.030, 0.064]** | 优势**变大** |
| AUC (gold vs 降级) | +0.046 | **+0.058 [0.034, 0.083]** | 优势**变大** |
| AUC (good vs bad) | +0.055 | **+0.064 [0.042, 0.088]** | 优势**变大** |
| 分数动态范围 | +0.081 | **+0.091 [0.064, 0.119]** | 优势**变大** |
| reward-hack: 冗长空洞得分 | −0.027 (p=0.078) | **−0.060 [−0.090, −0.031], p=3e-4** | 由不显著**转为显著** |
| 饱和率 | +0.055（变差） | +0.024 (p=0.061) | 劣势**缩小到不显著** |

**结论明确**：agentic 的判别力优势**不是**来自条目更多。截断到相同条数后优势反而**普遍变大**，
说明 agentic 的**前 k 条**质量更高——它的条目是有优先级的，而 baseline 的不是。

一个副产品：`agentic-noval` 在受控后从"与 baseline 无差异"变成"显著优于 baseline"
（margin +0.042, p=3e-4）。它写了最多的条目（12.41 条）但**靠后的条目在稀释信号**。
这从另一个角度说明 Stage 6 的作用之一是"知道该停在哪里"。

### 4.2 极性敏感性（P0-2）：赢不是靠算错对照组的分

取证 F6 发现两个 domain 的 Pitfall 极性约定**相反**（science 80.3% 避免式，medicine 88.1% 失败式），
而权重都是 −1/−2，论文自身的正文、prompt、聚合公式三处也互相矛盾。如果打分代码对所有 Pitfall
套用同一种符号约定，**必有一个 domain 的 shipped rubric 被系统性算错分**。

本 harness 的做法：judge **只判断陈述是否字面为真**（prompt 里明确说"你不是在判断回答好不好"），
极性作为 schema 里的显式字段单独处理；然后用同一批 judge 判定，按四种极性口径分别聚合。

**gold − off_topic 的 margin，每个来源取对它最有利的口径**：

| 来源 | 最有利口径 | 该口径下 margin | `detected`（正确口径）下 |
|---|---|---|---|
| `shipped` | detected（= all_avoidance） | 0.556 | 0.556 |
| `baseline` | detected（= all_avoidance） | 0.560 | 0.560 |
| `agentic-noval` | all_avoidance / detected | 0.532 | 0.532 |
| **`agentic`** | all_avoidance / detected | **0.694** | **0.694** |

**结论**：即使给每个对照组挑对它最有利的整体极性口径，`agentic` 依然领先（0.694 vs 0.556 / 0.560）。
这个赢是硬赢。

> **⚠️ 一处已修正的错误声明。** 本报告此前（以及 §2 的设置表、README）声称"对照组另外还享有
> 一层逐条的 benefit of the doubt"（`PolarityMode.FAVOURABLE`）。**这层让利从未生效。**
> `detect_polarity` 对 `unclassified` 本就返回 POSITIVE，而 `category_default` 只出现在非 Pitfall
> 条目上、那一支在 `effective_polarity` 里提前返回，因此每个能走到的分支都与 `DETECTED` 相同。
> 实测 **12,484 条 criterion 中 0 条不同，18,849 条 judge 行 max abs diff = 0.0**
> ——证据一直摆在 `report_facts.txt` 的极性摆动表里（每行 `favourable=` 与 `detected=` 完全相等）。
>
> 该分支已删除。为什么不"真正实现"它：逐条取最有利极性是**退化的**——
> 把每条都判成"满足"会让任何回答都得 1.0 分。"从宽"只有在**整份 rubric 统一一种约定**时才有意义，
> 那正是上面这张表做的事。所以现在的口径是：所有来源一律 `detected` 打分，
> 然后从 `all_avoidance` / `all_failure` 里给每个对照组挑对它最有利的那个。上表的结论不受影响。
>
> **一个无法消除的不对称，必须写明**：agentic 系全部 Pitfall 的 `polarity_method` 都是
> `"enforced"`（生成时自己声明），而 shipped / baseline 走正则检测（baseline 有 141 条 `unclassified`）。
> **对照组的极性可能被判错，agentic 不可能。** 上面的整体口径扫描是对这一点的补偿，但不是消除。

#### 4.2.1 一个值得单独写的观察：常规 sanity check 抓不到极性 bug

极性口径的选择会让分数移动多少？

| 来源 | gold 分数摆动 | off_topic 分数摆动 |
|---|---|---|
| `shipped` | **3.84 pp** | 2.15 pp |
| `baseline` | **15.55 pp** | 7.98 pp |
| `agentic-noval` | 15.63 pp | 6.53 pp |
| `agentic` | 15.62 pp | 5.18 pp |

`baseline` 的 gold 分数在 `all_avoidance` 口径下是 0.570，在 `all_failure` 口径下是 0.726——**相差 15.6 个百分点**。
这足以颠覆任何方法间比较。（`shipped` 摆动较小是因为它的 Pitfall 占比低且极性表述更规整；
`baseline` 摆动大是因为它有 6.7% 的条目极性无法分类、且 Pitfall 镜像率高达 25.8%。）

**而关键在于**：我们检查了全部 4 个来源 × 4 种极性口径 = 16 种组合，
**"gold 分数 > off_topic 分数"这个 sanity check 在所有 16 种组合下都通过**。

这意味着：**一个把整个 domain 的 Pitfall 算反了的实现，仍然会通过"好答案得分高于坏答案"这类常规检查**，
却已经在 15 个百分点的尺度上污染了 reward。做 RL 的同学如果只用这类粗粒度 sanity check 来验证 reward 实现，
是抓不到这个 bug 的——必须显式地对极性做单元测试。这本身就是一个可以写进论文的工程发现。

---

### 4.3 泄漏审计（最严重的一个混淆）：Stage 6 的 gold 与评测的 gold 是同一个对象

> **这一节推翻了 §3.2 的表述，并实质性地弱化了主结论。** 结论见 §4.3.5。
> 全部数字：`results/pilot_v2/confound_audit.txt` / `.json`，复现命令
> `python scripts/confound_audit.py --run-name pilot_v2 --reference baseline`。

#### 4.3.1 问题

agentic 的 Stage 6 用 `example.reference_answer` 当 gold 去判定候选 criterion：gold 判不通过的条目被
`drop_gold_fail` 删掉，gold 反驳的条目被 `drop_contradicted` 删掉。而判别效度的主指标里，
**`gold` 这一档就是同一个 `reference_answer`**。

也就是说 **Stage 6 直接以"提高 reference answer 的得分"为目标做筛选，而主指标恰好以同一个
reference answer 作为正例**。这是循环论证。它同时污染两件事：主结果 §3.1 的 margin，
以及 §3.2 "验证在环是唯一有效成分"的归因——因为 Stage 6 恰好也是**唯一泄漏评测目标的阶段**，
现有数据无法区分"验证在环真的改善了 reward signal"和"验证在环只是把 rubric 调得对这个 gold 更友好"。

审计过程中还发现两处比预想更糟：`drop_contradicted` 和"gold 判不过则降权"这两条**原本没有任何开关**，
只 gate 了 `drop_gold_failures` 并不能真正关掉 gold 侧。已新增 `use_gold_signal` /
`use_negative_signal` 两个总开关。

#### 4.3.2 绝对分数分解：增益确实主要是"gold 涨"，但并非全面放松

把 margin 拆回绝对分数（n=150，即七个来源都有完整 7 档梯子的题）：

| response 档 | 质量级 | shipped | baseline | agentic | Δ(agentic−baseline) |
|---|---|---|---|---|---|
| **gold** | 5 | 0.717 | 0.767 | 0.872 | **+0.105*** |
| missing_step | 3 | 0.653 | 0.666 | 0.744 | **+0.078*** |
| terse_correct | 4 | 0.425 | 0.522 | 0.565 | **+0.043*** |
| right_method_wrong_answer | 2 | 0.519 | 0.499 | 0.505 | +0.006 |
| numeric_error | 2 | 0.466 | 0.442 | 0.455 | +0.013 |
| verbose_empty | 1 | 0.330 | 0.313 | 0.279 | −0.034* |
| off_topic | 0 | 0.085 | 0.121 | 0.113 | −0.008 |

审稿意见推测"六档降级里五档分数反而更高，rubric 整体更松"。**实测只部分成立，而且方向比推测的有利**：
显著上升的三档是 `gold`、`missing_step`、`terse_correct`——恰好是质量级 ≥3 的**全部"好"答案**；
四档真正的坏答案（rmwa / numeric_error / verbose_empty / off_topic）**没有一档显著上升**，
`verbose_empty` 还显著下降。

所以这不是无差别的放松，而是"把正确但不完整的答案也认成好答案"。这本身是 rubric 该有的行为。
但它确实抬高了整体水位（`mean_level` +0.029, p=0.015），所以原始 margin 被高估了。

#### 4.3.3 决定性检验：把泄漏的正例整个删掉

把 `gold` 这一档从指标里完全删除，只在**六个降级档之间**按构造时的质量级算判别力。
评测的降级档由 `harness/eval/responses.py` 独立生成，generator 从未见过（Stage 6 自己合成的 negative
与评测梯子的 token Jaccard 中位数只有 0.171，820 对里仅 3 对 >0.8），所以这一步是干净的。

对 5 个指标 × 4 个来源共 20 个对比做 BH-FDR 校正后：

| 指标（已删 gold） | agentic Δ | q | 存活 |
|---|---|---|---|
| separation（好−坏） | +0.066 | 0.0005 | **是** |
| z_separation（尺度归一化） | +0.134 | 0.018 | **是** |
| spread（动态范围） | +0.071 | 0.0005 | **是** |
| AUC | +0.032 | 0.092 | 否 |
| ranking accuracy | +0.014 | 0.281 | 否 |

**主结论没有被击穿，但被显著收窄。** 删掉泄漏的正例、并对尺度归一化之后，agentic 相对 baseline
仍有稳健的优势（z_separation +0.134, q=0.013），所以增益**不是纯循环的**。
但**排序类指标（ranking accuracy、AUC）在 FDR 之后都不再显著**。也就是说：
存活下来的效应是"把好答案和坏答案的分数拉得更开"，**不是"更准确地把答案排对顺序"**。

一个必须说清的残余泄漏：评测的六个降级档（除 `off_topic` 外）**都派生自同一个 reference_answer**。
Stage 6 拟合 reference 的内容，理论上仍可能顺带提高所有 reference 派生文本的分数。
所以 §4.3.3 大幅削弱但没有 100% 消除泄漏，真正稳健的推断是下一节的**相对比较**。

#### 4.3.4 拆开 Stage 6：泄漏的那一半贡献了几乎全部增益

Stage 6 有两个机制，泄漏程度完全不同：

- **gold 侧**（`drop_gold_fail` + `drop_contradicted` + gold 不过则降权）→ 以 reference_answer 为准 → **对评测泄漏**
- **negative 侧**（没有任何 negative 判失败则降权、全部失败则升权）→ negative 由 generator 独立合成 → **不泄漏**

新增两个消融各生成 200 份 rubric（`agentic-goldonly` 只开 gold 侧，`agentic-negonly` 只开 negative 侧），
在**同一批 response 梯子**上判定。删掉 gold 之后的结果：

| 指标（已删 gold，n=150） | agentic | **goldonly**（泄漏） | **negonly**（干净） | realneg（真实反例） | noval |
|---|---|---|---|---|---|
| separation | **+0.066 (q=.0007)** | **+0.077 (q=.0007)** | +0.024 (q=0.11) | −0.002 (q=0.99) | −0.013 (q=0.50) |
| spread | **+0.069 (q=.0007)** | **+0.085 (q=.0007)** | +0.008 (q=0.83) | −0.013 (q=0.60) | −0.029 (q=0.13) |
| z_separation | **+0.134 (q=.013)** | +0.112 (q=0.072) | +0.066 (q=0.26) | +0.015 (q=0.96) | +0.002 (q=1.00) |
| AUC | +0.032 (q=0.089) | +0.018 (q=0.42) | +0.006 (q=0.85) | +0.002 (q=0.99) | −0.002 (q=0.99) |
| ranking accuracy | +0.014 (q=0.31) | +0.002 (q=0.99) | +0.000 (q=1.00) | −0.009 (q=0.67) | −0.016 (q=0.31) |

q 值现在由 `scripts/confound_audit.py` 计算并落盘到 `confound_audit.json` 的 `gold_excluded.*.fdr`
（此前只活在这段散文里，无从核对——审计员按声明的族手算复原了它们，但那不是一条可持续的路）。
**族 = 5 指标 × 6 个非参照来源 = 30 个对比**，随 `agentic-realneg` 的加入从 15 扩到 30，
因此本表 q 值与本轮之前的版本不同（点估计一个未变；变的只是多重比较的分母）。
族的定义由 `scripts/selftest.sh` 的回归测试守住，不会再随表的内容漂移。

**判读结果对我们不利，如实记录**：

1. **增益由泄漏的那一半携带。** `goldonly` 在两个存活指标上**等于或超过**完整 agentic
   （separation +0.077 vs +0.066；spread +0.086 vs +0.071）。
2. **干净的那一半几乎没有贡献。** `agentic-negonly` 在 FDR 之后**没有任何一个指标显著**，
   最好的 separation +0.024 也只有 q=0.11。
3. 因此**"用独立构造的反例筛选 criterion"这个卖点，本次实验没有支持**。真正起作用的是
   "删掉 reference answer 都满足不了的条目"——那是一个**接地/自洽性过滤**，不是判别力过滤，
   而且正是触及评测正例的那一半。

绝对分数也印证同一件事：`goldonly` 的 gold 分数涨得最多（+0.130***，比完整 agentic 的 +0.105 还高），
而 `negonly` 的 gold 几乎不动（+0.019，不显著）。

#### 4.3.5 修订后的主结论

原表述（§3.2）："验证在环是唯一有效成分。" **这个表述必须收窄成：**

> 在删除泄漏正例并做尺度归一化、FDR 校正之后，agentic rubric 相对单次生成 baseline 仍有稳健优势，
> 但该优势**只体现在"拉开好坏答案的分数间距"（separation / z_separation / spread），
> 不体现在"排序更准"（ranking accuracy、AUC 均不过 FDR）**。
> 该优势几乎完全由 Stage 6 的 **gold 侧过滤**（删掉 reference answer 无法满足的条目）产生；
> 而 gold 侧正是与评测正例共享同一份 reference answer 的那一半，因此**即使删掉 gold 档，
> 也无法完全排除循环解释**。Stage 6 中唯一在构造上干净的 negative 侧过滤，
> 在本次实验中**没有产生任何通过 FDR 的收益**。

换句话说，`docs` 里那个漂亮的说法——"criterion 不是写出来的，是搜索+验证出来的"——
**本次实验只支持它的一半**：起作用的是"拿一个已知正确的答案去核对条目写得对不对"，
而不是"拿反例去筛选条目有没有判别力"。这是一个更弱、但站得住的主张。

#### 4.3.5b 但这条"最硬的结论"是纯 science 域效应

§4.3.3 的 pooled 数字掩盖了一件必须说的事。把同一份删 gold 的审计按域拆开：

| 指标（已删 gold，Δ agentic − baseline） | science (n=56) | medicine (n=94) |
|---|---|---|
| separation | **+0.119**, p=1e-4, q=.001 | +0.034, p=0.037, q=0.097 |
| **z_separation** | **+0.325**, p=0.0024, q=0.009 | +0.021, **p=0.61, q=0.73** |
| spread | **+0.106**, p=3e-4, q=.002 | +0.048, p=0.0085, q=0.036 |
| AUC | **+0.074**, p=0.016, q=0.040 | +0.007, **p=0.68, q=0.79** |
| ranking accuracy | +0.036, p=0.043, q=0.088 | +0.001, **p=0.95, q=0.95** |

**在占审计样本 62% 的 medicine 上，z_separation、AUC、ranking accuracy 全是彻底的零效应**
（p 分别为 0.61 / 0.68 / 0.95）。pooled 的 z_separation +0.134 (q=0.013) 由 38% 的 science 样本拉动。
medicine 侧只有 separation 和 spread 还有小而正的效应，且都比 science 弱 3 倍以上。

审计样本本身还按域失衡：science 从主结果的 49.5%（98/198）掉到 37.3%（56/150）。
被剔的题里绝大多数是 science，主因是 `terse_correct` 与 gold 逐字相同被判 `identical to gold` 丢弃
（science 36 题 / medicine 4 题——science 的参考答案本身常常就很短）。
**流失的正是效应最强的那个域**，所以 pooled 数字既被 medicine 稀释、又在 science 上样本不足。

q 值为**域内**校正（每个域各自 5 指标 × 6 来源 = 30 个对比），不是跨域联合校正；
分域看时 science 的 z_separation 仍过 FDR（q=0.009），medicine 仍不过（q=0.73）。

（这个拆分一直存在于 `confound_audit.json` 的 `by_domain` 块和 `confound_audit.txt`，
但此前从未写进本报告——**而且它此前藏在一个默认关闭的 `--by-domain` 开关后面**。
现在分域是默认行为，要用 `--no-by-domain` 才能关掉：一个默认不输出的拆分，
就是一个迟早会被漏掉的拆分，这次就漏掉了。）

#### 4.3.6 为什么没有做"无泄漏 gold"（第三步）

原计划是从 Stage 2 的 k 次独立 rollout 里挑一个判定正确的当评测正例。
`rollout_verdicts` 确实存了每次 rollout 的正确性判定，实现成本不高，**但这样做会用一个泄漏换另一个泄漏**：
rollout 会喂给 Stage 3（reconcile）和 Stage 5（drafting），所以"正确 rollout"对 agentic 的**起草阶段**
同样不独立，而对 baseline / shipped 则完全无关——反而会制造一个偏向 agentic 的新混淆。

相比之下 §4.3.3 的做法（把正例整个删掉，只在 generator 从未见过的降级档之间比较）更干净，
且已经涵盖了"用一个非 reference 的正确答案当正例"这件事——`terse_correct`（质量级 4）就是
一个正确但不含推导的答案，它已经在"好"的一侧参与 separation 与 AUC 的计算。所以第三步是冗余的，
放弃它是出于设计正确性而不是成本。

---

## 4.4 面向 rollout 的评测（RL 忠实协议）

> 全部数字：`results/pilot_v2/rollout_report.md` / `.json`、`results/pilot_v2/reward_hacking_probe.md`。
> 复现：`scripts/build_rollouts.py` → `scripts/eval_rubrics.py --metrics rollout`
> → `scripts/rollout_report.py` + `scripts/reward_hacking_probe.py`。

### 4.4.1 为什么必须换一个协议：第一个协议对 gold 侧不公平

§4.3 得出"增益来自泄漏的那一半"，但这个判决建立在一个**对 gold 侧过滤本身不公平**的评测设计上。

分清两件事：

* **部署时的泄漏**：不存在。RL 训练时 rubric 生成器看得到 reference answer 是**天经地义**的——
  rubric 本来就是离线、逐题、拿着标准答案写出来的；被打分的是策略采出来的 rollout，
  而 rollout 与 reference answer 是两个不同的对象。"拿已知正确答案核对条目"是一个
  **完全合法的离线步骤**，不是作弊。
* **评测时的循环**：存在。协议一把 `reference_answer` **本身**当作正例去打分。
  于是"gold 侧过滤"被同时用作训练目标和考试题目，它赢是必然的，
  而"负例侧过滤"从未被给过表现机会——协议一里根本没有真实的模型失败样本。

所以 §4.3 只证明了**"在以 reference answer 为正例的考试上，针对 reference answer 优化的那一半会赢"**，
这几乎是同义反复。要判断这两半在 RL 里谁真的有用，正例必须换成**生成器没见过的、模型自己采出来的 rollout**。

### 4.4.2 协议二的设计

1. **全新 rollout**：对每题采 k=6 个 rollout，缓存 salt 命名空间为 `eval-rollout:*`，
   与生成器 Stage 2 的 `rollout{i}` 完全隔离。落盘时逐字比对，
   **验证 2039 条评测 rollout 与 3000 条生成器 rollout 零碰撞**（`disjoint=True`）。
2. **独立 oracle**：逐条判定正确性，判定 prompt **完全看不到任何 rubric**，只看 question / reference / candidate。
3. **主指标**：AUC(rubric 分数, oracle 正确性)、best-of-n 选择准确率、Spearman、
   分数间隔、尺度归一化间隔——全部逐题计算、按题配对、BH-FDR 校正（族固定为 6 个判别指标，写在
   `scripts/rollout_report.py:FDR_FAMILY` 里）。
4. **真实 reward hacking 探针**：在 **oracle 判错**的 rollout 里，看分数与长度的相关性。

### 4.4.3 rollout 质量分布：这是本轮最大的实际限制

| 项 | 值 |
|---|---|
| 题数 / rollout 总数 | 200 / 2148 |
| rollout 总体正确率 | **0.835** |
| **有效题数（对错混合）** | **62** |
| 全对题数（排除） | 123 |
| 全错题数（排除） | 15 |
| oracle 二次判定自洽率 | 0.974（n=196，全题分层抽查） / 0.857（n=35，仅有效题子集） |

**策略太强了。** 在 high reasoning effort 下模型解对 83.5%，而且是双峰的：多数题它每次都对。
AUC / best-of-n / separation 都需要题内同时存在对与错，**138 题因此完全不进入任何指标**，
只剩 62 题。为此做过的努力：

* 用 prompt 制造质量差（`terse` 不许写推导、`rushed` 不许检查）——有效但不够。
* 把 `reasoning_effort` 调到 medium/low——**实测无效且更慢**（同一题 high 34s、medium 245s、low 316s，
  并且触发大量空回复）。该参数经 `chat_template_kwargs` 传递，服务端 profile 把它钉死在 high。
* 换更弱的 policy 模型——**外部模型全部 401 无权限，内部旧 checkpoint 全部超时**，本地无路可走。
* 对 138 题定向加采到 k=12（只加采标签未混合的题，`--top-up`）——**只救回 20 题**，
  说明剩下的题是真的简单，不是运气问题。

所以下面所有数字都建立在 **n≈61 题**上。这个样本量下，6 指标 × 6 对比的 FDR 家族里，
只有很大的效应才可能显著。**"没通过 FDR"在这里主要说明检验力不足，不等于"没有效应"。**

### 4.4.4 主结果：没有任何来源在 FDR 之后显著优于 baseline

| 指标 | n | `shipped` | `baseline` | `agentic` | `agentic-goldonly` | `agentic-negonly` | `agentic-realneg` |
|---|---|---|---|---|---|---|---|
| **AUC** | 61 | 0.711 | 0.747 | 0.784 | 0.745 | **0.810** | 0.780 |
| Δ vs baseline | | −0.036 (q=0.38) | — | +0.036 (q=0.72) | −0.002 (q=0.94) | **+0.063 (p=0.041, q=0.24)** | +0.033 (q=0.66) |
| Best-of-n 准确率 | 61 | 0.808 | **0.816** | 0.803 | 0.781 | 0.802 | 0.781 |
| Spearman | 58 | 0.301 | 0.373 | 0.432 | 0.381 | **0.440** | 0.406 |
| separation | 61 | 0.178 | 0.202 | 0.225 | 0.195 | 0.219 | 0.213 |
| z_separation | 57 | 0.873 | 1.026 | 1.091 | 0.953 | **1.123** | 1.078 |

**没有一格通过 FDR（全部 q>0.05）。** 最强的信号是 `agentic-negonly` 的 AUC +0.063（p=0.041, q=0.244）。

**但结论的方向整个反过来了**：协议一里 `agentic-goldonly` 携带全部增益、`agentic-negonly` 一无所获；
**协议二里 `agentic-goldonly` 是 agentic 家族里最差的（AUC 0.745，与 baseline 持平），
而 `agentic-negonly` 是最好的（AUC 0.810）**。

Best-of-n 是最实用的一条，结果最不客气：**没有任何 rubric 变体比 baseline 更会挑答案**
（baseline 0.816 最高，随机下界 0.613，oracle 上界 1.0）。

### 4.4.5 分域：science 上负例侧有信号，medicine 上什么都没有

| 指标（Δ vs baseline） | science (n=30) | medicine (n=31) |
|---|---|---|
| AUC `agentic-negonly` | **+0.115 (p=0.022, q=0.13)** | +0.013 (p=0.74) |
| AUC `agentic` | +0.030 (p=0.56) | +0.043 (p=0.27) |
| AUC `agentic-goldonly` | −0.025 (p=0.66) | +0.019 (p=0.65) |
| AUC `shipped` | −0.001 (p=0.98) | −0.071 (p=0.042) |
| Spearman `agentic-negonly` | +0.150 (p=0.058) | −0.006 (p=0.92) |
| z_separation `agentic-negonly` | +0.237 (p=0.23) | −0.020 (p=0.87) |

与 §4.3.5b 同一个模式：**能看到的信号都在 science，medicine 上是零**。
medicine 上唯一显著的是 `shipped` **比** baseline 差（AUC −0.071, p=0.042），
即论文原版 rubric 不如用同一个模型重跑论文 prompt——这是"模型差异"而非"方法差异"。

### 4.4.6 `agentic-realneg`：没有救活"反例筛选"

用真实失败 rollout 替代合成反例的消融。执行细节：Stage 2 的 3 次 rollout 在 **67% 的题上全部正确**，
没有真实反例可用，因此新增了 hard-negative mining——在 `rushed` / `terse` / `shortcut` 三种
提示下重采，用一个独立的正确性判定只留下真错的那些（salt 命名空间 `negmine{i}`，
与 Stage 2 和评测 rollout 都不相交）。

结果：**AUC 0.780，低于 `agentic-negonly` 的 0.810**；best-of-n 0.781，也低于 negonly 的 0.802。
在 science 上 realneg AUC +0.047 vs negonly +0.115。

**结论：没有救活。** "换成真实反例会更好"这个假设，本次实验**不支持**——
真实失败反而不如构造反例。一个可信的解释是：模型真实失败的方式集中且单调
（多为最后一步算错），构造反例覆盖的错误模式更宽，因此更能筛出有判别力的条目。
但 n=61 下这只是一个推测。

### 4.4.7 唯一方向一致、且分域显著的发现：真实 reward hacking

在 oracle 判错的 rollout 里，rubric 分数与长度的相关性（越低越好，按题聚类 bootstrap）：

| 来源 | 合并 (n=61) | science (n=31) | medicine (n=31) |
|---|---|---|---|
| `shipped` | +0.491 | +0.228 | +0.614 (**Δ +0.213, p=0.005**) |
| `baseline` | +0.402 | +0.272 | +0.402 |
| `agentic` | +0.239 (p=0.10) | **−0.040 (Δ −0.312, p=0.044)** | +0.550 (p=0.082) |
| `agentic-goldonly` | +0.228 (p=0.17) | **−0.109 (Δ −0.381, p=0.018)** | +0.545 (p=0.11) |
| `agentic-negonly` | +0.274 (p=0.22) | −0.012 (Δ −0.284, p=0.069) | +0.615 (**Δ +0.214, p=0.029**) |
| `agentic-realneg` | +0.238 (p=0.071) | **+0.048 (Δ −0.224, p=0.050)** | +0.484 (p=0.34) |

**在 science 上 agentic 把这个相关性从 +0.272 压到 −0.040**（p=0.044），
即长而错的答案不再拿高分，这正是 RL 里最该避免的漏洞。

**但在 medicine 上符号反过来**：agentic 让相关性从 +0.402 升到 +0.550（p=0.082），
`agentic-negonly` 升到 +0.615（p=0.029，方向错误且显著）。medicine 的题多为需要覆盖多个要点的
论述题，条目更多的 rubric 天然给更长的回答更多得分机会，这个副作用在 medicine 上压过了收益。

同样的域依赖，第三次出现。

### 4.4.8 两个协议放在一起怎么读

| | 协议一（合成降级梯子） | 协议二（真实 rollout） |
|---|---|---|
| 正例 | `reference_answer` **本身**（生成器见过） | 模型 rollout（生成器没见过） |
| 负例 | 按配方构造的降级答案 | 模型真实失败 |
| n | 151–198 题 | 61 题 |
| agentic vs baseline | separation/spread 显著（science） | 全部不显著 |
| **哪一半 Stage 6 有用** | **gold 侧**（negonly 全零） | **negative 侧**（goldonly 最差） |
| reward hacking | agentic 更好 | agentic 更好（仅 science） |
| 域依赖 | science 强、medicine 零 | science 强、medicine 零或反向 |

**一致的部分**（两个协议、不同正负例、不同 n 都指向同一件事，可信度最高）：

1. **所有效应都是 science 域效应**，medicine 上无论哪个协议都接近零。
2. **agentic 系压低错误答案的得分**（协议一 verbose_empty −0.034；协议二错误 rollout −0.090, p=0.003），
   而不是抬高正确答案的得分（协议二里 agentic 的正确 rollout 平均分反而比 baseline **低** 0.067）。
   agentic 的作用是**更严格的下压**，不是更准的上抬。
3. **agentic 在 science 上更抗长度型 reward hacking。**

**冲突的部分**（必须如实报告，不能挑一个信）：

* 协议一说增益来自 gold 侧，协议二说 gold 侧最差。
* 最直接的解释是**协议一的判决是设计造成的**：以 reference answer 为正例的考试必然偏袒
  以 reference answer 为目标的过滤。换成模型自己的 rollout 当正例，这个优势就消失了。
* 但**不能就此断言"negative 侧才是真的有用"**：协议二里 negonly 的优势同样没过 FDR，
  且只在 science 上、且 n=61。诚实的表述是**两半谁更有用，本次实验都没有定论**。

---

## 4.5 换 judge 复现：agentic 的优势不是"和 judge 同源"造出来的

产物：`results/judgeswap/`（`SUMMARY.md`、`main.*`、`confound_audit.*`、
`cross_judge_agreement.*`、`criterion_agreement.*`、`headtohead_compare.*`）、
`runs/judgeswap_arms/`（四个运行臂的原始判定）、`configs/judgeswap*.yaml`、`scripts/judgeswap_*.py`。

### 4.5.1 被检验的威胁

agentic 的 rubric 由 `hy-t2t-glm-5.2-384k-fp8-L20A-t1-v2` 生成，而主 judge 是**同一个模型**。
`baseline` 只控制住了"同一个模型 + 论文 prompt"，**没有控制住 judge 与 generator 共享模型**——
agentic 的措辞风格可能天然合这个 judge 的口味。这是一个能解释掉全部判别力增益的竞争假说。

### 4.5.2 设计

**只换 judge**。六个来源的 rubric 与整套降级 response 梯子**逐字节复用** `runs/pilot_v2`
（硬链接，SHA-256 记录在 `judgeswap_manifest.json`），judge prompt、`paper_explicit` 聚合、
极性处理、条目乱序种子（7）全部不变。新 judge 为 `api_azure_openai_gpt-5.1`——
**不同厂商、不同模型家族**（同家族的 `api_ali_glm-5.2` 被刻意排除，它只是同一个 GLM-5.2 的另一条服务路径）。

题目取自主结果的配对队列，按域配平到 57 science + 57 medicine，
**每一处对比都在同一道题上配对**。4788 次判定 **0 失败**；5 格（0.10%）因 judge 在 `why` 字段里
写 LaTeX 反斜杠破坏 JSON 而丢失，分布在 `baseline`(2) / `agentic-goldonly`(2) / `shipped`(1)，
**来源中性**。最终 **n=109（52 science + 57 medicine）**。

关键设计是**三个对照臂**：同 judge 子集、同 judge 重采样（seed 7）、同 judge 重测（seed 8）。
没有"judge 与自己有多一致"这个地板，跨 judge 一致率是无法解读的。

### 4.5.3 效应量在两个 judge 之间统计上无法区分

因为两个 judge 判的是同一批题，"效应有没有变小"可以直接检验，而不是靠比两条 CI：

`DiD_q = (new_source_q − new_baseline_q) − (old_source_q − old_baseline_q)`

`agentic`，已删 gold，n=109：

| 指标 | 旧 judge 优势 | 新 judge 优势 | DiD | 95% CI | p |
|---|---|---|---|---|---|
| ranking_accuracy | +0.022 | +0.020 | −0.002 | [−0.019, +0.015] | 0.86 |
| auc | +0.043 | +0.052 | +0.009 | [−0.013, +0.032] | 0.47 |
| separation | +0.077 | +0.062 | −0.015 | [−0.039, +0.008] | 0.21 |
| z_separation | +0.187 | +0.209 | +0.022 | [−0.044, +0.086] | 0.52 |

**全部交互 CI 都包含 0。** 唯一的例外是 `spread`（动态范围，不是判别力）：DiD −0.049, p=0.004——
gpt-5.1 压缩**所有** agentic 系 rubric 的分数range（`agentic-noval` 动得最多，−0.059），
所以这是家族级的尺度效应，不是"共享模型的那个来源"独有的。

### 4.5.4 没有自我偏好

原始跨 judge 一致率确实是 `baseline` 最高，单看这一条像自我偏好。**但它不是**，
因为旧 judge 与**自己**的一致率也按同样的顺序排列——来源之间本来就有"能被稳定判定的程度"之差。
扣掉同 judge 地板之后（配对到题，`agentic` vs `baseline`）：

| 层级 | 指标 | Δ | 95% CI | p |
|---|---|---|---|---|
| 分数 | 平均绝对差 | −0.000 | [−0.010, +0.010] | 0.94 |
| 分数 | **排序一致率** | **+0.021** | [+0.003, +0.041] | **0.030** |
| 条目 | 逐条一致率 | −0.002 | [−0.012, +0.008] | 0.68 |

换 judge 让每个来源都损失约 2 个点的一致率，**幅度相同**。而在对 reward signal 真正重要的那个
指标上——两个 judge 是否把同一道题的若干回答**排成同样的顺序**——`agentic` 衰减得**更少**
（+0.021, p=0.030），与自我偏好的预测**方向相反**。审计员的第三个追问可以关掉了。

### 4.5.5 域间反差完整复现（这让主结论更硬）

| 已删 gold，agentic vs baseline | 旧 judge | 新 judge |
|---|---|---|
| science | **+0.320** (p=0.0039, n=52) | **+0.340** (p=0.0013) |
| medicine | +0.066 (p=0.18, n=57) | +0.088 (p=0.13) |
| **science − medicine 差距** | **+0.254** [+0.034, +0.481], p=0.028 | **+0.252** [+0.027, +0.478], p=0.030 |

**§4.3.5b 那个"纯 science 域效应"是真实性质，不是 judge 伪影。** 两个 judge 下域间差距的
点估计几乎完全相同（+0.254 / +0.252），且都显著。这是本报告里被独立复现得最干净的一条结论。

一个必须说的限定：这里 medicine 只有 57 题（主审计是 94 题），检验力更低。
**数据支持的是"medicine 效应小到测不出"，不是"medicine 效应恰好为零"。**

### 4.5.6 head-to-head 的反直觉结果是真的，但消融排序不是

| 对 | judge | B−A | 位移 | 结论 |
|---|---|---|---|---|
| baseline vs agentic | 旧 | −0.202 | | |
| | 新 | −0.132 | +0.070, p=0.46 | **未翻转** |
| shipped vs agentic | 旧 → 新 | +0.667 → +0.605 | −0.061, p=0.44 | 未翻转 |
| baseline vs agentic-noval | 旧 → 新 | +0.202 → −0.114 | −0.316, **p=0.0005** | **翻转** |

"judge 在盲评里更偏好 baseline，而所有判别力指标都说反话"这个矛盾（§5.2）
**在不同家族的 judge 下复现**，所以它是"问 LLM 哪份 rubric 看起来更好"这件事本身的性质，
不是旧 judge 的怪癖。两个 judge 在**逐题**层面只有 0.491 的一致率——
它们逐题吵架，却吵出同一个总体偏好。

**但消融之间的排序在两个 judge 下会翻转**：旧 judge 下 `agentic` 在 ranking accuracy、AUC、
z_separation 三项上压过 `agentic-goldonly`；新 judge 下三项**全部反过来**
（已删 gold 的 z_separation：agentic +0.209 vs goldonly +0.226）。

### 4.5.7 三方证据交叉：Stage 6 哪一半有用，仍然没有定论

这是本报告最需要放在一起读的三条，此前散在三处：

| 证据来源 | 对"Stage 6 哪一半有用"的判决 |
|---|---|
| 协议一（合成梯子，旧 judge） | **gold 侧**携带几乎全部增益，negonly 全零 |
| 协议二（真实 rollout，旧 judge） | **negative 侧**最好，goldonly 与 baseline 持平 |
| 换 judge（合成梯子，gpt-5.1） | **gold 侧**进一步反超完整 agentic |

三方**两票投 gold 侧、一票投 negative 侧**，但这不是简单的多数决：
两票投 gold 侧的都用**合成梯子 + reference answer 当正例**，而这正是 §4.4.1 论证过的、
对 gold 侧过滤系统性有利的协议。换 judge 没有改变协议，所以它复现的是**同一个偏袒**，
不是独立的第三票。唯一改了正例来源的协议二投给 negative 侧，但它 n=61、没过 FDR。

**结论：证据结构本身不允许判决。** 要判决必须在**足够功效的 rollout 协议**下重跑——
这正是 §4.6 做的事。

---

## 5. agentic 没有赢的地方（诚实清单）

这一节不做辩护，只陈述事实和最可能的原因。

### 5.1 明确失败：权重没有携带信息（P0-3 的靶子没打中）

取证 F5 指出 shipped 的权重与等权平均相关性高达 0.94，几乎不携带信息。这给了 agentic 一个明确目标：
**权重应该显著偏离等权，并且这种偏离要能换来判别力提升**。

实测（用真实 judge 判定，比较 `paper_explicit` 加权分数与等权分数）：

| 来源 | 加权 vs 等权相关性（越低越说明权重有信息） | 平均 \|加权 − 等权\| |
|---|---|---|
| `shipped` | 0.983 | 0.042 |
| `baseline` | 0.989 | 0.033 |
| `agentic-noval` | 0.985 | 0.043 |
| **`agentic`** | **0.989** | 0.034 |

**agentic 的权重与等权的相关性是 0.989，不但没有低于 shipped，反而是最高的之一。**
weight-calibration 阶段（Stage 7）**没有产出有信息量的权重**。

（注：本表的 0.98–0.99 与取证的 0.94 不是同一个量——取证算的是 criterion 层面的权重向量相关性，
这里算的是**聚合后 response 分数**的相关性，后者天然更高。可比的是**列与列之间的相对关系**，
而 agentic 在这个相对关系上没有优势。）

**这意味着**：agentic 在 §3.1 拿到的全部判别力提升，都来自**criterion 本身写得更好**，
而不是权重分配得更好。如果换成等权聚合，结论大概率不变。对后续同学的直接含义是：
**不必在权重设计上投入**，RaR 论文 Table 2"去掉 category label 反而更好"的发现在这里得到了呼应。

### 5.2 明确失败：盲评 head-to-head 中 baseline 被更偏好

匿名化、随机左右位置、双向各判一次（2,376 次判定，0 次失败），位置偏置校正后：

| 对比 | A 胜率 | B 胜率 | 位置翻转率 |
|---|---|---|---|
| `shipped` vs `baseline` | 0.020 | **0.904** | 0.076 |
| `shipped` vs `agentic` | 0.177 | **0.712** | 0.111 |
| `shipped` vs `agentic-noval` | 0.025 | **0.909** | 0.066 |
| **`baseline` vs `agentic`** | **0.495** | 0.253 | 0.253 |
| `baseline` vs `agentic-noval` | 0.192 | **0.515** | 0.293 |
| `agentic` vs `agentic-noval` | 0.152 | **0.631** | 0.197 |

**judge 在盲评中更偏好 `baseline` 而不是 `agentic`（0.495 vs 0.253）**，甚至更偏好 `agentic-noval`
而不是 `agentic`（0.631 vs 0.152）。这与判别效度的结论**完全相反**。

可能的原因（未验证，只是假设）：

1. `agentic` 的 rubric 更长、更技术化、包含具体数值和公式，可能在"看起来像一份好评分标准"上不占优——
   judge 可能偏好简洁、覆盖面广、措辞规整的 rubric。
2. `agentic` 的 RaR 格式合规率只有 71.9%（见 §5.4），judge 可能在惩罚格式偏离。
3. 位置翻转率高达 25.3%（`baseline` vs `agentic`），说明这个判定本身就很不稳定。

**如何看待**：用户事前就指出这条指标"容易被质疑，只作为辅助证据"。这里它确实与主结果冲突。
我倾向于认为**判别效度是更可信的指标**——它衡量的是 rubric 作为 reward signal 的实际功能
（能不能把好答案和坏答案分开），而 head-to-head 衡量的是"看起来像不像一份好 rubric"。
RL 训练用的是前者，不是后者。但这终究是我的解释，**这条负面结果必须原样呈现，让后续同学自己判断**。

### 5.3 变差：饱和率与自信但错的探针

- **饱和率**（降级 response 仍得 ≥0.8 的比例）：agentic 0.151 vs baseline 0.095，**显著变差**（p=1e-4）。
  原因是 agentic 给 gold 打的分本来就高（0.814 vs 0.674），整体分布右移。条数受控后劣势缩小到
  +0.024（p=0.061，不显著），说明部分来自条目数量。**但这仍然是一个真实的隐患**，详见下面 §5.3.1。
- **"自信但错"探针**：agentic 0.504 vs baseline 0.490，**没有改善**（p=0.45）。
  这是 reward hacking 里最危险的一档（框架正确、过程漂亮、最终答案错），
  agentic 的 Stage 4 pitfall 挖掘**没有解决它**。条数受控后 agentic 为 0.466（p=0.25，仍不显著），
  而 `agentic-noval` 反而做到了 0.433（p=0.0009，显著改善）。这一档值得后续专门优化。
- **judge 自一致 kappa**：配对均值 agentic 0.767 vs baseline 0.847（**n=81**，Δ=−0.079，
  p=0.040，**q=0.053，未通过 FDR**）。此前此处写作 p=0.0026 与 0.722，均为笔误：
  0.722 是**边际**均值（在 agentic 有定义的 101 题上），而 Δ 是**配对**差值（在 81 题上），
  两者不可混用。现全表统一为配对均值，Δ 恰为两列之差。
  另外 kappa 是唯一存在来源间缺失差异的指标（agentic 有定义 101 题，shipped 171 题），
  且 n=81 的入选条件之一是"agentic 在该题上未饱和"，**并非随机缺失**。
  **这是指标假象，不是 judge 真的变得不稳定**，证据见 §5.3.2。

#### 5.3.1 饱和率对 RL 意味着什么（比预想的更细致）

审稿意见的推测是：分数堆在高位 → 好 rollout 之间的 advantage 差异变小 → GRPO 梯度变稀。
我把这个推测拆成可测量的量（`confound_audit.txt` §4），结果**只对了一半**：

| 量 | 含义 | baseline | agentic | Δ | goldonly | negonly |
|---|---|---|---|---|---|---|
| `good_spread` | 好答案之间的分数标准差 = **可用的 advantage 信号** | 0.176 | 0.203 | **+0.027**, p=0.002 | +0.028*** | +0.005 (ns) |
| `headroom` | `1 − gold`，**触顶前的余量** | 0.233 | 0.128 | **−0.105**, p=1e-4 | −0.130*** | −0.019 (ns) |
| `saturation` | 降级答案 ≥0.8 的比例 | 0.108 | 0.158 | **+0.050**, p=0.0007 | +0.056*** | +0.002 (ns) |

> ⚠️ **这三行是未经 FDR 校正的原始 p 值**，与紧邻的 §4.3.3 采用的 FDR 口径不同。
> 它们属于 §4.3 泄漏审计的 reward-geometry 家族（`confound_audit.py`，5 指标 × 4 来源），
> 而 §3 主表的 FDR 家族由 `METRIC_SPEC` 固定为 50 个指标。**跨节比较 p 值时请注意口径不同。**

**推测错的那一半**：好答案之间的分数并没有被压平，反而**拉得更开了**（`good_spread` +0.027, p=0.002）。
在一组"都还不错"的 rollout 里，agentic 给出的 advantage 信号比 baseline **更强**，不是更弱。

**推测对的那一半，而且更具体**：真正的风险是**天花板**。agentic 给 gold 打 0.87，只剩 **0.128** 的余量。
一旦策略学到 reference 级别，reward 就基本饱和，之后的改进拿不到任何梯度；同时 15.8% 的降级答案已经
≥0.8，处在同一个饱和区里，与真正的好答案挤在一起难以区分。所以对 RL 的判断应该是
**"顶部截断/触顶"风险，而不是"advantage 消失"风险**。

**一个可直接采纳的处方**：这两项损害**完全由 gold 侧造成**（goldonly 全部显著，negonly 全部不显著）。
所以如果后续同学更在意 reward 的动态范围而不是绝对分距，`agentic-negonly` 配置是免费的改良——
它保留了干净的一半、不动天花板（headroom −0.019 ns、saturation +0.002 ns），代价是几乎放弃了增益。
更实际的做法是保留完整 agentic 但在 RL 侧做 **reward 归一化/白化**（group-wise z-score），
把绝对水位问题交给归一化处理，这也是 GRPO 常规做法。

#### 5.3.2 kappa 下降是 base-rate 假象，不要误读

原始数据（`confound_audit.txt` §5）：

| 来源 | 原始一致率 `p_o` | Cohen's kappa | **PABAK** | criterion 通过率 | gold 通过率 | kappa 有定义的题数 |
|---|---|---|---|---|---|---|
| shipped | 0.929 | 0.737 | 0.858 | 0.432 | 0.621 | 171 |
| baseline | 0.966 | 0.847 | 0.932 | 0.437 | 0.657 | 153 |
| agentic-noval | 0.963 | 0.835 | 0.925 | 0.401 | 0.616 | 168 |
| **agentic** | **0.967** | **0.722** | **0.934** | 0.472 | 0.789 | **101** |
| agentic-goldonly | — | — | — | **0.482** | **0.806** | — |
| agentic-negonly | — | — | — | 0.399 | 0.637 | — |

四条互相独立的证据说明这是假象而非真实的不稳定：

1. **原始一致率 agentic 最高**（0.967），不是最低。两次判定实际上更少分歧。
2. **PABAK（prevalence-adjusted bias-adjusted kappa，= 2·p_o − 1）agentic 也最高**（0.934）。
   PABAK 正是为"类别极不平衡时 kappa 失效"设计的。
3. **通过率最高**（0.472，gold 上 0.789）。kappa 的期望一致率 `p_e` 随边缘分布偏斜而上升，
   `(p_o − p_e)/(1 − p_e)` 的分母被压缩，kappa 因此塌陷——即使 `p_o` 更高。
4. **kappa 有定义的题数骤降**（101 vs 153–171）。kappa 未定义正是"两次判定全部相同"的情形，
   它本身就是极端偏斜的直接证据；而这些题被排除后，剩下的样本对 agentic 反而更不利。

`agentic-goldonly` 的通过率最高（0.482 / gold 0.806）进一步确认了因果方向：
**是 gold 侧过滤删掉了"gold 都过不了"的难条目，抬高了通过率，从而压低了 kappa。**

**结论：本设置下 Cohen's kappa 不是合适的稳定性指标，应改用原始一致率或 PABAK，
两者都显示 agentic 的 judge 稳定性没有下降。** §3.5 表里的 kappa 一列请据此理解。

### 5.4 设计取舍导致的"变差"

- **RaR prompt 合规率**：agentic 71.9% vs baseline 87.9%。这是**故意的**——合规率衡量的是
  "是否符合 RaR 原版 prompt 的硬约束"（7–20 条、description 单句、Pitfall 以 `Does not mention` 开头等）。
  agentic 刻意统一了极性（不用失败式表述），也会写多句 description。**这个指标对 agentic 不适用**，
  列出只为透明。
- **子问题覆盖率**：agentic 97.4% vs 其余 100%（p=0.032）。绝对量很小（198 题里约 5 题），
  但方向是错的，而 F8 恰恰是"多小问覆盖不足"——**agentic 没有修好 F8**。
  机制上这与 §4.3.4 是同一件事：能把子问题漏掉的正是 gold 侧过滤，它按"reference 能否满足"删条目，
  而多小问题目的 reference 常常只详写了其中一两问，于是覆盖另外几问的条目被误删。
  这是"用 reference 当唯一裁判"的直接代价，也是 §4.3 泄漏问题在覆盖率上的镜像表现。
  修法很清楚：删条目前先按子问题分组，保证每组至少保留一条。

### 5.5 无法判断

- **覆盖率类指标**（claim recall 96%+）存在天花板效应，本设置下区分不出来源。
- **条数自适应性**：agentic 的 CV=0.310（vs shipped 0.176）说明条数确实更分散，
  `r(条数, 子问题数)`=0.159（vs shipped −0.129）方向正确，但 `r(条数, 题目长度)`=0.090
  反而低于 baseline 的 0.251。**"条数随复杂度自适应"这个卖点只得到了很弱的支持**，
  远没有达到"干净的卖点"的程度。

  分域拆开看，有一个支持 §3.3 主结论的细节。science 域上：

  | 来源 | 平均条数 | CV | modal share |
  |---|---|---|---|
  | shipped | 7.47 | 0.098 | 64.3% |
  | baseline | 10.58 | 0.179 | 19.4% |
  | agentic-noval | 14.09 | 0.153 | 40.8% |
  | **agentic** | **10.30** | **0.349** | 28.6% |

  关掉验证环之后，条数不但没有更自适应，反而**回落成一个偏大的近似常数**（14.09 条，CV 0.153）。
  也就是说条数的分散性主要不是 Stage 1 的复杂度估计带来的，而是 **Stage 6 按题逐条淘汰**的副产品：
  简单题上被淘汰得多、难题上留得多，条数才拉开。这与 §3.3 的结论一致——**验证环是唯一真正起作用的组件**，
  连"自适应条数"这个看起来属于生成侧的卖点，实际也依赖它。

---

## 6. Case study

完整原文见 `results/pilot_v2/case_study_wins.md`（agentic 优势最大的 2 题）和
`case_study_losses.md`（agentic 劣势最大的 1 题）。

### 6.1 `med-cdb031db3eb3`（medicine，MCQ：袖带血压 vs 动脉内压）

**`shipped`（7 条）** 的 3 条 Pitfall 是模板化的"Avoids Incorrect Choices — Does not mention or support
choice (A)/(C)/(D)"，逐字重复，只换字母。前 4 条正向条目里有"Discusses Measurement Method"
这类**在任何一道血压题上都成立**的条目。

**`baseline`（6 条）** 好一些，但仍有"Conciseness — Remains concise and avoids unnecessary detail"
（纯文体、与本题无关）和"Explanation Precedes Answer"（纯格式）。

**`agentic`（6 条）** 的差异集中在两条**带执行证据**的条目：

> **[Pitfall w=−2] Avoids Korotkoff reversal** — Avoids reasoning that because Korotkoff sounds appear
> when cuff pressure falls below systolic pressure, the sphygmomanometer reading must be less than
> intravascular pressure.
> `来源证据: pitfall · 验证: gold_pass=True, negatives_failed=2/3, discrimination=0.67`

这条抓的是**本题特有的、看起来很有道理的错误推理链**——它来自 Stage 2 的独立求解 rollout 里模型
真实犯过的错，并在 Stage 6 被验证为"gold 能过、3 个 negative 里 2 个不过"。
模板化的"Avoids choice (A)"做不到这件事：它在任何选择题上都成立，且任何没提到 A 的回答都能通过。

**同时这个 case 也暴露了 agentic 的缺陷**：第 1、5、6 条其实是同一件事的三种说法
（"Selects option B" / "States cuff BP exceeds intravascular" / "Selects option B"），
**Stage 7 的跨类别去重没有抓住它们**。6 条里有 3 条冗余，这是 §5 之外的一个实打实的实现缺陷。

---

## 7. F1–F10 逐条对照

| # | 取证发现 | 是否修复 | 证据 / 说明 |
|---|---|---|---|
| **F1** | 条数是常数而非自适应（science 64.6% 恰好 7 条，r(题长)=0.19） | **部分** | agentic CV 0.310 vs shipped 0.176，modal share 0.192 vs 0.343；但 r(题长)=0.090 反而低于 baseline 0.251。见 §5.5 |
| **F2** | 条目与题目弱接地 | **是** | 弱接地率 27.1% vs 41.0%（baseline）；锚点数 2.82 vs 2.19 |
| **F3** | 大量通用/可回收条目（science 28.9%） | **是** | 通用条目率 9.5% vs 21.0%；加权后 9.0% vs 18.5% |
| **F4** | 主观/文体条目占比高 | **是** | 主观率 7.8% vs 18.1%（baseline）、vs 32.0%（shipped）；客观可判定 98.9% vs 92.7% |
| **F5** | 权重与等权相关 0.94，几乎无信息 | **否** | agentic 0.989，未低于对照。**明确失败**，见 §5.1 |
| **F6** | 两 domain 的 Pitfall 极性约定相反 | **是** | agentic 单一极性率 100%，未分类极性 0%；评测侧做了四口径敏感性分析，见 §4.2 |
| **F7** | Essential 与 Pitfall 双重计分（medicine 16.2%） | **部分** | agentic 镜像率 20.2% vs baseline 25.8%，**p=0.22 不显著**。跨类别去重不够强，§6.1 的 case 也证实了这点 |
| **F8** | 多小问覆盖不足（每小问从 7.49 掉到 2.07） | **否** | agentic 子问题覆盖率 97.4%，反而低于其余的 100%。见 §5.4 |
| **F9** | 条目可验证性差 | **是** | 客观可判定比例 98.9% vs 70.5%（shipped） |
| **F10** | 与下游 reward 的关系未经检验 | **不适用** | 本地无法验证，属于后续 RL 实验的范围，见 §8 |

> **F8 补充**：§4.3.4 之后可以给出机制解释——漏掉子问题的正是 gold 侧过滤，
> 它按"reference 能否满足"删条目，而多小问题目的 reference 常只详写了其中一两问。
> 这是"用 reference 当唯一裁判"的直接代价。修法：删条目前按子问题分组，每组至少保留一条。

---

## 8. 给后续同学的建议

### 8.1 必须先处理的两个数据问题

1. **RaR-Science 的 test 集有 12.3% 的问题在 train 里出现过**。不去重直接做 RL + 评测，
   测出来的收益里有相当一部分是记忆。**做任何训练前先跨 split 去重**。
2. **9.15% 的行是重复题面**。同一题在训练集里出现多次会让 reward 权重被隐式放大。
   本 harness 的 `sample_examples(deduplicate_questions=True)` 已按题面去重，可直接复用这个逻辑。

### 8.2 RL 验证怎么设计

> **⚠️ 方法论硬要求（由 §4.3 得出，优先级最高）**：
> **rubric 生成阶段用到的参考答案，绝对不能同时充当评测/奖励的正例。**
> 本次实验就踩了这个坑：agentic 的 Stage 6 用 `reference_answer` 筛选条目，而判别效度的 `gold`
> 档是同一段文本，于是"rubric 变好"和"rubric 被拟合到这段文本"无法区分。
> 落到 RL 上，这个坑有两种形态，都必须避免：
>
> 1. **数据层面**：如果 rubric 由某题的 reference 生成，就不要再用同一个 reference（或其改写）
>    作为该题的评测答案或 reward 的锚点。评测集的正例必须来自 rubric 生成时未见过的来源。
> 2. **流程层面**：rubric 生成与 reward 评测如果共享任何"标准答案"，就必须像 §4.3.4 那样
>    做一次拆半消融，证明收益不是来自共享的那一半。**成本很低（我们这次全部复用缓存，0 次新 API 调用
>    就重算完了泄漏侧的对比），但不做的话整套结论都可能被推翻。**
> 3. **正例一律用模型 rollout，不要用 reference answer**（由 §4.4 得出）。
>    这是上面两条的建设性版本：不是"小心别泄漏"，而是"换一个正例来源就没有泄漏问题"。
>    做法见 `harness/eval/rollouts.py`——独立 salt 命名空间采样、逐字校验与生成器 rollout 不相交、
>    再用一个**看不到任何 rubric** 的 oracle 判正确性。
>    这样"生成器见过 reference"就完全不构成优势，因为它被考的是别的文本。
>    **注意这不是说"生成时看 reference 是作弊"**——离线拿标准答案写 rubric 完全合法，
>    RL 部署时也确实如此；不合法的只是**拿同一段文本当考题**。
> 4. **只在"结局混合"的题上评估 rubric 质量**（由 §4.6 得出，重要性被我们低估了）。
>    我们第一次跑协议二时对 200 题全量生成 rubric，结果 123 题全对、15 题全错，
>    只有 61 题能进指标——**一次功效失败，而不是一个发现**。
>
>    但这不只是省钱的技巧，它是**总体定义**问题：**GRPO 的 advantage 是对 group 均值取的，
>    所以一个全对（或全错）的 group 里每个成员的 advantage 都是 0，对梯度没有任何贡献。**
>    rubric 在这种题上打分准不准，按构造就不影响训练。
>    **结局混合的题不是为了凑样本量的权宜子集，而恰恰是 reward signal 唯一起作用的那个总体。**
>
>    做法见 `scripts/screen_questions.py`：先用 k=4 的 rubric-blind rollout 筛，
>    只对留下来的题生成 rubric。k=4 时"每道入选题的成本"最低（46 次调用，k=8 是 63 次），
>    而且**在 k=4 混合必然在 k=8 混合**（多采样只会加样本，不会抹掉已有的对错分歧），
>    所以筛选没有假阳性。筛选阶段必须**看不到任何 rubric**，且所有来源事后共享同一批 rollout。
>
>    附带的现实提醒：如果 policy 已经很强，务必先看正确率分布再决定 k。
>    我们试过用 prompt 逼模型"简答/赶时间"来制造分歧，**效果很弱**（正确率只从 84.9% 降到 79.9%）；
>    降 `reasoning_effort` 在这个端点上是无效的（见 `ROLLOUT_SETTINGS` 的注释），
>    压 `max_tokens` 会得到空回复而不是差回复。**真正可靠的杠杆是扩大题池后筛选，不是降低采样质量。**

**最小可信实验**：固定 policy 模型、固定 RL 算法（GRPO 或 PPO）、固定超参，只换 reward rubric 的来源。
至少三臂：`shipped` / `baseline` / `agentic`。**强烈建议加上第四臂 `agentic-noval`**——
§3.2 显示它在判别力上等同于 baseline 但结构性指标远好于 baseline，
这是一个天然的对照，能直接检验"判别效度是不是预测下游收益的正确指标"。

**如果预算允许，第五臂用 `agentic-negonly`**（`scripts/gen_rubrics.py --sources agentic-negonly`）。
它是 Stage 6 里**构造上不泄漏**的那一半：本地代理指标上它几乎没有增益（§4.3.4），
但它也**完全没有付出天花板代价**（headroom/saturation 均不显著变差，§5.3.1）。
如果下游 RL 里 `negonly` 反而追平甚至超过完整 `agentic`，那就说明本地测到的 gold 侧增益
确实主要是拟合假象——这是**最直接、最便宜的一个判决性检验**。

**benchmark**：medicine 用 HealthBench（论文用的就是它，且有"纯通用 rubric 把 29.7 砸到 12.5"这个
已知对照点）；science 用 GPQA-Diamond 或论文用的同一套。**务必报告去重后的结果**。

**预期现象**：如果本地的判别效度指标确实预测下游，应该看到 `agentic` > `baseline` ≈ `agentic-noval` > `shipped`。
`agentic-noval` 与 `baseline` 打平是这个假设的**关键检验点**：如果 `agentic-noval` 下游明显更好，
说明判别效度不是正确的代理指标，本地评测体系需要重做。

**失败时最可能的原因**（按可能性排序）：

1. **触顶**。agentic 给 gold 打 0.87，只剩 0.128 余量，且 15.8% 的降级答案已 ≥0.8（§5.3.1）。
   注意**不是**"好答案之间 advantage 变小"——实测 `good_spread` 反而更大（+0.027, p=0.002）；
   风险是策略学到 reference 水平后 reward 饱和、后续改进拿不到梯度。
   建议做 group-wise reward 归一化/白化，并监控 reward 触顶比例；这两项损害全部来自 gold 侧，
   `agentic-negonly` 不带这个副作用。
2. **"自信但错"未被惩罚**（§5.3）。policy 很可能收敛到"过程写得漂亮但答案错"，
   因为这一档在 agentic rubric 下得分 0.504，与 baseline 无差异。**这是最需要盯的 hacking 方向**。
3. **极性实现错误**。§4.2.1 表明这类 bug 不会被常规 sanity check 抓到，却能造成 15pp 的污染。
   接手时请**先对 reward 函数的极性处理写单元测试**，再开始训练。
4. **判别效度与下游脱钩**。这是最根本的风险，也正是 `agentic-noval` 这一臂要检验的。

### 8.3 方法层面的下一步

- **Stage 6 是唯一的有效成分**（§3.2），值得单独深挖：加大 negative response 的数量和多样性、
  用更强的 negative（而不只是 rollout 的错误）、把 discrimination 分数直接用作权重。
- **不要在权重设计上投入**（§5.1）。
- **修 Stage 7 的跨类别去重**（§6.1 的 case 里 6 条有 3 条冗余，F7 未显著改善）。
- **修多小问的预算分配**（F8 未修复，§5.4）。

---

## 9. 局限性

1. **样本量**。每域 100 题、配对 n=198。主结果的效应量很大（margin +0.113，CI 不含 0，p=1e-4），
   但子群分析（如按 `question_source` 分层）没有功效。
2. **单一 judge、单一模型**。生成和评测用的是同一个模型（`hy-t2t-glm-5.2`）。
   agentic rubric 是这个模型写的，也是这个模型判的，**存在自我偏好的风险**。
   `shipped`（由 o3-mini/GPT-4o 生成）在所有指标上垫底，一部分可能就是这个原因——
   这也是为什么方法层面的结论全部以 `baseline`（同模型、同 judge）为准。**换 judge 模型复核是必要的后续工作**。
3. **降级 response 由 LLM 生成**。虽然一次生成、所有来源复用（保证公平），
   但"降级"的真实性未经人工核验。构造出来的 `numeric_error` 未必是真实 policy 会犯的错。
4. **judge 单次判定**。自一致率 96–97%，但仍有噪声；本报告未做多次判定取多数。
5. **论文附录的统计量与公开数据对不上**。取证发现 Table 4/5/7/8 的数字与公开数据不符
   （标注 medicine 的那张表几乎精确匹配 science 数据）。因此本报告**没有引用论文附录的任何统计量**，
   所有对照基线都取自我们自己在全量数据上的取证。
6. **head-to-head 与主结果冲突**（§5.2），且位置翻转率高达 25%。这个矛盾没有被解决。
7. **本地指标全部是代理指标**。判别效度、query-specificity 都不是下游性能。
   F10 无法在本地验证。**本报告不能、也没有声称 agentic rubric 会带来更好的 RL 结果。**
8. **gold 泄漏没有被 100% 消除**（§4.3.3）。评测梯子的六个降级档除 `off_topic` 外全部派生自
   同一个 `reference_answer`，而 Stage 6 的 gold 侧正是拟合这段文本。删掉 `gold` 档已经去掉了
   最直接的循环，但"拟合 reference 内容"仍可能顺带抬高所有 reference 派生文本的分数。
   本报告因此把 `goldonly` vs `negonly` 的**相对比较**当作比绝对数值更可信的推断。
   彻底解决需要一批与 reference 完全独立、且人工核验过的正例，本地不具备条件。
9. **泄漏审计的样本量小于主结果**。审计要求七个来源都有完整 7 档梯子，n=150（主结果 n=198）。
   方向与主结果一致，但 CI 更宽。
10. **三个新消融只跑了判别效度与结构性指标**，没有跑 transfer / coverage / intrinsic / head-to-head，
    因为它们回答的不是泄漏问题。`agentic-negonly` / `agentic-realneg` 在那些指标上的表现未知。
11. **结论有强域依赖，而样本只覆盖两个域。** §4.3.5b / §4.4.5 / §4.4.7 显示几乎所有效应
    都集中在 rar_science，medicine 上为零或反向。**本报告无法判断这是"理科题 vs 论述题"的一般规律，
    还是这两个具体语料的特性。** 任何把结论外推到第三个域的做法都缺乏依据。
12. **协议二样本量小（n=61）。** 策略在 83.5% 的 rollout 上答对，138/200 题因题内标签不混合
    而完全无法进入指标。加采到 k=12 只救回 20 题。本地只有一个可用的 policy 模型
    （外部模型 401 无权限、内部旧 checkpoint 全部超时），无法用更弱的策略拉开质量分布。
    **§4.4 的"不显著"主要是检验力问题。**
13. **极性判定存在来源间的不对称。** agentic 系全部 Pitfall 的 `polarity_method` 是 `"enforced"`
    （生成时自己声明），shipped / baseline 走正则检测（baseline 有 141 条 `unclassified`）。
    **对照组的极性可能被判错，agentic 不可能。** §4.2 的整体口径扫描是补偿而非消除。
14. **幸存者偏差（方向对 agentic 不利，但仍是被静默丢弃的结果）。**
    `harness/pipeline.py` 丢掉任何来源产出空 rubric 的 uid，被丢的 2 题正是 **baseline 唯一的
    两次 JSON 解析失败**（200→198）。方向上这对 baseline 有利、低估 agentic，
    但"**baseline 在 1% 的题上直接产不出 rubric**"本身就是一个结果，不该被排除掉。
    intent-to-treat 口径（失败记 0 分）下，baseline 的所有均值应按 198/200 折算再加两个 0，
    即各指标约下调 1%——不改变任何结论的方向，但应当在引用 baseline 绝对值时说明。
    被丢弃的 uid 现在会写进 `runs/<run>/intent_to_treat_failures.json`。
15. **可复现性边界。** judge 的 `temperature` 从未显式设置，走服务端默认，
    因此**结果只在 `runs/cache/` 在场时可精确复现，而 cache 被 gitignore**。
    `EvalConfig.judge_temperature` 已加好但默认 `None`（与本次 pilot 一致），
    下一次完整运行应设为 0.0——代价是作废全部已缓存判定。
    另外 `max_tokens` **不在缓存键里**：只差 token 预算的两次调用共用一条缓存，
    因此提高预算不会重新拉取一个曾被截断的回复。本次运行零截断，但这对后来者是个真实的坑。
16. **`min_margin` 对缺档最敏感，而 judge 失败时用的是更少的档而非 NaN。**
    `_question_metrics` 在某一档判定失败时会用剩下的档计算 margin，只影响 3 个单元，
    但方向不可控，引用 `min_margin` 时应注意。
17. **换 judge 复现只换了 judge，没有换协议。** §4.5 用的仍是合成降级梯子、
    仍以 `reference_answer` 当正例。它排除掉的是"judge 与 generator 同源"这一个竞争假说，
    **没有、也不可能排除协议本身对 gold 侧过滤的偏袒**（§4.4.1）。
    因此 §4.5.7 里"两票投 gold 侧"不是两个独立证据，而是同一个协议被测了两次。
18. **消融之间的排序不能只用一个 judge 下结论。** §4.5.6：换 judge 后
    `agentic-goldonly` 在 ranking accuracy / AUC / z_separation 三项上全部反超完整 `agentic`。
    主结论（agentic vs baseline）稳健，但**任何"哪个 agentic 变体最好"的说法都不稳健**。
19. **换 judge 臂的 medicine 只有 57 题**（主审计 94 题）。域间差距本身在两个 judge 下都显著，
    但"medicine 为零"在该臂上的检验力更低。数据支持的是**效应小到测不出**，不是**效应为零**。

---

## 10. 工程记录：踩到的坑

| 坑 | 表现 | 处理 |
|---|---|---|
| reasoning model 返回 dict | `chat()` 返回 `{'reasoning_content','response'}` | `harness/llm.py` 统一兼容 str/dict |
| token 被 reasoning 吃光 | `response` 为空字符串 | 检测到空响应后**逐级提升 max_tokens** 重试；本次共 142 次空响应、全部恢复 |
| LaTeX 反斜杠破坏 JSON | baseline 2/200 题解析失败（`Invalid \escape`） | 多级 JSON 修复；仍失败的题**从所有来源一并剔除**以保持配对（200 → 198） |
| judge 输出被 ```json fence 包裹 | 7,024 次调用中 59 次解析失败（0.84%） | 修复器剥离 fence；失败的单条判定记为 error，不影响该题其他条目 |
| 并发上限 | 探测到 96 路并发 0 错误 | 生成用 48×2 并行，评测用 64，留余量 |
| **shell 命令被并发执行 3 次** | 3 个 `gen_rubrics` 同时写同一个 jsonl，文件涨到 598 行；后来一次并发写把 `aggregate_results.py` **截断为空文件** | 所有长任务改用 `mkdir` 原子锁保护（`scripts/run_pilot_driver.sh`）；产物用 `scripts/_repair_jsonl.py` 按 uid 去重；被清空的脚本已重写 |
| 子任务接口不一致 | `build_responses.py` 传 `resume=`，但函数签名是 `reuse=`；`load_response_sets` 收到 `RunDir` 而非路径 | 修正调用方，并让 `load_response_sets` 同时接受两种形式；补了签名绑定检查 |
| 聚合表全是空值 | `summarize_metric` 把结果放在 `per_group` 下，聚合脚本按顶层键取 | 修正取值路径 + 8 处列名不匹配 |
| **`--sources` 会静默删掉其他来源的结果** | 指标行文件用 `open("w")` 截断写。为两个新消融单独跑一次评测后，文件里**只剩这两个来源**，原来四个来源的 8000 行判定全没了；表现是聚合突然变空，不报任何错 | 改成按 `rubric_source` 合并：只替换本次评测的来源、保留其余（`_write_rows_preserving_other_sources`）。**修正时只带了 `--metrics discriminative`，所以当时只有 discriminative 恢复了**；grounding / lint / adaptivity 三族的逐题数据（共 18 项指标）一直到本轮才补回。这三族是零 LLM 指标，重算 0 次 API 调用 |
| **同一个 bug 的第二种形态：重跑会静默翻倍** | 上面的合并逻辑对**没有 `rubric_source` 字段**的行是累加而非替换，而 head-to-head 的行正好没有这个字段 | 合并时若本批含无来源归属的行，则整批替换旧的无归属行；并加了**写后断言**：任一来源的行数不得减少，减少即 ERROR |
| **聚合表的配对队列会被新来源静默改变** | `compute_table` 把**文件里的所有来源**交给 `summarize_metric`，而后者对全部组求 uid 交集。加入题集不同的 `agentic-realneg` 会静默缩小主表队列且不报错 | 配对前先按 `sources` 过滤；新增 `--expect-paired` 断言，队列大小变化即报错 |
| **逐条 verdict 没落盘** | 只存聚合分，导致任何与极性相关的事后分析都算不出来，且重算依赖 `runs/cache/` 还在（而 cache 被 gitignore） | discriminative 与 rollout 两条路径都新增 `per_criterion_rows`：criterion id、weight、polarity、`literally_true`、`met`、判定口径全部落盘 |
| **`PolarityMode.FAVOURABLE` 是恒等变换** | "给对照组 benefit of the doubt"这层保险**从未生效**（12,484 条 criterion 中 0 条不同），但报告里当作已生效的让利在写 | 删除该分支；报告改为只声明真正做了的事——整份 rubric 级别的最有利口径扫描（§4.2） |
| **rollout 采样的 `reasoning_effort` 无效** | 想用 high/medium/low 制造质量梯度，实测 medium/low **更慢**（245s/316s vs 34s）且大量空回复 | 改用 prompt 制造梯度（`terse` 不许写推导、`rushed` 不许检查）；温度上限压到 1.0，超过会让模型在推理里空转不出结果 |
| **rollout 评测把 7 个来源串行了** | `score_question` 里 `for source in sources` 是顺序 await，外层信号量又设成 1，引擎并发几乎闲置（14 请求/2 分钟） | 工作单元改成 (题, 来源) 对，一次性 gather 434 个；吞吐提高约 20 倍 |

**成本（累计口径）**。此前报告引用的 `calls=1755` 是 `llm_stats_gen.json` 的**最后一次分片**
（该文件每次生成都覆盖），现已过时。截至本轮的累计：

| 阶段 | 调用数 | 缓存命中率 |
|---|---|---|
| rubric 生成（7 个来源，含消融） | ~2,700 | 87–94% |
| response 梯子 | 2,161 | — |
| 评测 judge（协议一，7 来源） | ~8,400 | 30–67% |
| rollout 采样 + oracle（协议二） | 6,411 | 17–52% |
| rollout judge（协议二，7 来源 × 62 题） | 3,443 | 0.6%（首跑）→ 100%（重跑） |

全流程墙钟约 6.5 小时。**逐项数字请以各 `llm_stats_*.json` 为准，不要引用本表做精确核算**——
这些文件按阶段覆盖写，只有最后一次分片是准确的。

### 10.1 版本控制（本轮补上的）

本仓库在本轮之前**没有 `.git`**，而且所有源码 mtime 都被一次批量重写压成了同一个时间戳。
后果是**代码演进史无从查证**：无法从证据上排除"先看结果再改指标定义"这类问题。
这不是指控，是**查无可查**——而查无可查本身就是交接时的一个缺陷。

现已 `git init` 并提交当前状态。**此前的历史无法追溯地重建，只能从这一次提交开始。**
后续所有改动请走正常提交流程，尤其是任何**指标定义**或**FDR 家族**的改动——
这两处一旦在看过结果之后被调整，全部 q 值都会失去意义。
`scripts/selftest.sh` 里已经加了守住 FDR 家族定义的回归测试。

产物的取舍：逐题行（`*_per_question_rows.jsonl`）**入库**，因为报告里每个数字都由它们算出，
不入库就无人能复核；逐条 verdict（`*_per_criterion_rows.jsonl`，约 50 MB）**不入库**，
它们服务于事后追问，留在产出机器上即可。`runs/cache/` 同样不入库（见 §9 第 15 条的可复现性边界）。

---

## 11. 结论

**用户的猜想「一次性合成的 rubric 质量不够，agentic 能做得更好」——**

**被支持的**（两个域一致，不经过 judge、不经过 reference，两个协议都不影响）：

- **结构质量。** 通用条目率 21.0%→9.5%、主观率 18.1%→7.8%、锚点数 2.19→2.82、
  客观可判定比例 92.7%→98.9%、极性一致性 100%（彻底修掉 F6）。
  这是**全篇最干净的一块证据**：零 LLM 调用、定义与 n=45k 全语料取证一致、
  两域同向。不受泄漏审计和 rollout 协议的影响。

**被支持但仅限 rar_science**（这一条以前被写成无条件成立，是报告的错误）：

- **判别力：把好答案与坏答案的分数拉得更开。** 删掉泄漏的 `gold` 档、尺度归一化、FDR 校正后，
  science 上 z_separation **+0.325（p=0.0024, q=0.009）**、separation +0.119、spread +0.106、AUC +0.074。
  **medicine 上 z_separation +0.021（p=0.61）、AUC +0.007（p=0.68）、
  ranking accuracy +0.001（p=0.95）——彻底的零效应。**
  此前报告引用的 pooled z_separation +0.134 (q=0.013) 是被 medicine 稀释后的数字，
  且审计样本里 science 只占 37.3%（主结果里占 49.5%，被剔的题绝大多数是 science）。
- **降低长度型 reward hacking。** science 上错误 rollout 的分数-长度相关性
  从 +0.272 压到 **−0.040（p=0.044）**。**medicine 上反向**：+0.402 升到 +0.550（p=0.082），
  `agentic-negonly` 升到 +0.615（p=0.029，方向错误且显著）。
- 上述优势不是靠堆条目（条数受控后更大），也不是靠算错对照组的分
  （给对照组各自最有利的整体极性口径后仍领先：0.694 vs 0.556/0.560）。

**没有被支持的**：

- **"排序更准"，两个协议都没有。** 协议一删掉泄漏正例后 ranking accuracy（+0.014, q=0.31）、
  AUC（+0.032, q=0.089）不过 FDR；**协议二（真实 rollout，n=61）没有任何来源在 FDR 之后
  优于 baseline**，best-of-n 选择准确率甚至是 baseline 最高（0.816 vs agentic 0.803）。
- **"用反例筛选 criterion"这个核心卖点。** 协议一里 `agentic-negonly` 没有任何指标过 FDR；
  协议二里它虽然名义最好（AUC 0.810）也没过 FDR。
  换成真实失败 rollout 的 `agentic-realneg` **反而更差**（AUC 0.780），没有救活这个卖点。
- 权重信息量（F5 的靶子）：agentic 0.989，**没有优于对照**。判别力提升与权重无关。
- 盲评 head-to-head：judge **更偏好 baseline**（0.495 vs 0.253）。
- 覆盖率：与对照无差异（天花板效应）；子问题覆盖率反而略低（F8 未修复）。
- 饱和率与天花板余量：**显著变差**（headroom 仅 0.128），且全部由 gold 侧造成。
- 条数自适应（F1）：只有很弱的支持。

**无法判断的**：

- 下游 RL 性能（F10）——本地无法验证，这正是交给后续同学的部分。
- **Stage 6 的哪一半在起作用。两个协议给出相反答案**：协议一说 gold 侧携带全部增益、
  negative 侧全零；协议二说 gold 侧最差（AUC 0.745，与 baseline 持平）、negative 侧最好。
  最可能的解释是协议一的判决由设计造成（以 reference answer 为正例的考试必然偏袒
  以 reference answer 为目标的过滤），但协议二 n=61、检验力不足，不足以反向定论。**这一条悬而未决。**
- **为什么所有效应都集中在 science。** 一个可查的假设：medicine 多为需覆盖多要点的论述题，
  条目更多的 rubric 天然给更长的回答更多得分机会（这与 §4.4.7 medicine 上长度相关性
  反而升高一致）；science 多为有唯一正确答案的计算题，条目的对错更二元。未验证。
- head-to-head 与判别效度冲突的原因。

**最重要的一条附带发现**：agentic 的判别力优势几乎全部来自 **Stage 6**——
去掉它（`agentic-noval`）之后判别力与 baseline 无法区分，尽管结构性指标依然大幅领先。
**但 Stage 6 内部哪一半有用，取决于用哪个协议评测**（见上）。

所以本研究可以主张的、站得住的说法是这一句：

> 在**有唯一正确答案的理科题**上，把 rubric 的每一条拿去实际执行一遍再筛选，
> 能提升它作为 reward signal 的**分辨力**（拉开分距、压低长而错的答案），
> 但**提不了排序准确性**；在**医学论述题**上这些收益全部消失。
> 结构质量（具体、可判、有锚点）的改善则是无条件的，两域一致。

三条给后续同学的警告：

1. **结构性指标改善 ≠ reward signal 改善。** `agentic-noval` 是干净的反例。
2. **rubric 生成用到的参考答案，不能同时当评测正例。** 我们踩了这个坑，
   而且它不会被任何常规 sanity check 抓到——只有做拆半消融才能发现。代价极低（复用缓存即可）。
   反过来同样重要：**用 reference answer 当评测正例会系统性偏袒"对着 reference 优化"的方法**，
   质量评估里的正例应当一律换成模型 rollout（§4.4.1）。
3. **任何 pooled 结论都要先按域拆开看。** 本报告最初把一个纯 science 效应写成了普遍结论，
   而分域数据一直就在产物里。
