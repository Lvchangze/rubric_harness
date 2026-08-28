# RaR-Science / RaR-Medicine 数据取证

> 目标：用**全量数据的定量证据**回答一个问题 —— "论文里一次性合成的 rubric 到底哪里不好？"
>
> 结论先行：**"rubric 写得烂"这个笼统判断是不准确的。**真实情况是 —— 单条 criterion 的措辞质量其实不差，句内冗余也**不严重**（这一条我们的假设被数据证伪了）；真正的问题集中在**四个可量化、可修复的结构性缺陷**上：预算不随复杂度变化、四分之一的 reward 质量压在与题目无关的通用条目上、权重体系不携带信息、Pitfall 的符号约定在两个 domain 里是**反的**。
>
> 配套代码在 `analysis/`，全部数值输出在 `analysis/out/`。本文档所有数字均来自这些脚本的真实运行结果。

---

## 0. TL;DR：最重要的 10 个数字

| # | 发现 | RaR-Science | RaR-Medicine |
|---|---|---|---|
| 1 | rubric 条数几乎是常数（**64.6% 的题恰好 7 条**），与题目长度相关性极弱 | r=0.19，CV=0.11 | r=0.26，CV=0.14 |
| 2 | **Eq. 1 中落在"通用条目"上的 reward 质量占比** | **24.6%** | 15.2% |
| 3 | **落在主观/风格条目上的 reward 质量占比** | **40.2%** | 21.3% |
| 4 | "可回收" criterion（无题目锚点 + 批量复用标题） | 17.7% | 11.6% |
| 5 | 用 TF-IDF 把 criterion 归位到自己题目（1000 干扰项）的 top-1 准确率 | **56.8%** | 71.9% |
| 6 | 权重几乎由 category 决定：给定 category 后权重的条件熵 | 0.72 bit（Essential 98.0% 为 5） | 0.78 bit（Essential **100%** 为 5） |
| 7 | **Pitfall 极性：science 是"避免式"，medicine 是"失败式"，完全相反** | 80.3% 避免式 | 88.1% 失败式 |
| 8 | 多小问题目的 rubric 预算几乎不增加 | 单问 7.49 → 多问 7.99 条 | n/a（几乎无多小问题目） |
| 9 | 单条 criterion 开头模板集中度（最高频 6-gram 开头） | 1.32% | **3.19%**（"remains concise and avoids unnecessary detail"） |
| 10 | **测试集问题在训练集中出现过的比例（数据泄漏）** | **12.3%** | 3.6% |

---

## 1. 方法与可复现性

### 1.1 脚本

| 脚本 | 作用 | 输出 | 运行时间 |
|---|---|---|---|
| `analysis/extract_pdf.py` | PDF → 带页码的纯文本 | `out/paper_text.txt` | <1s |
| `analysis/common.py` | 数据加载 / 分词 / 展平 criteria | `out/criteria_{domain}.parquet` | — |
| `analysis/rubric_stats.py` | 基础统计、prompt 约束合规、权重信息量、可验证性 | `out/basic_stats.json`, `out/basic_summary.csv` | ~35s |
| `analysis/rubric_similarity.py` | 标题/模板样板化、跨题近重复、句内冗余、Pitfall 镜像 | `out/similarity_stats.json`, `out/top_titles.csv`, `out/within_question_redundancy_*.csv` | ~130s |
| `analysis/rubric_grounding.py` | 锚点接地、归属检索、reference 耦合与泄漏、子问题覆盖 | `out/grounding_stats.json` | ~20s |
| `analysis/collect_examples.py` | 复合指标（可回收率、reward 质量分布、Pitfall 极性、权重序违规）+ 逐字例证 | `out/composite_stats.json`, `out/examples.md` | ~15s |

复现命令：

```bash
cd analysis
python extract_pdf.py
python rubric_stats.py
python rubric_similarity.py --sample-criteria 60000 --sample-questions 8000
python rubric_grounding.py  --sample-criteria 60000 --distractors 1000
python collect_examples.py
```

### 1.2 数据范围与抽样说明

- **除特别注明外，所有统计都在全量 train+val+test 上计算**（science 22,917 题 / 172,017 条 criteria；medicine 22,408 题 / 214,658 条）。
- 三处使用了抽样，因为是 $O(n^2)$ 计算：
  - 跨题近重复：每域随机抽 **60,000 条** criteria，只在样本内互相搜索。**样本内搜索只会漏掉匹配、不会凭空造出匹配，因此报出的重复率是全量真实值的下界。**
  - 句内冗余：每域随机抽 **8,000 题**。
  - 权重仿真：随机抽 **4,000 题** × 8 次随机 0/1 抽样 = 32,000 次。
  - 归属检索：每域随机抽 **60,000 条** criteria，干扰池 **1,000 题**。
- 所有随机数种子固定为 0（部分函数用 1/2/3），脚本可重复运行得到相同结果。

### 1.3 方法学诚实性说明（三条重要的自我修正）

1. **句内冗余最初被严重高估。** 用普通 TF-IDF 时，medicine 有 57.8% 的题存在 cos≥0.5 的 criterion 对，看起来冗余很严重。但检查样例后发现，最相似的那些对其实是**平行但不同**的条目 —— 例如 "每 3 个月随访，持续 2 年" vs "每 6 个月随访，持续 2 年"，或 "识别 Pimozide" vs "识别 Penfluridol"。默认分词器会丢掉数字、符号和单字符，把这些判成一样。加入**"判别性标记"过滤**（数字 + 符号 + 语料中的稀有实体词必须一致才算真冗余）后，真实冗余率降到 medicine 7.9% / science 0.8%。**下文报告的是修正后的数字。**
2. **子问题检测最初把选择题选项误判成小问。** `(a)(b)(c)(d)` 的选项列表和 `(a)(b)(c)` 的小问在正则层面完全一样。加入"该 span 是否真的在提问"（含 `?` 或祈使动词）的判据后，science 的多小问题目从 12.32% 修正到 **2.95%**，另有 9.36% 被正确识别为选择题。
3. **"通用条目率"配了置换对照。** 把每条 criterion 与**随机另一题**的锚点比对，通用率跳到 96.4%（science）/ 98.5%（medicine），说明这个指标确实在测量"题目特异性"而不是词表巧合，判别力约 67–77 个百分点。

---

## 2. 数据概览

### 2.1 基本规模

| | RaR-Science | RaR-Medicine |
|---|---|---|
| train / val / test | 18,333 / 2,292 / 2,292 | 17,926 / 2,240 / 2,242 |
| 总题数 | 22,917 | 22,408 |
| 总 criteria 数 | 172,017 | 214,658 |
| 每题条数 mean ± std | **7.51 ± 0.80** | 9.58 ± 1.34 |
| 每题条数 median / max | 7 / 12 | 10 / 17 |
| description 词数 mean | 24.4 | 15.0 |
| title 词数 mean | 2.37 | 2.83 |
| question 词数 median | 42 | 34 |
| **reference_answer 词数 median** | **23** | 76 |
| **reference_answer 为空（0 词）的题** | **2,180（9.5%）** | 45（0.2%） |
| 主要来源 | natural_reasoning 50.4%、SCP-116K 38.4%、VNet 11.2% | medical-o1-reasoning-SFT 87.9%、VNet 9.9% |

### 2.2 类别分布

| 类别 | Science | Medicine |
|---|---|---|
| Essential | 30.58% | 21.60% |
| Important | 33.92% | 35.99% |
| Optional | 22.02% | 28.25% |
| Pitfall | 12.99% | 14.16% |
| 缺前缀 / 前缀非法 | 0.49%（837 条） | 0.007%（15 条） |

非法前缀实例（medicine）：`Importance`、`Esssential`（拼写错误）、`Option`、`Universal`、`Mandatory`。

### 2.3 ⚠️ 论文附录统计表与公开数据对不上

我们把论文 Appendix A.1/A.2 的统计表和实际数据（train+val，与论文口径一致）做了逐项比对：

| | 论文 Table 4+5（标注 **Medicine**） | 我们测到的 **Science** | 我们测到的 **Medicine** |
|---|---|---|---|
| Total examples | 20,166 | 20,625 | **20,166 ✓** |
| Avg. rubrics per question | 7.5 | **7.51 ✓** | 9.58 ✗ |
| Avg. question length (words) | 45.0 | **45.4 ✓** | 41.3 ✗ |
| Essential % | 30.7 | **30.6 ✓** | 21.6 ✗ |
| Important % | 34.1 | **33.9 ✓** | 36.0 |
| Optional % | 22.1 | **22.0 ✓** | 28.3 ✗ |
| Pitfall % | 13.1 | **13.0 ✓** | 14.2 |

**题目数量对得上 medicine，但所有内容统计量几乎精确匹配 science。** 论文 Table 7 的 caption 本身也写错了（在 science 一节里写 "Aggregate statistics for the full **medical** ... dataset"）。Table 7+8 标注为 science 的统计量（7.5 条 / 52.6 词 / 28.4-34.8-22.3-14.5）则与任一公开 split 都不精确匹配。

> **实用结论：不要引用论文的 Table 4/5/7/8，一律以本文档从公开数据重算的数字为准。** 这也顺带说明"论文对自己发布的 rubric 数据的基本画像都没有核对过"，与"从未做过 rubric 质量内在度量"的判断一致。

### 2.4 Prompt 硬约束的合规率

拿 `docs/00_paper_notes.md` §2.4 列出的可机检约束逐条核验：

| 约束 | Science | Medicine |
|---|---|---|
| 条数 ∈ [7,20] | 99.97% | 99.78% |
| title 为 2–4 词 | 98.67% | 95.46% |
| description 单句 | 99.94% | 99.65% |
| 有合法 category 前缀 | 99.51% | 99.99% |
| weight 在 prompt 允许范围内 | 100% | 100% |
| weight 落在 Ess=5 / Imp=3–4 / Opt=1–2 区间 | 77.59%<sup>†</sup> | 99.99% |
| Pitfall 以 "Does not mention"/"Recommends" 开头 | 3.54%<sup>‡</sup> | 78.56% |

<sup>†</sup> **注意：这两个区间只在 medicine 的 prompt 里写明了**（"Essential ... (weight 5) / Important ... (weight 3–4) / Optional ... (weight 1–2)"），science 的 prompt 只说 "use 1–5"，没给数值锚点。所以 science 的 77.59% **不算违反 prompt**，只是说明它没有形成一致的数值约定。science 真正的问题是**类别序被破坏**（见 §9.1，41.3% 的题目里 Optional 权重 ≥ Important 权重），那是与 prompt 无关的内在矛盾。

<sup>‡</sup> science 的 prompt 同样没有强制这个开头（只在示例里出现过），所以也不算违规，但**说明两个 domain 的 Pitfall 写法完全不同** —— 这直接导致了 §9.3 的符号问题。

> 值得注意：**格式层面的合规率其实很高（>95%）**。单次生成的失败不在"格式写错"，而在"内容层面的结构性偏差"。这是我们的 harness 需要瞄准的地方 —— 加一个 JSON schema 校验器是不够的。

---

## 3. 【发现 1】条数预算是常数，不随题目复杂度变化

Prompt 明确要求 *"Choose 7–20 rubric items **based on the complexity of the question**"*。实际上：

| | Science | Medicine |
|---|---|---|
| 众数条数 & 占比 | **7 条，占 64.56%** | 10 条，占 36.18% |
| 前两个众数合计占比 | **88.89%**（7 或 8 条） | 61.63%（9 或 10 条） |
| 变异系数 CV | **0.107** | 0.140 |
| 实际用到的区间 | 5–12（占允许区间 7–20 的 57%） | 5–17 |
| 与 question 词数的相关性 | **r = 0.19** | r = 0.26 |
| 与 reference 词数的相关性 | r = 0.24 | r = 0.30 |

**Science 的 rubric 条数基本是一个常数 7。** 一道 12 词的选择题和一道 300 词的三小问推导题，拿到的 criteria 数量几乎一样。

真实对照（同一数据集内）：

- `rar_science:1`，问题 15 词（"How many visible photons (∼ 5000 Å) does a 100-W bulb with 3% efficiency emit per second?"），reference 只有 `10^{19}` → **8 条 criteria**
- `rar_science:63`，问题 100+ 词、三个小问 (a)(b)(c)、reference 730 词的量子散射推导 → **8 条 criteria**

**这解释了后面几乎所有问题**：预算被钉死在 7 条，而 Essential 事实往往就占掉 3–5 条，剩下的名额只够塞通用条目；复杂题目则被迫欠覆盖。

---

## 4. 【发现 2】样板化程度：medicine 是"复制粘贴"，science 是"同一个模子"

两个域的样板化形态**不同**，需要用不同指标才能测到。

### 4.1 标题复用

| 指标 | Science | Medicine |
|---|---|---|
| 唯一标题数 / 总条数 | 73,164 / 172,017 = **42.5%** | 134,023 / 214,658 = 62.4% |
| 标题被复用 ≥10 次的 criteria 占比 | **42.27%** | 25.26% |
| 标题被复用 ≥100 次的 criteria 占比 | 21.87% | 13.11% |
| 标题被复用 ≥1000 次的 criteria 占比 | 4.53% | 3.92% |
| Top-50 标题覆盖的 criteria 占比 | 14.82% | 10.89% |

**Top-10 高频标题**（完整 top-50 见 `analysis/out/top_titles.csv`）：

| Science | 次数 | Medicine | 次数 |
|---|---|---|---|
| unit consistency | 1,942 | **conciseness** | **4,140** |
| conciseness | 1,766 | avoids unnecessary detail | 1,664 |
| clarity and conciseness | 1,611 | contextual relevance | 1,451 |
| concise explanation | 1,441 | avoids unnecessary details | 1,169 |
| final answer clarity | 1,026 | concise explanation | 960 |
| step by step reasoning | 956 | clear conclusion | 895 |
| conciseness and clarity | 852 | brevity and clarity | 746 |
| clarity and organization | 691 | identifies condition | 744 |
| avoid misinterpretation | 654 | identifies correct answer | 620 |
| numerical accuracy | 572 | avoids irrelevant details | 595 |

> Top-10 里 science 有 7 个、medicine 有 8 个是**纯风格标签**（简洁、清晰、组织），而不是学科内容。

### 4.2 措辞模板

| 指标 | Science | Medicine |
|---|---|---|
| 完全相同的 description 出现在**不同题目**上 | 0.02%（14 个字符串） | **3.50%（1,240 个字符串）** |
| 掩去稀有词后模板重复率 | 0.03% | 4.58% |
| 只保留语料 top-200 高频词的"骨架"重复率 | 0.24% | **15.12%** |
| 最高频的 6 词开头占全部 criteria 比例 | 1.32% | **3.19%** |
| Top-10 开头合计占比 | 4.01% | **5.62%** |

**Medicine 最高频的 criterion 开头**（6,851 次，占全部 criteria 的 3.19%）：

> `Optional Criteria: Remains concise and avoids unnecessary detail ...`

这句话几乎逐字来自 **生成 prompt 里的示例**（*"If brevity is valued, include a rubric like: – Optional Criteria: Remains concise and avoids unnecessary detail."*）。模型把 prompt 里的示例当模板抄了 6,851 次。

**Science 最高频开头**（2,272 次，1.32%）：`The response must clearly state that ...`

### 4.3 跨题近重复（TF-IDF cosine，60k 抽样，下界）

| 与另一道题的 criterion 最大余弦相似度 | Science | Medicine |
|---|---|---|
| ≥ 0.60 | 9.48% | **24.89%** |
| ≥ 0.80 | 0.86% | **7.86%** |
| ≥ 0.90 | 0.13% | 4.71% |
| ≥ 0.99（基本逐字相同） | 0.02% | **3.50%** |
| 平均最大跨题相似度 | 0.427 | 0.517 |

真实例证（不同题目上出现的完全相同文本）：

- Science，出现在 `rar_science:2501` 和 `rar_science:20372`：
  > `Essential Criteria: The response must clearly state that the correct answer is (b) False.`
- Science，出现在 `rar_science:5254` 和 `rar_science:20552`：
  > `Pitfall Criteria: The response should avoid rounding intermediate values too early, as doing so can lead to inaccuracies in the final result.`
- Medicine，`rar_medicine:691`、`rar_medicine:1302`、`rar_medicine:8600` …：
  > `Optional Criteria: Remains concise and avoids unnecessary detail while explaining the diagnosis.`

### 4.4 "通用条目率"（generic criterion rate）—— 核心指标

**定义**：为每道题抽取"锚点集合"= 该题 question + reference_answer 中的稀有词（语料文档频率 ≤ 1% 的题）∪ 数字 ∪ 符号。若一条 criterion 的 description 不含**任何**本题锚点，判为 **generic**。

**置换对照**：把同一条 criterion 拿去和**随机另一题**的锚点比对。

| 指标 | Science | Medicine |
|---|---|---|
| **通用条目率** | **28.87%** | **21.97%** |
| 置换对照下的通用率 | 96.39% | 98.54% |
| 判别力（真实 vs 置换的差） | **67.5 个百分点** | 76.6 个百分点 |
| 锚点 <2 个的"弱接地"条目率 | 52.61% | 43.53% |
| 每条 criterion 平均锚点数 | 1.78 | 2.07 |
| **过半 criteria 都是 generic 的题目占比** | **16.66%** | 4.37% |

**按类别拆开看，问题极度集中在 Optional 和 Pitfall：**

| 类别 | Science 通用率 | Medicine 通用率 |
|---|---|---|
| Essential | 14.93% | 8.73% |
| Important | 23.00% | 10.09% |
| **Optional** | **52.32%** | **43.93%** |
| **Pitfall** | **37.22%** | 28.55% |

> 也就是说：**Essential/Important 条目基本是贴着题目写的，问题几乎全在 Optional 和 Pitfall 这两类"填充位"上。** 这是一个非常可操作的结论 —— agentic 流程不需要重写整个 rubric，只需要重做后半段。

### 4.5 归属检索测试（query-specificity 的直接度量）

给定一条 criterion，用 TF-IDF 在 **1,000 道随机干扰题 + 它自己的题** 中找它属于哪一题：

| 指标 | Science | Medicine |
|---|---|---|
| **Top-1 归位准确率** | **56.82%** | 71.88% |
| Top-5 归位准确率 | 71.77% | 82.88% |
| 随机基线 | 0.10% | 0.10% |
| 与自己题目**零词汇重叠**的 criteria | 10.08% | 7.56% |
| 被 ≥10 个干扰题击败的 criteria | **23.66%** | 15.07% |

**43% 的 science criteria 无法被归位到自己的题目**，哪怕干扰池只有 1000 道题。

### 4.6 复合指标："可回收 criterion 率"

**定义**：`generic`（无本题锚点）**且** `标题被复用 ≥10 次` —— 即这条 criterion 可以原样贴到本域任何一道题上。这正是论文 RaR-PREDEFINED 的形态。

| 指标 | Science | Medicine |
|---|---|---|
| **可回收 criterion 率** | **17.67%** | 11.60% |
| 每题平均可回收条数 | 1.33 | 1.11 |
| 有 ≥2 条可回收条目的题 | **36.86%** | 29.18% |
| 有 ≥3 条可回收条目的题 | 16.21% | 8.65% |
| 完全没有可回收条目的题 | 28.63% | 29.17% |

真实例证（science，标题复用次数在括号里）：

- （852×）`Optional Criteria: Provides a succinct explanation that is easy to follow without extraneous details, ensuring the response is accessible.`
- （495×）`Optional Criteria: The response may discuss or apply appropriate significant figures based on the given data in the problem.`
- （341×）`Optional Criteria: The explanation should be clear and concise, avoiding unnecessary details that do not contribute to understanding the reasoning behind the choice.`

对照论文 RaR-PREDEFINED 的 4 条固定 rubric，例如 *"The response is concise and to the point, avoiding unnecessary verbosity or repetition."* —— 形态完全一致。

> **这是本次取证最有力的论证：论文自己证明了纯通用 rubric 会把 HealthBench 从 29.7 砸到 12.5（比不训练的 instruct 模型还低 10 分），但它发布的 instance-specific 数据里，仍有 17.7%（science）的条目本质上就是那种通用 rubric。**

---

## 5. 【发现 3】句内冗余：**假设被证伪**，不是主要问题

我们原本预期同一题内的 criteria 会大量语义重叠。**数据不支持这个判断。**

| 指标（8,000 题抽样） | Science | Medicine |
|---|---|---|
| 每题最大两两余弦（平均） | 0.417 | 0.536 |
| 每题平均两两余弦 | 0.139 | 0.137 |
| 每题最大 Jaccard（平均） | 0.195 | 0.288 |
| 朴素口径：存在 cos≥0.5 对的题 | 22.05% | 57.81% |
| **锚点校正后：存在真冗余对（cos≥0.5 且数字/符号/实体一致）的题** | **0.82%** | **7.90%** |
| 锚点校正后：真冗余对占全部对的比例 | 0.03% | 0.23% |
| 每题平均真冗余对数 | 0.008 | 0.097 |

朴素口径和校正口径差了一个数量级，原因见 §1.3。被误判的典型例子：

> A: `Essential Criteria: Specifies a follow-up schedule of every 3 months for the first 2 years.`
> B: `Essential Criteria: Specifies a follow-up schedule of every 6 months for the next 2 years.`
> （朴素 cos = 1.0，但这是两个不同的事实）

**结论：单纯的"重复造轮子"不是这批 rubric 的主要缺陷，我们不应把它当卖点。** 但下面这个变种是真问题。

### 5.1 真正的冗余形态：Pitfall 镜像正面条目

同一题里，一条 Pitfall 只是把某条 Essential/Important **取反重述**，同一个事实被计分两次（一次正权重、一次负权重）。

| 极性剥离后 Pitfall 与正面条目的最大余弦 | Science | Medicine |
|---|---|---|
| ≥ 0.6 | 0.95% | **16.21%** |
| ≥ 0.7 | 0.16% | 8.18% |
| ≥ 0.8 | 0.01% | 3.12% |
| ≥ 0.9 | 0.00% | 0.78% |

真实例证（medicine，全部为同一题内）：

- `rar_medicine:6222`
  > Essential (w=5)：`Identifies PKD1 as the gene not associated with Autosomal Dominant Tubulointerstitial Kidney Disease.`
  > Pitfall (w=−1)：`Does not mention PKD1 as the gene not associated with Autosomal Dominant Tubulointerstitial Kidney Disease.`
- `rar_medicine:17658`
  > Essential (w=5)：`Identifies Taenia solium as the causative agent of the disease.`
  > Pitfall (w=−1)：`Does not mention Taenia solium as the causative agent of the disease.`

在 Eq. 1 下，这一个事实同时贡献 +5 和 −1，且两条的判定必然反相关 —— 相当于对同一个判断做了带偏置的双重计数。

---

## 6. 【发现 4】可验证性：主观词与风格条目吃掉大量 reward 质量

| 指标 | Science | Medicine |
|---|---|---|
| 含主观词的 criteria | **34.16%** | 18.13% |
| 风格/表述类 criteria | 14.57% | 13.30% |
| **主观 或 风格（合计）** | **39.60%** | 23.64% |
| 断言"正确"但不说正确是什么的（非 self-contained） | 2.04% | 0.69% |
| **按 Eq. 1 权重计，落在主观/风格条目上的 reward 质量** | **40.18%** | **21.25%** |
| **按 Eq. 1 权重计，落在通用条目上的 reward 质量** | **24.56%** | 15.20% |

**按类别拆分的主观词率：**

| 类别 | Science | Medicine |
|---|---|---|
| Essential | 38.45% | 22.94% |
| Important | 24.47% | 7.75% |
| **Optional** | **59.14%** | 33.96% |
| Pitfall | 6.97% | 5.60% |

最高频主观词：science 是 `clearly`(26,144)、`clear`(12,103)、`concise`(8,852)、`appropriate`(4,842)；medicine 是 `concise`(14,908)、`clear`(9,216)、`clearly`(6,613)。

真实例证（这些在 RL 里极易被 reward hack）：

- (w=3) `Optional Criteria: The answer should be concise, avoiding unnecessary verbiage while still providing a complete and logical derivation of the expressions and limits.`
- (w=1) `Optional Criteria: Remains concise and avoids unnecessary detail while conveying all necessary information.`
- (w=1) `Optional Criteria: Conveys Empathy — Expresses understanding of the mother's situation and the complexity of the diagnosis.`（`rar_medicine:32`，一道"最可能的诊断是什么"的题）

**为什么这危险**：这些条目对 judge 来说几乎总是"满足"（简洁、清晰是默认状态），于是它们在 GRPO 的 group 内部**方差接近 0**，对 advantage 没有贡献，纯粹稀释了真正有区分度的事实性条目的梯度权重。更糟的是，当 policy 学会了"写短一点、加小标题"就能稳拿这 40% 的 reward 质量时，就出现了经典的风格性 reward hacking。

**注意 science 上 Essential 类的主观词率高达 38.45%** —— 这不是"Optional 位置放风格条目"那么简单，而是连关键事实条目都被主观修饰语污染了。例如：

> (w=5) `Essential Criteria: the response must clearly state the final answer as (a) and (d), ensuring that the answer choices are properly referenced.`

这里 "clearly" 和 "properly referenced" 让一个本来完全可验证的条目（答案是不是 (a) 和 (d)）变成了需要主观判断的条目。

### 6.1 modal 动词与类别的一致性

两个域的写作风格差异巨大（也印证了 science=o3-mini / medicine=GPT-4o 的说法）：

| | Science | Medicine |
|---|---|---|
| 含 `must` | 28.61% | 0.10% |
| 含 `should` | 30.72% | 0.60% |
| 含 `may/might/could` | 7.44% | 2.13% |

Science 用完整句（"The response must ..."），medicine 用祈使短语（"Identifies X"）。两个域的 criterion **语法形态完全不同**，同一个 judge prompt 要同时处理这两种，判定一致性存疑。类别与 modal 的冲突率本身很低（Essential 用 may/might/could 只有 0.39% / 0.45%），所以这不是一个大问题，但**跨域风格不统一**是复现时要注意的。

---

## 7. 【发现 5】Reference 依赖与泄漏

### 7.1 与 reference answer 的 n-gram 重叠

| 指标 | Science | Medicine |
|---|---|---|
| 平均 trigram 重叠率 | 0.017 | **0.109** |
| 平均 4-gram 重叠率 | 0.007 | 0.059 |
| 与 question 的 4-gram 重叠 | 0.010 | 0.016 |
| trigram 重叠 ≥30% 的 criteria | 1.36% | **15.11%** |
| trigram 重叠 ≥50% 的 criteria | 0.43% | **7.41%** |
| trigram 重叠为 0 的 criteria | 90.36% | 66.78% |

**两个域是两种失败方向：**

- **Medicine 抄 reference**：7.41% 的 criteria 有一半以上的 trigram 直接来自 reference answer，违反了 prompt 里 *"Do not copy large blocks of the question or reference_answer into the text"* 的明确要求。这些 criterion 本质上是把参考答案切碎成 checklist，rubric 相对于 reference 没有增加任何信息 —— 那么 RaR-EXPLICIT 相比 REFERENCE-LIKERT 的增益从哪来就成了疑问。
- **Science 几乎不碰 reference**：90.36% 的 criteria 与 reference 零 trigram 重叠。结合 §7.3 的 reference 长度分布，这不是"独立思考"，而是 **reference 本来就没什么内容可抄**。

### 7.2 最终答案数值泄漏

统计"出现在 reference 但不出现在 question 里的数字"（即答案值）被 criterion 直接写出的情况：

| | Science | Medicine |
|---|---|---|
| 泄漏答案数值的 criteria | **6.82%** | 2.97% |
| 至少有一条泄漏的题目 | **28.49%** | 12.15% |

例（`rar_science:1`，reference = `10^{19}`）：

> (w=5) `Essential Criteria: The answer must clearly state the final result as approximately 10^19 visible photons emitted per second.`

**这本身未必是坏事**（对客观题来说，这就是一条可验证的答案检查条目，等价于 RLVR 的 exact match）。但它意味着：对这 28.5% 的题目，rubric 的核心价值就是一条答案匹配，剩下 6 条多是过程和风格描述 —— **用 rubric 而不是 RLVR 的理由在这类题上很弱**。

### 7.3 Reference 越短，rubric 越通用（接地能力随 reference 塌陷）

| reference 词数 | Science 通用率 | Science 条数 | Medicine 通用率 | Medicine 条数 |
|---|---|---|---|---|
| 1–5 | **37.00%** | 35,606 | **36.50%** | 14,631 |
| 6–20 | 31.71% | 35,373 | **38.27%** | 10,486 |
| 21–60 | 24.25% | 56,904 | 22.55% | 35,064 |
| 61–150 | 26.99% | 27,698 | 20.20% | 131,493 |
| >150 | 24.34% | 16,436 | **14.58%** | 22,984 |

Science 有 **2,180 题（9.5%）的 reference 完全为空**，25% 的题 reference ≤6 词。论文 §5 自己说 *"rubric quality is impacted by reference quality"*，Table 1 也证明了无 reference 会掉 3.9 分（35.9 → 32.0）。**我们的数据显示：RaR-Science 里有相当一部分题目实际上处于"接近无 reference"的状态**，但生成器依然照常产出 7 条 criteria。

真实例证（`rar_science:0`，reference 仅 `-44.9 kcal`，却生成了 7 条 criteria，其中 5 条权重都是 5）：

> - (w=5) Essential: The response should mention converting the given temperatures from Celsius to Kelvin ...
> - (w=5) **Important**: The response must detail applying the least squares regression technique ...
> - (w=5) Essential: The response should clearly state that the slope of the regression line equals -ΔH/R ...
> - (w=5) **Important**: The response must include a step converting energy units appropriately ...
> - (w=5) Essential: The response should provide a clear and precise final calculation of the enthalpy, approximating -44.9 kcal ...
> - (w=3) Optional: It is beneficial if the response outlines intermediate steps ...
> - (w=−1) Pitfall: The response should not omit mentioning that non-ideal effects ... are assumed negligible.

这一例同时展示了三个问题：**7 条固定预算**、**5 条同为权重 5（权重退化）**、**Important 类别却给了权重 5（违反 category 序）**。

---

## 8. 【发现 6】覆盖率缺口

### 8.1 多小问题目

Science 有 **677 题（2.95%）** 是真正的多小问题目（另有 2,146 题即 9.36% 是选择题，已排除）。Medicine 几乎没有（93 题全是选择题形态，多小问 0 题）。

| Science | 单小问 | 2 小问 | 3 小问 | 4 小问 |
|---|---|---|---|---|
| 题数 | 22,240 | 391 | 185 | 78 |
| 平均 rubric 条数 | **7.49** | 7.92 | 7.92 | **8.29** |
| **平均每小问条数** | 7.49 | **3.96** | **2.64** | **2.07** |

**从 1 个小问到 4 个小问，rubric 预算只增加了 0.8 条**（回归斜率 = 每多一个小问 +0.27 条）。每小问的 criteria 数从 7.49 掉到 2.07。

| 语义覆盖指标（677 道多小问题目，1,777 个小问） | 值 |
|---|---|
| 小问与本题任一 criterion 的最佳余弦（平均） | 0.358 |
| **孤儿小问率**（最佳余弦 < 0.08，即无任何 criterion 与之对应） | **4.45%** |
| 至少有一个孤儿小问的多小问题目 | **8.71%** |
| **有任一 criterion 明确点名某个小问标签（如 "part (a)"）的题目** | **33.97%** |

> 最后一行是关键：**66% 的多小问题目里，没有任何一条 criterion 说明自己是在评哪个小问。** 在 RaR-EXPLICIT 的逐条判定模式下，judge 拿到 "The response must correctly apply the formula μ = N I A" 时无法知道这属于 (a)，也无法知道 (b)(c) 是否被单独评估过。这是 self-contained 原则的实质性违反。

### 8.2 选择题上的"填充"

Science 有 2,146 题（9.36%）是选择题，答案唯一且完全可验证。这些题平均拿到 **7.44 条 criteria**，其中只有 **42.19%** 提到了答案选项，**平均每题有 4.3 条与答案无关的条目**。

> 对这近一成的数据，RLVR 的 exact match 就够了；多出来的 4.3 条主要是过程描述和风格要求，只会引入噪声。

---

## 9. 【发现 7】权重几乎不携带信息

### 9.1 权重是 category 的确定性函数

| | Science | Medicine |
|---|---|---|
| 权重的边际熵 | 2.246 bit | 2.698 bit |
| **给定 category 后的条件熵** | **0.723 bit** | **0.775 bit** |
| category 与 weight 的互信息 | 1.523 bit | 1.923 bit |
| **P(weight=5 \| Essential)** | **97.99%** | **100.00%** |
| P(weight=4 \| Important) | 74.73% | 50.53% |
| P(weight=2 \| Optional) | 31.6% | 60.16% |

**Medicine 的 Essential 权重 100% 都是 5，Optional 只取 {1,2}，Important 只取 {3,4}** —— 数值权重完全由 category 决定，没有任何额外信息。所谓"细粒度数值权重"实际上只是 category 的四值编码。

Science 略好（Essential 有 2% 不是 5），但代价是**破坏了类别序**：

| | Science | Medicine |
|---|---|---|
| 同一 rubric 内"低类别权重 ≥ 高类别权重"的有序对占比 | **8.64%** | 0.00% |
| **至少存在一处类别序违反的题目** | **41.30%** | 0.00% |

Science 有 66.77% 的 Optional 条目权重为 3，而 5.9% 的 Important 条目权重也是 3、还有 8 条 Important 权重为 2 —— 于是 **41.3% 的 science 题目里，某条 Optional 的权重不低于某条 Important**。这不是"违反 prompt"（science prompt 本来就没给数值区间），而是生成器在没有锚点时无法自洽地维持"Essential > Important > Optional"这个它自己声明的序（见 `00_paper_notes.md` §2.3）。Medicine 因为 prompt 给了明确区间，违反率为 0。

### 9.2 加权和 ≈ 简单平均

用 4,000 题 × 8 次随机 0/1 判定（32,000 次抽样）模拟 Eq. 1：

| | Science | Medicine |
|---|---|---|
| 数据权重加权 vs 等权平均，Pearson | 0.856 | 0.756 |
| 数据权重加权 vs 等权平均，Spearman | 0.845 | 0.746 |
| **论文 RaR-EXPLICIT 的重映射权重 vs 等权平均，Pearson** | **0.943** | **0.932** |
| 平均绝对差 | 0.092 | 0.112 |

**论文实际使用的权重方案（Essential 1.0 / Important 0.7 / Optional 0.3 / Pitfall 0.9）与"直接取平均"的相关性高达 0.93–0.94。** 整套权重体系在 reward 层面几乎是恒等变换。

> 这与论文自己的消融 **完全一致**：Table 2 显示 "No Categorical Labels" 得 38.8%，比 "All Rubrics" 的 37.2% **更高**。权重不但不携带信息，还略有害。论文用一句 "minimal performance differences" 带过了这个结果。

### 9.3 ⚠️ Pitfall 符号问题：两个 domain 的约定是**相反的**

这是本次取证发现的最严重的**可直接修复的 bug**。

Pitfall criterion 有两种写法，含义正好相反：

- **"失败式"（failure-describing）**：`Does not mention X` —— 满足该句 = 回答**漏了** X = 回答**差**
- **"避免式"（avoidance-describing）**：`Avoids X` / `should not X` —— 满足该句 = 回答**避开了**错误 = 回答**好**

实测分布：

| | Science | Medicine |
|---|---|---|
| **失败式** | 6.77% | **88.11%** |
| **避免式** | **80.33%** | 2.90% |
| 未能分类 | 12.90% | 8.99% |
| 权重为负的比例 | 100% | 100% |
| 同一题内两种写法混用的题目 | 0.18% | 1.77% |

**两个公开数据集用了完全相反的极性约定，却都配了 −1/−2 的负权重。** 后果：

| | 用数据里的 −1/−2 | 用论文 RaR-EXPLICIT 的 +0.9 |
|---|---|---|
| **Science**（避免式为主）：回答成功避开错误 → c=1 | 得 **−1**，**做对了反被扣分** ❌ | 得 +0.9 ✓ |
| **Medicine**（失败式为主）：回答犯了错误 → c=1 | 得 −1 ✓ | 得 **+0.9**，**犯错反被奖励** ❌ |

**无论选哪一种符号约定，必有一个 domain 的 13–14% criteria 被训反。**

这还提供了对论文 Table 2 中 "No Pitfall Criteria: 37.2% = All Rubrics: 37.2%" 的一个更简单的解释。论文归因为 *"synthetically generating effective pitfall criteria is inherently difficult"*，但我们的数据显示 —— **Pitfall 内容其实写得挺具体的，问题是符号约定错乱，正负相消，净效果归零。**

更棘手的是，即使在 medicine 内部，同样的 `Does not ...` 句式也可能有相反含义，取决于 X 本身是好事还是坏事：

- `Pitfall Criteria: Does not mention exploring with the patient any potential alternatives to permanent surgical solutions.` (w=−1)
  → X = "和患者讨论替代方案"，是**好事**；不提 = 差 → 失败式，符号正确
- `Pitfall Criteria: Does not suggest delaying fluids or antibiotics despite identification of septic shock symptoms.` (w=−2)
  → X = "延迟补液和抗生素"，是**坏事**；不建议延迟 = 好 → 避免式，符号**反了**

**这种歧义无法用正则消解，也很难指望 judge 稳定判对。** 它要求生成阶段就强制单一极性。

### 9.4 负权重让 Eq. 1 的归一化失效

| | Science | Medicine |
|---|---|---|
| 含至少一条负权重的题 | **91.80%** | **95.55%** |
| 负权重 criteria 占比 | 13.05% | 14.16% |
| 每题 Σw 均值 | 25.28 | 24.70 |
| 每题 Σ\|w\| 均值 | 27.66 | 28.83 |

Eq. 1 的分母 $\sum_j w_j$ 把负权重也加进去了，使分母比 $\sum_j |w_j|$ 小 9%（science）/ 14%（medicine），而且缩小幅度依题而异。这让"归一化使 reward 跨 prompt 可比"的设计意图打了折扣。（好消息：没有任何一题的 Σw ≤ 0，所以不会出现除零或符号翻转。）

---

## 10. 【发现 8】数据卫生：训练/测试泄漏

| | Science | Medicine |
|---|---|---|
| 重复的 question 行数 | 2,096（**9.15%**） | 574（2.56%） |
| 同一 question 跨多个 split 出现 | 509 组 | 119 组 |
| train 与 test 共有的 question | 261 | 69 |
| **test 集中问题也在 train 里出现过的比例** | **12.30%** | 3.57% |

RaR-Science 的 test split 有 **12.3% 的题目在 train 里出现过**。任何在这个 split 上报告的结果都被污染了。（论文主实验不在这些 split 上评测，用的是 HealthBench / GPQA，所以论文结论不受影响；但**我们如果用 val/test 做 rubric 质量评估，必须先去重**。）

---

## 11. 结论：直接合成 rubric 的失效模式清单

每条给出：**现象 → 量化 → agentic 流程应如何针对性修复 → 建议的监控指标**。

---

### **F1 — 预算固化：rubric 条数不随题目复杂度变化**

- **量化**：Science 64.6% 的题恰好 7 条，88.9% 是 7 或 8 条，CV=0.107，与题目长度相关性仅 r=0.19。多小问题目从 1 问到 4 问只多 0.8 条（斜率 +0.27 条/小问）。prompt 允许 7–20，实际只用到 5–12。
- **为什么致命**：这是下游一半问题的根因。预算钉死 → 简单题被塞通用条目凑数（F3）、复杂题欠覆盖（F8）。
- **agentic 修复**：把"决定条数"变成一个**显式的前置步骤**而非让生成器隐式决定。先做 **claim decomposition**：从 question + reference 中抽出可判定的原子断言/子任务清单（含小问、每步中间结论、单位/量纲、边界条件），条数由这个清单的长度**导出**，而不是由 prompt 里的 "7–20" 锚定。
- **监控指标**：`items_per_question` 的 CV（目标 > 0.35）、与题目复杂度代理（小问数、reference 长度、解题步骤数）的相关系数（目标 r > 0.5）、条数分布的众数占比（目标 < 25%）。

---

### **F2 — 样板化：prompt 里的示例被当成模板批量复制**

- **量化**：Medicine 单个 6 词开头 `Remains concise and avoids unnecessary detail` 占全部 criteria 的 **3.19%（6,851 次）**，且逐字来自生成 prompt 的示例；3.50% 的 description 逐字出现在**不同题目**上（1,240 个字符串）；15.12% 的条目共享同一"高频词骨架"。Science 42.27% 的条目标题被复用 ≥10 次，top-50 标题覆盖 14.82% 的条目。
- **agentic 修复**：(a) prompt 里**不要给可直接照抄的完整示例句**，只给结构说明；(b) 生成后加一道**去样板 pass**：对照一个跨题维护的"高频短语黑名单"（可从上一批产出中自动挖掘），命中则强制重写或删除；(c) 标题由内容自动派生（如"被检查的断言"的短摘要），而不是自由生成。
- **监控指标**：`top1_opener_share`（目标 < 0.5%）、`exact_duplicate_across_different_questions_pct`（目标 ≈ 0）、`criteria_whose_title_reused_ge_10x_pct`（目标 < 10%）。

---

### **F3 — 通用条目：四分之一的 reward 质量押在与本题无关的条目上**

- **量化**：通用条目率 **28.87%（science）/ 21.97%（medicine）**（置换对照 96.4%/98.5%，判别力 67–77 个百分点）。按 Eq. 1 权重计，**24.56%（science）/ 15.20%（medicine）的 reward 质量落在通用条目上**。"可回收"条目（通用 + 批量复用标题）占 **17.67% / 11.60%**，36.86% 的 science 题目有 ≥2 条。归属检索 top-1 只有 **56.82%（science）**，23.66% 的条目被 ≥10 个随机干扰题击败。
- **问题极度集中**：Optional 类通用率 **52.32%/43.93%**，Pitfall **37.22%/28.55%**，而 Essential 只有 14.93%/8.73%。
- **为什么致命**：论文自己证明纯通用 rubric（RaR-PREDEFINED）把 HealthBench 从 29.7 砸到 **12.5**。这批数据里近两成条目就是那个形态。
- **agentic 修复**：(a) 强制每条 criterion 引用本题的**至少一个具体锚点**（数值、公式、实体、单位、选项标签），生成时把锚点列表作为约束传入；(b) 加一道 **grounding verifier**：自动检测无锚点条目并要求重写或直接删除，而不是保留占位；(c) **不要为了凑够 N 条而生成 Optional 条目** —— 允许 rubric 只有 4 条高质量条目。
- **监控指标**：`generic_criterion_rate`（目标 < 5%）、`recyclable_criterion_rate`（目标 < 2%）、`attribution_top1`（目标 > 90% @1000 干扰项）、**`pct_of_weight_on_generic_criteria`（目标 < 5%，这是最贴近 reward 的指标）**。

---

### **F4 — 主观/风格条目稀释梯度，且是 reward hacking 的入口**

- **量化**：39.60%（science）/ 23.64%（medicine）的条目含主观词或属风格类；按权重计占 **40.18% / 21.25%** 的 reward 质量。Science 连 Essential 类都有 38.45% 含主观词。最高频词：`clearly` 26,144 次、`concise` 14,908 次。
- **为什么致命**：这类条目对 judge 几乎恒为"满足"，在 GRPO 的 group 内方差趋近 0，不贡献 advantage，只稀释真正有区分度的事实条目；同时给 policy 提供了"写短一点、加小标题"就能拿分的捷径。
- **agentic 修复**：(a) 维护一份**主观词黑名单**，生成时禁用，验证时拦截；(b) 要求每条 criterion 必须是**二元可判定**的 —— 加一道 "两个独立 judge 能否对同一 response 给出一致判定" 的自检；(c) 风格维度如果确实要保留，**从加权和里剥离**成独立的辅助 reward，不要混进事实性 checklist；(d) 把 `clearly state X` 改写成 `states X`（去掉主观修饰语，保留可验证内核）。
- **监控指标**：`subjective_or_style_rate`（目标 < 10%）、`pct_of_weight_on_subjective_criteria`（目标 < 10%）、以及一个**判定一致性指标**：同一 (criterion, response) 用两个 judge / 两个 temperature 判定的 agreement（目标 > 0.9）。

---

### **F5 — 权重退化：整套权重体系是 category 的恒等变换**

- **量化**：给定 category 后权重的条件熵仅 0.72/0.78 bit；medicine 的 Essential **100%** 为 5，Optional 只取 {1,2}，Important 只取 {3,4}。论文 RaR-EXPLICIT 的重映射权重与**等权平均**的 Pearson 相关性 **0.943 / 0.932**。论文自己的消融也显示去掉 category label 反而更好（38.8% vs 37.2%）。
- **附带 bug（仅 science）**：**41.30%** 的题目里存在 Optional 权重 ≥ Important 权重的情况（8.64% 的有序对违反），因为 science prompt 没给数值锚点。
- **agentic 修复**：两条路，二选一并明确论证：
  - **(a) 放弃权重**（论文数据支持这条）：所有 criteria 等权，把精力放在"生成正确的条目集合"上。
  - **(b) 让权重真正携带信息**：权重不由生成器自由发挥，而是从**可观测量**导出 —— 例如"该断言在参考解中出现的位置深度""该断言错误时后续步骤是否全错（因果依赖度）""在一批 rollout 上该条目的判定方差"。后者尤其有价值：**方差为 0 的条目权重应为 0**。
- **监控指标**：`H(weight | category)`（目标 > 1.5 bit）、权重与等权平均的 Spearman（目标 < 0.85）、`weight_order_violation_rate`（目标 = 0）、以及在真实 rollout 上的**每条目判定方差分布**（目标：无大量零方差条目）。

---

### **F6 — Pitfall 符号错乱：两个 domain 的极性约定相反**

- **量化**：Science **80.33%** 是"避免式"（满足=好），Medicine **88.11%** 是"失败式"（满足=坏），两者都配 −1/−2 权重。91.8%/95.6% 的题含至少一条负权重条目。无论采用哪一种符号约定，必有一个 domain 的 **13–14%** 条目被训反。
- **这解释了论文的 Table 2**：去掉 Pitfall 完全不影响性能（37.2% = 37.2%），不是因为"合成 pitfall 很难"，而是**符号错乱导致正负相消**。
- **额外难点**：即使在同一 domain 内，`Does not mention X` 的含义仍取决于 X 本身是好事还是坏事（见 §9.3 的两个例子），正则和 judge 都难以稳定消解。
- **agentic 修复**：(a) **强制单一极性** —— 所有 criterion（包括 pitfall）统一写成"满足 = 好"的正向形式（`Avoids recommending X` 而非 `Recommends X`），权重一律为正；(b) 生成后加一道 **polarity linter**，检出失败式表述并自动改写；(c) 如果确实需要负权重的惩罚项，**在 schema 里显式加一个 `polarity` 字段**，不要靠自然语言句式推断。
- **监控指标**：`polarity_consistency_rate`（目标 = 100% 单一极性）、`negative_weight_criteria_pct`（目标 = 0，惩罚通过显式字段而非负权重表达）、以及一个端到端 sanity check：**人为构造一个"完美回答"和一个"典型错误回答"，验证 Eq. 1 给前者的分严格高于后者**（当前数据在一个 domain 上必然不满足）。

---

### **F7 — 同一事实被 Essential 和 Pitfall 双重计分**

- **量化**：Medicine **16.21%** 的题存在 Pitfall 与正面条目在极性剥离后余弦 ≥0.6 的情况，8.18% ≥0.7，3.12% ≥0.8。Science 可忽略（0.95% / 0.16%）。
- **为什么致命**：同一个判断同时贡献 +5 和 −1，且两条判定必然反相关，等于对该事实做了带偏置的双重计数；与 F6 叠加时符号完全不可控。
- **agentic 修复**：生成完成后做一次 **rubric 级别的去重/合并 pass**：对极性归一后的条目做互相似度检查，把镜像对合并为单条（保留正向表述 + 较高权重）。这个 pass 应该看到整份 rubric，而不是逐条生成时的局部视角 —— **这正是 agentic 流程相对单次生成的天然优势**。
- **监控指标**：`pitfall_mirror_rate @cos≥0.6`（目标 < 1%）、极性归一后的**句内最大余弦**分布。
- **注意**：与之相对，**一般意义上的句内语义冗余不是问题**（锚点校正后只有 0.82%/7.90% 的题存在真冗余对），不要在这上面浪费预算或夸大宣传。

---

### **F8 — 多小问覆盖不足 + 条目无法定位到小问**

- **量化**：Science 677 题（2.95%）为真多小问题目。单小问 7.49 条 vs 4 小问 8.29 条，**每小问的条目数从 7.49 掉到 2.07**。孤儿小问率 4.45%，8.71% 的多小问题目至少有一个小问完全没有对应条目。**只有 33.97% 的多小问题目里有任一条目明确点名小问标签** —— 即 66% 的情况下 judge 无法知道某条 criterion 在评哪个小问，直接违反论文自己的 self-contained 原则。
- **附带**：9.36% 的 science 题目是选择题（答案唯一可验证），却仍拿到 7.44 条 criteria，其中只有 42.19% 提到答案选项，平均每题 4.3 条与答案无关。
- **agentic 修复**：(a) 前置的 **question decomposition** 步骤显式产出子任务列表，并作为结构化字段传给后续步骤；(b) 每条 criterion 携带一个 `covers` 字段指向它评的子任务，最后做**覆盖矩阵检查** —— 任何未被覆盖的子任务触发补充生成；(c) 对选择题走**独立的轻量路径**（一条答案检查 + 少量推理过程条目），不要套用同一套 7–20 条模板。
- **监控指标**：`orphan_subpart_rate`（目标 = 0）、`subpart_label_addressability`（目标 = 100%）、`items_per_subpart` 的方差、以及一个明确的 `question_type` 路由准确率。

---

### **F9 — 接地能力随 reference 质量塌陷，但生成器不感知**

- **量化**：Science **9.5% 的题 reference 完全为空**，25% 的题 ≤6 词。reference 1–5 词时通用条目率 **37.00%**（vs 21–60 词时 24.25%）；medicine 6–20 词时 **38.27%**（vs >150 词时 14.58%）。论文自己承认 *"rubric quality is impacted by reference quality"*，Table 1 显示无 reference 掉 3.9 分。
- **反向问题（medicine）**：reference 足够长时又变成照抄 —— **7.41%** 的条目有 ≥50% 的 trigram 直接来自 reference，违反 prompt 的明文禁令。
- **agentic 修复**：这是 agentic 方案**最有说服力的差异化点** —— 单次生成拿到什么 reference 就用什么，agent 可以：(a) 先**评估 reference 的信息量**，不足时**自己解题**（调用推理/工具/检索）来补足接地信息，而不是拿空 reference 硬编 7 条；(b) reference 充足时，要求 criterion 是对 reference 的**判定式改写**而非**片段复制**（加 n-gram 重叠上限的 lint）。
- **监控指标**：分 reference 长度桶的 `generic_rate`（目标：各桶之间差异 < 5 个百分点，即质量不依赖 reference 长度）、`trigram_overlap_with_reference`（目标：< 30% 的条目超过 0.3）。

---

### **F10 — Schema 与数据卫生问题**

- **量化**：Science 837 条（0.49%）缺 category 前缀；Medicine 有 `Esssential`/`Importance`/`Option`/`Universal`/`Mandatory` 等非法前缀；Science **77.59%** 的权重落在推荐区间（medicine 99.99%）。数据层面：Science **9.15%** 的题目重复，**test 集 12.30% 的问题在 train 中出现过**（medicine 3.57%）。
- **agentic 修复**：(a) 输出走**结构化 schema 校验 + 自动重试**（category 用枚举而非自然语言前缀，polarity/weight 用独立字段）；(b) 建库前做 **question 级别去重和跨 split 隔离**，这是我们做任何 rubric 质量评估的前置条件。
- **监控指标**：`schema_violation_rate`（目标 = 0）、`cross_split_question_overlap`（目标 = 0）。

---

### 11.1 优先级建议

给 harness 团队的排序（按"影响 × 可修复性"）：

| 优先级 | 失效模式 | 理由 |
|---|---|---|
| **P0** | **F6**（Pitfall 符号） | 唯一一个**确定性 bug**，修复成本最低，且能解释论文 Table 2 的空结果 |
| **P0** | **F3**（通用条目占 24.6% reward 质量） | 影响最大，论文自己证明了 generic rubric 的破坏力（29.7 → 12.5） |
| **P0** | **F1**（条数固化） | 是 F3/F8 的共同根因，修它能连带改善两条 |
| **P1** | **F4**（主观/风格占 40.2% reward 质量） | 影响大，但需要设计"判定一致性"指标才能可靠度量 |
| **P1** | **F9**（reference 依赖） | agentic 方案最有说服力的差异化点（agent 能自己解题补足接地） |
| **P2** | **F8**（多小问覆盖） | 影响的题目占比不高（science 2.95%），但对"agentic 分解"的叙事很有说服力 |
| **P2** | **F5**（权重退化） | 最简单的修法是**直接放弃权重**（有论文消融背书） |
| **P3** | **F2 / F7 / F10** | 工程性问题，加 lint pass 即可 |

### 11.2 不要过度宣称的两点

1. **句内语义冗余不是主要问题**（锚点校正后 science 0.82% / medicine 7.90% 的题存在真冗余对）。真正的冗余只有 Pitfall 镜像那一种（F7）。
2. **格式合规率其实很高**（>95%）。单次生成的失败不在"不听指令"，而在"指令本身没有约束到关键的结构性属性"（条数是否自适应、条目是否接地、极性是否统一）。**因此，仅仅把 prompt 写得更长、约束更多，大概率无效** —— 这也是为什么需要 agentic 流程里的**验证与修订闭环**，而不只是更好的 prompt。

---

## 附：产物索引

| 文件 | 内容 |
|---|---|
| `analysis/out/basic_stats.json` | 全部基础统计、prompt 合规、权重信息量、可验证性、重复统计 |
| `analysis/out/basic_summary.csv` | 两个域的一行式摘要 |
| `analysis/out/similarity_stats.json` | 标题/模板样板化、跨题近重复、句内冗余、Pitfall 镜像（含例证） |
| `analysis/out/top_titles.csv` | 两个域各 top-50 高频标题及占比 |
| `analysis/out/within_question_redundancy_*.csv` | 每题的两两相似度统计（逐题明细） |
| `analysis/out/grounding_stats.json` | 锚点接地、归属检索、reference 耦合与泄漏、子问题覆盖 |
| `analysis/out/composite_stats.json` | 可回收率、reward 质量分布、Pitfall 极性、权重序违规、选择题填充 |
| `analysis/out/examples.md` | 每类失效模式的逐字例证（含完整 rubric 样例） |
| `analysis/out/paper_text.txt` | 论文全文（带页码标记），供回溯引用 |
| `analysis/out/criteria_*.parquet` | 展平后的 criterion 级数据（含解析出的 category / body），供后续脚本复用 |
