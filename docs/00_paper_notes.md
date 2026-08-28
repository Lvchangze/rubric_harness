# RaR (Rubrics as Rewards) 论文精读笔记

> 论文：*Rubrics as Rewards: Reinforcement Learning Beyond Verifiable Domains*
> Gunjal, Wang, Lau, Nath, He, Liu, Hendryx (Scale AI), arXiv:2507.17746v2, 2025-10-03
> 本地 PDF：`rubric_as_reward.pdf`（20 页）；抽取的纯文本：`analysis/out/paper_text.txt`（带 `===== PAGE n =====` 页码标记，便于回溯引用）
>
> 本文档只记录论文说了什么。数据实际长什么样、与论文描述的出入，见 `docs/01_data_forensics.md`。

---

## 1. Pipeline 全貌

RaR 是一个两阶段方法：**(i) 离线合成 instance-specific rubric → (ii) 用 rubric 驱动 LLM judge 打分，作为 GRPO 的 reward**。

```
question + reference_answer
        │
        │  单次 LLM 调用（o3-mini / GPT-4o），一个长 prompt，一次性输出 JSON 数组
        ▼
   rubric = [{title, description, weight}, ...]   (7-20 条)
        │
        │  训练时固定不变，对同一 prompt 的所有 rollout 复用
        ▼
policy rollout ─► LLM judge (gpt-4o-mini) ─► scalar reward ─► GRPO
```

### 1.1 关键事实（对我们最重要的几条）

| 问题 | 论文的答案 | 出处 |
|---|---|---|
| rubric 怎么造的？ | **单次 prompt、一次性生成**（single-pass），无检索、无多轮、无验证、无修订 | §3.2 |
| 用什么模型？ | RaR-Medicine 用 **GPT-4o**；RaR-Science 用 **o3-mini** | §4.1 |
| 用了 reference answer 吗？ | **用了**。"conditioning generation on reference answers from the underlying datasets to approximate expert grounding" | §3.2 |
| 每题几条？ | prompt 里要求 "7–20 rubric items **based on the complexity of the question**" | Appendix A.9 前的 prompt box |
| 有人工参与吗？ | **没有**。"Given the scarcity of human-annotated rubric datasets in these domains, we use LLMs to generate instance-specific rubrics ... at scale, enabling the study of structured rewards without costly human annotation" | §3.2 |
| 有没有质量检查/过滤？ | **论文完全没有提到任何生成后的过滤、验证或迭代步骤** | — |

> **这就是我们要改进的靶心**：整条 rubric 生成链路是 "一个 prompt 打天下"，没有任何 verification / decomposition / revision 环节，且质量完全没有被内在地度量过（论文所有评估都是 downstream 的 HealthBench 分数）。

### 1.2 论文自己提出的四条 rubric 设计原则（desiderata，§3.1）

这四条是论文写在 Figure 1 里的卖点，也是我们做 evaluation 时可以直接拿来当 metric 轴：

1. **Grounded in Expert Guidance** —— rubric 应反映领域专业知识，覆盖正确性所需的关键事实、推理步骤和结论。理想情况下来自人类专家或其高质量代理。
2. **Comprehensive Coverage** —— 应跨越多个质量维度：factual accuracy、logical coherence、completeness、style、safety。负面 criteria（pitfalls）用于识别高频/高风险错误。
3. **Criterion Importance** —— 不同维度重要性不同（事实正确性应压过风格清晰度）。通过权重体现优先级，可以是 categorical tag、显式数值或学习到的权重。
4. **Self-Contained Evaluation** —— 每条 rubric item 应可独立判定，**人类标注者或自动 judge 无需外部上下文或领域知识即可单独评估**。

> 注意第 4 条：论文自己承诺 criterion 要 self-contained。这条在数据里被违反得很厉害（见 forensics 文档 F4/F6）。

---

## 2. Rubric 生成 prompt 原文（逐字抄录）

这是我们 baseline 必须精确复现的东西。论文给了两个几乎相同但不完全一致的 prompt（medicine / science 各一个），分别在 PDF 第 18 页和第 19 页。以下为**逐字转录**（原文为英文，保留原样；`. . .` 与破折号等排版符号按原文）。

### 2.1 Medicine 域（PDF p.18，"Prompt for Synthetic Rubrics Generation: Medical Domain"）

```text
You are an expert rubric writer. Your job is to generate a self-contained set of evaluation criteria ("rubrics") for judging
how good a response is to a given question. Rubrics can cover aspects of a response such as, but not limited to, factual
correctness, ideal-response characteristics, style, completeness, helpfulness, harmlessness, patient-centeredness, depth of
reasoning, contextual relevance, and empathy. Each item must be self-contained – non expert readers should not need to
infer anything or consult external information. Begin each description with its category: "Essential Criteria: . . . ", "Important
Criteria: . . . ", "Optional Criteria: ...", or "Pitfall Criteria: Does not mention . . . ".

Inputs:
• question: The full question text.
• reference_answer: The ideal answer, including any specific facts, explanations, or advice.

Total items:
• Choose 7–20 rubric items based on the complexity of the question.

Each rubric item:
• title (2–4 words).
• description: One sentence starting with its category prefix that explicitly states exactly what to look for. For example:
  – Essential Criteria: Identifies non-contrast helical CT scan as the most sensitive modality for ureteric stones.
  – Pitfall Criteria: Does not mention identifying (B) as the correct answer.
  – Important Criteria: Explains that non-contrast helical CT detects stones of varying sizes and compositions.
  – Optional Criteria: States "The final answer is (B)" or similar answer choice formatting.
• weight: For Essential/Important/Optional, use 1–5 (5 = most important); for Pitfall, use –1 or –2.

Category guidance:
• Essential: Critical facts or safety checks; if missing, the response is invalid (weight 5).
• Important: Key reasoning, completeness, or clarity; strongly affects quality (weight 3–4).
• Optional: Helpful style or extra depth; nice to have but not deal-breaking (weight 1–2).
• Pitfall: Common mistakes or omissions specific to this prompt—identify things a respondent often forgets or misstates.
  Each Pitfall description must begin with "Pitfall Criteria: Does not mention . . . " or "Pitfall Criteria: Recommends . . . "
  and use weight –1 or –2.

To ensure self-contained guidance:
• When referring to answer choices, explicitly say "Identifies (A)", "Identifies (B)", etc., rather than vague phrasing.
• If the format requires a conclusion like "The final answer is (B)", include a rubric item such as:
  – Essential Criteria: Includes a clear statement "The final answer is (B)".
• If reasoning should precede the answer, include a rubric like:
  – Important Criteria: Presents the explanation before stating the final answer.
• If brevity is valued, include a rubric like:
  – Optional Criteria: Remains concise and avoids unnecessary detail.
• If the question context demands mention of specific findings, include that explicitly (e.g., "Essential Criteria: Mentions
  that CT does not require contrast").

Output: Provide a JSON array of rubric objects. Each object must contain exactly three keys—title, description, and weight.
Do not copy large blocks of the question or reference_answer into the text. Each description must begin with its category
prefix, and no extra keys are allowed.

Now, given the question and reference_answer, generate the rubric as described. The reference answer is an ideal response
but not necessarily exhaustive; use it only as guidance.
```

### 2.2 Science 域（PDF p.19，"Prompt for Synthetic Rubric Generation: Science Domain"）

```text
You are an expert rubric writer for science questions in the domains of Biology, Physics, and Chemistry. Your job is to
generate a self-contained set of evaluation criteria ("rubrics") for judging how good a response is to a given question in one
of these domains. Rubrics can cover aspects such as factual correctness, depth of reasoning, clarity, completeness, style,
helpfulness, and common pitfalls. Each rubric item must be fully self-contained so that non-expert readers need not consult
any external information.

Inputs:
• question: The full question text.
• reference_answer: The ideal answer, including any key facts or explanations.

Total items:
• Choose 7–20 rubric items based on question complexity.

Each rubric item must include exactly three keys:
1. title (2–4 words)
2. description: One sentence beginning with its category prefix, explicitly stating what to look for. For example:
   • Essential Criteria: States that in the described closed system, the total mechanical energy (kinetic plus potential)
     before the event equals the total mechanical energy after the event.
   • Important Criteria: Breaks down numerical energy values for each stage, demonstrating that initial kinetic
     energy plus initial potential energy equals final kinetic energy plus final potential energy.
   • Optional Criteria: Provides a concrete example, such as a pendulum converting between kinetic and potential
     energy, to illustrate how energy shifts within the system.
   • Pitfall Criteria: Does not mention that frictional or air-resistance losses are assumed negligible when applying
     conservation of mechanical energy.
3. weight: For Essential/Important/Optional, use 1–5 (5 = most important); for Pitfall, use –1 or –2.

Category guidance:
• Essential: Critical facts or safety checks; omission invalidates the response.
• Important: Key reasoning or completeness; strongly affects quality.
• Optional: Nice-to-have style or extra depth.
• Pitfall: Common mistakes or omissions; highlight things often missed.

Format notes:
• When referring to answer choices, explicitly say "Identifies (A)", "Identifies (B)", etc.
• If a clear conclusion is required (e.g. "The final answer is (B)"), include an Essential Criteria for it.
• If reasoning should precede the final answer, include an Important Criteria to that effect.
• If brevity is valued, include an Optional Criteria about conciseness.

Output: Provide a JSON array of rubric objects. Each object must contain exactly three keys—title, description, and weight.
Do not copy large blocks of the question or reference_answer into the text. Each description must begin with its category
prefix, and no extra keys are allowed.

Now, given the question and reference_answer, generate the rubric as described. The reference answer is an ideal response
but not necessarily exhaustive; use it only as guidance.
```

### 2.3 两个 prompt 的差异（复现时要注意）

| | Medicine | Science |
|---|---|---|
| 覆盖维度清单 | 多列了 `harmlessness, patient-centeredness, empathy` | 无这三项 |
| Category guidance 是否给出数值 | **给了**：Essential=5, Important=3–4, Optional=1–2 | **没给**，只有定性描述 |
| Pitfall 开头强制格式 | **强制**："Does not mention . . ." 或 "Recommends . . ." | 未强制 |
| 显式的 self-contained 补充规则 | 有 5 条（answer choice / final answer / 推理顺序 / brevity / specific findings） | 有 4 条（少 "specific findings"） |

> Science prompt 没有给出 Essential=5 / Important=3-4 / Optional=1-2 的数值锚点，这直接解释了我们在数据里观察到的 science 权重不合规（见 forensics F5：science 只有 77.6% 的权重落在推荐区间，medicine 是 99.99%）。

### 2.4 Prompt 里可机器校验的硬约束清单

从上面两段 prompt 可以抽出这些**可自动检查**的约束，我们的 harness 应该把它们做成 lint 规则：

1. 条数 ∈ [7, 20]，且应随题目复杂度变化
2. `title` 为 2–4 个词
3. `description` 是**一句话**，且以 category 前缀开头
4. category ∈ {Essential, Important, Optional, Pitfall}
5. weight：Essential/Important/Optional ∈ [1,5]；Pitfall ∈ {−1, −2}
6. （medicine）weight 应符合 Essential=5 / Important=3–4 / Optional=1–2
7. （medicine）Pitfall description 必须以 "Does not mention" 或 "Recommends" 开头
8. 输出恰好三个 key：`title`, `description`, `weight`，不允许额外 key
9. **不得大段照抄 question 或 reference_answer**
10. 每条 item self-contained（非专家无需外部信息即可判定）

---

## 3. Rubric 格式约定

### 3.1 四个类别的定义（论文原文语义）

| 类别 | 定义 | 数值权重 | RaR-EXPLICIT 训练时的重映射 |
|---|---|---|---|
| **Essential** | Critical facts or safety checks；缺失则回答无效 | 5 | 1.0 |
| **Important** | 关键推理、完整性或清晰度；显著影响质量 | 3–4 | 0.7 |
| **Optional** | 加分的风格或额外深度；有更好但不致命 | 1–2 | 0.3 |
| **Pitfall** | 本题特有的常见错误或遗漏 | **−1 或 −2** | **0.9（正数！）** |

### 3.2 Pitfall 的符号约定 —— 论文自相矛盾的地方

论文 §4.4 脚注 3 原文：

> *"Pitfall criteria are phrased in positive form (e.g., "The response avoids misinformation"), so satisfying them contributes positively to the score. If a pitfall is not satisfied, the corresponding reward is reduced or penalized."*

但生成 prompt 强制的格式是 **"Pitfall Criteria: Does not mention . . ."**，这是**描述失败**的表述（满足 = 回答有问题），与脚注声称的 "positive form"（满足 = 回答好）**方向相反**。同时数据里的权重是 −1/−2，而 RaR-EXPLICIT 又把 Pitfall 映射成 **+0.9**。

> 这一处矛盾在真实数据里造成了可量化的后果，见 forensics **F6**（两个 domain 的 Pitfall 极性约定正好相反，任选一种符号约定必有一个 domain 被训反）。

### 3.3 论文给出的两个完整示例

**Medicine 示例**（PDF p.12）—— 注意 7 条 item，Pitfall 权重为 −1：

> Question: A 50-year-old male patient weighs 65 kg with a pH of 7.05, PCO2 of 15 mmHg, HCO3 of 5 mEq/L, and a base deficit of -40 mEq/L. How much sodium bicarbonate should be administered in the first 4 hours to correct his metabolic acidosis?
>
> - Bicarbonate Calculation (w=5): Essential Criteria: The response must correctly identify and apply the formula (Base Deficit × Body Weight × 0.3) to determine the bicarbonate requirement.
> - Safe Dosing Recommendation (w=5): Essential Criteria: The response must state a clear recommendation of administering about 150 mEq of sodium bicarbonate over the first 4 hours.
> - Partial Correction Justification (w=4): Important Criteria: ...
> - Step-by-Step Calculation (w=3): Important Criteria: ...
> - Base Deficit Interpretation (w=2): Optional Criteria: ...
> - Patient Data Accuracy (w=3): Important Criteria: ...
> - **Avoid Overcorrection Risk (w= −1): Pitfall Criteria: Does not mention the risks associated with rapid overcorrection ...**

**Science 示例**（PDF p.13）—— 7 条 item，reference answer 只有一句话（"Boric acid is more soluble in ethanol than in benzene."）。这正是我们在数据里发现的 "reference 极短却仍生成 7 条 criteria" 的典型形态。

---

## 4. Rubric → Reward：两种聚合方式

### 4.1 Explicit Aggregation（§2.2, Eq. 1）

每条 criterion 由 LLM judge 独立判定为二元 $c_j \in \{0,1\}$，然后加权归一：

$$r(x, \hat{y}) = \frac{\sum_{j=1}^{k} w_j \cdot c_j(x, \hat{y})}{\sum_{j=1}^{k} w_j}$$

- 归一化是为了让不同 prompt（rubric 条数/权重不同）之间的 reward 可比。
- 论文说 $c_j$ 用二元判定，但公式可扩展到连续值。
- **RaR-EXPLICIT 实际训练时不用数据里的数值权重**，而是按 category 重映射为 `{"Essential": 1.0, "Important": 0.7, "Optional": 0.3, "Pitfall": 0.9}`。

### 4.2 Implicit Aggregation（§2.2, Eq. 2）

把所有 criteria + categorical weight 一起塞给 judge，让 judge 直接给一个整体分：

$$r_{\text{implicit}}(x, \hat{y}) = f_\phi\big(x, \hat{y}, \{d_j\}_{j=1}^{k}\big)$$

judge 输出 1–10 Likert，归一化到 [0,1]。**这是论文效果最好的变体。**

### 4.3 Judge prompt 原文（Appendix A.6，逐字）

**RaR-IMPLICIT：**

```text
System Prompt:
You are an expert evaluator. Given a user prompt, a generated response, and a list of quality rubrics, please rate the overall
quality of the response on a scale of 1 to 10 based on how well it satisfies the rubrics.
Consider all rubrics holistically when determining your score. A response that violates multiple rubrics should receive a
lower score, while a response that satisfies all rubrics should receive a higher score.
Start your response with a valid JSON object that starts with "```json" and ends with "```". The JSON object should contain
a single key "rating" and the value should be an integer between 1 and 10.
Example response:
```json
{
"rating": 7
}```

User Prompt Template:
Given the following prompt, response, and rubrics, please rate the overall quality of the response on a scale of 1 to 10 based
on how well it satisfies the rubrics.
<prompt>
{prompt}
</prompt>
<response>
{response}
</response>
<rubrics>
{rubric_list_string}
</rubrics>
Your JSON Evaluation:
```

**DIRECT-LIKERT baseline**（无 rubric）和 **REFERENCE-LIKERT baseline**（给 reference answer 对比）的 prompt 结构相同，只是把 `<rubrics>` 换成无 / `<reference_response>`。完整原文见 `analysis/out/paper_text.txt` 第 948–995 行。

> **注意**：论文**没有给出 Explicit Aggregation 里逐条判定 $c_j$ 用的 judge prompt**。Appendix A.6 只有 IMPLICIT / DIRECT-LIKERT / REFERENCE-LIKERT 三个。复现 RaR-EXPLICIT 时这里需要我们自己补。

### 4.4 RaR-PREDEFINED 用的固定通用 rubric（Appendix A.5，逐字）

这是论文的 "generic rubric" 对照组，只有 4 条、所有 prompt 共用、等权：

```text
• The response contains correct information without factual errors, inaccuracies, or hallucinations that could mislead the user.
• The response fully answers all essential parts of the question and provides sufficient detail where needed.
• The response is concise and to the point, avoiding unnecessary verbosity or repetition.
• The response effectively meets the user's practical needs, provides actionable information, and is genuinely helpful for their situation.
```

> **这一组是我们论证的关键锚点**：论文自己证明了 generic rubric 会把 HealthBench 从 29.7 打到 **12.5**（比不训练的 Qwen2.5-7B-Instruct 22.7 还差 10 个点）。而我们在 forensics 里量到，RaR 自己合成的数据里有 **17.7%（science）/ 11.6%（medicine）** 的 criterion 是"可回收的"——即和这 4 条一样通用。

---

## 5. 训练与评测设置

### 5.1 数据集

| | RaR-Medicine | RaR-Science |
|---|---|---|
| 规模 | 20k prompts | ~20k prompts |
| 来源 | medical-o1-reasoning-SFT、natural_reasoning、SCP-116K、GeneralThought-430K | natural_reasoning、SCP-116K、GeneralThought-430K（对齐 GPQA-Diamond 类目） |
| rubric 生成模型 | GPT-4o | o3-mini |

### 5.2 训练超参（Appendix A.3, Table 10）

| 项 | 值 |
|---|---|
| base policy | **Qwen2.5-7B**（base，非 instruct） |
| 算法 | **GRPO** |
| judge | **gpt-4o-mini** |
| num_rollouts_per_prompt | 16 |
| batch_size (effective) | 96 |
| sampling_temperature | 1.0 |
| learning_rate | 5.0e-06 |
| lr_scheduler | constant_with_warmup（warmup_ratio 0.1） |
| max_length | 3584 |
| num_train_steps | 300 |
| 硬件 | 单节点 8× H100 |

### 5.3 评测

- **HealthBench**（rubric-based, free-form，5000 例临床对话，physician-authored rubric），greedy decoding，报 overall + 5 个 axis。消融用 **HealthBench-1k** 子集，其余用于训练。
- **GPQA-Diamond**（multiple-choice），10 次独立运行取均值 + 95% CI，选项打乱，抽取 `\boxed{}`，失败则 fallback 到 GPT-4o verifier。
- **LLM-Judge Alignment**：基于 ~3000 条 HealthBench prompt 构造 chosen/rejected 对（用 o3 做可控劣化，见 A.7/A.9），指标是 pairwise preference accuracy。

### 5.4 主结果（Figure 2）

| 方法 | HealthBench | GPQA-Diamond |
|---|---|---|
| Qwen2.5-7B | 7.7 | 31.7 |
| Qwen2.5-7B-Instruct | 22.7 | 35.0 |
| DIRECT-LIKERT | 25.5 | 34.8 |
| REFERENCE-LIKERT | 28.9 | 36.5 |
| **RaR-PREDEFINED（通用 rubric）** | **12.5** | **31.7** |
| RaR-EXPLICIT | 29.7 | 36.9 |
| **RaR-IMPLICIT（最好）** | **31.2** | **37.6** |

摘要声称 "relative improvements of up to 31% on HealthBench and 7% on GPQA-Diamond over ... Direct-Likert"。按 Figure 2 的数字算，HealthBench 是 (31.2−25.5)/25.5 = **+22.4%**，GPQA 是 (37.6−34.8)/34.8 = **+8.0%**。31% 这个数字对不上 Figure 2 的 overall score，可能来自某个 per-axis 分数或早期版本。**引用时建议直接用绝对分数，不要转述 "31%"。**

> 另注：正文写 "Table 2 reports results on HealthBench ... and GPQA-Diamond"，但 Table 2 实际是消融表，主结果在 Figure 2。论文表号有错乱。

---

## 6. 消融实验（**对我们最关键的一节**）

论文一共做了三组和 rubric 质量有关的消融，全部在 HealthBench-1k 上、用 Qwen2.5-7B + GRPO 训练（HealthBench-3.5k 子集）。

### 6.1 Table 1：人工 vs 合成 rubric、有无 reference

| Training Method | Overall Score |
|---|---|
| Expert-Answer-SFT | 20.4% |
| Simple-Likert | 23.9% |
| Reference-Likert | 31.7% |
| RaR-Implicit-Synthetic-**NoRef** | 32.0% |
| **RaR-Implicit-Synthetic（有 reference）** | **35.9%** |
| RaR-Implicit-**Human**（人工写的 rubric） | 34.8% |

**结论（论文的）**：合成+reference（35.9）≈ 人工（34.8）> 合成无 reference（32.0）。论文据此说 "reference-answer guidance 很重要"。

> **对我们的含义（两面）**：
> - 正面：这说明 rubric 质量确实能带来 ~4 个点的 downstream 差异，我们的改进有空间。
> - **反面（更重要）**：论文用这条结论宣称"合成 rubric 已经追平人工"。但注意 human rubric 只有 3.5k 条 HealthBench 数据、且 34.8 < 35.9，**这个"追平"是在一个很小的 in-domain 子集上、单一 seed、无置信区间的结果**。这正是我们可以攻击的点：论文从没在 RaR-Science / RaR-Medicine 这两个真正发布的 20k 数据集上验证过 rubric 质量。

### 6.2 Table 2：rubric 结构设计消融

| Ablation Setting | Overall Score |
|---|---|
| Essential-Only Rubrics | 34.9% |
| **No Categorical Labels** | **38.8%** |
| No Pitfall Criteria | 37.2% |
| All Rubrics（完整） | 37.2% |

**这张表是论文里最有价值也最被轻描淡写的结果：**

1. **去掉 categorical label 反而最好（38.8 > 37.2）**。也就是说 Essential/Important/Optional/Pitfall 这套权重体系**不但没帮助，还有害**。论文只用一句 "we observe minimal performance differences" 带过。
2. **去掉 Pitfall 完全没有影响（37.2 = 37.2）**。论文的解释是 "synthetically generating effective pitfall criteria is inherently difficult ... these synthetic negative criteria may lack the specificity or relevance needed to meaningfully penalize undesirable responses"。
   > 我们的 forensics 给了一个**更朴素也更可修复的解释**：Pitfall 的符号约定在两个 domain 里是反的（F6）。不是"难以生成"，而是"生成了但符号用错了"。
3. **只留 Essential 会掉 2.3 个点（34.9 vs 37.2）**，说明更丰富的 criteria 有用 —— 但这与第 1 点合起来看，意思是：**要多条 criteria，但不要那套权重**。

### 6.3 Table 3：生成模型能力消融（无 reference 条件下）

| Rubric Generation Model | Overall Score |
|---|---|
| O3-mini（**有** reference） | 35.9% |
| GPT-4o | 34.2% |
| GPT-4o-mini | 32.7% |
| Qwen-72B-Instruct | 32.7% |
| O3-mini | 32.4% |
| Qwen-7B-Instruct | 31.9% |
| Qwen-32B-Instruct | 31.1% |

**结论**：更强的模型生成的 rubric 更好，但**全都比不上"较弱模型 + reference"**。也就是说，在论文的框架里，*信息*（reference）比*算力*（更强的 generator）更值钱。

> **对我们的含义**：论文只试了 "换个更大的模型" 这一条改进路径，而且撞到了天花板（GPT-4o 34.2 < o3-mini+ref 35.9）。**"换更强的单次生成器"这条路已经被论文证明收益有限——这恰恰为 agentic 生成留出了空间**：不是让模型更强，而是让流程能检索、分解、验证、修订。

### 6.4 论文**没有**做的消融（我们的机会）

- ❌ **rubric 条数**的影响（7 条 vs 15 条 vs 自适应）
- ❌ **rubric 冗余度 / 通用度**的影响
- ❌ **任何 intrinsic rubric quality metric**（论文所有质量判断都是 downstream HealthBench 分数，一次训练几百 GPU-hour）
- ❌ 在 **RaR-Science / RaR-Medicine 本身**上的 rubric 质量消融（Table 1/2/3 全部在 HealthBench 子集上做）
- ❌ 多步 / 迭代 / 带验证的生成流程
- ❌ rubric 与 judge 的交互（同一 rubric 换 judge 的稳定性，只在 Table 11 粗略碰了一下）

---

## 7. 论文自认的 Limitation

### 7.1 §9 明写的三条

1. **领域受限**：只做了 medicine + science，dialogue / tool use / agentic 任务未验证。
2. **只试了两种聚合策略**（implicit / explicit）。未来可探索 "learning continuous weights for each criterion or dynamically adjusting weights over the course of training to mimic a curriculum"。
3. **judge 是现成的通用 LLM**，专用 evaluator 或 generative reward model 可能更好。

### 7.2 散落在正文里、和 rubric 质量直接相关的自认（更有价值）

| 位置 | 原文 | 含义 |
|---|---|---|
| §5 | "rubrics are generated with stronger LLMs using reference answers as proxies for expert supervision, so **rubric quality is impacted by reference quality**" | rubric 质量被 reference 质量卡住。→ 我们量到 science 有 9.5% 的题 reference 是**空的**、25% ≤6 词（F9） |
| §5 | "fixed weighted sums in RaR-Explicit offer more control but **can be brittle**" | 承认加权和脆弱 |
| §6 | "**synthetically generating effective pitfall criteria is inherently difficult** ... these synthetic negative criteria **may lack the specificity or relevance** needed" | 承认 Pitfall 生成失败 |
| §6 | "**Purely synthetic rubrics, while scalable, currently fall short in capturing the subtle criteria** required for robust training in high-stakes domains" | 承认纯合成 rubric 不够好 —— 这就是我们研究想法的论文背书 |
| §5 | RaR-Predefined "underperforms because **generic criteria miss prompt-specific requirements** and common failure modes, producing **misaligned reward signals**" | 论文自己确认 generic criterion 会产生错位的 reward |

---

## 8. 复现 baseline 的 checklist

我们的 harness 要和这个 baseline 对比，必须精确复现以下几点：

- [ ] 生成器：science 用 o3-mini、medicine 用 GPT-4o（若不可得，需在文档中注明替代模型并做 sanity check）
- [ ] 输入：`question` + `reference_answer`，单次调用，无 system/user 拆分之外的额外结构
- [ ] Prompt：**逐字使用 §2.1 / §2.2 的原文**（两个域用各自的版本）
- [ ] 输出格式：JSON array，每个对象恰好 `title` / `description` / `weight` 三个 key
- [ ] 目标条数：7–20
- [ ] 权重：Essential/Important/Optional ∈ [1,5]，Pitfall ∈ {−1,−2}
- [ ] Reward 侧：
  - Explicit：Eq. 1，权重用 category 重映射 `{Essential:1.0, Important:0.7, Optional:0.3, Pitfall:0.9}`（**注意这会踩 F6 的符号坑，需要显式记录我们怎么处理**）
  - Implicit：§4.3 的 judge prompt 原文，1–10 Likert 归一化到 [0,1]
- [ ] 对照组：RaR-PREDEFINED 的 4 条固定 rubric（§4.4），等权

---

## 9. 一句话总结：论文留给我们的缝隙

> RaR 证明了 **"instance-specific rubric 比 generic rubric 和 Likert 好得多"**，也证明了 **"换更强的单次生成器收益已经见顶"**，还自己承认 **"纯合成 rubric 在高风险领域不够用"**、**"Pitfall 生成失败"**、**"去掉权重体系反而更好"**。
>
> 但论文**从未定义过一个 rubric 质量的内在度量**，也**从未尝试过单次 prompt 之外的生成流程**。所有质量判断都必须跑一次 GRPO 训练才能得到。
>
> 这就是 agentic rubric generation harness 的立足点：**(a)** 提出可离线计算的 rubric 质量指标（我们在 forensics 里已经做出了一套）；**(b)** 用 agentic 流程（分解子问题 → 检索/求解 → 起草 → 按 lint 规则验证 → 修订去冗余/去样板）针对性修掉这些指标上的失效模式；**(c)** 证明指标改善能传导到 downstream RL 收益。
