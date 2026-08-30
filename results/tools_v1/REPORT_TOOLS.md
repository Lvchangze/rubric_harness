# 给模型可执行工具，rubric 会更好吗？（`tools_v1`）

**结论：不会。在 128 道混合结局题上，`agentic-tools` 与 `agentic` 在所有指标上都无法区分，
best-of-n 甚至方向不利。工具机制确实完整运行了，所以这是一个真实的否定结果，不是实现失败。**

---

## 1. 实验设置

| | |
|---|---|
| 题集 | `rollout_v2` 的 **128 道混合结局题**（65 medicine / 63 science），逐字复用 |
| 为什么是这批题 | GRPO 里全对/全错的 group 对每个成员的 advantage 都是 0，rubric 在那些题上好坏不影响训练。混合结局题是 reward signal 唯一起作用的总体 |
| 对照 | `shipped`（论文原始）、`baseline`（RaR 单次合成）、`agentic`（本方法，无工具） |
| 处理 | `agentic-tools` = `agentic` + `investigate` + `critic_tools` 两个阶段，**其余 stage / prompt / 阈值完全继承** |
| 生成成本 | 3093 秒，9988 次 LLM 调用，2593 次工具调用，0 次失败 |
| 判定 | 与 rubric 无关的 oracle 标注 rollout 对错；judge 与生成同模型；BH-FDR，族 = 6 个判别类指标 |

两臂只差「让模型自己取证」这一件事，所以差值可归因。

## 2. 主结果：rollout 协议（RL 忠实，无 gold 泄漏）

参照列 = `agentic`，配对 n=128。

| 指标 | 方向 | `agentic` | `agentic-tools` | Δ | p | q |
|---|:--:|--:|--:|--:|--:|--:|
| AUC(rubric 分数, oracle 正确性) | ↑ | 0.768 | 0.777 | +0.009 | 0.61 | 0.73 |
| **Best-of-n 选择准确率** | ↑ | 0.824 | **0.796** | **−0.027** | 0.21 | 0.55 |
| Spearman(分数, 正确性) | ↑ | 0.397 | 0.418 | +0.021 | 0.43 | 0.64 |
| 分数间隔（正确 − 错误） | ↑ | 0.236 | 0.221 | −0.016 | 0.27 | 0.55 |
| 尺度归一化间隔 | ↑ | 1.065 | 1.044 | −0.021 | 0.75 | 0.75 |

**没有任何一格通过 FDR。** 方向还在两域之间互相矛盾：AUC 在 science 是 +0.043、在 medicine 是
−0.024；best-of-n 两域都是负的（science −0.054, medicine −0.002）。

放进四来源全表看，`agentic-tools` 的 best-of-n 是**四者最低**（0.796 vs `baseline` 0.840、
`agentic` 0.824、`shipped` 0.812；随机地板 0.614）。这条没有统计显著性，但方向不利，
按 `HANDOFF.md` §5 预先写死的判读规则——**best-of-n 与 AUC 冲突时默认相信 best-of-n**——
不能把 AUC 上的 +0.009 说成趋势向好。

真实 reward hacking 探针（oracle 判错的 rollout 内，分数与长度的相关）同样没有帮助：
`agentic` +0.287（vs baseline **−0.173, p=0.018**），`agentic-tools` +0.367（−0.093, p=0.18）。
无工具那一臂的抗 hacking 优势，加了工具之后**反而变弱**。

## 3. 结构性指标（零 LLM，可完全复现）

18 项里 17 项无显著差异。唯一通过 FDR 的是 `lint/compliance_rate`（0.675 → 0.735,
q=0.0072），而它衡量的是对论文 RaR 格式的贴合度，不是 rubric 质量。

| 指标 | `agentic` | `agentic-tools` | Δ | q |
|---|--:|--:|--:|--:|
| 通用条目率 ↓ | 0.079 | 0.078 | −0.001 | 1.0 |
| specificity lift ↑ | 0.888 | 0.888 | −0.000 | 1.0 |
| 锚点数/条 ↑ | 3.034 | 2.946 | −0.088 | 0.44 |
| 主观条目率 ↓ | 0.073 | 0.060 | −0.013 | 0.29 |
| 单一极性率 ↑ | 1.000 | 1.000 | 0.000 | 1.0 |

这是可以预期的：这些属性由确定性的 `lint` 阶段**强制**保证，无工具流水线已经把可测空间占满
（通用条目率 0.079 对比 `baseline` 的 0.188，单一极性率已经是 100%）。工具在这里没有剩余空间。

## 4. 机制确实运行了 —— 所以这是真实的否定结果

一个否定结果只有在被否定的东西真的发生过时才有意义。

| | |
|---|---|
| 工具调用/题 | mean **20.3**（median 20，range 11–31） |
| `investigate` 成功 | **124/128**，其中**零工具的有 0 题** |
| `critic_tools` 成功 | **128/128**，零工具 0 题 |
| 已验证事实/题 | 4.4，其中有区分度 3.1 |
| **标出参考答案本身的问题** | **165 处，涉及 108/128 题** |
| 主动否掉「不具区分度」的检查 | **347 条** |
| 工具 critic 的编辑 | 删除 **196** 条、改写 **60** 条 |

按工具分布：`execute_criterion` 1710、`make_counterexample` 434、`python_eval` 163、
`extract_reference_claims` 121、`find_similar_questions` 96、`check_specificity` 42、
`check_equivalence` 26、`check_units` 1。

模型**压倒性地偏好三个走 LLM 的工具**，几乎不用五个零成本的确定性工具。这既解释了成本
（2593 次工具调用里绝大多数自己就是一次 LLM 调用），也指向一个具体的改进方向：
`check_specificity` 全程只被调用 42 次，而它恰恰是唯一能直接测「这条判据是否可回收」的工具。

`investigate` 有 **112/124 用满轮次预算**（`max_rounds_closed`），只有 12 题是自然收尾。
所以调查普遍是被截断的。

## 5. 诚实的解读

**被支持的**：无。没有任何指标显示工具改善了 rubric。

**被否定的**：「让 agent 自主调用可执行工具会产生更好的 rubric」这一主张，在本设置下**没有得到支持**，
且 best-of-n 与 reward-hacking 两项方向不利。

**无法判断的**：下游 RL 性能；以及在 `investigate` 不被截断、或强制使用确定性工具的配置下会不会不同。

**为什么可能是这样**——三条推测，都还没验证：

1. **结构性空间已被占满。** 无工具流水线用确定性 lint 强制了 grounding、极性、去主观化，
   工具想改善的正是这些，没有剩余空间。
2. **验证的是错误的东西。** `execute_criterion` 占了 66% 的工具调用，它测的是「判据能否区分
   已有的答案」；但 `critic` 阶段本来就在做这件事，`critic_tools` 只是又做了一遍。
3. **调查被截断。** 90% 的 investigate 用满了预算，收尾时靠的是「基于已查到的写」，
   可能稀释了证据质量。

**一个副产品值得单独提**：工具在 **108/128 道题上标出了参考答案本身的问题**（共 165 处）。
即使工具没有改善 rubric，这个能力对**数据集质量审计**是直接有用的——RaR 的参考答案被论文
和本仓库都当作 ground truth，而它们并非都正确。

## 6. 产出

- `results/tools_v1/rollout_report.md` — 四来源主表，分域，FDR
- `results/tools_v1/reward_hacking_probe.md` — 真实失败 rollout 内的分数-长度相关
- `results/tools_v1/summary_table.md` — 结构性指标，参照 `baseline`
- `results/tools_v1/tools_vs_agentic.md` — 结构性指标，参照 `agentic`
- `results/tools_v1/vs_agentic/tools_v1/` — 以 `agentic` 为参照的完整聚合
- `runs/tools_v1/rubrics_agentic-tools.jsonl` + `traces/` — 每一次工具调用的参数、结果、耗时
