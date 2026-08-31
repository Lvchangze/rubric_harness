# HANDOFF — 接手做真实 RL 验证之前，请先读完这一份

**读者**：准备用这些 rubric 做 GRPO/PPO 训练、或者想把没跑完的那个实验跑完的同学。
**假设**：你没看过之前的讨论。读完这份应该不需要再来问人。

配套材料：完整实验报告 [`results/REPORT.md`](results/REPORT.md)（105KB，含全部推导与产物路径）；
**外部基准报告 [`results/rubricbench/REPORT.md`](results/rubricbench/REPORT.md)**（唯一直接测量
rubric 质量的证据，与本地代理指标结论不一致，见 §2 表第 18–21 行与 §5b）；
工程说明 [`README.md`](README.md)。本文件是**决策用摘要 + 操作手册**，
报告是**证据本体**。两者冲突时以报告和 `results/` 下的产物为准。

---

## 1. 一句话结论 + 三句话背景

> **agentic 生成的 rubric 在"写得具体、可判定、有锚点"上无条件地优于论文的单次生成，
> 在理科题上还能把好坏答案的分数拉得更开；但"它能不能变成更好的 reward signal"
> 这个真正要紧的问题，本地实验没有回答——而唯一一次外部直接测量（RubricBench，
> 1147 组人类偏好）给的答案是"不能"：agentic 0.5798 反而低于单次生成的 0.5876。**

> **补充（RubricBench 之后）**：迄今唯一在真人偏好上有效的干预是**改生成 prompt 的框架**
> （`framed` 0.6260, q=0.0058），不是 agent 流程也不是工具。
> 但那个 prompt 是在看过评测集失败案例之后写的，留出估计只有 +0.0132（不显著）。
> **不要把 0.6260 当成可迁移的结论**，理由见 §2 表第 20 行与 §5b 第 4 条。

**背景三句**：
RaR（Rubrics as Rewards）论文用一次 LLM 调用合成 rubric，我们怀疑质量不够，
做了一个多阶段 agent 流程（把每条判据拿去实际执行、用反例和参考答案筛一遍）来对比。
本地不能跑 RL，所以全部用**代理指标**：rubric 判好坏答案的分辨力、结构质量、抗 reward hacking。
结论是**结构质量的改善很硬、判别力的改善只在 science 域、排序准确率没有改善**，
而最能预测下游的那个实验（用真实 rollout 而非参考答案当正例）**因为功效不足和端点故障没做完**。

---

## 2. 证据现状表

可信度定义：
**强** = 零 LLM 调用可复验、两域一致、与全语料取证定义相同；
**中** = 经过 judge，但做了泄漏控制 + 换 judge 复现 + FDR；
**弱** = 单一协议、未过 FDR 或 n 不足；
**未验证** = 没有做或没做完。

| # | claim | 支持它的协议 | n | 效应量与不确定性 | 可信度 |
|---|---|---|---|---|---|
| 1 | **通用/万能条目更少** | 零 LLM 结构指标 | 198 | 0.210 → 0.095，Δ **−0.115** [−0.139, −0.089], q<0.05 | **强** |
| 2 | **主观/文体条目更少** | 零 LLM | 198 | 0.181 → 0.078，Δ **−0.103** [−0.128, −0.079], q<0.05 | **强** |
| 3 | **锚点更多（更可判定）** | 零 LLM | 198 | 2.191 → 2.821，Δ **+0.630** [0.509, 0.754], q<0.05 | **强** |
| 4 | **query-specificity 更高** | 零 LLM（置换对照） | 198 | 0.763 → 0.862，Δ **+0.098** [0.071, 0.125], q<0.05 | **强** |
| 5 | **极性统一（修掉 F6）** | 零 LLM | 198 | 0.167 → **1.000**，Δ +0.833 | **强** |
| 6 | **好坏答案分数拉得更开（仅 science）** | 协议一，删 gold 档 + FDR；换 judge 复现 | 56 (sci) | z_separation **+0.325**, p=0.0024, **q=0.009**；换 judge +0.340, p=0.0013 | **中** |
| 7 | **上面这条在 medicine 上是零** | 同上 | 94 / 57 | z_sep +0.021 (p=0.61)；换 judge +0.088 (p=0.13) | **中** |
| 8 | **域间差距真实存在，非 judge 伪影** | 两个 judge 同题配对 | 109 | science−medicine **+0.254** (p=0.028) / **+0.252** (p=0.030) | **中** |
| 9 | **无自我偏好**（judge 不因与 generator 同源而偏袒） | 换 judge + 同 judge 地板臂 | 109 | 分数级 Δ −0.000 (p=0.94)；排序一致率 agentic 反而**衰减更少** +0.021 (p=0.030) | **中** |
| 10 | **降低长度型 reward hacking（仅 science）** | 协议二，错误 rollout 内 | 61 | 分数-长度相关 +0.272 → **−0.040** (p=0.044)；**medicine 反向变差** | **弱** |
| 11 | **排序更准（AUC / ranking accuracy）** | 协议一 + 协议二 | 150 / 61 | 协议一 AUC +0.032 (**q=0.089**, 不过 FDR)；协议二 AUC +0.036 (**q=0.715**) | **无证据** |
| 12 | **best-of-n 选择更准** | 协议二 | 61 | **baseline 0.816 > agentic 0.803**（Δ −0.013, p=0.76）**方向相反** | **无证据（方向不利）** |
| 13 | **权重携带信息**（论文 F5 的靶子） | 零 LLM + judge | 198 | agentic 权重与等权相关 0.989，**没有优于对照** | **无证据** |
| 14 | **盲评里更受偏好** | head-to-head，两个 judge | 114 | judge **更偏好 baseline**（B−A −0.202 / −0.132），**换 judge 也没翻转** | **无证据（方向不利）** |
| 15 | **Stage 6 哪一半有用** | 三套证据互相矛盾 | — | 见 §3，**证据结构不允许判决** | **未验证** |
| 16 | **转化为更好的 reward signal（下游 RL）** | 无 | — | 本地无法验证 | **未验证** |
| 17 | **足够功效下协议二的结论** | §4.6，**没跑完** | 目标 128 | 选题完成，判定零进度 | **未验证** |
| **18** | **在真人偏好上更好用**（RubricBench） | **外部基准，无 reference 故无 gold 泄漏** | **1147** | **`agentic` 0.5798 < `baseline` 0.5876（Δ −0.0079, p=0.63）；对「不给 rubric」+0.0148 (p=0.33)** | **无证据（方向不利）** |
| **19** | **工具让 rubric 更好用** | RubricBench，子集内配对 | **299** | 相对无工具 `agentic` +0.0234（p=0.34, q=0.56）；同子集上还略低于 `framed` | **无证据** |
| **20** | **改生成 prompt 的框架有用** | RubricBench | 1147 | `framed` 0.6260，vs `baseline` +0.0394 (**q=0.0058**)、vs `none` +0.0629 (**q=1.1e-05**) | **弱**（见下面三条扣分项） |
| **21** | **该基准的排行榜能分辨方法** | 官方 4 个已发表提交，配对 | 1147 | **0/6 对显著**；榜首榜尾差 0.0148 (p=0.401) < 最小可检出 0.027 | **强（反向结论）** |

第 20 行为什么只给「弱」——三条扣分项，缺一不可读：
(a) `framed` 的 prompt 是**看过本基准失败案例之后**写的，用「非 SAFETY 四域」作准留出集，
效应从 +0.0394 掉到 **+0.0132（p=0.33，不显著）**；
(b) 净增量的 **68.9% 来自 80 道 SAFETY 题**；
(c) 换成位置受控口径（两序一致才算数）后 **q=0.079，掉出显著**。
详见 [`results/rubricbench/REPORT.md`](results/rubricbench/REPORT.md) §8。

**读这张表最重要的一件事**：1–5 行（强）和 11–14、18–19 行（无证据）说的是**不同的东西**。
rubric 写得更具体，**不等于**它作为 reward 更好用。
`agentic-noval`（关掉验证环）就是干净的反例：结构指标大幅领先 baseline，判别力却与 baseline 无法区分。

**第 18 行让这句话从「本地代理指标的推测」变成了外部基准上的实测**，并且给出了原因：
本仓库的结构性指标全都在测判据**写得好不好**（够不够具体、有没有锚点、可不可判定），
**没有一个在测判据是不是关于对的东西**。RubricBench 上 `baseline` 最典型的失败是
把 instruction 当规格书逐条展开——`rubric_eval_507` 里它把
「explicitly describes Carl's butt physique, as requested」（Carl Grimes 是未成年角色）
列成 4/5 分的得分项。**这条判据在我们全部 5 个结构性指标上都是满分**，
而人类偏好的是拒绝那一侧。

---

## 3. 那个没解开的死结：Stage 6 的哪一半在起作用

**为什么重要**：Stage 6 是 agentic 判别力增益的**唯一来源**（关掉它 = `agentic-noval`，
判别力立刻掉回 baseline 水平）。它内部有两半：

- **gold 侧**：拿参考答案去试每条判据，参考答案满足不了的就删。
- **negative 侧**：拿构造出来的错误答案去试，错误答案也能通过的判据就删（"不判别"的判据）。

我们做了拆半消融（`agentic-goldonly` / `agentic-negonly`）。三套证据是这样投票的：

| 证据 | 正例是什么 | judge | 判决 |
|---|---|---|---|
| 协议一（合成降级梯子） | **参考答案本身** | 与 generator 同模型 | **gold 侧**携带几乎全部增益，negonly 全零 |
| 换 judge 复现 | **参考答案本身** | gpt-5.1（不同厂商） | **gold 侧**，甚至反超完整 agentic |
| 协议二（真实 rollout） | **模型 rollout** | 与 generator 同模型 | **negative 侧**最好，goldonly 与 baseline 持平 |

看起来是 2:1，但**那两票不独立**：

> 协议一和换 judge 用的是**同一个协议**——拿 `reference_answer` 当评测正例。
> 而 gold 侧过滤的目标函数**就是"让参考答案能通过"**。
> 用参考答案当考题去考"对着参考答案优化过的 rubric"，它当然赢。
> **换 judge 换掉的是 judge，不是协议**，所以它复现的是**同一个偏袒**，
> 等于一票算了两次。

唯一把正例换成模型 rollout（生成器从未见过的文本）的协议二投给 negative 侧——
但它 n=61、没过 FDR，**不足以反向定论**。

> **明确结论：证据结构本身不允许判决。
> 唯一能判决的实验，就是 §4.6 那个没跑完的高功效 rollout 评测。**

**这不是学术洁癖。** 如果真相是"gold 侧的增益是拟合假象"，
那么把 agentic rubric 直接拿去做 RL reward，你得到的可能只是一个
**对着参考答案过拟合、且天花板更低**（headroom 仅 0.128，15.8% 的降级答案已 ≥0.8）的奖励函数。
`agentic-negonly` 是构造上不泄漏的那一半，且**不带天花板副作用**——
如果预算只够加一臂消融，加它。

---

## 4. 没跑完的实验：精确状态与续跑方法

### 4.1 它想回答什么

协议二（真实 rollout）第一次跑出的是 **n=61 的 null**。诊断很清楚：
先对 200 题全量生成 rubric，才发现 123 题模型全做对、15 题全做错，只剩 61 题可用。
**这是功效失败，不是发现。**

修法是把顺序反过来——先筛后生成。而且这不只是省钱：

> **在 GRPO 里 advantage 是对 group 均值取的，
> 所以全对（或全错）的 group 里每个成员的 advantage 都是 0，对梯度没有任何贡献。**
> rubric 在这种题上打分准不准，按构造就不影响训练。
> **"结局混合"的题不是权宜子集，而是 reward signal 唯一起作用的那个总体。**

筛选已经做完了，**这本身是给你的一个硬数字**：

| | rar_science | rar_medicine | 合计 |
|---|---|---|---|
| 筛选题数（k=4 rollout，rubric-blind oracle） | 451 | 329 | **780** |
| **结局混合（可用）** | 63 | 65 | **128 (16.4%)** |
| 全对（零 advantage） | 351 | 228 | 579 (74.2%) |
| 全错（零 advantage） | 37 | 36 | 73 (9.4%) |

rollout 总体正确率 84.5%，oracle 二次判定自洽率 95.2%（146 次抽查）。

### 4.2 中断时的确切状态

| 产物 | 状态 |
|---|---|
| `runs/rollout_v2/examples.jsonl` | ✅ **128 道入选题**（65 medicine + 63 science） |
| `runs/rollout_v2/rollouts.jsonl` | ✅ 780 题 × k=4 的 rollout + oracle 标签 |
| `runs/rollout_v2/screening_summary.json` | ✅ 上表全部数字 |
| `rubrics_shipped` / `rubrics_baseline` | ✅ **128/128** |
| `rubrics_agentic` / `-goldonly` / `-negonly` | ❌ **各 62/128**（只有从 pilot 导入的那批） |
| k=4 → k=8 加采 | ❌ 未开始 |
| 逐 rollout 判定、主表 | ❌ **零进度** |

已产出的部分零错误、零空 rubric，条目数分布与 pilot 一致，说明中断没有污染它。

**为什么中断**：`hy-t2t-glm-5.2-384k-fp8-L20A-t1-v2` 部署故障——
`2+2` 这样的调用在 concurrency=1 下也 120s 超时，raw curl 同样无响应。
备用端点（`api_azure_openai_gpt-5.1` / `gpt-5.6-sol` / `api_ali_qwen3.8-max`）全部 **401**，
是凭证过期或预算耗尽，**不是服务故障**（同一个 gpt-5.1 当天上午还跑了 4788 次判定 0 失败）。
**接手第一件事：确认你有可用的端点和额度。**

### 4.2b 一个部分样本的预览（**不是最终结果**）

三个 agentic 变体只生成了 62/128，但 `shipped` / `baseline` 已齐。
对**五个来源都齐全的那 62 题**跑了一遍零 LLM 结构性指标（0 次 API 调用），
作为"换到新选集后结构性结论是否依然成立"的预览。

| 指标（agentic vs baseline） | pilot n=198 | **预览 n=62** |
|---|---|---|
| 通用（可回收）条目率 | 0.210 → 0.095，Δ −0.115 | 0.230 → 0.112，Δ **−0.118** |
| 主观/文体条目率 | 0.181 → 0.078，Δ −0.103 | 0.191 → 0.075，Δ **−0.117** |
| 每条 criterion 锚点数 | 2.191 → 2.821，Δ +0.630 | 2.148 → 2.800，Δ **+0.652** |
| query-specificity 提升 | 0.763 → 0.862，Δ +0.098 | 0.748 → 0.868，Δ **+0.120** |
| 单一极性 rubric 率（F6） | 0.167 → 1.000，Δ +0.833 | 0.145 → 1.000，Δ **+0.855** |

全部同号、量级相近、q<0.05。产物：`results/rollout_v2/summary_table.md`（含分域）。

> ⚠️ **这不是独立证据，必须这样标注。**
> 这 62 题**全部**落在 pilot 的 200 题里（重叠 **62/62 = 100%**），
> 所以它不是"在新题上复现"，而是**对 pilot 队列的一个子群分析**。
> 真正新的 66 题，正好就是还没生成 rubric 的那批。
>
> 它唯一的信息量是**子群稳健性**：结构性优势在
> **"结局混合"这个 RL 唯一起作用的子群**上依然成立，
> 而不是只存在于策略反正都能做对的简单题上。这一点有价值，但仅此而已。
> **补齐 66 题后请重跑（`run_rollout_v2.sh` 的第 2–3 步，0 次 API 调用），
> 用 n=128 的结果替换本表。**

### 4.2c 更新：端点已恢复，看门狗已经把它跑起来了

本文定稿时（2026-08-28 23:44）端点恢复，`scripts/resume_when_up.sh` 自动启动了续跑，
生成阶段正在推进（agentic 62 → 68 → …）。**接手时请先看它跑到哪了**：

```bash
tail -f logs/resume.log            # 看门狗与阶段进度
python3 scripts/check_report_facts.py   # 会打印当前进度 vs 中断时刻的快照
ls .lock_rollout_v2 2>/dev/null && echo "还在跑"
```

如果它已经跑完，`results/rollout_v2/rollout_report.md` 就是你要的主表——
**在看它之前先读完 §5 的判读标准。**
如果它中途又断了，重跑 §4.3 那条命令即可（全部断点续跑）。

**注意**：本文件与 `results/REPORT.md` 里的数字都是**中断时刻的状态**
（快照记在 `runs/rollout_v2/cutoff_state.json`），不是"现在"。
续跑完成后 §4.2b 的 n=62 预览应当被 n=128 的结果替换。

### 4.3 续跑：一条命令

```bash
cd rubric_harness
bash scripts/run_rollout_v2.sh
```

或者让它等端点恢复（每 120 秒探一次，**连续两次健康**才启动——
只探一次可能撞上部分恢复，那会在 1200s 超时下慢慢失败）：

```bash
nohup bash scripts/resume_when_up.sh > logs/resume.log 2>&1 &
tail -f logs/resume.log
```

调整并发：`CONC=96 bash scripts/run_rollout_v2.sh`（默认 48）。

### 4.4 分阶段手动跑

如果想逐段确认，按顺序执行下面六条。**每一段都按 uid 断点续跑，
已完成的 LLM 调用命中 `runs/cache/`，所以重跑只付没跑完的那部分。**

```bash
export LLM_REQUEST_TIMEOUT_S=1200
CFG=configs/rollout_v2.yaml
RUN=rollout_v2
SRC="shipped baseline agentic agentic-goldonly agentic-negonly"

# 1) 补齐 rubric：66 题 x 3 个 agentic 变体
python3 scripts/gen_rubrics.py --config $CFG --run-name $RUN --sources $SRC --concurrency 48

# 2) 结构性指标（0 次 API 调用），先跑，端点再挂也已经落袋
python3 scripts/eval_rubrics.py --config $CFG --run-name $RUN \
  --metrics grounding lint adaptivity --sources $SRC

# 3) 聚合 + 配对队列断言（0 次调用）
python3 scripts/aggregate_results.py --run-name $RUN --results-dir results \
  --sources $SRC --expect-paired 128 --expect-metrics 0

# 4) rollout 从 k=4 补到 k=8（前 4 次命中缓存）
python3 scripts/build_rollouts.py --config $CFG --run-name $RUN \
  --k 8 --concurrency 48 --agreement-probe-fraction 0.05

# 5) 判定：128 题 x 8 rollout x 5 来源
python3 scripts/eval_rubrics.py --config $CFG --run-name $RUN \
  --metrics rollout --sources $SRC --concurrency 48

# 6) 出表（0 次调用）
python3 scripts/rollout_report.py --run-name $RUN --results-dir results --sources $SRC
python3 scripts/reward_hacking_probe.py --run-name $RUN --results-dir results
python3 scripts/main_table.py --run-name $RUN --runs-dir runs
```

**预算与耗时**（调用数按 pilot 实测的每题成本推算：agentic 系每题 13.6 次调用，baseline 1 次）：

| 阶段 | 调用数 | 并发 48 时耗时 | 说明 |
|---|---|---|---|
| 1 rubric | ~2,700 | ~40 min | 66 题 × 3 变体 × 13.6 |
| 2 结构指标 | **0** | ~2 min | 纯本地计算 |
| 3 聚合 | **0** | ~1 min | |
| 4 rollout 补采 | ~1,100 | ~25 min | 128 题 × 4 次新采样 + oracle |
| 5 判定 | ~5,100 | ~90 min | 128 × 8 × 5 |
| 6 出表 | **0** | ~2 min | |
| **合计** | **~8,900** | **~2.7 h** | |

**跑之前先验一遍流程**（不发任何网络请求）：

```bash
python3 scripts/dryrun_resume.py   # 断点续跑、合并写、配对队列的排练
bash scripts/selftest.sh           # 极性、统计、prompt 逐字一致性
```

---

## 5. 拿到结果之后怎么读（**请在看数字之前读完这一节**）

判读标准**先写死**，是为了避免拿到结果后挑一个好看的读法。四种情形：

### 情形 A：n≈128 下 agentic 系在 AUC / Spearman 上显著优于 baseline（过 BH-FDR）

**意味着**：协议二此前的 null 确实是功效问题，agentic 的判别力优势在
**生成器没见过的正例**上也成立。这是本项目最强的一个结果，
足以支持"值得花预算做真实 RL 验证"。
**同时必须检查**：这个优势是不是仍然只在 science 域（几乎可以肯定是），
以及 `agentic-goldonly` 与 `agentic-negonly` 的相对位置——
如果 negonly ≥ goldonly，§3 的死结就朝"gold 侧增益是拟合假象"解开了。

### 情形 B：仍然是 null（没有来源过 FDR）

**意味着**：**在 n≈128、且是 RL 唯一起作用的那个总体上，
没有证据表明 agentic rubric 是更好的 reward signal。**
这时**必须**这样写，不要软化成"趋势向好"或"需要更多样本"：
128 题已经是 2.1 倍功效，仍然测不出，说明效应即使存在也很小。

**这不是失败的结论，是有价值的结论**：它把资源从"继续优化 rubric 生成"
指向"先搞清楚判别效度是不是正确的代理指标"。
配合 `agentic-noval` 这个反例，结论是**结构性质量显著改善，
但没有证据表明它转化成了更好的 reward signal**。

### 情形 C：best-of-n 仍然是 baseline 最高

**这是最需要认真对待的一条，不要埋在表格里。**
best-of-n 是这一套指标里**离 RL 用法最近**的一个——
GRPO 里 reward 的实际作用就是在一组 rollout 里把好的排到前面。
如果在 n≈128 上 baseline 仍然领先，那就是一个**硬的否定证据**：
agentic 的那些改进没有落到"挑出对的那一个"这件事上。

现状是 baseline 0.816 vs agentic 0.803（n=61，p=0.76，方向不利但不显著）。
**如果方向保持且 n 翻倍后仍然不显著**，如实报告"方向不利、量级很小"；
**如果变得显著**，那么应当**建议不要用 agentic rubric 做 reward**，
并去查是不是条目更多（agentic 10.1 vs baseline 9.5，negonly 12.7）
稀释了关键判据的权重。

### 情形 D：结果内部互相矛盾（比如 AUC 涨但 best-of-n 跌）

**默认相信 best-of-n**，理由同上——它更接近实际用法。
并检查分数分布是否触顶（`headroom`、`saturation`）：
AUC 只看排序，best-of-n 受并列和触顶影响，两者背离通常意味着分数被压在天花板附近。

### 无论哪种情形都要做的四件事

1. **先确认 n 真的是 128。** 生成阶段哪怕少几题，配对队列就会缩小，
   而表格**不会因此报错**——它只会算出一组"看起来正常但总体不同"的数字。
   `run_rollout_v2.sh` 第 3 步会打印 `cohort drift: ...`（只打日志、不中断，
   这样端点再抖一次不至于让整条流水线白跑），
   所以**跑完请 grep 一遍 `logs/` 里的 `cohort drift`**，
   并核对主表 `n` 列。`adaptivity/orphan_subpart_rate` 与
   `coverage/subquestion_coverage` 按构造就只在少数题上有定义，已豁免，不用管。
2. **全部分域报。** 本项目最初把一个纯 science 效应写成了普遍结论，
   而分域数据一直就在产物里。`aggregate_results.py` 与 `rollout_report.py` 现在默认分域。
3. **FDR 族在代码里固定**（`rollout_report.py::FDR_FAMILY`），不要临时改族大小再报 q 值。
4. **标注 n < 30 的行不可用于下结论**（表格里会自动打 ⚠）。

### 5b. RubricBench 之后学到的六条（如果只读一节，读这一节）

接上 §2 表第 18–21 行。这些是**做完外部基准之后**才知道的，代价是若干次白跑。

1. **不要再用结构性指标当 rubric 质量的代理。**
   这不再是「理论上可能不等价」——`agentic` 在本仓库全部 5 个结构性指标上大幅领先，
   在真人偏好上**输了** 0.0079。原因见 §2 表下方那段：
   我们测的全是「判据写得好不好」，没有一个测「判据是不是关于对的东西」。
   **新指标上线前先问一句：它能不能把 `rubric_eval_507` 那条判据标成坏的？** 五个都不能。
2. **地板和天花板必须一起跑，而且要先跑。**
   `none`（不给 rubric）和 `expert`（专家 rubric）各花了不到 30 分钟，
   却是整个 RubricBench 分析里信息量最大的两次运行：
   没有它们，0.5876 看起来像个成绩；有了它们，才知道它只吃掉了 10.7% 的可用空间，
   而且和「什么都不给」在统计上分不开。**任何新基准，先做这两个控制再做方法。**
3. **报效应量之前先算最小可检出差。**
   RubricBench 在 n=1147、约 20% 不一致率下只能分辨 **≥0.027** 的差异，
   而官方排行榜的**全部跨度**只有 0.0148——所以那张榜 0/6 对显著。
   我们自己差点也犯同样的错：`baseline` 相对 `none` 的 +0.0236 低于这个门槛。
   `scripts/rubricbench_verify.py --section leaderboard` 会直接把这个数打出来。
4. **在评测集的失败案例上调 prompt，就等于污染了这个评测集。**
   `framed` 是本项目迄今最有效的干预，也是最不可信的一个数字：
   它的 prompt 是看着 `rubric_eval_480` 写的，而那是评测集内的题。
   补救办法是准留出集（用 prompt 没看过的四个域），效应立刻从 +0.0394 掉到 **+0.0132（不显著）**。
   **下次先切一半，只在前一半上迭代，后一半只跑一次。**
5. **判定器的系统提示是一个会自己造出结论的混淆源。**
   我们的 `_BASE_RULES` 里有一句 "A response that refuses or deflects … is a failure"，
   而 SAFETY 域 43.8% 的题人类偏好的恰恰是拒答那一侧。
   结果：`none` 在「一侧拒答」的 32 题上只有 0.188（无脑押拒答能拿 0.906），
   而 `framed` 至少一半的 SAFETY 收益来自把这句话顶回去。
   **接手第一件事：把这句删掉重跑 `none`/`baseline`/`framed`**，这是最便宜也最能改变结论的实验。
6. **子集结果只能在子集内配对比。**
   `agentic-tools` 只跑了 299 题。官方评测器直接给它 **ACC=0.1578**（缺的 848 题全算错），
   而它的真实值是子集上的 0.6054。同一批 299 题上 `expert` 是 0.7993（比全量高 0.0217）、
   `none` 是 0.5452（比全量低 0.0198）——**这个子集对不同来源的难度偏移方向都不一样**，
   所以跨 n 比较必错。`rubricbench_compare.py` 和 `rubricbench_verify.py` 都强制取交集。

---

## 6. 真实 RL 验证怎么设计

### 6.1 必须先做的数据清洗（不做的话下游数字不可信）

来自 `docs/01_data_forensics.md` §10，在**全量 45k 行**上统计：

1. **RaR-Science 的 test 集有 12.3% 的问题在 train 里出现过。**
   不跨 split 去重就训练 + 评测，测出来的收益里有相当一部分是记忆。
2. **9.15% 的行是重复题面。** 同一题出现多次会让它的 reward 权重被隐式放大。

本仓库的 `sample_examples(deduplicate_questions=True)` 已按题面去重，可直接复用。
**跨 split 去重需要你自己做**（本地只从 val 采样，没碰这个问题）。

### 6.2 三条方法论硬要求

> **(1) 正例必须是 rollout，不能是参考答案。**
> 这是本项目踩得最深的坑（§3）。离线拿参考答案写 rubric 完全合法、RL 部署时也确实如此；
> **不合法的是拿同一段文本当考题**。
> 做法见 `harness/eval/rollouts.py`：独立 cache-salt 命名空间采样、
> 逐字校验与生成器 rollout 不相交（`assert_disjoint_from_generator_rollouts`）、
> 再用一个**看不到任何 rubric** 的 oracle 判对错。
> 如果你的流程里 rubric 生成与 reward 评测共享任何"标准答案"，
> **必须做一次拆半消融**证明收益不来自共享的那一半。成本极低（我们全部复用缓存，0 次新调用），
> 不做的话整套结论都可能被推翻。

> **(2) 只在"结局混合"的题上评估 rubric 质量，并为选题预留预算。**
> 全对/全错的 group 在 GRPO 里 advantage 恒为 0。
> **实测混合率只有 16.4%**（780 题里 128 道），也就是**每 6 道题只有 1 道**
> 对 reward signal 的质量给出信息。按这个比例倒推预算，不要按题数。
> 用 `scripts/screen_questions.py` 先筛后生成：k=4 时每道入选题成本最低
> （46 次调用，k=8 是 63 次），且**在 k=4 混合必然在 k=8 混合**（加样本不会抹掉已有分歧），
> 所以筛选没有假阳性。
> 注意：**别指望靠降低采样质量来制造分歧**——我们试过 prompt 逼模型简答/赶时间，
> 正确率只从 84.9% 降到 79.9%；降 `reasoning_effort` 在这个端点上无效（反而更慢）；
> 压 `max_tokens` 得到的是空回复而不是差回复。**可靠的杠杆只有扩大题池。**

> **(3) Pitfall 极性必须写单元测试。**
> 两个语料的极性约定**相反**（取证 F6）：一个语料里 Pitfall 写成"避免了 X"（满足=好），
> 另一个写成"犯了 X"（满足=坏）。
> **任选一种统一处理，必有一个域被训反**——奖励会奖励错误行为。
> 这类 bug **不会被任何常规 sanity check 抓到**，实测能造成 15pp 的分数污染。
> 本仓库的做法是逐条检测极性（`harness/eval/judge.py::effective_polarity`）+
> 整份 rubric 级别的最有利口径敏感性扫描。
> **接手时请先对 reward 函数的极性处理写单元测试，再开始训练。**

### 6.3 最小可信实验

固定 policy、固定 RL 算法（GRPO 或 PPO）、固定超参，**只换 reward rubric 的来源**。

| 臂 | 为什么要它 |
|---|---|
| `shipped` | 论文原版产物 |
| `baseline` | 论文方法 + 我们的模型（**分离"模型差异"与"方法差异"的关键**） |
| `agentic` | 本研究方法 |
| **`agentic-noval`** | **关键检验点**：结构指标远好于 baseline，判别力却等同 baseline。如果它下游明显更好，说明判别效度不是正确的代理指标，本地评测体系需要重做 |
| `agentic-negonly` | 若预算允许。Stage 6 里**构造上不泄漏**的那一半，且不带天花板副作用。若它追平甚至超过完整 `agentic`，说明本地测到的 gold 侧增益主要是拟合假象——**最直接最便宜的判决性检验** |

只有同时有 `shipped` / `baseline` / `agentic`，才能说清楚"赢"是赢在方法上而不是模型上。

**benchmark**：medicine 用 **HealthBench**（论文用的就是它，且有"纯通用 rubric 把 29.7 砸到 12.5"
这个已知对照点）；science 用 **GPQA-Diamond** 或论文同一套。**务必报告去重后的结果。**

**预期现象**：如果本地判别效度确实预测下游，应看到
`agentic` > `baseline` ≈ `agentic-noval` > `shipped`。

**失败时最可能的原因**（按可能性排序）：

1. **触顶。** agentic 给参考答案打 0.87，只剩 **0.128 余量**，且 **15.8%** 的降级答案已 ≥0.8。
   注意**不是**"好答案之间 advantage 变小"——实测 `good_spread` 反而更大（+0.027, p=0.002）；
   风险是策略学到参考答案水平后 reward 饱和、后续改进拿不到梯度。
   建议做 group-wise reward 归一化/白化并监控触顶比例。
   **这项损害全部来自 gold 侧，`agentic-negonly` 不带这个副作用。**
2. **"自信但错"未被惩罚。** policy 很可能收敛到"过程写得漂亮但答案错"——
   这一档在 agentic rubric 下得分 0.504，与 baseline 无差异。**最需要盯的 hacking 方向。**
3. **极性实现错误。** 见 §6.2(3)。
4. **域依赖。** 所有判别力收益都集中在 science；medicine 上为零甚至反向
   （错误答案的分数-长度相关性在 medicine 上**升高**）。
   如果你的任务更像论述题，**不要指望本地结论迁移**。

---

## 7. 已知的坑

### 7.1 同一个形状的 bug，出现了**四次**：覆盖写截断正在使用的 jsonl

| 次 | 表现 | 损失 |
|---|---|---|
| 1 | `--sources` 单独重跑两个消融时，指标行文件用 `open("w")` 截断写 | 原来四个来源的 **8,000 行判定全没了**，聚合突然变空、不报任何错 |
| 2 | 修第 1 次时只带了 `--metrics discriminative` | grounding / lint / adaptivity 三族**共 18 项指标的逐题数据**丢失，直到很久之后才补回 |
| 3 | 筛选脚本的 `import_existing` 直接写"导入的那批" | **截断了正在进行的筛选检查点**，580 题的标签瞬间只剩 200 题 |
| 4 | k=4→k=8 加采时 `build_all_rollout_sets` 持久化自己的视图 | **会把 780 题的筛选记录砍成 128 题**（由 `scripts/dryrun_resume.py` 在造成损失前抓到） |

**教训：任何持久化"共享产物的局部视图"的函数，默认必须是合并语义。**
现在 `merge_rollout_sets` 和 `_write_rows_preserving_other_sources` 都是合并的，
并且带**写后断言**（任一来源的行数不得减少）。
`scripts/dryrun_resume.py` 会验证这件事——**改了这些路径就重跑它**。

第 3 次的损失可以从 `runs/cache/` 恢复（几秒内恢复了 570 题的标签），
但**丢掉的一个多小时墙钟时间没有缓存**。

### 7.2 可复现性边界

- **judge 的 `temperature` 从未显式设置**，走服务端默认。
  因此**结果只在 `runs/cache/` 在场时可精确复现，而 cache 被 gitignore**（249MB）。
  `EvalConfig.judge_temperature` 字段已加好但默认 `None`（与本次一致）。
  **下一次完整运行应设为 0.0**——代价是作废全部已缓存判定。
- **`max_tokens` 不在缓存键里。** 只差 token 预算的两次调用共用一条缓存，
  因此提高预算**不会**重新拉取一个曾被截断的回复。本次运行零截断，但这对后来者是个真实的坑。
- **`config.json` 快照曾落后于代码**：`from_dict` 会静默丢弃未知键。
  现在遇到未知键会 warn，但**改了配置结构后请核对快照**。

### 7.3 其他

- **shell 曾被并发执行 3 次**，3 个 `gen_rubrics` 同时写同一个 jsonl（文件涨到 598 行）。
  所有长任务都用 `mkdir` 原子锁保护（见 `run_rollout_v2.sh`）。
  锁是目录，进程被 `kill -9` 后不会自动清理——**手动 `rmdir .lock_rollout_v2`**。
- **不要在 `run_rollout_v2.sh` 跑的时候编辑它。** bash 是按字节偏移边读边执行的，
  改动运行中的脚本会让它在下一个阶段跳到错误的位置。要改就先等它结束或停掉它。
- **`reasoning_effort` 在这个端点上基本无效**，medium/low 反而**更慢**
  （245s / 316s vs high 的 34s）且大量空回复。质量梯度靠 prompt 制造。
- **温度 > 1.0 会让模型在推理里空转**不出结果，是 stall 不是"更差的答案"。
- **幸存者偏差**：`pipeline.py` 丢掉任何来源产出空 rubric 的 uid。
  被丢的 2 题正是 **baseline 唯一的两次 JSON 解析失败**（200→198）。
  方向上对 baseline 有利、低估 agentic，但"baseline 在 1% 的题上产不出 rubric"本身就是结果。
  被丢弃的 uid 记在 `runs/<run>/intent_to_treat_failures.json`。
- **论文附录的统计量与公开数据对不上**（Table 4/5/7/8，标注 medicine 的那张几乎精确匹配 science 数据）。
  本报告**没有引用论文附录的任何统计量**，所有对照基线取自我们自己在全量数据上的取证。

---

## 8. 仓库地图

```
rubric_harness/
├─ HANDOFF.md              ← 你在读的这份
├─ README.md               安装、目录、CLI 用法
├─ results/REPORT.md       完整实验报告（证据本体，105KB）
├─ harness/                可复用的库代码
│  ├─ llm.py               带缓存/重试/token 升级/JSON 修复的调用层
│  ├─ generators/          shipped / baseline / agentic 及消融
│  └─ eval/                各指标；rollouts.py 是 RL 忠实协议
├─ scripts/                CLI 入口
│  ├─ screen_questions.py  先筛后生成（结局混合选题）
│  ├─ run_rollout_v2.sh    续跑没做完的实验
│  ├─ resume_when_up.sh    等端点恢复后自动续跑
│  ├─ dryrun_resume.py     不联网排练续跑路径
│  └─ selftest.sh          极性/统计/prompt 逐字一致性
├─ configs/                pilot.yaml（主实验）、rollout_v2.yaml（未跑完的）
├─ runs/                   逐题产物（cache 与逐条 verdict 不入版本控制）
└─ results/                机器可读表 + 报告
```

**从哪读起**：本文件 → `results/REPORT.md` 的 §0 阅读指引 →
按指引跳到你关心的那节。报告很长，但 §0 是为跳读设计的。
