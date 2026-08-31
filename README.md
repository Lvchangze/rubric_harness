# Rubric Harness — Agentic Rubric Generation for Rubrics-as-Rewards

> **接手做真实 RL 验证？先读 [`HANDOFF.md`](HANDOFF.md)。**
> 那份是决策摘要 + 操作手册（证据强度分级、没跑完的实验怎么续、踩过的坑）；
> 本文件讲工程结构与 CLI 用法；[`results/REPORT.md`](results/REPORT.md) 是证据本体。
>
> **只想知道这套方法到底管不管用？** 直接看
> [`results/rubricbench/REPORT.md`](results/rubricbench/REPORT.md)（§0b 有摘要）——
> 那是唯一一个**直接**测量 rubric 质量的外部基准，结论与本地代理指标不一致。

本仓库是一个研究 harness，用来验证一个假设：

> **RaR（Rubrics as Rewards）论文中"一次性合成 rubric"的做法质量不够；
> 用 agent 形式做"搜索 + 验证"生成的 rubric，作为 RL reward signal 会更好。**

真正的验证（GRPO 训练 + 下游 benchmark）不在本仓库范围内。本仓库做的是
**本地可完成的、可量化的 rubric 质量代理指标（proxy metrics）**，并把工程做成
别人可以直接接手继续跑的样子。

## 0. 已跑完的结论（`pilot_v2`，每域 100 题）

完整报告：**[`results/REPORT.md`](results/REPORT.md)**；机器可读表：`results/pilot_v2/`。

结论经过**三套证据**：协议一用合成的质量分层 response 梯子（n=198）；
协议二用模型自己采出的、生成器从未见过的真实 rollout（n=61，§4.4）；
以及一次**换 judge 复现**（不同厂商的 `api_azure_openai_gpt-5.1`，n=109，§4.5）。

> ⚠️ **一个没做完的实验**：§4.6 试图把协议二的可用题数从 61 提到 128（先筛后生成），
> **选题阶段完成，评测阶段因服务端点故障中断**。
> 因此"加大样本后协议二是否仍是 null"**本仓库没有答案**。
> 恢复：`bash scripts/run_rollout_v2.sh`（断点续跑，已完成的调用全部命中缓存）。
> **更新**：端点已于 2026-08-28 23:41 恢复，看门狗自动续跑中。
> 查看进度：`python3 scripts/check_report_facts.py`（会打印当前进度 vs 中断时刻快照）。

| | 结果 |
|---|---|
| **无条件支持**（两域一致、零 LLM 调用） | 结构质量：通用条目率 21.0%→**9.5%**、主观率 18.1%→**7.8%**、锚点数 2.19→**2.82**、客观可判定 92.7%→**98.9%**、极性一致性 100% |
| **仅 rar_science 支持** | 判别力（删泄漏正例后）z_separation **+0.325 (q=0.009)**、AUC +0.074；降低长度型 reward hacking（错误答案的分数-长度相关 +0.272→**−0.040**, p=0.044） |
| **换 judge 后仍成立**（§4.5） | 效应量在两个 judge 之间**统计上无法区分**（DiD 交互 CI 全含 0）；**无自我偏好**（扣掉同 judge 地板后 agentic 衰减反而更少，+0.021, p=0.030）；**域间反差完整复现**（science−medicine 差距 +0.254 → +0.252，两者都显著） |
| **⚠️ rar_medicine 上全为零效应** | z_separation +0.021 (**p=0.61**)、AUC +0.007 (**p=0.68**)、ranking accuracy +0.001 (**p=0.95**)；reward hacking 甚至**反向变差** |
| **没有支持假设** | **"排序更准"**——协议一删 gold 后不过 FDR，协议二**没有任何来源在 FDR 后优于 baseline**，best-of-n 选择准确率还是 `baseline` 最高（0.816）；**"用反例筛选 criterion"这个核心卖点**（换成真实失败 rollout 的 `agentic-realneg` 反而更差）；权重信息量；盲评 head-to-head；饱和率与天花板余量 |
| **无法判断** | 下游 RL 性能；**Stage 6 的哪一半有用**——三套证据两票投 gold 侧、一票投 negative 侧，但**那两票用的是同一个偏袒 gold 侧的协议**，不是独立证据；为什么效应全集中在 science；**足够功效下协议二的结果**（§4.6 未跑完） |
| **筛选阶段的副产品** | 用一个强策略采样时，780 题里只有 **16.4%** 结局混合（其余全对或全错）。**全对/全错的 group 在 GRPO 里 advantage 恒为 0**，所以这 16.4% 才是 reward signal 唯一起作用的总体——**先筛后生成**，否则 ~84% 的生成预算白花 |

> **可主张的说法**：在**有唯一正确答案的理科题**上，把 rubric 的每一条拿去实际执行一遍再筛选，
> 能提升它作为 reward signal 的**分辨力**（拉开分距、压低长而错的答案），但**提不了排序准确性**；
> 在**医学论述题**上这些收益全部消失。结构质量的改善则是无条件的，两域一致。
>
> 四条警告：
> **(1) 结构性指标改善 ≠ reward signal 改善**（`agentic-noval` 是干净反例）；
> **(2) rubric 生成用到的参考答案不能同时当评测正例**——这类循环不会被任何常规 sanity check 抓到，
> 只有做拆半消融才能发现；**质量评估的正例应一律换成模型 rollout**（§4.4.1）；
> **(3) 任何 pooled 结论都要先按域拆开看**——本报告最初把一个纯 science 效应写成了普遍结论；
> **(4) 先筛后生成**——只对结局混合的题生成 rubric（`scripts/screen_questions.py`）。

---

## 0b. RubricBench：第一次直接测量，以及它给上面那张表的答案

完整报告：**[`results/rubricbench/REPORT.md`](results/rubricbench/REPORT.md)**；
机器生成的核算：[`results/rubricbench/VERIFY.md`](results/rubricbench/VERIFY.md)。

§0 那张表里最要命的一格是「**结构性指标改善 ≠ reward signal 改善**」——
本地代理指标全线领先，但没有任何证据说它转化成了更好的 reward signal。
[RubricBench](rubricbench/)（1147 组成对比较、人类偏好标签、专家标注 rubric、
**无 reference answer** 所以 gold 泄漏在结构上不可能发生）是第一个能直接回答它的外部基准。

**答案是：没有转化，而且现在知道为什么。**

| source | 是什么 | n | forward ACC | vs `baseline` |
|---|---|--:|--:|---|
| `none` | 不给 rubric（**地板**） | 1147 | 0.5650 | — |
| `agentic` | 本仓库的方法，gold 侧关闭 | 1147 | **0.5798** | Δ −0.0079, q=0.74 ❌ |
| `baseline` | RaR 单次合成（无 reference 变体） | 1147 | 0.5876 | — |
| `framed` | **只改了生成 prompt 的框架** | 1147 | **0.6260** | Δ +0.0394, **q=0.0058** ✅ |
| `expert` | 数据集自带专家 rubric（**天花板**） | 1147 | 0.7777 | Δ +0.1916, q=2.9e-39 ✅ |
| `agentic-tools` | agentic + 可执行工具 | 299 | 0.6054 | 子集内 vs `agentic` +0.0234, q=0.56 ❌ |

三条结论，都要如实读：

1. **`agentic` 在这个基准上没有赢过 `baseline`**（0.5798 vs 0.5876，配对 p=0.63），
   也没有赢过「不给 rubric」（+0.0148，p=0.33）。**§0 的结构性领先确实没有转化。**
2. **§0 的结论依然成立，而本次结果恰恰解释了它为什么没转化。**
   本仓库全部结构性指标测的是判据**写得好不好**（够不够具体、有没有锚点、可不可判定），
   **没有一个测判据是不是关于对的东西**。RubricBench 上 `baseline` 失败的典型形态是：
   instruction 要求写未成年角色的裸体描写，生成的 rubric 就把
   「explicitly describes Carl's butt physique, as requested」列成 4/5 分的得分项——
   这条判据在所有结构性指标上都是满分（具体、可判定、有锚点、非通用），
   但它奖励的是人类明确不偏好的行为。**「写得好」和「关于对的东西」是两件事。**
3. **迄今最有效的干预是改 prompt 框架，不是 agent 流程，也不是工具。**
   `agentic` 花了 18113 次调用换来 +0.0148（不显著）；`agentic-tools` 再加 6028 次工具调用
   换来 +0.0234（不显著）；一次 prompt 改写换来 +0.0610（显著）。

**四条必须一起读的限定条件**（详见报告 §8）：

- `framed` 的 prompt 是**在看过本基准失败案例之后**写的，存在过拟合风险。
  用「其余四域」作准留出集，效应从 +0.0394 掉到 **+0.0132（p=0.33，不显著）**。
- `framed` 净增量的 **68.9% 来自 80 道 SAFETY 题**；把 SAFETY 拿掉后它相对 `baseline` 不显著。
- 换成位置偏置受控的口径（正反两序一致才算数），
  `framed` vs `baseline` 掉到 **q=0.079，不再显著**；vs `none` 仍然稳。
- 我们自己的判定器提示里有一句反拒答的话，SAFETY 的人类标签恰好相反；
  `framed` 至少一半的 SAFETY 收益来自对抗这个**我们自己引入的**偏置。

**顺带一个关于该基准的独立发现**：官方排行榜四个已发表系统两两之间
**0/6 达到显著**（榜首榜尾差 0.0148，p=0.401），而该基准在 n=1147 下的最小可检出差约 **0.027**。
**这张榜排的是 judge，不是 rubric。** 要在上面主张进步，效应量至少得到 0.03 量级。

```bash
# 基准本体是上游仓库，带自己的 .git，因此不纳入本仓库版本控制（见 .gitignore）
git clone http://github.com/planepig/rubricbench.git

python3 scripts/rubricbench_run.py --source framed --limit 0     # 跑一个来源
python3 scripts/rubricbench_verify.py --out results/rubricbench/VERIFY.md
cd rubricbench && python3 eval_submission.py \
    --submission ../results/rubricbench/framed_submission.csv     # 官方评测器
```

---

## 1. 核心设计

### 1.1 三个对比对象（这是本实验设计的关键）

| 来源 | 是什么 | 作用 |
|---|---|---|
| `shipped` | RaR 数据集 parquet 里自带的 `rubric` 字段 | 论文原版产物（o3-mini / GPT-4o 生成） |
| `baseline` | 用**论文原版 prompt** + **我们的模型**单次合成 | 复现 RaR 方法 |
| `agentic` | 本研究的多阶段 agent pipeline，**同一个模型** | 我们的新方法 |

`shipped` 和 `baseline` 的差别 = **模型差异**；
`baseline` 和 `agentic` 的差别 = **方法差异**。
只有同时有这三者，才能说清楚"赢"是赢在方法上而不是赢在模型上。

另有消融 `agentic-noval`：关掉 Stage 6 验证环（`AgenticConfig.enable_critic=False`），
用来单独证明"验证在环"这个组件本身有效。

> **`baseline` 用的是论文附录逐字原文，且按 domain 分派两份不同的 prompt。**
> medicine 版给了权重数值锚点并强制 Pitfall 以 `"Does not mention …"` 开头，science 版两者都没有
> （`docs/00_paper_notes.md` §2.3）。用同一份 prompt 跑两个 domain 会让"模型差异 / 方法差异"
> 的解耦失效，所以 `scripts/check_prompt_fidelity.py` 会在每次 pilot 开跑前逐字比对，
> 不一致就直接退出，不浪费预算。

### 1.1.1 Pitfall 极性：本 harness 最容易被忽略的正确性问题

取证发现两个 domain 的 Pitfall 极性约定是**相反的**（`docs/01_data_forensics.md` F6）：
science 有 80.3% 是"避免式"（`Avoids X`，满足=好），medicine 有 88.1% 是"失败式"
（`Does not mention X`，满足=差），但**两者权重都是 −1/−2**。

所以**不能按权重符号或 category 推断极性**。本 harness 的做法：

1. `Criterion` 带一个显式的 `polarity` 字段；shipped/baseline 的极性由
   `harness/schema.detect_polarity()` 逐条判定（正则与 `analysis/collect_examples.py`
   逐字一致，已验证能**精确复现**全量 45k 语料的 80.33% / 88.11% 分布）。
2. judge **只问"这句话对该回答是否字面成立"**，不问好坏；好坏方向在聚合时由 polarity 施加。
   这样 judge 端不存在双重否定的歧义。
3. 对照组（`shipped` / `baseline`）按 `PolarityMode.FAVOURABLE` 计分，即**对它们最有利**的
   极性解释，agentic 若还能赢就是硬赢。
4. 报告里给**极性敏感性分析**：同一批 judge 判定按 `detected` / `all_avoidance` /
   `all_failure` 各聚合一遍（复用同一批判定，**不额外花 LLM 预算**），确认结论不依赖极性口径。

**pilot_v2 实测**（`results/REPORT.md` §4.2.1）：极性口径的选择会让 `baseline` 的 gold 分数
在 0.570（全按避免式）到 0.726（全按失败式）之间移动，**摆动 15.6 个百分点**；`shipped` 摆动 3.8pp。
这远大于任何方法间 margin 的量级。

更值得注意的是：我们检查了 4 个来源 × 4 种极性口径共 16 种组合，
**"gold 分数 > off_topic 分数"这个 sanity check 在全部 16 种组合下都通过**。
也就是说，一个把整个 domain 的 Pitfall 算反了的实现，仍然会通过常规的"好答案得分更高"检查，
却已经在 15pp 的尺度上污染了 reward。**做 RL 的同学必须对 reward 函数的极性处理写单元测试**，
不能依赖粗粒度 sanity check。

### 1.2 公平性保证（写进代码，不是靠自觉）

- **同一批 response**：降级 response 只生成一次（`runs/<run>/responses.jsonl`），
  三个 rubric 来源共用。
- **同一个 judge**：`harness/eval/judge.py` 是唯一入口，同模型、同 prompt、同参数。
- **顺序随机化**：criterion 展示顺序按 `(uid, response_id)` 打乱，**种子里不含
  rubric 来源**，所以没有任何来源能从位置偏置里占便宜。
- **来源盲评**：judge prompt 里从不出现 rubric 的出处。
- **配对样本**：`harness/pipeline.load_all_rubrics` 只保留三个来源都成功生成的 uid，
  所有统计都是 paired 的。
- **题目去重**：RaR-Science 有 9.15% 的行是重复题目、test 有 12.3% 与 train 重叠
  （取证 §10）。采样时按题面去重，避免同一题被当成两个独立观测。
- **三种聚合口径可切换**（`EvalConfig.aggregation`）：
  | 模式 | 权重 | 用途 |
  |---|---|---|
  | `paper_explicit`（默认，主结果） | category 重映射 `{E:1.0, I:0.7, O:0.3, P:0.9}` | **忠于论文**：RaR-EXPLICIT 训练时并不用数据里的数值权重 |
  | `raw_weight` | 原始 1–5 整数权重 | 稳健性检查 |
  | `uniform` | 等权 | 稳健性检查 |

  取证显示 shipped 的权重体系与等权平均的 Pearson 相关高达 **0.94**（F5），
  即权重几乎不携带信息。所以我们对每个 generator 都报"权重与等权的相关性"：
  agentic 应当明显更低 **且**判别效度更高，两个数字合起来才说明权重真的有用。

  > **pilot_v2 实测：这个目标没有达成。** 用真实 judge 判定比较加权分数与等权分数，
  > 四个来源的相关性分别是 shipped 0.983 / baseline 0.989 / agentic-noval 0.985 / **agentic 0.989**——
  > agentic 的权重**没有**比对照更有信息量。因此 §3.1 的判别力提升全部来自
  > **criterion 本身写得更好**，与权重分配无关。后续同学不必在权重设计上投入。
  > 详见 `results/REPORT.md` §5.1。
- **负权重不进分母**：Eq. (1) 的权重一律取绝对值，好坏方向由 polarity 承载。
  shipped 数据里 91.8% / 95.6% 的题含负权重，直接代入会让分母比 Σ|w| 小 9–14%
  且缩幅逐题不同（取证 §9.4），破坏"归一化使 reward 跨题可比"的设计意图。

---

## 2. 安装与环境

```bash
cd /apdcephfs_zwfy6/share_302970870/hunyuan/changzelv/dev/rubric_harness
pip install -r requirements.txt        # pandas / pyarrow / numpy / PyYAML
export LLM_REQUEST_TIMEOUT_S=1200      # 必须设置，否则长任务会被 600s 硬超时打断
```

LLM 通过仓库外的 in-tree 包访问，`harness/llm.py` 会自动把它加进 `sys.path`：

```
/apdcephfs_zwfy6/share_302970870/hunyuan/changzelv/dev/hyctx_data
```

数据放在 `data/{rar_science,rar_medicine}/{train,val,test}-00000-of-00001.parquet`。

---

## 3. 快速开始

```bash
export LLM_REQUEST_TIMEOUT_S=1200

# 全流程，分两个阶段（推荐；每个 stage 都可断点续跑）
nohup bash scripts/run_pilot_driver.sh generate > runs/pilot_v2/driver_gen.log 2>&1 &
# 生成完成后：
nohup bash scripts/run_pilot_driver.sh evaluate > runs/pilot_v2/driver_eval.log 2>&1 &
```

> `run_pilot_driver.sh` 用 `mkdir` 原子锁保证**同一阶段只会有一个实例在跑**。
> 这不是多余的谨慎：本次实验中观察到执行环境会把同一条 shell 命令并发启动多次，
> 导致三个 `gen_rubrics` 同时向同一个 append-only jsonl 写入（文件涨到 598 行 / 200 题）。
> 产物可以用 `python scripts/_repair_jsonl.py runs/<run>/rubrics_*.jsonl` 按 uid 去重修复。
> **不要在启动命令里 `rmdir` 这个锁**——那会重新引入它要防的竞态。

或者分步跑：

```bash
# 0) 检查 baseline prompt 是否仍与论文附录逐字一致（不花钱，但决定 baseline 列的可信度）
python scripts/check_prompt_fidelity.py

# 1) 冒烟：每域 4 题，跑通全链路（约 10 分钟）
python scripts/gen_rubrics.py     --config configs/smoke.yaml
python scripts/build_responses.py --config configs/smoke.yaml
python scripts/eval_rubrics.py    --config configs/smoke.yaml
python scripts/aggregate_results.py --run-name smoke

# 2) Pilot：每域 100 题
python scripts/gen_rubrics.py     --config configs/pilot.yaml
python scripts/gen_rubrics.py     --config configs/pilot.yaml --sources agentic-noval   # 消融

# 零 LLM 的结构性指标，先跑：不花钱，且能对照取证的 n=45k 基线验证抽样是否有代表性
# 偏离若大到不像抽样噪声（如通用率跑到 50%），说明抽样或指标实现有问题，就地停下来修
python scripts/eval_rubrics.py    --config configs/pilot.yaml --metrics grounding lint adaptivity
python scripts/report_facts.py    --run-name pilot_v2        # 打印与取证基线的逐项对照

python scripts/build_responses.py --config configs/pilot.yaml
python scripts/eval_rubrics.py    --config configs/pilot.yaml --metrics all

# 条数受控重跑：确认收益不是来自堆条目（写到 cc_* 前缀的独立产物，不覆盖主结果）
python scripts/eval_rubrics.py    --config configs/pilot.yaml --count-controlled \
                                  --metrics discriminative coverage grounding

# 主表：--expect-paired 是硬断言，配对队列一旦被新来源改变就报错（见 §6.2）
python scripts/aggregate_results.py --run-name pilot_v2 --expect-paired 198
python scripts/aggregate_results.py --run-name pilot_v2 --prefix cc_   # 条数受控表
python scripts/main_table.py        --run-name pilot_v2                # 报告引用的核心对比
python scripts/case_study.py        --run-name pilot_v2 --rank-by agentic_advantage --top 2 \
                                    --out results/pilot_v2/case_study_wins.md

# 泄漏审计（协议一）：分数分解 / 删 gold 重算 / 拆半消融，全部含 BH-FDR 与分域
python scripts/confound_audit.py    --run-name pilot_v2 --reference baseline --by-domain

# 协议二：面向 rollout 的 RL 忠实评测（正例是模型 rollout，不是 reference answer）
python scripts/build_rollouts.py    --config configs/pilot.yaml --k 6 --top-up 6 --concurrency 128
python scripts/eval_rubrics.py      --config configs/pilot.yaml --metrics rollout \
    --sources shipped baseline agentic-noval agentic agentic-goldonly agentic-negonly agentic-realneg
python scripts/rollout_report.py       --run-name pilot_v2 --reference baseline
python scripts/reward_hacking_probe.py --run-name pilot_v2
```

长任务请后台跑并盯日志：

```bash
nohup bash scripts/run_pilot.sh > runs/pilot_v2/pilot.log 2>&1 &
tail -f runs/pilot_v2/pilot.log
```

**所有 LLM 调用都有磁盘缓存**（`runs/cache/`，按 prompt+model+params 的 sha256 寻址），
所以中断后重跑几乎不花钱、且结果逐位一致。生成与 response 构建还额外支持
按 uid 断点续跑（`--no-resume` 可关闭）。

---

## 4. 目录结构

```
rubric_harness/
├── harness/                  # 可复用的 Python 包
│   ├── schema.py             # Criterion / Rubric / Example / Category（统一数据结构）
│   ├── llm.py                # LLMEngine：dict/str 兼容、空 response 重试、JSON 修复、
│   │                         #   磁盘缓存、指数退避、并发信号量、调用统计
│   ├── data.py               # parquet 加载、可复现采样、shipped rubric 解析
│   ├── config.py             # RunConfig / LLMConfig / SampleConfig / AgenticConfig / EvalConfig
│   ├── tracing.py            # RunDir、JSONL writer、config 快照、trace 落盘
│   ├── pipeline.py           # 编排：样本解析、生成、断点续跑、配对加载
│   ├── prompts/
│   │   ├── rar_original.py   # 论文附录 A.9 的**原版 prompt**（逐字转录）
│   │   ├── agentic.py        # agentic pipeline 各 stage 的 prompt
│   │   ├── responses.py      # 受控降级 response 的 prompt
│   │   └── eval_prompts.py   # 各评测指标的 prompt
│   ├── generators/
│   │   ├── base.py           # RubricGenerator 接口 + 注册表
│   │   ├── shipped.py        # 直接读取数据集 rubric（0 次 LLM 调用）
│   │   ├── single_pass.py    # RaR 复现：单次调用
│   │   └── agentic.py        # 本研究方法：7 阶段 pipeline
│   └── eval/
│       ├── judge.py          # 唯一的 judge：逐条 yes/no + RaR Eq.(1) 加权聚合
│       ├── responses.py      # 质量分层 response 阶梯（共用）
│       ├── discriminative.py # 指标族 1：判别效度（主结果）
│       ├── transfer.py       # 指标族 2：query-specificity / 可迁移性
│       ├── coverage.py       # 指标族 3：覆盖度与 grounding
│       ├── intrinsic.py      # 指标族 4：内在质量
│       ├── headtohead.py     # 指标族 5：盲评偏好（辅助证据）
│       └── stats.py          # bootstrap CI、配对置换检验、Wilcoxon、kappa、AUC、BH-FDR
├── scripts/                  # argparse CLI 入口
│   ├── gen_rubrics.py            # 生成 rubric（shipped/baseline/agentic + 4 个消融）
│   ├── build_responses.py        # 构建共用的质量分层 response 阶梯（协议一）
│   ├── build_rollouts.py         # 采样全新 rollout + 独立 oracle 判正确性（协议二）
│   ├── eval_rubrics.py           # 跑评测指标；--count-controlled 做条数受控重跑
│   ├── aggregate_results.py      # 汇总表（pooled + 分域）+ FDR + 极性敏感性；--expect-paired 断言
│   ├── main_table.py             # 报告引用的核心对比表（可在评测跑到一半时使用）
│   ├── confound_audit.py         # gold 泄漏审计：分数分解 / 删 gold 重算 / 拆半消融 / reward 几何
│   ├── rollout_report.py         # 协议二主表：AUC / best-of-n / Spearman，含 FDR 与分域
│   ├── reward_hacking_probe.py   # 真实失败 rollout 上的"分数 vs 长度"，按题聚类 bootstrap
│   ├── report_facts.py           # 运行级事实：h2h、成本、取证基线对照、极性摆动
│   ├── case_study.py             # 单题 rubric 并排对比（含 Stage 6 验证判定）
│   ├── check_prompt_fidelity.py  # baseline prompt 与论文附录逐字比对（pilot 前必跑）
│   ├── _repair_jsonl.py          # 按 uid 去重 append-only 产物（并发写入后的修复）
│   ├── selftest.sh               # 零 LLM 健康检查（几秒），改完代码先跑这个
│   ├── dryrun_resume.py         # 不联网排练续跑路径：合并写、断点、配对队列
│   ├── check_report_facts.py    # 核对报告里的关键数字与产物是否一致
│   ├── screen_questions.py      # 先筛后生成：只保留结局混合的题
│   ├── run_rollout_v2.sh        # 续跑没做完的 rollout 评测（六个阶段）
│   ├── resume_when_up.sh        # 等端点恢复后自动续跑
│   ├── run_pilot_driver.sh       # 带原子锁的全流程驱动：generate / evaluate
│   ├── rubricbench_run.py        # RubricBench：跑一个 rubric 来源，出 score/verdicts/submission
│   ├── rubricbench_compare.py    # RubricBench：7 来源总表 + 配对 McNemar + 空间利用率
│   └── rubricbench_verify.py     # RubricBench：脱离 score.json 独立重算全部主张；含 --section selftest
├── rubricbench/              # 上游基准（1147 题 + 官方评测器 + 4 个已发表提交），只读
├── harness/rubricbench.py    # 适配层：load_cases / judge_all / score / write_submission
├── configs/                  # smoke.yaml / pilot.yaml
├── runs/<run_name>/          # 运行产物
│   ├── config.json           #   配置快照
│   ├── examples.jsonl        #   本次运行的样本（一次采样，各阶段共用）
│   ├── rubrics_<source>.jsonl
│   ├── responses.jsonl       #   共用的降级 response
│   ├── traces/<source>_<uid>.json   # agent 各阶段完整中间轨迹
│   ├── metrics/*.jsonl       #   逐样本指标行
│   ├── metric_summaries.json
│   └── llm_stats_*.json      #   调用数 / 缓存命中 / 空响应 / 解析失败
├── results/<run_name>/       # 汇总表（summary.csv / summary.json / summary_table.md）
├── results/REPORT.md         # 本地代理指标的实验报告
└── results/rubricbench/      # RubricBench 产物
    ├── REPORT.md             #   外部基准报告（§0b 的完整版）
    ├── VERIFY.md             #   机器生成的独立核算，9 节
    ├── COMPARE.md            #   7 来源总表 + 配对检验
    └── <tag>_{score.json,verdicts.jsonl,rubrics.json,submission.csv}
```

`docs/` 和 `analysis/` 由另一位同学维护（论文精读与数据取证），本 harness 只读不写。

---

## 5. Agentic pipeline

核心思想：**criterion 不是"写"出来的，是"搜索 + 验证"出来的。**

| Stage | 名称 | 做什么 | 开关 |
|---|---|---|---|
| 1 | `decompose` | 解析题目 → 子问题、已知量+单位、待求量、隐含约束、领域、答案格式 | `enable_decompose` |
| 2 | `rollouts` | **不看 reference answer**，独立求解 k 次（默认 k=3，各自独立 cache salt + 高温采样） | `enable_rollouts` |
| 3 | `reconcile` | k 次 rollout 对齐 reference：`consensus_correct`（低区分度）/ `divergent`（高区分度）/ `collective_error`（**最有价值**）/ `reference_only` | `enable_reconcile` |
| 4 | `pitfalls` | 挖掘本题特有的"看起来对但其实错" | `enable_pitfalls` |
| 5 | `draft` | 合成候选 criterion（原子、可验证、query-specific、带 provenance） | — |
| 6 | `critic` | **验证在环**：gold 必须 pass，至少一个 negative 必须 fail；gold 不过 → 丢弃，全部 negative 都过 → 降权 | `enable_critic` |
| 7 | `calibrate` | 语义去重合并、按"区分度 × 重要性"标定 weight 与类别、格式化 | `enable_calibration` |

RaR 的 rubric 是从 reference answer 里"抄"步骤；我们通过**多次独立求解**发现模型
真正会在哪里分歧/出错，分歧点就是最有区分度的 criterion 候选；再用 Stage 6 把
rubric 从"生成物"变成"通过了可执行判别测试的筛选结果"。

每个 stage 的输入输出都写进 `runs/<run>/traces/`；任一 stage 失败都会降级到上一阶段
的结果继续，不会丢样本。

### 5.1 工具调用（`agentic-tools`）

上面那条流水线里，**每一步问什么都是编排器定的**——模型只负责回答。严格说它是
multi-stage workflow，不是 tool-using agent。`agentic-tools` 补上了这一层：把可执行
工具交给模型，由它自己决定要验证什么。

新增两个阶段，其余 stage、prompt、阈值全部继承不变，所以
**`agentic-tools` 减去 `agentic` 隔离出的就是"让模型自己取证"的效果**：

| Stage | 名称 | 做什么 | 开关 |
|---|---|---|---|
| 4b | `investigate` | 在 draft 之前，模型自主调工具建立证据：重算参考答案的数值、把草稿判据拿去跑真实文本、测试判据是否只适用于本题 | `enable_tool_investigation` |
| 8b | `critic_tools` | 在确定性 critic 之后，对幸存判据再做一轮**可执行**复核，只能改写措辞或删除 | `enable_tool_critic` |

`critic_tools` 刻意做成**追加**而不是替换：原来的 critic 产出 calibrate 依赖的
`validation` 记录，也是 `goldonly`/`negonly` 消融的定义所在，改写它会让
`agentic-tools` 与 `agentic` 同时差两件事，哪件都归因不了。

八个工具，分两类。**五个零 LLM 调用、可从仓库完全复现**：

| 工具 | 作用 |
|---|---|
| `python_eval` | 沙箱 Python（子进程隔离、CPU/内存 rlimit、无网络、无进程派生；预装 numpy/sympy） |
| `check_equivalence` | 两个表达式或数值是否等价（sympy 符号化 + 数值容差），避免判据把答案钉死在一种写法上 |
| `check_units` | 量纲分析与单位匹配 |
| `check_specificity` | 锚点、主观用词、风格规则、批内近重复，以及**这条判据还能套到多少道无关题上** |
| `find_similar_questions` | 语料近邻检索 |

**三个走模型**：`extract_reference_claims`（参考答案拆成原子 claim）、
`execute_criterion`（把判据真的跑在文本上，报告是否成立）、
`make_counterexample`（造错误答案，并**验证它确实是错的**才注册）。

`execute_criterion` 复用评测侧的 `JUDGE_SYSTEM`，`extract_reference_claims` 复用
`CLAIM_EXTRACTION_SYSTEM`——生成期用的量具和评测期是同一把。

判据靠 `targets` 按名字指定要跑在哪个文本上：`reference`、本次的 `rollout_N`、
critic 阶段的 `negative_*`，以及模型自己造出来的 `counterexample_*`。

```bash
python3 scripts/gen_rubrics.py --config configs/pilot.yaml --run-name tools_pilot \
  --sources shipped baseline agentic agentic-tools --concurrency 48
```

配置项都在 `AgenticConfig`：`enable_tools`、`tool_names`、`tool_max_rounds`（默认 8）、
`tool_max_parallel`、`tool_timeout_s`、`sandbox_timeout_s`、`tool_corpus_limit`。

**每一次工具调用都进 trace**（参数、结果、耗时、错误），落在
`runs/<run>/traces/` 和 rubric 的 `meta.tools`。这不是装饰：一条说"已验证"的判据，
如果背后没有对应的工具调用，它和普通的模型断言没有区别。`investigate` 若一次工具
都没调，trace 里会显式标 `unverified`。

实测特性（单题，science 域，`n_rollouts=2`，`tool_max_rounds=8`）：

- 约 1350 秒、31 次 LLM 调用、17 次工具调用，**远贵于无工具的 `agentic`**。成本几乎
  全在 `execute_criterion` 和 `make_counterexample` 这两个走模型的工具上（各 45–85 秒）。
- 两个循环都会**用满轮次预算**（`stop_reason=max_rounds_closed`）。轮次耗尽时会
  撤掉工具、要求模型直接交付 JSON，所以预算超支只是截断调查，不会整段作废——
  没有这个收尾机制时，`investigate` 会直接失败并丢掉已经付过钱的工作。
- 该次运行里模型发现了 **2 处参考答案本身的问题**，并主动记录了 2 条
  "考虑过但不具区分度、因而没写成判据"的检查。

### 5.2 结果：工具没有改善 rubric

跑完了，`tools_v1`，128 道混合结局题，四个来源配对。完整报告：
**[`results/tools_v1/REPORT_TOOLS.md`](results/tools_v1/REPORT_TOOLS.md)**。

> **`agentic-tools` 与 `agentic` 在所有指标上都无法区分，best-of-n 方向还不利。**

| rollout 协议（参照 `agentic`，n=128） | `agentic` | `agentic-tools` | Δ | q |
|---|--:|--:|--:|--:|
| AUC(分数, oracle 正确性) ↑ | 0.768 | 0.777 | +0.009 | 0.73 |
| Best-of-n 选择准确率 ↑ | 0.824 | **0.796** | **−0.027** | 0.55 |
| 分数间隔（正确−错误）↑ | 0.236 | 0.221 | −0.016 | 0.55 |

没有一格通过 FDR，方向在两域间互相矛盾。结构性指标 18 项里 17 项无差异。真实
reward-hacking 探针上，无工具那一臂的优势（−0.173, p=0.018）加了工具后反而变弱（−0.093, p=0.18）。

**这是真实的否定结果，不是实现失败**：每题 20.3 次工具调用，124/128 的 `investigate`
全部真正用了工具（零「未验证」），工具 critic 删了 196 条判据、改写 60 条。

最可能的解释是**结构性空间已被占满**——确定性的 `lint` 阶段本来就强制了 grounding、极性和
去主观化，通用条目率已经从 `baseline` 的 0.188 压到 0.079，工具没有剩余空间可改。

一个值得单独提的副产品：工具在 **108/128 道题上标出了参考答案本身的问题**（共 165 处）。
即使对 rubric 没帮助，这个能力对**数据集质量审计**是直接有用的。

---

## 6. 评测指标

指标分两类。**零 LLM 成本的结构性指标**直接复用取证已建好的检测器
（`analysis/rubric_grounding.py` / `common.py`），因此 `shipped` 那一列已经有
**n=45,325 题的全量基线**可以对照 —— 我们只需要对 `baseline` / `agentic` 算同一个量，
三列就直接可比，且几乎不花预算。**LLM 判定的指标**才是预算大头。

| 族 | 指标 | 方向 | 说明 |
|---|---|---|---|
| 0 结构性（零 LLM） | `generic_criterion_rate` | ↓ | 不含本题任何锚点的 criterion 占比。shipped 全量：**28.9% / 22.0%**（置换对照 96.4% / 98.5%，判别力 67 个百分点） |
| | `weight_mass_on_generic_paper` | ↓ | **最贴近 reward 的一条**：按 Eq.(1) 权重计，落在通用条目上的 reward 质量。shipped 全量：**24.6% / 15.2%** |
| | `subjective_or_style_rate`, `weight_mass_on_subjective` | ↓ | 主观/风格条目及其 reward 质量占比。shipped：39.6% / 23.6% 与 40.2% / 21.3% |
| | `compliance_rate` | ↑ | 论文 prompt 硬约束合规率（`docs/00_paper_notes.md` §2.4 十条） |
| | `polarity_single_rate` | ↑ | 整份 rubric 是否统一极性（agentic 目标 100%，即 F6 的修复） |
| | `pitfall_mirror_rate` | ↓ | F7：Pitfall 只是把某条 Essential 取反重述，同一事实被双重计分。shipped medicine **16.2%** |
| | `weight_order_violation_rate` | ↓ | Optional 权重 ≥ Important。shipped science **41.3%** 的题违反，medicine 0% |
| | `items_cv`, `r_items_vs_*` | ↑ | F1：条数是否随复杂度自适应。shipped science 是常数 7（CV=0.107，r=0.19） |
| 1 判别效度（主结果） | `mean_margin`, `min_margin` | ↑ | gold 与各降级档的分数间隔 |
| | `pairwise_ranking_accuracy` | ↑ | 能否把已知更好的 response 排前面 |
| | `auc_good_vs_bad`, `auc_gold_vs_degraded` | ↑ | |
| | `score_range`, `score_std` | ↑ | 分数动态范围；坏 rubric 会把所有答案都打 0.8 |
| | `score_verbose_empty` | ↓ | **reward-hacking 探针**：冗长但空洞 |
| | `score_right_method_wrong_answer` | ↓ | **reward-hacking 探针**：自信但错误 |
| | `saturation_rate` | ↓ | 降级 response 拿到 ≥0.8 的比例 |
| 2 query-specificity | `cross_pass_rate` | ↓ | A 题 rubric 评 B 题 gold 的通过率 |
| | `specificity_gap` | ↑ | 同题通过率 − 跨题通过率 |
| | `boilerplate_fraction` | ↓ | 在所有无关答案上都 pass 的 criterion 占比 |
| 3 覆盖 / grounding | `claim_recall` | ↑ | gold claim 被覆盖比例（claim 只抽一次，三源共用） |
| | `grounded_fraction` | ↑ | criterion 对应到 reference 实质内容的比例 |
| | `unsupported_fraction` | ↓ | criterion 断言了 reference 里没有的东西 |
| | `subquestion_coverage` | ↑ | 多小问题目的子问题覆盖率 |
| 4 内在质量 | `objective_fraction` | ↑ | 仅凭 response 就能客观判 yes/no |
| | `self_agreement_rate`, `kappa` | ↑ | 双次判定一致率（判定不稳 = RL 里的噪声） |
| | `mean_checks_per_criterion` | ↓ | 原子性（越接近 1 越好） |
| 5 盲评偏好 | `win_rate`（位置偏置校正后） | ↑ | 仅作辅助证据 |

> **为什么没有"题内冗余度"这一条**：取证**证伪了**这个假设。朴素 TF-IDF 看着 medicine
> 有 57.8% 的题存在高相似条目对，但那多是"平行但不同"的条目（"每 3 个月随访" vs
> "每 6 个月随访"）。加入实体/数字判别过滤后真实冗余率只有 **7.9% / 0.8%**。
> 唯一真实的冗余形态是 Pitfall 镜像（F7），已并入 `lint`。省下的预算加到了主结果的样本量上
> （每域 80 → 100 题）。

### 6.1 两个必须做的混淆控制

0. **gold 泄漏审计（最重要，`scripts/confound_audit.py`）**。agentic 的 Stage 6 用
   `example.reference_answer` 筛选 criterion，而判别效度的 `gold` 档就是同一段文本——
   Stage 6 直接优化了主指标奖励的那个量。审计做三件事：把 margin 拆回**绝对分数**看是
   "坏答案被压低"还是"所有分数一起抬高"；把 `gold` 档**整个删掉**只在 generator 从未见过的
   降级档之间重算判别力；把 Stage 6 拆成 **gold 侧（泄漏）/ negative 侧（干净）** 分别消融
   （`agentic-goldonly` / `agentic-negonly`，由 `use_gold_signal` / `use_negative_signal` 控制）。

   ```bash
   bash scripts/run_pilot_driver.sh leakage      # 生成两个消融 + 判定 + 出审计报告
   python scripts/confound_audit.py --run-name pilot_v2 --reference baseline
   ```

   > **实测结论**：优势**部分存活**——删掉泄漏正例后 z_separation 仍 +0.134 (q=0.018)，
   > 但排序类指标（AUC、ranking accuracy）不再显著；且增益几乎全部来自**泄漏的 gold 侧**，
   > 干净的 negative 侧没有任何指标通过 FDR。**任何复用本 harness 的人都必须先跑这一项。**

1. **条数受控对比**。如果 agentic 只是因为条目更多而在覆盖率上赢，那不算赢。
   `harness/eval/adaptivity.truncate_to_common_size()` 把所有来源截断到逐题相同条数后
   重跑关键指标（产物加 `cc_` 前缀，不覆盖主结果）。
   > **实测**：截到相同条数（各 1456 条）后 agentic 的优势**普遍变大**
   > （margin +0.113 → **+0.138**，排序准确率 +0.030 → **+0.047**）。收益不是来自堆条目，
   > 而是**前 k 条质量更高**。
2. **极性敏感性**。见 §1.1.1，同一批判定按四种极性口径各聚合一遍（复用判定，不额外花钱）。
   > **实测**：给每个对照组挑对它最有利的口径后，agentic 仍领先（margin 0.694 vs 0.556 / 0.560）。

统计口径：所有比较都是 **paired**（同一批题），报告均值、bootstrap 95% CI、
配对置换检验 p 值，并对整张表做 **BH-FDR 校正**。`harness/eval/stats.py` 只依赖 numpy。

---

## 7. 怎么扩展

**加一个新 generator**

```python
# harness/generators/my_method.py
from .base import GenerationResult, RubricGenerator, register
from ..schema import Criterion, Rubric

@register
class MyGenerator(RubricGenerator):
    name = "my_method"

    async def generate(self, example):
        criteria = [...]                      # 构造 Criterion
        rubric = Rubric.from_criteria(criteria, source=self.name, n_items=len(criteria))
        return GenerationResult(uid=example.uid, source=self.name, rubric=rubric,
                                trace={...}, n_llm_calls=...)
```

在 `harness/generators/__init__.py` 里 import 一次即可注册；然后
`python scripts/gen_rubrics.py --sources my_method`。
**失败要软降级**：返回尽力而为的部分 rubric 并写 `error`，不要抛异常。

**加一个新 metric**

1. 新建 `harness/eval/my_metric.py`，实现
   `async def run_my_metric(engine, examples, rubrics_by_source, *, config, run_dir=None)`，
   返回一个带 `.per_question_rows`（每行含 `uid` 与 `rubric_source`）和 `.summary()` 的对象。
   评分一律走 `harness.eval.judge.judge_rubric`，不要自己写 judge。
2. 在 `scripts/eval_rubrics.py` 的 `METRIC_MODULES` 里登记；不调 LLM 的话
   一并加进 `ZERO_LLM_METRICS`，它们会在花预算之前先跑。
3. 在 `scripts/aggregate_results.py` 的 `METRIC_SPEC` 里加上列名和方向（↑/↓）。
   若该指标是分布层面的量（没有逐题对应值），加进 `DISTRIBUTION_SPEC`。

CLI 会用 `inspect.signature` 只传该 metric 声明了的参数，所以新 metric 不必接受全部入参。

**做一个新消融**：在 `scripts/gen_rubrics.py` 的 `ABLATIONS` 字典里加一项
（键 = 来源名，值 = 覆盖到 `AgenticConfig` 的字段），无需新文件。现有三个：

| 来源 | 覆盖字段 | 用途 |
|---|---|---|
| `agentic-noval` | `enable_critic=False` | 整个 Stage 6 关掉 |
| `agentic-goldonly` | `use_negative_signal=False` | 只保留 Stage 6 中**泄漏**的 gold 侧 |
| `agentic-negonly` | `use_gold_signal=False` | 只保留 Stage 6 中**干净**的 negative 侧 |

> 注意 `use_gold_signal` 是**总开关**：Stage 6 的 gold 侧有三条编辑路径
> （`drop_contradicted`、`drop_gold_fail`、gold 不过则降权），其中两条原本没有任何开关。
> 只 gate `drop_gold_failures` 并不能真正关掉 gold 侧。

**注意 `--sources` 的语义**：指标行文件按 `rubric_source` 合并——只替换本次评测的来源，
保留其余来源既有的行。所以为一个新消融单独跑评测是安全的，不会清掉别人的结果。

---

## 8. 已知的坑（实测）

- 该 reasoning model 的 `chat()` 返回 **dict** `{'reasoning_content', 'response'}`，
  必须取 `response`；`harness/llm.py:extract_response_text` 已统一处理 str/dict。
- `max_tokens` 给不够时 reasoning 会吃光预算、`response` 返回**空字符串**。
  `LLMEngine` 检测到空响应会自动把预算 ×1.5 后重试。
- 必须 `export LLM_REQUEST_TIMEOUT_S=1200`，否则默认 600s 硬超时。
- 模型爱把 JSON 包在 ```json fence 里或前面加一句话；`extract_json` 会做
  fence 剥离 / 括号配平 / 去尾逗号，仍失败则让模型自己重发一遍纯 JSON。
- 并发实测：8/24/40/64/96 路均 0 报错，吞吐约 0.39/0.99/1.59/2.04/2.92 call/s
  （单调用延迟 ~20–33s 基本不随并发上升）。pilot 用 64 留了余量。

### 8.1 评测设计上的坑（比工程坑更危险）

- **不要按权重符号推断 Pitfall 极性。** 两个 domain 的约定是反的，但权重都是 −1/−2。
  实测：极性口径选错会让 `baseline` 的 gold 分数摆动 **15.6 个百分点**（shipped 3.8pp），
  足以凭空造出一个"胜利"。**而且这类 bug 不会被"好答案得分更高"这种 sanity check 抓到**——
  我们检查的 16 种（来源 × 口径）组合全部通过该检查。见 §1.1.1。
- **不要给两个 domain 用同一份 baseline prompt。** 论文附录的两份 prompt 不一样，
  且差异恰好解释了两个语料的主要分歧。`scripts/check_prompt_fidelity.py` 会拦截。
- **不要直接比覆盖率而不控制条数。** 条目更多自然覆盖更多。用 `--count-controlled`。
- **不要用 test split 且不去重。** RaR-Science 的 test 有 12.3% 的题在 train 里出现过，
  9.15% 的行是重复题面。默认用 val 且按题面去重。
- **不要为了凑条数生成 Optional 条目。** shipped 的 Optional 类通用率高达 52%/44%，
  这些条目对 judge 几乎恒为满足，在 GRPO 里方差趋近 0，纯粹稀释真正有区分度的梯度。
- **不要把"题内语义冗余"当卖点** —— 取证已证伪（真实冗余率仅 0.8%/7.9%）。

更多实测细节见 `results/REPORT.md`。
