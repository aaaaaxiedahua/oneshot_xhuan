# Module 2: Query-Conditioned Virtual Hub Predictor

## 1. Goal

本模块是对原始预测器的一种小改动创新，目标不是推翻原 GNN，而是解决两个很具体的问题：

1. 采样子图可能不连通，信息传不过去
2. 原 GNN 传播时不知道当前 query 到底在问什么关系

因此，本方案不改动原有采样主流程，也不重写 GNN 层，只在送入 GNN 前，对每个子图增加一个 **query-conditioned virtual hub**。

---

## 2. Original Predictor and Its Problem

### 2.1 Original Paper Behavior

原论文的整体逻辑是：

1. 先为每个 query 采一个小子图
2. 再在这个子图上跑 GNN
3. 最后对候选实体打分

也就是说，预测器本质上是：

- 输入：采样后的子图
- 传播：在子图上多层消息传递
- 输出：给真实实体节点打分

### 2.2 Current Code Patch: `add_manual_edges`

当前代码里还有一个可选工程开关：`add_manual_edges`。

它做的事情不是论文主方法，而是代码里的一个补丁：

- 从 `head -> all sampled nodes` 加一批人工边
- 再从 `all sampled nodes -> head` 加一批人工边

这样做确实能缓解子图断连问题，但问题也很明显：

1. 它是分散式捷径，不是结构化建模
2. 所有节点都直接连 head，拓扑改动过于粗暴
3. 这些人工边本身不携带 query relation 语义
4. 它只是“让图连起来”，不是“让图按 query 传播”

所以，这个补丁更像工程修补，不像一个可单独讲清楚的方法点。

---

## 3. Core Idea of Query Hub

本方案把“给 head 到处拉捷径边”，改成“给每个子图增加一个唯一的虚拟中枢节点”。

这个 hub 节点有两个职责：

1. **Connectivity bridge**
   - 让原本不容易互相通信的节点，能通过 hub 形成稳定的信息通路
2. **Query semantic anchor**
   - hub 的初始表示直接由当前 query relation 决定
   - 这样 hub 在传播一开始就知道“这次要找什么关系”

一句话概括：

> 原来的 `add_manual_edges` 只是给图补路；
> query hub 是给子图加一个“带查询语义的中枢节点”。

---

## 4. Final Method

设某个 query 采样得到的子图为：

\[
\mathcal G_s = (\mathcal V_s, \mathcal E_s)
\]

其中 query 为 \((h, q, ?)\)，`h` 是 head 实体，`q` 是 query relation。

### 4.1 Step 1: Add One Virtual Hub Node

为该子图增加一个新的虚拟节点：

\[
\tilde{\mathcal V}_s = \mathcal V_s \cup \{v_{hub}\}
\]

这个节点不是实体，不参与最终实体候选排名。

### 4.2 Step 2: Add Hub Edges

将 hub 与子图中的真实节点连接。

#### Option A: `hub_rel_mode = directed`

使用两种特殊关系：

\[
\tilde{\mathcal E}_s
=
\mathcal E_s
\cup
\{(v_{hub}, r_{hub}^{out}, v_i) \mid v_i \in \mathcal V_s\}
\cup
\{(v_i, r_{hub}^{in}, v_{hub}) \mid v_i \in \mathcal V_s\}
\]

含义：

- `r_hub_out` 表示 hub 向外广播
- `r_hub_in` 表示节点向 hub 汇聚

#### Option B: `hub_rel_mode = shared`

如果不想区分方向，可只用一个共享特殊关系：

\[
\tilde{\mathcal E}_s
=
\mathcal E_s
\cup
\{(v_{hub}, r_{hub}, v_i)\}
\cup
\{(v_i, r_{hub}, v_{hub})\}
\]

这个版本参数更少，但方向语义更弱。

### 4.3 Step 3: Initialize Hub with Query Semantics

真实节点仍按原模型方式初始化。

如果原模型使用实体初始化，则普通节点仍然是实体特征；如果 query 子节点使用关系初始化，也保持原逻辑不变。

新增的只是 hub 节点的初始表示。

#### Option A: `hub_init = query_relation`

\[
h_{hub}^{(0)} = W_{hub}\, e_q
\]

其中：

- `e_q` 是 query relation embedding
- `W_hub` 是一个轻量映射矩阵

直观理解：

- hub 一开始就带着“这次在问什么关系”的语义

#### Option B: `hub_init = zero`

\[
h_{hub}^{(0)} = \mathbf 0
\]

这个版本用于消融，检验“hub 的提升到底来自结构桥接，还是来自 query relation 注入”。

### 4.4 Step 4: Run the Original GNN on the Augmented Graph

后续消息传递不需要重写一套新 GNN，只要在扩展后的图

\[
\tilde{\mathcal G}_s = (\tilde{\mathcal V}_s, \tilde{\mathcal E}_s)
\]

上运行原有传播层即可：

\[
h_i^{(l+1)} = \operatorname{GNNLayer}\Big(h_i^{(l)}, \mathcal N_{\tilde{\mathcal G}_s}(i)\Big)
\]

这里最关键的变化是：

1. 真实节点可以通过 hub 交换全局信息
2. hub 会把 query relation 语义传播给整张小图
3. hub 也会从所有节点回收上下文，逐层形成全局 summary

### 4.5 Step 5: Readout

最终只对真实实体节点打分，不给 hub 打分。

设 `v` 为候选实体节点，最终打分有两种方式。

#### Option A: `hub_readout = head`

保持原模型风格，用 head 对应表示作为查询锚点：

\[
s(v) = \langle h_v^{(L)}, h_h^{(L)} \rangle
\]

这表示：

- hub 只参与传播
- 最终读出仍然沿用原模型的 head-centered scoring

#### Option B: `hub_readout = hub`

用 hub 的最终表示作为读出锚点：

\[
s(v) = \langle h_v^{(L)}, h_{hub}^{(L)} \rangle
\]

这表示：

- 候选节点是否匹配，不再只看它和 head 的相似性
- 而是看它是否匹配“经过全图聚合后的 query hub 表示”

---

## 5. Why This Is Better Than `add_manual_edges`

| 对比项 | `add_manual_edges` | Query Hub |
| --- | --- | --- |
| 结构形式 | head 对所有节点直接拉边 | 每个子图只有一个 hub |
| 连接方式 | 分散式捷径 | 结构化中枢 |
| 是否带 query 语义 | 没有 | 有，来自 `e_q` |
| 作用方式 | 只是补连通 | 同时做连通桥接和语义广播 |
| 是否适合写成方法点 | 更像工程补丁 | 更像独立模块创新 |

核心区别不是“有没有人工边”，而是：

- `add_manual_edges` 是无语义的粗暴 shortcut
- query hub 是有语义的中心节点建模

---

## 6. Why This Counts as a Good Micro-Innovation

### 6.1 It Fixes a Real Defect

这个方案不是凭空加模块，而是正面解决两个已有缺陷：

1. 小图断连带来的传播受限
2. GNN 传播阶段缺少 query-aware 全局锚点

### 6.2 It Does Not Overturn the Original Framework

它没有改动：

- PPR 采样主框架
- 原始 GNN 层定义
- 损失函数
- 训练主流程

所以这是一个成本低、风险可控的小创新。

### 6.3 It Has a Clear Story

故事线非常顺：

1. 原模型子图可能断连
2. 代码里已有 `add_manual_edges` 补丁，但缺少 query 语义
3. 因此引入一个 query-conditioned virtual hub
4. 让连通性修复和 query 注入在同一个结构里完成

这个叙事比“再拼几个交互项”要干净很多。

---

## 7. Hyperparameters

本模块只保留 4 个新参数，不再引入多余开关。

### 7.1 `use_query_hub`

- 类型：`bool`
- 含义：是否启用 query hub 模块
- 推荐默认：`False`

说明：

- 这是总开关
- 开启后，每个子图都会增加一个 hub 节点

### 7.2 `hub_init`

- 类型：`query_relation | zero`
- 含义：hub 的初始特征怎么来
- 推荐默认：`query_relation`

解释：

- `query_relation`：用 query relation embedding 初始化 hub
- `zero`：用全零向量初始化，用来做消融

### 7.3 `hub_readout`

- 类型：`head | hub`
- 含义：最终给实体打分时，用谁作为读出锚点
- 推荐默认：`head`

解释：

- `head`：保留原模型读出方式，hub 只辅助传播
- `hub`：让最终读出也使用 hub 的全局语义表示

### 7.4 `hub_rel_mode`

- 类型：`directed | shared`
- 含义：hub 边是否区分输入和输出关系
- 推荐默认：`directed`

解释：

- `directed`：`hub -> node` 和 `node -> hub` 用不同特殊关系
- `shared`：双向都使用同一个特殊关系

---

## 8. Configuration Rule

为了避免方法语义冲突，建议加入以下硬约束：

\[
\texttt{use\_query\_hub = True} \Rightarrow \texttt{add\_manual\_edges = False}
\]

也就是说：

- 不能同时开 `use_query_hub=True` 和 `add_manual_edges=True`
- 如果两个都开，程序应直接报错

原因很简单：

1. 两者都在做“额外连通结构”补充
2. 同时开会让实验解释变脏
3. 无法分清性能提升来自 hub 还是来自 manual edges

---

## 9. Minimal Ablation Plan

建议至少做下面几组对比：

1. **Baseline**
   - 原始模型
   - 不开 `add_manual_edges`
   - 不开 `use_query_hub`

2. **Baseline + Manual Edges**
   - 只开 `add_manual_edges`

3. **Baseline + Query Hub**
   - `use_query_hub=True`
   - `hub_init=query_relation`
   - `hub_readout=head`
   - `hub_rel_mode=directed`

4. **Query Hub w/o Query Init**
   - `hub_init=zero`

5. **Query Hub with Hub Readout**
   - `hub_readout=hub`

这样可以直接回答 3 个问题：

1. hub 是否比 manual edges 更有效
2. 提升来自结构桥接，还是来自 query semantic initialization
3. hub 更适合只做传播辅助，还是也适合参与最终读出

---

## 10. Recommended Default Setting

如果把它作为论文里的主推版本，建议默认配置为：

```yaml
use_query_hub: true
hub_init: query_relation
hub_readout: head
hub_rel_mode: directed
```

理由：

1. `query_relation` 最符合这个模块的核心思想
2. `head` 读出最稳，改动最小
3. `directed` 比 `shared` 更容易表达“广播”和“汇聚”两个方向

---

## 11. One-Sentence Positioning

本模块可以概括为：

> 我们用一个由 query relation 初始化的虚拟 hub 节点，替代粗暴的分散式 manual shortcuts，使采样子图在保持原 GNN 主体不变的前提下，同时获得更稳定的连通性和更明确的 query-aware 全局传播锚点。
