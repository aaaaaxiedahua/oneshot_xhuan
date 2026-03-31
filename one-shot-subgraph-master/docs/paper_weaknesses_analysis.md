# One-shot Subgraph 原论文缺陷分析

> 论文：Less is More: One-shot Subgraph Reasoning on Large-scale Knowledge Graphs (ICLR 2024)

---

## 一、论文方法概述

论文将知识图谱链接预测解耦为两步：

1. **采样器 $g_\phi$**：用 Personalized PageRank (PPR) 为查询 $(u, q, ?)$ 提取一个 query-dependent 子图 $\mathcal{G}_s$
2. **预测器 $f_\theta$**：在子图 $\mathcal{G}_s$ 上用 GNN 做消息传递和打分

核心公式：

- PPR 迭代：$p^{(k+1)} = \alpha \cdot s + (1-\alpha) \cdot D^{-1}A \cdot p^{(k)}$
- 消息传递：$h_o^{(\ell+1)} = \text{Dropout}\big(\text{Act}(\text{Agg}\{\text{Mess}(h_x^{(\ell)}, h_r^{(\ell)}, h_o^{(\ell)}) : (x,r,o) \in \mathcal{E}_s\})\big)$
- 子图提取：$\mathcal{V}_s \leftarrow \text{TopK}(\mathcal{V}, p, K=r_V^q \times |\mathcal{V}|)$

---

## 二、缺陷分析

### 缺陷 1：PPR 将知识图谱退化为同质图，完全丢弃关系语义

**论文相关位置：** Section 4.1, Eq. 2

**问题描述：**

PPR 算法的前提是在**同质无向图**上做随机游走。论文在应用 PPR 前，将知识图谱 $\mathcal{G}=(\mathcal{V}, \mathcal{R}, \mathcal{E})$ 转化为同质图 $G=(\mathcal{V}, E)$，转化规则为：

$$E = \{(h, t) \mid \exists r, (h, r, t) \in \mathcal{E}\}$$

这一步将所有关系类型的边合并为一条无类型的边。**知识图谱区别于普通图的核心特征就是边具有不同的关系语义**（如 `father_of`、`born_in`、`works_at`），PPR 在转化过程中将这一核心信息完全抹除。

**具体后果：**

对于同一个查询实体 $u$，无论查询关系 $q$ 是什么，PPR 计算的采样分布 $p$ 完全相同。也就是说：

- 查询 $(Alice, \text{mother\_of}, ?)$
- 查询 $(Alice, \text{works\_at}, ?)$

这两个语义完全不同的查询，会采出**完全相同的子图**。两个完全不同的问题，用完全相同的证据去回答。

**论文的忽视：**

论文在 Section 4.1 声称 PPR 是 *"query-dependent"*，但实际上 PPR 只是 **entity-dependent**，不是 **relation-dependent**。论文没有讨论这一区别，也没有分析不同关系类型下 PPR 采样质量的差异。

---

### 缺陷 2：One-shot 采样不可纠错，存在不可恢复的信息损失

**论文相关位置：** Section 3, Definition 1; Section 5, Table 3

**问题描述：**

论文将 "one-shot" 作为方法论的核心优势（Advantage 3），强调只需一个子图即可回答一个查询。但 one-shot 同时意味着**采样过程没有纠错机制**——如果 PPR 在采样阶段遗漏了关键节点或关键证据路径，后续的 GNN 预测器无论能力多强，都无法恢复被丢失的信息。

**论文自身数据的佐证：**

论文 Table 3 报告了不同采样启发式的 Coverage Ratio（答案实体是否在子图中）：

| 数据集 | PPR, $r_V^q=0.1$ | PPR, $r_V^q=0.2$ | PPR, $r_V^q=0.5$ |
|--------|-------------------|-------------------|-------------------|
| WN18RR | 87.6% | 89.6% | 92.9% |
| NELL-995 | 96.5% | 97.7% | 98.7% |
| YAGO3-10 | 72.8% | 76.0% | 84.8% |

以 YAGO3-10 为例，$r_V^q=0.1$ 时 **27.2% 的正确答案不在子图中**。这些查询不管模型多好都无法正确预测，是方法论层面的硬上限。

**与其他方法的对比：**

| 方法 | 采样策略 | 纠错能力 |
|------|---------|---------|
| AdaProp (KDD 2023) | 每层 GNN 动态采样 | 有，逐层根据推理状态修正采样范围 |
| A*Net (NeurIPS 2023) | 启发式搜索逐步扩展 | 有，按需扩展搜索边界 |
| **One-shot-subgraph** | **PPR 一次性采样** | **无，采样错误不可恢复** |

论文在 Section 3 讨论了与 layer-wise sampling 和 subgraph-wise sampling 的效率对比，但**没有讨论 one-shot 在采样质量上的代价**。

---

### 缺陷 3：Coverage Ratio 不等于推理质量，评估指标有盲区

**论文相关位置：** Section 5, Table 3, Figure 3

**问题描述：**

论文使用 Coverage Ratio (CR) 作为衡量采样质量的指标：

$$\text{CR} = \frac{1}{|\mathcal{E}^{test}|} \sum_{(u,q,v) \in \mathcal{E}^{test}} \mathbb{1}\{v \in \mathcal{V}_s\}$$

CR 衡量的是**答案实体是否在子图中**。但答案在子图中 ≠ 模型能正确推理到答案。

**关键忽视——证据路径覆盖率：**

链接预测不仅需要答案实体在子图中，还需要从查询实体 $u$ 到答案实体 $v$ 的**推理证据路径**也在子图中。

举例：查询 $(A, \text{grandfather\_of}, C)$

- 正确推理路径：$A \xrightarrow{\text{father\_of}} B \xrightarrow{\text{father\_of}} C$
- 假设子图包含 $A$ 和 $C$（CR=100%），但中间节点 $B$ 不在子图中
- 此时覆盖率满分，但推理不可能成功，因为证据路径断裂了

**论文没有定义或报告"证据路径覆盖率"（Evidence Path Coverage Ratio）。** 仅凭 CR 无法判断 PPR 采样是否真正保留了推理所需的完整证据。

---

### 缺陷 4：PPR 的局部性参数 α 对所有关系一刀切

**论文相关位置：** Section 4.1, Eq. 2

**问题描述：**

PPR 公式中的 damping coefficient $\alpha$（论文默认 $\alpha=0.85$）控制随机游走的局部性偏置：$\alpha$ 越大，PPR 分数越集中在查询实体 $u$ 的近邻；$\alpha$ 越小，分数越均匀地扩散到远端。

**论文对所有关系类型使用同一个 $\alpha=0.85$。** 但不同关系的正确答案分布在不同的跳数距离上：

| 关系类型 | 典型推理距离 | 对 α 的需求 |
|---------|-----------|-----------|
| `has_part`、`member_of` | 1 hop，直接关系 | 大 α（强局部性） |
| `grandfather_of`、`uncle_of` | 2 hops，关系组合 | 中等 α |
| `nationality`、`language_spoken` | 3-4 hops（born_in → located_in → country） | 小 α（广扩散） |

固定的 $\alpha$ 导致：
- 对短距离关系：采样了过多不相关的远端节点（噪声）
- 对长距离关系：扩散不够远，遗漏了关键的中间节点和答案实体

**论文没有分析不同关系下 PPR 采样质量的差异**，也没有按关系类型报告 CR 或 MRR 的细分结果。

---

### 缺陷 5：预测器对子图边界无感知，无法区分截断与稀疏

**论文相关位置：** Section 4.1 Step-3, Eq. 4

**问题描述：**

GNN 在子图 $\mathcal{G}_s$ 上做消息传递时，将子图视为一个自包含的完整图。但子图是从原始 KG 中采样得到的，子图边界处的节点存在系统性的信息截断：

- 一个节点 $v$ 在原始 KG 中可能有 300 个邻居
- PPR 采样后，子图中可能只保留了 $v$ 的 5 个邻居
- GNN 在聚合 $v$ 的邻居信息时，只看到 5 个邻居

**GNN 无法区分两种本质不同的情况：**

1. $v$ 在原始 KG 中确实只有 5 个邻居（真实稀疏节点）
2. $v$ 在原始 KG 中有 300 个邻居，但 295 个被采样丢弃了（信息截断节点）

这两种情况下，GNN 接收到的输入完全一致，但节点表示 $h_v$ 的语义含义完全不同。情况 1 中 $h_v$ 是完整的，情况 2 中 $h_v$ 是严重残缺的。

**论文没有讨论这种边界效应**，也没有任何机制让预测器感知信息截断的程度。

---

### 缺陷 6：GNN 只能隐式捕获关系路径组合，效率低且受层数限制

**论文相关位置：** Section 4.1 Step-3, Appendix B Table 10, Appendix D.4

**问题描述：**

知识图谱推理的核心能力之一是**关系组合推理**（relational composition）：

- $\text{grandfather} = \text{father} \circ \text{father}$
- $\text{uncle} = \text{parent} \circ \text{brother}$
- $\text{nationality} = \text{born\_in} \circ \text{located\_in} \circ \text{country\_of}$

论文的 GNN 通过多层消息传递来**隐式**学习这些组合模式。第 $\ell$ 层的消息传递聚合 1-hop 邻居信息，因此理论上 $L$ 层 GNN 能捕获 $L$-hop 范围内的关系组合。

**但隐式捕获存在根本局限：**

1. **层数瓶颈：** 捕获 $N$-hop 关系组合至少需要 $N$ 层 GNN。论文 Table 4 的实验显示，8 层 GNN 效果最好，10 层不再提升甚至下降，说明 over-smoothing 限制了可用层数。这意味着模型难以学习超过 8 步的关系组合。

2. **学习效率低：** 隐式方法需要从大量训练样本中"悟出"关系组合规律。模型并不知道 `grandfather = father + father`，它只能通过梯度下降从数据中间接学习这种模式。

3. **缺乏可解释性：** 隐式学到的关系组合嵌入在高维向量中，无法直接解读模型学到了哪些组合规则。

**论文的自我认知：**

论文在 Appendix D.4 (Extension) 中自己提到了这个局限，将 *"relation composition"* 列为未来工作方向，但没有在正文中给出具体的解决方案或分析。

---

### 缺陷 7：双层优化的两步搜索不保证联合最优

**论文相关位置：** Section 4.2, Eq. 5, Figure 2(b)

**问题描述：**

论文将超参数优化分为两个子问题（Eq. 5）：

$$\phi^*_{\text{hyper}} = \arg\max_{\phi_{\text{hyper}}} \mathcal{M}(f_{(\theta^*_{\text{hyper}}, \theta^*_{\text{learn}})}, g_{\phi^*_{\text{hyper}}}, \mathcal{E}^{val})$$

$$\theta^*_{\text{hyper}} = \arg\max_{\theta_{\text{hyper}}} \mathcal{M}(f_{(\theta_{\text{hyper}}, \theta^*_{\text{learn}})}, g_{\phi^*_{\text{hyper}}}, \mathcal{E}^{val})$$

实际搜索过程（Figure 2(b)）是：

1. 先固定采样器 $g_\phi$（即固定 $r_V^q, r_E^q$），搜索预测器的最优超参数 $\theta^*_{\text{hyper}}$
2. 再固定预测器 $f_{\theta^*}$，搜索采样器的最优超参数 $\phi^*_{\text{hyper}}$

**这种交替搜索的问题：**

- 最优子图大小依赖于预测器的表达能力：强模型能从更大、更嘈杂的子图中提取有用信息；弱模型需要更小、更干净的子图
- 最优预测器架构依赖于子图大小：小子图需要深层网络做远距离传播；大子图可能用浅层网络就够了
- **两步交替搜索可能陷入局部最优：** 第一步找到的 $\theta^*$ 是基于当前 $\phi$ 的条件最优，但当 $\phi$ 改变后，$\theta^*$ 可能不再最优

论文没有分析这种两步搜索与联合搜索之间的 gap 有多大。

---

## 三、缺陷严重程度总结

| 严重程度 | 缺陷编号 | 缺陷描述 | 影响范围 |
|---------|---------|---------|---------|
| 致命 | 缺陷 1 | PPR 丢弃关系语义，采样与查询关系无关 | 所有数据集、所有查询 |
| 严重 | 缺陷 2 | One-shot 不可纠错，存在覆盖率硬上限 | 尤其影响 YAGO（CR=72.8%） |
| 严重 | 缺陷 3 | CR 不反映推理质量，评估有盲区 | 论文的核心评估指标不完整 |
| 中等 | 缺陷 4 | α 固定，不适应不同关系的推理距离 | 长距离关系受影响最大 |
| 中等 | 缺陷 5 | 预测器不感知子图边界截断 | 边界节点表示失真 |
| 中等 | 缺陷 6 | 隐式路径推理受 over-smoothing 限制 | 深层网络退化 |
| 轻微 | 缺陷 7 | 两步优化非联合最优 | HPO 结果可能次优 |

---

## 四、后续方向

以上 7 个缺陷为后续创新提供了明确的切入点。每个缺陷都可以独立设计解决方案，也可以组合多个缺陷提出统一的改进框架。具体的解决方案将在单独的文档中给出。
