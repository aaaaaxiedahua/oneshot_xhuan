# Module 2: Progressive Query Updater

## 1. Goal

本方案希望解决一个很具体的问题：

- 当前子图 GNN 在所有层都使用同一个静态 query 向量
- 但多跳推理本质上是逐层展开、逐层修正关注重点的过程
- 因此，深层传播继续使用初始 query，会出现“静态 query 指导动态推理”的不匹配

PQU 的核心目标是：

- 把 query 从“固定条件”改成“逐层更新的推理状态”
- 让每层 GNN 的注意力都能使用当前阶段的 query state
- 让模型具备“边传播、边修正目标”的能力

---

## 2. Current Model Limitation

当前模型的 GNN attention 里，query 是静态注入的。

对应代码位置：

- `GNNLayer.forward`
- `h_qr = self.rela_embed(q_rel)[r_idx]`

也就是说：

1. 每个 query relation `q_rel` 会映射成一个 embedding
2. 这个 embedding 会在所有层重复使用
3. 第 1 层和第 8 层看到的是同一个 query 表示

当前 attention 形式可以写成：

\[
\alpha_{uv}^{(l)} = f\big(h_u^{(l)}, e_r, e_q\big)
\]

其中：

- \(h_u^{(l)}\) 是源节点特征
- \(e_r\) 是边关系特征
- \(e_q\) 是固定的 query relation embedding

问题在于：

- 第 1 层更像是“从 head 朝哪些关系方向展开”
- 第 3~8 层更像是“在已经看到一部分关系上下文后，下一步该关注什么”
- 这两种阶段对 query 的使用方式不应该完全相同

因此，当前模型存在一个结构盲点：

- **传播深度在变化**
- **query 条件却完全不变**

---

## 3. Core Idea

PQU 的核心思想是：

- 每一层根据当前层实际聚合到的关系上下文
- 去更新下一层要使用的 query state

因此：

- 第 1 层使用初始 query
- 第 2 层使用“吸收了第 1 层上下文”的 query
- 第 3 层使用进一步更新后的 query

整个流程从：

```text
static query -> all layers
```

改成：

```text
q^(0) -> layer 1 -> update -> q^(1) -> layer 2 -> update -> q^(2) -> ...
```

这不是机械地把 query 拆词，而是：

- 让 query 作为一个**动态推理状态**
- 在传播过程中持续吸收“当前走到哪里了”的关系信息

---

## 4. Final PQU Scheme

### 4.1 Initialize Query State

对第 \(b\) 个 query，初始化：

\[
q_b^{(0)} = e_{q_b}
\]

其中：

- \(e_{q_b}\) 是 query relation 的初始 embedding

这一点和当前模型一致，区别只在于后续不再固定不变。

### 4.2 Layer-wise Query-Aware Attention

在第 \(l\) 层，对边 \((u, r, v)\)，attention 改为依赖当前层的 query state：

\[
\alpha_{uv}^{(l)}
=
\sigma \Big(
w^\top
\mathrm{ReLU}(W_s h_u^{(l)} + W_r e_r + W_q q_b^{(l)})
\Big)
\]

这里：

- \(q_b^{(l)}\) 替代了原始固定的 \(e_q\)
- 不同层会使用不同的 query state

对应 message 仍保持原结构：

\[
m_{uv}^{(l)} = \alpha_{uv}^{(l)} \cdot (h_u^{(l)} \odot e_r)
\]

这样做的好处是：

- 不用重写整套 message passing
- 只改 attention 中 query 的使用方式

### 4.3 Relation Context Aggregation

第 \(l\) 层传播结束后，用这一层所有边的关系信息聚合一个关系上下文向量：

\[
c_b^{(l)} =
\frac{
\sum_{e \in \mathcal{E}_b} \alpha_e^{(l)} \, e_{r_e}
}{
\sum_{e \in \mathcal{E}_b} \alpha_e^{(l)} + \epsilon
}
\]

其中：

- \(\mathcal{E}_b\) 是第 \(b\) 个 query 的子图边集合
- \(\alpha_e^{(l)}\) 是该边在第 \(l\) 层的 attention
- \(e_{r_e}\) 是该边关系 embedding

直观上：

- 如果本层模型重点依赖了某些关系边
- 那这些关系 embedding 的 attention 加权平均就构成了当前层的“关系上下文摘要”

### 4.4 Progressive Query Update

利用当前 query state 和本层关系上下文，更新下一层 query：

\[
q_b^{(l+1)}
=
\mathrm{LayerNorm}
\left(
q_b^{(l)}
+
\mathrm{MLP}\big([q_b^{(l)} \parallel c_b^{(l)}]\big)
\right)
\]

说明：

1. `MLP` 负责计算 query 的更新量
2. 残差连接保证不会完全抹掉原始 query
3. `LayerNorm` 用于稳定训练

这一步是整个 PQU 的核心。

### 4.5 Next Layer Uses Updated Query

下一层 attention 不再使用初始 query，而是使用更新后的：

\[
q_b^{(l+1)}
\]

如此循环，直到最后一层结束。

---

## 5. Step-by-Step Pipeline

完整步骤如下：

1. 用当前 query relation 初始化 \(q^{(0)}\)
2. 第 \(l\) 层 GNN 使用 \(q^{(l)}\) 计算边 attention
3. 根据本层 edge attention 聚合关系上下文 \(c^{(l)}\)
4. 用小型 MLP 更新 query state，得到 \(q^{(l+1)}\)
5. 第 \(l+1\) 层继续使用新的 query state
6. 最后一层结束后，readout 保持原样

这意味着 PQU 只改“传播条件”，不改最终打分头。

---

## 6. What Changes in the Original Model

### 6.1 Original Model

原模型中：

\[
\alpha_{uv}^{(l)} = f(h_u^{(l)}, e_r, e_q)
\]

其中 \(e_q\) 在所有层固定不变。

### 6.2 With PQU

加入 PQU 后：

\[
\alpha_{uv}^{(l)} = f(h_u^{(l)}, e_r, q^{(l)})
\]

同时新增两条方程：

\[
c_b^{(l)} =
\frac{
\sum_{e \in \mathcal{E}_b} \alpha_e^{(l)} e_{r_e}
}{
\sum_{e \in \mathcal{E}_b} \alpha_e^{(l)} + \epsilon
}
\]

\[
q_b^{(l+1)}
=
\mathrm{LayerNorm}
\left(
q_b^{(l)}
+
\mathrm{MLP}\big([q_b^{(l)} \parallel c_b^{(l)}]\big)
\right)
\]

所以本质变化是：

- **原模型只有静态 query 注入**
- **PQU 增加了层间 query state 更新**

---

## 7. Why This Modification Is Reasonable

### 7.1 Multi-hop reasoning is staged

多跳推理不是一步完成的。

随着层数增加：

- 模型已经看到的关系上下文在变化
- 节点状态在变化
- 下一层真正需要的 query 条件也应变化

因此，继续使用固定 query 不够合理。

### 7.2 The current attention already depends on query

当前模型 attention 本来就依赖 query。

因此 PQU 不是强行新增一个毫不相关的模块，而是：

- 延续原模型的 query-aware attention 设计
- 进一步把 query 从静态条件升级为动态状态

### 7.3 It avoids a second scoring branch

PQU 不会像某些额外模块那样：

- 单独再开一套 logits
- 与主模型 readout 抢最终分数

它只作用于 attention 条件，所以更稳。

### 7.4 The extra cost is small

PQU 每层只增加：

- 一个关系上下文聚合
- 一个很小的 MLP
- 一个 LayerNorm

相比新增独立推理分支，这个开销很低。

---

## 8. Code-Level Modification Plan

如果后续实现，主要改动点如下。

### 8.1 `GNNLayer`

当前 `GNNLayer.forward` 内部通过 `q_rel` 查 relation embedding：

```python
h_qr = self.rela_embed(q_rel)[r_idx]
```

实现 PQU 后，更合适的方式是：

- 上层直接传入当前的 `q_state`
- 每条边根据 `r_idx` 拿到对应 query 的当前层状态

即改成类似：

```python
h_qr = q_state[r_idx]
```

### 8.2 Main Forward Loop

在 `GNN_auto.forward` 中：

1. 初始化 `q_state = query_rela_embed(q_rel)`
2. 每层运行 GNN 时使用当前 `q_state`
3. 拿到该层 edge attention 后聚合 `c_rel`
4. 用 updater 得到新 `q_state`

### 8.3 New Query Updater Module

新增模块可采用共享参数形式：

```python
query_updater: Linear(2d -> h) -> ReLU -> Linear(h -> d)
query_norm: LayerNorm(d)
```

不建议每层独立一套 updater，因为：

- 参数增长没有必要
- 实验更难控
- 小创新最好保持轻量

---

## 9. Suggested Hyperparameters

建议只加 2 个新参数：

- `--use_progressive_query`
- `--query_update_hidden`

推荐默认值：

- `use_progressive_query = False`
- `query_update_hidden = hidden_dim`

如果要极简，也可以固定：

- `query_update_hidden = hidden_dim`

只暴露一个开关参数。

---

## 10. Advantages

PQU 的主要优点：

1. 改动集中在 attention 条件层
2. 不改采样器，不改最终 readout
3. 不引入独立打分分支，稳定性更高
4. 计算开销远低于额外记忆模块或 path-level 分支
5. 学术故事清楚：从 static query 到 progressive query state

---

## 11. Risks

实现时要注意两点：

1. 如果 query update 太强，可能把原始 query 语义冲掉  
   所以建议保留 residual update 和 LayerNorm

2. 如果每层 updater 独立、参数过多，容易过拟合  
   所以建议先用一套共享 updater

---

## 12. One-Sentence Summary

PQU 的核心是：

- **让每层 GNN 使用的 query 不再固定，而是根据上一层实际聚合到的关系上下文动态更新。**

这使得模型从“静态 query 条件匹配”提升为“动态目标驱动推理”。
