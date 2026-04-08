# Module 1: ComplEx-Initialized Relation-Conditioned Weighted PPR Sampler

## 1. Goal

本方案的目标是把当前模型的第一阶段检索从：

\[
\text{head-only PPR}
\]

改成：

\[
\text{head + relation aware weighted PPR}
\]

核心要求有三个：

1. 保留原始 PPR 的全局扩散能力和覆盖率
2. 让 query relation \(r_q\) 真正参与第一阶段小图检索
3. 尽量少改后续 GNN 主体

本方案建议使用：

1. `ComplEx` 预训练关系嵌入
2. `relation-conditioned weighted PPR` 做第一阶段检索
3. 主任务训练时对 sampler 关系参数进行小学习率联合微调

---

## 2. Current Model Limitation

当前仓库的第一阶段 sampler 逻辑是：

1. 在无关系同构图上预计算每个 head 的 PPR 分数
2. 查询时读取 \( \operatorname{PPR}(v \mid h) \)
3. 直接取 topk 节点
4. 在这些节点上抽边形成子图

这个过程的问题是：

1. 第一阶段只依赖 head \(h\)，不依赖 query relation \(r_q\)
2. sampler 使用的是无关系图，关系语义在第一阶段被丢掉了
3. 现有 `RelationRefine` 只是在粗图内再次裁剪，不能补回第一阶段漏掉的答案

因此，本方案的创新点不是“粗图后精排”，而是：

\[
\text{直接改写第一阶段扩散算子}
\]

---

## 3. Core Idea

原始 PPR 的随机游走转移是固定的：

\[
\mathbf p^{(t+1)}
=
\gamma \mathbf e_h
+
(1-\gamma)\mathbf P^\top \mathbf p^{(t)}
\]

其中：

- \(\mathbf e_h\) 是以 head 为起点的 one-hot 向量
- \(\mathbf P\) 是固定图上的转移矩阵
- \(\gamma\) 是 restart 系数

本方案把固定转移矩阵 \(\mathbf P\) 改成与 query relation \(r_q\) 有关的转移矩阵：

\[
\mathbf p_q^{(t+1)}
=
\gamma \mathbf e_h
+
(1-\gamma)(\mathbf P^{(q)})^\top \mathbf p_q^{(t)}
\]

其中：

\[
\mathbf P^{(q)}
\]

不是固定图结构，而是根据当前查询关系 \(r_q\) 动态构造的加权转移矩阵。

这意味着第一阶段检索从：

\[
\operatorname{PPR}(v \mid h)
\]

变成：

\[
\operatorname{PPR}(v \mid h, r_q)
\]

---

## 4. Why Use ComplEx Embeddings

我们需要一套关系语义向量去决定：

\[
\text{当前 query relation 下，哪些类型的边更值得扩散}
\]

因此需要先把每个关系 \(r\) 映射成向量：

\[
r \mapsto \mathbf e_r
\]

本方案建议优先使用 `ComplEx`，原因如下：

1. `ComplEx` 可以表达反对称关系和逆关系
2. 对 `WN18RR` 这类数据集通常比 `DistMult` 更合适
3. 工程实现比 `RotatE` 更轻
4. 它提供的是关系语义空间，不直接替代 PPR，只作为边权先验

`ComplEx` 的三元组打分函数为：

\[
f(h,r,t)
=
\operatorname{Re}\left(
\langle \mathbf e_h, \mathbf e_r, \overline{\mathbf e_t} \rangle
\right)
\]

训练结束后，保留关系向量 \(\mathbf e_r\) 即可。

如果实现中不希望引入复数运算，可把关系向量写成实部与虚部拼接形式：

\[
\tilde{\mathbf e}_r
=
\left[
\operatorname{Re}(\mathbf e_r);
\operatorname{Im}(\mathbf e_r)
\right]
\]

后续 weighted PPR 中统一使用 \(\tilde{\mathbf e}_r\)。

---

## 5. Step 0: Pretrain Relation Embeddings

在正式训练主模型前，先在训练图上进行一次轻量 KGE 预训练。

### 5.1 Training Data

训练三元组：

\[
\mathcal T = \{(h,r,t)\}
\]

建议保留：

1. 原始关系 \(r\)
2. 反向关系 \(r^{-1}\)
3. 不把自环关系作为 KGE 主训练目标

### 5.2 KGE Objective

设 `ComplEx` 打分为：

\[
f(h,r,t)
=
\operatorname{Re}\left(
\langle \mathbf e_h, \mathbf e_r, \overline{\mathbf e_t} \rangle
\right)
\]

可使用标准负采样损失：

\[
\mathcal L_{\text{kge}}
=
- \log \sigma(f(h,r,t))
- \sum_{(h',r,t') \in \mathcal N^-}
\log \sigma(-f(h',r,t'))
\]

### 5.3 Output

预训练结束后得到一套关系向量表：

\[
E^{\text{kge}}
=
\{\tilde{\mathbf e}_r \mid r \in \mathcal R \cup \mathcal R^{-1}\}
\]

这套向量用于初始化 sampler 的关系语义空间。

---

## 6. Step 1: Build Query-Aware Edge Scores

对图中每条边 \((u,r,v)\)，定义它在当前查询 \((h,r_q,?)\) 下的边权打分：

\[
a_q(u,r,v)=\phi(\tilde{\mathbf e}_r,\tilde{\mathbf e}_{r_q})
\]

这里的 \(\phi\) 是关系兼容性打分函数。

### 6.1 Recommended First Version

第一版推荐使用双线性打分：

\[
a_q(u,r,v)
=
\tilde{\mathbf e}_r^\top W \tilde{\mathbf e}_{r_q}
\]

优点：

1. 参数少
2. 容易稳定训练
3. 适合作为第一版实现

### 6.2 Stronger Version

如果后续需要更强表达力，可改成 MLP 型：

\[
a_q(u,r,v)
=
\mathbf w^\top
\tanh
\left(
W_1 \tilde{\mathbf e}_r + W_2 \tilde{\mathbf e}_{r_q} + \mathbf b
\right)
\]

这个式子的含义很简单：

1. \(\tilde{\mathbf e}_r\) 表示边关系的语义
2. \(\tilde{\mathbf e}_{r_q}\) 表示当前查询关系的语义
3. \(a_q\) 越大，说明这类边在当前 query 下越值得被随机游走优先经过

---

## 7. Step 2: Convert Edge Scores to Query-Aware Transition Probabilities

对于源节点 \(u\) 的所有出边，做局部归一化：

\[
\alpha_q(u,r,v)
=
\frac{\exp(a_q(u,r,v))}
{\sum_{(u,r',v')\in \mathcal N^+(u)} \exp(a_q(u,r',v'))}
\]

其中：

- \(\mathcal N^+(u)\) 表示从 \(u\) 出发的所有有向边
- \(\alpha_q(u,r,v)\) 是当前查询下从 \(u\) 走到 \(v\) 的概率分配项

然后定义 query-aware 转移矩阵：

\[
P^{(q)}_{u\to v}
=
\sum_{r : (u,r,v)\in \mathcal E} \alpha_q(u,r,v)
\]

注意：

1. 如果 \(u \to v\) 之间存在多种关系边，需要把它们的贡献加起来
2. 当前第一版不建议再把节点特征引入边权，先只用关系语义

---

## 8. Step 3: Run Relation-Conditioned Weighted PPR

初始化：

\[
\mathbf p_q^{(0)} = \mathbf e_h
\]

迭代更新：

\[
\mathbf p_q^{(t+1)}
=
\gamma \mathbf e_h
+
(1-\gamma)(\mathbf P^{(q)})^\top \mathbf p_q^{(t)}
\]

直到收敛，得到：

\[
\mathbf p_q
=
\operatorname{PPR}(h,r_q)
\]

与原始 PPR 的关键区别是：

1. 原始 PPR 使用固定 \(\mathbf P\)
2. 本方案使用由 query relation 决定的 \(\mathbf P^{(q)}\)

因此最终分数是：

\[
s_q(v)=\mathbf p_q(v)
\]

它明确依赖于查询关系 \(r_q\)。

---

## 9. Step 4: Retrieve the Coarse Subgraph

根据 weighted PPR 分数取第一阶段节点集合：

\[
V_k(h,r_q)
=
\operatorname{TopK}_{v \in \mathcal V}\ s_q(v)
\]

然后构造诱导子图：

\[
E_k(h,r_q)
=
\{(u,r,v)\in\mathcal E \mid u\in V_k,\ v\in V_k\}
\]

得到 query-aware 粗子图：

\[
G_q^{\text{coarse}}=(V_k,E_k)
\]

这个粗子图再送入现有 GNN 主体。

也就是说，本方案尽量不改：

1. GNN 层
2. `ProgressiveQuery`
3. `LayerRefine`
4. `ReadoutRefine`

创新一只负责：

\[
\text{第一阶段检索}
\]

---

## 10. Recommended Training Strategy

本方案不建议一上来做完全端到端训练。推荐采用三阶段训练。

### 10.1 Stage A: KGE Pretraining

先训练 `ComplEx`，得到：

\[
E^{\text{kge}}
\]

### 10.2 Stage B: Initialize Sampler

新建 sampler 自己的关系嵌入参数：

\[
E^{\text{sam}}
\in
\mathbb R^{(2n_{\text{rel}}+1)\times d_s}
\]

初始化：

\[
E^{\text{sam}}_0 \leftarrow E^{\text{kge}}
\]

### 10.3 Stage C: Joint Fine-Tuning

主模型训练时，用较小学习率微调 sampler 参数。

总损失可写为：

\[
\mathcal L
=
\mathcal L_{\text{main}}
+
\lambda_{\text{kge}}\mathcal L_{\text{kge}}
+
\lambda_{\text{reg}}
\left\|
E^{\text{sam}} - E^{\text{kge}}
\right\|_2^2
\]

其中：

- \(\mathcal L_{\text{main}}\) 是当前主模型的链路预测损失
- \(\mathcal L_{\text{kge}}\) 是可选的轻量辅助 KGE 损失
- 第三项用于防止 sampler 关系语义在主任务微调中漂移过大

推荐学习率关系：

\[
\eta_{\text{main}} > \eta_{\text{sam}} > \eta_{\text{kge-aux}}
\]

例如：

1. `main lr = 1e-4`
2. `sampler lr = 2e-5`
3. `aux kge lr = 1e-5`

---

## 11. Why This Is Better Than PRA / PCRW as the Main Route

`PRA / PCRW` 的核心是路径模板匹配：

\[
x_{\pi}(h,v)=\mathbf e_h^\top T_{r_1}T_{r_2}\cdots T_{r_L}\mathbf e_v
\]

再做路径加权：

\[
s_{\text{pra}}(h,r_q,v)
=
\sum_{\pi\in\Pi(r_q)} w_{r_q,\pi}x_\pi(h,v)
\]

这个思路有价值，但不适合作为当前仓库创新一的主线，原因是：

1. 更稀疏，召回不稳定
2. 强依赖关系路径模板挖掘
3. 不像 PPR 那样天然适合做全局粗检索
4. 改造成本更高

本方案更适合作为主路线，因为：

1. 仍保留 PPR 的扩散覆盖率
2. 关系语义直接进入第一阶段扩散
3. 更符合当前 `sampler -> subgraph -> GNN` 的代码结构

---

## 12. Code-Level Landing Points

如果后续按这份文档落地，建议改动位置如下。

### 12.1 New Files

建议新增：

1. `pretrain_kge.py`
2. `sampler_kge_utils.py`

用于训练和加载 `ComplEx` 关系嵌入。

### 12.2 Existing Files

#### `PPR_sampler.py`

需要改造为：

1. 保留关系有向边，而不是只保留同构 \((h,t)\) 边
2. 新增 relation embedding loading
3. 新增 query-aware edge scoring
4. 新增 weighted transition construction
5. 新增 online weighted PPR or query cache

#### `train_auto.py`

需要新增：

1. sampler 相关参数
2. 是否加载预训练关系嵌入
3. sampler 学习率和权重衰减

#### `search_auto.py`

需要新增：

1. sampler 相关超参搜索空间
2. 是否开启 weighted PPR sampler
3. edge score function 的选择

#### `base_model.py`

需要新增：

1. sampler 参数分组
2. 可选的辅助 KGE loss
3. 联合微调逻辑

---

## 13. Suggested Hyperparameters

第一版建议先保持简单，不要一次加太多自由度。

### 13.1 KGE Pretraining

1. `kge_model = ComplEx`
2. `kge_dim = 64 or 128`
3. `kge_dropout = 0.0 or 0.1`
4. `kge_neg_num = 32 or 64`

### 13.2 Weighted PPR

1. `ppr_restart = 0.15`
2. `ppr_iter = 20 ~ 50`
3. `score_fn = bilinear`
4. `sampler_dim = kge_dim`

### 13.3 Joint Fine-Tuning

1. `sampler_lr = 2e-5`
2. `sampler_weight_decay = 1e-5`
3. `lambda_kge = 0.01`
4. `lambda_reg = 0.001`

---

## 14. Evaluation Protocol

对于这个创新，不能只看最终模型 MRR。至少要看三层指标。

### 14.1 KGE Validation

验证 `ComplEx` 预训练是否学到合理关系语义：

1. `valid MRR`
2. `Hits@1`
3. `Hits@10`

### 14.2 Retrieval Metrics

这是创新一最关键的指标。

设第一阶段节点集合为 \(V_k(h,r_q)\)，真实答案为 \(t\)，则：

\[
\text{Recall@K}_{\text{node}}
=
\frac{1}{|Q|}
\sum_{(h,r_q,t)\in Q}
\mathbf 1[t \in V_k(h,r_q)]
\]

还建议记录：

1. coarse graph 中真实答案的 rank
2. coarse graph 节点数
3. coarse graph 边数
4. query-aware sampler 相比原始 PPR 的召回增益

### 14.3 Final End-to-End Metrics

最终仍以主任务指标为准：

1. `valid MRR`
2. `test MRR`
3. `Hits@1`
4. `Hits@10`

理想目标是：

\[
\text{Recall@K}_{\text{node}}^{\text{new}}
>
\text{Recall@K}_{\text{node}}^{\text{PPR}}
\]

并且：

\[
\text{MRR}_{\text{valid}}^{\text{new}}
>
\text{MRR}_{\text{valid}}^{\text{baseline}}
\]

---

## 15. Minimal First Version

为了降低实现风险，第一版建议只做下面这些：

1. `ComplEx` 预训练关系嵌入
2. sampler 中引入关系有向边
3. 用双线性边权：

\[
a_q(u,r,v)=\tilde{\mathbf e}_r^\top W \tilde{\mathbf e}_{r_q}
\]

4. 用 query-aware weighted PPR 取代原始 head-only PPR
5. 后续 GNN 结构先不改

不建议第一版同时做：

1. soft topk
2. fully end-to-end differentiable sampler
3. 节点语义与关系语义混合打分
4. 路径模板和 weighted PPR 的同时融合

第一版先把问题收敛为：

\[
\text{Can relation semantics improve stage-1 retrieval without hurting PPR coverage?}
\]

---

## 16. Final Summary

本方案的核心可以概括为三句话：

1. 用 `ComplEx` 学一套关系语义向量
2. 用关系语义去重写 PPR 的边转移概率
3. 用 `query-aware weighted PPR` 替换当前第一阶段的 `head-only PPR`

最终第一阶段不再是：

\[
\operatorname{PPR}(v \mid h)
\]

而是：

\[
\operatorname{PPR}(v \mid h,r_q)
\]

这比“粗图里再精排”更符合创新一的定位，也更有机会真正提升第一阶段小图检索质量。
