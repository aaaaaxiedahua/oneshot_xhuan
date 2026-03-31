# Module 1: Two-Stage Relation Refinement

## 1. Goal

本方案用于替代单纯的 `PPR + relation prior` 融合思路，改成一个更干净的两阶段采样框架：

1. **Stage 1: Coarse Retrieval**
   - 仍然使用原始 PPR 做全图粗召回
   - 目标是保证效率和召回率
2. **Stage 2: Relation Refinement**
   - 只在 Stage 1 得到的小图上做关系语义精化
   - 目标是让扩散方向真正受 query relation 和动态关系状态控制

这条路线的核心不是“再加一个静态关系表”，而是：

- 第一阶段负责找范围
- 第二阶段负责在范围内按关系语义重新分配传播质量

---

## 2. Why Not Static Relation Prior

已有的静态融合方法本质上是：

\[
\text{score}(v) = (1-\lambda)\,\text{PPR}(h,v) + \lambda\,\text{RelPrior}(q,v)
\]

这个问题在于：

1. relation 只在最后融合，扩散过程本身没变
2. query relation 一旦确定，整个扩散过程的关系偏好就固定
3. 无法表达“第 1 跳”和“第 3 跳”的关系语义差异
4. 更像统计后处理，不像真正的采样算法改造

因此，本方案把 relation 信息放进 **Stage 2 的局部扩散算子** 中，而不是最后再加一项。

---

## 3. Final Two-Stage Scheme

### 3.1 Stage 1: Coarse PPR Retrieval

从 query head `h` 出发，用原始 PPR 得到节点分布：

\[
\mathbf p^{\text{coarse}} = \alpha \mathbf e_h + (1-\alpha)\mathbf P^\top \mathbf p^{\text{coarse}}
\]

从中选出粗子图节点集合：

\[
V_q^{(0)} = \operatorname{TopK}(\mathbf p^{\text{coarse}}, K_1)
\]

并诱导出粗子图：

\[
G_q = (V_q^{(0)}, E_q^{(0)})
\]

说明：

- `K1` 来自原有 `topk` 或 Stage 1 节点预算
- 这一阶段不改原始 PPR 公式
- 这一阶段只负责粗召回，不负责关系精化

### 3.2 Stage 2: Small-Graph Relation Refinement

只在 `G_q` 上做少量轮次的关系驱动扩散。

#### Step 1: Initialize

\[
\mathbf p_0 = \mathbf e_h,\qquad \mathbf z_0 = \mathbf c_q
\]

其中：

- `p_t` 是小图内当前节点分布
- `z_t` 是当前关系状态
- `c_q` 是 query relation 的轻量参数向量

#### Step 2: Current Relation Intent

\[
\mathbf g_t = W_q \mathbf c_q + W_z \mathbf z_t
\]

这里 `g_t` 表示第 `t` 轮扩散时的**当前关系意图向量**。

直观上：

- `W_q c_q` 表示 query relation 的初始语义需求
- `W_z z_t` 表示当前扩散到这一步形成的关系状态
- 两者合成当前这一轮更应该走哪些关系边

#### Step 3: Relation-Driven Edge Scoring

对小图中每条边 `(u, r, v)`，只根据边关系 `r` 和当前关系意图 `g_t` 计算打分：

\[
e_t(u,r,v) = \mathbf c_r^\top \mathbf g_t
\]

其中 `c_r` 是边关系 `r` 的轻量参数向量。

说明：

- 这里不再额外加 `\log p_t(u)` 或 `\psi(v,q)` 等项
- 目的是让公式保持干净，避免拼凑感

#### Step 4: Local Transition Normalization

对每个源点 `u` 的邻边做 softmax，得到局部转移概率：

\[
T_t(v\mid u) =
\operatorname{softmax}_{(u,r',v')\in \mathcal N_{G_q}(u)}
\big(e_t(u,r',v')\big)
\]

说明：

- `T_t` 不是全图统一转移矩阵
- `T_t` 只定义在粗子图 `G_q` 上
- `T_t` 会随 `t` 变化，因为 `g_t` 和 `z_t` 在变

#### Step 5: Node Distribution Update

\[
\mathbf p_{t+1} =
\alpha \mathbf e_h + (1-\alpha)\mathbf T_t^\top \mathbf p_t
\]

说明：

- 这一条保留了 PPR / RWR 的 restart diffusion 骨架
- 但固定的 `P` 被替换成了小图上的动态 `T_t`

#### Step 6: Relation State Update

\[
\mathbf z_{t+1}
=
(1-\eta)\mathbf z_t
+ \eta \sum_{(u,r,v)\in G_q} p_t(u)\,T_t(v\mid u)\,\mathbf c_r
\]

说明：

- `z_t` 由当前真正经过的关系边进行更新
- 这不是 MuRWR 的原始大状态传播
- 这是一个轻量 query-level relation-state memory

#### Step 7: Final Node Selection

完成 `T` 轮 refinement 后，再从最终节点分布中选出精化后的节点集合：

\[
\hat V_q = \operatorname{TopK}(\mathbf p_T, K_2)
\]

说明：

- 推荐只在最后做一次 `TopK`
- 不建议每轮都硬裁节点
- 否则方法会变得离散、脆弱、参数过多

---

## 4. What Comes From Prior Work vs What Is New

### 4.1 PPR / RWR

参考公式：

\[
\mathbf p = \alpha \mathbf e_h + (1-\alpha)\mathbf P^\top \mathbf p
\]

借鉴点：

- restart diffusion 骨架
- 头实体驱动的 personalized propagation

本方案中的对应位置：

\[
\mathbf p_{t+1} =
\alpha \mathbf e_h + (1-\alpha)\mathbf T_t^\top \mathbf p_t
\]

区别：

- 原始 PPR 用固定 `P`
- 本方案在 Stage 2 中使用小图上的动态 `T_t`

### 4.2 Supervised Random Walks (SRW)

参考核心形式：

\[
a_{uv}^{(q)} = \exp(\theta^\top \psi(u,r_{uv},v,q))
\]

\[
Q_{uv}^{(q)} =
\frac{a_{uv}^{(q)}}{\sum_z a_{uz}^{(q)}}
\]

借鉴点：

- 转移概率不应固定
- 应先对边打分，再局部归一化成 query-dependent transition

本方案中的对应位置：

\[
e_t(u,r,v) = \mathbf c_r^\top \mathbf g_t
\]

\[
T_t(v\mid u) =
\operatorname{softmax}_{(u,r',v')\in \mathcal N_{G_q}(u)}
\big(e_t(u,r',v')\big)
\]

区别：

- SRW 是全图边级 supervised random walk
- 本方案不学习全图 walker
- 本方案只在粗子图上做局部 refinement

### 4.3 MuRWR

参考核心形式：

\[
\mathbf R_{t+1}
=
\alpha \mathbf Q_s
+ (1-\alpha)\sum_k \tilde{\mathbf A}_k \mathbf R_t \mathbf S_k
\]

借鉴点：

- 关系语义不应是静态的
- 关系状态应随扩散过程动态演化

本方案中的对应位置：

\[
\mathbf z_{t+1}
=
(1-\eta)\mathbf z_t
+ \eta \sum_{(u,r,v)\in G_q} p_t(u)\,T_t(v\mid u)\,\mathbf c_r
\]

区别：

- MuRWR 维护的是 `节点 × 关系标签` 大状态矩阵
- 本方案只维护一个小的 query-level state `z_t`
- 本方案只在粗子图上运行

---

## 5. Innovation Summary

本方案真正的创新点不在于双线性项本身，而在于以下四点的组合：

1. **Coarse-to-fine framework**
   - 第一阶段全图 PPR 粗召回
   - 第二阶段小图关系精化

2. **Lightweight relation-state memory**
   - 不做 MuRWR 的大状态展开
   - 只保留轻量 `z_t`

3. **Query- and state-conditioned local diffusion**
   - 第二阶段的局部转移不再固定
   - 由 query relation 和当前关系状态共同决定

4. **Relation-driven refinement instead of static reranking**
   - relation 信息不再只是结果融合项
   - 而是直接进入第二阶段扩散算子

---

## 6. New Hyperparameters

### 6.1 Recommended Minimal New Hyperparameters

建议第一版只新增以下 4 个超参数：

| Name | Meaning | Recommended Values | Notes |
|---|---|---|---|
| `refine_dim` | 第二阶段关系向量和状态向量维度 | `16`, `32` | 最核心的新超参数 |
| `refine_steps` | 第二阶段 refinement 轮数 | `2`, `3`, `4` | 建议先用 `2` 或 `3` |
| `refine_eta` | 关系状态更新强度 `\eta` | `0.1`, `0.3`, `0.5` | 控制 `z_t` 更新快慢 |
| `final_topk` | 第二阶段最终保留节点预算比例 | `0.07`, `0.08`, `0.10` | 语义与原 `topk` 完全一致，二者都表示“占整个实体集的比例” |

### 6.2 Existing Parameters That Should Not Be Reused

| Parameter | Whether Used in Stage 2 | Reason |
|---|---|---|
| `fact_ratio` | No | `fact_ratio` 是训练图切分比例，不是 refinement 轮次预算 |
| `topk` | Stage 1 only | 用于粗召回节点预算 |
| `topm` | Optional | 属于边采样预算，不等于第二阶段关系扩散预算 |

### 6.3 Recommended Default Configuration

建议第一版的默认值：

| Hyperparameter | Default |
|---|---|
| `refine_dim` | `16` |
| `refine_steps` | `2` |
| `refine_eta` | `0.3` |
| `final_topk` | 若想最终仍为 `0.1`，可设 `topk=0.14, final_topk=0.1` |

理由：

- 参数语义更直观
- `topk` 继续负责粗召回
- `final_topk` 直接对应最终送进 GNN 的预算

---

## 7. What Not To Add in the First Version

第一版不建议再加下面这些项，否则方法会开始臃肿：

1. 每轮节点硬裁剪
2. 每轮动态调整 `fact_ratio`
3. `log p_t(u)` 直接并入边打分
4. 额外的目标节点匹配项 `\psi(v,q)`
5. 太多 gate / teleport / temperature 参数

原因：

- 会让公式过多
- 会让方法显得像 heuristic 拼装
- 不利于讲清主线创新

---

## 8. Final Clean Formula Set

最终建议在主文中只保留下面这组公式：

\[
V_q^{(0)} = \operatorname{TopK}(\mathbf p^{\text{coarse}}, K_1)
\]

\[
\mathbf p_0 = \mathbf e_h,\qquad \mathbf z_0 = \mathbf c_q
\]

\[
\mathbf g_t = W_q \mathbf c_q + W_z \mathbf z_t
\]

\[
e_t(u,r,v) = \mathbf c_r^\top \mathbf g_t
\]

\[
T_t(v\mid u) =
\operatorname{softmax}_{(u,r',v')\in \mathcal N_{G_q}(u)}
\big(e_t(u,r',v')\big)
\]

\[
\mathbf p_{t+1} =
\alpha \mathbf e_h + (1-\alpha)\mathbf T_t^\top \mathbf p_t
\]

\[
\mathbf z_{t+1}
=
(1-\eta)\mathbf z_t
+ \eta \sum_{(u,r,v)\in G_q} p_t(u)\,T_t(v\mid u)\,\mathbf c_r
\]

\[
\hat V_q = \operatorname{TopK}(\mathbf p_T, K_2)
\]

这组公式已经足够表达：

- coarse retrieval
- query-conditioned relation intent
- relation-driven local transition
- lightweight relation-state update
- final node refinement

---

## 9. Short Conclusion

这个方案的关键不是“换一个现成 random walk 算法”，而是：

- 保留原始 PPR 的高效粗召回
- 在粗召回小图内加入轻量关系状态精化
- 用 query relation 和动态 relation state 共同控制局部扩散方向

相比 `PPR + static relation prior`，它更像一个真正的采样算法改造；相比直接照搬 MuRWR 或 SRW，它更轻、更贴合 one-shot subgraph reasoning 的两阶段框架。
