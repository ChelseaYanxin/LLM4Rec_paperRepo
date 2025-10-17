# LLM4Rec_paperRepo
# 📚 LLM4Rec 发展路线

## 🕰️ 早期模型（Before LLM）

- **Caser**：引入 CNN 进行推荐序列建模  
- **GRU4Rec**：使用 GRU 建模用户行为序列  
- **SASRec (2018)**  
  - [Paper](https://arxiv.org/abs/1808.09781)
- **BERT4Rec (2019)**  
  - [Paper](https://arxiv.org/abs/1904.06690)

## 🧠 预训练模型与 Prompt 驱动推荐

- **PPR (Prompt Tuning)**
- **LMRS**
  - 利用提示词（Prompt）将推荐任务表述为多 token 的完形填空任务（Cloze Task）  
  - [Paper](https://www.amazon.science/publications/language-models-as-recommender-systems-evaluations-and-limitations)
- **ZSRLM (2021)**  
  - 将交互行为转化为 Prompt，探索零样本推荐（zero-shot recommendation）  
  - 比较了 BERT 与 GPT2 的表现  
  - [Paper](https://arxiv.org/pdf/2112.04184)
- **PLM4Rec (2022)**  
  - 推荐系统中预训练语言模型（PLM）的综述  
  - [Paper](https://arxiv.org/pdf/2302.03735)

## 🚀 生成式推荐模型（Generative Recommendation）

### 2023 年
- **TIGER**
  - 经典生成式推荐模型  
  - 使用语义 ID 生成方式，RQ-VAE 优化为残差 KMeans 量化  
  - [Paper](https://arxiv.org/pdf/2305.05065)
- **LLM4RS**
  - 验证 GPT-3.5 的推荐能力  
  - 三种范式：point-wise、pair-wise、list-wise  
  - [Paper](https://arxiv.org/pdf/2305.02182) | [Code](https://github.com/rainym00d/LLM4RS)
- **LLMRec**
  - 利用图结构与 ID 信息嵌入到 LLM 中  
  - 对比：E4SRec 更注重 ID 注入，而 LLMRec 更重图结构  
  - [Paper](https://arxiv.org/pdf/2311.00423)
- **CoLLM**
  - 将通用协同过滤模型预训练的 user/item embedding 注入到 LLM  
  - [Paper](https://arxiv.org/pdf/2310.19488)
- **GenRec**
  - 通过 prompt 对 item 名称格式化，微调 LLM 预测下一个交互 item  
  - [Paper](https://arxiv.org/pdf/2307.00457)
- **E4SRec**
  - 可插拔组件，提取 item embeddings，使用 Adapter 处理个性化需求  
  - [Paper](https://arxiv.org/abs/2312.02443)
- **TALLRec**
  - 微调 LLM 以适配推荐任务（指令微调 + 推荐微调）  
  - [Paper](https://arxiv.org/abs/2305.00447) | [Code](https://github.com/SAI990323/TALLRec)
- **BIGRec**
  - 引入流行度与协同信息增强  
  - [Paper](https://arxiv.org/pdf/2308.08434)

### 2024 年
- **LANE**
  - 通过 LLM 为用户序列生成偏好摘要  
  - 结合协同过滤 (CF) 与语义 embedding  
  - [Paper](https://arxiv.org/pdf/2407.02833)
- **MMGRec**
  - 编码方式：Graph RQ-VAE (语义 token + 流行度 token)  
  - [Paper](https://arxiv.org/pdf/2404.16555)
- **LC-Rec（腾讯）**
  - 基于 Tiger 架构，使用 RQ-VAE 进行多级离散化表征  
  - 引入均匀分布约束缓解 ID 冲突  
  - 多任务微调：
    1. 用户历史序列预测未来 item  
    2. 文本生成 item ID  
    3. 跨模态预测（文本 ↔ ID）  
    4. 根据用户意图预测感兴趣 item  
  - [Paper](https://arxiv.org/abs/2311.09049) | [Code](https://github.com/RUCAIBox/LC-Rec)
- **ColaRec（腾讯）**
  - 将协同过滤信号直接引入生成式建模  
  - 使用 LightGCN + KMeans 聚类编码  
  - [Paper](https://arxiv.org/pdf/2403.18480) | [Code](https://github.com/Junewang0614/ColaRec)
- **EAGER（浙大 × 华为）**
  - 行为与语义分开建模，引入全局对比学习与摘要 token  
  - 推理时通过预测熵融合信息  
  - [Paper](https://arxiv.org/pdf/2406.14017)
- **CCF-LLM**
  - 将 CF embedding 与语义 embedding 加权融合  
  - 提升 CoLLM 效果  
  - [Paper](https://arxiv.org/pdf/2408.08564)
- **HLLM**
  - 构建 Item/User 双塔 LLM  
  - 同时支持生成式推荐与判别式推荐  
  - [Paper](https://arxiv.org/pdf/2409.12740)
- **LASER**
  - 利用 MoE 与 Q-Former，从 CF embedding 中提取有效特征  
  - 进行 Prompt Tuning  
  - [Paper](https://arxiv.org/pdf/2409.01605)
- **LLM4RERANK（CityU × 华为）**
  - 将不同重排序需求抽象为节点，构建全连接图  
  - 使用 CoT 综合多因素决策，实现个性化重排序  
  - [Paper](https://arxiv.org/pdf/2406.12433)

### 2025 年
- **HSTU**
  - Meta 生成式推荐，比传统 SASRec 提升约 60pt  
  - [Paper](https://arxiv.org/pdf/2402.17152)
- **OneRec（快手）**
  - 残差 KMeans 量化 + session-wise 建模 + IPA 偏好对齐优化 + 稀疏 MMOE  
  - [Paper](https://arxiv.org/pdf/2502.18965)
- **GRAB（百度）**
  - 生成式排序模型，侧重性能优化  
  - 与 HSTU 类似结构  
  - [Link](https://mp.weixin.qq.com/s/mT8DmHzgc3ag57PVMqZ3Rw)
- **RPG（Meta）**
  - 并行生成长语义 ID 的生成式推荐  
  - 技术创新：
    1. 基于 PQ 的语义 ID 生成  
    2. Multi-token prediction  
    3. 图解码生成策略  
  - 在 Amazon 数据集上超过 Tiger 与 HSTU  
  - [Paper](https://arxiv.org/pdf/2506.05781) | [Code](https://github.com/facebookresearch/RPG_KDD2025)
- **DFGR（美团）**
  - 针对小红书 GenRank 优化  
  - 提出双流 GR 设计与动态 mask 机制  
  - 避免 label 泄露与样本拆分问题  
  - [Paper](https://arxiv.org/pdf/2505.16752)
- **GenRank（小红书）**
  - 生成式推荐探索性研究：
    1. 自回归范式有效  
    2. 样本聚合形式影响小  
    3. 传统推荐模块仍兼容  
    4. 滑动窗口特征显著  
  - [Paper](https://arxiv.org/pdf/2505.04180)
- **GR-LLMs (Alibaba)**
  - 生成式推荐最新综述  
  - [Paper](https://arxiv.org/pdf/2507.06507)

---

## 📊 小结

| 年份 | 模型 | 核心特点 |
|------|------|----------|
| 2018–2020 | SASRec / BERT4Rec | 序列建模、预训练 Encoder |
| 2021–2022 | ZSRLM / PLM4Rec | 引入 Prompt 与 PLM |
| 2023 | LLM4RS / TIGER / CoLLM | LLM 接入与生成式探索 |
| 2024 | LC-Rec / EAGER / HLLM | 多模态、协同语义融合 |
| 2025 | HSTU / OneRec / RPG / DFGR | 元生成式与动态优化趋势 |

---

> ✨ **未来方向**：
> - Tokenizer + Generator 联合优化（ETEGRec, OneRec）  
> - 语义与行为的双流建模（EAGER, DFGR）  
> - 可解释的生成式推荐与个性化重排序（LLM4RERANK）

