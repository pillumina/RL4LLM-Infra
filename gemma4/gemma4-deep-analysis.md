# Gemma 4 深度架构解析：小模型如何重新定义参数效率的边界

> 2026 年 4 月 2 日，Google DeepMind 发布 Gemma 4。31B 参数的 dense 模型在 AIME 2026 上拿到 89.2%，MoE 变体以 3.8B 活跃参数跑出 27B 级别的性能。与此同时，Qwen3-235B 的总参数量是它的 7.6 倍，GLM-5 的规模也在同等量级。

这引出了一个被反复追问却始终没有得到清晰回答的问题：**大模型的参数规模，到底还有多大意义？**

本文不打算罗列 benchmark 数字——那是对读者时间的浪费。我们要拆解的是 Gemma 4 在架构和训练层面做出的每一项技术选择，理解它们为什么有效，以及它们共同指向的一个核心洞察。

## 核心洞察：参数效率正在取代参数规模，成为模型能力的真正决定因素

这不是一个温和的观点。它意味着，在训练数据质量、合成数据策略和架构效率三个维度上取得突破的团队，可以用一个数量级更少的参数达到相同的能力水平。Gemma 4 是目前这个观点最有力的证据。

但"效率"是一个被滥用到失去意义的词。我们需要拆开来看：Gemma 4 的效率到底从哪里来？

## 一、架构篇：每一分算力都要花在刀刃上

Gemma 4 的架构并不是凭空发明了一套全新范式。它的基础仍然是 decoder-only Transformer，但 Google 在几乎每一个组件上都做了有针对性的优化。这些优化单独看都不算革命性，但叠加起来产生的效果是显著的。

### 1.1 Shared KV Cache：最简单也最容易被忽视的优化

Transformer 推理时，显存消耗的大头是 KV Cache——每一层都要缓存所有历史 token 的 Key 和 Value 向量。对于 256K 上下文的模型来说，这部分显存可以轻松超过模型权重本身。

Gemma 4 的做法极其直接：**让最后 N 层直接复用前面同类型 attention 层的 KV 张量，不再自己计算 K 和 V 投影。**

这不是一种近似或压缩——最后一层复用的就是之前某一层精确计算过的 KV。HuggingFace 团队在实际测试中确认，这种做法对输出质量的影响微乎其微，但显存和计算开销的节省是实实在在的。

为什么这样做效果这么好？一个直觉性的解释是：深层网络中，相邻层学到的 KV 表示往往高度相似。强行让每一层都独立计算 K 和 V，本质上是在做冗余工作。

### 1.2 Global Attention 的极致压缩

在混合注意力机制中，global attention 层是最昂贵的——它们需要对整个上下文序列做全注意力计算。Gemma 4 在 global attention 层上叠加了五重优化，每一层都在压缩 KV Cache 的体积：

**第一重：GQA 极端分组。** Local attention 层使用 2 个 Query 头共享 1 个 KV 头（GQA 2:1），而 global attention 层直接拉到 8:1。8 个 Query 共享 1 个 KV head，缓存直接缩小 8 倍。

这样做显然会损失信息。但 Google 的应对方式是：

**第二重：Key 维度翻倍。** 每个 KV head 的维度从标准值扩大到两倍。更宽的 KV 向量能在更少的 head 数量下保留足够的信息容量，补偿 GQA 带来的信息瓶颈。

**第三重：K = V。** 在 global attention 层中，直接让 Key 向量等于 Value 向量。这意味着不再需要单独存储 V，KV Cache 进一步减半。效果上，这相当于强制模型在"检索"（Key 的角色）和"读取"（Value 的角色）之间使用同一套表示，降低了表示的自由度，但也降低了过拟合的风险。

**第四重：p-RoPE（比例位置编码）。** 标准 RoPE 对向量的所有维度都施加位置旋转。但在长上下文场景下，低频维度的旋转极其微小，累积起来反而会在远距离 token 之间引入对齐误差——这些误差对语义匹配是有害的。p-RoPE 只对 25% 的高频维度施加旋转，让低频维度纯粹保留语义信息，不被位置噪声污染。

**第五重：最后一层强制 Global。** 无论 interleaving pattern 怎么排，最后一层一定是 global attention。这确保了输出 token 在生成之前，能够"看到"完整的输入上下文——这对指令跟随和事实准确性至关重要。

把这五重优化放在一起看，你会发现一个清晰的设计哲学：**global attention 层的目标不是保留最多的信息，而是在保留足够信息的前提下，把开销压缩到最低。** 这是一种"够用就好"的工程思维，与学术界追求极致精度的风格形成鲜明对比。

### 1.3 双 RoPE：不同层用不同的位置编码策略

Gemma 4 在 sliding window 层使用标准 RoPE，在 global 层使用 p-RoPE。这不是随意的选择。

Sliding window 的上下文窗口很短（小模型 512 token，大模型 1024 token），位置编码不需要覆盖很大的距离范围，标准 RoPE 完全够用。但 global 层要处理 256K 的完整上下文——在这个尺度上，标准 RoPE 的低频分量会因为旋转过小而变得不可靠。

分开处理的好处是：sliding window 层保留完整的位置信息（包括低频语义分量），global 层专注于大范围的语义关联，不受位置噪声干扰。

### 1.4 Per-Layer Embeddings：给每一层一个独立的"记忆通道"

这是 Gemma 4 小模型（E2B、E4B）最独特的架构特征，也是理解"为什么 2B 能做这么多事"的关键。

在标准 Transformer 中，每个 token 的 embedding 向量在整个模型的所有层中是同一个。这意味着第一层的 embedding 需要预先编码这个 token 在所有层次上可能需要的信息——这对一个固定维度的向量来说是不现实的约束。

PLE 的做法是：**为每一个 decoder 层维护一个独立的小型 embedding table。** 每个 token 在每一层都会收到一个专属的信号，这个信号由两部分组成：
- 一个 token-identity 分量（通过 embedding lookup 获得）
- 一个 context-aware 分量（通过对主 embedding 的学习投影获得）

每一层通过一个轻量级的 residual block，将这些信号注入到 hidden states 中。

效果上，E2B 的总参数是 5.1B，但有效参数只有 2.3B——另外 2.8B 就是 PLE 的 embedding table。这些参数在磁盘上占空间，但计算成本极低（只是 embedding lookup + 小型投影），所以推理速度完全是一个 2B 模型的水平。

这是参数量和计算量之间的精妙分离：**用存储换计算，用参数空间换算力空间。** 对于边缘设备来说，存储空间比算力便宜得多，这种 trade-off 非常合理。

### 1.5 MoE 128 小专家：细粒度条件计算

26B-A4B 是 Gemma 4 家族中最值得深入分析的模型。25.2B 总参数，每 token 只激活 3.8B（8 个专家 + 1 个共享专家），但性能达到 31B dense 模型的 97%。

关键设计选择是 **128 个小专家**，而不是业界更常见的 8-16 个大专家。

这两种设计的本质区别在于**路由的粒度**。少量大专家的路由器只能做粗粒度的分发——"这段文本归你"。128 个小专家的路由器可以做微分发——"这个词的这种语义组合归这个专家"。

更细的路由粒度意味着：
- 每个专家学习的模式更专注，专家间的功能重叠更少
- 参数利用率更高——总参数中真正被用到的比例更大
- 路由器本身的决策空间更大，能学到更复杂的分发策略

但 128 个专家也带来了挑战：路由器训练更难（选择空间从 8-16 扩大到 128），负载均衡更难保证。Google 没有公开具体的路由训练细节，但从结果来看，他们显然解决了这些问题。

一个值得注意的数据点：26B-A4B 在 AIME 2026 上拿到 88.3%，31B dense 是 89.2%——差距只有 0.9 个百分点。但在推理成本上，前者大约是后者的 1/8。对于生产部署来说，这个性价比差距是决定性的。

## 二、训练篇：Gemini 3 的知识蒸馏才是真正的护城河

架构优化解释了 Gemma 4 的效率，但没有完全解释它为什么这么强。一个架构精良但训练数据平庸的 31B 模型，不可能在 benchmark 上碾压训练充分的 235B 模型。

**Gemma 4 真正的差异化优势来自训练，而非架构。**

### 2.1 Gemini 3 作为 Teacher：一个不对称的优势

Google 明确表示 Gemma 4 的架构源自 Gemini 3 研究。虽然他们没有公开 Gemma 4 的训练细节，但从 ResearchGate 上已经发表的论文来看，Google 团队已经在系统性地研究从 Gemini 3 到 Gemma 的知识蒸馏方法。

这种蒸馏不是简单的"让大模型生成答案，小模型模仿"。根据已发表的 Chain-of-Thought 蒸馏研究，流程更接近：
1. 用 Gemini 3 对大量 prompt 生成详细的推理链（Chain-of-Thought）
2. 过滤质量，保留推理逻辑清晰、步骤完整的样本
3. 用这些合成数据训练 Gemma 4 的 instruction-tuning 阶段

这意味着 Gemma 4 在训练过程中"看到"的推理数据质量，远超任何公开数据集能提供的水平。而且这种优势是不对称的——Qwen 和智谱没有同等规模的 teacher 模型来生成合成数据。

### 2.2 QAT：训练时就考虑量化

Gemma 4 提供了 Quantization-Aware Training (QAT) checkpoint。这不是事后量化，而是在训练阶段就引入量化噪声，让模型学会在低精度表示下保持输出质量。

实际效果是：Gemma 4 的 QAT checkpoint 在量化到 NVFP4（4-bit 浮点）后，质量损失极小。NVIDIA 已经发布了基于此的 Gemma-4-31B-IT-NVFP4 模型。

对于生产环境来说，这意味着：
- 相同硬件上可以跑更大的 batch size
- 更长的上下文不会因显存不足而失败
- 推理吞吐量直接翻倍（4-bit vs 16-bit）

### 2.3 原生多模态训练：不是拼接，是融合

Gemma 4 的多模态能力不是后期拼接的结果，而是在预训练阶段就与文本一起训练的。视觉编码器基于 ViT，但做了关键改进：
- 使用 2D RoPE 而非 1D RoPE，直接编码 patch 的二维空间位置
- 支持可变宽高比输入，通过自适应 resize + padding 保持原始比例
- 可配置的 soft token budget（70/140/280/560/1120 tokens），让开发者根据任务需求在精度和速度之间做 trade-off

音频编码器沿用了 Gemma-3n 的 USM-style conformer 架构，但从 681M 参数压缩到 305M，帧时长从 160ms 降到 40ms。这意味着更低的延迟和更好的实时转录体验。

## 三、对比篇：与 Qwen3.5、GLM-5 的正面交锋

### 3.1 参数效率对比

要公平比较，我们需要关注的是**活跃参数量**而非总参数量：

| 模型 | 总参数 | 活跃参数 | AIME 2026 | LiveCodeBench v6 | 活跃参数效率 |
|------|--------|----------|-----------|-------------------|-------------|
| Gemma 4 31B | 30.7B | 30.7B | 89.2% | 80.0% | 2.90 /B |
| Gemma 4 26B-A4B | 25.2B | 3.8B | 88.3% | 77.1% | **23.2** /B |
| Qwen3-235B-A22B | 235B | 22B | ~85%* | ~72%* | ~3.9 /B |
| Qwen3.5-Omni | 未公开 | 未公开 | 未公开 | 未公开 | — |
| GLM-5 | 未公开 | 未公开 | ~87%* | ~75%* | — |

*注：Qwen3 和 GLM-5 的数字为估算值，基于公开 benchmark 报告。

这个表格揭示了一个关键事实：**Gemma 4 26B-A4B 的每活跃参数效率是 Qwen3-235B 的近 6 倍。** 即使拿 31B dense 和 235B-A22B 比，Gemma 4 的活跃参数效率也领先约 30%。

### 3.2 架构选择对比

| 设计选择 | Gemma 4 | Qwen3 | GLM-5 |
|---------|---------|-------|-------|
| MoE 专家数 | 128 小专家 | 8-16 大专家 | 未公开 |
| 位置编码 | Dual RoPE + p-RoPE | 标准 RoPE | 标准 RoPE |
| KV Cache 优化 | Shared KV + K=V + GQA 8:1 | 标准 GQA | 标准 GQA |
| 混合注意力 | Sliding + Global 交错 | 全局注意力为主 | 全局注意力 |
| 量化策略 | QAT checkpoint | 后量化 | 后量化 |
| 边缘部署 | PLE + 2-bit 量化 | 无专门优化 | 无专门优化 |

Gemma 4 在几乎每一个架构组件上都选择了"效率优先"的方案。这种一致性不是偶然的——它反映了一个明确的工程目标：**在给定参数预算下最大化推理效率。**

### 3.3 生态与许可证

这可能是 Gemma 4 最被低估的优势。

Gemma 4 采用 Apache 2.0 许可证——这是开源社区最宽松的标准许可证之一。没有 MAU 限制，没有使用场景限制，没有再分发限制。

与之形成对比的是：
- **Qwen3.5-Omni 和 Qwen 3.6 Plus** 已经开始从完全开源转向更受限的发布策略，不再公开全部权重
- **GLM-5** 的开源策略也在收紧
- **Meta Llama 4** 使用社区许可证，限制比 Apache 2.0 更严格

Google 在中国 AI 实验室收紧开源的同时选择全面开放，这个时间点的选择意味深长。对于需要商业部署确定性的企业来说，许可证的清晰度有时比 benchmark 数字更重要。

### 3.4 公平性声明

任何对比都需要承认 benchmark 的局限性。Gemma 4 在数学推理和编程上的优势是真实的，但：
- **长尾知识**：235B 级别的模型在罕见领域知识上仍然有优势，这不是 benchmark 能完全覆盖的
- **中文能力**：Gemma 4 支持 140+ 语言，但中文场景的深度可能不如 Qwen3 和 GLM-5（这两者的训练数据中中文比例更高）
- **多轮对话体验**：实际使用中的主观感受可能与 benchmark 排名有差异

## 四、对行业的深层影响

### 4.1 "小模型 + 大 Teacher"正在成为新范式

Gemma 4 不是孤例。Microsoft 的 Phi 系列和 Apple 的 OpenELM 也在走类似路线——用大模型生成高质量训练数据，用精巧的架构训练小模型。

这个范式如果成立，意味着：
- 模型能力的上限由 Teacher 模型（通常是闭源的）决定
- 开源模型之间的竞争将从"谁的参数多"转向"谁的蒸馏策略好"
- 拥有最强闭源模型的团队（Google、OpenAI、Anthropic）在开源领域也拥有结构性优势

### 4.2 MoE 128 小专家可能成为新标准

传统的 8-16 大专家 MoE 设计（DeepSeek-V3、Qwen3-235B）可能在下一代模型中被重新审视。128 小专家的细粒度路由在参数效率上的优势已经足够明显，值得其他团队跟进。

### 4.3 架构效率将成为推理引擎的核心竞争力

Gemma 4 的架构选择（Shared KV Cache、K=V、p-RoPE）对推理引擎的实现提出了新要求。vLLM、SGLang 等框架需要针对这些优化做专门适配才能发挥最大效率。这也意味着，**能最好地支持这些高效架构的推理引擎，将在下一轮竞争中胜出。**

## 五、结论

Gemma 4 告诉我们的事情，远比一个 benchmark 排名重要：

**参数效率的边界比我们想象的要远得多。** 31B 参数可以达到过去需要 200B+ 参数才能达到的性能水平，这不是因为 31B 模型"超常发挥"，而是因为过去的大模型在架构和训练上存在大量可以优化的空间。

**拥有最强 Teacher 模型的团队，在开源领域也有结构性优势。** Google 通过 Gemini 3 蒸馏，让 Gemma 4 获得了其他开源团队无法复制的训练数据优势。这打破了"开源模型之间公平竞争"的假设。

**效率优化正在从推理工程层面上升到架构设计层面。** Shared KV Cache、p-RoPE、PLE 这些优化不是推理引擎的后处理技巧，而是被直接设计进模型架构中。未来的模型设计需要在训练时就考虑推理效率，而不是训练完再想办法优化。

对于 AI Infra 工程师来说，Gemma 4 的发布意味着一个新的工作重心：**不再只是让大模型跑得更快，而是让高效的模型架构得到最高效的执行。**

---

*本文基于 Google 官方发布信息、HuggingFace 技术博客、VentureBeat 报道以及 Maarten Grootendorst 的视觉化架构分析综合撰写。部分训练细节（如具体的蒸馏数据量和 RL 策略）为作者基于公开信息的合理推断，以官方技术报告为准。*

*参考来源：*
- [Google Blog: Gemma 4](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/)
- [HuggingFace Blog: Welcome Gemma 4](https://huggingface.co/blog/gemma4)
- [A Visual Guide to Gemma 4 — Maarten Grootendorst](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-gemma-4)
- [Google Open Source Blog: Gemma 4 Expanding the Gemmaverse](https://opensource.googleblog.com/2026/03/gemma-4-expanding-the-gemmaverse-with-apache-20.html)
- [VentureBeat: Google releases Gemma 4 under Apache 2.0](https://venturebeat.com/technology/google-releases-gemma-4-under-apache-2-0-and-that-license-change-may-matter)
- [WaveSpeed: What Is Google Gemma 4?](https://wavespeed.ai/blog/posts/what-is-google-gemma-4/)
- [NVIDIA: Gemma-4-31B-IT-NVFP4](https://huggingface.co/nvidia/Gemma-4-31B-IT-NVFP4)
