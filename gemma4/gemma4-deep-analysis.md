# Gemma 4 深度架构解析：小模型如何重新定义参数效率的边界

> 2026 年 4 月 2 日，Google DeepMind 发布 Gemma 4。31B 参数的 dense 模型在 AIME 2026 上拿到 89.2%，MoE 变体以 3.8B 活跃参数跑出 27B 级别的性能。与此同时，Qwen3-235B 的总参数量是它的 7.6 倍，GLM-5 更是达到 744B 总参数。

这引出了一个被反复追问却始终没有得到清晰回答的问题：**大模型的参数规模，到底还有多大意义？**

本文不打算罗列 benchmark 数字——那是对读者时间的浪费。我们要拆解的是 Gemma 4 在架构和训练层面做出的每一项技术选择，理解它们为什么有效，以及它们共同指向的一个核心洞察。

## 核心洞察：在推理和编程任务上，参数效率正在取代参数规模

这个洞察需要加一个限定词——"在推理和编程任务上"。因为在需要长尾知识广度和复杂 agentic 规划的场景中，参数规模仍然不可替代。Gemma 4 用 31B 参数在数学推理和编程 benchmark 上追平甚至超越 200B+ 级别的模型，证明了架构效率和训练质量可以在特定任务上大幅压缩参数需求。但 GLM-5 用 744B 参数在 SWE-bench（77.8%）和 Vending Bench 2 等长周期 agentic 任务上取得开源 SOTA，也证明了复杂规划能力的上限仍然与总参数量正相关。

效率优化有边界。理解这个边界在哪里，比单纯崇拜"小模型"更有价值。

## 一、架构篇：每一分算力都要花在刀刃上

Gemma 4 的基础仍然是 decoder-only Transformer，但 Google 在几乎每一个组件上都做了有针对性的优化。这些优化单独看都不算革命性，但叠加起来产生的效果是显著的。

### 1.1 Shared KV Cache：最简单也最容易被忽视的优化

Transformer 推理时，显存消耗的大头是 KV Cache——每一层都要缓存所有历史 token 的 Key 和 Value 向量。对于 256K 上下文的模型来说，这部分显存可以轻松超过模型权重本身。

Gemma 4 的做法极其直接：**让最后 N 层直接复用前面同类型 attention 层的 KV 张量，不再自己计算 K 和 V 投影。** 这不是近似或压缩——最后一层复用的就是之前某一层精确计算过的 KV。HuggingFace 团队确认这种做法对输出质量的影响微乎其微，但显存和计算开销的节省是实实在在的。

为什么这样做效果这么好？深层网络中，相邻层学到的 KV 表示往往高度相似。强行让每一层都独立计算 K 和 V，本质上是在做冗余工作。

### 1.2 Global Attention 的极致压缩

在混合注意力机制中，global attention 层是最昂贵的——它们需要对整个上下文序列做全注意力计算。Gemma 4 在 global attention 层上叠加了五重优化，形成一个完整的设计链：

起点是 **GQA**——global 层 16 个 Query 头共享 8 个 KV head（31B 模型，GQA 2:1），26B-A4B 的 16Q/8KV 配置类似。进一步在 global 层拉到更极端的分组（文章提及 8:1，但 config.json 显示实际 GQA 比例为 2:1——这可能意味着 8:1 仅应用于 global attention 层，而 config.json 记录的是模型整体配置）。但这显然会损失信息容量。Google 的应对是 **Key 维度翻倍**：更宽的 KV 向量能在更少的 head 数量下保留足够的信息。在这个基础上，进一步让 **K = V**——直接让 Key 向量等于 Value 向量，KV Cache 再减半。这强制模型在"检索"（Key 的角色）和"读取"（Value 的角色）之间使用同一套表示，降低了表示的自由度，但也降低了过拟合的风险。

到这里，KV Cache 已经被压缩到了极致。但还有一个问题需要解决：长上下文下的位置编码失真。标准 RoPE 对向量的所有维度都施加旋转，但在 256K 的距离上，低频维度的旋转极其微小，累积起来会在远距离 token 之间引入对齐误差，干扰语义匹配。Gemma 4 的解决方案是 **p-RoPE**：只对 25% 的高频维度施加旋转，低频维度纯粹保留语义信息，不被位置噪声污染。

最后，**最后一层强制 Global**——无论 interleaving pattern 怎么排，最后一层一定是 global attention，确保输出 token 能看到完整的输入上下文。

把这五层因果关系放在一起：GQA 8:1 压缩 KV head 数量 → Key 维度翻倍补偿信息损失 → K=V 进一步减半 → p-RoPE 解决长距离位置噪声 → 最后一层 Global 保证全局可见。**这不是五条独立措施的堆叠，而是一条完整的设计链。** 设计哲学很清晰：global attention 层的目标不是保留最多的信息，而是在保留足够信息的前提下，把开销压缩到最低。

### 1.3 双 RoPE：不同层用不同的位置编码策略

Gemma 4 在 sliding window 层使用标准 RoPE，在 global 层使用 p-RoPE。这不是随意的选择——sliding window 的上下文窗口很短（小模型 512 token，大模型 1024 token），标准 RoPE 完全够用。但 global 层要处理 256K 的完整上下文，标准 RoPE 的低频分量在这个尺度上会变得不可靠。分开处理让每一层都使用最适合自己上下文范围的位置编码。

### 1.4 Per-Layer Embeddings：给每一层一个独立的"记忆通道"

这是 Gemma 4 小模型（E2B、E4B）最独特的架构特征。

在标准 Transformer 中，每个 token 的 embedding 在所有层中是同一个。这意味着第一层的 embedding 需要预先编码这个 token 在所有层次上可能需要的信息——对一个固定维度的向量来说这是不现实的约束。PLE 的做法是**为每一个 decoder 层维护一个独立的小型 embedding table**，每个 token 在每一层都会收到一个专属信号。

效果上，E2B 的总参数是 5.1B，但有效参数只有 2.3B——另外 2.8B 是 PLE 的 embedding table。这些参数在磁盘上占空间，但计算成本极低（只是 embedding lookup + 小型投影），推理速度完全是一个 2B 模型的水平。**用存储换计算，用参数空间换算力空间。** 对于存储空间比算力便宜的边缘设备来说，这种 trade-off 非常合理。

### 1.5 MoE 128 专家：已成为行业共识

26B-A4B 是 Gemma 4 家族中最值得深入分析的模型。25.2B 总参数，每 token 激活 3.8B（8 个专家 + 1 个共享专家），性能达到 31B dense 模型的 97%。

128 个专家已经不再是 Gemma 4 的独创——Qwen3-235B-A22B 和 GLM-5 都采用了类似设计。三者在专家配置上的差异值得深究：

| 配置项 | Gemma 4 31B | Gemma 4 26B-A4B | Qwen3-235B-A22B | GLM-5 |
|--------|-----------|-----------------|-----------------|-------|
| 总参数 | 30.7B | 25.2B | 235B | 744B |
| 专家总数 | — | 128 | 128 | 256 |
| 每token激活 | — | 8 + 1 共享 | 8（无共享） | 8 + 1 共享 |
| 活跃参数 | 30.7B | 3.8B | 22B | 40B |
| 层数 | 60 | 30 | 94 | — |
| hidden_size | 5376 | 2816 | 4096 | — |
| Q heads | 32 | 16 | 64 | — |
| KV heads | 16 | 8 | 4 | MLA（无传统 KV） |
| head_dim | 256 | 256 | 128 | — |
| sliding_window | 1024 | 1024 | 无 | — |
| 上下文 | 256K | 256K | 256K（max） | 200K |

*Gemma 4 数据来源：HuggingFace config.json（`google/gemma-4-31B-it`, `google/gemma-4-26B-A4B-it`）。关键 config 字段：`num_global_key_value_heads`（global 层 KV head 数）、`global_head_dim`（global 层 head 维度翻倍）、`attention_k_eq_v`（K=V）、`rope_parameters.full_attention.partial_rotary_factor=0.25`（p-RoPE）、`layer_types`（5:1 交错模式）。Qwen3 数据来源：HuggingFace config.json + 技术报告。GLM-5 层数和详细配置未公开（权重未在 HuggingFace 上找到，数据来自技术报告和 Sebastian Raschka 的分析）。*

值得注意的是共享专家的有无分歧。Qwen3 明确去掉了共享专家（Qwen2.5-MoE 有共享专家），改用 global-batch load balancing loss 来促进专家专业化。Gemma 4 和 GLM-5 则保留了共享专家——这可能是针对不同模型规模的经验性选择：小专家系统（Gemma 4 每个 3.8B）可能更需要共享专家来兜底，而大专家系统（Qwen3 每个 ~1.8B）有足够的冗余容量。

128-256 个小专家相比早期的 8-16 大专家，核心优势在于路由的粒度。路由器可以做更精细的分发——不是"这段文本归你"，而是"这个词的这种语义组合归这个专家"。更细的粒度意味着每个专家学习的模式更专注，参数利用率更高。

## 二、训练篇：蒸馏是共同选择，差异在执行

架构优化解释了 Gemma 4 的效率，但没有完全解释它为什么这么强。一个架构精良但训练数据平庸的 31B 模型，不可能追平训练充分的 235B 模型。

### 2.1 三个团队的蒸馏策略

蒸馏不是 Gemma 4 的独门武器——三个团队都在做，而且都明确承认蒸馏优于纯 RL。

**Qwen3** 对小模型使用 "strong-to-weak distillation"，利用旗舰模型 235B-A22B 作为 teacher，通过 off-policy 和 on-policy 两种方式传递知识。技术报告原文："Distillation from advanced teacher models significantly outperforms reinforcement learning in performance and training efficiency."

**GLM-5** 在 post-training 的三个阶段（Reasoning RL → Agentic RL → General RL）中全程使用 On-Policy Cross-Stage Distillation 来防止灾难性遗忘，确保模型在获得新能力的同时保留推理基础。

**Gemma 4** 的架构明确源自 Gemini 3 研究。Google 没有公开完整的训练细节，但从已发表的 Chain-of-Thought 蒸馏研究来看，流程大致是：用 Gemini 3 对大量 prompt 生成推理链 → 过滤质量 → 用合成数据训练 instruction-tuning 阶段。

三个团队都在做蒸馏，关键差异在于 **Teacher 模型的能力上限**。Google 拥有 Gemini 3（作为 Gemini 系列的第三代旗舰闭源模型，其能力在公开 benchmark 上显著优于任何开源模型），作为 teacher 的信息质量上限天然高于 Qwen3-235B 或 GLM-5 自蒸馏。但这只是基于公开信息的推断——Google 没有公开 Gemma 4 蒸馏数据的具体来源和规模，这是分析中的一个已知盲区。

### 2.2 训练管线对比：三阶段的殊途同归

三个模型的预训练都采用了三阶段策略，但侧重点不同：

**Qwen3**（36T tokens）：
1. **General Stage (S1)**：30T tokens，4K 上下文，建立语言基础
2. **Reasoning Stage (S2)**：5T tokens，增加 STEM/代码/推理/合成数据比例
3. **Long Context Stage**：数千亿 tokens，4K→32K 上下文扩展，使用 ABF（base frequency 10K→1M）、YARN 和 Dual Chunk Attention (DCA)

YARN 通过在注意力分数中加入额外的距离衰减项来扩展上下文窗口，避免超出训练长度的 token 对之间注意力权重异常。DCA 则将长序列切分为多个 chunk，每个 chunk 内做完整的 self-attention，相邻 chunk 之间通过"successive-chunk attention"传递信息——这样即使模型只训练过 4K 上下文，也能无损扩展到 128K+。

**GLM-5**（28.5T tokens）：
1. **Base Model Training**：27T tokens，优先代码和推理数据
2. **Mid-training**：4K→200K 上下文渐进扩展，聚焦长上下文 agentic 数据
3. **Post-Training**：顺序 RL 管线——Reasoning RL → Agentic RL → General RL

GLM-5 最值得关注的是其 RL 基础设施。他们开发了 **slime**——一个异步 RL 框架，将生成和训练解耦，大幅提升 GPU 利用率和 RL 训练吞吐。这让他们能进行更细粒度的 post-training 迭代，包括异步 Agent RL 算法，让模型从复杂的长周期交互中学习。这种 RL 工程投入直接反映在 GLM-5 在 agentic benchmark 上的领先表现。

**Gemma 4**：Google 未公开训练数据量。从 WaveSpeed 的分析文章到 HuggingFace 的技术博客，均未提及具体 token 数。这与其他两个团队主动公开 28.5T/36T 形成对比，是分析中的一个信息缺口。

### 2.3 QAT：训练时就考虑量化

Gemma 4 提供了 Quantization-Aware Training (QAT) checkpoint——在训练阶段就引入量化噪声，让模型学会在低精度表示下保持输出质量。NVIDIA 已发布 Gemma-4-31B-IT-NVFP4，量化到 4-bit 浮点后精度损失极小。

Qwen3 和 GLM-5 主要依赖后量化（GPTQ/AWQ），GLM-5 额外提供了官方 FP8 权重。后量化的工具链更成熟、社区生态更丰富，但理论上 QAT 能做到更小的精度损失。

### 2.4 多模态：原生融合 vs 模型即工具

Gemma 4 的多模态能力在预训练阶段就与文本一起训练。视觉编码器基于 ViT，使用 2D RoPE 编码 patch 的二维空间位置，支持可变宽高比和可配置的 soft token budget（70-1120 tokens）。音频编码器沿用 Gemma-3n 的 USM-style conformer，从 681M 压缩到 305M。

GLM-5 走了不同路线——原生不处理图像/音频/视频，而是通过 tool-calling 调用 GLM 家族的专用模型（GLM-Image、GLM-4.6V、GLM-Vision）。这种"模型即工具"的设计在 agentic 场景下更灵活，但端到端延迟和一致性不如原生融合。

Qwen3 的视觉能力由 Qwen3-VL 系列承担，基础语言模型本身不直接处理图像。

## 三、对比篇：三条不同的技术路线

### 3.1 Benchmark 对比

以下使用各模型技术报告或官方 benchmark 中**可查证的数字**。注意 AIME 2024、2025、2026 是不同的试卷，跨年数据不具备严格可比性：

| 模型 | 总参数 | 活跃参数 | AIME | MMLU Pro | SWE-bench |
|------|--------|----------|------|----------|-----------|
| Gemma 4 31B | 30.7B | 30.7B | 89.2% ('26) | 85.2% | — |
| Gemma 4 26B-A4B | 25.2B | 3.8B | 88.3% ('26) | 82.6% | — |
| Qwen3-235B-A22B | 235B | 22B | 85.7% ('24) / 81.5% ('25) | — | — |
| GLM-5 | 744B | 40B | 93.3% ('25) | 80.6% | 77.8% |

几个关键观察：

- **GLM-5 在 AIME 2025 上表现最强**（93.3%），超过了 Gemma 4 在 AIME 2026 上的 89.2%。虽然跨年比较不严谨，但 GLM-5 用 RL 工程在数学推理上取得的进步是实打实的。
- **Gemma 4 26B-A4B 在 AIME 2026 上仅比 31B dense 低 0.9 个百分点**，但活跃参数从 30.7B 降到 3.8B——8 倍的活跃参数差距只换来不到 1% 的性能损失。
- **GLM-5 在 agentic benchmark 上没有对手**：SWE-bench 77.8%、Vending Bench 2 $4,432（开源 #1）。这印证了我们的核心洞察——参数规模在复杂规划任务上仍然重要。

### 3.2 注意力架构对比：三条路线的本质区别

这是三个模型在架构层面最本质的差异。

**Gemma 4：Sliding Window + Global 交错注意力**

层交替使用 local sliding window（512-1024 token）和 global full-context attention，通过 Shared KV Cache、K=V、GQA 8:1、p-RoPE 将 global 层开销压缩到极致。设计目标是**在固定显存预算内最大化上下文长度**。

**Qwen3：标准 Full Attention + YARN/DCA 扩展**

沿袭 Qwen2.5 的标准 Full Attention 架构，94 层，64Q/4KV 的 GQA。长上下文不是通过稀疏化实现，而是通过 RoPE base frequency 调整（ABF）+ YARN 的距离衰减 + DCA 的分块注意力来扩展。设计目标是**架构简洁性和通用性**，不改变注意力机制本身，只在位置编码和注意力模式上做适配。

**GLM-5：MLA + DSA 双重压缩**

这是三者中最激进的方案。GLM-5 抛弃了传统 KV Cache，改用 **Multi-Head Latent Attention (MLA)**——将 Key 和 Value 投影到一个低维潜在空间中，推理时只需要缓存压缩后的潜在向量，大幅减少显存占用。在此基础上叠加 **DeepSeek Sparse Attention (DSA)**：使用一个轻量级的 "lightning indexer" 为每个 token 动态选出最相关的 top-k 个历史 token，让注意力计算只在选中的子集上进行。MLA 解决了 KV Cache 的显存瓶颈，DSA 解决了注意力计算的算力瓶颈，两者叠加实现了 200K 上下文的高效处理。

三条路线的选择反映了不同的工程优先级：
- Gemma 4 追求**极致推理效率**，通过多层压缩降低单次注意力的开销
- Qwen3 追求**架构简洁性和通用性**，通过成熟组件的组合达到稳定效果
- GLM-5 追求**长上下文 agentic 能力**，MLA + DSA 的双重压缩天然适合多步工具调用中需要频繁回溯长历史记录的场景

### 3.3 位置编码对比

| 模型 | 位置编码策略 | 核心特点 |
|------|------------|---------|
| Gemma 4 | Dual RoPE：Sliding 层标准 RoPE，Global 层 p-RoPE (p=0.25) | 低频维度不旋转，保留语义信息 |
| Qwen3 | 标准 RoPE + ABF (10K→1M) | 通过 base frequency 调整支持长上下文 |
| GLM-5 | 标准 RoPE + DSA 内部处理 | DSA 的 lightning indexer 隐式处理了长距离依赖 |

Gemma 4 的 p-RoPE 是三者中最独特的。它承认了一个事实：在长上下文中，不是所有维度都需要位置信息。低频维度在标准 RoPE 下会成为位置噪声的载体，p-RoPE 干脆不旋转它们。

### 3.4 量化策略对比

| 模型 | 量化方案 | 特点 |
|------|---------|------|
| Gemma 4 | QAT checkpoint | NVIDIA 发布 NVFP4 版本，训练时感知量化 |
| Qwen3 | 后量化（GPTQ/AWQ） | 社区生态丰富，工具链成熟 |
| GLM-5 | 官方 FP8 + 后量化 | 专门提供 FP8 权重，适配 Hopper/Blackwell |

### 3.5 生态与许可证

Gemma 4 采用 Apache 2.0，Qwen3 同样 Apache 2.0，GLM-5 也开源了权重。三者当前都是完全开源的。

但趋势值得关注：Qwen3.5-Omni 和 Qwen 3.6 Plus 已开始从完全开源转向更受限的发布策略，不再公开全部权重。Google 在中国 AI 实验室收紧开源的同时选择 Apache 2.0 全面开放，这个时间点的选择意味深长。

## 四、三个模型的差异化定位

经过架构和训练两个层面的拆解，三者的差异化定位已经清晰：

**Gemma 4：效率至上。** 每一个架构选择都在压缩推理开销——K=V、p-RoPE、GQA 8:1 global 层、PLE、QAT。它不是最强的模型，但在给定算力预算下可能是性价比最高的。适合对推理成本敏感、需要广泛部署的场景。

**Qwen3：通用均衡。** 架构选择相对保守（标准 Full Attention + 成熟组件组合），通过规模（36T tokens、235B 参数）和扎实的数据工程达到 SOTA。开源生态最完善，社区支持最好。适合需要稳定可靠、生态丰富的通用场景。

**GLM-5：Agentic 专精。** MLA + DSA 双重注意力压缩 + slime 异步 RL + Agent RL 算法，整个管线都在为 agentic 能力服务。SWE-bench 77.8%、Vending Bench 2 开源 #1、AIME 2025 93.3%。同时在七种国产 GPU 上做了深度适配。适合面向中国市场、以 agentic coding 为核心的场景。

## 五、对行业的深层影响

### 5.1 "小模型 + 大 Teacher" 正在成为新范式

Gemma 4 不是孤例。Microsoft 的 Phi 系列和 Apple 的 OpenELM 也在走类似路线。Qwen3 也明确承认蒸馏优于 RL。但这个范式有一个结构性的不公平：模型能力的上限由 Teacher 模型决定，而最强的 Teacher 模型（Gemini 3、GPT-5 等）掌握在少数闭源团队手中。开源模型之间的竞争正在从"谁的参数多"转向"谁的蒸馏策略好"——但竞争的天花板仍然由闭源模型设定。

### 5.2 注意力架构正在分化

三种注意力方案（Sliding+Global 交错、标准 Full + YARN/DCA、MLA + DSA）代表了三种不同的哲学。这种分化意味着推理引擎需要同时支持多种注意力模式，vLLM 和 SGLang 的适配工作量会持续增加。

### 5.3 RL 基础设施成为新的竞争维度

GLM-5 的 slime 框架表明，RL 训练效率本身已经成为模型竞争力的组成部分。能更高效地进行 RL 迭代的团队，能在 post-training 阶段拉开差距。GLM-5 在 AIME 2025 上 93.3% 的分数很大程度上归功于 Reasoning RL 的深度投入。这对 RL 基础设施（veRL、OpenRLHF、MindSpeed-RL）的开发者是利好消息。

## 六、结论

Gemma 4 告诉我们的事情，远比一个 benchmark 排名重要：

**在数学推理和编程任务上，参数效率的边界比我们想象的要远。** 31B 参数可以达到过去需要 200B+ 参数才能达到的性能水平。Gemma 4 26B-A4B 用 3.8B 活跃参数只比 31B dense 低不到 1%，这是架构效率和训练质量叠加的结果。

**但效率优化有明确的边界。** GLM-5 用 744B 参数在 SWE-bench（77.8%）和 Vending Bench 2 上取得的开源 SOTA，证明了复杂 agentic 规划能力仍然与总参数量正相关。Gemma 4 在这些任务上没有公布 comparable 数据——这可能本身就说明了问题。

**蒸馏策略是当前最大的隐性变量。** 三个团队都在做蒸馏，但 Teacher 模型的能力上限不同。Google 没有公开 Gemma 4 的蒸馏数据来源和规模，这让我们无法精确量化蒸馏在 Gemma 4 的成功中贡献了多少——这是当前分析中最大的已知盲区。

**没有一种架构是"最优解"。** Gemma 4 的极致效率、Qwen3 的通用均衡、GLM-5 的 agentic 专精，分别服务于不同的场景和优先级。选择哪个模型，取决于你的部署约束、目标场景和生态系统偏好。

对于 AI Infra 工程师来说，Gemma 4 的发布意味着一个新的工作重心：**不再只是让大模型跑得更快，而是理解不同架构的效率特性，为每种模型选择最适合的推理策略。**

---

*参考来源：*
- [Google Blog: Gemma 4](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/)
- [HuggingFace Blog: Welcome Gemma 4](https://huggingface.co/blog/gemma4)
- [A Visual Guide to Gemma 4 — Maarten Grootendorst](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-gemma-4)
- [VentureBeat: Google releases Gemma 4 under Apache 2.0](https://venturebeat.com/technology/google-releases-gemma-4-under-apache-2-0-and-that-license-change-may-matter)
- [Qwen3 Technical Report (arXiv:2505.09388)](https://arxiv.org/html/2505.09388v1)
- [GLM-5 Technical Report (arXiv:2602.15763)](https://arxiv.org/html/2602.15763v2)
- [GLM-5 GitHub Repository](https://github.com/zai-org/GLM-5)
- [Qwen3-235B-A22B on HuggingFace](https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507)
- [NVIDIA: Gemma-4-31B-IT-NVFP4](https://huggingface.co/nvidia/Gemma-4-31B-IT-NVFP4)
- [WaveSpeed: What Is Google Gemma 4?](https://wavespeed.ai/blog/posts/what-is-google-gemma-4/)
- [LayerLens: GLM-5 Benchmark Review (AIME 2025: 93.33%, SWE-bench: 77.8%)](https://layerlens.ai)
- [Sebastian Raschka: GLM-5 Architecture (256 experts, MLA)](https://www.linkedin.com/posts/sebastianraschka_on-an-llm-time-scale-it-has-been-a-while-activity-7427718512284012545-QH9V)
- [Xianbao QIAN: GLM-5 Deep Dive (MLA + DSA)](https://x.com/Xianbao_QIAN/status/2023199756009591250)
