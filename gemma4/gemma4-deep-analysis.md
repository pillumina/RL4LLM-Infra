# Gemma 4 架构深度拆解：31B 参数的压缩艺术（附 Qwen3.5、GLM-5 对比）

> 2026 年 4 月 2 日，Google DeepMind 发布 Gemma 4。31B 参数的 dense 模型在 AIME 2026 上拿到 89.2%，MoE 变体 26B-A4B 以 3.8B 活跃参数在 MMLU Pro 上达到 31B dense 模型的 97%。同期 Qwen3.5-27B 以 27B dense 参数在 SWE-bench 上追平 GPT-5 mini（72.4），Qwen3.5-35B-A3B 仅 3B 活跃参数就达到相近水平——而 GLM-5 以 744B 总参数拿下开源 agentic SOTA。四款模型在 3B-744B 的参数跨度上展开了一场技术路线的全面对抗。

本文要拆解的是 Gemma 4 在架构和训练层面做出的每一项技术选择：K=V、p-RoPE、128 小专家 + Dense MLP 双路径、Per-Layer Embeddings——每一项单独看都不算革命性，但叠加在一起的效果是惊人的。

## 核心洞察

Gemma 4 和 Qwen3.5 同时证明了**架构效率和训练质量可以在特定任务上大幅压缩参数需求**——Gemma 4 31B 在数学推理 benchmark 上超越 200B+ 级别模型，Qwen3.5-27B 以 27B dense 在 SWE-bench 上追平 GPT-5 mini，Qwen3.5-35B-A3B 仅 3B 活跃参数就达到接近水平。

但这个效率增益不是无条件的。GLM-5 用 744B 参数在 SWE-bench（77.8%）和 Vending Bench 2 等长周期 agentic 任务上取得开源 SOTA，说明**复杂规划能力的上限仍然与总参数量正相关**。

理解效率优化在哪里生效、在哪里失效，比单纯崇拜"小模型"更有价值。

## 一、架构篇：每一分算力都要花在刀刃上

Gemma 4 的基础仍然是 decoder-only Transformer，但 Google 在几乎每一个组件上都做了有针对性的优化。这些优化单独看都不算革命性，但叠加起来产生的效果是显著的。

### 1.1 KV 共享与边缘模型优化

Transformer 推理时，显存消耗的大头是 KV Cache——每一层都要缓存所有历史 token 的 Key 和 Value 向量。对于 256K 上下文的模型来说，这部分显存可以轻松超过模型权重本身。

Gemma 4 在边缘模型（E2B、E4B）中采用了一种直接的做法：**让最后 N 层直接复用前面同类型 attention 层的 KV 张量，不再自己计算 K 和 V 投影。** 从 HuggingFace config.json 可以验证：E2B 有 35 层，其中 20 层共享 KV（`num_kv_shared_layers=20`）；E4B 有 42 层，其中 18 层共享 KV（`num_kv_shared_layers=18`）。这不是近似或压缩——复用的是精确计算过的 KV。深层网络中，相邻层学到的 KV 表示往往高度相似，独立计算本质上是冗余工作。

但值得注意的是，31B 和 26B-A4B 的 `num_kv_shared_layers=0`——发布版的大模型**并未启用**这一优化。这说明 Google 认为在参数量较大的模型中，每层独立计算 KV 的信息增益值得保留，KV 共享主要服务于算力受限的边缘设备场景。

### 1.2 Global Attention 的五重压缩

在混合注意力机制中，global attention 层是最昂贵的——它们需要对整个上下文序列做全注意力计算。Gemma 4 在 global attention 层上叠加了五重优化，形成一个完整的设计链：

起点是 **GQA**——模型整体使用适度的分组（31B 为 32Q/16KV，26B-A4B 为 16Q/8KV），但在 global attention 层，KV head 数量进一步压缩：31B 从 16 个降到 4 个（`num_global_key_value_heads=4`），26B-A4B 从 8 个降到 2 个（`num_global_key_value_heads=2`），均达到 **GQA 8:1**。但这显然会损失信息容量。Google 的应对是 **Key 维度翻倍**：global 层的 head_dim 从 256 翻倍到 512（`global_head_dim=512`），更宽的 KV 向量能在更少的 head 数量下保留足够的信息。在这个基础上，进一步让 **K = V**（`attention_k_eq_v=True`）——直接让 Key 向量等于 Value 向量，KV Cache 再减半。这强制模型在"检索"（Key 的角色）和"读取"（Value 的角色）之间使用同一套表示，降低了表示的自由度——共享表示约束了模型的拟合空间，客观上起到了正则化效果。

到这里，KV Cache 已经被压缩到了最小。但还有一个问题需要解决：长上下文下的位置编码失真。标准 RoPE 对向量的所有维度都施加旋转，但在 256K 的距离上，低频维度的旋转极其微小，累积起来会在远距离 token 之间引入对齐误差，干扰语义匹配。Gemma 4 的解决方案是 **p-RoPE**：只对 25% 的高频维度施加旋转，低频维度纯粹保留语义信息，不被位置噪声污染。

最后，**最后一层强制 Global**——无论 interleaving pattern 怎么排，最后一层一定是 global attention，确保输出 token 能看到完整的输入上下文。

把这五层因果关系放在一起：GQA 8:1 压缩 KV head 数量 → Key 维度翻倍补偿信息损失 → K=V 进一步减半 → p-RoPE 解决长距离位置噪声 → 最后一层 Global 保证全局可见。设计哲学很清晰：global attention 层的目标不是保留最多的信息，而是在保留足够信息的前提下，把开销压缩到最低。

### 1.3 双 RoPE：参数差异背后的设计逻辑

1.2 节提到的 p-RoPE 和标准 RoPE 的分工，从 config.json 可以看到完整的参数差异：sliding window 层使用 `rope_theta=10000`，global 层使用 `partial_rotary_factor=0.25` + `rope_theta=1000000`。两者的上下文窗口也不同——边缘模型 E2B/E4B 的 sliding window 为 512 token，31B/26B-A4B 为 1024 token。短窗口 + 标准 RoPE，长窗口 + p-RoPE，各取所需。

### 1.4 Per-Layer Embeddings：给每一层一个独立的"记忆通道"

这是 Gemma 4 小模型（E2B、E4B）最独特的架构特征。

在标准 Transformer 中，每个 token 的 embedding 在所有层中是同一个。这意味着第一层的 embedding 需要预先编码这个 token 在所有层次上可能需要的信息——对一个固定维度的向量来说这是不现实的约束。PLE 的做法是**为每一个 decoder 层维护一个独立的小型 embedding table**，每个 token 在每一层都会收到一个专属信号。

效果上，E2B 的总参数是 5.1B，但有效参数只有 2.3B——另外 2.8B 是 PLE 的 embedding table。这些参数在磁盘上占空间，但计算成本极低（只是 embedding lookup + 小型投影），推理速度完全是一个 2B 模型的水平。**用存储换计算，用参数空间换算力空间。** 对于存储空间比算力便宜的边缘设备来说，这种 trade-off 非常合理。

### 1.5 MoE 128 专家 + Dense MLP 双路径：独特的混合架构

26B-A4B 是 Gemma 4 家族中最值得深入分析的模型。25.2B 总参数，每 token 激活约 3.8B，性能达到 31B dense 模型的 97%（MMLU Pro 82.6% vs 85.2%），AIME 2026 仅低 0.9 个百分点。

但它的 MoE 实现方式与其他模型有本质区别。从 HuggingFace 的模型实现代码（`modeling_gemma4.py`）可以确认，26B-A4B 每层包含**两个并行的 FFN 路径**：一个标准的 dense MLP（`intermediate_size=2112`）和一个 routed MoE block（128 个专家，每 token 选 8 个，每个专家 `moe_intermediate_size=704`）。两条路径的输出通过 layer norm 后相加合并。

这与 Qwen3（纯 MoE，无辅助路径）和 Qwen3.5/GLM-5（256 experts + shared expert）的架构都不同。dense MLP 充当了类似共享专家的角色，但它是一个独立的、不依赖路由的全量计算路径，提供了不依赖路由器决策的稳定基础信号。Qwen3.5 在 Qwen3 基础上做了同样的判断——加入了 shared expert——说明纯 MoE 缺乏稳定基底已成为共识。

128-256 个小专家已成为 2025-2026 年旗舰 MoE 模型的标准配置。Qwen3.5 在 Qwen3 的基础上做了关键架构转向——从纯 MoE 改为 256 experts + shared expert，与 GLM-5 的设计方向趋同。四款 MoE 模型的配置对比（Gemma 4 31B 为 dense 模型，不参与 MoE 对比）：

| 配置项 | Gemma 4 26B-A4B | Qwen3.5-35B-A3B | Qwen3-235B-A22B | GLM-5 |
|--------|-----------------|-----------------|-----------------|-------|
| 总参数 | 25.2B | 35B | 235B | 744B |
| 专家总数 | 128 | 256 | 128 | 256 |
| 辅助路径 | dense MLP (2112) | shared expert (512) | 无 | shared expert |
| 每token激活 | 8 MoE + dense MLP | 8 MoE + shared | 8（纯 MoE） | 8 + 1 共享 |
| 活跃参数 | 3.8B | 3B | 22B | 40B |
| 层数 | 30 | 40 | 94 | 78 |
| hidden_size | 2816 | 2048 | 4096 | 6144 |
| Q heads | 16 | 16 | 64 | 64（MLA 压缩） |
| KV heads | 8 | 2 | 4 | MLA（kv_lora_rank=512） |
| head_dim | 256 | 256 | 128 | 64（qk_head_dim=256） |
| 注意力 | sliding + global | linear + full | full attention | MLA + DSA |
| 上下文 | 256K | 262K | 256K（max） | 200K |
| 多模态 | ❌ text-only | ✅ native | ❌ (VL 分离) | ❌ (tool-calling) |

*数据来源：Gemma 4 来自 HuggingFace config.json（[31B](https://huggingface.co/google/gemma-4-31B-it/blob/main/config.json)、[26B-A4B](https://huggingface.co/google/gemma-4-26B-A4B-it/blob/main/config.json)）。Qwen3.5 来自 HuggingFace config.json（[configuration_qwen3_5_moe.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5_moe/configuration_qwen3_5_moe.py)）和官方 blog。Qwen3 来自 HuggingFace config.json + 技术报告。GLM-5 来自 HuggingFace config.json（[zai-org/GLM-5](https://huggingface.co/zai-org/GLM-5/blob/main/config.json)）和技术报告。*

**MoE 路由策略的趋同与分化。** Qwen3.5 从 Qwen3 的纯 MoE 转向 256 experts + shared expert，与 GLM-5 的设计方向一致——两者都认为需要一个不依赖路由的全量计算路径来稳定基础表达能力。Gemma 4 的 dense MLP 本质上也扮演了类似角色，但实现方式不同：它是与 MoE 并行的独立路径，而非被路由器选中的共享专家。三种设计的选择反映了不同的工程哲学：
- **Gemma 4 的 dense MLP**：独立于路由的全量计算路径，每次前向传播都执行，提供最稳定的基底信号
- **GLM-5/Qwen3.5 的 shared expert**：被路由器视为一个可选专家，理论上可以被跳过，但通过 router bias 确保始终被选中
- **Qwen3 的纯 MoE**：追求最激进的参数效率，但牺牲了基础表达稳定性

128-256 个小专家相比早期的 8-16 大专家，核心优势在于路由的粒度。更细的粒度意味着每个专家学习的模式更专注，参数利用率更高。Gemma 4 每个 MoE 专家的 intermediate_size 仅 704 维（对比 Qwen3 的约 12K 维、Qwen3.5 的 512 维），路由粒度最细。

## 二、训练篇：蒸馏是共同选择，差异在执行

架构优化解释了 Gemma 4 的效率，但没有完全解释它为什么这么强。一个架构精良但训练数据平庸的 31B 模型，不可能追平训练充分的 235B 模型。

### 2.1 三个团队的蒸馏策略

蒸馏不是 Gemma 4 的独门武器——三个团队都在做，而且都明确承认蒸馏优于纯 RL。

**Qwen3 / Qwen3.5** 对小模型使用 strong-to-weak distillation，利用旗舰模型作为 teacher。Qwen3 技术报告原文：“Distillation from advanced teacher models significantly outperforms reinforcement learning in performance and training efficiency.” Qwen3.5 延续这一策略，397B 旗舰作为 teacher 蒸馏出 35B-A3B、27B、122B-A10B 等中端模型。

**GLM-5** 在 post-training 的三个阶段（Reasoning RL → Agentic RL → General RL）中全程使用 On-Policy Cross-Stage Distillation 来防止灾难性遗忘，确保模型在获得新能力的同时保留推理基础。

**Gemma 4** 的架构明确源自 Gemini 3 研究。Google 没有公开完整的训练细节，但从已发表的 Chain-of-Thought 蒸馏研究来看，流程大致是：用 Gemini 3 对大量 prompt 生成推理链 → 过滤质量 → 用合成数据训练 instruction-tuning 阶段。

三个团队都在做蒸馏，关键差异在于 **Teacher 模型的能力上限**。Google 拥有 Gemini 3 作为 teacher。作为 Gemini 系列的第三代旗舰闭源模型，其能力在公开 benchmark 上显著优于任何开源模型，信息质量上限天然高于 Qwen3-235B 或 GLM-5 自蒸馏。但这只是基于公开信息的推断——Google 没有公开 Gemma 4 蒸馏数据的具体来源和规模，这是分析中的一个已知盲区。

### 2.2 训练管线对比：三阶段的殊途同归

四个模型的预训练都采用了三阶段策略，但侧重点不同。以下只讲训练流程和 RL 策略，架构层面的位置编码和注意力扩展技术见第三章。

**Qwen3**（36T tokens）和 **Qwen3.5**：
1. **General Stage (S1)**：30T tokens，4K 上下文，建立语言基础
2. **Reasoning Stage (S2)**：5T tokens，增加 STEM/代码/推理/合成数据比例
3. **Long Context Stage**：数千亿 tokens，4K→32K 上下文扩展（Qwen3 使用 ABF + YARN + DCA）
4. **Qwen3.5 的架构跃迁**：从标准 Full Attention 切换到 Gated DeltaNet linear attention + full attention 的 hybrid 架构（3:1 交替），上下文扩展到 262K。linear attention 层将序列压缩为固定大小状态，消除了传统 KV Cache 的显存瓶颈

**GLM-5**（28.5T tokens）：
1. **Base Model Training**：27T tokens，优先代码和推理数据
2. **Mid-training**：4K→200K 上下文渐进扩展，聚焦长上下文 agentic 数据
3. **Post-Training**：顺序 RL 管线——Reasoning RL → Agentic RL → General RL

GLM-5 最值得关注的是其 RL 基础设施。他们开发了 **slime**——一个异步 RL 框架，将生成和训练解耦，大幅提升 GPU 利用率和 RL 训练吞吐。这让他们能进行更细粒度的 post-training 迭代，包括异步 Agent RL 算法，让模型从复杂的长周期交互中学习。这种 RL 工程投入直接反映在 GLM-5 在 agentic benchmark 上的领先表现。

**Gemma 4**：Google 未公开训练数据量。从 WaveSpeed 的分析文章到 HuggingFace 的技术博客，均未提及具体 token 数。这与其他两个团队主动公开 28.5T/36T 形成对比，是分析中的一个信息缺口。

### 2.3 QAT：训练时就考虑量化

Gemma 4 提供了 Quantization-Aware Training (QAT) checkpoint——在训练阶段就引入量化噪声，让模型学会在低精度表示下保持输出质量。NVIDIA 已发布 Gemma-4-31B-IT-NVFP4，量化到 4-bit 浮点后精度损失极小。

Qwen3 和 GLM-5 主要依赖后量化（GPTQ/AWQ），GLM-5 和 Qwen3.5 额外提供了官方 FP8 权重。后量化的工具链更成熟、社区生态更丰富，但理论上 QAT 能做到更小的精度损失。

### 2.4 多模态：原生融合 vs 模型即工具

Gemma 4 的多模态能力在预训练阶段就与文本一起训练。视觉编码器基于 ViT，使用 2D RoPE 编码 patch 的二维空间位置，支持可变宽高比和可配置的 soft token budget（70-1120 tokens）。音频编码器沿用 Gemma-3n 的 USM-style conformer，从 681M 压缩到 305M。

GLM-5 走了不同路线——原生不处理图像/音频/视频，而是通过 tool-calling 调用 GLM 家族的专用模型（GLM-Image、GLM-4.6V、GLM-Vision）。这种"模型即工具"的设计在 agentic 场景下更灵活，但端到端延迟和一致性不如原生融合。

Qwen3 的视觉能力由 Qwen3-VL 系列承担。**Qwen3.5 做了根本性的改变**——所有中端模型（35B-A3B、27B、122B-A10B）都是原生多模态的，视觉编码器与语言模型一起预训练。这是四款模型中唯一做到中端模型原生多模态的。

## 三、对比篇：四条不同的技术路线

### 3.1 Benchmark 对比

以下使用各模型技术报告或 HuggingFace 官方 model card 中**可查证的数字**。注意 AIME 2024、2025、2026 是不同的试卷，跨年数据不具备严格可比性：

| 模型 | 总参数 | 活跃参数 | AIME | MMLU Pro | SWE-bench |
|------|--------|----------|------|----------|-----------|
| Gemma 4 31B | 30.7B | 30.7B | 89.2% ('26) | 85.2% | — |
| Gemma 4 26B-A4B | 25.2B | 3.8B | 88.3% ('26) | 82.6% | — |
| Qwen3.5-27B | 27B | 27B | — | 86.1% | 72.4% |
| Qwen3.5-35B-A3B | 35B | 3B | — | 85.3% | 69.2% |
| Qwen3.5-122B-A10B | 122B | 10B | — | 86.7% | 72.0% |
| Qwen3-235B-A22B | 235B | 22B | 85.7% ('24) / 81.5% ('25) | — | — |
| GLM-5 | 744B | 40B | 93.3% ('25) | 80.6% | 77.8% |

几个关键观察：

- **GLM-5 在 AIME 2025 上表现最强**（93.3%），超过了 Gemma 4 在 AIME 2026 上的 89.2%。虽然跨年比较不严谨，但 GLM-5 用 RL 工程在数学推理上取得的进步是实打实的。
- **Gemma 4 26B-A4B 在 AIME 2026 上仅比 31B dense 低 0.9 个百分点**，但活跃参数从 30.7B 降到 3.8B——8 倍的活跃参数差距只换来不到 1% 的性能损失。
- **Qwen3.5-27B 以 27B dense 在 SWE-bench 上追平 GPT-5 mini（72.4）**，Qwen3.5-35B-A3B 仅 3B 活跃参数就达到 69.2%。这是架构效率的又一次验证——与 Gemma 4 的结论相互印证。
- **GLM-5 在 agentic benchmark 上没有对手**：SWE-bench 77.8%、Vending Bench 2 $4,432（开源 #1）。这印证了我们的核心洞察——参数规模在复杂规划任务上仍然重要。回到开头的判断：Gemma 4 和 Qwen3.5 在推理和编程任务上确实逼近了 200B+ 级别的水平，但这个“逼近”有明确的适用边界。

### 3.2 注意力架构对比：四条路线的本质区别

这是四个模型在架构层面最本质的差异。

**Gemma 4：Sliding Window + Global 交错注意力**

层交替使用 local sliding window（1024 token）和 global full-context attention（5:1 交错比例，从 `layer_types` 验证），通过 K=V、GQA 8:1、p-RoPE 将 global 层开销压到最低。设计目标是**在固定显存预算内最大化上下文长度**。

**Qwen3.5：Gated DeltaNet Linear Attention + Full Attention 混合**

Qwen3.5 在注意力架构上做了代际跃迁——从 Qwen3 的标准 Full Attention 切换到 hybrid 架构。每 4 层中 3 层使用 Gated DeltaNet linear attention（结合 Mamba2 的 gated decay 和 delta rule），1 层使用标准 full attention。linear attention 层将输入序列压缩为固定大小状态，计算复杂度从 O(n²) 降到近 O(n)，消除了传统 KV Cache 的显存瓶颈。

值得注意的是，Qwen3.5 与 Gemma 4 在注意力设计上选择了**相同的 hybrid 策略**——用高效注意力处理大部分层，用 full attention 保留精确推理能力。但具体实现不同：
- Gemma 4 用 sliding window（局部注意力）+ global（全局注意力）交替，KV Cache 仍然存在，只是通过 K=V 和 GQA 压缩
- Qwen3.5 用 linear attention + full attention 交替，linear attention 层**完全消除了 KV Cache**

这使得 Qwen3.5-35B-A3B 能在 8GB+ 显存上运行（配合 GGUF 量化），同时支持 262K 上下文。设计目标是**长上下文效率**，兼顾 MoE 的参数效率。

**GLM-5：MLA + DSA 双重压缩**

这是三者中最激进的方案。GLM-5 抛弃了传统 KV Cache，改用 **Multi-Head Latent Attention (MLA)**——将 Key 和 Value 投影到一个低维潜在空间中，推理时只需要缓存压缩后的潜在向量，大幅减少显存占用。在此基础上叠加 **DeepSeek Sparse Attention (DSA)**：使用一个轻量级的 lightning indexer（`index_n_heads=32`, `index_head_dim=128`, `index_topk=2048`）为每个 token 动态选出最相关的 top-k 个历史 token，让注意力计算只在选中的子集上进行。MLA 解决了 KV Cache 的显存瓶颈，DSA 解决了注意力计算的算力瓶颈，两者叠加实现了 200K 上下文的高效处理。设计目标是**长上下文 agentic 能力**，MLA + DSA 的双重压缩天然适合多步工具调用中需要频繁回溯长历史记录的场景。

四条路线的选择反映了不同的工程优先级：
- Gemma 4 追求**推理效率最大化**，通过 K=V、GQA、sliding window 等传统组件的组合降低单次注意力开销
- Qwen3.5 追求**长上下文效率**，通过 linear attention 消除 KV Cache，是架构变动最激进的选择
- GLM-5 追求**长上下文 agentic 能力**，MLA + DSA 的双重压缩天然适合多步工具调用中需要频繁回溯长历史记录的场景
- Qwen3 追求**架构简洁性和通用性**，通过成熟组件的组合达到稳定效果

**一个值得注意的趋势：** Gemma 4 和 Qwen3.5 都选择了 hybrid attention（高效注意力 + 间歇性 full attention），说明“全量注意力太贵、纯稀疏注意力不够”已成为共识。差异在于如何实现“高效”——Gemma 4 用 sliding window，Qwen3.5 用 linear attention，GLM-5 用 MLA + DSA。三种高效注意力的实现各有取舍，但方向一致。

### 3.3 量化策略对比

| 模型 | 量化方案 | 特点 |
|------|---------|------|
| Gemma 4 | QAT checkpoint | NVIDIA 发布 NVFP4 版本，训练时感知量化 |
| Qwen3 / Qwen3.5 | 后量化（GPTQ/AWQ）+ 官方 FP8 | Qwen3.5 额外提供官方 FP8 权重 |
| GLM-5 | 官方 FP8 + 后量化 | 专门提供 FP8 权重，适配 Hopper/Blackwell |

### 3.4 生态与许可证

四款模型均采用 Apache 2.0 完全开源，对社区和商业用户都非常友好。

## 四、四款模型的差异化定位

经过架构和训练两个层面的拆解，四者的差异化定位已经清晰：

**Gemma 4：效率至上。** 每一个架构选择都在压缩推理开销——K=V、p-RoPE、GQA 8:1 global 层、PLE、QAT。它不是最强的模型，但在给定算力预算下可能是性价比最高的。适合对推理成本敏感、需要广泛部署的场景。

**Qwen3.5：架构激进，全面进化。** 从 Qwen3 的标准架构跃迁到 Gated DeltaNet hybrid attention + MoE shared expert，是四款模型中架构变动最大的。所有中端模型都是原生多模态，35B-A3B 仅 3B 活跃参数就能在 SWE-bench 上拿到 69.2%。适合需要长上下文、多模态、高性价比推理的场景。

**GLM-5：Agentic 专精。** MLA + DSA 双重注意力压缩 + slime 异步 RL + Agent RL 算法，整个管线都在为 agentic 能力服务。SWE-bench 77.8%、Vending Bench 2 开源 #1、AIME 2025 93.3%。并提供了 Ascend NPU 的官方部署适配。适合面向中国市场、以 agentic coding 为核心的场景。

**Qwen3：通用均衡，生态最完善。** 架构选择相对保守——标准 Full Attention、纯 MoE（128 experts，无 shared expert）、成熟的位置编码方案（RoPE + ABF + YARN/DCA）。但通过 36T tokens 的训练规模和扎实的蒸馏策略，在开源社区建立了最完善的支持生态。Qwen3-235B-A22B 仍是当前最广泛部署的开源 MoE 模型之一。适合需要稳定可靠、社区支持最好的通用场景。

## 五、行业影响

### 5.1 "小模型 + 大 Teacher" 范式的结构性天花板

Gemma 4 不是孤例——Microsoft Phi、Apple OpenELM 都在走类似路线，Qwen3/Qwen3.5 也明确承认蒸馏优于 RL。但这个范式有一个结构性问题：模型能力的上限由 Teacher 决定，而最强的 Teacher（Gemini 3、GPT-5）掌握在闭源团队手中。开源模型之间的竞争正从"谁的参数多"转向"谁的蒸馏策略好"，但竞争的天花板仍由闭源模型设定。

### 5.2 RL 工程投入拉开差距

GLM-5 的 slime 异步 RL 框架表明，RL 训练效率本身已成为模型竞争力的组成部分。GLM-5 在 AIME 2025 上 93.3% 和 SWE-bench 77.8% 很大程度上归功于 Reasoning RL + Agent RL 的深度投入。这对 RL 基础设施（veRL、OpenRLHF、MindSpeed-RL）的开发者是明确的利好信号。

## 六、结论

Gemma 4 的架构选择值得逐条学习，但比单个技巧更重要的是整体思路：

**在数学推理和编程任务上，参数效率的边界比我们想象的要远。** 31B 参数可以达到过去需要 200B+ 参数才能达到的性能水平。Gemma 4 26B-A4B 用 3.8B 活跃参数只比 31B dense 低不到 1%，这是架构效率和训练质量叠加的结果。

**但效率优化有明确的边界。** GLM-5 用 744B 参数在 SWE-bench（77.8%）和 Vending Bench 2 上取得的开源 SOTA，证明了复杂 agentic 规划能力仍然与总参数量正相关。Gemma 4 在这些任务上没有公布 comparable 数据。

**蒸馏策略是当前最大的隐性变量。** 四个团队都在做蒸馏，但 Teacher 模型的能力上限不同。Google 没有公开 Gemma 4 的蒸馏数据来源和规模，这让我们无法精确量化蒸馏在 Gemma 4 的成功中贡献了多少——这是当前分析中最大的已知盲区。

**没有一种架构是"最优解"。** Gemma 4 的效率、Qwen3.5 的激进创新、Qwen3 的通用均衡、GLM-5 的 agentic 专精，分别服务于不同的场景和优先级。选择哪个模型，取决于你的部署约束、目标场景和生态系统偏好。

**Hybrid attention 正在成为新共识。** Gemma 4（sliding + global）和 Qwen3.5（linear + full）殊途同归——全量注意力太贵，纯稀疏不够，hybrid 是当前的最优解。GLM-5 的 MLA + DSA 本质上也是一种 hybrid（DSA 的 sparse attention 只在需要时激活）。这个趋势对推理引擎开发者意味着：同时支持多种注意力模式的适配工作量只会增加。

对于 AI Infra 工程师来说，这意味着：**理解每种架构的效率特性，比单纯堆算力更重要。**

---

*参考来源：*
- [Google Blog: Gemma 4](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/)
- [HuggingFace Blog: Welcome Gemma 4](https://huggingface.co/blog/gemma4)
- [A Visual Guide to Gemma 4 — Maarten Grootendorst](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-gemma-4)
- [VentureBeat: Google releases Gemma 4 under Apache 2.0](https://venturebeat.com/technology/google-releases-gemma-4-under-apache-2-0-and-that-license-change-may-matter)
- [Qwen3 Technical Report (arXiv:2505.09388)](https://arxiv.org/html/2505.09388v1)
- [Qwen3.5 Official Blog](https://qwen.ai/blog?id=qwen3.5)
- [Qwen3.5-35B-A3B on HuggingFace](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)
- [Qwen3.5-27B on HuggingFace](https://huggingface.co/Qwen/Qwen3.5-27B)
- [Qwen3.5 MoE Config (transformers)](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5_moe/configuration_qwen3_5_moe.py)
- [Qwen3.5 Dense Config (transformers)](https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen3_5/configuration_qwen3_5.py)
- [GLM-5 Technical Report (arXiv:2602.15763)](https://arxiv.org/html/2602.15763v2)
- [GLM-5 GitHub Repository](https://github.com/zai-org/GLM-5)
- [Qwen3-235B-A22B on HuggingFace](https://huggingface.co/Qwen/Qwen3-235B-A22B-Thinking-2507)
- [NVIDIA: Gemma-4-31B-IT-NVFP4](https://huggingface.co/nvidia/Gemma-4-31B-IT-NVFP4)
- [WaveSpeed: What Is Google Gemma 4?](https://wavespeed.ai/blog/posts/what-is-google-gemma-4/)
- [LayerLens: GLM-5 Benchmark Review (AIME 2025: 93.33%, SWE-bench: 77.8%)](https://layerlens.ai)
- [Sebastian Raschka: GLM-5 Architecture (256 experts, MLA)](https://www.linkedin.com/posts/sebastianraschka_on-an-llm-time-scale-it-has-been-a-while-activity-7427718512284012545-QH9V)
- [Xianbao QIAN: GLM-5 Deep Dive (MLA + DSA)](https://x.com/Xianbao_QIAN/status/2023199756009591250)
- [Qwen 3.5 Medium: Benchmarks & Guide (Digital Applied)](https://www.digitalapplied.com/blog/qwen-3-5-medium-model-series-benchmarks-pricing-guide)
- [Qwen3.5-35B-A3B Model Card (HuggingFace, official benchmarks)](https://huggingface.co/Qwen/Qwen3.5-35B-A3B)
- [HuggingFace transformers: Gemma 4 modeling source (MLP+MoE dual path)](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma4/modeling_gemma4.py)
