# Gemma 4 深度架构解析：小模型如何重新定义参数效率的边界

> 2026 年 4 月 2 日，Google DeepMind 发布 Gemma 4。31B 参数的 dense 模型在 AIME 2026 上拿到 89.2%，MoE 变体以 3.8B 活跃参数跑出 27B 级别的性能。与此同时，Qwen3-235B 的总参数量是它的 7.6 倍，GLM-5 更是达到 744B 总参数。

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

关键设计选择是 **128 个小专家**。值得注意的是，Qwen3-235B-A22B 同样采用了 128 个专家的设计（每 token 激活 8 个）——这说明 128 专家已经成为当前 MoE 架构的主流选择，而非 Gemma 4 的独创。

但两者在专家配置上的差异值得深究：

| 配置项 | Gemma 4 26B-A4B | Qwen3-235B-A22B |
|--------|-----------------|-----------------|
| 总参数 | 25.2B | 235B |
| 专家总数 | 128 | 128 |
| 每token激活 | 8 + 1 共享 | 8（无共享专家） |
| 活跃参数 | 3.8B | 22B |
| GQA (MoE) | 32Q / 4KV | 64Q / 4KV |
| 层数 | 未公开 | 94 层 |
| 上下文 | 256K | 128K（原生） |

Qwen3 的技术报告中明确提到，他们去掉了共享专家（Qwen2.5-MoE 有共享专家），采用 global-batch load balancing loss 来促进专家专业化。Gemma 4 则保留了共享专家——这是一个有趣的分歧，说明团队之间对"共享专家是否有益"还没有达成共识。

128 个专家的核心优势在于**路由的粒度**。相比早期的 8-16 大专家设计，128 个小专家的路由器可以做更细粒度的分发——不是"这段文本归你"，而是"这个词的这种语义组合归这个专家"。更细的路由粒度意味着每个专家学习的模式更专注，参数利用率更高。

但 128 个专家也带来了挑战：路由器训练更难（选择空间从 8-16 扩大到 128），负载均衡更难保证。Qwen3 的 global-batch load balancing loss 和 Gemma 4 的共享专家，是两种不同的应对策略。

一个值得注意的数据点：Gemma 4 26B-A4B 在 AIME 2026 上拿到 88.3%，Qwen3-235B-A22B 在 AIME'24 上拿到 85.7。Gemma 4 用 3.8B 活跃参数追平甚至超越了 22B 活跃参数的 Qwen3。这背后的原因，需要从训练策略中去寻找。

## 二、训练篇：蒸馏策略的差异才是真正的分水岭

架构优化解释了 Gemma 4 的效率，但没有完全解释它为什么这么强。一个架构精良但训练数据平庸的 31B 模型，不可能在 benchmark 上碾压训练充分的 235B 模型。

### 2.1 Teacher 模型的差距：Gemini 3 vs 自蒸馏

Google 明确表示 Gemma 4 的架构源自 Gemini 3 研究。虽然 Google 没有公开 Gemma 4 的完整训练细节，但从已发表的 Chain-of-Thought 蒸馏研究来看，Google 团队已经在系统性地研究从 Gemini 3 到 Gemma 的知识蒸馏方法。

这种蒸馏不是简单的"让大模型生成答案，小模型模仿"。流程更接近：
1. 用 Gemini 3 对大量 prompt 生成详细的推理链（Chain-of-Thought）
2. 过滤质量，保留推理逻辑清晰、步骤完整的样本
3. 用这些合成数据训练 Gemma 4 的 instruction-tuning 阶段

Qwen3 也采用了类似的蒸馏策略。根据 Qwen3 技术报告，他们对小模型使用了"strong-to-weak distillation"，利用旗舰模型（235B-A22B）作为 teacher，通过 off-policy 和 on-policy 两种方式传递知识。报告明确指出："Distillation from advanced teacher models significantly outperforms reinforcement learning in performance and training efficiency."

所以，**蒸馏本身不是 Gemma 4 的独门武器——三个团队都在做。真正的差异在于 Teacher 模型的能力上限。** Google 拥有 Gemini 3（很可能万亿参数级的闭源模型），作为 teacher 的能力上限远超 Qwen3-235B 或 GLM-5 自身。这就像用 985 大学教授的讲义来教高中生，和用高中尖子生的笔记来教普通学生——虽然都在"蒸馏"，但信息质量的天花板不同。

### 2.2 训练管线对比：三阶段的殊途同归

三个模型的预训练都采用了三阶段策略，但侧重点不同：

**Qwen3**（36T tokens）：
1. **General Stage (S1)**：30T tokens，4K 上下文，建立语言基础
2. **Reasoning Stage (S2)**：5T tokens，增加 STEM/代码/推理/合成数据比例
3. **Long Context Stage**：数千亿 tokens，4K→32K 上下文扩展，使用 ABF（base frequency 10K→1M）、YARN 和 Dual Chunk Attention (DCA)

**GLM-5**（28.5T tokens）：
1. **Base Model Training**：27T tokens，优先代码和推理数据
2. **Mid-training**：4K→200K 上下文渐进扩展，聚焦长上下文 agentic 数据
3. **Post-Training**：顺序 RL 管线——Reasoning RL → Agentic RL → General RL，全程使用 On-Policy Cross-Stage Distillation 防止遗忘

**Gemma 4**（未公开 token 数，但架构源自 Gemini 3 研究）：
- 从已知信息推断，采用了类似的分阶段策略
- 区别在于合成数据的质量和多样性（Gemini 3 作为 teacher）

GLM-5 的训练管线中最值得关注的是其 RL 基础设施。他们开发了 **slime**——一个异步 RL 框架，将生成和训练解耦，大幅提升了 GPU 利用率和 RL 训练吞吐。这让他们能进行更细粒度的 post-training 迭代，包括异步 Agent RL 算法，让模型从复杂的长周期交互中学习。这种在 RL 工程上的投入，直接反映在 GLM-5 在 agentic benchmark（SWE-bench、Vending Bench 2）上的领先表现。

### 2.3 QAT：训练时就考虑量化

Gemma 4 提供了 Quantization-Aware Training (QAT) checkpoint。这不是事后量化，而是在训练阶段就引入量化噪声，让模型学会在低精度表示下保持输出质量。

实际效果是：Gemma 4 的 QAT checkpoint 在量化到 NVFP4（4-bit 浮点）后，质量损失极小。NVIDIA 已经发布了基于此的 Gemma-4-31B-IT-NVFP4 模型。

Qwen3 和 GLM-5 目前主要依赖后量化（GPTQ、AWQ 等），虽然效果也不错，但训练时就考虑量化的 QAT 在理论上能做到更小的精度损失。这是 Gemma 4 在推理效率上的一个实际优势。

### 2.4 原生多模态训练：不是拼接，是融合

Gemma 4 的多模态能力不是后期拼接的结果，而是在预训练阶段就与文本一起训练的。视觉编码器基于 ViT，但做了关键改进：
- 使用 2D RoPE 而非 1D RoPE，直接编码 patch 的二维空间位置
- 支持可变宽高比输入，通过自适应 resize + padding 保持原始比例
- 可配置的 soft token budget（70/140/280/560/1120 tokens），让开发者根据任务需求在精度和速度之间做 trade-off

音频编码器沿用了 Gemma-3n 的 USM-style conformer 架构，但从 681M 参数压缩到 305M，帧时长从 160ms 降到 40ms。

GLM-5 在多模态上走了不同的路线——原生不处理图像/音频/视频，而是通过 tool-calling 调用 GLM 家族的其他专用模型（GLM-Image、GLM-4.6V、GLM-Vision）。这种"模型即工具"的设计在 agentic 场景下更灵活，但端到端的延迟和一致性不如 Gemma 4 的原生融合方案。

Qwen3 同样采用原生多模态路线，但视觉能力主要由 Qwen3-VL 系列承担，而非基础语言模型本身。

## 三、对比篇：与 Qwen3、GLM-5 的正面交锋

### 3.1 参数效率对比

要公平比较，我们需要关注的是**活跃参数量**而非总参数量。以下是三个团队旗舰模型的核心数据：

| 模型 | 总参数 | 活跃参数 | AIME | LiveCodeBench | 注意力架构 |
|------|--------|----------|------|---------------|-----------|
| Gemma 4 31B | 30.7B | 30.7B | 89.2% (AIME'26) | 80.0% (v6) | Sliding+Global 交错 |
| Gemma 4 26B-A4B | 25.2B | 3.8B | 88.3% (AIME'26) | 77.1% (v6) | Sliding+Global 交错 |
| Qwen3-235B-A22B | 235B | 22B | 85.7% (AIME'24) | 70.7% (v5) | 标准 Full Attention |
| GLM-5 | 744B | 40B | SOTA on agentic* | SOTA on SWE* | DSA (DeepSeek Sparse Attention) |

*GLM-5 的核心优势在 agentic benchmark 而非纯推理 benchmark。它在 ArtificialAnalysis Intelligence Index v4.0 上拿到 50 分，是首个达到此分数的开源模型。

一个关键发现：**Gemma 4 26B-A4B 用 3.8B 活跃参数，在 AIME 上超越了用 22B 活跃参数的 Qwen3-235B。** 活跃参数效率差距接近 6 倍。即使拿 31B dense 和 235B-A22B 比，Gemma 4 的活跃参数效率也领先约 30%。

但这只是数学推理和编程 benchmark 上的效率。在 agentic 能力上，GLM-5 通过 DSA + slime RL + Agent RL 的组合，在 SWE-bench、Vending Bench 2 等长周期任务上展现出明显优势。GLM-5 在 Vending Bench 2（模拟自动售货机业务运营一年）上以 $4,432 的最终余额排名第一（开源模型），接近 Claude Opus 4.5 的水平。

### 3.2 注意力架构对比：三条不同的路线

这是三个模型在架构层面最本质的差异：

**Gemma 4：Sliding Window + Global 交错注意力**
- Local 层用 sliding window（小模型 512 token，大模型 1024 token），Global 层做全序列注意力
- 5:1 或 4:1 交错模式（E2B 为 4:1，其他为 5:1），最后一层强制 Global
- 通过 Shared KV Cache、K=V、GQA 8:1、p-RoPE 将 Global 层开销压缩到极致
- 256K 上下文（31B/26B）或 128K（E2B/E4B）

**Qwen3：标准 Full Attention + YARN/DCA 扩展**
- 基础架构沿袭 Qwen2.5，使用标准 Full Attention
- Qwen3-235B-A22B：94 层，64Q / 4KV 的 GQA，128 专家 MoE
- 长上下文通过 RoPE base frequency 调整（ABF, 10K→1M）+ YARN + Dual Chunk Attention (DCA) 实现
- 原生 128K 上下文，可扩展到 262K

**GLM-5：DeepSeek Sparse Attention (DSA)**
- 从 GLM-4.7 的标准 GQA（96Q/8KV）转向 DSA——一种基于 token 重要性动态分配注意力资源的稀疏注意力机制
- DSA 显著降低了训练和推理成本，同时保持长上下文理解能力
- 744B 总参数，40B 活跃参数，原生 200K 上下文
- 适配七种国产 GPU 平台（华为昇腾等），从底层 kernel 到上层推理框架做了深度优化

三条路线的选择反映了不同的工程优先级：
- Gemma 4 追求**极致推理效率**，不惜牺牲部分注意力精度
- Qwen3 追求**架构简洁性和通用性**，通过成熟组件的组合达到稳定效果
- GLM-5 追求**长上下文 agentic 能力**，DSA 的动态稀疏性天然适合多步工具调用场景

### 3.3 位置编码对比

| 模型 | 位置编码策略 | 特殊处理 |
|------|------------|---------|
| Gemma 4 | Dual RoPE：Sliding 层标准 RoPE，Global 层 p-RoPE (p=0.25) | 低频维度不施加旋转，保留语义信息 |
| Qwen3 | 标准 RoPE + ABF（base freq 10K→1M） | 使用 YARN 和 DCA 处理超长上下文 |
| GLM-5 | 标准 RoPE（具体参数待确认） | DSA 内部可能有自己的位置编码处理 |

Gemma 4 的 p-RoPE 是三者中最独特的。它承认了一个事实：在长上下文中，不是所有维度都需要位置信息。低频维度（旋转极小的那些）在标准 RoPE 下会成为位置噪声的载体，反而干扰语义匹配。p-RoPE 干脆不旋转它们，让这些维度纯粹编码语义内容。

### 3.4 量化策略对比

| 模型 | 量化方案 | 特点 |
|------|---------|------|
| Gemma 4 | QAT checkpoint（训练时量化感知） | NVIDIA 发布 NVFP4 版本，精度损失极小 |
| Qwen3 | 后量化（GPTQ/AWQ） | 社区生态丰富，工具链成熟 |
| GLM-5 | FP8 官方版 + 后量化 | 专门提供 FP8 checkpoint，适配 Hopper/Blackwell GPU |

GLM-5 直接提供官方 FP8 权重（GLM-5-FP8），并且在 vLLM 和 SGLang 的部署配置中专门针对 FP8 做了优化。这在实际部署中的价值很大——不需要用户自己做量化，官方保证质量。

### 3.5 生态与许可证

Gemma 4 采用 Apache 2.0 许可证，Qwen3 同样使用 Apache 2.0，GLM-5 也开源了权重。三者都是真正意义上的开源。

但趋势值得关注：
- **Qwen3.5-Omni 和 Qwen 3.6 Plus** 已经开始从完全开源转向更受限的发布策略，不再公开全部权重
- **GLM-5** 的开源权重是完整的，但后续版本的开放程度存在不确定性
- **Gemma 4** 在中国 AI 实验室收紧开源的同时选择 Apache 2.0 全面开放

对于需要商业部署确定性的企业来说，许可证的清晰度和稳定性有时比 benchmark 数字更重要。

## 四、三个模型的差异化定位

经过架构和训练两个层面的拆解，我们可以清晰地看到三个模型的差异化定位：

**Gemma 4：效率至上。** 每一个架构选择都在压缩推理开销——Shared KV Cache、K=V、p-RoPE、PLE、QAT。它不是最强的模型，但在给定算力预算下可能是性价比最高的。适合对推理成本敏感、需要广泛部署的场景。

**Qwen3：通用均衡。** 架构选择相对保守（标准 Full Attention + 成熟组件组合），通过规模（36T tokens、235B 参数）和扎实的数据工程达到 SOTA。开源生态最完善，社区支持最好。适合需要稳定可靠、生态丰富的通用场景。

**GLM-5：Agentic 专精。** DSA 稀疏注意力 + slime 异步 RL + Agent RL 算法，整个训练管线都在为 agentic 能力服务。在 SWE-bench、Vending Bench 2 等长周期任务上表现突出，同时在七种国产 GPU 上做了深度适配。适合面向中国市场、以 agentic coding 为核心的场景。

## 五、对行业的深层影响

### 5.1 "小模型 + 大 Teacher"正在成为新范式

Gemma 4 不是孤例。Microsoft 的 Phi 系列和 Apple 的 OpenELM 也在走类似路线——用大模型生成高质量训练数据，用精巧的架构训练小模型。Qwen3 也明确承认蒸馏优于 RL。

这个范式如果成立，意味着：
- 模型能力的上限由 Teacher 模型（通常是闭源的）决定
- 开源模型之间的竞争将从"谁的参数多"转向"谁的蒸馏策略好"
- 拥有最强闭源模型的团队（Google、OpenAI、Anthropic）在开源领域也拥有结构性优势

### 5.2 128 专家正在成为 MoE 的新标准

Qwen3-235B 和 Gemma 4 26B-A4B 都选择了 128 个专家，说明行业已经从"少量大专家"（DeepSeek-V3 初代的风格）向"大量小专家"收敛。但两者在共享专家的有无上存在分歧——这个问题的答案可能需要在更多实验中验证。

### 5.3 注意力架构正在分化

三个模型的三种注意力方案（Sliding+Global 交错、标准 Full + YARN/DCA、DSA 稀疏注意力）代表了三种不同的哲学。这种分化意味着推理引擎需要同时支持多种注意力模式，vLLM 和 SGLang 的适配工作量会持续增加。

### 5.4 RL 基础设施成为新的竞争维度

GLM-5 的 slime 框架表明，RL 训练效率本身已经成为模型竞争力的组成部分。能更高效地进行 RL 迭代的团队，能在 post-training 阶段拉开差距。这对 RL 基础设施（如 veRL、OpenRLHF、MindSpeed-RL）的开发者来说是利好消息。

## 六、结论

Gemma 4 告诉我们的事情，远比一个 benchmark 排名重要：

**参数效率的边界比我们想象的要远得多。** 31B 参数可以达到过去需要 200B+ 参数才能达到的性能水平，这不是因为 31B 模型"超常发挥"，而是因为过去的大模型在架构和训练上存在大量可以优化的空间。

**蒸馏策略的质量差异，比架构差异更能解释性能差距。** 三个团队都在做蒸馏，但 Teacher 模型的能力上限不同。这打破了"开源模型之间公平竞争"的假设——拥有更强闭源模型的团队，在开源领域也有结构性优势。

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
