# Voice Agent / Speech-to-Speech 对话系统学术前沿调研（2023–2025）

> **方法与可信度声明**
> 本报告基于 arXiv、ACL Anthology、ISCA Archive、NeurIPS/ICLR/ICML Proceedings、OpenAI 官方博客等检索结果撰写。
> **本会话的网页抓取工具（web_fetch）被沙箱网络策略整体阻断**（对 `arxiv.org`、`huggingface.co`、`aclanthology.org`、`example.com` 等一律返回 `resolves to a non-public IP address`），因此**无法逐篇打开原文核对**。所有论文标题与链接均来自 `web_search` 返回的真实索引条目；凡检索结果中未直接出现、我无法证实的 arXiv 编号或数值，均显式标注「**未核实**」。当前系统时间为 2026-09，检索结果中包含部分 2026 年工作，正文中一律显式标注年份，请勿把 2026 年结果误当作 2025 年及以前的结论。

---

## 一、研究版图总览

### 1.1 代表系统对比表

| 系统（年份） | 语音表示 | 理解/生成架构 | 双工与轮次 | 语音输出 | 备注 |
|---|---|---|---|---|---|
| **dGSLM**（2022/2023）[1] | HuBERT 离散单元，**无文本** | 双塔 Transformer，双通道联合建模 | 天然双流，可生成笑声/附和 | 是 | 证明"无文本纯语音对话"可行，但语义可控性差 |
| **AudioPaLM**（2023）[2] | 文本 token + AudioLM 式声学 token 统一词表 | 单一自回归 LLM（PaLM-2） | 半双工/离线 | 是 | ASR/AST/S2ST 统一，但非流式低延迟 |
| **SpeechGPT**（EMNLP 2023 Findings）[3] | HuBERT 离散单元 | 文本-语音交错自回归，"模态思维链" | 半双工 | 是 | 保留 LLM 语义能力，延迟高 |
| **SpiRit-LM**（TACL 2025）[4] | 语音单元 + 文本交错 | 交错预训练 | 半双工 | 是 | 语音-文本对齐的早期探索 |
| **Moshi**（2024-09）[5] | **Mimi codec**，12.5 Hz 帧率、8 码本、约 1.1 kbps | 双流并行 + RQ-Transformer depth transformer + **inner monologue**（先文本后声学） | **原生全双工**，用户流/系统流同时建模 | 是 | 开源权重；论文声称理论延迟约 160 ms、单卡实测约 200 ms（**具体数值以原文为准**） |
| **GLM-4-Voice**（2024-12）[6] | 单码本、12.5 Hz 语音 tokenizer + 流式 decoder | 端到端 S2S，流式 | 半双工，支持打断 | 是 | 支持情感/方言/语速控制；用截断音频样本做流式推理 |
| **Qwen2.5-Omni**（2025-03）[7] | AuT 音频编码器 + **Talker 以 6.25 Hz 输出离散语音 token** + 块级 DiT code2wav | **Thinker–Talker 分离**；TMRoPE 时间对齐位置编码 | 流式、块级；Talker 与 Thinker 并行 | 是 | 理解与生成解耦，可分别优化；7B 量级 |
| **Qwen3-Omni**（2025-09）[8] | 同上路线演进，MoE | Thinker–Talker | 流式低延迟 | 是 | 30B-A3B 级 MoE，主打全模态与低延迟 |
| **Step-Audio**（2025-02）[9] | **双码本** tokenizer（语言码本 + 语义码本）[10] | 统一理解+生成，130B 级 | 半双工，支持工具/RAG | 是 | 引入生成式 reward 与 ASR/TTS 引擎协同；支持情感、方言、语速控制 |
| **Step-Audio 2**（2025-07）[11] | 双码本延续 | 统一音频理解与语音生成 + 工具调用 | 半双工 | 是 | 引入 RL 优化 |
| **MiniCPM-o 2.6**（2025-01）[12] | 语音 tokenizer + 流式 | 端到端全模态（视觉+语音+文本） | **声称手机端实时全双工** | 是 | arXiv 编号**未核实**（检索仅命中 HF/GitHub 模型卡） |
| **LLaMA-Omni**（ICLR 2025）[13] | Whisper 类语音编码器 + 适配器 | LLM + **流式非自回归语音解码器** | 半双工、低延迟 | 是 | 适配器+流式 decoder 的轻量范式 |
| **Minmo / LLaMA-Omni 2.5**（2025-06）[14] | 同上延续 | Thinker–Talker 类 | 半双工 | 是 | 被后续论文作为 Talker 类基线引用 |
| **Mini-Omni**（2024-08）[15] | 语音单元 + 文本 | "边想边说"流式 | 半双工 | 是 | 开源、轻量，适合做研究基线 |
| **Freeze-Omni**（2024-11）[16] | 分块流式语音输入 | **冻结 LLM** + 轻量适配 | 支持打断的类双工 | 是 | 三阶段训练；ICML 2025 poster |
| **Kimi-Audio**（2025-04）[17] | 12.5 Hz 音频 tokenizer | LLM + 文本/音频并行头 + 流式 detokenizer | 半双工 | 是 | ASR/音频理解/语音聊天统一 |
| **SLAM-Omni**（ACL 2025 Findings）[18] | 语音 token | 单阶段训练、音色可控 | 半双工 | 是 | 强调训练简洁性与 zero-shot 音色控制 |
| **Ultravox**（2024–2025）[19] | Whisper 编码器投影进 LLM 嵌入空间 | 语音适配器，**文本输出** | 不适用 | **否** | 重要的理解侧开源基线，但**不是 S2S** |
| **GPT-4o Realtime / gpt-realtime**（2024–2025）[20][21] | 未公开 | 未公开 | 官方主打实时双工与打断 | 是 | 闭源；GPT-4o System Card 有安全评测[22] |
| **FlexDuo**（2025-02）[23] | 复用已有模型 | **可插拔**地把回合制模型改造成双工 | 通过额外状态预测实现双工 | 复用 | "不改主干即得双工"的工程化路线 |

### 1.2 技术路线分类

- **A. 级联流水线（VAD → ASR → LLM → TTS）**：工业界仍是主流，优势是各模块可独立替换、可观测、可接入文本工具生态；代价是误差累积、延迟叠加、副语言信息在 ASR 文本瓶颈处被丢弃。
- **B. 语音适配器 / 语音理解 LLM（speech-in, text-out）**：以 Ultravox[19]、Qwen2-Audio[24]、SALMONN[25] 为代表，把语音编码器投影进文本 LLM。它不是 S2S，但提供了"语音语义能力上限"的参照系。
- **C. 文本-语音交错自回归**：AudioPaLM[2]、SpeechGPT[3]、SpiRit-LM[4]。核心动机是保留 LLM 的语言能力，代价是自回归链条变长、延迟高。
- **D. 端到端双流全双工**：dGSLM[1]、Moshi[5]、MiniCPM-o[12]、PersonaPlex（2026）[26]、BayLing-Duplex（2026）[27]。核心是"两个并行 token 流 + 一个决定谁在说的隐状态"。
- **E. Thinker–Talker 解耦**：Qwen2.5/3-Omni[7][8]、Step-Audio[9][11]、Kimi-Audio[17]、Minmo[14]。理解侧与生成侧分开，允许生成侧用更小的模型、更激进的流式策略。
- **F. 冻结 LLM + 轻量适配**：Freeze-Omni[16]、SLAM-Omni[18]、LLaMA-Omni[13]。用最小训练成本把已有文本 LLM 变成语音对话体。
- **G. 闭源实时 API**：GPT-4o Realtime / gpt-realtime[20][21]。技术细节不公开，是能力上限与用户体验的参照，但不可复现。

**版图判断**：2023 年的主线是"如何把语音塞进 LLM 的词表"；2024 年的主线是"如何让语音 token 足够短、足够语义化以支撑实时"；2025 年的主线已经变成"**如何建模'该谁说'**"，即从 turn-based 走向 full-duplex。Moshi[5] 与 Full-Duplex-Bench[28] 是这一转折的两个标志性节点。

---

## 二、逐热点深入

### 2.1 端到端 S2S 大模型：语义 vs 声学 token、codec、双工建模

**问题定义**：如何在单一模型里同时完成"听懂内容"和"发出有韵律、有声线、能被打断的语音"，而不经过文本瓶颈。

**代表工作与关键结论**：

1. **Moshi[5]** 给出目前最完整的公开设计：Mimi codec 把 24 kHz 语音压到 12.5 Hz、8 个码本、约 1.1 kbps；模型同时维护**用户流**与**系统流**两条并行的 token 流，因此"该谁说"不是外挂的 VAD 判断，而是模型自己生成的一部分。其 RQ-Transformer（外层时间 Transformer + 内层 depth Transformer）处理多码本层级依赖。**Inner monologue** 是关键工程技巧：系统流的文本 token 先于对应音频 token 生成，相当于给声学生成加了一个语义"草稿"，显著缓解纯声学建模的语言退化。
2. **Qwen2.5-Omni[7]** 走 Thinker–Talker 路线：Thinker 负责多模态理解，Talker 以 6.25 Hz 输出离散语音 token，再由块级 DiT（code2wav）还原波形；TMRoPE 解决音视频/文本的时间对齐。**结论：把"理解"和"发声"解耦，比强行用一条自回归链同时做两件事更容易做到低延迟与高质量并存。**
3. **Step-Audio[9][10]** 用**双码本** tokenizer（语言码本承载语义、语义/声学码本承载音色韵律），配合生成式 reward 模型与外部 ASR/TTS 引擎做数据与评测闭环；支持情感、方言、语速显式控制。
4. **GLM-4-Voice[6]** 选择更激进的"单码本 + 12.5 Hz"，牺牲部分重建质量换取极短的序列长度与流式友好性，并用截断音频样本做流式推理。
5. **LLaMA-Omni[13] / Mini-Omni[15] / Freeze-Omni[16]** 代表"低成本改造"路线：冻结或轻量微调文本 LLM，用流式非自回归语音 decoder 或分块流式输入换低延迟。

**未解难题**：
- **语义-声学耦合的两难**：单码本短序列（GLM-4-Voice[6]）流式友好但音质/韵律上限低；多码本（Moshi[5] Mimi）质量好但层级建模复杂、推理成本高。目前**没有公认的最优折中**。
- **语音 tokenizer 与 LLM 预训练分布不匹配**：文本 LLM 是在离散词表上预训练的，声学 token 的分布完全不同，导致"语音输入时语言能力下降"的 modality gap（见 2.8）。
- **统一理解与生成的相互干扰**：Step-Audio[9] 与 Kimi-Audio[17] 都采用并行头方案，但理解与生成共享主干是否一定互相伤害，尚无定论。

**链接**：[Moshi](https://arxiv.org/abs/2410.00037)｜[Qwen2.5-Omni](https://arxiv.org/abs/2503.20215)｜[Qwen3-Omni](https://arxiv.org/abs/2509.17765)｜[GLM-4-Voice](https://arxiv.org/abs/2412.02612)｜[Step-Audio](https://arxiv.org/abs/2502.11946)｜[Step-Audio 2](https://arxiv.org/abs/2507.16632)｜[LLaMA-Omni](https://arxiv.org/abs/2409.06666)｜[LLaMA-Omni 2.5 / Minmo](https://arxiv.org/abs/2506.04518)｜[MiniCPM-o 2.6](https://huggingface.co/openbmb/MiniCPM-o-2_6)｜[Ultravox](https://github.com/fixie-ai/ultravox)｜[Kimi-Audio](https://arxiv.org/abs/2504.18425)｜[Mini-Omni](https://arxiv.org/abs/2408.16725)｜[Freeze-Omni](https://arxiv.org/abs/2411.00774)

### 2.2 全双工与轮次转换（full-duplex / turn-taking / barge-in）

**问题定义**：对话系统除了"说什么"，还必须决策"**此刻该不该说**"——包括用户停顿时不抢话、用户打断时立刻停、在合适位置插入"嗯/对"这类附和（backchannel）。这是一个**时序决策问题**，与内容生成解耦。

**代表工作与数据集**：
- **dGSLM[1]** 是最早证明"双通道语音对话可以无文本端到端建模"的工作，能自发产生笑声与附和类发声，但无法保证语义一致性。
- **Full-Duplex-Bench**[28] 是该方向第一个专门的评测基准，用**分块流式输入 + 实时输出**的方式，从四个维度衡量：**pause handling（停顿处理）、backchannel（附和）、smooth turn-taking（平滑轮转）、user interruption（用户打断/barge-in）**。其核心发现对工程界非常重要：**现有模型普遍"过于急于回应"——在用户只是思考性停顿时就抢话**。
- **Full-Duplex-Bench v1.5**[29] 把重点转向**overlap handling（重叠语音处理）**；**v2**[30] 引入自动化的"考官"做多轮评测，把评测从单轮反应扩展到多轮交互稳定性。
- **FLEXI**[31] 进一步区分**用户侧附和**与**模型侧附和**，发现模型主动产生 backchannel 的比例远低于人类。
- **FlexDuo**[23] 提供"可插拔"方案：在已有回合制 S2S 模型上增加一个轻量的"听/说"状态决策模块，即可获得双工能力，无需重训主干。
- **综述**：《From Turn-Taking to Synchronous Dialogue: A Survey of Full-Duplex Spoken Language Models》[32] 系统梳理了从轮次到同步的范式迁移；另有 2026 年的全双工口语对话系统综述[33]提出"决策状态机"式的统一视角。

**未解难题**：
- **评估本身未被解决**：人工评价双工行为成本极高，自动指标（如抢话率、响应延迟）与人类主观"自然度"的相关性**尚未验证**。
- **训练数据稀缺**：真实的双通道、带重叠语音、带附和的对话数据（如 Fisher、Switchboard 类）与当前的语音 LLM 训练配方并不天然匹配；多数模型是靠合成/拼接数据学"不抢话"。
- **"沉默"的表达力**：模型可以学会不抢话，但很难学会用停顿、叹气、吸气来参与对话。

**链接**：[dGSLM](https://arxiv.org/abs/2203.16502)｜[Full-Duplex-Bench](https://arxiv.org/abs/2503.04721)｜[v1.5](https://arxiv.org/abs/2507.23159)｜[v2](https://arxiv.org/abs/2510.07838)｜[FLEXI](https://arxiv.org/abs/2509.22243)｜[FlexDuo](https://arxiv.org/abs/2502.13472)｜[Full-Duplex Survey](https://arxiv.org/abs/2509.14515)｜[Full-Duplex-Bench GitHub](https://github.com/DanielLin94144/Full-Duplex-Bench)

### 2.3 语音 tokenizer / neural audio codec 的取舍

**问题定义**：语音 tokenizer 决定了整个 S2S 系统的"词表"，它同时约束了序列长度（→延迟）、重建质量（→音质）、以及语义保真度（→语言能力）。

**代表工作与关键结论**：

| 方案 | 时间 | 帧率/码率 | 语义性 | 主要取舍 |
|---|---|---|---|---|
| **SoundStream**[34] | 2021 | 3–16 kbps，RVQ + 量化器 dropout | 低（纯声学） | 首个端到端可流式神经音频 codec；码率可伸缩 |
| **EnCodec**[35] | 2022 | 1.5–24 kbps，RVQ + 熵编码 | 低 | 音质强、生态成熟；帧率偏高（约 75 Hz），序列太长不利于 LLM |
| **SpeechTokenizer**[36] | 2023 | RVQ 第 1 层由 HuBERT 蒸馏为语义层，其余为声学层 | **中高** | 显式做"语义/声学分层"，但帧率仍偏高 |
| **Mimi**（Moshi 配套）[5] | 2024 | **12.5 Hz**，8 码本，约 1.1 kbps，24 kHz | 中（首码本偏语义） | **为流式双工专门设计**：因果卷积、低帧率、超低延迟 |
| **Step-Audio Tokenizer**[10] | 2025 | 双码本（语言 + 语义） | **高** | 显式拆成两个语义角色，但需要两套解码路径 |

**关键结论**：
1. **帧率是延迟的第一性约束**。语音 LLM 的自回归步数与帧率成正比；从 EnCodec 的约 75 Hz 降到 Mimi 的 12.5 Hz，是"实时对话"从不可能变为可能的直接原因。这就是为什么 2024 年之后的 S2S 系统几乎都收敛到 **12.5 Hz 或 6.25 Hz** 量级（Moshi[5]、GLM-4-Voice[6]、Kimi-Audio[17]、Qwen2.5-Omni[7] 的 Talker）。
2. **"语义 token 好还是声学 token 好"这个二选一已经被绕过**。主流做法不是二选一，而是**分层/多码本**：第一层承担语义、其余层承担声学细节（SpeechTokenizer[36]）、或干脆用多码本 + depth transformer（Moshi[5]）。
3. **纯声学 token 会让 LLM 的语言能力退化**，这是 SpeechGPT[3]、AudioPaLM[2] 到 Moshi[5] 一路上反复出现的现象，Moshi 的 inner monologue 正是对这个问题的工程回应。

**未解难题**：tokenizer 与下游 LLM 的**联合训练**仍然罕见（多数是"先训 codec 再冻住"）；缺少一个能同时给出"语义保真度、重建质量、流式延迟、码率"四维可比数字的标准测试集。

**链接**：[SoundStream](https://arxiv.org/abs/2107.03312)｜[EnCodec](https://arxiv.org/abs/2210.13438)｜[SpeechTokenizer](https://arxiv.org/abs/2308.16692)｜[Step-Audio Tokenizer](https://huggingface.co/stepfun-ai/Step-Audio-Tokenizer)｜[Dual-Codebook 说明](https://deepwiki.com/stepfun-ai/Step-Audio/2.1-dual-codebook-tokenization)

### 2.4 延迟与流式：可感知延迟指标、首包延迟、投机解码

**问题定义**：人类对话的轮次间隔通常在**数百毫秒**量级，超过这个量级就会被感知为"卡"。因此延迟不是一个优化项，而是**可用性门槛**。

**关键指标（工程与学术已趋同）**：
- **TTFT / TTFA（Time to First Token / Time to First Audio）**：首包延迟，语音场景中 TTFA 比文本 TTFT 更贴近用户感知[37]。
- **端到端轮次延迟（turn-by-turn latency）**：从用户说完到系统首个音频帧播出。
- **打断响应延迟（barge-in latency）**：用户开始说话到系统停声的时间，Full-Duplex-Bench[28] 把它作为核心维度。

**技术手段**：
1. **降帧率**：见 2.3，是延迟的最大杠杆。
2. **块级/分块流式**：Qwen2.5-Omni[7] 的 Talker 与 code2wav 都按块推进；GLM-4-Voice[6] 用截断音频样本实现流式；Freeze-Omni[16] 用分块流式输入。
3. **非自回归/并行语音解码**：LLaMA-Omni[13] 的流式非自回归语音解码器一次性生成一段。
4. **投机解码（speculative decoding）**：Leviathan 等人的原始工作[38]提出用草稿模型并行验证；在语音场景中，"文本先于声学生成"本身就构成一种天然的草稿（Moshi 的 inner monologue[5]），近期也出现了把 transducer 与语音 LLM 结合的工作[39]以及双路径投机生成[40]。
5. **流式 ASR 侧的经典方法**：FastEmit[41] 用序列级发射正则化压低 streaming ASR 的延迟，是理解"延迟 vs 精度"权衡的经典参考。

**未解难题**：
- **延迟指标不统一**：不同论文报告的"延迟"口径各异（是否含 VAD、是否含网络、是否含首帧播放），导致**跨论文数字几乎不可比**。这是本领域最严重的方法学问题之一。
- **投机解码在语音 token 上是否划算存疑**：语音 token 的"多码本 + 高熵"特性使草稿模型的接受率不稳定，公开的端到端收益证据不足（**该判断基于原理解读，具体收益数值未核实**）。

**链接**：[FastEmit](https://arxiv.org/abs/2010.11148)｜[Speculative Decoding](https://arxiv.org/abs/2211.17192)｜[TRADE](https://arxiv.org/abs/2606.08486)｜[RelayS2S](https://arxiv.org/abs/2603.23346)｜[TTFT 工程解读](https://www.assemblyai.com/blog/time-to-first-token-voice-agents)

### 2.5 评测基准与指标

**问题定义**：语音对话的"好"包含至少四个正交维度——**内容正确、时序正确、副语言正确、鲁棒**。WER 只能覆盖第一维的一小部分，无法覆盖"是否在错误的时机说话"。

**关键结论**：
1. **VoiceBench**[42]（TACL）用 8 个数据集评测 LLM-based 语音助手，主体是短指令；其核心发现是**语音输入版本的模型普遍落后于同一模型的文本版本**，且 ASR 类错误会向下游传播。
2. **Full-Duplex-Bench**[28][29][30] 是**唯一以时序行为为主目标**的基准族，覆盖停顿处理、附和、平滑轮转、打断与重叠。
3. **SpokenWOZ**[43]（NeurIPS 2023 D&B）提供 249 小时语音、约 20 万轮次的任务型口语对话，是检验"语音端到端能否追上文本 pipeline"的关键数据集；其结论是**口语对话状态追踪仍显著难于文本版**。
4. **AudioBench**[44]、**SD-Eval**[45]、**WavBench**[46] 分别覆盖通用音频任务、情感/韵律/年龄/口音维度、以及"推理 + 口语化 + 副语言"三维；WavBench 明确把"口语化表达"单列，这是文本基准无法覆盖的。
5. **综述层面**：EMNLP 2025 的 LALM 评测综述[47]系统整理了指标碎片化问题；SpeechIQ[48] 尝试用认知层次（记忆/理解/推理）重组语音理解评测。

**未解难题**：**没有公认的"韵律自然度/交互自然度"自动指标**。MOS 类指标依赖昂贵人工听测，而"打断响应时间"这类客观指标与主观体验的相关性尚未被系统验证。

### 2.6 语音中的工具调用 / function calling

**问题定义**：文本 function calling 的输入是干净字符串；语音输入则会带来（a）**口语化参数**（"帮我订后天下午三点左右"）、（b）**ASR 造成的专有名词/数字错误**、（c）**需要澄清的模糊指代**、（d）**必须边说边调用的流式约束**。

**代表工作**：
- **VoiceAgentBench**[49]（2025-10）专门评测语音助手在 agentic 任务上的表现，明确把"语音管道带来的额外失败模式"作为评测对象。
- **Spoken Function Calling**[50]（2026）把该任务拆成**函数名识别 / 参数抽取 / 调用决策**三个子任务并设计细粒度 reward，指出"语音输入下的函数名与参数抽取"是一个独立的、此前被忽视的 SLU 问题。
- **Audio2Tool**[51]（2026）发布语音工具使用数据集；**BFCL Audio**[52]（ICML 2026）把 Berkeley Function Calling Leaderboard 扩展到音频输入。

**关键结论**：目前公开证据一致指向"**语音 function calling 的瓶颈不在推理能力，而在参数级的信息损失与口语规范化**"。数字、日期、ID、代码标识符在口语中被读错或读成一串数字，是主要失败来源。

**未解难题**：**流式/边说边调用**（用户还没说完就要决定是否发起调用）几乎没有成熟方案；工具调用失败后的**语音澄清策略**（怎么用口语问"你是指 A 还是 B"）也缺少系统研究。

**链接**：[VoiceAgentBench](https://arxiv.org/abs/2510.07978)｜[Spoken Function Calling](https://arxiv.org/abs/2608.05126)｜[Audio2Tool](https://arxiv.org/abs/2604.22821)｜[BFCL Audio](https://icml.cc/virtual/2026/poster/61489)

### 2.7 副语言信息：情感、韵律、笑声、语气

**问题定义**：同一句话用不同语气说，含义可能相反。副语言信息（情感、态度、强调、笑、叹气、语速）在**文本瓶颈处会被完全丢弃**——这是纯级联方案的结构性缺陷，也是端到端 S2S 最重要的动机之一。

**代表工作与现状**：
- **dGSLM**[1] 证明双流语音模型可以自发产生**笑声与附和**类发声，是副语言生成的早期证据。
- **emotion2vec**[53]（ACL 2024 Findings）提供通用语音情感自监督表示，成为情感相关下游任务的标准前端。
- **NVSpeech**[54]（2025）构建了包含**副语言发声（paralinguistic vocalizations）**的大规模建模流水线，把"叹气、笑、吸气"等非词汇发声当作一等公民。
- **Style Amnesia**[55]（2025-12）报告了多轮语音对话中的**说话风格退化**：模型在首轮能保持的风格，随轮次增加而丢失——这是端到端 S2S 特有的新问题。
- **WavBench**[46] 把副语言拆成可测维度；**SD-Eval**[45] 覆盖情感、韵律、年龄、口音四个维度。
- 对话互动层面，"嗯/对/yeah"这类**反馈标记（backchannel）**的产生与预测已有专门研究[56]；情感层面的**对话崩溃（dialogue breakdown）**预测也出现了专门的语音数据集[57]。

**关键结论**：**理解侧**的副语言建模相对成熟（emotion2vec[53] 类表示 + 专门基准）；**生成侧**则明显落后——模型可以"听出"情绪，但很难稳定地"说出"情绪，且风格的**跨轮一致性**是新暴露的短板（Style Amnesia[55]）。

**未解难题**：副语言的**评价**没有可靠自动指标；副语言控制与内容正确性之间是否存在 trade-off，缺乏受控实验。

### 2.8 鲁棒性：口音、方言、噪声、专有名词与代码标识符

**问题定义**：口语对话的现实分布远比朗读语料恶劣。四类主要失效来源：口音/方言、噪声与远场、命名实体与代码标识符、以及语码转换（code-switching）。

**代表工作**：
- **CS3-Bench**[58]（2025-10）专门评测并改进**中英语码转换**的 S2S LLM，是少见的把 S2S（而非只是 ASR）作为评测对象的鲁棒性工作。
- **AfriSpeech-MultiBench**[59]（IJCNLP 2025）用多国、多领域的非裔口音英语构建垂直化 ASR 基准，说明口音鲁棒性仍是开放问题。
- **Speak & Spell**[60] 用 LLM 驱动**可控音素错误增强**来提升对话状态追踪的鲁棒性——一种"用合成错误换鲁棒性"的实用思路。
- **Modality Gap 研究**：EMNLP 2025 的实证研究[61]与更细粒度的内部状态分析[62]都指出，语音 LLM 在语音输入下的性能下降，部分来自**语音与文本表征对齐不充分**，而非单纯的识别错误。这意味着"提高 ASR 准确率"不足以解决鲁棒性问题。

**关键结论**：**鲁棒性问题的根因在层级上高于 ASR**。即使把识别错误降到很低，语言学能力损失（modality gap[61][62]）依然存在。这是端到端 S2S 目前最被低估的难点。

---

## 三、评测基准汇总表

| 基准（年份） | 侧重 | 覆盖维度 | 链接 |
|---|---|---|---|
| **VoiceBench**（2024，TACL） | LLM 语音助手综合能力 | 8 个数据集，短指令为主 | [arXiv:2410.17196](https://arxiv.org/abs/2410.17196) |
| **Full-Duplex-Bench**（2025） | 全双工轮次行为 | 停顿处理、附和、平滑轮转、打断 | [arXiv:2503.04721](https://arxiv.org/abs/2503.04721) |
| **Full-Duplex-Bench v1.5**（2025） | 重叠语音处理 | overlap handling | [arXiv:2507.23159](https://arxiv.org/abs/2507.23159) |
| **Full-Duplex-Bench v2**（2025） | 多轮双工 | 自动化考官、多轮稳定性 | [arXiv:2510.07838](https://arxiv.org/abs/2510.07838) |
| **FLEXI**（2025） | 人-LLM 双工交互 | 用户/模型双向 backchannel | [arXiv:2509.22243](https://arxiv.org/abs/2509.22243) |
| **SpokenWOZ**（NeurIPS 2023） | 任务型口语对话 | 249h 语音、约 20 万轮次、DST/端到端 | [arXiv:2305.13040](https://arxiv.org/abs/2305.13040) |
| **AudioBench**（2024） | 通用音频 LLM | 8 类任务、26 个数据集 | [arXiv:2406.16020](https://arxiv.org/abs/2406.16020) |
| **SD-Eval**（NeurIPS 2024） | 副语言理解 | 情感、韵律、年龄、口音 | [arXiv:2406.13340](https://arxiv.org/abs/2406.13340) |
| **WavBench**（2026） | **端到端口语对话** | 推理、口语化（colloquialism）、副语言 | [arXiv:2602.12135](https://arxiv.org/abs/2602.12135)｜[数据集](https://huggingface.co/datasets/WavBench/WavBench) |
| **AIR-Bench**（2024） | 音频理解 | 多任务音频理解 | [arXiv:2402.07729](https://arxiv.org/abs/2402.07729) |
| **Dynamic-SUPERB**（2023–） | 指令化语音任务 | 动态扩展、社区协作 | [arXiv:2309.09510](https://arxiv.org/abs/2309.09510) |
| **MMAU**（ICLR 2025） | 音频理解与推理 | 多任务音频推理 | [ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/d36f208919582785db965fe648b9fe59-Abstract-Conference.html) |
| **VoiceAgentBench**（2025） | 语音 agent 任务 | agentic 工具使用 | [arXiv:2510.07978](https://arxiv.org/abs/2510.07978) |
| **MLC-SLM Challenge**（Interspeech 2025） | 多语言对话语音 LM | 多语种对话 ASR/理解 | [ISCA](https://www.isca-archive.org/mlcslm_2025/mei25_mlcslm.html) |
| **CS3-Bench**（2025） | 中英语码转换 S2S | 语码转换鲁棒性 | [arXiv:2510.07881](https://arxiv.org/abs/2510.07881) |
| **SpeechIQ**（ACL 2025） | 认知层次化语音理解 | 记忆/理解/推理分层 | [ACL 2025](https://aclanthology.org/2025.acl-long.1466/) |
| **LALM 评测综述**（EMNLP 2025） | 指标与基准全景 | 碎片化问题梳理 | [ACL 2025](https://aclanthology.org/2025.emnlp-main.514/) |

**ASR WER 之外的指标现状**：
| 指标类别 | 代表指标 | 成熟度 |
|---|---|---|
| 语义准确率 | 意图/槽位准确率、VoiceBench 类问答准确率 | 较成熟 |
| 时序行为 | 打断响应延迟、抢话率、backchannel 命中率 | **指标已提出，与主观体验的相关性未验证** |
| 副语言 | 情感分类准确率（理解侧） | 理解侧较成熟 |
| 韵律/自然度 | MOS、偏好胜率（人工） | **无可靠自动指标** |
| 风格一致性 | 跨轮风格保持度（Style Amnesia[55] 提出） | 早期 |

---

## 四、学术上的核心开放问题（按重要性排序）

1. **"该谁说"的可训练、可评测建模仍是未解核心。**
   理由：这是 2025 年全领域的范式转移点，但 Full-Duplex-Bench[28] 已显示现有模型在停顿处理上系统性失败（过于急于回应），而 FLEXI[31] 显示模型几乎不主动产生 backchannel。更关键的是，**评估这些行为的方法本身没有被验证**——我们甚至不确定自动指标能否代表人的感受。没有评测就没有可靠的进步。

2. **语音-文本的 modality gap 不是识别问题，而是表征问题。**
   理由：多项独立研究[61][62]指向同一结论：即使转写正确，语音输入下的语言能力仍低于文本。这意味着整个领域把"提高 ASR 质量"当作主要抓手的方向可能是错的，需要**训练目标层面**的对齐（如 Moshi 的 inner monologue[5]、SpeechTokenizer 的语义蒸馏[36]）。

3. **副语言的生成与跨轮一致性。**
   理由：Style Amnesia[55] 暴露了一个此前未被命名的问题：模型能在单轮"演"出风格，却不能在多轮保持。副语言是端到端 S2S 相对级联方案的**唯一不可替代优势**，如果这一点做不好，端到端的技术合理性会被削弱。

4. **延迟报告缺乏统一口径。**
   理由：这不是"难"的问题，而是"没人做"的问题，却直接导致跨论文结论不可比、工程选型无依据。属于低成本高收益的社区基建缺口。

5. **语音 function calling 的口语规范化与流式调用。**
   理由：这是离商业价值最近、公开研究最少的方向。VoiceAgentBench[49] 与 Spoken Function Calling[50] 刚起步，而真实产品已经把语音 agent 与工具调用绑在一起。

6. **鲁棒性的层级问题**：口音/方言/语码转换/噪声仍然显著（CS3-Bench[58]、AfriSpeech-MultiBench[59]），但更危险的是"看似识别正确、实则语义漂移"的失败，这类失败**在现有基准上不可见**。

7. **Tokenizer 与 LLM 的联合优化**：目前几乎所有系统都是"先训 codec、冻住、再训 LLM"，这个两阶段解耦是否是次优的，缺乏系统实验。

---

## 五、对工程落地的启示

### 现在就能用的学术结论

1. **帧率优先于一切延迟优化。** 把语音 token 帧率压到 12.5 Hz 或 6.25 Hz 量级（Moshi[5]、Qwen2.5-Omni[7]）是从架构层面降低延迟，比在推理侧做工程优化收益大一个数量级。自建 S2S 系统时 tokenizer 选型应作为第一决策。
2. **Thinker–Talker 解耦是可复制的架构范式。** 理解用大模型、生成用小模型 + 流式 decoder（Qwen2.5-Omni[7]、Step-Audio[9]、LLaMA-Omni[13]），这种解耦已经在多个独立团队上被验证有效，且便于分别迭代。
3. **"先文本后声学"是廉价且有效的质量补丁。** Moshi 的 inner monologue[5] 证明给声学生成加一个文本草稿能显著改善语言质量，这个思路可以直接移植到级联与半级联系统。
4. **不要在停顿时急于回应。** Full-Duplex-Bench[28] 的"过度回应"结论是当前的普遍缺陷，工程上可通过显式提升停顿容忍阈值来改善体感——这是一个不需要重训模型的调参收益。
5. **用 Full-Duplex-Bench / VoiceBench / SD-Eval 做回归测试。** 这三个基准分别覆盖时序行为、综合能力、副语言，是目前最成熟的公开选择。
6. **可插拔双工改造优于重训。** FlexDuo[23] 的思路说明"给现有回合制模型加一个听/说状态模块"是性价比很高的路径。

### 现在还不能用的

1. **不要指望模型自发获得自然的多轮副语言表达。** Style Amnesia[55] 表明风格在多轮中会退化；NVSpeech[54] 类方法仍是研究阶段。产品上若要稳定的情感/语气，短期内仍需在 TTS 侧做显式控制。
2. **不要把语音 agent 直接接复杂工具调用。** VoiceAgentBench[49] 与 Spoken Function Calling[50] 显示参数级错误是主要失败来源，尤其是数字、ID、代码标识符。稳妥做法是**语音识别 + 文本澄清确认**，而非一步到位的语音 function calling。
3. **不要相信论文里报告的"延迟"数字可直接横比。** 口径不统一（见 2.4），必须自己在目标链路上端到端测量 TTFA 与打断响应延迟。
4. **不要假设"识别准确率高 = 对话体验好"。** modality gap 研究[61][62] 说明语言能力损失独立于识别错误，需要单独评测与单独干预。
5. **不要在没有人工听测的情况下上线韵律/自然度相关的改动。** 自动韵律指标目前不可靠。

### 一句话总结

2023–2025 年这个领域的技术重心已经从"**让模型会说话**"（tokenizer + 端到端架构）转移到"**让模型知道什么时候该说话、用什么语气说话**"（全双工 + 副语言）。前者已经有可复制的工程答案，后者连**怎么衡量**都还没解决——这正是当前学术与工程之间最大的缺口。

---

## 参考文献

**端到端 S2S 模型**

1. Generative Spoken Dialogue Language Modeling (dGSLM) — https://arxiv.org/abs/2203.16502
2. AudioPaLM: A Large Language Model That Can Speak and Listen — https://arxiv.org/abs/2306.12925
3. SpeechGPT: Empowering Large Language Models with Intrinsic Cross-Modal Conversational Abilities (EMNLP 2023 Findings) — https://aclanthology.org/2023.findings-emnlp.1055/
4. SpiRit-LM: Interleaved Spoken and Written Language Model (TACL 2025) — https://aclanthology.org/2025.tacl-1.2/
5. Moshi: a speech-text foundation model for real-time dialogue — https://arxiv.org/abs/2410.00037 ｜原文 PDF: https://kyutai.org/Moshi.pdf ｜代码: https://github.com/kyutai-labs/moshi
6. GLM-4-Voice: Towards Intelligent and Human-Like End-to-End Spoken Chatbot — https://arxiv.org/abs/2412.02612
7. Qwen2.5-Omni Technical Report — https://arxiv.org/abs/2503.20215
8. Qwen3-Omni Technical Report — https://arxiv.org/abs/2509.17765
9. Step-Audio（Unified Understanding and Generation in Intelligent Speech Interaction）— https://arxiv.org/abs/2502.11946
10. Step-Audio-Tokenizer / Dual-Codebook Tokenization — https://huggingface.co/stepfun-ai/Step-Audio-Tokenizer ｜https://deepwiki.com/stepfun-ai/Step-Audio/2.1-dual-codebook-tokenization
11. Step-Audio 2 Technical Report — https://arxiv.org/abs/2507.16632
12. MiniCPM-o 2.6 模型卡 — https://huggingface.co/openbmb/MiniCPM-o-2_6 ｜代码: https://github.com/OpenBMB/MiniCPM-o （**arXiv 编号未核实**）
13. LLaMA-Omni: Seamless Speech Interaction with Large Language Models (ICLR 2025) — https://arxiv.org/abs/2409.06666
14. Minmo / LLaMA-Omni 2.5 — https://arxiv.org/abs/2506.04518
15. Mini-Omni: Language Models Can Hear, Talk While Thinking in Streaming — https://arxiv.org/abs/2408.16725
16. Freeze-Omni: A Smart and Low Latency Speech-to-speech Dialogue Model with Frozen LLM (ICML 2025) — https://arxiv.org/abs/2411.00774 ｜https://icml.cc/virtual/2025/poster/43854
17. Kimi-Audio Technical Report — https://arxiv.org/abs/2504.18425
18. SLAM-Omni: Timbre-Controllable Voice Interaction System with Single-Stage Training (ACL 2025 Findings) — https://aclanthology.org/2025.findings-acl.115.pdf
19. Ultravox（Fixie AI）— https://github.com/fixie-ai/ultravox ｜https://huggingface.co/fixie-ai/ultravox-v0_4_1-mistral-nemo （**arXiv 编号未核实**）
20. OpenAI: Introducing the Realtime API — https://openai.com/index/introducing-the-realtime-api/
21. OpenAI: Introducing gpt-realtime — https://openai.com/index/introducing-gpt-realtime/
22. GPT-4o System Card — https://openai.com/index/gpt-4o-system-card/
23. FlexDuo: A Pluggable System for Enabling Full-Duplex Capabilities in Speech Dialogue Systems — https://arxiv.org/abs/2502.13472
24. Qwen2-Audio — https://arxiv.org/abs/2407.10759
25. SALMONN: Towards Generic Hearing Abilities for Large Language Models — https://arxiv.org/abs/2310.13289
26. PersonaPlex: Voice and Role Control for Full Duplex Conversational Speech Models（2026）— https://arxiv.org/abs/2602.06053
27. BayLing-Duplex: Native Full-Duplex Speech Dialogue with a Single Autoregressive LLM（2026）— https://arxiv.org/abs/2606.14528

**全双工 / 轮次转换 / 评测**

28. Full-Duplex-Bench: A Benchmark to Evaluate Full-duplex Spoken Dialogue Models on Turn-taking Capabilities — https://arxiv.org/abs/2503.04721 ｜https://github.com/DanielLin94144/Full-Duplex-Bench
29. Full-Duplex-Bench v1.5: Evaluating Overlap Handling for Full-Duplex Speech Models — https://arxiv.org/abs/2507.23159
30. Full-Duplex-Bench v2: A Multi-Turn Evaluation Framework for Duplex Dialogue Systems with an Automated Examiner — https://arxiv.org/abs/2510.07838
31. FLEXI: Benchmarking Full-duplex Human-LLM Speech Interaction — https://arxiv.org/abs/2509.22243
32. From Turn-Taking to Synchronous Dialogue: A Survey of Full-Duplex Spoken Language Models — https://arxiv.org/abs/2509.14515
33. A Survey of Full-Duplex Spoken Dialogue Systems: Architectural Hierarchy, Interaction Ontology, and Decision State Machine（2026）— https://arxiv.org/abs/2606.19453

**语音 tokenizer / codec**

34. SoundStream: An End-to-End Neural Audio Codec — https://arxiv.org/abs/2107.03312
35. High Fidelity Neural Audio Compression (EnCodec) — https://arxiv.org/abs/2210.13438
36. SpeechTokenizer: Unified Speech Tokenizer for Speech Large Language Models — https://arxiv.org/abs/2308.16692

**延迟 / 流式 / 投机解码**

37. Time to First Token: The Voice Agent Latency Metric（AssemblyAI 工程博客）— https://www.assemblyai.com/blog/time-to-first-token-voice-agents
38. Fast Speculative Decoding for Transformer（Leviathan, Kalman, Matias）— https://arxiv.org/abs/2211.17192
39. TRADE: Transducer-Augmented Decoder for Speech LLM（2026）— https://arxiv.org/abs/2606.08486
40. RelayS2S: Dual-Path Speculative Generation for Real-Time Dialogue（2026）— https://arxiv.org/abs/2603.23346
41. FastEmit: Low-latency Streaming ASR with Sequence-level Emission Regularization — https://arxiv.org/abs/2010.11148

**评测基准**

42. VoiceBench: Benchmarking LLM-Based Voice Assistants（TACL）— https://arxiv.org/abs/2410.17196 ｜https://direct.mit.edu/tacl/article-pdf/doi/10.1162/TACL.a.628/2593104/tacl.a.628.pdf
43. SpokenWOZ: A Large-Scale Speech-Text Benchmark for Spoken Task-Oriented Dialogue Agents (NeurIPS 2023 D&B) — https://arxiv.org/abs/2305.13040 ｜https://proceedings.neurips.cc/paper_files/paper/2023/file/7b16688a2b053a1b01474ab5c78ce662-Paper-Datasets_and_Benchmarks.pdf
44. AudioBench: A Universal Benchmark for Audio Large Language Models — https://arxiv.org/abs/2406.16020
45. SD-Eval: A Benchmark Dataset for Spoken Dialogue Understanding Beyond Words (NeurIPS 2024 D&B) — https://arxiv.org/abs/2406.13340
46. WavBench: Benchmarking Reasoning, Colloquialism, and Paralinguistics for End-to-End Spoken Dialogue Models — https://arxiv.org/abs/2602.12135 ｜https://huggingface.co/datasets/WavBench/WavBench
47. Towards Holistic Evaluation of Large Audio-Language Models: A Comprehensive Survey (EMNLP 2025) — https://aclanthology.org/2025.emnlp-main.514/
48. SpeechIQ: Speech-Agentic Intelligence Quotient Across Cognitive Levels in Voice Understanding by Large Language Models (ACL 2025) — https://aclanthology.org/2025.acl-long.1466/
49. VoiceAgentBench: Are Voice Assistants ready for agentic tasks? — https://arxiv.org/abs/2510.07978
50. Spoken Function Calling: A New Perspective on Spoken Language Understanding for Large Audio Language Models — https://arxiv.org/abs/2608.05126
51. Audio2Tool: Speak, Call, Act — A Dataset for Benchmarking Speech Tool Use — https://arxiv.org/abs/2604.22821
52. BFCL Audio: An Audio Function Calling Evaluation for Large Language Models (ICML 2026) — https://icml.cc/virtual/2026/poster/61489
53. emotion2vec: Self-Supervised Pre-Training for Speech Emotion Representation (ACL 2024 Findings) — https://aclanthology.org/2024.findings-acl.931.pdf
54. NVSpeech: An Integrated and Scalable Pipeline for Human-Like Speech Modeling with Paralinguistic Vocalizations — https://arxiv.org/abs/2508.04195
55. Style Amnesia: Investigating Speaking Style Degradation and Mitigation in Multi-Turn Spoken Language Models — https://arxiv.org/abs/2512.23578
56. From "um" to "yeah": Producing, predicting, and regulating information flow in human conversation — https://arxiv.org/abs/2403.08890
57. Exploring Emotional Nuances in Spoken Dialogue: Dataset Construction and Prediction of Emotional Dialogue Breakdown (IWSDS 2026) — https://aclanthology.org/2026.iwsds-1.9/
58. CS3-Bench: Evaluating and Enhancing Speech-to-Speech LLMs for Mandarin-English Code-Switching — https://arxiv.org/abs/2510.07881
59. AfriSpeech-MultiBench: A Verticalized Multidomain Multicountry Benchmark Suite for African Accented English ASR (IJCNLP 2025) — https://aclanthology.org/2025.ijcnlp-long.190/
60. Speak & Spell: LLM-Driven Controllable Phonetic Error Augmentation for Robust Dialogue State Tracking — https://arxiv.org/abs/2409.06263
61. Understanding the Modality Gap: An Empirical Study on the Speech-Text Alignment Mechanism of Large Speech Language Models (EMNLP 2025) — https://aclanthology.org/2025.emnlp-main.262/
62. Anatomy of the Modality Gap: Dissecting the Internal States of End-to-End Speech LLMs（2026）— https://arxiv.org/abs/2603.01502

**其他相关**

63. Dynamic-SUPERB: Towards a Dynamic, Collaborative, and Comprehensive Instruction-Tuning Benchmark for Speech — https://arxiv.org/abs/2309.09510
64. AIR-Bench: Benchmarking Large Audio-Language Models via Generative Comprehension (ACL 2024) — https://arxiv.org/abs/2402.07729 ｜https://aclanthology.org/2024.acl-long.109.pdf
65. MMAU: A Massive Multi-Task Audio Understanding and Reasoning Benchmark (ICLR 2025) — https://proceedings.iclr.cc/paper_files/paper/2025/hash/d36f208919582785db965fe648b9fe59-Abstract-Conference.html
66. Interspeech 2025 MLC-SLM Challenge（多语言对话语音语言模型挑战赛）— https://www.isca-archive.org/mlcslm_2025/mei25_mlcslm.html
67. Step-Audio-AQAA: a Fully End-to-End Expressive Large Audio Language Model — https://arxiv.org/abs/2506.08967
68. Baichuan-Omni-1.5 Technical Report — https://arxiv.org/abs/2501.15368
69. VITA-1.5: Towards GPT-4o Level Real-Time Vision and Speech Interaction — https://arxiv.org/abs/2501.01957
70. Closing the Gap Between Text and Speech Understanding in LLMs (ICLR 2026) — https://proceedings.iclr.cc/paper_files/paper/2026/hash/ec4c05921d70202023898568a2a8c002-Abstract-Conference.html
