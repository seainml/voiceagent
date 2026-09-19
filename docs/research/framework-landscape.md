# Voice Agent 框架与工具生态调研（2026-09 版）

> 数据采集时间：**2026-09-19（UTC）**。Star 数由 GitHub API / 仓库页实时抓取；价格与版本号来自各官网页面，均可能随时变动。凡未在官方页面直接核到的数字，一律标注「需核实」并给出核实链接。

---

## 一、生态全景图（分层）

| 层 | 职责 | 代表组件 | 关键约束 |
|---|---|---|---|
| **传输层** | 音频/视频/数据双向实时通道 | WebRTC（LiveKit、Daily、SmallWebRTC、Vonage、Beyond Presence）、WebSocket（FastAPI WS、各家实时 API）、SIP/电话（Twilio、Telnyx、Plivo、Exotel、Vonage）、Media over QUIC（Pipecat 新增）、本地音频（macOS AVFoundation、ESP32） | WebRTC 抗弱网、天然带抖动缓冲与（部分）3A；WebSocket 简单、服务端接力友好，但弱网与回声处理要自己做 |
| **编排层** | 把 VAD→ASR→LLM→TTS 串成可打断的流水线，或对接端到端实时模型 | 开源：Pipecat、LiveKit Agents、Vocode、Bolna、TEN-Agent、Home Assistant Assist、Rasa Pro；托管：Vapi、Retell、Bland、Synthflow、ElevenLabs Agents | 打断（barge-in）、轮次检测、工具调用、电话接续是四大难点 |
| **模型层** | 听、想、说 | ASR（Whisper 系、SenseVoice/FunASR、Deepgram、AssemblyAI、Paraformer）；TTS（ElevenLabs、OpenAI、Azure、MiniMax、CosyVoice、Fish Speech、Kokoro、GPT-SoVITS）；端到端实时（OpenAI Realtime、Gemini Live、Qwen-Omni-Realtime、豆包、Azure OpenAI Realtime）；VAD/AEC（Silero、ten-vad、WebRTC VAD、AEC3、SpeexDSP、Krisp） | 中文质量与延迟差距最大的就是这一层 |
| **工具层** | 让语音 Agent 真的"能干活" | Function calling（各实时 API 原生）、MCP（LiveKit `MCPToolset`、`pipecat-mcp-server`、OpenAI Realtime 文档的 Tools and MCP 章节）、Agent CLI 语音壳（Heard、AgentVoice） | 语音场景对工具延迟极敏感，工具设计必须"快返回、可播报" |

**一句话结论**：2026 年的分水岭是「**级联式（ASR+LLM+TTS）** vs **端到端实时（S2S）**」。级联式可控、可换供应商、便于审计；端到端延迟与语气更自然，但调试、成本与工具调用可控性更差。多数生产系统实际是混合：主链路用 S2S，工具与兜底走级联。

---

## 二、编排框架逐项对比

| 框架 | 定位 | 传输层 | 开源/许可 | 生态成熟度（2026-09-19） | 适合场景 |
|---|---|---|---|---|---|
| **Pipecat** | Python 优先的「帧-处理器-流水线」编排框架，官方定位"实时语音与多模态 AI" | Daily、LiveKit、SmallWebRTC(P2P)、FastAPI WebSocket、WebSocket、Media over QUIC、WhatsApp、Vonage、HeyGen/Tavus 数字人 | 开源，**BSD-2-Clause** | 仓库 15,660★；PyPI `pipecat-ai` **1.11.0**（2026-09-18）；1.0 起统一 `LLMContext`、`LLMUserAggregatorParams`，生态插件最多 | 快速试错、需要自由拼装 STT/LLM/TTS、多 Agent 与分布式 Worker |
| **LiveKit Agents** | 构建"可编程实时参与者"，与 LiveKit WebRTC 云/自托管深度耦合 | 自研 WebRTC（客户端 SDK 全平台）、SIP/电话栈、数据通道 RPC | 开源，**Apache-2.0** | 14,270★；PyPI `livekit-agents` **1.8.2**（2026-09-15）；有 JS/TS 版 `agents-js`；内置作业调度、语义轮次检测、LiveKit Inference 模型网关 | 生产级 WebRTC/电话场景，需要调度、多端、可观测性 |
| **Vocode** | 早期最流行的开源语音 Agent 库（电话/会议/助手） | 电话、Web、Zoom | 开源，**MIT** | 3,793★，但主干**最后提交 2024-11-15**，文档仍在，实际处于低维护状态 → **新项目不建议作主选** | 仅作历史参考或已有存量系统 |
| **Bolna** | "全栈语音 Agent"：OSS 自托管 + 托管云，强电话与多语言/多供应商抽象 | 电话为主（多家 Telephony），支持 Speech-to-Speech 与级联两条链路 | 开源，**MIT** | 764★；PyPI `bolna` **0.10.244**（2026-09-18），迭代非常活跃；有 Agent Studio、Campaign、图式 Agent | 想做"自托管 Vapi"、外呼/呼入、多语言切换 |
| **Rasa** | 对话式 AI 平台，CALM（LLM 驱动流程）+ 传统 NLU 双轨 | 通过 Rasa Pro 的 voice 能力对接 ASR/TTS（社区 demo 用 Deepgram+Rime 编排） | 核心 `rasa` OSS **Apache-2.0**；Rasa Pro 为商业产品 | 21,325★；OSS PyPI **3.6.21**（2025-01-14，更新放缓），商业版是主推 | 企业级对话流程、合规、与业务系统深度集成的语音坐席 |
| **Home Assistant Assist** | 智能家居语音助手流水线（唤醒词→STT→意图→TTS） | WebSocket API（`assist_pipeline/run`）+ Wyoming 协议连接各语音服务 | 开源，**Apache-2.0**（HA Core 90,788★） | 极成熟的家居生态。官方数据：Speech-to-Phrase 在 HA Green / 树莓派 4 上**<1 秒**；Whisper 在树莓派 4 上**约 8 秒**、Intel NUC 上**<1 秒**；Piper 在树莓派上每秒可生成 **1.6 秒**语音 | 本地/私有语音控制家居，或作为"本地端到端"的参考实现 |
| **Vapi** | 托管语音 Agent 平台（开发者导向） | Vapi Telephony/SIP、Vapi WebSocket、Daily WebRTC（页面标注均为 Free） | 闭源（SDK 开源） | 模型目录全、并发与合规模块清晰；定价见下 | 想几小时上线电话 Agent、不想运维 |
| **Retell AI** | 托管电话 Agent，强合规与电话能力 | 自带电话基础设施，支持 Twilio/Telnyx 等 | 闭源（有 Python SDK 59★） | 定价透明（下面数字来自官网定价页） | 呼叫中心、外呼、需要 SLA |
| **Bland AI** | 托管电话 Agent，"无 token 转嫁"打包计价 | 自建电话链路，支持 on-prem/VPC（Enterprise） | 闭源 | 主打稳定与合规，企业功能多 | 大规模外呼、受监管行业 |
| **Synthflow** | 无代码/低代码语音 Agent 平台 | 托管，含 WhatsApp 等渠道 | 闭源 | 面向业务侧，G2 评价量大 | 业务团队自助搭建客服/前台 |

**托管平台价格（官网页面，2026-09-19）**：

| 平台 | 计价（PAYG） | 备注 |
|---|---|---|
| Vapi | 模型透传：Deepgram 转写 **$0.0095–0.0099/min**、OpenAI 智能层 **$0.0077–0.0452/min**、ElevenLabs 声音 **$0.0146–0.0238/min**；Core **$29/月**、Pro **$999/月起**（Vapi 托管费 10% 计）；并发加线 **$10/线/月**；HIPAA **$2,000/月** | 电话侧 Twilio 呼入 $0.008/min、呼出 $0.014/min |
| Retell | 总价 **$0.07–$0.31/min**；拆分：LLM $0.04/min、Retell 语音基础设施 $0.055/min、TTS $0.015/min（ElevenLabs 音色 $0.04/min） | 含 20 并发，PAYG |
| Bland | Start **$0.14/min**（$0 平台费）；Build **$0.12/min + $299/月**；转账 $0.05/$0.04/min | 明确"无 token 费、无模型转嫁费" |
| Synthflow | 官网定价页与计费文档未给出可引用的每分鐘价（JS 渲染）→ **需核实**：[定价页](https://synthflow.ai/pricing)、[Usage & Billing](https://docs.synthflow.ai/usage) | 计费按秒累计、按整分钟出账 |
| ElevenLabs Agents | Starter 档 **$0.08/min**（2026-05-07 公告自 $0.10 下调） | 见 [官方博客](https://elevenlabs.io/blog/weve-lowered-api-agents-pricing-and-introduced-pay-as-you-go) |

**选型要点与坑**：
- **WebRTC vs WebSocket**：浏览器端优先 WebRTC；服务端到服务端、电话网关用 WebSocket 更省事。Pipecat 文档把这条作为默认建议。
- **Vocode 的"僵尸化"**是最大坑：Star 高但主干停更近两年，依赖的第三方 API 变更会直接踩雷。
- **托管平台的隐藏成本**在并发线与电话分钟；Vapi 并发 $10/线/月、Pro 需 $999/月最低消费，量小反而比自建贵。
- **自建的最大成本是"打断与回声"**，不是模型钱（见第六节）。

---

## 三、端到端实时语音 API 对比

| 产品 | 协议 | 原生 function calling | 打断（barge-in） | 中文效果 | 计费（官网，2026-09-19） |
|---|---|---|---|---|---|
| **OpenAI Realtime** | WebRTC / WebSocket / SIP（[文档](https://platform.openai.com/docs/guides/realtime)） | 是，且官方文档目录含 **Tools and MCP**（functions + MCP servers + connectors） | 是（文档首推 barge-in、低首音延迟、工具调用） | 好，中文语音自然度业界前列 | `gpt-realtime-2.1`：音频输入 **$32/1M token**、缓存 $0.40、音频输出 **$64/1M**；文本 $4/$24；`gpt-realtime-2.1-mini`：音频 $10/$20；`gpt-live-1` 语音会话 **$0.05/min**；`gpt-realtime-translate` $0.034/min、`gpt-realtime-whisper` $0.017/min（[定价页](https://platform.openai.com/docs/pricing)） |
| **Google Gemini Live** | 有状态 WebSocket（WSS）为主，文档亦提及 WebRTC 路径 | 是（function calling + Google Search 组合） | 是，官方明确 "Users can interrupt the model at any time" | 好，多语言强 | `gemini-3.8-live`：音频输入 **$3/1M（≈$0.005/min）**、音频输出 **$12/1M（≈$0.018/min）**；文本 $0.75/$4.50；`gemini-2.5-flash-native-audio` 音频 $3/$12（[定价页](https://ai.google.dev/gemini-api/docs/pricing)、[Live 文档](https://ai.google.dev/gemini-api/docs/live)） |
| **阿里云百炼 Qwen-Omni-Realtime** | **WebSocket / WebRTC / AOQ 三选**（WebRTC 音频走 UDP，内置回声消除与降噪；弱网优先 AOQ） | 是（"支持 Function Calling，模型可自主判断是否调用外部工具"） | 是，且有**语义打断**（过滤附和声与背景音误触发） | 中文母语级；支持 **55 种音色**（47 多语言 + 8 方言），Plus/Flash 支持**声音复刻** | `qwen3.5-omni-flash-realtime`：文本/图像输入 **$0.55**、音频输入 **$4.5**、音频输出档最高 **$17.7**（每 1M token）；`qwen3.5-omni-plus-realtime` 对应 $2.1 / $16.5 / $62。分档字段较多，**务必以[计费页](https://www.alibabacloud.com/help/en/model-studio/model-pricing)为准**（[实时模型文档](https://help.aliyun.com/zh/model-studio/realtime)） |
| **火山引擎豆包实时语音** | 端到端实时语音走 WebSocket/RTC 方案（AI 音视频互动） | 依模型能力，**需核实** | **需核实** | 中文优秀，国内落地最广 | 官方计费页为 JS 渲染，未能抓到可引用数字 → **需核实**：[豆包语音计费说明](https://www.volcengine.com/docs/6561/1359370)、[AI 音视频互动方案计费](https://docs.volcengine.com/docs/6348/2123214) |
| **Azure OpenAI Realtime** | **WebRTC（推荐）/ WebSocket / SIP**，GA 端点 `/openai/v1` | 是（与 OpenAI 同族工具模型） | 是 | 同 OpenAI 模型族 | 模型清单（[官方 quickstart](https://learn.microsoft.com/en-us/azure/ai-services/openai/realtime-audio-quickstart)）：`gpt-4o-realtime-preview`(2024-12-17) … `gpt-realtime`(2025-08-28)、`gpt-realtime-mini`(2025-12-15)、`gpt-realtime-1.5`(2026-02-23)、`gpt-realtime-2`(2026-05-07)、`gpt-realtime-translate`/`-whisper`；转写/翻译类按**时长**计费，具体单价见 Azure OpenAI 定价页（**需核实**） |
| **ElevenLabs Conversational AI (ElevenAgents)** | WebSocket/WebRTC 客户端 + 电话集成 | 是（Agent tools / 客户端工具） | 是 | 中文 TTS 顶级，LLM 侧自带模型 | Agents **$0.08/min**（Starter 档）；TTS Flash **$0.05/1k token**、Scribe v2 STT **$0.22/1k token**（Starter），[官方降价公告 2026-05-07](https://elevenlabs.io/blog/weve-lowered-api-agents-pricing-and-introduced-pay-as-you-go) |

**结论**：
- **中文 + 合规在国内落地，Qwen-Omni-Realtime 与豆包是首选**；Qwen 的文档把 WebRTC/AOQ 的内置 AEC 写得很明确，能省掉一大块音频前处理工作。
- **OpenAI Realtime 的单价按 token 计，很容易超预算**：音频输入 $32/1M token 折算下来通常远高于 Gemini 的 $3/1M 音频输入。做成本对比时，**务必按"每分钟"折算**再决策。
- Azure 版本迭代明显滞后于 OpenAI 第一方（`gpt-realtime-2` 为 2026-05），但胜在区域/合规/企业采购。

---

## 四、ASR 对比

| 方案 | 本地/云 | 中文 | 流式 | 速度/延迟（官方口径） | 许可 |
|---|---|---|---|---|---|
| **OpenAI Whisper** | 本地 | 好 | 否（需自行切窗） | 基线实现，见 faster-whisper 对比表 | MIT（109,364★） |
| **faster-whisper** | 本地 | 好 | 伪流式（VAD 切段） | CTranslate2 重写，"相同精度下最高快 4 倍、内存更省"；官方 benchmark：13 分钟音频 large-v2，whisper fp16 2m23s vs faster-whisper fp16 1m03s / int8 59s / batch8+int8 16s（RTX 3070 Ti） | MIT（25,469★，最后提交 2025-11） |
| **whisper.cpp** | 本地（含端侧） | 好 | 伪流式 | CPU 上比 openai/whisper 快得多（small 模型 13 分钟音频：2m05s vs 6m58s，8 线程 i7-12700K） | MIT（53,777★，活跃） |
| **Distil-Whisper** | 本地 | 一般（蒸馏主要面向英文） | 否 | 更快更小，但仓库最后提交 **2025-01-08** | MIT（4,117★） |
| **NVIDIA Parakeet TDT 0.6B** | 本地（NeMo / NIM） | **不支持中文**（v3 为 25 种欧洲语言，v2 为英语） | 是（TDT 流式） | 官方模型卡只给 WER 指标，速度需实测 → 具体 RTFx **需核实** | **CC-BY-4.0**（模型），[模型卡](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) |
| **SenseVoice / FunASR** | 本地 | **强**（官方称中文、粤语优于 Whisper） | 准流式（VAD 分段 + 非自回归） | 非自回归端到端：同参数下比 Whisper-Small **快 5 倍以上**、比 Whisper-Large **快 15 倍**；已迁至 [QwenAudio/SenseVoice](https://github.com/QwenAudio/SenseVoice)（9,328★） | MIT（FunASR 20,430★，`funasr` 1.4.16） |
| **阿里 Paraformer（百炼）** | 云 | 强 | `paraformer-realtime-v2` 是流式 | 计费按音频秒数：录音文件 `paraformer-v2` **$0.000012/秒**；实时 `paraformer-realtime-v2` **$0.000035/秒（≈$0.0021/min）**，无免费额度 | 商业 API |
| **Deepgram Nova / Flux** | 云 | 一般（Flux 主打英语/多语） | 是 | Nova-3 流式 **$0.0048/min**（促销价，原价 $0.0077）；多语 $0.0058；Flux English **$0.0065/min**（内置轮次检测与"自然打断"） | 商业 API，[定价页](https://deepgram.com/pricing) |
| **AssemblyAI Universal** | 云 | 多语覆盖有限 | 是 | `Universal-Streaming` **$0.15/小时 = $0.0025/min**；`Universal-3.5 Pro Realtime` **$0.45/小时 = $0.0075/min**（18 语种、上下文延续） | 商业 API，[定价页](https://www.assemblyai.com/pricing) |
| **字节/火山 ASR** | 云 | 强 | 是 | 官方计费页 JS 渲染，未取到可引用数字 → **需核实** | 商业 API |

**坑**：Whisper 系「流式」本质是 VAD 切段 + 快速解码，**首字延迟取决于切段策略**，别指望它像 Paraformer/Flux 那样真正的 chunk 级流式。若做中文实时对话，**SenseVoice/FunASR 自建或 Paraformer 云 API 是更优解**。

---

## 五、TTS 对比

| 方案 | 流式首包延迟 | 中文自然度 | 声音克隆 | 成本（官网，2026-09-19） | 许可 |
|---|---|---|---|---|---|
| **ElevenLabs** | 低（Flash/低延迟档） | 顶级的非中文母语产品，中文强 | 支持（Instant / Professional） | Flash **$0.05/1k token**（Creator 档，原 $0.11）；Business 档宣称低至 **$0.05/min** 级别低延迟 TTS；订阅 $0–$990/月 | 商业 API |
| **OpenAI TTS** | 中低 | 好 | 不支持克隆（有 Custom voices 能力，需核实） | `tts-1` **$15/1M 字符**、`tts-1-hd` **$30/1M 字符**；`gpt-4o-mini-tts` 音频输出 **$12/1M token** | 商业 API |
| **Azure Speech** | 低（SDK 流式） | 好，HD 音色自然 | 支持 Custom Neural Voice / Personal Voice | 免费层：Neural TTS **50 万字符/月**免费；**中文每字按 2 字符计费**；实际单价需在定价页按区域查看 → **需核实** [定价页](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/speech-services/) |
| **火山豆包 TTS** | 低 | 中文顶级 | 支持（声音复刻） | 未能取到可引用数字 → **需核实** [计费说明](https://www.volcengine.com/docs/6561/1359370) | 商业 API |
| **MiniMax Speech** | 低 | 中文顶级，情感表现力强 | 支持快速克隆（套餐赠送 10/30/300 个音色位） | HD 系列 **¥630 / 200 万字符 ≈ ¥0.315/千字符**；Turbo **¥360 / 200 万字符 ≈ ¥0.18/千字符**（套餐一，[官方定价](https://platform.minimaxi.com/document/price)） | 商业 API |
| **CosyVoice 2 / Fun-CosyVoice 3.0** | **官方称双流式下延迟低至 150ms** | 好（9 语种 + 18 种以上中文方言/口音） | 支持零样本克隆（跨语种） | 自建算力成本 | Apache-2.0（[仓库](https://github.com/FunAudioLLM/CosyVoice)，最新 Fun-CosyVoice3-0.5B-2512） |
| **Fish Speech / S1-S2** | 需实测 | 好，表现力强 | 支持 | 自建 | **非 OSI 开源**：FISH AUDIO RESEARCH LICENSE，商用需读条款（32,745★，[仓库](https://github.com/fishaudio/fish-speech)） |
| **Kokoro-82M** | 快（82M 小模型） | 一般（英文优先，中文走 `misaki[zh]`） | 不支持克隆 | 自建，成本极低 | **Apache-2.0**（权重可商用，8,895★） |
| **ChatTTS** | 需实测 | 中文对话语气自然 | 不支持 | 自建 | **AGPL-3.0**（商用需注意传染性，39,859★） |
| **GPT-SoVITS** | 需实测（非实时优化） | 好，少样本克隆强 | 强（1 分钟素材） | 自建 | MIT（61,931★） |
| **macOS `say` / `AVSpeechSynthesizer`** | 极低（系统本地） | 中文可懂、机械感明显 | 不支持（可用 Personal Voice，需核实） | **免费**（随系统） | 系统内置，App Store 分发需注意 API 使用条款 |
| **edge-tts** | 低 | 好（微软在线音色） | 不支持 | 免费 | **GPL-3.0**；且是**非官方**调用 Edge 在线服务，无 SLA、随时可能失效（11,982★） |

**坑**：
- **许可陷阱**：Fish Speech 已改为研究许可；ChatTTS 是 AGPL-3.0；edge-tts 是 GPL-3.0 + 非官方服务。用于商业闭源产品前必须逐一确认。
- **"流式首包延迟"官方数字几乎都不可比**：CosyVoice 3 的 150ms 是双流式理想值，实际端到端要叠加 LLM 首 token 与网络。除 CosyVoice 外多数未标称，标 **需实测**。
- **中文按"字符"计费的 Azure 要把汉字算 2 个**，做预算时容易算错一倍。

---

## 六、VAD / 打断 / 回声消除

| 组件 | 类型 | 许可 | 说明 |
|---|---|---|---|
| **Silero VAD** | VAD | **MIT** | 事实标准，10,251★，PyTorch/ONNX，社区默认选择 |
| **WebRTC VAD**（`py-webrtcvad` 等） | VAD | BSD 系（WebRTC 项目） | 极轻量、纯能量/高斯混合，噪声下误报明显，适合兜底 |
| **ten-vad**（TEN-framework） | VAD | **Apache-2.0 + 附加条款**（不得用于与 Agora 竞争的部署） | 2,270★；2025-06 开源 ONNX 模型与预处理代码，2025-07 被 `sherpa-onnx` 集成；README 提供与 Silero 的 PR 曲线复现脚本（谁更优需按域调阈值） |
| **Krisp VIVA SDK** | 语音隔离 + 轮次预测 | 商业 | **端上/CPU 运行**，官方称部署于 2 亿+ 设备、月处理 750 亿+ 分钟；已集成 Discord、RingCentral、Synthflow、Vapi，并作为模块接入 Pipecat 与 Daily；VIVA 主打"Voice Isolation 修掉误打断 + Turn-Taking 预测" |
| **WebRTC AEC3**（`webrtc-audio-processing`） | 回声消除 | BSD-3 | 浏览器外做全双工的**首选**；WebRTC 传输（LiveKit/Daily）通常已含 |
| **SpeexDSP** | AEC / 降噪 / AGC | BSD-3 | 732★，老牌、轻量，AEC 效果弱于 AEC3，适合端侧低算力 |

**工程结论（也是最大的坑）**：
1. **打断 = VAD + 轮次检测 + 播放侧可中断 + 上下文回滚**，四件事缺一不可。VAD 只是第一环。
2. **外放场景必须先做 AEC**，否则 TTS 自己的声音会触发 VAD → 无限自我打断。用 WebRTC 传输（LiveKit/Daily/SmallWebRTC）或端到端 WebRTC API（Qwen-Omni 的 WebRTC/AOQ、OpenAI WebRTC）可以白拿大部分 AEC。
3. **纯 WebSocket 链路（含电话 8k 采样）要自己接 AEC3**，这是自建方案最容易被低估的工作量。
4. 商业方案（Krisp）解决的是"多人/嘈杂环境下的误打断"，成本敏感场景先用 Silero + 语义轮次检测（LiveKit 的语义轮次模型、Pipecat 的 turn analyzer）顶着。

---

## 七、工具与协议层

**1) MCP 在语音场景的可用性 —— 已经从"能不能"变成"怎么用"**
- 规范版本：**2026-07-28**（[modelcontextprotocol.io](https://modelcontextprotocol.io/docs/getting-started/intro)）。
- **LiveKit Agents 已一等公民支持 MCP**：`livekit-agents[mcp]`，用 `MCPToolset` 包装 `MCPServerHTTP`（streamable HTTP / SSE 自动识别）或 `MCPServerStdio`（本地 CLI 型 MCP server）；旧的 `mcp_servers` 参数已弃用。文档：[LiveKit MCP](https://docs.livekit.io/agents/logic/tools/mcp/)。
- **Pipecat 官方提供 `pipecat-mcp-server`**（143★），把语音/视频能力反向暴露给 AI Agent：[仓库](https://github.com/pipecat-ai/pipecat-mcp-server)。
- **OpenAI Realtime 官方文档目录包含 "Tools and MCP"**（functions、MCP servers、connectors），说明第一方实时 API 已把 MCP 纳入工具路径：[Realtime 指南](https://platform.openai.com/docs/guides/realtime)（具体字段与限制以该页为准，建议二次核实）。
- **语音场景的特殊约束**：MCP 工具调用要"快返回、结果可播报"。慢工具（>1s）必须走"先应答再回填"的异步模式，否则体验崩塌。

**2) OpenAI Realtime 的 function calling 实践要点**
- 用 **Agents SDK 的 `RealtimeAgent` + `RealtimeSession`** 起步最省事：浏览器走 WebRTC、服务端走 WebSocket，会话内自动处理音频轮次、工具、打断与 handoff（见官方 Realtime 指南）。
- 服务端先签发 **ephemeral client secret**，前端不要暴露长期 API Key。
- 工具设计原则：语音里"工具名+参数"由模型口播，**函数名要能听懂、参数要少、返回要短**。

**3) Agent CLI 被语音层包装的可行性 —— 已有现成项目**
- **Heard**（Apache-2.0，183★）：自称 "Jarvis for your coding agents"，为 **Claude Code、Codex、OpenClaw、Hermes** 等提供语音层：朗读 Agent 输出、支持免手（hands-free）回话，内置端上 STT 与 Wispr Flow 式听写：[github.com/heardlabs/heard](https://github.com/heardlabs/heard)。
- **AgentVoice**（3★，早期）：自托管语音桥，面向 Cursor / Codex / Claude Code，可从手机说话、可插拔托管：[github.com/dixonSolutions/AgentVoice](https://github.com/dixonSolutions/AgentVoice)。
- **可行性判断**：技术上完全可行（CLI 的 stdout 就是天然的 TTS 输入流，回答走本地 STT 回填 stdin/剪贴板）。真正的坑是**权限与确认交互**：Agent CLI 常需要 y/n 确认，语音层必须实现"读题—确认—回传"的闭环，否则会卡死。做 DSH/Codex 的语音壳，建议走 **Pipecat 或 LiveKit Agents 做编排 + 本地 VAD/TTS**，而不是自己糊 WebSocket。

---

## 八、三档选型推荐

### 档位 A：零成本本地跑通（学习/家居/私有化验证）
**组合**：`Home Assistant Assist 思路` 或 `Pipecat`（BSD-2）+ **faster-whisper / whisper.cpp / SenseVoice** 做 ASR + **Ollama 本地 LLM** + **Piper / Kokoro / macOS say** 做 TTS + **Silero VAD** + **SmallWebRTC 或本地音频**。
- 理由：全链路可离线、零 API 费用、许可干净（MIT/Apache-2.0/BSD）。macOS 上可直接用 `AVSpeechSynthesizer` 把 TTS 成本压到 0。
- 坑：树莓派级硬件上 Whisper 会到 **8 秒级**延迟（HA 官方数据），必须换 Speech-to-Phrase 这类闭集模型或上更强硬件；Kokoro 中文需额外 G2P，别指望开箱即用。

### 档位 B：低延迟中文生产（国内业务首选）
**组合**：**LiveKit Agents 或 Pipecat** 编排 + **端到端实时模型（Qwen-Omni-Realtime / 豆包实时语音）** 或 **Paraformer-realtime-v2 + 豆包/MiniMax TTS** 的级联链路 + **Silero/ten-vad** 做兜底 VAD + **WebRTC 传输（自带 AEC）**。
- 理由：Qwen-Omni-Realtime 原生支持 **WebSocket/WebRTC/AOQ**、**Function Calling**、**语义打断**、**55 音色与声音复刻**，中文场景几乎无短板；AsR 侧 `paraformer-realtime-v2` 约 **$0.0021/min** 是全国产链路的成本基线。
- 坑：① Qwen 计费分档复杂且 Plus/Flash 差异大，**必须按自己的音频时长做实测折算**；② AOQ/WebRTC 的 SDK 生态仍在完善，"需核实"你所用语言的客户端支持度；③ 火山引擎价格页需登录/JS 渲染，**采购前一定走商务确认**，不要按博客数字做预算。

### 档位 C：企业级可托管（合规、SLA、快速上线）
**组合**：**Retell / Vapi / Bland** 任选（电话与合规成熟）或 **Azure OpenAI Realtime**（区域与采购合规），编排与可观测走 **LiveKit Agents / Pipecat**，噪声与误打断加 **Krisp VIVA**。
- 理由：Retell 定价透明（$0.07–0.31/min，含 20 并发）；Vapi 模型目录与并发管理最灵活；Azure Realtime 提供 WebRTC/SIP/WebSocket 与 GA 端点，便于企业架构审查。
- 坑：① 托管平台按月最低消费（Vapi Pro $999/月）与并发线费用会显著抬高小规模成本；② 模型版本迭代快（Azure 的 `gpt-realtime-2` 为 2026-05-07，落后第一方），**上线前锁定版本并订阅其弃用公告**；③ 音频单价按 token 折算极易失控——OpenAI 音频输入 $32/1M token，请务必按分钟做容量模型。

**三条通用建议**：
1. **先做"打断与回声"的 PoC**，再选模型。90% 的语音 Agent 体验问题不在 LLM。
2. **始终准备级联兜底**：S2S 模型换版/涨价/限流时，能一键切到 ASR+LLM+TTS。
3. **所有价格当作时点快照**：本报告数字采集于 **2026-09-19**，下单前复查官方定价页。

---

## 九、参考链接

**编排框架**
- Pipecat：[GitHub](https://github.com/pipecat-ai/pipecat) · [文档](https://docs.pipecat.ai/) · [1.0 迁移指南](https://docs.pipecat.ai/pipecat/migration/migration-1.0) · [MCP Server](https://github.com/pipecat-ai/pipecat-mcp-server)
- LiveKit Agents：[GitHub](https://github.com/livekit/agents) · [文档](https://docs.livekit.io/agents/) · [MCP 工具](https://docs.livekit.io/agents/logic/tools/mcp/) · [Realtime 模型](https://docs.livekit.io/agents/models/realtime/)
- Vocode：[GitHub](https://github.com/vocodedev/vocode-core) · [文档](https://docs.vocode.dev/welcome)
- Bolna：[GitHub](https://github.com/bolna-ai/bolna) · [文档](https://www.bolna.ai/docs/introduction)
- Rasa：[GitHub](https://github.com/RasaHQ/rasa) · [Rasa Pro Voice Assistants](https://rasa-pro-docs.netlify.app/docs/rasa-pro/voice-assistants/)
- Home Assistant：[Core](https://github.com/home-assistant/core) · [Assist 流水线](https://developers.home-assistant.io/docs/voice/pipelines/) · [本地语音助手教程](https://www.home-assistant.io/voice_control/voice_remote_local_assistant/)

**托管平台定价**
- [Vapi Pricing](https://vapi.ai/pricing) · [Retell Pricing](https://www.retellai.com/pricing) · [Bland Pricing](https://www.bland.ai/pricing) · [Synthflow Pricing](https://synthflow.ai/pricing) / [Usage & Billing](https://docs.synthflow.ai/usage) · [ElevenLabs Pricing](https://elevenlabs.io/pricing) / [降价公告](https://elevenlabs.io/blog/weve-lowered-api-agents-pricing-and-introduced-pay-as-you-go)

**实时语音 API**
- [OpenAI Realtime 指南](https://platform.openai.com/docs/guides/realtime) · [OpenAI 定价](https://platform.openai.com/docs/pricing)
- [Gemini Live API](https://ai.google.dev/gemini-api/docs/live) · [Gemini 定价](https://ai.google.dev/gemini-api/docs/pricing)
- [Qwen-Omni 实时模型](https://help.aliyun.com/zh/model-studio/realtime) · [阿里云百炼模型计费](https://www.alibabacloud.com/help/en/model-studio/model-pricing)
- 火山引擎：[豆包语音计费说明](https://www.volcengine.com/docs/6561/1359370) · [AI 音视频互动方案计费](https://docs.volcengine.com/docs/6348/2123214)
- [Azure OpenAI Realtime 快速开始](https://learn.microsoft.com/en-us/azure/ai-services/openai/realtime-audio-quickstart) · [Azure Speech 定价](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/speech-services/) · [Azure TTS 文档](https://learn.microsoft.com/en-us/azure/ai-services/speech-service/text-to-speech)

**ASR**
- [openai/whisper](https://github.com/openai/whisper) · [faster-whisper](https://github.com/SYSTRAN/faster-whisper) · [whisper.cpp](https://github.com/ggml-org/whisper.cpp) · [distil-whisper](https://github.com/huggingface/distil-whisper)
- [QwenAudio/SenseVoice](https://github.com/QwenAudio/SenseVoice) · [modelscope/FunASR](https://github.com/modelscope/FunASR)
- [Parakeet TDT 0.6B v3 模型卡](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) · [NVIDIA-NeMo/Speech](https://github.com/NVIDIA-NeMo/Speech)
- [Deepgram 定价](https://deepgram.com/pricing) · [AssemblyAI 定价](https://www.assemblyai.com/pricing)

**TTS**
- [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice) · [fishaudio/fish-speech](https://github.com/fishaudio/fish-speech) · [hexgrad/kokoro](https://github.com/hexgrad/kokoro) · [2noise/ChatTTS](https://github.com/2noise/ChatTTS) · [RVC-Boss/GPT-SoVITS](https://github.com/RVC-Boss/GPT-SoVITS) · [rany2/edge-tts](https://github.com/rany2/edge-tts) · [MiniMax 语音定价](https://platform.minimaxi.com/document/price)

**VAD / AEC**
- [snakers4/silero-vad](https://github.com/snakers4/silero-vad) · [TEN-framework/ten-vad](https://github.com/TEN-framework/ten-vad) · [xiph/speexdsp](https://github.com/xiph/speexdsp) · [Krisp AI Voice SDK 文档](https://sdk-docs.krisp.ai/docs/getting-started.md)

**工具/协议与语音助手整机项目**
- [Model Context Protocol](https://modelcontextprotocol.io/docs/getting-started/intro)（规范 2026-07-28）
- [heardlabs/heard](https://github.com/heardlabs/heard) · [dixonSolutions/AgentVoice](https://github.com/dixonSolutions/AgentVoice)
- [78/xiaozhi-esp32](https://github.com/78/xiaozhi-esp32)（30,058★）· [huangjunsen0406/py-xiaozhi](https://github.com/huangjunsen0406/py-xiaozhi)（3,478★）· [toverainc/willow](https://github.com/toverainc/willow)（3,105★）· [rhasspy/rhasspy](https://github.com/rhasspy/rhasspy)（2,745★）· [OpenInterpreter/01](https://github.com/OpenInterpreter/01)（5,157★）· [TEN-framework/TEN-Agent](https://github.com/TEN-framework/TEN-Agent)（11,137★）· [k2-fsa/sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)（14,850★）
