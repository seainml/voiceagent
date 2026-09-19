# 架构

> 设计取舍与理由。想直接跑起来看 [README](../README.md)；想调参看 [配置](configuration.md)；
> 想知道为什么这么设计看 [最佳实践](best-practices.md)。

---

## 1. 全景

```
                         ┌──────────────────────────────────────────┐
   麦克风 ──► VAD ──────►│            transport                     │
   (浏览器 / 终端)        │  server/app.py  (WebSocket)              │
                         │  cli.py         (terminal + Speaker)     │
                         └───────────────────┬──────────────────────┘
                                             │  Event 流（唯一出口）
                         ┌───────────────────▼──────────────────────┐
                         │  session.py — 会话状态机                  │
                         │  · turn 生命周期   · barge-in            │
                         │  · 确认握手        · 指标埋点             │
                         └───┬───────────────────────────┬──────────┘
                             │                           │
              ┌──────────────▼────────────┐   ┌──────────▼───────────┐
              │ agent/loop.py             │   │ providers/            │
              │  流式 tool-calling 循环    │◄──┤  asr · tts · llm      │
              │  · 队列化事件流            │   │  三个小 Protocol       │
              │  · 并发执行独立工具         │   └──────────────────────┘
              │  · 历史 sanitize          │
              └──────────────┬────────────┘
                             │
              ┌──────────────▼────────────────────────────────────┐
              │ tools/                                            │
              │  registry（超时/截断/异常统一处理）                 │
              │  ├─ shell · files · http · memory                 │
              │  ├─ agent_cli：claude / codex / dsh（子进程 + 流式解析）│
              │  └─ mcp：stdio JSON-RPC 客户端                     │
              └───────────────────────────────────────────────────┘
                             │
              句子切分 ──► TTS ──► 扬声器（浏览器 Web Audio / sounddevice）
```

---

## 2. 分层与依赖方向

依赖**单向向下**，没有循环：

```
cli / server  →  session  →  agent  →  tools  →  providers  →  audio
                    ↓                                          ↑
                    └────────────── events ────────────────────┘
```

关键约束：

- **`audio/` 不认识任何 provider**，只定义一种 PCM 方言（mono / 16-bit LE）。
- **`providers/` 不认识 `tools/`**，三个 Protocol 各自 2–3 个方法。
- **`tools/` 不认识 `agent/`**，通过 `ToolContext` 拿到 workspace / 设置 / 回调。
- **`session` 是唯一的编排者**，只有它同时知道 ASR、TTS、agent、tools。

这让"换掉任意一层"成为局部改动：本仓库把 ASR 从 `MockASR` 换成 `faster-whisper` 再换成云端，业务代码零改动。

---

## 3. 三条主数据流

### 3.1 音频输入（浏览器）

```
AudioWorklet(128 帧) ──主线程重组──► 20ms 帧 ──能量 VAD──► 16kHz PCM16
       │                                                      │
       └─ 客户端持有 VAD：判定"开始说/说完了"                  │
                                                              ▼
              {audio.start} {audio.chunk}×N {audio.end}  ── WebSocket ──►
```

**为什么 VAD 放在客户端**：
- 少一次往返，端点判定延迟直接省掉
- 只在检测到语音时才上传，带宽降一个数量级
- 静音不进 ASR，直接省钱

**代价**：客户端和服务端必须对阈值达成一致。解决方式是 `ready` 事件把 `vad` 参数下发给客户端，只在一处调参。

### 3.2 一轮对话

```
audio.end
   │
   ├─► ASR ──────────────────────► transcript 事件
   │
   ├─► AgentLoop.run()  ──► assistant.delta ──► SentenceSplitter
   │        │                                        │
   │        │                                        ▼
   │        │                                  TTS 队列（预取下一句）
   │        │                                        │
   │        └─► tool.call / tool.progress ──┐        ▼
   │                     / tool.result      │   audio 事件 ×N
   │                                        │        │
   └────────────────────────────────────────┴────────┴──► audio.end
```

**句子级切分 + 预取**是延迟上的关键设计：播放第 N 句时，第 N+1 句已经在合成。切换点在 `session._tts_worker`。

### 3.3 工具执行

```
Model 请求 tool_calls
   │
   ├─ 危险工具？ ── 是 ──► CONFIRM_REQUEST 事件 + 朗读问题
   │                          │
   │                    等语音/按钮/终端回答（45s 超时即拒绝）
   │                          │
   └─ 否 ─────────────────────┴──► ToolRegistry.call()
                                     ├─ asyncio.timeout(spec.timeout_s)
                                     ├─ 异常 → ToolResult.failure（含 traceback 尾部）
                                     └─ 输出超长 → 截断
                                          │
                                    工具内部 ctx.progress() ──► 汇入同一条事件流
```

**独立且安全的多个工具并发执行**（"让 Claude 和 Codex 都看看"不应该等两倍时间）；**需要确认的串行执行**（确认必须一次一个，否则用户不知道在回答哪个）。

---

## 4. WebSocket 协议

单连接、JSON 文本帧、无子协议。所有服务端→客户端帧都带 `type` 与 `ts`。

### 客户端 → 服务端

| `type` | 其它字段 | 含义 |
|---|---|---|
| `audio.start` | — | 用户开始说话（PTT 按下 / VAD 起始） |
| `audio.chunk` | `pcm`(base64)、`sample_rate`、可选 `mime` | 一段 PCM。**不是一轮** |
| `audio.end` | — | 这段话说完了，开始识别 |
| `text` | `text` | 打字输入，或浏览器侧 ASR 的最终结果 |
| `barge_in` | — | 用户在播放中开口（传输层判定，服务端从不推断） |
| `interrupt` | — | 用户显式要求"停下" |
| `confirm` | `approved`(bool) | 对确认请求的回答 |
| `metric` | `name`、`value` | 客户端侧测量（用于真实 TTFA） |
| `ping` | — | 保活 |

### 服务端 → 客户端

| `type` | 关键字段 | 含义 |
|---|---|---|
| `ready` | `session_id`、`providers`、`tools`、`vad`、`warnings`、`agent` | 会话就绪，附带全部运行时信息 |
| `state` | `state` | `idle` / `listening` / `transcribing` / `thinking` / `acting` / `speaking` / `awaiting_confirmation` |
| `transcript` | `text`、`language`、`duration_s` | 用户这句话的最终转写 |
| `assistant.delta` | `text` | 助手输出的增量。**流式**，直接追加显示 |
| `assistant.done` | `text` | 文本生成完毕。**不代表音频播完** |
| `tool.call` | `id`、`name`、`arguments` | 模型请求了一个工具 |
| `tool.progress` | `message` | 长任务的进度播报 |
| `tool.result` | `id`、`ok`、`summary`、`declined?` | 工具结果 |
| `confirm.request` | `question` | 需要用户确认，前端应显示并可点按 |
| `audio` | `mime`、`sample_rate`、`seq`、`data`(base64) | 一段 TTS 音频 |
| `audio.end` | `seq`、`reason` | **本轮音频结束**（`complete` / `interrupted` / `too_short` / `empty_transcript`） |
| `error` | `message`、`where` | 某个环节失败 |
| `log` | `message`、`level` | 调试信息（`debug` 级别前端可忽略） |
| `metric` | `name`、`value` | 延迟埋点 |
| `pong` | `t` | ping 的回应 |

### 三条容易搞错的语义

1. **`assistant.done` ≠ 说完了。** 文本生成完时音频通常才播到一半。要知道"说完了"必须等 `audio.end`。
2. **`audio.end` 每轮恰好一次**，无论有没有产生音频。失败的一轮也要发，否则客户端会一直等。这是本项目实际修过的一个协议漏洞。
3. **`barge_in` 与 `interrupt` 不同。** 前者是"我在说话"，服务端在等待确认时只停音频、不取消；后者是"停下来"，一律取消。

---

## 5. 为什么是级联而不是端到端语音模型

| | 级联（本仓库） | 端到端 S2S（Moshi / Qwen-Omni / Realtime API） |
|---|---|---|
| 可替换性 | 每个模块独立可换 | 整体换 |
| 可观测性 | 每段延迟、每次转写可见 | 黑盒 |
| 工具调用 | 直接用文本 function calling | 需要模型原生支持，参数精度另说 |
| 本地可跑 | 可以（本项目零密钥可跑） | 通常需要 GPU/云 |
| 副语言 | **在文本瓶颈处全部丢失** | 保留（这是它唯一不可替代的优势） |
| 延迟上限 | 各段叠加 | 更低 |

**结论**：办公场景的核心诉求是"可靠地调用工具"，这恰好是级联的强项、端到端语音模型的弱点（学术调研指出语音 function calling 的瓶颈在**参数级信息损失**）。副语言能力在办公场景收益有限。

**演进方向**：`providers/base.py` 里 `LLMDelta` 已经能承载任意增量；接入 Realtime API 只需新增一个 provider，把它当作"ASR+LLM+TTS 合一"的实现即可，`session` 不用改。

---

## 6. 关键文件索引

| 文件 | 职责 | 值得注意的地方 |
|---|---|---|
| `audio/pcm.py` | 唯一的 PCM 方言 + 重采样 + WAV 编解码 | 只用 `numpy` + 标准库 `wave`，不需要 ffmpeg |
| `audio/vad.py` | 帧级 VAD + 端点判定 | 自适应噪声基底、预滚、最长话语上限 |
| `audio/player.py` | 终端播放（sounddevice 流式 / afplay 兜底） | 可取消，这是终端能打断的前提 |
| `text.py` | 句子切分 + 面向语音的文本清洗 | ASCII 句点只在后接空白时才算句末（避免切开 `a.py`） |
| `providers/registry.py` | `auto` 解析 + 能力探测 | 本地优先；`doctor` 与前端设置面板都基于它 |
| `agent/loop.py` | 流式 tool-calling 循环 | 队列化事件流；历史 `_sanitize()` 修复被打断的 tool_calls |
| `session.py` | 状态机、打断、确认、TTS 预取 | 项目里最值得读的一个文件 |
| `tools/exec.py` | 子进程执行 | 超时 + 取消 + 流式行回调 + 输出上限 |
| `tools/agent_cli.py` | claude / codex / dsh 适配 | 每个 CLI 一个流解析器，解析失败降级为纯文本 |
| `tools/mcp.py` | stdio JSON-RPC MCP 客户端 | 死掉的 server 会让 pending 请求失败而不是挂住 |
| `server/static/app.js` | 浏览器客户端 | 本地优先停止音频；防自打断的三层防御 |

---

## 7. 已知限制

诚实清单，以及为什么暂时这样：

| 限制 | 影响 | 为什么 | 怎么补 |
|---|---|---|---|
| **ASR 是批式的** | 本地实测 ASR 占端到端延迟的 1/3 | 流式 ASR 需要 provider 接口扩展为"喂帧 + 取 interim" | 接口已按可替换设计；新增流式 provider 即可，`session` 改动很小 |
| **终端路径无 AEC** | 终端默认半双工，说话时它不听 | PortAudio 侧没有现成的 AEC | 用 Ctrl-C 打断；或用浏览器路径（浏览器有 AEC） |
| **无唤醒词** | 必须按 PTT 或开免提 | 唤醒词（Porcupine/openWakeWord）是额外依赖与误唤醒调参 | `audio/` 已隔离，加一层前置检测即可 |
| **记忆是关键词检索** | 笔记多了以后召回会退化 | 避免引入向量库依赖 | `tools/memory.py` 的 `recall` 可换成 embedding |
| **无持久化会话** | 进程重启后历史丢失 | 办公场景单次会话足够 | `AgentLoop.history` 可直接序列化 |
| **并发是单会话级别** | 一个 WebSocket 一个 session | 刻意为之：共享一个麦克风和扬声器 | 多用户需按 session 隔离 workspace |
| **无鉴权体系** | 只有可选的静态 token | 默认只监听 127.0.0.1 | 对外暴露前必须加 `server.auth_token` 或反代鉴权 |

---

## 8. 扩展点

按"改动量从小到大"排列：

1. **加一个 provider**：在 `providers/{asr,tts,llm}/` 下实现对应 Protocol，在 `registry.py` 注册。不动其他任何文件。
2. **加一个工具**：写一个 `ToolSpec`（名字 + JSON Schema + async handler），加进 `tools/builtin.py` 的列表。确认、超时、截断自动生效。
3. **加一个 MCP server**：改配置即可，工具自动出现为 `mcp__<server>__<tool>`。
4. **换前端**：实现 WebSocket 协议表即可，`server/static/app.js` 只是一个参考客户端。
5. **加一个 transport**：写一个 `on_event` 回调 + 调用 `session` 的公开方法。`cli.py` 就是 ~80 行的例子。
6. **接入端到端语音模型**：新增一个"三位一体"的 provider，`session` 无需改动。
