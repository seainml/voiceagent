# 配置参考

配置优先级（后者覆盖前者）：

```
内置默认  <  voiceagent.toml  <  .env  <  环境变量 (VA_*)  <  CLI 参数
```

- 生成模板：`voiceagent init`（或直接 `cp voiceagent.example.toml voiceagent.toml`）
- 查看当前生效值：`voiceagent config`
- 查看本机可用能力：`voiceagent doctor`
- **密钥永不写进 TOML**，只写"存放密钥的环境变量名"（`api_key_env`）

环境变量用双下划线表示嵌套：

```bash
export VA_ASR__PROVIDER=openai
export VA_LLM__MODEL=deepseek-chat
export VA_TTS__VOICE=Tingting
export VA_PIPELINE__VAD_SILENCE_MS=800
```

---

## 1. 快速选择

| 我想要 | 配置 |
|---|---|
| 零密钥、离线、立刻能用 | 默认即可（macOS `say` + 浏览器识别） |
| 离线语音识别 | `uv pip install -e '.[local-asr]'` → `asr.provider` 保持 `auto` |
| 更好的中文音色（免费） | `uv pip install -e '.[edge]'` → `tts.provider = "edge"` |
| 真实的大模型 | `export DEEPSEEK_API_KEY=...`（或任意 OpenAI 兼容端点） |
| 完全本地大模型 | `export VA_LLM__BASE_URL=http://127.0.0.1:11434/v1`（Ollama） |
| 云端 ASR | `asr.provider="openai"` + `OPENAI_API_KEY` |
| 终端里说话 | `uv pip install -e '.[local-audio]'` → `voiceagent talk` |

---

## 2. `[asr]` — 语音识别

| 键 | 默认 | 说明 |
|---|---|---|
| `provider` | `auto` | `auto` / `faster_whisper` / `openai` / `browser` / `mock` |
| `model` | `base` | 本地：`tiny`/`base`/`small`/`medium`/`large-v3`；云端：模型 id |
| `language` | `zh` | 设为 `null` 自动检测 |
| `hotwords` | `[]` | **性价比最高的一个旋钮**，见下 |
| `base_url` | 见配置 | 云端专用，任何 OpenAI 兼容端点 |
| `api_key_env` | `OPENAI_API_KEY` | 存放密钥的环境变量名 |
| `device` | `cpu` | `cpu` / `cuda` |
| `compute_type` | `int8` | `int8` / `int8_float16` / `float16` |
| `beam_size` | `1` | 越大越准越慢；实时场景保持 1 |
| `vad_filter` | `true` | 用 whisper 内置 VAD 再过滤一次 |
| `download_root` | `.voiceagent/models` | 本地模型缓存目录 |

`auto` 的解析顺序：**faster_whisper（已安装）→ openai（有 key）→ browser**。

### 关于 `hotwords`

实测把「明天上午十点开会」识别成「明天上五十點開會」——"午"→"五"。这类错误（数字、金额、人名、代号）换模型也修不掉，但热词能修掉大部分：

```toml
[asr]
hotwords = ["voiceagent", "张三", "李四", "季度复盘", "Kubernetes", "DSH"]
```

把**同事姓名、项目代号、你常用的 CLI 名称**放进去。Whisper 系通过 `prompt` 实现，效果比换更大的模型更明显。

### `provider = "browser"`

识别发生在浏览器（Web Speech API），服务端只收文本。零安装、零成本，中文效果可用，但**只在 Chrome 系可用**，且会把音频送到浏览器厂商的云。前端会在 `ready` 事件里看到 `asr: browser` 并自动切换。

---

## 3. `[tts]` — 语音合成

| 键 | 默认 | 说明 |
|---|---|---|
| `provider` | `auto` | `auto` / `macos_say` / `edge` / `openai` / `mock` |
| `voice` | `null` | `null` = 按语言/文本自动挑 |
| `rate` | `195` | 语速（词/分），仅 `macos_say` |
| `speed` | `1.0` | 倍速，云端 provider 用 |
| `model` | `tts-1` | 仅 `openai` |
| `response_format` | `mp3` | `pcm` 可省掉容器解码，但兼容性略差 |
| `api_key_env` | `OPENAI_API_KEY` | 存放密钥的环境变量名 |

### 音色自动选择

`voice = null` 时按以下顺序决定：

1. 文本含中日韩字符 → 中文音色（`Tingting` / `zh-CN-XiaoxiaoNeural`）
2. 否则用 `asr.language` 对应的音色
3. 都没有 → 交给系统默认

> **⚠️ 一个真实的坑**：macOS `say` 在**未指定 voice** 时，若系统默认音色没有中文数据，会对中文文本**静默输出 0 帧的 WAV**——一切看起来正常，但没有任何声音。本仓库因此在两处做了防御：文本级的中文检测（步骤 1），以及解码后的空音频校验（会抛出带修复建议的明确错误）。
>
> 所以：**中文场景请显式设置 `voice`**。

常用音色：

| provider | 中文 | 英文 |
|---|---|---|
| `macos_say` | `Tingting`(zh-CN) / `Meijia`(zh-TW) / `Sinji`(zh-HK) | `Samantha` / `Daniel` |
| `edge` | `zh-CN-XiaoxiaoNeural` / `zh-CN-YunxiNeural` | `en-US-AriaNeural` |

列出本机可用音色：`say -v '?'`；edge 走 `tts.voices()`。

---

## 4. `[llm]` — 推理模型

任何 OpenAI 兼容的 `/chat/completions` 都能用。

| 键 | 默认 | 说明 |
|---|---|---|
| `provider` | `auto` | `auto` / `openai` / `ollama` / `echo` / `mock` |
| `model` | `deepseek-chat` | |
| `base_url` | `https://api.deepseek.com/v1` | |
| `api_key_env` | `DEEPSEEK_API_KEY` | |
| `temperature` | `0.5` | |
| `max_tokens` | `1200` | |
| `timeout_s` | `120` | 流式连接可能长时间保持 |
| `extra_headers` | `{}` | 需要自定义头网关时 |
| `extra_body` | `{}` | 需要额外字段时（如 `top_p`） |

### 常见端点

```toml
# OpenAI
base_url = "https://api.openai.com/v1"
model    = "gpt-4o-mini"
api_key_env = "OPENAI_API_KEY"

# 月之暗面 / 智谱 / 硅基流动 / Groq / Together … 同样都是这个形状
base_url = "https://api.moonshot.cn/v1"
model    = "moonshot-v1-8k"
api_key_env = "MOONSHOT_API_KEY"

# 本地（Ollama / vLLM / LM Studio）——不需要真实密钥
base_url = "http://127.0.0.1:11434/v1"
model    = "qwen2.5:7b"
```

**本地端点识别**：`base_url` 的 host 是 `127.0.0.1` / `localhost` / `::1` 时，缺密钥会用占位符，不会报错。远程端点缺密钥会**明确报错**，而不是发一个必然 401 的请求。

### `provider = "echo"`

没有可用模型时的兜底。它会**说出**听到了什么，并提示如何配置密钥。作用是：零密钥也能把整条语音链路（麦克风、VAD、打断、UI、确认）验证完，只有"思考"这一环缺失，且它明说。

---

## 5. `[agent]` — 人设

| 键 | 默认 | 说明 |
|---|---|---|
| `name` | `Aria` | 出现在 UI 与提示词里 |
| `persona` | 见配置 | 一句话人设 |
| `system_prompt` | `null` | 完全替换内置提示词 |
| `system_prompt_file` | `null` | 从文件读（适合版本管理） |
| `max_tool_iterations` | `6` | 单轮最多几次工具往返 |

> 自定义 `system_prompt` 会**替换掉**内置的语音规则（禁止 Markdown、1–3 句、不朗读代码…）。环境信息与工具列表仍会自动附加。如果你只想改人设，用 `persona`。

---

## 6. `[pipeline]` — 轮次与流式

| 键 | 默认 | 说明 |
|---|---|---|
| `barge_in` | `true` | 允许打断 |
| `sentence_flush_chars` | `28` | 累积到多少字且遇到自然停顿就提前合成 |
| `max_sentence_chars` | `180` | 单句上限，超过则在标点处硬切 |
| `max_spoken_chars` | `700` | **单轮朗读量硬上限**，超出部分只显示在屏幕上 |
| `history_turns` | `12` | 提示词里保留多少轮 |
| `vad_silence_ms` | `600` | 尾随静音判定 |
| `vad_threshold` | `0.015` | RMS 能量阈值 |
| `asr_min_utterance_ms` | `250` | 短于此时长的话丢弃 |
| `max_utterance_s` | `60` | 硬上限，防止麦克风卡住 |

### 怎么调 `vad_silence_ms`

这是**最值得按人调**的参数，因为存在真实的取舍：调小响应快，但会在用户思考停顿时抢话（Full-Duplex-Bench 指出这是现有模型的系统性缺陷）；调大不抢话，但显得慢。

| 场景 | 建议 |
|---|---|
| 说话慢、常停顿思考 | 800–1000 |
| 指令式快节奏 | 400–500 |
| 嘈杂环境 | 保持 600，把 `vad_threshold` 提到 0.02–0.03 |
| 用免提 + 外放 | 保持 600 并在 UI 上关掉「打断」 |

改这里会**同时**影响服务端（`ready` 事件下发给浏览器）和终端路径，只改一处。

---

## 7. `[tools]` — 能力与安全

| 键 | 默认 | 说明 |
|---|---|---|
| `workspace` | `.` | 工具可访问的根目录，路径逃逸会被拒绝 |
| `enabled` | 见配置 | 启用的工具名单；`"mcp"` 是通配符 |
| `confirm` | 见配置 | **必须先确认才执行**的工具 |
| `max_output_chars` | `4000` | 工具输出截断上限 |
| `shell_timeout_s` | `120` | |
| `shell_denylist` | 见配置 | 直接拒绝的命令片段 |
| `http_allow_hosts` | `[]` | 空 = 允许任意 host |
| `memory_file` | `.voiceagent/memory.jsonl` | |
| `allow_unsafe_agent_writes` | `false` | 是否让 agent CLI 绕过自身权限 |

### 三层安全边界

```
1. 拒绝（shell_denylist / workspace 逃逸检查 / http_allow_hosts）
2. 确认（confirm 列表 + 名称启发式）        ← 真正的控制在这一层
3. 超时与截断（registry 统一施加）
```

> **黑名单是安全带，不是沙箱。** `run_shell` 能被绕过的写法无穷无尽；真正的门是"说出来再等确认"。
>
> 同理，**MCP 的 `destructiveHint` 只是提示，不是强制**。本仓库对 MCP 工具同时使用注解、名称启发式（`write`/`delete`/`exec`…）和配置的确认列表。

`allow_unsafe_agent_writes = true` 会给 agent CLI 加上 `--dangerously-skip-permissions` 之类的参数。**默认关闭是刻意的**——它和字面意思一样危险。

---

## 8. `[tools.agents.*]` — 三个 coding agent

每个都是一次独立的子进程调用。

```toml
[tools.agents.dsh]
enabled = true
command = "dsh"
args = ["--profile", "headless"]
preamble = "..."          # 前置提示，见下
timeout_s = 900
env = {}                  # 额外环境变量
working_dir = null        # 默认用 tools.workspace
unsafe_args = []          # 仅当 allow_unsafe_agent_writes = true 时追加
```

三个已验证的非交互调用方式：

| 工具名 | `command` + `args` | 输出格式 |
|---|---|---|
| `ask_claude_code` | `claude --print --output-format stream-json --verbose` | JSONL（解析 `assistant` 与 `result` 事件） |
| `ask_codex` | `codex exec --color never --skip-git-repo-check --json` | JSONL（另用 `-o` 取最终消息作保险） |
| `ask_dsh` | `dsh --profile headless` | 纯文本 |

### `preamble` 为什么必须有

这些 CLI 默认返回 Markdown。**原样读出来是灾难。** 默认前置提示要求它们最后用不超过三句口语总结。想改成别的风格就改这个字段。

### 认证状态

三个 CLI 都需要各自登录，语音层无法代劳：

| CLI | 未登录时的表现 | 修复 |
|---|---|---|
| Claude Code | `--print` 正常退出但 `result` 事件带 `is_error:true`、内容为 `Not logged in · Please run /login` | 终端跑 `claude` 后 `/login`，或设 `ANTHROPIC_API_KEY` |
| Codex | 401，且会**重试 5 次**（约 30 s）后才失败 | `codex login`，或设有效的 `OPENAI_API_KEY` |
| DSH | 取决于其模型配置 | 检查 dsh 配置 |

本仓库会识别这些错误并**用中文说清楚该怎么修**，而不是丢一段原始报错。

---

## 9. `[tools.mcp.*]` — MCP 服务器

任何有 MCP server 的东西都能变成语音工具，无需写 Python。

```toml
[tools.mcp.filesystem]
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "/Users/me/notes"]
timeout_s = 30          # 启动 + initialize 超时
tool_timeout_s = 120    # 单次 tools/call 超时

[tools.mcp.github]
command = "npx"
args = ["-y", "@modelcontextprotocol/server-github"]
env = { GITHUB_PERSONAL_ACCESS_TOKEN = "..." }   # 更推荐用环境变量
```

- 工具名形如 `mcp__<server>__<tool>`
- 某个 server 起不来**不会阻止助手启动**，只在 `ready.warnings` 与 `voiceagent tools` 里报告
- 只有 `tools/list` 与 `tools/call` 被实现；sampling / roots 有意不做

---

## 10. `[server]`

| 键 | 默认 | 说明 |
|---|---|---|
| `host` | `127.0.0.1` | |
| `port` | `8765` | |
| `auth_token` | `null` | 设置后客户端需带 `?token=` |
| `cors_origins` | `["*"]` | |

> **对外暴露前必须做两件事**：设置 `auth_token`，或者放在带鉴权的反向代理后面。启动时如果绑到非 localhost 且没有 token，会打印警告。
>
> 注意：`/api/health` 和 `/api/capabilities` 不做 token 校验（便于探活），因此它们会暴露已启用的 provider 与工具名称。

---

## 11. 环境变量速查

```bash
# 模型
DEEPSEEK_API_KEY / OPENAI_API_KEY / VA_LLM__API_KEY
VA_LLM__BASE_URL / VA_LLM__MODEL

# 语音
VA_TTS__PROVIDER / VA_TTS__VOICE / VA_ASR__PROVIDER / VA_ASR__MODEL
VA_ASR__HOTWORDS=["张三","voiceagent"]

# 行为
VA_PIPELINE__VAD_SILENCE_MS=800
VA_TOOLS__ALLOW_UNSAFE_AGENT_WRITES=true
VA_SERVER__AUTH_TOKEN=change-me

# 配置文件位置
VA_CONFIG=voiceagent.local.toml
```

`.env` 会被自动读取（`python-dotenv`）。参见 [`.env.example`](../.env.example)。

---

## 12. 排障

| 现象 | 先看这里 |
|---|---|
| 有对话文字但没声音 | `voiceagent say "测试"` ；中文务必显式设 `tts.voice` |
| 第一句话等很久 | `voiceagent doctor` 看 ASR 是否本地模型（冷启动约 20 s，之后会自动预热） |
| 识别结果乱 | `asr.hotwords` 是否覆盖了专有名词；`asr.model` 是否太小 |
| 助手总打断自己 | UI 关掉「打断」，或调高 `vad_threshold`；外放时优先用耳机 |
| 助手总抢话 | 调大 `vad_silence_ms` 到 800–1000 |
| 助手说太久 | 调小 `max_spoken_chars` |
| agent CLI 工具失败 | `voiceagent doctor` 看二进制是否存在；再确认该 CLI 已登录 |
| MCP 工具没出现 | `voiceagent tools` 看 warnings；确认 `command` 在 PATH 上 |
| 想确认当前到底用了什么 | `voiceagent config` + `/api/capabilities` |
