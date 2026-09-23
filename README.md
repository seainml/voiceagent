<div align="center">

# voiceagent

**能听的办公助手：说话进去，事情出来。**

可插拔语音链路 · 可插拔工具 · 本地优先、零密钥可跑

[![tests](https://img.shields.io/badge/tests-214%20passing-brightgreen)](#开发)
[![smoke](https://img.shields.io/badge/e2e%20smoke-14%2F14-brightgreen)](#开发)
[![python](https://img.shields.io/badge/python-3.11%2B-blue)](#快速开始)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

</div>

---

## 这是什么

一个语音优先的办公助手。你在浏览器或终端里说话，它回答，并且**能真的做事**——
执行命令、调用 HTTP 接口、读写文件、记忆笔记，以及把整块编码任务交给
**Claude Code**、**Codex** 或 **DSH** 去跑。

语音链路的每一环（ASR / TTS / LLM）和每一项能力都是可替换的 provider，
所以你可以从"零密钥、零安装"开始，再逐块升级到生产配置。

```bash
uv pip install -e ".[dev]"
voiceagent doctor        # 看看这台机器上什么可用
voiceagent serve         # 打开 http://127.0.0.1:8765
```

---

## 已验证的状态

不是"应该能跑"，是**实测跑通了**。所有数字来自本机（Intel Mac，CPU int8）：

| 验证项 | 结果 |
|---|---|
| 单元测试 | **214 个通过**，4.4 s |
| 真实端到端冒烟 | **14/14 通过**（`scripts/smoke.py`） |
| 完整语音闭环 | 合成 3.04 s 语音 → Whisper 识别 → LLM → **721 KB TTS 音频回传**，端到端 **4.0 s** |
| 真实 `dsh` 子进程调用 | ✓ 7.6 s 返回 |
| 语音确认门 | ✓ 说"取消"不执行、说"确认"才执行 |
| 真实 WebSocket 往返 | ✓ 对运行中的 uvicorn 实测通过 |

**实测延迟分解**（3.0 s 音频）：

```
ASR            1261 ms   ████████████████████░░░░░░░░░░  本地瓶颈在这
LLM 首 token      2 ms   ░░░░░░░░░░░░░░░░░░░░░░░░░░░░  （echo 模型）
TTS 首包       1768 ms   ████████████████████████████  含上面两段
端到端         4000 ms
```

> ⚠️ 识别并不是完美的。实测「明天**上午**十点开会」被识别成「明天**上五**十點開會」。
> 这正是研究里反复指出的**数字/专有名词 ASR 错误**，也是本项目内置热词偏置与
> 复述确认的原因。详见[最佳实践](docs/best-practices.md#5-asr-错误不可消除只能设计成可恢复)。

---

## 快速开始

需要 **Python 3.11+**。默认路径**不需要 ffmpeg、PortAudio 或任何 API key**。

```bash
git clone https://github.com/seainml/voiceagent.git
cd voiceagent

uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

### 三个入口

```bash
voiceagent chat            # 打字对话，工具全部可用
voiceagent talk            # 麦克风进、扬声器出（需 .[local-audio]）
voiceagent serve           # 浏览器 UI：按住说话 + 免提连续对话
```

### 可选增强

```bash
uv pip install -e ".[local-asr]"    # 离线 Whisper 识别（首次会下载模型）
uv pip install -e ".[local-audio]"  # 终端麦克风与流式播放
uv pip install -e ".[edge]"         # 免费神经 TTS，中文音色明显更好
```

### 接上真正的大模型

```bash
cp .env.example .env
echo 'DEEPSEEK_API_KEY=sk-...' >> .env      # 或任意 OpenAI 兼容端点
```

没配模型也能完整验证语音链路——内置的 `echo` provider 会**说出**它听到了什么，
并告诉你还缺什么。只有"思考"这一环不在。

---

## 运行与运维

### 前台运行（调试用）

```bash
voiceagent serve                    # Ctrl-C 停止
voiceagent serve --port 9000        # 换端口
voiceagent serve --reload           # 改代码自动重启
```

`Ctrl-C` 会优雅关闭：取消进行中的回合、关掉 MCP 子进程、断开 WebSocket。

### 后台运行（日常用）

用附带的脚本，它会 detached 启动、记录 PID、写日志：

```bash
./scripts/service.sh start          # 后台启动，默认 http://127.0.0.1:8765
./scripts/service.sh status         # 进程、启动时长、健康检查
./scripts/service.sh logs           # 最近 60 行日志
./scripts/service.sh logs -f        # 实时跟随
./scripts/service.sh restart        # 重启（改了配置或代码后用）
./scripts/service.sh stop           # 停止
```

其它用法：

```bash
./scripts/service.sh start --port 9000        # 换端口
./scripts/service.sh start --reload           # 开发模式
LINES=300 ./scripts/service.sh logs           # 看更多日志
./scripts/service.sh doctor                   # 本机能力体检
```

脚本产生的文件都在 `.voiceagent/`（已 gitignore）：

```
.voiceagent/run/voiceagent.pid      # PID 文件
.voiceagent/logs/voiceagent.log     # 日志（会累积，可随时删除）
.voiceagent/models/                 # 本地 ASR 模型缓存
```

`status` 在未运行时返回退出码 1，方便写进脚本判断。它也能识别**不是它启动的**服务
（比如你在另一个终端里 `voiceagent serve` 的），并通过端口找到对应进程。

### 更新

```bash
./scripts/service.sh update          # 拉代码 + 按需重装依赖 + 重启
./scripts/service.sh update --deps   # 强制重装依赖
```

`update` 会：`git fetch` → 显示将要应用的提交 → 检查工作区是否干净 →
`git merge --ff-only` → 依赖有变化时重装 → 只在真的有变化时重启。
**没有变化就什么都不做**，不会白白重启。

手动等价操作：

```bash
git pull
uv pip install -e ".[dev]"           # 依赖有变化时
./scripts/service.sh restart
```

想更新本地 ASR 模型（例如换更大的模型）：

```bash
# 1. 改配置
#    [asr] model = "small"      （或 medium / large-v3）
# 2. 删掉旧缓存并重启，会重新下载
rm -rf .voiceagent/models
./scripts/service.sh restart
```

### 暂停而不是停止

如果只是想让它**暂时不响应**，不需要停进程：

- 浏览器里关掉「免提」并松开按钮即可——它只在被唤起时才会听
- 想彻底静默：`./scripts/service.sh stop`，之后再 `start`（启动约 2–3 秒，
  本地 ASR 模型会重新预热）

### 开机自启（macOS）

`launchd` 是 macOS 上正确的做法（`nohup` 撑不过重启）。存成
`~/Library/LaunchAgents/com.local.voiceagent.plist`：

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.local.voiceagent</string>
  <key>ProgramArguments</key>
  <array>
    <string>/绝对路径/voiceagent/.venv/bin/voiceagent</string>
    <string>serve</string>
  </array>
  <key>WorkingDirectory</key><string>/绝对路径/voiceagent</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>/绝对路径/voiceagent/.voiceagent/logs/launchd.log</string>
  <key>StandardErrorPath</key><string>/绝对路径/voiceagent/.voiceagent/logs/launchd.err</string>
</dict>
</plist>
```

```bash
launchctl load  ~/Library/LaunchAgents/com.local.voiceagent.plist   # 启用
launchctl unload ~/Library/LaunchAgents/com.local.voiceagent.plist  # 停用
```

> 用 launchd 时先把 `./scripts/service.sh stop`，避免两个进程抢同一个端口。

### 常见运维问题

| 现象 | 处理 |
|---|---|
| `start` 报端口被占用 | 脚本会打印占用进程的 PID；`kill <pid>` 或换个端口 |
| `status` 说 running 但打不开页面 | `./scripts/service.sh logs` 看最后几行，通常是模型预热或配置错误 |
| 改了 `voiceagent.toml` 没生效 | 配置在启动时读取一次 → `./scripts/service.sh restart` |
| 改了 `.env` 没生效 | 同上，`restart` |
| 日志文件越来越大 | `./scripts/service.sh stop && rm .voiceagent/logs/*.log && ./scripts/service.sh start` |
| 忘了改过什么配置 | `voiceagent config`（不含密钥）看生效值 |
| 想彻底重来 | `rm -rf .voiceagent .venv && uv venv --python 3.12 .venv && uv pip install -e ".[dev]"` |

---

## 架构

```
 麦克风 ─► VAD/端点判定 ─► ASR ─┐
  (浏览器 / 终端)                │
                    ┌───────────▼──────────────────────┐
                    │ session.py  回合状态机 + 打断      │
                    └───────────┬──────────────────────┘
        ┌───────────────────────▼───────────────────────┐
        │ agent/loop.py  流式 tool-calling               │
        └──────┬──────────────────────────────┬─────────┘
               │                              │
     ┌─────────▼──────────┐        ┌──────────▼─────────┐
     │ tools/             │        │ providers/         │
     │  shell · http      │        │  asr · tts · llm   │
     │  files · memory    │        │  三个小 Protocol    │
     │  claude/codex/dsh  │        └────────────────────┘
     │  MCP servers       │
     └────────────────────┘
               │
     句子切分 ─► TTS ─► 扬声器
```

**三个关键设计**（理由见[架构文档](docs/architecture.md)）：

1. **句子级流式 + 预取**：LLM 还在生成第 3 句时，第 1 句已经在播、第 2 句已在合成。
2. **打断是传输层的事实**：由检测到的一方显式声明，服务端**从不**从"有音频到达"推断。
3. **一切走同一条有序事件流**：助手增量、工具调用、工具进度、音频、指标，前端只需理解一套词汇。

| 目录 | 内容 |
|---|---|
| `audio/` | 唯一的 PCM 方言、重采样、VAD/端点判定、播放 |
| `providers/` | ASR / TTS / LLM 适配器，`auto` 本地优先解析 |
| `tools/` | 能力层：子进程、文件、HTTP、记忆、agent CLI、MCP |
| `agent/` | 推理循环 + 面向语音的系统提示词 |
| `session.py` | 编排：回合、打断、确认握手、指标 |
| `server/` | FastAPI + WebSocket + 浏览器 UI |

---

## 工具

| 工具 | 需确认 | 用途 |
|---|---|---|
| `ask_claude_code` | ✓ | 交给 Claude Code（`claude --print --output-format stream-json`） |
| `ask_codex` | ✓ | 交给 Codex CLI（`codex exec --json --skip-git-repo-check`） |
| `ask_dsh` | ✓ | 交给 DSH（`dsh --profile headless`） |
| `run_shell` | ✓ | 在工作区内执行命令 |
| `write_file` | ✓ | 创建 / 覆盖 / 追加文件 |
| `read_file` · `list_files` | | 查看工作区 |
| `http_request` | | 调用任意 HTTP 接口 / webhook |
| `remember` · `recall` | | 跨会话的持久笔记 |
| `mcp__<server>__<tool>` | 按名字判定 | 任何 MCP server 的工具 |

标记"需确认"的工具**不会被静默执行**：助手会用一句短话说清楚要做什么，然后**停下来等你**——
你可以直接说"确认"或"取消"，也可以点按钮或在终端输入。

```toml
# 任何 MCP server 都能变成语音工具，不用写一行 Python
[tools.mcp.filesystem]
command = "npx"
args = ["-y", "@modelcontextprotocol/server-filesystem", "/Users/me/notes"]
```

---

## 配置

优先级：`默认 < voiceagent.toml < .env < 环境变量(VA_*) < CLI 参数`

```bash
voiceagent init          # 生成配置模板
voiceagent doctor        # 本机可用能力与已解析的 provider
voiceagent config        # 当前生效配置（不含密钥）
```

```bash
export VA_ASR__HOTWORDS='["张三","voiceagent","DSH"]'   # 最划算的一个旋钮
export VA_TTS__VOICE=Tingting                            # 中文请显式指定
export VA_PIPELINE__VAD_SILENCE_MS=800                   # 说话慢的人
export VA_LLM__BASE_URL=http://127.0.0.1:11434/v1        # 本地 Ollama
```

完整参考见[配置文档](docs/configuration.md)。

---

## 调研

构建之前先做了三份调研，它们直接决定了上面的设计：

| 文档 | 内容 |
|---|---|
| [学术前沿](docs/research/academic-landscape.md) | 2023–2025 S2S 研究版图、20+ 系统对比、17 个评测基准、开放问题排序、70 条带链接文献 |
| [框架生态](docs/research/framework-landscape.md) | Pipecat/LiveKit/Vocode 等编排框架、OpenAI/Gemini/Qwen 实时 API、ASR/TTS 厂商实测价格与许可陷阱、三档选型推荐 |
| [工程难点](docs/research/engineering-pitfalls.md) | 分环节延迟预算、11 个工程难点、25 条反模式、可观测性指标、成本控制、350 条来源 |

三份报告共同的结论：**这个领域的重心已经从"让模型会说话"转移到"让模型知道什么时候该说话、
用什么语气说话"**——而后者连"怎么衡量"都还没解决。

本项目据此选择了**级联架构**（办公场景真正需要的是"可靠地调用工具"，这是级联的强项、
端到端语音模型的弱点），并把工程预算压在打断、端点判定和工具安全上。

---

## 开发

```bash
.venv/bin/ruff check src/ tests/ scripts/     # lint
.venv/bin/python -m pytest -q                 # 214 个单元测试，约 4 s
.venv/bin/python scripts/smoke.py             # 14 项真实端到端检查
```

`scripts/smoke.py` 用**真实 provider**跑完整闭环：真实 TTS 合成一段话 → 把它当作用户语音回灌 →
真实 ASR → 真实 LLM → 真实 TTS。还包括真实 `dsh` 子进程调用、真实确认门、真实 WebSocket 往返。

> 经验：**mock 测试证明逻辑对，真实闭环测试证明能用。**
> 这个脚本抓到了两个单元测试永远抓不到的 bug——TTS 对中文静默输出空音频、
> 以及每轮对话都在自我打断。

---

## 已知限制

诚实清单（理由与补法见[架构文档](docs/architecture.md#7-已知限制)）：

- **ASR 是批式的**，占了实测端到端延迟的 1/3。流式 ASR 是下一个大杠杆；provider 接口已按可替换设计。
- **终端路径没有 AEC**，因此默认半双工（说话时它不听）。浏览器路径有 AEC，是全双工的。
- **无唤醒词**，需要按住说话或开启免提。
- **记忆是关键词检索**，笔记很多以后召回会退化。
- **单会话**：一个连接一个 session（因为共享一个麦克风和扬声器）。
- **对外暴露前必须加鉴权**：默认只监听 `127.0.0.1`，另有可选的静态 token。

---

## 许可

MIT — 见 [LICENSE](LICENSE)。
