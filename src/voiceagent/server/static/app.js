/* voiceagent browser client.
 *
 * Responsibilities, in order of importance:
 *   1. Capture 16 kHz mono PCM and detect turn boundaries (client-side VAD).
 *   2. Stop audio *locally and instantly* on barge-in — never wait for the
 *      server round-trip, or interrupting feels laggy.
 *   3. Schedule TTS chunks gaplessly with the Web Audio API.
 *   4. Degrade gracefully: if the server has no ASR, use the browser's.
 */
(() => {
  "use strict";

  const $ = (id) => document.getElementById(id);

  const FRAME_MS = 20;
  // Defaults; overwritten by the `ready` event so the browser and the server
  // agree on turn-taking without editing two files.
  const DEFAULT_MIN_SPEECH_MS = 250;
  const DEFAULT_SILENCE_MS = 600;
  const DEFAULT_THRESHOLD = 0.015;
  const PREROLL_FRAMES = 15;      // 300 ms so the first phoneme is not clipped
  const BARGE_PREROLL_FRAMES = 5; // less, because it may still contain our own tail
  const BARGE_FRAMES = 4;         // ~80 ms of sustained speech before we cut in
  //: How much louder than normal speech must be while we are playing audio.
  //: Browser echo cancellation is not enough on laptop speakers, so without
  //: this gate the assistant hears itself and interrupts itself, forever.
  const BARGE_GAIN = 3.0;
  const SEND_FRAMES = 2;          // batch 40 ms per WebSocket message

  const state = {
    ws: null,
    connected: false,
    reconnectDelay: 500,
    ready: false,

    mode: "ptt",              // 'ptt' | 'handsfree'
    browserAsr: false,        // recognition in the browser instead of the server
    autoBargeIn: true,        // cut the assistant off when the user speaks
    recording: false,
    awaitingConfirm: false,

    // turn-taking thresholds, synced from the server's `ready` event
    minSpeechMs: DEFAULT_MIN_SPEECH_MS,
    silenceMsLimit: DEFAULT_SILENCE_MS,
    baseThreshold: DEFAULT_THRESHOLD,
    bargeFrames: 0,

    captureCtx: null,
    playCtx: null,
    worklet: null,
    source: null,
    stream: null,
    sampleRate: 16000,

    // capture pipeline
    leftover: null,
    sendBuf: [],
    preroll: [],
    speechStarted: false,
    speechMs: 0,
    silenceMs: 0,
    noiseFloor: 0.003,

    // playback pipeline
    nextTime: 0,
    activeSources: [],
    //: When the user's turn was submitted, so we can report the only latency
    //: number that actually matters: submission -> sound coming out.
    turnSentAt: 0,
    ttfaReported: false,

    // rendering
    currentAssistant: null,
    currentAssistantText: "",
    pendingTools: new Map(),
    metrics: {},
    capabilities: null,

    recognition: null,
    recognitionActive: false,
  };

  // ---------------------------------------------------------------- utils

  const b64FromBytes = (bytes) => {
    let bin = "";
    const CHUNK = 0x8000;
    for (let i = 0; i < bytes.length; i += CHUNK) {
      bin += String.fromCharCode.apply(null, bytes.subarray(i, i + CHUNK));
    }
    return btoa(bin);
  };

  const bytesFromB64 = (b64) => {
    const bin = atob(b64);
    const out = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  };

  const floatToInt16 = (frame) => {
    const out = new Int16Array(frame.length);
    for (let i = 0; i < frame.length; i++) {
      const s = Math.max(-1, Math.min(1, frame[i]));
      out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    return out;
  };

  const rms = (frame) => {
    let sum = 0;
    for (let i = 0; i < frame.length; i++) sum += frame[i] * frame[i];
    return Math.sqrt(sum / frame.length);
  };

  const send = (payload) => {
    if (state.ws && state.ws.readyState === WebSocket.OPEN) {
      state.ws.send(JSON.stringify(payload));
    }
  };

  const scroll = () => {
    const el = $("transcript");
    el.scrollTop = el.scrollHeight;
  };

  // ------------------------------------------------------------ transcript

  function addTurn(role, text) {
    const wrap = document.createElement("div");
    wrap.className = `turn ${role}`;
    const who = document.createElement("div");
    who.className = "who";
    who.textContent = { user: "你", assistant: agentName(), system: "系统", error: "错误" }[role] || role;
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;
    wrap.append(who, bubble);
    $("transcript").appendChild(wrap);
    scroll();
    return bubble;
  }

  function agentName() {
    return ($("agent-name").textContent || "助手").trim();
  }

  function beginAssistant() {
    state.currentAssistantText = "";
    state.currentAssistant = addTurn("assistant", "");
    state.currentAssistant.classList.add("typing");
  }

  function appendAssistant(text) {
    if (!state.currentAssistant) beginAssistant();
    state.currentAssistantText += text;
    state.currentAssistant.textContent = state.currentAssistantText;
    scroll();
  }

  function endAssistant(text) {
    const finalText = (text || state.currentAssistantText || "").trim();
    if (!state.currentAssistant) {
      // A tool-only turn can produce a final message without any deltas.
      if (finalText) addTurn("assistant", finalText);
      state.currentAssistantText = "";
      return;
    }
    if (finalText) {
      state.currentAssistant.textContent = finalText;
    } else {
      state.currentAssistant.closest(".turn").remove();
    }
    state.currentAssistant.classList.remove("typing");
    state.currentAssistant = null;
    state.currentAssistantText = "";
    scroll();
  }

  function addToolCall(data) {
    const el = document.createElement("details");
    el.className = "tool";
    const args = JSON.stringify(data.arguments || {}, null, 2);
    el.innerHTML = `
      <summary>
        <span class="name"></span>
        <span class="status">运行中…</span>
      </summary>
      <div class="log"></div>`;
    el.querySelector(".name").textContent = data.name;
    el.querySelector(".log").textContent = args;
    $("transcript").appendChild(el);
    state.pendingTools.set(data.id || data.name, el);
    scroll();
    return el;
  }

  function toolProgress(data) {
    // Attach to the most recent running tool card.
    const cards = state.pendingTools;
    let el = null;
    for (const value of cards.values()) {
      if (value.querySelector(".status").textContent === "运行中…") el = value;
    }
    if (!el) return;
    const log = el.querySelector(".log");
    log.textContent += (log.textContent ? "\n" : "") + (data.message || "");
    log.scrollTop = log.scrollHeight;
    scroll();
  }

  function toolResult(data) {
    let el = state.pendingTools.get(data.id) || null;
    if (!el) {
      for (const value of state.pendingTools.values()) {
        if (value.querySelector(".status").textContent === "运行中…") el = value;
      }
    }
    if (!el) return;
    const status = el.querySelector(".status");
    status.textContent = data.declined ? "已取消" : data.ok ? "完成" : "失败";
    status.className = `status ${data.ok ? "ok" : "fail"}`;
    el.open = false;
  }

  // -------------------------------------------------------------- playback

  function schedule(buffer) {
    if (!state.playCtx) return;
    const ctx = state.playCtx;
    const start = Math.max(state.nextTime, ctx.currentTime + 0.02);
    const src = ctx.createBufferSource();
    src.buffer = buffer;
    src.connect(ctx.destination);
    src.start(start);
    state.nextTime = start + buffer.duration;
    state.activeSources.push(src);
    src.onended = () => {
      state.activeSources = state.activeSources.filter((s) => s !== src);
    };

    if (state.turnSentAt && !state.ttfaReported) {
      state.ttfaReported = true;
      // Measured here, not on the server: the server cannot know when the
      // browser actually started playing. Without this you never really
      // measure time-to-first-audio.
      send({
        type: "metric",
        name: "client_ttfa_ms",
        value: Math.round(performance.now() - state.turnSentAt),
      });
    }
  }

  function markTurnSubmitted() {
    state.turnSentAt = performance.now();
    state.ttfaReported = false;
  }

  function playChunk(data) {
    if (!state.playCtx) return;
    const bytes = bytesFromB64(data.data);
    if (data.mime === "audio/pcm") {
      const rate = data.sample_rate || 24000;
      const count = Math.floor(bytes.length / 2);
      const buffer = state.playCtx.createBuffer(1, count, rate);
      const channel = buffer.getChannelData(0);
      const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
      for (let i = 0; i < count; i++) channel[i] = view.getInt16(i * 2, true) / 32768;
      schedule(buffer);
      return;
    }
    // Encoded containers (wav/mp3/ogg) — let the browser decode them.
    state.playCtx.decodeAudioData(
      bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
      (buffer) => schedule(buffer),
      (err) => console.warn("decode failed", err)
    );
  }

  function stopPlayback() {
    for (const src of state.activeSources) {
      try { src.stop(); } catch (_) { /* already finished */ }
    }
    state.activeSources = [];
    state.nextTime = 0;
  }

  const isPlaying = () => state.activeSources.length > 0;

  // --------------------------------------------------------------- capture

  async function ensureCapture() {
    if (state.source) return true;
    try {
      state.stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
    } catch (err) {
      addTurn("error", `无法访问麦克风：${err.message}`);
      return false;
    }
    // Ask for 16 kHz so the browser resamples for us; fall back if refused.
    try {
      state.captureCtx = new AudioContext({ sampleRate: 16000 });
    } catch (_) {
      state.captureCtx = new AudioContext();
    }
    state.sampleRate = state.captureCtx.sampleRate;
    await state.captureCtx.audioWorklet.addModule("/static/pcm-worklet.js");
    state.source = state.captureCtx.createMediaStreamSource(state.stream);
    state.worklet = new AudioWorkletNode(state.captureCtx, "pcm-capture");
    state.worklet.port.onmessage = (event) => onSamples(event.data);
    state.source.connect(state.worklet);
    // The worklet must be connected to something for process() to run.
    state.worklet.connect(state.captureCtx.destination);
    return true;
  }

  function ensurePlayback() {
    if (!state.playCtx) state.playCtx = new AudioContext();
    if (state.playCtx.state === "suspended") state.playCtx.resume();
  }

  function onSamples(frame) {
    if (state.browserAsr) return;
    const frameLen = Math.max(1, Math.round(state.sampleRate * FRAME_MS / 1000));

    // Re-assemble 128-sample worklet blocks into 20 ms VAD frames.
    let buf;
    const leftover = state.leftover;
    if (leftover && leftover.length) {
      buf = new Float32Array(leftover.length + frame.length);
      buf.set(leftover, 0);
      buf.set(frame, leftover.length);
    } else {
      buf = frame;
    }

    let offset = 0;
    while (buf.length - offset >= frameLen) {
      handleFrame(buf.subarray(offset, offset + frameLen));
      offset += frameLen;
    }
    state.leftover = offset ? buf.slice(offset) : buf;
  }

  function handleFrame(frame) {
    if (!state.recording) return;
    if (state.mode === "ptt") {
      queueForSend(frame);
      return;
    }

    // Hands-free: energy VAD with an adaptive noise floor.
    const level = rms(frame);
    if (level < state.noiseFloor) {
      state.noiseFloor = 0.9 * state.noiseFloor + 0.1 * level;
    } else {
      state.noiseFloor = 0.999 * state.noiseFloor + 0.001 * level;
    }

    const playing = isPlaying();
    const gate = playing && state.autoBargeIn ? BARGE_GAIN : 1.0;
    const threshold = Math.max(state.baseThreshold, state.noiseFloor * 3) * gate;

    let speech = level > threshold;
    if (playing && !state.autoBargeIn) {
      // Half-duplex: hands-free listening is paused while we speak, so the
      // microphone cannot pick up our own voice at all.
      speech = false;
    } else if (speech && playing) {
      state.bargeFrames += 1;
      if (state.bargeFrames < BARGE_FRAMES) speech = false;
    } else if (!speech || !playing) {
      state.bargeFrames = 0;
    }

    const prerollLimit = playing ? BARGE_PREROLL_FRAMES : PREROLL_FRAMES;

    if (!state.speechStarted) {
      state.preroll.push(frame);
      if (state.preroll.length > prerollLimit) state.preroll.shift();
      if (speech) {
        state.speechStarted = true;
        state.speechMs = FRAME_MS;
        state.silenceMs = 0;
        bargeIn();
        send({ type: "audio.start" });
        state.preroll.forEach(queueForSend);
        state.preroll = [];
      }
      return;
    }

    queueForSend(frame);
    if (speech) {
      state.speechMs += FRAME_MS;
      state.silenceMs = 0;
    } else {
      state.silenceMs += FRAME_MS;
    }
    if (state.silenceMs >= state.silenceMsLimit) {
      if (state.speechMs >= state.minSpeechMs) finishUtterance();
    } else if (state.speechMs > 60000) {
      finishUtterance();
    }
  }

  function queueForSend(frame) {
    state.sendBuf.push(frame);
    if (state.sendBuf.length >= SEND_FRAMES) flushSendBuffer();
  }

  function flushSendBuffer() {
    if (!state.sendBuf.length) return;
    let total = 0;
    for (const frame of state.sendBuf) total += frame.length;
    const merged = new Float32Array(total);
    let offset = 0;
    for (const frame of state.sendBuf) {
      merged.set(frame, offset);
      offset += frame.length;
    }
    state.sendBuf = [];
    const pcm = floatToInt16(merged);
    send({
      type: "audio.chunk",
      pcm: b64FromBytes(new Uint8Array(pcm.buffer)),
      sample_rate: state.sampleRate,
    });
  }

  function bargeIn(force = false) {
    if (!isPlaying()) return;
    if (!force && !state.autoBargeIn) return;
    // Stop locally first: waiting for the server makes interruption feel broken.
    // `barge_in` (not `interrupt`) is deliberate: the server knows that a
    // pending confirmation must survive, whereas `interrupt` is an explicit
    // "stop everything" from the user.
    stopPlayback();
    send({ type: "barge_in" });
  }

  function finishUtterance() {
    flushSendBuffer();
    send({ type: "audio.end" });
    markTurnSubmitted();
    state.speechStarted = false;
    state.speechMs = 0;
    state.silenceMs = 0;
    state.preroll = [];
  }

  async function startRecording() {
    if (state.recording) return;
    if (!state.browserAsr && !(await ensureCapture())) return;
    ensurePlayback();

    state.recording = true;
    state.speechStarted = false;
    state.speechMs = 0;
    state.silenceMs = 0;
    state.preroll = [];
    state.sendBuf = [];
    state.leftover = null;

    $("btn-ptt").classList.add("recording");
    $("btn-ptt").querySelector(".ptt-label").textContent =
      state.mode === "ptt" ? "松开发送" : "点击停止";

    bargeIn(true);

    if (state.browserAsr) {
      startBrowserAsr();
    } else if (state.mode === "ptt") {
      // Hands-free sends `audio.start` itself, on speech onset.
      send({ type: "audio.start" });
    }
  }

  function stopRecording() {
    if (!state.recording) return;
    state.recording = false;
    $("btn-ptt").classList.remove("recording");
    $("btn-ptt").querySelector(".ptt-label").textContent =
      state.mode === "ptt" ? "按住说话" : "点击开始 / 停止";

    if (state.browserAsr) {
      stopBrowserAsr();
      return;
    }
    if (state.mode === "ptt") {
      finishUtterance();
      return;
    }
    // Hands-free: close whatever was captured, if anything.
    flushSendBuffer();
    if (state.speechStarted) send({ type: "audio.end" });
    state.speechStarted = false;
    state.preroll = [];
  }

  // ---------------------------------------------------------- browser ASR

  function startBrowserAsr() {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SR) {
      addTurn("error", "这个浏览器不支持语音识别，请改用 Chrome，或在服务端配置 ASR。");
      return;
    }
    if (!state.recognition) {
      const recog = new SR();
      recog.lang = "zh-CN";
      recog.interimResults = true;
      recog.continuous = state.mode === "handsfree";
      recog.onresult = (event) => {
        for (let i = event.resultIndex; i < event.results.length; i++) {
          const result = event.results[i];
          const text = result[0].transcript.trim();
          if (!text) continue;
          if (result.isFinal) {
            markTurnSubmitted();
            send({ type: "text", text });
          } else {
            showPartial(text);
          }
        }
      };
      recog.onerror = (event) => {
        if (event.error !== "no-speech" && event.error !== "aborted") {
          addTurn("error", `浏览器识别出错：${event.error}`);
        }
      };
      recog.onend = () => {
        state.recognitionActive = false;
        if (state.recording && state.mode === "handsfree") {
          try { recog.start(); state.recognitionActive = true; } catch (_) {}
        }
      };
      state.recognition = recog;
    }
    state.recognition.continuous = state.mode === "handsfree";
    try {
      state.recognition.start();
      state.recognitionActive = true;
    } catch (_) { /* already started */ }
  }

  function stopBrowserAsr() {
    if (state.recognition && state.recognitionActive) {
      try { state.recognition.stop(); } catch (_) {}
    }
  }

  let partialEl = null;
  function showPartial(text) {
    if (!partialEl) {
      partialEl = addTurn("user", "");
      partialEl.classList.add("typing");
    }
    partialEl.textContent = text;
    scroll();
  }
  function clearPartial() {
    if (partialEl) {
      const turn = partialEl.closest(".turn");
      if (turn) turn.remove();
      partialEl = null;
    }
  }

  // -------------------------------------------------------------- events

  function handleEvent(event) {
    switch (event.type) {
      case "ready":
        state.ready = true;
        if (event.agent) $("agent-name").textContent = event.agent;
        renderReady(event);
        break;

      case "state":
        setStateBadge(event.state);
        break;

      case "transcript":
        clearPartial();
        addTurn("user", event.text);
        break;

      case "assistant.delta":
        appendAssistant(event.text || "");
        break;

      case "assistant.done":
        endAssistant(event.text || "");
        break;

      case "tool.call":
        addToolCall(event);
        break;

      case "tool.progress":
        toolProgress(event);
        break;

      case "tool.result":
        toolResult(event);
        break;

      case "confirm.request":
        showConfirm(event.question);
        break;

      case "audio":
        playChunk(event);
        break;

      case "metric":
        state.metrics[event.name] = event.value;
        renderMetrics();
        break;

      case "error":
        clearPartial();
        addTurn("error", `${event.where ? `[${event.where}] ` : ""}${event.message}`);
        break;

      case "log":
        if (event.level !== "debug") addTurn("system", event.message);
        break;

      default:
        break;
    }
  }

  function setStateBadge(value) {
    const badge = $("state-badge");
    const labels = {
      idle: "待机", listening: "聆听中", transcribing: "识别中",
      thinking: "思考中", acting: "执行中", speaking: "回答中",
      awaiting_confirmation: "等待确认",
    };
    badge.textContent = labels[value] || value;
    badge.className = `badge ${value}`;
  }

  function showConfirm(question) {
    state.awaitingConfirm = true;
    $("confirm-text").textContent = question;
    $("confirm-bar").hidden = false;
  }

  function hideConfirm() {
    state.awaitingConfirm = false;
    $("confirm-bar").hidden = true;
  }

  function answerConfirm(approved) {
    hideConfirm();
    send({ type: "confirm", approved });
  }

  function renderReady(event) {
    const providers = event.providers || {};
    const box = $("panel-body");
    box.innerHTML = "";
    const rows = [
      ["会话", event.session_id],
      ["ASR", providers.asr],
      ["TTS", providers.tts],
      ["LLM", providers.llm],
    ];
    for (const [key, value] of rows) {
      const row = document.createElement("div");
      row.className = "row";
      row.innerHTML = `<span>${key}</span><span></span>`;
      row.querySelector("span:last-child").textContent = value || "-";
      box.appendChild(row);
    }
    (event.warnings || []).forEach((w) => {
      const div = document.createElement("div");
      div.className = "warn";
      div.textContent = `⚠ ${w}`;
      box.appendChild(div);
    });

    const tools = $("tools");
    tools.innerHTML = "";
    (event.tools || []).forEach((name) => {
      const div = document.createElement("div");
      div.className = "tool-item";
      div.innerHTML = "<b></b>";
      div.querySelector("b").textContent = name;
      tools.appendChild(div);
    });

    // If the server has no ASR of its own, use the browser's.
    if (providers.asr === "browser" || providers.asr === "mock") {
      state.browserAsr = true;
      $("btn-asr-mode").textContent = "识别：浏览器";
    }

    // Take the server's turn-taking thresholds so both ends agree.
    const vad = event.vad || {};
    if (vad.silence_ms) state.silenceMsLimit = Number(vad.silence_ms);
    if (vad.threshold) state.baseThreshold = Number(vad.threshold);
    if (vad.min_utterance_ms !== undefined) {
      state.minSpeechMs = Number(vad.min_utterance_ms);
    }
  }

  function renderMetrics() {
    const names = {
      asr_ms: "识别耗时",
      llm_first_token_ms: "首个 token",
      tts_first_audio_ms: "首包音频（服务端）",
      client_ttfa_ms: "首包音频（真实）",
      turn_total_ms: "本轮总耗时",
    };
    const box = $("metrics");
    box.innerHTML = "";
    for (const [key, label] of Object.entries(names)) {
      if (state.metrics[key] === undefined) continue;
      const div = document.createElement("div");
      div.className = "metric";
      div.innerHTML = `<span>${label}</span><span></span>`;
      div.querySelector("span:last-child").textContent = `${Math.round(state.metrics[key])} ms`;
      box.appendChild(div);
    }
  }

  // ------------------------------------------------------------------ ws

  function connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const params = new URLSearchParams(location.search);
    const token = params.get("token");
    const url = `${proto}://${location.host}/ws${token ? `?token=${encodeURIComponent(token)}` : ""}`;
    const ws = new WebSocket(url);
    state.ws = ws;

    ws.onopen = () => {
      state.connected = true;
      state.reconnectDelay = 500;
      $("conn-dot").classList.add("on");
      loadCapabilities();
    };

    ws.onmessage = (raw) => {
      let event;
      try { event = JSON.parse(raw.data); } catch (_) { return; }
      handleEvent(event);
    };

    ws.onclose = () => {
      state.connected = false;
      state.ready = false;
      $("conn-dot").classList.remove("on");
      setStateBadge("disconnected");
      $("state-badge").textContent = "已断开";
      setTimeout(connect, state.reconnectDelay);
      state.reconnectDelay = Math.min(state.reconnectDelay * 2, 8000);
    };

    ws.onerror = () => { /* onclose handles it */ };
  }

  async function loadCapabilities() {
    try {
      const res = await fetch("/api/capabilities");
      state.capabilities = await res.json();
    } catch (_) { /* non-fatal */ }
  }

  // ------------------------------------------------------------------ ui

  function bindPushToTalk() {
    const btn = $("btn-ptt");
    let active = false;

    const down = async (event) => {
      event.preventDefault();
      if (active || state.awaitingConfirm) return;
      active = true;
      await startRecording();
      if (state.mode === "handsfree") return; // hands-free keeps running
    };
    const up = (event) => {
      if (event) event.preventDefault();
      if (!active) return;
      active = false;
      if (state.mode === "ptt") stopRecording();
    };

    btn.addEventListener("pointerdown", down);
    btn.addEventListener("pointerup", up);
    btn.addEventListener("pointercancel", up);
    btn.addEventListener("pointerleave", up);

    document.addEventListener("keydown", (event) => {
      if (event.code !== "Space" || event.repeat) return;
      const tag = (event.target.tagName || "").toLowerCase();
      if (tag === "input" || tag === "textarea") return;
      if (active) return;
      event.preventDefault();
      active = true;
      startRecording();
    });
    document.addEventListener("keyup", (event) => {
      if (event.code !== "Space") return;
      if (!active) return;
      event.preventDefault();
      active = false;
      if (state.mode === "ptt") stopRecording();
    });
  }

  function toggleHandsFree() {
    const handsfree = state.mode === "ptt";
    if (handsfree) stopRecording();
    state.mode = handsfree ? "handsfree" : "ptt";
    $("btn-handsfree").textContent = handsfree ? "免提：开" : "免提：关";
    $("btn-handsfree").classList.toggle("primary", handsfree);
    $("btn-ptt").querySelector(".ptt-label").textContent = handsfree ? "点击开始 / 停止" : "按住说话";
    if (handsfree) startRecording();
  }

  function bind() {
    bindPushToTalk();

    $("btn-handsfree").addEventListener("click", toggleHandsFree);

    $("btn-bargein").addEventListener("click", () => {
      state.autoBargeIn = !state.autoBargeIn;
      $("btn-bargein").textContent = state.autoBargeIn ? "打断：开" : "打断：关";
      $("btn-bargein").classList.toggle("primary", state.autoBargeIn);
    });

    $("btn-asr-mode").addEventListener("click", () => {
      state.browserAsr = !state.browserAsr;
      $("btn-asr-mode").textContent = state.browserAsr ? "识别：浏览器" : "识别：服务端";
      if (!state.browserAsr) stopBrowserAsr();
    });

    $("btn-panel").addEventListener("click", () => {
      const app = $("app");
      app.classList.toggle("with-panel");
      $("panel").hidden = !app.classList.contains("with-panel");
    });

    $("btn-stop").addEventListener("click", () => {
      stopPlayback();
      hideConfirm();
      send({ type: "interrupt" });
    });

    $("btn-confirm-yes").addEventListener("click", () => answerConfirm(true));
    $("btn-confirm-no").addEventListener("click", () => answerConfirm(false));

    $("text-form").addEventListener("submit", (event) => {
      event.preventDefault();
      const input = $("text-input");
      const text = input.value.trim();
      if (!text) return;
      input.value = "";
      ensurePlayback();
      bargeIn(true);
      markTurnSubmitted();
      send({ type: "text", text });
    });

    window.addEventListener("beforeunload", () => {
      if (state.ws) state.ws.close();
    });
  }

  bind();
  connect();
})();
