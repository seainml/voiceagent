#!/usr/bin/env bash
#
# voiceagent service management.
#
#   ./scripts/service.sh start [--host H] [--port P] [--reload]
#   ./scripts/service.sh stop
#   ./scripts/service.sh restart
#   ./scripts/service.sh status
#   ./scripts/service.sh logs [-f]
#   ./scripts/service.sh update [--deps]
#   ./scripts/service.sh doctor
#
# Runs the server detached in the background, keeps a PID file, and writes logs
# to .voiceagent/logs/. Everything it creates already lives under .voiceagent/,
# which is gitignored.
#
# For a service that survives reboots, see the launchd notes in the README.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_DIR="$ROOT/.voiceagent/run"
LOG_DIR="$ROOT/.voiceagent/logs"
PID_FILE="$RUN_DIR/voiceagent.pid"
LOG_FILE="$LOG_DIR/voiceagent.log"

HOST=""
PORT=""
RELOAD=""
DEPS=""
LOGS_ARGS=()

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

c_ok()   { printf '\033[32m%s\033[0m\n' "$*"; }
c_warn() { printf '\033[33m%s\033[0m\n' "$*"; }
c_err()  { printf '\033[31m%s\033[0m\n' "$*" >&2; }
c_dim()  { printf '\033[2m%s\033[0m\n' "$*"; }
die()    { c_err "$*"; exit 1; }

# Resolve the executable: prefer the project venv, fall back to PATH.
voiceagent_bin() {
  if [[ -x "$ROOT/.venv/bin/voiceagent" ]]; then
    echo "$ROOT/.venv/bin/voiceagent"
  elif command -v voiceagent >/dev/null 2>&1; then
    command -v voiceagent
  else
    die "voiceagent not found. Install it first:
    uv venv --python 3.12 .venv && source .venv/bin/activate
    uv pip install -e \".[dev]\""
  fi
}

# Read a value straight out of the resolved config, so the script and the
# server never disagree about which port is in use.
config_value() {
  local key="$1" fallback="$2"
  "$(voiceagent_bin)" config 2>/dev/null \
    | python3 -c "import json,sys;print(json.load(sys.stdin)$key)" 2>/dev/null \
    || echo "$fallback"
}

pid_from_file() {
  [[ -f "$PID_FILE" ]] || return 1
  local pid
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  [[ -n "$pid" ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  echo "$pid"
}

# Find a listener on the port even if we did not start it (e.g. a foreground
# `voiceagent serve` in another terminal).
pid_from_port() {
  local port="$1"
  command -v lsof >/dev/null 2>&1 || return 1
  lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null | head -1
}

resolve_endpoint() {
  local port="${PORT:-$(config_value "['server']['port']" 8765)}"
  local host="${HOST:-$(config_value "['server']['host']" 127.0.0.1)}"
  echo "$host $port"
}

health_ok() {
  local port="$1"
  curl -fsS --max-time 3 "http://127.0.0.1:${port}/api/health" >/dev/null 2>&1
}

# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

cmd_start() {
  read -r host port <<<"$(resolve_endpoint)"

  local existing
  if existing="$(pid_from_file)"; then
    c_warn "already running (pid $existing) — use 'restart' to pick up changes"
    return 0
  fi
  if existing="$(pid_from_port "$port")"; then
    c_warn "port $port is already in use by pid $existing (started outside this script)"
    c_dim "         stop it with: kill $existing"
    return 1
  fi

  mkdir -p "$RUN_DIR" "$LOG_DIR"

  local bin args=()
  bin="$(voiceagent_bin)"
  args=(serve --host "$host" --port "$port")
  [[ -n "$RELOAD" ]] && args+=(--reload)

  # nohup + disown so it survives the terminal that started it.
  nohup "$bin" "${args[@]}" >>"$LOG_FILE" 2>&1 &
  local pid=$!
  echo "$pid" >"$PID_FILE"

  for _ in $(seq 1 40); do
    if health_ok "$port"; then
      c_ok "started (pid $pid)  →  http://${host}:${port}"
      c_dim "  logs:   $LOG_FILE"
      c_dim "  stop:   ./scripts/service.sh stop"
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      rm -f "$PID_FILE"
      c_err "server exited during startup. Last lines of the log:"
      tail -n 20 "$LOG_FILE" >&2 || true
      return 1
    fi
    sleep 0.25
  done

  c_warn "started (pid $pid) but health check did not pass within 10s"
  c_dim "  last lines of $LOG_FILE:"
  tail -n 20 "$LOG_FILE" || true
  return 1
}

cmd_stop() {
  read -r _host port <<<"$(resolve_endpoint)"
  local pid stopped=0

  if pid="$(pid_from_file)"; then
    c_dim "stopping pid $pid …"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 40); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 0.25
    done
    if kill -0 "$pid" 2>/dev/null; then
      c_warn "did not exit in 10s, sending SIGKILL"
      kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$PID_FILE"
    stopped=1
  elif pid="$(pid_from_port "$port")"; then
    # Not started by this script, but the port is clearly ours to free.
    c_dim "no pid file; stopping listener on port $port (pid $pid)"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 40); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 0.25
    done
    stopped=1
  fi

  if [[ "$stopped" == "1" ]]; then
    c_ok "stopped"
  else
    c_dim "not running"
  fi
}

cmd_restart() {
  cmd_stop
  cmd_start
}

cmd_status() {
  read -r host port <<<"$(resolve_endpoint)"
  local pid="" source=""

  if pid="$(pid_from_file)"; then
    source="pid file"
  elif pid="$(pid_from_port "$port")"; then
    source="port $port (started outside this script)"
  fi

  if [[ -z "$pid" ]]; then
    c_warn "not running"
    c_dim "  start with: ./scripts/service.sh start"
    return 1
  fi

  local started elapsed health
  started="$(ps -o lstart= -p "$pid" 2>/dev/null | sed 's/^ *//' || echo '?')"
  elapsed="$(ps -o etime= -p "$pid" 2>/dev/null | tr -d ' ' || echo '?')"
  c_ok "running (pid $pid via $source)"
  echo "  url       http://${host}:${port}"
  echo "  started   $started  (up $elapsed)"
  echo "  log       $LOG_FILE"

  if health="$(curl -fsS --max-time 3 "http://127.0.0.1:${port}/api/health" 2>/dev/null)"; then
    echo "  health    $health"
  else
    c_warn "  health    endpoint not answering yet (still loading models?)"
  fi
}

cmd_logs() {
  [[ -f "$LOG_FILE" ]] || die "no log file yet at $LOG_FILE"
  if [[ "${1:-}" == "-f" || "${1:-}" == "--follow" ]]; then
    tail -f "$LOG_FILE"
  else
    tail -n "${LINES:-60}" "$LOG_FILE"
  fi
}

cmd_update() {
  local deps="${1:-}"
  local changed=0

  c_dim "fetching…"
  git fetch --quiet origin
  local branch
  branch="$(git rev-parse --abbrev-ref HEAD)"

  if [[ "$(git rev-parse HEAD)" == "$(git rev-parse "origin/$branch")" ]]; then
    c_dim "already up to date ($branch)"
  else
    c_dim "commits to apply:"
    git log --oneline "HEAD..origin/$branch" | sed 's/^/    /'
    if ! git diff --quiet || ! git diff --cached --quiet; then
      die "you have uncommitted changes; commit or stash them first"
    fi
    git merge --ff-only "origin/$branch"
    changed=1
    deps="--deps"   # dependency changes are likely; be safe
  fi

  if [[ "$deps" == "--deps" ]]; then
    c_dim "reinstalling dependencies…"
    UV_CACHE_DIR="$ROOT/.uv-cache" uv pip install -e ".[dev]" \
      --python "$ROOT/.venv/bin/python" >/dev/null
    changed=1
  fi

  if [[ "$changed" == "0" ]]; then
    c_ok "nothing to do — already current"
    return 0
  fi

  if pid_from_file >/dev/null 2>&1 \
     || pid_from_port "$(resolve_endpoint | awk '{print $2}')" >/dev/null 2>&1; then
    c_dim "restarting to pick up the new code…"
    cmd_restart
  else
    c_ok "updated (server was not running)"
  fi
}

cmd_doctor() {
  "$(voiceagent_bin)" doctor
}

usage() {
  cat <<'EOF'
voiceagent service management

  ./scripts/service.sh start [--host H] [--port P] [--reload]
  ./scripts/service.sh stop
  ./scripts/service.sh restart
  ./scripts/service.sh status
  ./scripts/service.sh logs [-f]
  ./scripts/service.sh update [--deps]
  ./scripts/service.sh doctor

`logs` prints the last 60 lines; set LINES=200 for more, or pass -f to follow.
`update` pulls, reinstalls only when needed, and restarts only if it changed.

Runs the server detached in the background, keeps a PID file and writes logs
to .voiceagent/logs/, which is gitignored.
EOF
  exit "${1:-0}"
}

# --------------------------------------------------------------------------
# argument parsing
# --------------------------------------------------------------------------

[[ $# -gt 0 ]] || usage 1
COMMAND="$1"; shift

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)   HOST="$2"; shift 2 ;;
    --port)   PORT="$2"; shift 2 ;;
    --reload) RELOAD=1; shift ;;
    --deps)   DEPS="--deps"; shift ;;
    -f|--follow) LOGS_ARGS+=("$1"); shift ;;
    -h|--help) usage 0 ;;
    *) die "unknown option: $1" ;;
  esac
done

case "$COMMAND" in
  start)   cmd_start ;;
  stop)    cmd_stop ;;
  restart) cmd_restart ;;
  status)  cmd_status ;;
  # `${arr[@]+...}` keeps `set -u` happy when the array is empty.
  logs)    cmd_logs ${LOGS_ARGS[@]+"${LOGS_ARGS[@]}"} ;;
  update)  cmd_update "$DEPS" ;;
  doctor)  cmd_doctor ;;
  *)       usage 1 ;;
esac
