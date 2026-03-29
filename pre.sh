#!/usr/bin/env bash
set -euo pipefail

ME="${USER:-$(whoami)}"

echo "[1/6] Stop Ray (best-effort)"
ray stop --force >/dev/null 2>&1 || true

kill_pat() {
  local sig="$1"; shift
  local pat="$1"; shift || true

  local pids
  pids="$(pgrep -u "$ME" -f "$pat" || true)"
  if [[ -n "${pids}" ]]; then
    echo "  -> kill -$sig pattern='$pat' pids: $pids"
    kill "-$sig" $pids >/dev/null 2>&1 || true
  fi
}

echo "[2/6] Graceful terminate key services"
# Ray core
kill_pat TERM "raylet"
kill_pat TERM "gcs_server"
kill_pat TERM "dashboard"
kill_pat TERM "ray::"            # ray worker

# vLLM
kill_pat TERM "VLLM::EngineCor"
kill_pat TERM "python.*-m vllm"
kill_pat TERM "vllm"

# kill_pat TERM "coldstart"
for pat in \
  "pred_serve" \
  "visionthink" \
  "run_generalqa" \
  "eval\.sh" \
  "run\.sh" \
  "judge" \
  "train" \
  "multiprocessing" \
  "spawn" \
  "mix.sh" \
  "prepa"
do
  kill_pat TERM "$pat"
done

sleep 2

echo "[3/6] Force kill remaining (KILL)"
# Ray core
kill_pat KILL "raylet"
kill_pat KILL "gcs_server"
kill_pat KILL "dashboard"
kill_pat KILL "ray::"

# vLLM
kill_pat KILL "VLLM::EngineCor"
kill_pat KILL "python.*-m vllm"
kill_pat KILL "vllm"

for pat in \
  "pred_serve" \
  "visionthink" \
  "run_generalqa" \
  "eval\.sh" \
  "run\.sh" \
  "judge" \
  "train" \
  "multiprocessing" \
  "spawn"
do
  kill_pat KILL "$pat"
done

echo "[4/6] Cleanup temp caches"
rm -rf /tmp/torchinductor_* 2>/dev/null || true
rm -rf /tmp/ray 2>/dev/null || true

echo "[5/6] Show remaining suspicious processes"
pgrep -u "$ME" -af "raylet|gcs_server|dashboard|VLLM::EngineCor|vllm|pred_serve|visionthink" || echo "  none"

echo "[6/6] GPU check"
command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || true

echo "Done."
