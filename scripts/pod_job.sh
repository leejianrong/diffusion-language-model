#!/usr/bin/env bash
# Pod-side job: smoke-train AR and DLM on a slice of OpenWebText, narrating itself the whole way.
# Shipped inside the create request (PodPlan.job_file), so nothing is fetched before the pod's
# dead-man's-switch is armed. Grep-able tokens: STAGE, hb, WARN, ERROR, STAGE failed, progress,
# RP_REPORT_BEGIN/END, JOB_DONE.
#
# Settings come from the pod's variables; LOCAL=1 rehearses the same script on a dev machine
# (no install, no clone, no GPU checks).
set -Eeuo pipefail

REPO_URL=${REPO_URL:-https://github.com/leejianrong/diffusion-language-model}
GIT_REF=${GIT_REF:-main}
WORK=${WORK:-/workspace}
RUN=${RUN:-$WORK/job}
TRAIN_TOKENS=${TRAIN_TOKENS:-40000000}
VAL_TOKENS=${VAL_TOKENS:-2000000}
SHARD_TOKENS=${SHARD_TOKENS:-16777216}   # 16384 * 1024
CONFIGS=${CONFIGS:-"smoke_ar smoke_dlm"}
HB_SECS=${HB_SECS:-30}
HB_PAUSE_AFTER=${HB_PAUSE_AFTER:-600}
T_INSTALL=${T_INSTALL:-900}
T_DATA=${T_DATA:-1500}
T_TRAIN=${T_TRAIN:-1200}
DEVICE=${DEVICE:-cuda}
LOCAL=${LOCAL:-0}
MODE=${MODE:-smoke}   # smoke | resume | diag
KILL_AFTER_STEP=${KILL_AFTER_STEP:-160}
TORCH_SPEC=${TORCH_SPEC:-torch==2.15.0}

mkdir -p "$RUN"
: > "$RUN/work.log"
echo starting > "$RUN/stage"
echo running > "$RUN/status"
START=$(date +%s)

stage() { echo "$1" > "$RUN/stage"; echo "STAGE $1 (t+$(( $(date +%s) - START ))s)"; }
fail() {
  echo "STAGE failed ($1; was: $(cat "$RUN/stage"))"
  echo "ERROR job failed in stage $(cat "$RUN/stage")"
  echo failed > "$RUN/status"
}
trap 'fail "line $LINENO"' ERR

heartbeat() {
  set +e; trap - ERR   # a failed probe here must never mark the job failed
  local low=0 paused=0
  while true; do
    sleep "$HB_SECS"
    local now quiet gpu disk ram util
    now=$(date +%s)
    quiet=$(( now - $(stat -c %Y "$RUN/work.log") ))
    disk=$(df --output=pcent "$WORK" 2>/dev/null | tail -1 | tr -dc 0-9 || true)
    ram=$(free | awk '/Mem:/ {printf "%d", $3*100/$2}')
    gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ' || true)
    if [ "$quiet" -gt "$HB_PAUSE_AFTER" ]; then
      # Stop talking so the pod's idle watchdog can see the job is hung and terminate the pod.
      [ "$paused" = 0 ] && echo "WARN no job output for ${quiet}s; heartbeat paused so the idle watchdog can fire"
      paused=1
      continue
    fi
    paused=0
    echo "hb stage=$(cat "$RUN/stage") t=$(( now - START ))s gpu=${gpu:-none} disk=${disk:-?}% ram=${ram:-?}% quiet=${quiet}s"
    [ "${disk:-0}" -gt 90 ] && echo "WARN disk ${disk}% full"
    [ "${ram:-0}" -gt 90 ] && echo "WARN RAM ${ram}% used"
    util=${gpu%%,*}
    case "$(cat "$RUN/stage")" in
      train-*)
        if [ -n "$util" ] && [ "$util" -lt 5 ] 2>/dev/null; then low=$((low + 1)); else low=0; fi
        [ "$low" -ge 4 ] && echo "WARN GPU under 5% for $((low * HB_SECS))s during training" ;;
    esac
  done
}

report() {
  echo "RP_REPORT_BEGIN"
  echo "{\"status\": \"$(cat "$RUN/status")\", \"stage\": \"$(cat "$RUN/stage")\", \"secs\": $(( $(date +%s) - START ))}"
  for c in $CONFIGS; do
    [ -f "$WORK/runs/$c/results.json" ] && echo "{\"run\": \"$c\", \"results\": $(tr -d '\n' < "$WORK/runs/$c/results.json")}"
    [ -f "$WORK/runs/$c/log.jsonl" ] && echo "{\"run\": \"$c\", \"last_log\": $(tail -1 "$WORK/runs/$c/log.jsonl")}"
  done
  echo "RP_REPORT_END"
  return 0
}
# Always publish what we have, success or failure; the pod lingers a grace period after this.
on_exit() {
  local code=$?
  kill "$HB" 2>/dev/null || true
  if [ "$(cat "$RUN/status")" = running ] && [ "$code" = 0 ]; then echo ok > "$RUN/status"; fi
  report
  echo "JOB_DONE status=$(cat "$RUN/status") exit=$code"
}
trap on_exit EXIT

heartbeat & HB=$!

# step <seconds> <command...>: run under a timeout, output to the pod log and work.log
step() {
  local secs=$1; shift
  timeout "$secs" "$@" 2>&1 | tee -a "$RUN/work.log"
}

stage boot
echo "host: $(uname -srm); python: $(command -v python3)"
if [ "$LOCAL" != 1 ]; then
  step 60 nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

  stage install
  step "$T_INSTALL" pip install -q uv
  step "$T_INSTALL" uv venv --python 3.12 "$WORK/venv"
  # shellcheck disable=SC1091
  source "$WORK/venv/bin/activate"
  step "$T_INSTALL" uv pip install -q "$TORCH_SPEC" numpy pyarrow pyyaml tiktoken
  step 300 git clone -q "$REPO_URL" "$WORK/repo"
  cd "$WORK/repo" && git checkout -q "$GIT_REF"
  step 300 uv pip install -q --no-deps -e .
  echo "repo at $(git rev-parse --short HEAD)"

  stage cuda-check
  step 300 python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not available to torch"
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
torch.cuda.synchronize()
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0),
      "bf16 matmul ok", float((x @ x).float().abs().mean()) > 0)
PY
fi

stage data
if [ "$LOCAL" != 1 ]; then
  step "$T_DATA" dlm prep-data --out "$WORK/data" --raw "$WORK/raw" \
    --train-tokens "$TRAIN_TOKENS" --val-tokens "$VAL_TOKENS" --shard-tokens "$SHARD_TOKENS"
  rm -rf "$WORK/raw"
fi
stage verify
step 300 dlm verify-data --data "${DATA_DIR:-$WORK/data}" --manifest "${DATA_DIR:-$WORK/data}/manifest.json"

if [ "$MODE" = diag ]; then
  stage diag
  step "$T_TRAIN" python scripts/resume_diag.py "${CONFIG_DIR:-configs}/smoke_dlm.yaml"
  CONFIGS=""
fi

for c in $CONFIGS; do
  stage "train-$c"
  step "$T_TRAIN" dlm train --config "${CONFIG_DIR:-configs}/$c.yaml" --device "$DEVICE"
done

if [ "$MODE" = diag ]; then
  :
elif [ "$MODE" = resume ]; then
  # Kill a run partway with SIGKILL, resume it from its checkpoint, and require the final
  # validation loss to land within 1% of the uninterrupted run above.
  mkdir -p "$RUN/cfg"
  for c in $CONFIGS; do
    stage "kill-$c"
    sed "s/^run_name: .*/run_name: ${c}_split/" "${CONFIG_DIR:-configs}/$c.yaml" > "$RUN/cfg/$c.yaml"
    rm -rf "$WORK/runs/${c}_split"
    dlm train --config "$RUN/cfg/$c.yaml" --device "$DEVICE" >> "$RUN/work.log" 2>&1 &
    TP=$!
    log="$WORK/runs/${c}_split/log.jsonl"
    until [ -f "$log" ] && python3 -c "import json,sys; sys.exit(0 if any(json.loads(l)['step'] >= $KILL_AFTER_STEP for l in open('$log')) else 1)"; do
      kill -0 "$TP" 2>/dev/null || { echo "ERROR training ended before the kill point"; exit 1; }
      sleep 2
    done
    kill -9 "$TP"; wait "$TP" 2>/dev/null || true
    echo "killed $c at: $(tail -1 "$log" | cut -c1-80)"
    stage "resume-$c"
    step "$T_TRAIN" dlm train --config "$RUN/cfg/$c.yaml" --device "$DEVICE" --resume
    python3 - "$c" "$WORK" <<'PY'
import json, sys
c, work = sys.argv[1:]
a = json.load(open(f"{work}/runs/{c}/results.json"))["eval_loss"]
b = json.load(open(f"{work}/runs/{c}_split/results.json"))["eval_loss"]
rel = abs(a - b) / a
print(f"RESUME_CHECK {c} straight={a:.5f} killed_and_resumed={b:.5f} rel_diff={rel:.5%}")
sys.exit(0 if rel < 0.01 else 1)
PY
  done
else
  stage resume-check   # loads the final checkpoint and does zero further steps
  first=${CONFIGS%% *}
  step 600 dlm train --config "${CONFIG_DIR:-configs}/$first.yaml" --device "$DEVICE" --resume
fi

stage done
