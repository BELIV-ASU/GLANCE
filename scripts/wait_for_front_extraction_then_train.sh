#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PROGRESS_LOG="${ROOT_DIR}/results/waymo_front_extraction_progress.log"
STATUS_LOG="${ROOT_DIR}/results/wait_then_train_status.log"

# Training policy
NUM_EPOCHS="${NUM_EPOCHS:-5}"
TRAJ_MAX_SAMPLES="${TRAJ_MAX_SAMPLES:-0}"  # 0 => all samples
FORCE_REGENERATE="${FORCE_REGENERATE:-1}"

# Completion detection policy
POLL_SECONDS="${POLL_SECONDS:-300}"
REQUIRED_STABLE_POLLS="${REQUIRED_STABLE_POLLS:-3}"

mkdir -p "${ROOT_DIR}/results"

echo "$(date '+%F %T') [watcher] starting" | tee -a "${STATUS_LOG}"
echo "$(date '+%F %T') [watcher] waiting for extraction to stabilize (stable polls=${REQUIRED_STABLE_POLLS}, poll=${POLL_SECONDS}s)" | tee -a "${STATUS_LOG}"

last_counts=""
stable=0

while true; do
  if [[ ! -f "${PROGRESS_LOG}" ]]; then
    echo "$(date '+%F %T') [watcher] progress log not found yet: ${PROGRESS_LOG}" | tee -a "${STATUS_LOG}"
    sleep "${POLL_SECONDS}"
    continue
  fi

  line="$(tail -n 1 "${PROGRESS_LOG}" || true)"
  if [[ -z "${line}" ]]; then
    echo "$(date '+%F %T') [watcher] progress log empty" | tee -a "${STATUS_LOG}"
    sleep "${POLL_SECONDS}"
    continue
  fi

  train="$(sed -n 's/.*train=\([0-9][0-9]*\).*/\1/p' <<<"${line}")"
  val="$(sed -n 's/.*val=\([0-9][0-9]*\).*/\1/p' <<<"${line}")"

  echo "$(date '+%F %T') [watcher] train=${train:-NA} val=${val:-NA}" | tee -a "${STATUS_LOG}"

  counts_key="${train:-NA}:${val:-NA}"
  if [[ "${counts_key}" == "${last_counts}" ]]; then
    stable=$((stable + 1))
  else
    stable=0
    last_counts="${counts_key}"
  fi

  # Consider extraction done when val has started and counts stop changing for N polls.
  if [[ -n "${val}" && "${val}" -gt 0 && "${stable}" -ge "${REQUIRED_STABLE_POLLS}" ]]; then
    echo "$(date '+%F %T') [watcher] extraction appears complete; launching training" | tee -a "${STATUS_LOG}"
    break
  fi

  sleep "${POLL_SECONDS}"
done

cd "${ROOT_DIR}"

source /scratch/rbaskar5/set.bash >/dev/null 2>&1
source activate new_beginning >/dev/null 2>&1

NUM_EPOCHS="${NUM_EPOCHS}" TRAJ_MAX_SAMPLES="${TRAJ_MAX_SAMPLES}" FORCE_REGENERATE="${FORCE_REGENERATE}" \
  bash "${ROOT_DIR}/scripts/run_waymo_qwen32b_trajectory.sh" 2>&1 | tee -a "${STATUS_LOG}"

echo "$(date '+%F %T') [watcher] training finished" | tee -a "${STATUS_LOG}"
