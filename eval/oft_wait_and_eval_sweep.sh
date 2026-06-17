#!/bin/bash
# Wait for the Stage-2a training to finish, then evaluate a spread of snapshots
# on the blue_chick task and aggregate task-success rates.
# Designed to run unattended (launched in background). Robust: validates the GPU
# eval path on the FIRST snapshot before sweeping the rest; per-step error capture.
OFT=/home/capstone/openvla-oft
RUNDIR=/home/capstone/openvla_ckpts/runs_oft
OUT="$RUNDIR/eval_sweep"
mkdir -p "$OUT"
SUMMARY="$OUT/SUMMARY.txt"
TRIALS=10
MAXS=400
# eval order: 30000 first (final ckpt = validation + most-trained), then a descending spread
STEPS="30000 25000 20000 15000 10000 5000"

echo "[sweep] started, waiting for training to finish ..." | tee "$SUMMARY"

# 1) wait for training process to exit (safety cap ~7h)
WAITED=0
while pgrep -f "vla-scripts/finetune.py" >/dev/null 2>&1; do
  sleep 120; WAITED=$((WAITED+120))
  if [ "$WAITED" -gt 25200 ]; then echo "[sweep] WARN: waited >7h, proceeding anyway" | tee -a "$SUMMARY"; break; fi
done
echo "[sweep] training process gone at $(date). Freeing GPU (sleep 30) ..." | tee -a "$SUMMARY"
sleep 30
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader | tee -a "$SUMMARY"

extract_rate () {  # parse success rate from an eval log
  grep -hiE "success rate|total .*success|Success: |successes" "$1" 2>/dev/null | tail -5
}

VALIDATED=0
for s in $STEPS; do
  SNAP=$(ls -d "$RUNDIR"/*stage2a*--"${s}"_chkpt 2>/dev/null | head -1)
  [ -z "$SNAP" ] && { echo "[sweep] step $s: snapshot missing, skip" | tee -a "$SUMMARY"; continue; }
  LOGS="$OUT/eval_step${s}.log"
  echo "[sweep] === step $s -> $LOGS ($(date)) ===" | tee -a "$SUMMARY"
  bash "$OFT/scripts/oft_eval_snapshot.sh" "$s" "$TRIALS" "$MAXS" > "$LOGS" 2>&1
  RC=$?
  RATE=$(extract_rate "$LOGS")
  echo "[sweep] step $s rc=$RC : ${RATE:-<no success line found>}" | tee -a "$SUMMARY"

  # after the FIRST attempt, decide whether the GPU eval path actually works
  if [ "$VALIDATED" -eq 0 ]; then
    if [ "$RC" -ne 0 ] && [ -z "$RATE" ]; then
      echo "[sweep] !!! VALIDATION FAILED on first snapshot (rc=$RC, no result). Aborting sweep." | tee -a "$SUMMARY"
      echo "[sweep] See $LOGS (likely merge auto_map / model-load issue to fix interactively)." | tee -a "$SUMMARY"
      exit 3
    fi
    VALIDATED=1
    echo "[sweep] first snapshot produced output -> GPU eval path OK, continuing sweep." | tee -a "$SUMMARY"
  fi
done
echo "[sweep] DONE at $(date). Summary above; per-step logs in $OUT/." | tee -a "$SUMMARY"
