#!/usr/bin/env bash
# M3 stage 2b (plan.md §12, af4 2026-10-06). One writer per rows file:
#  - llama: full main group (all arms incl. amlp) + agentsentry, now (CaMeL skipped on local models by the runner).
#  - hub models: amlp arm only, after stage 2a (other arms) has released the main rows file.
#  - qwen: after stage 1 -> select -> full main group + agentsentry.
cd "$(dirname "$0")"
A2=$1   # pid of m3_stage2_baselines_hub.sh
( python3 m3_online.py --model llama31-8b-local --group main --workers 8
  baselines/agentsentry_env/bin/python m3_online.py --model llama31-8b-local --group agentsentry --workers 8
  echo "$(date -u +%FT%TZ) stage2b llama done" ) > logs/m3_stage2b_llama.log 2>&1 &
( while kill -0 "$A2" 2>/dev/null; do sleep 60; done
  for m in gpt-4o-mini-2024-07-18 gpt-4.1-mini-2025-04-14; do
    python3 m3_online.py --model "$m" --group main --arms amlp --workers 8 &
  done; wait
  echo "$(date -u +%FT%TZ) stage2b hub amlp done" ) > logs/m3_stage2b_hub_amlp.log 2>&1 &
( while pgrep -f "m3_(mine|holdout_monitor).py --model qwen3-8b-local" >/dev/null; do sleep 60; done
  python3 m3_select.py --model qwen3-8b-local
  python3 m3_online.py --model qwen3-8b-local --group main --workers 16
  baselines/agentsentry_env/bin/python m3_online.py --model qwen3-8b-local --group agentsentry --workers 8
  echo "$(date -u +%FT%TZ) stage2b qwen done" ) > logs/m3_stage2b_qwen.log 2>&1 &
wait
