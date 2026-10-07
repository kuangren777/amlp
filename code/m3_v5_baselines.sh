#!/usr/bin/env bash
# v5 (plan.md §13): baselines to 3 reps on the holdout. One writer per rows file: main then agentsentry per model;
# camel (own file) in parallel on hub models. llama: part c (benign) only.
cd "$(dirname "$0")"
B=block_all,spotlighting,sandwich,tool_filter,pi_detector,melon,tripwire,progent
hub() { local m=$1
  ( python3 m3_online.py --model "$m" --group main --arms $B --baseline-reps 3 --workers 8
    baselines/agentsentry_env/bin/python m3_online.py --model "$m" --group agentsentry --baseline-reps 3 --workers 8 ) &
  baselines/camel_env/bin/python m3_online.py --model "$m" --group camel --baseline-reps 3 --workers 8 &
  wait; echo "$(date -u +%FT%TZ) v5 baselines done $m"; }
hub gpt-4o-mini-2024-07-18 > logs/m3_v5_4omini.log 2>&1 &
hub gpt-4.1-mini-2025-04-14 > logs/m3_v5_41mini.log 2>&1 &
( python3 m3_online.py --model llama31-8b-local --group main --arms $B --parts c --baseline-reps 3 --workers 8
  baselines/agentsentry_env/bin/python m3_online.py --model llama31-8b-local --group agentsentry --parts c --baseline-reps 3 --workers 8
  echo "$(date -u +%FT%TZ) v5 baselines done llama" ) > logs/m3_v5_llama.log 2>&1 &
( python3 m3_online.py --model qwen3-8b-local --group main --arms $B --baseline-reps 3 --workers 16
  baselines/agentsentry_env/bin/python m3_online.py --model qwen3-8b-local --group agentsentry --baseline-reps 3 --workers 8
  echo "$(date -u +%FT%TZ) v5 baselines done qwen" ) > logs/m3_v5_qwen.log 2>&1 &
wait
