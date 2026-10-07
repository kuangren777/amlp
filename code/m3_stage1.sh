#!/usr/bin/env bash
# M3 stage 1 (af-man approved 2026-10-06): per model, mining then the A3 holdout monitor runs, sequential so that
# each local vLLM port carries one af4 client process (vllm-A <= 16 guarded inside the runner, vllm-B <= 8).
cd "$(dirname "$0")"
chain() { local m=$1 w=$2
  python3 m3_mine.py --model "$m" --workers "$w" && python3 m3_holdout_monitor.py --model "$m" --workers "$w"
  echo "$(date -u +%FT%TZ) stage1 done $m exit $?"; }
mkdir -p logs
chain qwen3-8b-local 16          > logs/m3_stage1_qwen.log 2>&1 &
chain llama31-8b-local 8         > logs/m3_stage1_llama.log 2>&1 &
chain gpt-4o-mini-2024-07-18 8   > logs/m3_stage1_4omini.log 2>&1 &
chain gpt-4.1-mini-2025-04-14 8  > logs/m3_stage1_41mini.log 2>&1 &
wait
