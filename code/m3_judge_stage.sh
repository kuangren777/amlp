#!/usr/bin/env bash
# M3 judging (af4): holdout target for all 4 models now (A3 monitor runs complete); attack target for each model once
# its online `none` arm is complete. No interim look: per-run flags only; the tests run once via m3_judge.py --test.
cd "$(dirname "$0")"
for m in gpt-4o-mini-2024-07-18 gpt-4.1-mini-2025-04-14 llama31-8b-local qwen3-8b-local; do
  ( python3 m3_judge.py --model "$m" --target holdout --workers 8
    if [ "$m" = qwen3-8b-local ]; then while pgrep -f "m3_online.py --model qwen3-8b-local" >/dev/null; do sleep 120; done; fi
    python3 m3_judge.py --model "$m" --target attack --workers 8
    echo "$(date -u +%FT%TZ) judge done $m" ) > "logs/m3_judge_${m%%-2*}.log" 2>&1 &
done
wait
