#!/usr/bin/env bash
# M3 stage 2a (hub models): online arms that do not depend on the AMLP configuration (af4, 2026-10-06).
cd "$(dirname "$0")"
ARMS=none,block_all,spotlighting,sandwich,tool_filter,pi_detector,melon,tripwire,progent
run() { local m=$1 t=$2
  python3 m3_online.py --model "$m" --group main --arms "$ARMS" --workers 8
  baselines/camel_env/bin/python m3_online.py --model "$m" --group camel --workers 8
  baselines/agentsentry_env/bin/python m3_online.py --model "$m" --group agentsentry --workers 8
  echo "$(date -u +%FT%TZ) stage2a done $m"; }
run gpt-4o-mini-2024-07-18 > logs/m3_stage2a_4omini.log 2>&1 &
run gpt-4.1-mini-2025-04-14 > logs/m3_stage2a_41mini.log 2>&1 &
wait
