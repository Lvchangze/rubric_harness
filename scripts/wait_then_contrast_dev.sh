#!/usr/bin/env bash
cd /apdcephfs_zwfy6/share_302970870/hunyuan/changzelv/dev/rubric_harness
loop_alive() { ps -eo comm,args | awk '$1 ~ /^python/ && /rubricbench_evolve_loop/ {f=1} END {exit !f}'; }
echo "$(date '+%F %T') waiting for rubricbench_evolve_loop to exit"
quiet=0
while [ $quiet -lt 3 ]; do
  if loop_alive; then quiet=0; else quiet=$((quiet + 1)); fi
  sleep 60
done
echo "$(date '+%F %T') loop gone for 3 min; starting agentic-tools-contrast on dev"
SOURCES=agentic-tools-contrast bash scripts/run_rubricbench_glm53.sh
