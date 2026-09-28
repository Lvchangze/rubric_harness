#!/usr/bin/env bash
# Forensics only: record what the machine looked like when the generation
# processes vanished. It restarts nothing.
#
# The run has been killed three times (09-25 15:57, 09-27 10:29, 09-28 10:47)
# with no OOM record, no reboot, and every other long-running process on the
# node untouched. The last time the generator was in its own session and still
# went at the same moment as the supervisor, so it was not a process-group kill.
# By the time anyone looks, whatever did it is gone. This takes a snapshot
# within a couple of seconds of the disappearance, while a cleanup script or
# the shell that ran it may still be visible.
#
# Run from a copy outside the repository, with cwd /, so it shares as few
# traits as possible with the processes that keep getting killed:
#
#   cp scripts/procwatch.sh /tmp/rh_procwatch.sh
#   (cd / && tmux new-session -d -s procwatch "bash /tmp/rh_procwatch.sh")
set -u
LOG=${LOG:-/tmp/rh_procwatch.log}
INTERVAL=${INTERVAL:-2}

# Matched on argument position rather than substrings: an interactive shell
# whose command text merely mentions these scripts is not one of them.
watched() {
  ps -eo pid=,args= | awk '
    ($2 ~ /python/ && $3 == "scripts/gen_rubrics.py" && index($0, "train_full")) ||
    ($2 == "bash" && ($3 == "scripts/supervise_train_full.sh" || $3 == "scripts/keep_supervisor.sh")) {print $1}'
}

# bash forks subshells for $(...) and pipelines that carry the parent's exact
# arguments and live for seconds, so a disappearance only counts for a process
# that had been alive for MIN_AGE_S. The real targets run for hours.
MIN_AGE_S=${MIN_AGE_S:-60}
declare -A first_seen

echo "$(date '+%F %T') procwatch started (pid $$)" >> "$LOG"
while true; do
  now=$(watched | sort -n | tr '\n' ' ')
  t=$(date +%s)
  gone=""
  for p in "${!first_seen[@]}"; do
    case " $now " in
      *" $p "*) ;;
      *) [ $(( t - first_seen[$p] )) -ge "$MIN_AGE_S" ] && gone="$gone $p"
         unset "first_seen[$p]" ;;
    esac
  done
  for p in $now; do
    [ -z "${first_seen[$p]:-}" ] && first_seen[$p]=$t
  done
  if [ -n "$gone" ]; then
    {
      echo "================================================================"
      echo "$(date '+%F %T') vanished:$gone   still watched: ${now:-none}"
      echo "--- processes started in the last 10 minutes ---"
      ps -eo pid,ppid,user,etimes,lstart,args --sort=start_time | awk 'NR==1 || $4 < 600' | cut -c1-240
      echo "--- full process list ---"
      ps -eo pid,ppid,user,etime,args | cut -c1-240
      echo "--- who ---"
      who 2>&1
    } >> "$LOG"
  fi
  sleep "$INTERVAL"
done
