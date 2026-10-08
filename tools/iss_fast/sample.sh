#!/bin/bash
# usage: sample.sh <pid> <nsamples> <out>  -- gdb stack sampling of all threads
pid=$1; n=$2; out=$3
: > $out
got=0
for i in $(seq 1 $((n * 3))); do
  [ $got -ge $n ] && break
  kill -0 $pid 2>/dev/null || break
  s=$(gdb -p $pid -batch -ex "thread apply all bt 14" 2>/dev/null | grep -E "^#|^Thread")
  if [ -n "$s" ]; then
    echo "$s" >> $out
    echo "----" >> $out
    got=$((got + 1))
  fi
  sleep 0.2
done
