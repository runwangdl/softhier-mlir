#!/usr/bin/env bash
# The H3 sweep (docs/SPATIAL_SPLIT.md): one simulation at a time. Isolated stages on their sets, then the concurrent
# pipelines (traced: utilisation + timeline), then the traced time-shared baseline. `ts` (untraced, every KV element
# dumped: the fp16 twin check) runs first: python tests/gvsoc/split.py run --name ts --mode time --kv-dump all
set -u
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
R="$PY -u tests/gvsoc/split.py run"
declare -A A=( [12_4]=rows:0-2 [8_8]=rows:0-1 [4_12]=rows:0-0 [10_6]=set:0x03ff [6_10]=set:0x003f )
declare -A B=( [12_4]=rows:3-3 [8_8]=rows:2-3 [4_12]=rows:1-3 [10_6]=set:0xfc00 [6_10]=set:0xffc0 )
for s in ${SPLITS:-12_4 8_8 4_12}; do
  $R --name s${s}_A --a ${A[$s]} --b ${B[$s]} --stages A --periods 1 || echo "FAILED s${s}_A"
  $R --name s${s}_B --a ${A[$s]} --b ${B[$s]} --stages B --periods 1 || echo "FAILED s${s}_B"
  $R --name s${s} --a ${A[$s]} --b ${B[$s]} --periods 2 ${TRACE---trace} || echo "FAILED s${s}"
done
