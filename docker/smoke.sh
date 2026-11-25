#!/usr/bin/env bash
# Smoke test of the whole stack: compiler tests, then one GEMM and the MLIR examples on gvsoc.
#   smoke          compiler + library GEMM + two examples
#   smoke full     + every example + one SigLIP encoder layer through the compiler
set -e
cd /app/softhier-mlir
mkdir -p .venv && [ -e .venv/bin ] || ln -s /opt/venv/bin .venv/bin   # scripts expect ./.venv/bin
echo "== 1. FileCheck (compiler only)";            bash tests/run_filecheck.sh | tail -3
echo "== 2. unit tests";                           python -m pytest -q tests/test_preload.py tests/test_testdata.py | tail -2
echo "== 3. softhier-ops GEMM on gvsoc";           python tests/gvsoc/run.py gemm
echo "== 4. linalg -> softhier -> C -> gvsoc";     python tests/gvsoc/run.py mlir examples/gemm512_linalg.mlir examples/mlp_linalg.mlir
if [ "$1" = "full" ]; then
  echo "== 5. every example";                      python tests/gvsoc/run.py mlir examples/*.mlir
  echo "== 6. SigLIP layer, seq 256, 16 clusters"; python tests/gvsoc/run.py siglip-mlir --seq 256 --cluster all
fi
echo "smoke: done"
