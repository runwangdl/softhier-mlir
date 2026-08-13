#!/usr/bin/env bash
# Run every FileCheck test by executing its own `// RUN:` line (lit-style).
# %s is substituted with the test file; softhier-opt/translate/filecheck come
# from the venv on PATH.
set -u
cd "$(dirname "$0")/.."
export PATH="$PWD/.venv/bin:$PATH"
pass=0
fail=0
for f in tests/filecheck/*.mlir; do
  runline=$(grep -m1 '// RUN:' "$f" | sed 's|.*// RUN: *||')
  cmd=${runline//%s/$f}
  if bash -c "$cmd" >/dev/null 2>&1; then
    echo "PASS $(basename "$f")"
    pass=$((pass + 1))
  else
    echo "FAIL $(basename "$f")"
    fail=$((fail + 1))
  fi
done
echo "=== $pass passed, $fail failed ==="
[ "$fail" -eq 0 ]
