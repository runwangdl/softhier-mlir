# softhier-mlir tutorial

Two things, hands-on:
1. **How to generate & run** — take a program in the `softhier` dialect (or in
   standard `linalg`), turn it into C, and run it on GVSoC.
2. **How to add a new op** — walked through with a real op, `softhier.l1_add`,
   that we add from scratch and run.

Everything here is copy-pasteable and was verified on `skylab` against the GVSoC
environment at `/scratch/runwang/softhier/gvsoc`.

---

## 0. Setup (once)

```bash
cd /scratch/runwang/softhier-mlir
python3 -m venv .venv
. .venv/bin/activate
pip install -e ".[dev]"       # installs xdsl + the softhier-opt / softhier-translate tools
```

Two command-line tools get installed:

| Tool | What it does |
| --- | --- |
| `softhier-opt`  | parse / verify / **run passes** on `.mlir` (e.g. `linalg-to-softhier`) |
| `softhier-translate` | lower a `softhier` module to **C** (against the `flex_` runtime) |

---

## 1. The generate-and-run flow

```
 your .mlir  ──softhier-opt -p linalg-to-softhier──▶  softhier dialect
             ──softhier-translate──▶  main.c   (flex_ runtime)
             ──SDK build + GVSoC──▶  runs, self-checks (prints *_PASS / *_FAIL)
```

### Worked example — a 512³ GEMM written in standard `linalg`

`examples/gemm512_linalg.mlir` is a plain `linalg.matmul` on HBM matrices, with
the inputs/verify expressed as `softhier` IO ops. Lower and inspect:

```bash
# 1) lower linalg.matmul onto SoftHier (a large HBM matmul -> tiled softhier.gemm)
softhier-opt examples/gemm512_linalg.mlir -p linalg-to-softhier
```

You'll see the `linalg.matmul` become `softhier.gemm %xh, %wh into %zh {fmt = "fp16"}`.

```bash
# 2) generate C
softhier-opt examples/gemm512_linalg.mlir -p linalg-to-softhier \
  | softhier-translate /dev/stdin -o /tmp/main.c
sed -n '1,40p' /tmp/main.c
```

The C is the real thing: `flex_redmule_config/trigger/wait`, `flex_dma_async_1d`
loads/stores, `flex_intra_cluster_sync`, all wrapped in the cluster-0 kernel.

### Run it on GVSoC

The C drops into the SoftHier SDK as an "app". Put `main.c` + a one-line
`CMakeLists.txt` in a folder under `soft_hier_sdk/generated/`, then build+run:

```bash
GV=/scratch/runwang/softhier/gvsoc
APP=$GV/soft_hier_sdk/generated/mygemm
mkdir -p "$APP"
softhier-opt examples/gemm512_linalg.mlir -p linalg-to-softhier \
  | softhier-translate /dev/stdin -o "$APP/main.c"
cat > "$APP/CMakeLists.txt" <<'EOF'
set(SRC_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/main.c)
set(SOURCES ${SRC_SOURCES} PARENT_SCOPE)
set(INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE)
EOF

cd "$GV"
source env_softhier.sh                                  # ETH gcc-14 / cmake / venv / SystemC / RISC-V
make sh-old-hs cfg=soft_hier_sdk/examples/SoftHier/config/arch_NoC512.py \
     app=soft_hier_sdk/generated/mygemm core_model=fast
./install/bin/gvsoc --target=pulp.chips.soft_hier_old.flex_cluster \
     --binary soft_hier_sdk/sw_build/softhier.elf --core-model=fast \
     run --trace=/chip/cluster_0/redmule
```

Expected: RedMule trace lines (`[LightRedmule] Finished ... GEMM id = ...`) and
the on-device check printing `GEMM_CHECK ok=262144 GEMM_PASS`.

> Tip: long builds/runs — launch under `setsid ... &` so a disconnect can't kill
> them, and `grep` the log for `GEMM_PASS` / `RUN_EXIT`.

### The memory model (what the offsets mean)

- `memref<...xf16, "tcdm">` — a per-cluster L1 (scratchpad) tile. The backend
  **bump-allocates** an offset for each `softhier.l1_buffer`; uses become `local(off)`.
- `memref<...xf16, "hbm_west">` (or `hbm_south/north/east`) — HBM; the byte offset
  comes from the `softhier.hbm_buffer {offset=...}` attribute; uses become `hbm_addr(off)`.
- fp16 constants are given as **raw 16-bit patterns**: `1.0=0x3C00 (15360)`,
  `0.5=0x3800 (14336)`, `0.75=0x3A00 (14848)`, `1.5=0x3E00 (15872)`, `1/256=0x1C00`,
  `1/512=0x1800`.

---

## 2. The pieces (where things live)

```
softhier_mlir/
  dialects/softhier.py        the dialect: op definitions + memory-space conventions
  backend/emit_c.py           softhier ops -> C (the code generator)
  transforms/linalg_to_softhier.py   linalg.matmul -> softhier (a lowering pass)
  tools/softhier_opt.py       the opt driver (registers dialect + passes)
  tools/softhier_translate.py the translate driver (calls emit_c)
tests/filecheck/*.mlir        one FileCheck test per feature (RUN line at the top)
examples/*.mlir               runnable examples
```

To add a capability you usually touch **two** files: define the op in
`dialects/softhier.py`, and teach `backend/emit_c.py` how to emit it. Optionally
add a lowering pattern in `transforms/`.

---

## 3. How to add a new op — `softhier.l1_add` (residual add)

Goal: an elementwise `dst += src` on two TCDM tiles (fp16) — the kind of op you
need for residual connections. Five steps.

### Step 1 — define the op (`softhier_mlir/dialects/softhier.py`)

```python
@irdl_op_definition
class L1AddOp(IRDLOperation):
    """Elementwise add of two TCDM tiles, in place: ``dst += src`` (fp16)."""

    name = "softhier.l1_add"
    src = operand_def(MemRefType)
    dst = operand_def(MemRefType)
    assembly_format = "$src `into` $dst attr-dict `:` type($src) `,` type($dst)"
```

then add `L1AddOp` to the `Dialect(...)` op list at the bottom of the file.

Rules of thumb:
- **operands** are SSA values (buffers/tensors); **`prop_def(...)`** are compile-time
  attributes (ints, strings). `L1AddOp` has two operands, no props.
- If you *do* add props and want them written inside `{...}` in the syntax, put
  `attr-dict` in the `assembly_format` **and** add
  `irdl_options = (ParsePropInAttrDict(),)` — otherwise xDSL errors that the prop
  is "missing from the declarative format". (Note: it must be a **tuple**.)

### Step 2 — teach the backend to emit it (`softhier_mlir/backend/emit_c.py`)

Add a branch in `emit_kernel`'s op loop. The `ew` counter gives you unique C
variable names so multiple copies of the op don't clash:

```python
elif isinstance(op, L1AddOp):
    src_off = bufs.raw_off(op.src)          # resolve operands -> L1 byte offsets
    dst_off = bufs.raw_off(op.dst)
    n = _nelem(bufs.memref(op.dst))         # element count from the memref shape
    a, d, i = f"a{ew}", f"d{ew}", f"i{ew}"
    ew += 1
    b("    if (flex_is_first_core()) {  // dst += src (fp16)")
    b(f"        volatile _Float16 *{a} = (volatile _Float16 *)local({src_off});")
    b(f"        volatile _Float16 *{d} = (volatile _Float16 *)local({dst_off});")
    b(f"        for (int {i} = 0; {i} < {n}; ++{i}) {d}[{i}] += {a}[{i}];")
    b("    }")
    b("    flex_intra_cluster_sync();")
```

Codegen conventions to copy:
- guard compute with `flex_is_first_core()`, DMA with `flex_is_dm_core()`, and end
  with `flex_intra_cluster_sync()` so the three cores in a cluster rendezvous.
- `bufs.raw_off(v)` → the byte offset; `bufs.addr(v)` → `local(off)` or `hbm_addr(off)`;
  `bufs.memref(v)` → the memref type (for shapes / element size).
- fp16 math: for **sign/zero** tricks use `uint16_t` bit ops (see `relu` = clear the
  sign bit; `l1_zero`); for **arithmetic** use `_Float16` (the RISC-V toolchain is
  built with `zfh`), as above.

### Step 3 — a tiny test program (`examples/add_test.mlir`)

```mlir
builtin.module {
  func.func @add_test() {
    %al = softhier.l1_buffer : memref<256x256xf16, "tcdm">
    %bl = softhier.l1_buffer : memref<256x256xf16, "tcdm">
    softhier.l1_fill %al {value_bits = 14336 : i32} : memref<256x256xf16, "tcdm">  // 0.5
    softhier.l1_fill %bl {value_bits = 15360 : i32} : memref<256x256xf16, "tcdm">  // 1.0
    softhier.l1_add %al into %bl : memref<256x256xf16, "tcdm">, memref<256x256xf16, "tcdm">
    softhier.check_const %bl {value_bits = 15872 : i32, tol = 4 : i32} : memref<256x256xf16, "tcdm">  // 1.5
    func.return
  }
}
```

### Step 4 — round-trip, then look at the generated C

```bash
pip install -e . >/dev/null            # re-install so the new op is picked up
softhier-opt examples/add_test.mlir                 # parses+verifies+reprints -> op is registered
softhier-translate examples/add_test.mlir           # see your emitted C
```

### Step 5 — run it on GVSoC (same recipe as §1)

Drop the generated `main.c` into `soft_hier_sdk/generated/add_test/`, `make sh-old-hs`,
run. `check_const` prints `MLP_PASS` if every element of `bl` is `1.5` — i.e. the
add worked. **This exact op was added and verified this way (`MLP_PASS`).**

### Step 6 — lock it in with a FileCheck test (`tests/filecheck/…mlir`)

```mlir
// RUN: softhier-translate %s | filecheck %s
...the module above...
// CHECK: d{{[0-9]+}}[i{{[0-9]+}}] += a{{[0-9]+}}[i{{[0-9]+}}];
// CHECK: MLP_PASS
```

Run the suite:

```bash
for f in tests/filecheck/*.mlir; do
  head -1 "$f" | grep -q -- '-p linalg' \
    && out=$(softhier-opt "$f" -p linalg-to-softhier) \
    || { head -1 "$f" | grep -q softhier-opt \
         && out=$(softhier-opt "$f") \
         || out=$(softhier-translate "$f"); }
  echo "$out" | filecheck "$f" && echo "PASS $f"
done
```

---

## 4. Variations

- **A new op that carries parameters** (like `softhier.hbm_fill {value_bits=...}`):
  add `prop_def(IntegerAttr)` fields + `irdl_options = (ParsePropInAttrDict(),)`,
  and read them in codegen via `op.value_bits.value.data`.
- **A new op that produces results** (like `softhier.cluster_pos -> index, index`):
  use `result_def(<type>)` and `result_types=[...]` when building it.
- **A macro op the backend expands into a loop** (like `softhier.gemm`): emit a C
  `for` nest in the backend branch; keep the RedMule/DMA/barrier order from the
  hand-written `example_one_cluster_gemm.h`.
- **A frontend lowering** (turn a standard-dialect op into softhier): add a
  `RewritePattern` in `transforms/` (see `linalg_to_softhier.py`) and register it
  in `tools/softhier_opt.py` via `register_pass`.

That's the whole loop: **define → emit → test → run**. Reuse mature dialects for
the math (`linalg`, `memref`, …) and keep `softhier` for what only the hardware
does.
