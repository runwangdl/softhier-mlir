"""Host-side checks of the test-input preload path (no simulator): the leapfrog LCG equals the scalar
loop, and `extract_preload` turns a module's fill ops into the same bytes + a sentinel wait."""
import numpy as np
from xdsl.context import Context
from xdsl.dialects import arith, scf
from xdsl.dialects.builtin import Builtin
from xdsl.dialects.func import Func
from xdsl.parser import Parser

from softhier_mlir.backend.emit_c import emit_c
from softhier_mlir.dialects.softhier import SoftHier
from softhier_mlir.sim.preload import SENTINEL_MAGIC, read_preload_elf, sentinel_array
from softhier_mlir.sim.testdata import extract_preload, write_preload
from softhier_mlir.testing import lcg


def _parse(text: str):
    ctx = Context()
    for d in (Builtin, Func, scf.Scf, arith.Arith, SoftHier):
        ctx.load_dialect(d)
    return Parser(ctx, text).parse_module()


def _module(x_off: int = 0x1000) -> str:
    return f'''builtin.module {{
  func.func @t() {{
    %x = softhier.hbm_buffer {{offset = {x_off} : i32}} : memref<64x96xf16, "hbm_west">
    %w = softhier.hbm_buffer {{offset = 0x10000 : i32}} : memref<96x32xf16, "hbm_west">
    %p = softhier.hbm_buffer {{offset = 0x20000 : i32}} : memref<8x16xf16, "hbm_west">
    %z = softhier.hbm_buffer {{offset = 0x40000 : i32}} : memref<64x32xf16, "hbm_west">
    softhier.hbm_fill_lcg %x {{seed = 7 : i32, lo = -16 : i32, hi = 16 : i32, scale = 0.125 : f32}} : memref<64x96xf16, "hbm_west">
    softhier.hbm_fill %w {{value_bits = 15360 : i32}} : memref<96x32xf16, "hbm_west">
    softhier.hbm_fill_col_parity %p {{even_bits = 15360 : i32, odd_bits = 14336 : i32}} : memref<8x16xf16, "hbm_west">
    softhier.gemm %x, %w into %z {{fmt = "fp16", tile_m = 64 : i32, tile_n = 32 : i32, tile_k = 96 : i32}} : memref<64x96xf16, "hbm_west">, memref<96x32xf16, "hbm_west">, memref<64x32xf16, "hbm_west">
    softhier.dump_samples %z {{seed = 3 : i32, n = 8 : i32, tag = "Z"}} : memref<64x32xf16, "hbm_west">
    func.return
  }}
}}
'''


def test_leapfrog_lcg_matches_scalar_loop():
    for seed, n in [(1, 1), (3, 4095), (3, 4096), (3, 4097), (11, 20000)]:
        assert (lcg._lcg_stream(seed, n) == lcg._lcg_stream_reference(seed, n)).all()
    assert lcg._lcg_stream(5, 0).shape == (0,)


def test_extract_preload_rewrites_module(tmp_path):
    m = _parse(_module())
    arrays = extract_preload(m, verbose=False)
    assert arrays is not None
    sent_off = max(arrays)
    assert sent_off == 0x41000 and arrays[sent_off].tobytes() == sentinel_array().tobytes()   # above z (0x40000 + 4 KB)
    assert (arrays[0x1000] == lcg.fill_fp16(64, 96, 7, -16, 16, 0.125)).all()
    assert arrays[0x10000].view(np.uint16).min() == 15360 and arrays[0x10000].view(np.uint16).max() == 15360
    p = arrays[0x20000].view(np.uint16)
    assert (p[:, ::2] == 15360).all() and (p[:, 1::2] == 14336).all()
    c = emit_c(m)
    assert "sh_test_fill" not in c and "sh_preload_wait(" in c
    assert c.index("sh_preload_wait(") < c.index("sh_gemm(")
    elf = write_preload(tmp_path / "p.elf", arrays)
    back = read_preload_elf(elf)
    assert set(back) == set(arrays)
    assert back[sent_off] == np.full(16, SENTINEL_MAGIC, np.uint32).tobytes()


def test_extract_preload_falls_back_below_allocator_region():
    m = _parse(_module(x_off=0))
    before = emit_c(m)
    assert extract_preload(m, verbose=False) is None
    assert emit_c(m) == before and before.count("sh_test_fill") == 3


def test_extract_preload_without_fills_is_a_noop():
    m = _parse('''builtin.module {
  func.func @t() {
    %z = softhier.hbm_buffer {offset = 0x1000 : i32} : memref<8x8xf16, "hbm_west">
    softhier.dump_samples %z {seed = 3 : i32, n = 8 : i32, tag = "Z"} : memref<8x8xf16, "hbm_west">
    func.return
  }
}
''')
    assert extract_preload(m, verbose=False) == {}
    assert "sh_preload_wait" not in emit_c(m)
