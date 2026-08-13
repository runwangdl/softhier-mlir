// RUN: softhier-translate %s | filecheck %s
// Non-uniform column-parity fill + tiled GEMM + verify.
builtin.module {
  func.func @gemm_nonconst() {
    %xh = softhier.hbm_buffer {offset = 0 : i32}       : memref<512x512xf16, "hbm_west">
    %wh = softhier.hbm_buffer {offset = 524288 : i32}  : memref<512x512xf16, "hbm_south">
    %zh = softhier.hbm_buffer {offset = 1048576 : i32} : memref<512x512xf16, "hbm_west">
    softhier.hbm_fill_col_parity %xh {even_bits = 15360 : i32, odd_bits = 14336 : i32} : memref<512x512xf16, "hbm_west">
    softhier.hbm_fill %wh {value_bits = 6144 : i32} : memref<512x512xf16, "hbm_south">
    softhier.gemm %xh, %wh into %zh {fmt = "fp16"} : memref<512x512xf16, "hbm_west">, memref<512x512xf16, "hbm_south">, memref<512x512xf16, "hbm_west">
    softhier.hbm_check_const %zh {value_bits = 14848 : i32, tol = 16 : i32} : memref<512x512xf16, "hbm_west">
    func.return
  }
}
// CHECK: ({{i[0-9]+}} % 2 == 0) ? 15360u : 14336u
// CHECK: GEMM 512x512x512
// CHECK: GEMM_PASS
