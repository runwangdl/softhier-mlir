// RUN: softhier-translate %s | filecheck %s
// A >256 GEMM must expand into a tile loop-nest with K-accumulation.
builtin.module {
  func.func @gemm512() {
    %xh = softhier.hbm_buffer {offset = 0 : i32}       : memref<512x512xf16, "hbm_west">
    %wh = softhier.hbm_buffer {offset = 524288 : i32}  : memref<512x512xf16, "hbm_south">
    %zh = softhier.hbm_buffer {offset = 1048576 : i32} : memref<512x512xf16, "hbm_west">
    softhier.hbm_fill %xh {value_bits = 15360 : i32} : memref<512x512xf16, "hbm_west">
    softhier.hbm_fill %wh {value_bits = 6144 : i32}  : memref<512x512xf16, "hbm_south">
    softhier.gemm %xh, %wh into %zh {fmt = "fp16"} : memref<512x512xf16, "hbm_west">, memref<512x512xf16, "hbm_south">, memref<512x512xf16, "hbm_west">
    softhier.hbm_check_const %zh {value_bits = 15360 : i32, tol = 16 : i32} : memref<512x512xf16, "hbm_west">
    func.return
  }
}
// CHECK: GEMM 512x512x512: 2x2 output tiles, 2 K-steps
// CHECK: for (int r{{[0-9]+}} = 0; r{{[0-9]+}} < 2
// CHECK: for (int k{{[0-9]+}} = 0; k{{[0-9]+}} < 2
// CHECK: flex_redmule_trigger(0, 131072, 262144, REDMULE_FP_16)
// CHECK: GEMM_PASS
