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
// CHECK: sh_test_fill_const_fp16(hb1, 512, 512, 512, 15360u)
// CHECK: sh_gemm_cfg cfg = { .tm = 0, .tn = 0, .tk = 0, .pipeline = 0, .accumulate = 0, .fmt = SH_FP16, .l1_base = 0 }
// CHECK: sh_gemm(hb1, hb2, hb3, 512, 512, 512, 512, 512, 512, &cfg, 0)
// CHECK: sh_test_check_const_fp16(hb3, 512, 512, 512, 15360u, 16, "GEMM512")
