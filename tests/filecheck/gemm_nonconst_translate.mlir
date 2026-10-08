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
// CHECK: sh_test_fill_colparity_fp16(hb1, 512, 512, 512, 15360u, 14336u)
// CHECK: sh_gemm(hb1, hb2, hb3, 512, 512, 512, 512, 512, 512, &cfg, 0)
// CHECK: sh_test_check_const_fp16(hb3, 512, 512, 512, 14848u
