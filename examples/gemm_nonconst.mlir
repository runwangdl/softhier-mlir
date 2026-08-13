// 512x512x512 GEMM with a NON-uniform input (beyond the constant trick):
//   X[i,k] = 1.0 if k even, 0.5 if k odd   (fp16 0x3C00 / 0x3800)
//   W[k,j] = 1/512                          (fp16 0x1800 = 6144)
//   Z[i,j] = sum_k X[i,k]/512 = (256*1.0 + 256*0.5)/512 = 384/512 = 0.75
// so Z must be 0.75 (fp16 0x3A00 = 14848). A wrong K-accumulation, or mixing up
// even/odd contributions, gives a different value -- so this checks that the
// GEMM handles position-dependent input values correctly, not just constants.
builtin.module {
  func.func @gemm_nonconst() {
    %xh = softhier.hbm_buffer {offset = 0 : i32}       : memref<512x512xf16, "hbm_west">
    %wh = softhier.hbm_buffer {offset = 524288 : i32}  : memref<512x512xf16, "hbm_south">
    %zh = softhier.hbm_buffer {offset = 1048576 : i32} : memref<512x512xf16, "hbm_west">

    softhier.hbm_fill_col_parity %xh {even_bits = 15360 : i32, odd_bits = 14336 : i32}
        : memref<512x512xf16, "hbm_west">
    softhier.hbm_fill %wh {value_bits = 6144 : i32} : memref<512x512xf16, "hbm_south">

    softhier.gemm %xh, %wh into %zh {fmt = "fp16"}
        : memref<512x512xf16, "hbm_west">, memref<512x512xf16, "hbm_south">, memref<512x512xf16, "hbm_west">

    softhier.hbm_check_const %zh {value_bits = 14848 : i32, tol = 16 : i32} : memref<512x512xf16, "hbm_west">
    func.return
  }
}
