// 512x512x512 GEMM written in standard linalg.matmul on HBM matrices. The
// size-aware linalg-to-softhier pass lowers it to a tiled softhier.gemm.
builtin.module {
  func.func @gemm512() {
    %xh = softhier.hbm_buffer {offset = 0 : i32}       : memref<512x512xf16, "hbm_west">
    %wh = softhier.hbm_buffer {offset = 524288 : i32}  : memref<512x512xf16, "hbm_south">
    %zh = softhier.hbm_buffer {offset = 1048576 : i32} : memref<512x512xf16, "hbm_west">

    softhier.hbm_fill %xh {value_bits = 15360 : i32} : memref<512x512xf16, "hbm_west">
    softhier.hbm_fill %wh {value_bits = 6144 : i32}  : memref<512x512xf16, "hbm_south">

    linalg.matmul ins(%xh, %wh : memref<512x512xf16, "hbm_west">, memref<512x512xf16, "hbm_south">)
                  outs(%zh : memref<512x512xf16, "hbm_west">)

    softhier.hbm_check_const %zh {value_bits = 15360 : i32, tol = 16 : i32} : memref<512x512xf16, "hbm_west">
    func.return
  }
}
