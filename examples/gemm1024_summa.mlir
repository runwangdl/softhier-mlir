// 1024x1024x1024 fp16 GEMM in standard linalg.matmul on HBM.
// With `-p linalg-to-softhier,distribute-summa` it lowers to a mesh-wide SUMMA:
// M=N=1024=4*256 -> a 4x4 grid of output tiles, one per cluster, so all 16
// clusters of arch_NoC512 compute in parallel (each accumulates 4 K-tiles).
// Verifiable: X=1.0 (0x3C00=15360), W=1/1024 (0x1400=5120) -> Z=1.0.
builtin.module {
  func.func @gemm1024() {
    %xh = softhier.hbm_buffer {offset = 0 : i32}       : memref<1024x1024xf16, "hbm_west">
    %wh = softhier.hbm_buffer {offset = 2097152 : i32} : memref<1024x1024xf16, "hbm_south">
    %zh = softhier.hbm_buffer {offset = 4194304 : i32} : memref<1024x1024xf16, "hbm_west">

    softhier.hbm_fill %xh {value_bits = 15360 : i32} : memref<1024x1024xf16, "hbm_west">
    softhier.hbm_fill %wh {value_bits = 5120 : i32}  : memref<1024x1024xf16, "hbm_south">

    linalg.matmul ins(%xh, %wh : memref<1024x1024xf16, "hbm_west">, memref<1024x1024xf16, "hbm_south">)
                  outs(%zh : memref<1024x1024xf16, "hbm_west">)

    softhier.hbm_check_const %zh {value_bits = 15360 : i32, tol = 16 : i32} : memref<1024x1024xf16, "hbm_west">
    func.return
  }
}
