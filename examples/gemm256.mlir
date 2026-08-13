// A single 256x256x256 fp16 GEMM tile on one cluster, in the softhier dialect.
// `softhier-translate` lowers this to a runnable main.c against the flex_ runtime.
//
//   Z[256x256] = X[256x256] @ W[256x256]      (RedMule, fp16)
//
// HBM byte offsets: X@0, W@131072, Z@262144 (256*256*2 = 131072 bytes/tile).
builtin.module {
  func.func @gemm256() {
    %xh = softhier.hbm_buffer {offset = 0 : i32}      : memref<256x256xf16, "hbm_west">
    %wh = softhier.hbm_buffer {offset = 131072 : i32} : memref<256x256xf16, "hbm_south">
    %zh = softhier.hbm_buffer {offset = 262144 : i32} : memref<256x256xf16, "hbm_west">

    %xl = softhier.l1_buffer : memref<256x256xf16, "tcdm">
    %wl = softhier.l1_buffer : memref<256x256xf16, "tcdm">
    %yl = softhier.l1_buffer : memref<256x256xf16, "tcdm">

    // stage X and W tiles from HBM into TCDM
    softhier.dma_2d %xh -> %xl {size = 131072 : i32, dst_stride = 0 : i32, src_stride = 0 : i32, repeat = 1 : i32}
        : memref<256x256xf16, "hbm_west"> -> memref<256x256xf16, "tcdm">
    softhier.dma_2d %wh -> %wl {size = 131072 : i32, dst_stride = 0 : i32, src_stride = 0 : i32, repeat = 1 : i32}
        : memref<256x256xf16, "hbm_south"> -> memref<256x256xf16, "tcdm">

    // Y = X @ W on RedMule
    softhier.redmule %xl, %wl into %yl {fmt = "fp16"}
        : memref<256x256xf16, "tcdm">, memref<256x256xf16, "tcdm">, memref<256x256xf16, "tcdm">

    // store the result tile back to HBM
    softhier.dma_2d %yl -> %zh {size = 131072 : i32, dst_stride = 0 : i32, src_stride = 0 : i32, repeat = 1 : i32}
        : memref<256x256xf16, "tcdm"> -> memref<256x256xf16, "hbm_west">

    func.return
  }
}
