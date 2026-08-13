// The same 2-layer MLP, but the GEMMs are written in *standard* linalg.matmul.
// `softhier-opt -p linalg-to-softhier` lowers each matmul onto RedMule
// (l1_zero + softhier.redmule); softhier-translate then emits runnable C.
// Only the accelerator-specific IO (fill/relu/check/dma/buffers) stays softhier.
builtin.module {
  func.func @mlp() {
    %zh  = softhier.hbm_buffer {offset = 0 : i32} : memref<256x256xf16, "hbm_west">

    %xl  = softhier.l1_buffer : memref<256x256xf16, "tcdm">
    %w1l = softhier.l1_buffer : memref<256x256xf16, "tcdm">
    %w2l = softhier.l1_buffer : memref<256x256xf16, "tcdm">
    %y1l = softhier.l1_buffer : memref<256x256xf16, "tcdm">
    %zl  = softhier.l1_buffer : memref<256x256xf16, "tcdm">

    softhier.l1_fill %xl  {value_bits = 15360 : i32} : memref<256x256xf16, "tcdm">
    softhier.l1_fill %w1l {value_bits = 7168 : i32}  : memref<256x256xf16, "tcdm">
    softhier.l1_fill %w2l {value_bits = 7168 : i32}  : memref<256x256xf16, "tcdm">

    // layer 1: Y1 = ReLU(X @ W1)
    linalg.matmul ins(%xl, %w1l : memref<256x256xf16, "tcdm">, memref<256x256xf16, "tcdm">)
                  outs(%y1l : memref<256x256xf16, "tcdm">)
    softhier.relu %y1l : memref<256x256xf16, "tcdm">

    // layer 2: Z = Y1 @ W2
    linalg.matmul ins(%y1l, %w2l : memref<256x256xf16, "tcdm">, memref<256x256xf16, "tcdm">)
                  outs(%zl : memref<256x256xf16, "tcdm">)

    softhier.check_const %zl {value_bits = 15360 : i32, tol = 8 : i32} : memref<256x256xf16, "tcdm">
    softhier.dma_2d %zl -> %zh {size = 131072 : i32, dst_stride = 0 : i32, src_stride = 0 : i32, repeat = 1 : i32}
        : memref<256x256xf16, "tcdm"> -> memref<256x256xf16, "hbm_west">

    func.return
  }
}
