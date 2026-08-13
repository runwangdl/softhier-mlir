// A 2-layer MLP tile on one cluster:  Z = ReLU(X @ W1) @ W2
// (all 256x256 fp16 tiles). Chosen so the result is exactly verifiable:
//   X    = 1.0      (fp16 0x3C00 = 15360)
//   W1   = 1/256    (fp16 0x1C00 = 7168)  -> Y1 = X@W1 = 256*(1/256) = 1.0
//   ReLU(1.0) = 1.0
//   W2   = 1/256                          -> Z  = Y1@W2 = 1.0
// so every output element must be 1.0. Inputs are filled on-chip; the output
// is self-checked (prints MLP_PASS / MLP_FAIL) and also stored to HBM.
builtin.module {
  func.func @mlp() {
    %zh  = softhier.hbm_buffer {offset = 0 : i32} : memref<256x256xf16, "hbm_west">

    %xl  = softhier.l1_buffer : memref<256x256xf16, "tcdm">
    %w1l = softhier.l1_buffer : memref<256x256xf16, "tcdm">
    %w2l = softhier.l1_buffer : memref<256x256xf16, "tcdm">
    %y1l = softhier.l1_buffer : memref<256x256xf16, "tcdm">
    %zl  = softhier.l1_buffer : memref<256x256xf16, "tcdm">

    // inputs
    softhier.l1_fill %xl  {value_bits = 15360 : i32} : memref<256x256xf16, "tcdm">
    softhier.l1_fill %w1l {value_bits = 7168 : i32}  : memref<256x256xf16, "tcdm">
    softhier.l1_fill %w2l {value_bits = 7168 : i32}  : memref<256x256xf16, "tcdm">

    // layer 1:  Y1 = ReLU(X @ W1)
    softhier.l1_zero %y1l : memref<256x256xf16, "tcdm">
    softhier.redmule %xl, %w1l into %y1l {fmt = "fp16"}
        : memref<256x256xf16, "tcdm">, memref<256x256xf16, "tcdm">, memref<256x256xf16, "tcdm">
    softhier.relu %y1l : memref<256x256xf16, "tcdm">

    // layer 2:  Z = Y1 @ W2
    softhier.l1_zero %zl : memref<256x256xf16, "tcdm">
    softhier.redmule %y1l, %w2l into %zl {fmt = "fp16"}
        : memref<256x256xf16, "tcdm">, memref<256x256xf16, "tcdm">, memref<256x256xf16, "tcdm">

    // verify Z == 1.0 (0x3C00) and store it out
    softhier.check_const %zl {value_bits = 15360 : i32, tol = 8 : i32} : memref<256x256xf16, "tcdm">
    softhier.dma_2d %zl -> %zh {size = 131072 : i32, dst_stride = 0 : i32, src_stride = 0 : i32, repeat = 1 : i32}
        : memref<256x256xf16, "tcdm"> -> memref<256x256xf16, "hbm_west">

    func.return
  }
}
