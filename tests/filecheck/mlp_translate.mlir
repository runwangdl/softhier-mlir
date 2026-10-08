// RUN: softhier-translate %s | filecheck %s
// The 2-layer MLP must emit two chained RedMule GEMMs (layer-1 output at 393216
// feeds layer-2 as the X operand), a ReLU, zeroed accumulators, and the verify.
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
    softhier.l1_zero %y1l : memref<256x256xf16, "tcdm">
    softhier.redmule %xl, %w1l into %y1l {fmt = "fp16"} : memref<256x256xf16, "tcdm">, memref<256x256xf16, "tcdm">, memref<256x256xf16, "tcdm">
    softhier.relu %y1l : memref<256x256xf16, "tcdm">
    softhier.l1_zero %zl : memref<256x256xf16, "tcdm">
    softhier.redmule %y1l, %w2l into %zl {fmt = "fp16"} : memref<256x256xf16, "tcdm">, memref<256x256xf16, "tcdm">, memref<256x256xf16, "tcdm">
    softhier.check_const %zl {value_bits = 15360 : i32, tol = 8 : i32} : memref<256x256xf16, "tcdm">
    softhier.dma_2d %zl -> %zh {size = 131072 : i32, dst_stride = 0 : i32, src_stride = 0 : i32, repeat = 1 : i32} : memref<256x256xf16, "tcdm"> -> memref<256x256xf16, "hbm_west">
    func.return
  }
}
// CHECK: sh_redmule(l1b2, l1b3, l1b5, 256, 256, 256, SH_FP16)
// CHECK: sh_l1_relu_fp16(l1b5, 65536)
// CHECK: sh_redmule(l1b5, l1b4, l1b6, 256, 256, 256, SH_FP16)
// CHECK: sh_test_check_const_l1_fp16(l1b6, 65536, 15360u, 8, "MLP")
