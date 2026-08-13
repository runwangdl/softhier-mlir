// RUN: softhier-opt %s -p linalg-to-softhier | filecheck %s
// linalg.matmul (memref form) must lower to l1_zero + softhier.redmule, and no
// linalg.matmul may remain.
builtin.module {
  func.func @two_gemms() {
    %xl  = softhier.l1_buffer : memref<256x256xf16, "tcdm">
    %w1l = softhier.l1_buffer : memref<256x256xf16, "tcdm">
    %w2l = softhier.l1_buffer : memref<256x256xf16, "tcdm">
    %y1l = softhier.l1_buffer : memref<256x256xf16, "tcdm">
    %zl  = softhier.l1_buffer : memref<256x256xf16, "tcdm">
    linalg.matmul ins(%xl, %w1l : memref<256x256xf16, "tcdm">, memref<256x256xf16, "tcdm">) outs(%y1l : memref<256x256xf16, "tcdm">)
    linalg.matmul ins(%y1l, %w2l : memref<256x256xf16, "tcdm">, memref<256x256xf16, "tcdm">) outs(%zl : memref<256x256xf16, "tcdm">)
    func.return
  }
}
// CHECK-LABEL: func.func @two_gemms
// CHECK:       softhier.l1_zero %y1l
// CHECK-NEXT:  softhier.redmule %xl, %w1l into %y1l {fmt = "fp16"}
// CHECK:       softhier.l1_zero %zl
// CHECK-NEXT:  softhier.redmule %y1l, %w2l into %zl {fmt = "fp16"}
// CHECK-NOT:   linalg.matmul
