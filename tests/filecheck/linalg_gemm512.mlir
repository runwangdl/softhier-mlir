// RUN: softhier-opt %s -p linalg-to-softhier | filecheck %s
// A large linalg.matmul on HBM operands lowers to a tiled softhier.gemm
// (not the single-tile redmule path).
builtin.module {
  func.func @g() {
    %xh = softhier.hbm_buffer {offset = 0 : i32}       : memref<512x512xf16, "hbm_west">
    %wh = softhier.hbm_buffer {offset = 524288 : i32}  : memref<512x512xf16, "hbm_south">
    %zh = softhier.hbm_buffer {offset = 1048576 : i32} : memref<512x512xf16, "hbm_west">
    linalg.matmul ins(%xh, %wh : memref<512x512xf16, "hbm_west">, memref<512x512xf16, "hbm_south">) outs(%zh : memref<512x512xf16, "hbm_west">)
    func.return
  }
}
// CHECK:     softhier.gemm %xh, %wh into %zh {fmt = "fp16"}
// CHECK-NOT: linalg.matmul
// CHECK-NOT: softhier.redmule
