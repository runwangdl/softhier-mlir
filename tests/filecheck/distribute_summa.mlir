// RUN: softhier-opt %s -p distribute-summa | filecheck %s
// distribute-summa marks softhier.gemm for the mesh-wide SUMMA schedule.
builtin.module {
  func.func @g() {
    %xh = softhier.hbm_buffer {offset = 0 : i32}       : memref<1024x1024xf16, "hbm_west">
    %wh = softhier.hbm_buffer {offset = 2097152 : i32} : memref<1024x1024xf16, "hbm_south">
    %zh = softhier.hbm_buffer {offset = 4194304 : i32} : memref<1024x1024xf16, "hbm_west">
    softhier.gemm %xh, %wh into %zh {fmt = "fp16"} : memref<1024x1024xf16, "hbm_west">, memref<1024x1024xf16, "hbm_south">, memref<1024x1024xf16, "hbm_west">
    func.return
  }
}
// CHECK: softhier.gemm %xh, %wh into %zh {summa, fmt = "fp16"}
