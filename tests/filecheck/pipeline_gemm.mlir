// RUN: softhier-opt %s -p pipeline-gemm | filecheck %s
// The pipeline-gemm pass marks each softhier.gemm for software pipelining.
builtin.module {
  func.func @g() {
    %xh = softhier.hbm_buffer {offset = 0 : i32}       : memref<512x512xf16, "hbm_west">
    %wh = softhier.hbm_buffer {offset = 524288 : i32}  : memref<512x512xf16, "hbm_south">
    %zh = softhier.hbm_buffer {offset = 1048576 : i32} : memref<512x512xf16, "hbm_west">
    softhier.gemm %xh, %wh into %zh {fmt = "fp16"} : memref<512x512xf16, "hbm_west">, memref<512x512xf16, "hbm_south">, memref<512x512xf16, "hbm_west">
    func.return
  }
}
// CHECK: softhier.gemm %xh, %wh into %zh {pipeline, fmt = "fp16"}
