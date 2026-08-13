// RUN: softhier-translate %s | filecheck %s
// The softhier -> C backend must emit the RedMule config/trigger/wait sequence
// and DMA loads/stores with the generator-assigned L1 offsets.
builtin.module {
  func.func @gemm256() {
    %xh = softhier.hbm_buffer {offset = 0 : i32}      : memref<256x256xf16, "hbm_west">
    %wh = softhier.hbm_buffer {offset = 131072 : i32} : memref<256x256xf16, "hbm_south">
    %zh = softhier.hbm_buffer {offset = 262144 : i32} : memref<256x256xf16, "hbm_west">
    %xl = softhier.l1_buffer : memref<256x256xf16, "tcdm">
    %wl = softhier.l1_buffer : memref<256x256xf16, "tcdm">
    %yl = softhier.l1_buffer : memref<256x256xf16, "tcdm">
    softhier.dma_2d %xh -> %xl {size = 131072 : i32, dst_stride = 0 : i32, src_stride = 0 : i32, repeat = 1 : i32} : memref<256x256xf16, "hbm_west"> -> memref<256x256xf16, "tcdm">
    softhier.dma_2d %wh -> %wl {size = 131072 : i32, dst_stride = 0 : i32, src_stride = 0 : i32, repeat = 1 : i32} : memref<256x256xf16, "hbm_south"> -> memref<256x256xf16, "tcdm">
    softhier.redmule %xl, %wl into %yl {fmt = "fp16"} : memref<256x256xf16, "tcdm">, memref<256x256xf16, "tcdm">, memref<256x256xf16, "tcdm">
    softhier.dma_2d %yl -> %zh {size = 131072 : i32, dst_stride = 0 : i32, src_stride = 0 : i32, repeat = 1 : i32} : memref<256x256xf16, "tcdm"> -> memref<256x256xf16, "hbm_west">
    func.return
  }
}
// CHECK: flex_dma_async_1d(local(0), hbm_addr(0), 131072)
// CHECK: flex_dma_async_1d(local(131072), hbm_addr(131072), 131072)
// CHECK: flex_redmule_config(256, 256, 256)
// CHECK: flex_redmule_trigger(0, 131072, 262144, REDMULE_FP_16)
// CHECK: flex_redmule_wait()
// CHECK: flex_dma_async_1d(hbm_addr(262144), local(262144), 131072)
