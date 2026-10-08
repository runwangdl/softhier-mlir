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
// CHECK: sh_dma_copy((uint64_t)sh_l1_addr(0), hb1, 131072)
// CHECK: sh_dma_copy((uint64_t)sh_l1_addr(131072), hb2, 131072)
// CHECK: sh_redmule(l1b4, l1b5, l1b6, 256, 256, 256, SH_FP16)
// CHECK: sh_dma_copy(hb3, (uint64_t)sh_l1_addr(262144), 131072)
