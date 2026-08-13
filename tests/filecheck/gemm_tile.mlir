// RUN: softhier-opt %s | filecheck %s
//
// One cluster's inner step of an output-stationary SUMMA GEMM tile, expressed
// with the `softhier` dialect. The diagonal cluster's HBM load + row/col
// broadcast, the in-place RedMule accumulate over the K loop, and the group
// barrier are all visible; only the accelerator specifics are `softhier` ops —
// the surrounding loop/control would be `scf`/`arith` in the real flow.

// CHECK-LABEL: func.func @summa_gemm_tile
func.func @summa_gemm_tile(
    %x_hbm: memref<128x128xf16, "hbm_west">,
    %w_hbm: memref<128x128xf16, "hbm_south">,
    %x_l1: memref<128x128xf16, "tcdm">,
    %w_l1: memref<128x128xf16, "tcdm">,
    %y_l1: memref<128x128xf16, "tcdm">) {

  // diagonal cluster stages the K-panels from the HBM edges into TCDM
  // CHECK: softhier.dma_2d
  softhier.dma_2d %x_hbm -> %x_l1 {size = 256 : i32, dst_stride = 256 : i32, src_stride = 1024 : i32, repeat = 128 : i32}
      : memref<128x128xf16, "hbm_west"> -> memref<128x128xf16, "tcdm">
  softhier.dma_2d %w_hbm -> %w_l1 {size = 256 : i32, dst_stride = 256 : i32, src_stride = 1024 : i32, repeat = 128 : i32}
      : memref<128x128xf16, "hbm_south"> -> memref<128x128xf16, "tcdm">

  // broadcast X along the mesh row, W along the mesh column (SUMMA)
  // CHECK: softhier.dma_broadcast
  softhier.dma_broadcast %x_l1 -> %x_l1 {row_mask = 15 : i32, col_mask = 0 : i32}
      : memref<128x128xf16, "tcdm"> -> memref<128x128xf16, "tcdm">
  softhier.dma_broadcast %w_l1 -> %w_l1 {row_mask = 0 : i32, col_mask = 15 : i32}
      : memref<128x128xf16, "tcdm"> -> memref<128x128xf16, "tcdm">

  // in-place accumulate this K-step:  y += x @ w
  // CHECK: softhier.redmule {{.*}} into {{.*}} {fmt = "fp16"}
  softhier.redmule %x_l1, %w_l1 into %y_l1 {fmt = "fp16"}
      : memref<128x128xf16, "tcdm">, memref<128x128xf16, "tcdm">, memref<128x128xf16, "tcdm">

  // synchronize the 4x4 cooperating group
  // CHECK: softhier.group_barrier
  softhier.group_barrier {grid_x = 4 : i32, grid_y = 4 : i32}

  func.return
}
