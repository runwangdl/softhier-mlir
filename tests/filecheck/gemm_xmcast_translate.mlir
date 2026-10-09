// RUN: softhier-translate %s | filecheck %s
// softhier.gemm {xmcast}: the small-M GEMM whose activation crosses HBM once (docs/XPANEL_MCAST.md). One cluster loads
// each X K-panel and multicasts it; every cluster streams only its own column slice of W. A unit `xmcast` picks the
// schedule automatically (whole X multicast once if it fits, else panels); "panel" / "whole" force one. The tile
// attributes keep their sh_gemm_cfg slots: tile_m = rows per block, tile_n = column granule, tile_k = K-panel (0 = auto).
builtin.module {
  func.func @xm() {
    %xn  = softhier.hbm_buffer {offset = 65536 : i32}  : memref<50x720xf16, "hbm_west">
    %w   = softhier.hbm_buffer {offset = 139264 : i32} : memref<720x1600xf16, "hbm_west">
    %qkv = softhier.hbm_buffer {offset = 2445312 : i32} : memref<50x1600xf16, "hbm_west">
    %wd  = softhier.hbm_buffer {offset = 2609152 : i32} : memref<2048x720xf16, "hbm_west">
    %m   = softhier.hbm_buffer {offset = 5558272 : i32} : memref<200x4096xf16, "hbm_west">
    %f2  = softhier.hbm_buffer {offset = 7196672 : i32} : memref<200x720xf16, "hbm_west">
    // CHECK: { sh_gemm_cfg cfg = { .tm = 50, .tn = 0, .tk = 0, .pipeline = 1, .accumulate = 0, .fmt = SH_FP16, .l1_base = 0 };  // X-panel multicast
    // CHECK-NEXT: sh_gemm_xmcast_ex({{.*}}, 50, 1600, 720, 720, 1600, 1600, &cfg, SH_XM_AUTO); }
    softhier.gemm %xn, %w into %qkv {fmt = "fp16", tile_m = 50 : i32, tile_n = 0 : i32, tile_k = 0 : i32, pipeline, xmcast, cluster = -1 : i32} : memref<50x720xf16, "hbm_west">, memref<720x1600xf16, "hbm_west">, memref<50x1600xf16, "hbm_west">
    %ga = softhier.view %m : memref<200x4096xf16, "hbm_west"> -> memref<200x2048xf16, strided<[4096, 1], offset: 0>, "hbm_west">
    // CHECK: sh_gemm_xmcast_ex({{.*}}, 200, 720, 2048, 4096, 720, 720, &cfg, SH_XM_PANEL); }
    softhier.gemm %ga, %wd into %f2 {fmt = "fp16", tile_m = 200 : i32, tile_n = 0 : i32, tile_k = 256 : i32, pipeline, xmcast = "panel", cluster = -1 : i32} : memref<200x2048xf16, strided<[4096, 1], offset: 0>, "hbm_west">, memref<2048x720xf16, "hbm_west">, memref<200x720xf16, "hbm_west">
    // CHECK-NOT: sh_gemm_xmcast
    // CHECK: sh_gemm({{.*}}, SH_ALL); }
    softhier.gemm %xn, %w into %qkv {fmt = "fp16", tile_m = 50 : i32, tile_n = 64 : i32, tile_k = 720 : i32, pipeline, cluster = -1 : i32} : memref<50x720xf16, "hbm_west">, memref<720x1600xf16, "hbm_west">, memref<50x1600xf16, "hbm_west">
    func.return
  }
}
