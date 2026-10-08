// RUN: softhier-translate %s | filecheck %s
// scf.for over index values -> C for loops; indexed hbm_buffer / view address offset + i * stride;
// indexed mark / dump_samples append the index to their tag.
builtin.module {
  func.func @looped() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    %x = softhier.hbm_buffer {offset = 65536 : i32} : memref<256x768xf16, "hbm_west">
    %o = softhier.hbm_buffer {offset = 458752 : i32} : memref<256x768xf16, "hbm_west">
    scf.for %L = %c0 to %c3 step %c1 {
      %L1 = arith.addi %L, %c1 : index
      %w = softhier.hbm_buffer %L {offset = 1048576 : i32, stride = 1179648 : i32} : memref<768x768xf16, "hbm_west">
      softhier.gemm %x, %w into %o {fmt = "fp16", tile_m = 256 : i32, tile_n = 256 : i32, tile_k = 256 : i32, pipeline, cluster = -1 : i32} : memref<256x768xf16, "hbm_west">, memref<768x768xf16, "hbm_west">, memref<256x768xf16, "hbm_west">
      scf.for %h = %c0 to %c2 step %c1 {
        %oh = softhier.view %o, %h {stride = 64 : i32} : memref<256x768xf16, "hbm_west"> -> memref<256x64xf16, strided<[768, 1], offset: 0>, "hbm_west">
        softhier.gelu %oh -> %oh {cluster = -1 : i32} : memref<256x64xf16, strided<[768, 1], offset: 0>, "hbm_west"> -> memref<256x64xf16, strided<[768, 1], offset: 0>, "hbm_west">
      }
      softhier.mark %L1 {tag = "layer"}
      softhier.dump_samples %o, %L1 {seed = 7 : i32, n = 4 : i32, tag = "L"} : memref<256x768xf16, "hbm_west">
    }
    func.return
  }
}
// CHECK: static void looped(void)
// CHECK: const uint64_t hb1 = sh_hbm_addr(65536);
// CHECK: const uint64_t hb2 = sh_hbm_addr(458752);
// CHECK: for (uint32_t i1 = 0; i1 < 3; i1 += 1) {
// CHECK-NEXT: const uint64_t hb3 = sh_hbm_addr((uint64_t)1048576 + (uint64_t)(i1) * 1179648u);
// CHECK: sh_gemm(hb1, hb3, hb2, 256, 768, 768, 768, 768, 768, &cfg, SH_ALL); }
// CHECK-NEXT: for (uint32_t i2 = 0; i2 < 2; i2 += 1) {
// CHECK-NEXT: sh_gelu((hb2 + (uint64_t)(i2) * 128u), (hb2 + (uint64_t)(i2) * 128u), 256, 64, 768, SH_ALL);
// CHECK-NEXT: }
// CHECK-NEXT: if (sh_cluster_id() == 0 && sh_is_first_core()) sh_printf("[mark] %s%u %u\n", "layer", (uint32_t)((i1 + 1)), sh_cycles());
// CHECK-NEXT: if (sh_cluster_id() == 0 && sh_is_first_core()) sh_test_dump_samples_idx(hb2, 256, 768, 768, 7, 4, "L", (uint32_t)((i1 + 1)));
// CHECK-NEXT: }
