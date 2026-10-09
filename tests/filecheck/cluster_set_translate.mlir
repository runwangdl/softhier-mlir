// RUN: softhier-translate %s | filecheck %s
// Spatial split (docs/SPATIAL_SPLIT.md): a `cluster_set` mask attribute deals an op over that set of clusters
// (SH_GROUP(mask)), `softhier.on_clusters` runs its body on the set only, so two regions over disjoint sets run at
// the same time; marks / dumps / barriers with `cluster_set` are printed by / scoped to the set.
builtin.module {
  func.func @split() {
    %x = softhier.hbm_buffer {offset = 65536 : i32} : memref<256x768xf16, "hbm_west">
    %w = softhier.hbm_buffer {offset = 458752 : i32} : memref<768x768xf16, "hbm_west">
    %z = softhier.hbm_buffer {offset = 1638400 : i32} : memref<256x768xf16, "hbm_west">
    %g = softhier.hbm_buffer {offset = 2031616 : i32} : memref<1x768xf16, "hbm_west">
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    scf.for %p = %c0 to %c3 step %c1 {
      softhier.on_clusters attributes {cluster_set = 4095 : i32} {
        softhier.gemm %x, %w into %z {fmt = "fp16", tile_m = 256 : i32, tile_n = 256 : i32, tile_k = 256 : i32, pipeline, cluster = -1 : i32, cluster_set = 4095 : i32} : memref<256x768xf16, "hbm_west">, memref<768x768xf16, "hbm_west">, memref<256x768xf16, "hbm_west">
        softhier.mark %p {tag = "a", cluster_set = 4095 : i32}
      }
      softhier.on_clusters attributes {cluster_set = 61440 : i32} {
        softhier.rmsnorm %x, %g -> %x {eps = 1.0e-5 : f32, cluster_set = 61440 : i32} : memref<256x768xf16, "hbm_west">, memref<1x768xf16, "hbm_west"> -> memref<256x768xf16, "hbm_west">
        softhier.group_barrier {cluster_set = 61440 : i32}
        softhier.mark %p {tag = "b", cluster_set = 61440 : i32}
      }
      softhier.group_barrier {grid_x = 4 : i32, grid_y = 4 : i32}
      softhier.mark %p {tag = "period"}
    }
    func.return
  }
}
// CHECK: static void split(void)
// CHECK: for (uint32_t i1 = 0; i1 < 3; i1 += 1) {
// CHECK-NEXT: if (sh_set_member(SH_GROUP(0x0fffu))) {   // spatial split: this set only
// CHECK-NEXT: sh_gemm_cfg cfg
// CHECK-NEXT: sh_gemm(hb1, hb2, hb3, 256, 768, 768, 768, 768, 768, &cfg, SH_GROUP(0x0fffu)); }
// CHECK-NEXT: if (sh_cluster_id() == sh_set_leader(SH_GROUP(0x0fffu)) && sh_is_first_core()) sh_printf("[mark] %s%u %u\n", "a", (uint32_t)(i1), sh_cycles());
// CHECK-NEXT: }
// CHECK-NEXT: if (sh_set_member(SH_GROUP(0xf000u))) {
// CHECK-NEXT: sh_rmsnorm(hb1, hb1, hb4, 256, 768, 768, {{.*}}, SH_GROUP(0xf000u));
// CHECK-NEXT: sh_set_barrier(SH_GROUP(0xf000u));
// CHECK-NEXT: if (sh_cluster_id() == sh_set_leader(SH_GROUP(0xf000u)) && sh_is_first_core())
// CHECK-NEXT: }
// CHECK-NEXT: sh_barrier_global();
// CHECK-NEXT: if (sh_cluster_id() == 0 && sh_is_first_core()) sh_printf("[mark] %s%u %u\n", "period"
