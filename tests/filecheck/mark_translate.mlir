// RUN: softhier-translate %s | filecheck %s
// softhier.mark prints a cluster-0 mcycle stamp (1 GHz -> ns) between ops; softhier.preload_wait
// blocks on the preload image's sentinel segment (all cores, ends with a global barrier).
builtin.module {
  func.func @timed() {
    %x = softhier.hbm_buffer {offset = 65536 : i32} : memref<256x768xf16, "hbm_west">
    %g = softhier.hbm_buffer {offset = 458752 : i32} : memref<1x768xf16, "hbm_west">
    %s = softhier.hbm_buffer {offset = 462848 : i32} : memref<1x32xf16, "hbm_west">
    softhier.preload_wait %s : memref<1x32xf16, "hbm_west">
    softhier.mark {tag = "start"}
    softhier.layernorm %x, %g, %g -> %x {eps = 1.0e-6 : f32, cluster = -1 : i32} : memref<256x768xf16, "hbm_west">, memref<1x768xf16, "hbm_west">, memref<1x768xf16, "hbm_west"> -> memref<256x768xf16, "hbm_west">
    softhier.mark {tag = "ln"}
    func.return
  }
}
// CHECK: static void timed(void)
// CHECK: sh_preload_wait(hb3);
// CHECK-NEXT: sh_printf("[mark] %s %u\n", "start", sh_cycles());
// CHECK-NEXT: sh_layernorm(hb1, hb1, hb2, hb2, 256, 768, 768, {{.*}}, SH_ALL);
// CHECK-NEXT: sh_printf("[mark] %s %u\n", "ln", sh_cycles());
