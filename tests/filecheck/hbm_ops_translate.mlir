// RUN: softhier-translate %s | filecheck %s
// HBM tensor ops + strided views lower 1:1 to softhier-ops calls; `cluster = -1` means SH_ALL.
builtin.module {
  func.func @attn_head() {
    %x  = softhier.hbm_buffer {offset = 0 : i32}      : memref<256x768xf16, "hbm_west">
    %g  = softhier.hbm_buffer {offset = 393216 : i32} : memref<1x768xf16, "hbm_west">
    %b  = softhier.hbm_buffer {offset = 397312 : i32} : memref<1x768xf16, "hbm_west">
    %y  = softhier.hbm_buffer {offset = 401408 : i32} : memref<256x768xf16, "hbm_west">
    %kT = softhier.hbm_buffer {offset = 794624 : i32} : memref<768x256xf16, "hbm_west">
    %s  = softhier.hbm_buffer {offset = 1187840 : i32} : memref<256x256xf16, "hbm_west">
    softhier.hbm_fill_lcg %x {seed = 1 : i32, lo = -16 : i32, hi = 16 : i32, scale = 0.125 : f32} : memref<256x768xf16, "hbm_west">
    softhier.layernorm %x, %g, %b -> %y {eps = 1.0e-6 : f32, cluster = -1 : i32}
        : memref<256x768xf16, "hbm_west">, memref<1x768xf16, "hbm_west">, memref<1x768xf16, "hbm_west"> -> memref<256x768xf16, "hbm_west">
    softhier.transpose %y -> %kT {cluster = -1 : i32} : memref<256x768xf16, "hbm_west"> -> memref<768x256xf16, "hbm_west">
    %q1  = softhier.view %y  : memref<256x768xf16, "hbm_west"> -> memref<256x64xf16, strided<[768, 1], offset: 64>, "hbm_west">
    %kT1 = softhier.view %kT : memref<768x256xf16, "hbm_west"> -> memref<64x256xf16, strided<[256, 1], offset: 16384>, "hbm_west">
    softhier.gemm %q1, %kT1 into %s {fmt = "fp16", tile_m = 256 : i32, tile_n = 256 : i32, tile_k = 64 : i32, pipeline, cluster = 1 : i32}
        : memref<256x64xf16, strided<[768, 1], offset: 64>, "hbm_west">, memref<64x256xf16, strided<[256, 1], offset: 16384>, "hbm_west">, memref<256x256xf16, "hbm_west">
    softhier.softmax %s -> %s {scale = 0.125 : f32, cluster = 1 : i32} : memref<256x256xf16, "hbm_west"> -> memref<256x256xf16, "hbm_west">
    softhier.gelu %y -> %y {cluster = -1 : i32} : memref<256x768xf16, "hbm_west"> -> memref<256x768xf16, "hbm_west">
    softhier.add_bias %y, %b -> %y {cluster = -1 : i32} : memref<256x768xf16, "hbm_west">, memref<1x768xf16, "hbm_west"> -> memref<256x768xf16, "hbm_west">
    softhier.add %x, %y -> %y {cluster = -1 : i32} : memref<256x768xf16, "hbm_west">, memref<256x768xf16, "hbm_west"> -> memref<256x768xf16, "hbm_west">
    softhier.dump_samples %s {seed = 9 : i32, n = 32 : i32, tag = "P"} : memref<256x256xf16, "hbm_west">
    func.return
  }
}
// CHECK: sh_test_fill_fp16(hb1, 256, 768, 768, 1, -16, 16, 0.125f)
// CHECK: sh_layernorm(hb4, hb1, hb2, hb3, 256, 768, 768, 9.999999974752427e-07f, SH_ALL)
// CHECK: sh_transpose(hb5, hb4, 256, 768, 768, 256, SH_ALL)
// CHECK: .tm = 256, .tn = 256, .tk = 64, .pipeline = 1
// CHECK: sh_gemm((hb4 + 128), (hb5 + 32768), hb6, 256, 256, 64, 768, 256, 256, &cfg, 1)
// CHECK: sh_softmax_rows(hb6, hb6, 256, 256, 256, 0.125f, 1)
// CHECK: sh_gelu(hb4, hb4, 256, 768, 768, SH_ALL)
// CHECK: sh_add_bias(hb4, hb4, hb3, 256, 768, 768, SH_ALL)
// CHECK: sh_add(hb4, hb1, hb4, 256, 768, 768, SH_ALL)
// CHECK: sh_test_dump_samples(hb6, 256, 256, 256, 9, 32, "P")
