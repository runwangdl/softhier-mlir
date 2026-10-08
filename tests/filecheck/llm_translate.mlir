// RUN: softhier-translate %s | filecheck %s
// Llama-style decoder ops (SmolVLA VLM prefix): rmsnorm / rope / silu_mul / masked softmax / grouped-query
// attention with a token mask / pixel shuffle lower 1:1 to runtime/sh_llm.inc.c; an attention without
// `mask` or `kv_heads` still lowers to sh_attention. arith.divui over an index feeds a two-region weight family.
builtin.module {
  func.func @vlm_layer() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c11 = arith.constant 11 : index
    %gap = arith.constant 536870912 : index
    %x   = softhier.hbm_buffer {offset = 65536 : i32}    : memref<256x960xf16, "hbm_west">
    %g   = softhier.hbm_buffer {offset = 557056 : i32}   : memref<1x960xf16, "hbm_west">
    %ln  = softhier.hbm_buffer {offset = 561152 : i32}   : memref<256x960xf16, "hbm_west">
    %tab = softhier.hbm_buffer {offset = 1052672 : i32}  : memref<256x64xf16, "hbm_west">
    %tok = softhier.hbm_buffer {offset = 1085440 : i32}  : memref<256xi16, "hbm_west">
    %kv  = softhier.hbm_buffer {offset = 1089536 : i32}  : memref<256x640xf16, "hbm_west">
    %o   = softhier.hbm_buffer {offset = 1417216 : i32}  : memref<256x960xf16, "hbm_west">
    %sc  = softhier.hbm_buffer {offset = 1908736 : i32}  : memref<256x256xf16, "hbm_west">
    %vis = softhier.hbm_buffer {offset = 2039808 : i32}  : memref<1024x768xf16, "hbm_west">
    %ps  = softhier.hbm_buffer {offset = 3612672 : i32}  : memref<64x12288xf16, "hbm_west">
    %k = softhier.view %kv : memref<256x640xf16, "hbm_west"> -> memref<256x320xf16, strided<[640, 1], offset: 0>, "hbm_west">
    %v = softhier.view %kv : memref<256x640xf16, "hbm_west"> -> memref<256x320xf16, strided<[640, 1], offset: 320>, "hbm_west">
    softhier.rmsnorm %x, %g -> %ln {eps = 1.0e-5 : f32, cluster = -1 : i32} : memref<256x960xf16, "hbm_west">, memref<1x960xf16, "hbm_west"> -> memref<256x960xf16, "hbm_west">
    softhier.rope %x, %tab -> %x {head_dim = 64 : i32, cluster = -1 : i32} : memref<256x960xf16, "hbm_west">, memref<256x64xf16, "hbm_west"> -> memref<256x960xf16, "hbm_west">
    softhier.rope %k, %tab -> %k {head_dim = 64 : i32, cluster = 2 : i32} : memref<256x320xf16, strided<[640, 1], offset: 0>, "hbm_west">, memref<256x64xf16, "hbm_west"> -> memref<256x320xf16, strided<[640, 1], offset: 0>, "hbm_west">
    softhier.silu_mul %x, %ln -> %o {cluster = -1 : i32} : memref<256x960xf16, "hbm_west">, memref<256x960xf16, "hbm_west"> -> memref<256x960xf16, "hbm_west">
    softhier.softmax %sc, %tok -> %sc {scale = 0.125 : f32, cluster = 3 : i32} : memref<256x256xf16, "hbm_west">, memref<256xi16, "hbm_west"> -> memref<256x256xf16, "hbm_west">
    softhier.softmax %sc -> %sc {scale = 0.125 : f32, cluster = 3 : i32} : memref<256x256xf16, "hbm_west"> -> memref<256x256xf16, "hbm_west">
    softhier.attention %x, %k, %v, %tok -> %o {scale = 0.125 : f32, heads = 15 : i32, kv_heads = 5 : i32, cluster = -1 : i32}
        : memref<256x960xf16, "hbm_west">, memref<256x320xf16, strided<[640, 1], offset: 0>, "hbm_west">, memref<256x320xf16, strided<[640, 1], offset: 320>, "hbm_west">, memref<256xi16, "hbm_west"> -> memref<256x960xf16, "hbm_west">
    softhier.attention %x, %k, %v -> %o {scale = 0.125 : f32, heads = 15 : i32, kv_heads = 5 : i32, cluster = 0 : i32}
        : memref<256x960xf16, "hbm_west">, memref<256x320xf16, strided<[640, 1], offset: 0>, "hbm_west">, memref<256x320xf16, strided<[640, 1], offset: 320>, "hbm_west"> -> memref<256x960xf16, "hbm_west">
    softhier.attention %x, %x, %x -> %o {scale = 0.125 : f32, heads = 15 : i32, cluster = -1 : i32}
        : memref<256x960xf16, "hbm_west">, memref<256x960xf16, "hbm_west">, memref<256x960xf16, "hbm_west"> -> memref<256x960xf16, "hbm_west">
    softhier.scale %x -> %x {scale = 30.983866769659336 : f32, cluster = -1 : i32} : memref<256x960xf16, "hbm_west"> -> memref<256x960xf16, "hbm_west">
    softhier.pixel_shuffle %vis -> %ps {scale = 4 : i32, cluster = -1 : i32} : memref<1024x768xf16, "hbm_west"> -> memref<64x12288xf16, "hbm_west">
    scf.for %L = %c0 to %c2 step %c1 {
      %r = arith.divui %L, %c11 : index
      %e = arith.muli %r, %gap : index
      %w = softhier.hbm_buffer %e {offset = 4194304 : i32, stride = 1 : i32} : memref<960x960xf16, "hbm_west">
      softhier.gemm %x, %w into %o {fmt = "fp16", tile_m = 256 : i32, tile_n = 320 : i32, tile_k = 320 : i32, pipeline, cluster = -1 : i32} : memref<256x960xf16, "hbm_west">, memref<960x960xf16, "hbm_west">, memref<256x960xf16, "hbm_west">
    }
    func.return
  }
}
// CHECK: sh_rmsnorm(hb3, hb1, hb2, 256, 960, 960, 9.999999747378752e-06f, SH_ALL);
// CHECK: sh_rope(hb1, hb1, hb4, 256, 960, 960, 64, 64, SH_ALL);
// CHECK: sh_rope(hb6, hb6, hb4, 256, 320, 640, 64, 64, 2);
// CHECK: sh_silu_mul(hb7, hb1, hb3, 256, 960, 960, SH_ALL);
// CHECK: sh_softmax_masked(hb8, hb8, 256, 256, 256, 0.125f, hb5, hb5, 3);
// CHECK: sh_softmax_rows(hb8, hb8, 256, 256, 256, 0.125f, 3);
// CHECK: sh_attention_gqa(hb1, hb6, (hb6 + 640), hb7, 256, 960, 15, 5, 960, 640, 640, 960, 0.125f, hb5, SH_ALL);
// CHECK: sh_attention_gqa(hb1, hb6, (hb6 + 640), hb7, 256, 960, 15, 5, 960, 640, 640, 960, 0.125f, 0, 0);
// CHECK: sh_attention(hb1, hb1, hb1, hb7, 256, 960, 15, 960, 960, 960, 960, 0.125f, SH_ALL);
// CHECK: sh_scale(hb1, hb1, 256, 960, 960, 30.983867645263672f, SH_ALL);
// CHECK: sh_pixel_shuffle(hb10, hb9, 32, 768, 4, SH_ALL);
// CHECK: for (uint32_t i1 = 0; i1 < 2; i1 += 1) {
// CHECK-NEXT: const uint64_t hb11 = sh_hbm_addr((uint64_t)4194304 + (uint64_t)(((i1 / 11) * 536870912)) * 1u);
