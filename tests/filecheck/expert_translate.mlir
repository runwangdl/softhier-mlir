// RUN: softhier-translate %s | filecheck %s
// SmolVLA action-expert ops: rmsnorm / rope / silu_mul / axpy / cross_attention (prefix KV + optional own causal
// keys + optional validity row) lower 1:1 to the sh_x_* library calls; a gemm with `step` + `fmt_steps` picks the
// RedMulE format per loop iteration (flow-matching steps of different precision); dump_all prints a whole tensor.
builtin.module {
  func.func @expert() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c10 = arith.constant 10 : index
    %h   = softhier.hbm_buffer {offset = 65536 : i32}  : memref<50x720xf16, "hbm_west">
    %g   = softhier.hbm_buffer {offset = 139264 : i32} : memref<1x720xf16, "hbm_west">
    %xn  = softhier.hbm_buffer {offset = 143360 : i32} : memref<50x720xf16, "hbm_west">
    %qkv = softhier.hbm_buffer {offset = 217088 : i32} : memref<50x1600xf16, "hbm_west">
    %w   = softhier.hbm_buffer {offset = 380928 : i32} : memref<720x1600xf16, "hbm_west">
    %tab = softhier.hbm_buffer {offset = 2686976 : i32} : memref<50x960xf16, "hbm_west">
    %kp  = softhier.hbm_buffer {offset = 2785280 : i32} : memref<241x320xf16, "hbm_west">
    %vp  = softhier.hbm_buffer {offset = 2940928 : i32} : memref<241x320xf16, "hbm_west">
    %m   = softhier.hbm_buffer {offset = 3096576 : i32} : memref<1x241xf16, "hbm_west">
    %o   = softhier.hbm_buffer {offset = 3100672 : i32} : memref<50x960xf16, "hbm_west">
    %gu  = softhier.hbm_buffer {offset = 3198976 : i32} : memref<50x4096xf16, "hbm_west">
    %mm  = softhier.hbm_buffer {offset = 3608576 : i32} : memref<50x2048xf16, "hbm_west">
    %x   = softhier.hbm_buffer {offset = 3813376 : i32} : memref<50x32xf16, "hbm_west">
    %v   = softhier.hbm_buffer {offset = 3817472 : i32} : memref<50x32xf16, "hbm_west">
    softhier.rmsnorm %h, %g -> %xn {eps = 1.0e-5 : f32, cluster = -1 : i32} : memref<50x720xf16, "hbm_west">, memref<1x720xf16, "hbm_west"> -> memref<50x720xf16, "hbm_west">
    scf.for %s = %c0 to %c10 step %c1 {
      softhier.gemm %xn, %w into %qkv step %s {fmt = "fp16", fmt_steps = ["fp8", "fp8", "fp16"], tile_m = 50 : i32, tile_n = 64 : i32, tile_k = 720 : i32, pipeline, cluster = -1 : i32} : memref<50x720xf16, "hbm_west">, memref<720x1600xf16, "hbm_west">, memref<50x1600xf16, "hbm_west">
      softhier.dump_all %x, %s {tag = "X"} : memref<50x32xf16, "hbm_west">
    }
    %q  = softhier.view %qkv : memref<50x1600xf16, "hbm_west"> -> memref<50x960xf16, strided<[1600, 1], offset: 0>, "hbm_west">
    %k  = softhier.view %qkv : memref<50x1600xf16, "hbm_west"> -> memref<50x320xf16, strided<[1600, 1], offset: 960>, "hbm_west">
    %vv = softhier.view %qkv : memref<50x1600xf16, "hbm_west"> -> memref<50x320xf16, strided<[1600, 1], offset: 1280>, "hbm_west">
    %tk = softhier.view %tab : memref<50x960xf16, "hbm_west"> -> memref<50x320xf16, strided<[960, 1], offset: 0>, "hbm_west">
    softhier.rope %q, %tab -> %q {dh = 64 : i32, cluster = -1 : i32} : memref<50x960xf16, strided<[1600, 1], offset: 0>, "hbm_west">, memref<50x960xf16, "hbm_west"> -> memref<50x960xf16, strided<[1600, 1], offset: 0>, "hbm_west">
    softhier.rope %k, %tk -> %k {dh = 64 : i32, cluster = -1 : i32} : memref<50x320xf16, strided<[1600, 1], offset: 960>, "hbm_west">, memref<50x320xf16, strided<[960, 1], offset: 0>, "hbm_west"> -> memref<50x320xf16, strided<[1600, 1], offset: 960>, "hbm_west">
    softhier.cross_attention %q, %kp, %vp own %k, %vv valid %m -> %o {scale = 0.125 : f32, heads = 15 : i32, kv_heads = 5 : i32, cluster = -1 : i32}
        : memref<50x960xf16, strided<[1600, 1], offset: 0>, "hbm_west">, memref<241x320xf16, "hbm_west">, memref<241x320xf16, "hbm_west">
          own memref<50x320xf16, strided<[1600, 1], offset: 960>, "hbm_west">, memref<50x320xf16, strided<[1600, 1], offset: 1280>, "hbm_west"> valid memref<1x241xf16, "hbm_west"> -> memref<50x960xf16, "hbm_west">
    softhier.cross_attention %q, %kp, %vp -> %o {scale = 0.125 : f32, heads = 15 : i32, kv_heads = 5 : i32, cluster = 2 : i32}
        : memref<50x960xf16, strided<[1600, 1], offset: 0>, "hbm_west">, memref<241x320xf16, "hbm_west">, memref<241x320xf16, "hbm_west"> -> memref<50x960xf16, "hbm_west">
    %ga = softhier.view %gu : memref<50x4096xf16, "hbm_west"> -> memref<50x2048xf16, strided<[4096, 1], offset: 0>, "hbm_west">
    %up = softhier.view %gu : memref<50x4096xf16, "hbm_west"> -> memref<50x2048xf16, strided<[4096, 1], offset: 2048>, "hbm_west">
    softhier.silu_mul %ga, %up -> %mm {cluster = -1 : i32} : memref<50x2048xf16, strided<[4096, 1], offset: 0>, "hbm_west">, memref<50x2048xf16, strided<[4096, 1], offset: 2048>, "hbm_west"> -> memref<50x2048xf16, "hbm_west">
    softhier.silu_mul %xn -> %xn {cluster = 0 : i32} : memref<50x720xf16, "hbm_west"> -> memref<50x720xf16, "hbm_west">
    softhier.axpy %x, %v -> %x {alpha = -0.1 : f32, cluster = -1 : i32} : memref<50x32xf16, "hbm_west">, memref<50x32xf16, "hbm_west"> -> memref<50x32xf16, "hbm_west">
    softhier.dump_all %x {tag = "A"} : memref<50x32xf16, "hbm_west">
    func.return
  }
}
// CHECK: sh_x_rmsnorm(hb3, hb1, hb2, 50, 720, 720, 720, 9.999999747378752e-06f, SH_ALL);
// CHECK: for (uint32_t i1 = 0; i1 < 10; i1 += 1) {
// CHECK-NEXT: { sh_gemm_cfg cfg = { .tm = 50, .tn = 64, .tk = 720, .pipeline = 1, .accumulate = 0, .fmt = ((const uint32_t[]){SH_FP8, SH_FP8, SH_FP16})[i1], .l1_base = 0 };
// CHECK-NEXT: sh_gemm(hb3, hb5, hb4, 50, 1600, 720, 720, 1600, 1600, &cfg, SH_ALL); }
// CHECK-NEXT: if (sh_cluster_id() == 0 && sh_is_first_core()) sh_test_dump_all_idx(hb13, 50, 32, 32, "X", (uint32_t)(i1));
// CHECK: sh_x_rope(hb4, hb4, hb6, 50, 960, 1600, 1600, 960, 64, SH_ALL);
// CHECK: sh_x_rope((hb4 + 1920), (hb4 + 1920), hb6, 50, 320, 1600, 1600, 960, 64, SH_ALL);
// CHECK: sh_x_attention(hb4, hb7, hb8, (hb4 + 1920), (hb4 + 2560), hb9, hb10, 50, 241, 50, 15, 5, 64, 1600, 320, 320, 1600, 1600, 960, 0.125f, SH_ALL);
// CHECK: sh_x_attention(hb4, hb7, hb8, 0, 0, 0, hb10, 50, 241, 0, 15, 5, 64, 1600, 320, 320, 0, 0, 960, 0.125f, 2);
// CHECK: sh_x_silu_mul(hb12, hb11, (hb11 + 4096), 50, 2048, 2048, 4096, 4096, SH_ALL);
// CHECK: sh_x_silu_mul(hb3, hb3, 0, 50, 720, 720, 720, 0, 0);
// CHECK: sh_x_axpy(hb13, hb13, hb14, 50, 32, 32, 32, 32, -0.10000000149011612f, SH_ALL);
// CHECK: if (sh_cluster_id() == 0 && sh_is_first_core()) sh_test_dump_all(hb13, 50, 32, 32, "A");
