// RUN: softhier-translate %s | filecheck %s
// Training ops (docs/TTT.md): gemm_t (transposed operands through an HBM scratch), rmsnorm / silu_mul / softmax
// backward, MSE gradient, attention backward (prefix frozen, own keys' grads summed per kv group), SGD / Adam on an fp32
// arena, the data-parallel REDADD gradient sum, per-cluster buffers via cluster_id + cluster = -2 (SH_SELF), f32 dumps.
builtin.module {
  func.func @train() {
    %x   = softhier.hbm_buffer {offset = 65536 : i32}  : memref<50x720xf16, "hbm_west">
    %dy  = softhier.hbm_buffer {offset = 139264 : i32} : memref<50x720xf16, "hbm_west">
    %g   = softhier.hbm_buffer {offset = 212992 : i32} : memref<1x720xf16, "hbm_west">
    %dx  = softhier.hbm_buffer {offset = 217088 : i32} : memref<50x720xf16, "hbm_west">
    %a   = softhier.hbm_buffer {offset = 290816 : i32} : memref<720x16xf16, "hbm_west">
    %t   = softhier.hbm_buffer {offset = 315392 : i32} : memref<50x16xf16, "hbm_west">
    %scr = softhier.hbm_buffer {offset = 319488 : i32} : memref<4096x720xf16, "hbm_west">
    %gu  = softhier.hbm_buffer {offset = 6291456 : i32} : memref<50x4096xf16, "hbm_west">
    %dm  = softhier.hbm_buffer {offset = 6713344 : i32} : memref<50x2048xf16, "hbm_west">
    %dgu = softhier.hbm_buffer {offset = 6918144 : i32} : memref<50x4096xf16, "hbm_west">
    %p   = softhier.hbm_buffer {offset = 7340032 : i32} : memref<50x320xf16, "hbm_west">
    %v   = softhier.hbm_buffer {offset = 7372800 : i32} : memref<50x32xf16, "hbm_west">
    %tg  = softhier.hbm_buffer {offset = 7376896 : i32} : memref<50x32xf16, "hbm_west">
    %dv  = softhier.hbm_buffer {offset = 7380992 : i32} : memref<50x32xf16, "hbm_west">
    %ls  = softhier.hbm_buffer {offset = 7385088 : i32} : memref<50x1xf32, "hbm_west">
    %g16 = softhier.hbm_buffer {offset = 7389184 : i32} : memref<1532x64xf16, "hbm_west">
    %w32 = softhier.hbm_buffer {offset = 7589888 : i32} : memref<1532x64xf32, "hbm_west">
    %w16 = softhier.hbm_buffer {offset = 7987200 : i32} : memref<1532x64xf16, "hbm_west">
    %m32 = softhier.hbm_buffer {offset = 8192000 : i32} : memref<1532x64xf32, "hbm_west">
    %v32 = softhier.hbm_buffer {offset = 8589312 : i32} : memref<1532x64xf32, "hbm_west">
    %q   = softhier.hbm_buffer {offset = 9437184 : i32} : memref<50x1600xf16, "hbm_west">
    %kp  = softhier.hbm_buffer {offset = 9601024 : i32} : memref<241x320xf16, "hbm_west">
    %vp  = softhier.hbm_buffer {offset = 9756672 : i32} : memref<241x320xf16, "hbm_west">
    %tok = softhier.hbm_buffer {offset = 9912320 : i32} : memref<1x241xi16, "hbm_west">
    %o   = softhier.hbm_buffer {offset = 9916416 : i32} : memref<50x960xf16, "hbm_west">
    %do  = softhier.hbm_buffer {offset = 10014720 : i32} : memref<50x960xf16, "hbm_west">
    %dq  = softhier.hbm_buffer {offset = 10113024 : i32} : memref<50x1600xf16, "hbm_west">
    %as  = softhier.hbm_buffer {offset = 10276864 : i32} : memref<50x1920xf16, "hbm_west">
    %sc  = softhier.hbm_buffer {offset = 10473472 : i32} : memref<1x32xf16, "hbm_west">
    %qq  = softhier.view %q : memref<50x1600xf16, "hbm_west"> -> memref<50x960xf16, strided<[1600, 1], offset: 0>, "hbm_west">
    %kk  = softhier.view %q : memref<50x1600xf16, "hbm_west"> -> memref<50x320xf16, strided<[1600, 1], offset: 960>, "hbm_west">
    %vv  = softhier.view %q : memref<50x1600xf16, "hbm_west"> -> memref<50x320xf16, strided<[1600, 1], offset: 1280>, "hbm_west">
    %dqq = softhier.view %dq : memref<50x1600xf16, "hbm_west"> -> memref<50x960xf16, strided<[1600, 1], offset: 0>, "hbm_west">
    %dkk = softhier.view %dq : memref<50x1600xf16, "hbm_west"> -> memref<50x320xf16, strided<[1600, 1], offset: 960>, "hbm_west">
    %dvv = softhier.view %dq : memref<50x1600xf16, "hbm_west"> -> memref<50x320xf16, strided<[1600, 1], offset: 1280>, "hbm_west">
    %ga  = softhier.view %gu : memref<50x4096xf16, "hbm_west"> -> memref<50x2048xf16, strided<[4096, 1], offset: 0>, "hbm_west">
    %up  = softhier.view %gu : memref<50x4096xf16, "hbm_west"> -> memref<50x2048xf16, strided<[4096, 1], offset: 2048>, "hbm_west">
    %dga = softhier.view %dgu : memref<50x4096xf16, "hbm_west"> -> memref<50x2048xf16, strided<[4096, 1], offset: 0>, "hbm_west">
    %dup = softhier.view %dgu : memref<50x4096xf16, "hbm_west"> -> memref<50x2048xf16, strided<[4096, 1], offset: 2048>, "hbm_west">
    softhier.gemm_t %x, %t into %a scratch %scr {trans_x, tile_m = 48 : i32, tile_n = 16 : i32, tile_k = 50 : i32, pipeline, cluster = -1 : i32} : memref<50x720xf16, "hbm_west">, memref<50x16xf16, "hbm_west">, memref<720x16xf16, "hbm_west">, memref<4096x720xf16, "hbm_west">
    softhier.gemm_t %t, %a into %dx scratch %scr {trans_w, accumulate, tile_m = 50 : i32, tile_n = 48 : i32, tile_k = 16 : i32, cluster = -1 : i32} : memref<50x16xf16, "hbm_west">, memref<720x16xf16, "hbm_west">, memref<50x720xf16, "hbm_west">, memref<4096x720xf16, "hbm_west">
    softhier.rmsnorm_bwd %x, %g, %dy res %dy -> %dx {eps = 1.0e-5 : f32, cluster = -1 : i32} : memref<50x720xf16, "hbm_west">, memref<1x720xf16, "hbm_west">, memref<50x720xf16, "hbm_west"> res memref<50x720xf16, "hbm_west"> -> memref<50x720xf16, "hbm_west">
    softhier.rmsnorm_bwd %x, %g, %dy -> %dx {eps = 1.0e-5 : f32, cluster = -2 : i32} : memref<50x720xf16, "hbm_west">, memref<1x720xf16, "hbm_west">, memref<50x720xf16, "hbm_west"> -> memref<50x720xf16, "hbm_west">
    softhier.silu_mul_bwd %ga, %up, %dm -> %dga, %dup {cluster = -1 : i32} : memref<50x2048xf16, strided<[4096, 1], offset: 0>, "hbm_west">, memref<50x2048xf16, strided<[4096, 1], offset: 2048>, "hbm_west">, memref<50x2048xf16, "hbm_west"> -> memref<50x2048xf16, strided<[4096, 1], offset: 0>, "hbm_west">, memref<50x2048xf16, strided<[4096, 1], offset: 2048>, "hbm_west">
    softhier.softmax_bwd %p, %p -> %p {scale = 0.125 : f32, cluster = 3 : i32} : memref<50x320xf16, "hbm_west">, memref<50x320xf16, "hbm_west"> -> memref<50x320xf16, "hbm_west">
    softhier.mse_grad %v, %tg -> %dv loss %ls {gscale = 1.0 : f32, cluster = -1 : i32} : memref<50x32xf16, "hbm_west">, memref<50x32xf16, "hbm_west"> -> memref<50x32xf16, "hbm_west"> loss memref<50x1xf32, "hbm_west">
    softhier.attention_bwd %qq, %kp, %vp own %kk, %vv mask %tok, %o, %do -> %dqq grads %dkk, %dvv scratch %as {scale = 0.125 : f32, heads = 15 : i32, kv_heads = 5 : i32, cluster = -1 : i32}
        : memref<50x960xf16, strided<[1600, 1], offset: 0>, "hbm_west">, memref<241x320xf16, "hbm_west">, memref<241x320xf16, "hbm_west">
          own memref<50x320xf16, strided<[1600, 1], offset: 960>, "hbm_west">, memref<50x320xf16, strided<[1600, 1], offset: 1280>, "hbm_west">
          mask memref<1x241xi16, "hbm_west">, memref<50x960xf16, "hbm_west">, memref<50x960xf16, "hbm_west"> -> memref<50x960xf16, strided<[1600, 1], offset: 0>, "hbm_west">
          grads memref<50x320xf16, strided<[1600, 1], offset: 960>, "hbm_west">, memref<50x320xf16, strided<[1600, 1], offset: 1280>, "hbm_west"> scratch memref<50x1920xf16, "hbm_west">
    softhier.attention_bwd %qq, %kp, %vp, %o, %do -> %dqq {scale = 0.125 : f32, heads = 15 : i32, kv_heads = 5 : i32, cluster = -1 : i32}
        : memref<50x960xf16, strided<[1600, 1], offset: 0>, "hbm_west">, memref<241x320xf16, "hbm_west">, memref<241x320xf16, "hbm_west">, memref<50x960xf16, "hbm_west">, memref<50x960xf16, "hbm_west"> -> memref<50x960xf16, strided<[1600, 1], offset: 0>, "hbm_west">
    softhier.optim_step %g16, %w32 -> %w16 {kind = "sgd", lr = 0.01 : f32, inv_scale = 0.5 : f32, cluster = -1 : i32} : memref<1532x64xf16, "hbm_west">, memref<1532x64xf32, "hbm_west"> -> memref<1532x64xf16, "hbm_west">
    softhier.optim_step %g16, %w32 -> %w16 moments %m32, %v32 {kind = "adam", lr = 0.001 : f32, inv_scale = 1.0 : f32, b1 = 0.9 : f32, b2 = 0.999 : f32, eps = 1.0e-8 : f32, bc1 = 0.1 : f32, bc2 = 0.001 : f32, cluster = -1 : i32} : memref<1532x64xf16, "hbm_west">, memref<1532x64xf32, "hbm_west"> -> memref<1532x64xf16, "hbm_west"> moments memref<1532x64xf32, "hbm_west">, memref<1532x64xf32, "hbm_west">
    %cid = softhier.cluster_id : index
    %gc  = softhier.hbm_buffer %cid {offset = 16777216 : i32, stride = 196608 : i32} : memref<1532x64xf16, "hbm_west">
    softhier.grad_allreduce %gc -> %g16, %sc {src_stride = 196608 : i32, mode = 0 : i32} : memref<1532x64xf16, "hbm_west"> -> memref<1532x64xf16, "hbm_west">, memref<1x32xf16, "hbm_west">
    softhier.grad_allreduce %gc -> %w32, %sc {src_stride = 196608 : i32, mode = 1 : i32} : memref<1532x64xf16, "hbm_west"> -> memref<1532x64xf32, "hbm_west">, memref<1x32xf16, "hbm_west">
    softhier.dump_samples %w32 {seed = 7 : i32, n = 16 : i32, tag = "W"} : memref<1532x64xf32, "hbm_west">
    softhier.lora_fwd %x, %a, %t -> %dx, %t {scale = 2.0 : f32, tile_k = 720 : i32, cluster = -1 : i32} : memref<50x720xf16, "hbm_west">, memref<720x16xf16, "hbm_west">, memref<50x16xf16, "hbm_west"> -> memref<50x720xf16, "hbm_west">, memref<50x16xf16, "hbm_west">
    softhier.linear_bwd %dm, %dgu -> %dx scratch %scr {tile_m = 48 : i32, tile_k = 512 : i32, cluster = -2 : i32} : memref<50x2048xf16, "hbm_west">, memref<50x4096xf16, "hbm_west"> -> memref<50x720xf16, "hbm_west"> scratch memref<4096x720xf16, "hbm_west">
    softhier.linear_bwd %dy, %a lora %a, %t, %t, %x grads %a, %t -> %dx scratch %scr {scale = 2.0 : f32, tile_m = 48 : i32, tile_k = 16 : i32, cluster = -1 : i32} : memref<50x720xf16, "hbm_west">, memref<720x16xf16, "hbm_west"> lora memref<720x16xf16, "hbm_west">, memref<50x16xf16, "hbm_west">, memref<50x16xf16, "hbm_west">, memref<50x720xf16, "hbm_west"> grads memref<720x16xf16, "hbm_west">, memref<50x16xf16, "hbm_west"> -> memref<50x720xf16, "hbm_west"> scratch memref<4096x720xf16, "hbm_west">
    softhier.copy %dy -> %dx {cluster = -1 : i32} : memref<50x720xf16, "hbm_west"> -> memref<50x720xf16, "hbm_west">
    softhier.add %dy, %x -> %dx {train, cluster = -2 : i32} : memref<50x720xf16, "hbm_west">, memref<50x720xf16, "hbm_west"> -> memref<50x720xf16, "hbm_west">
    softhier.cross_attention %qq, %kp, %vp mask %tok -> %o {train, scale = 0.125 : f32, heads = 15 : i32, kv_heads = 5 : i32, cluster = -1 : i32} : memref<50x960xf16, strided<[1600, 1], offset: 0>, "hbm_west">, memref<241x320xf16, "hbm_west">, memref<241x320xf16, "hbm_west"> mask memref<1x241xi16, "hbm_west"> -> memref<50x960xf16, "hbm_west">
    func.return
  }
}
// CHECK: sh_t_gemm_tr(hb1, hb6, hb5, 720, 16, 50, 720, 16, 16, 1, 0, hb7, &cfg, SH_ALL); }
// CHECK: .accumulate = 1
// CHECK-NEXT: sh_t_gemm_tr(hb6, hb5, hb4, 50, 720, 16, 16, 16, 720, 0, 1, hb7, &cfg, SH_ALL); }
// CHECK: sh_t_rmsnorm_bwd(hb4, hb1, hb2, hb2, hb3, 50, 720, 720, 720, 720, 720, 9.999999747378752e-06f, SH_ALL);
// CHECK: sh_t_rmsnorm_bwd(hb4, hb1, hb2, 0, hb3, 50, 720, 720, 720, 720, 0, 9.999999747378752e-06f, SH_SELF);
// CHECK: sh_t_silu_mul_bwd(hb10, (hb10 + 4096), hb8, (hb8 + 4096), hb9, 50, 2048, 4096, 4096, 4096, 4096, 2048, SH_ALL);
// CHECK: sh_t_softmax_bwd(hb11, hb11, hb11, 50, 320, 320, 0.125f, 3);
// CHECK: sh_t_mse_grad(hb14, hb15, hb12, hb13, 50, 32, 32, 1.0f, SH_ALL);
// CHECK: sh_t_attention_bwd(hb21, hb22, hb23, (hb21 + 1920), (hb21 + 2560), hb24, hb25, hb26, hb27, (hb27 + 1920), (hb27 + 2560), hb28, 50, 241, 50, 15, 5, 64, 1600, 320, 320, 1600, 1600, 960, 960, 1600, 1600, 1600, 0.125f, SH_ALL);
// CHECK: sh_t_attention_bwd(hb21, hb22, hb23, 0, 0, 0, hb25, hb26, hb27, 0, 0, 0, 50, 241, 0, 15, 5, 64, 1600, 320, 320, 0, 0, 960, 960, 1600, 0, 0, 0.125f, SH_ALL);
// CHECK: sh_t_optim(hb17, hb18, 0, 0, hb16, 2, 1532, 64, 0, {{[0-9.]+}}f, 0.5f,
// CHECK: sh_t_optim(hb17, hb18, hb19, hb20, hb16, 2, 1532, 64, 1, {{[0-9.e+-]+}}f, 1.0f, {{[0-9.e+-]+}}f, {{[0-9.e+-]+}}f, {{[0-9.e+-]+}}f, {{[0-9.e+-]+}}f, {{[0-9.e+-]+}}f, SH_ALL);
// CHECK: const uint64_t hb{{[0-9]+}} = sh_hbm_addr((uint64_t)16777216 + (uint64_t)(sh_cluster_id()) * 196608u);
// CHECK: sh_t_allreduce(hb16, hb{{[0-9]+}}, 196608u, 98048, 0, hb29);
// CHECK: sh_t_allreduce(hb17, hb{{[0-9]+}}, 196608u, 98048, 1, hb29);
// CHECK: sh_t_lora_fwd(hb4, hb1, hb5, hb6, hb6, 50, 720, 720, 16, 720, 720, 2.0f, 720, SH_ALL);
// CHECK: sh_t_linear_bwd(hb4, hb9, hb10, 50, 50, 2048, 720, 2048, 4096, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0f, hb7, 48, 512, SH_SELF);
// CHECK: sh_t_linear_bwd(hb4, hb2, hb5, 50, 720, 720, 720, 720, 16, hb5, hb6, hb6, hb1, hb5, hb6, 16, 16, 720, 2.0f, hb7, 48, 16, SH_ALL);
// CHECK: sh_t_copy(hb4, hb2, 50, 720, 720, 720, SH_ALL);
// CHECK: sh_t_add(hb4, hb2, hb1, 50, 720, 720, 720, 720, SH_SELF);
// CHECK: sh_t_attention_fwd(hb21, hb22, hb23, 0, 0, hb24, hb25, 50, 241, 0, 15, 5, 64, 1600, 320, 320, 0, 0, 960, 0.125f, SH_ALL);
// CHECK: sh_t_dump_samples_f32(hb17, 1532, 64, 64, 7, 16, "W");
