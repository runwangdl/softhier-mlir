// One SigLIP/ViT encoder layer at full resolution (1024 tokens, d=768, 12 heads, FFN 3072) as emitted by
// `python -m softhier_mlir.frontend.siglip --seq 1024 --layers 1 --no-test`: every activation/parameter is an
// HBM buffer, per-head attention works on strided views, `cluster = -1` = all 16 clusters, `cluster = n` pins a head.
builtin.module {
  func.func @siglip_encoder() {
    %x = softhier.hbm_buffer {offset = 0 : i32} : memref<1024x768xf16, "hbm_west">
    %ln1 = softhier.hbm_buffer {offset = 1572864 : i32} : memref<1024x768xf16, "hbm_west">
    %q = softhier.hbm_buffer {offset = 3145728 : i32} : memref<1024x768xf16, "hbm_west">
    %k = softhier.hbm_buffer {offset = 4718592 : i32} : memref<1024x768xf16, "hbm_west">
    %v = softhier.hbm_buffer {offset = 6291456 : i32} : memref<1024x768xf16, "hbm_west">
    %kT = softhier.hbm_buffer {offset = 7864320 : i32} : memref<768x1024xf16, "hbm_west">
    %sc = softhier.hbm_buffer {offset = 9437184 : i32} : memref<12288x1024xf16, "hbm_west">
    %o = softhier.hbm_buffer {offset = 34603008 : i32} : memref<1024x768xf16, "hbm_west">
    %ao = softhier.hbm_buffer {offset = 36175872 : i32} : memref<1024x768xf16, "hbm_west">
    %h = softhier.hbm_buffer {offset = 37748736 : i32} : memref<1024x768xf16, "hbm_west">
    %ln2 = softhier.hbm_buffer {offset = 39321600 : i32} : memref<1024x768xf16, "hbm_west">
    %f1 = softhier.hbm_buffer {offset = 40894464 : i32} : memref<1024x3072xf16, "hbm_west">
    %g = softhier.hbm_buffer {offset = 47185920 : i32} : memref<1024x3072xf16, "hbm_west">
    %f2 = softhier.hbm_buffer {offset = 53477376 : i32} : memref<1024x768xf16, "hbm_west">
    %out = softhier.hbm_buffer {offset = 55050240 : i32} : memref<1024x768xf16, "hbm_west">
    %wq0 = softhier.hbm_buffer {offset = 56623104 : i32} : memref<768x768xf16, "hbm_west">
    %wk0 = softhier.hbm_buffer {offset = 57802752 : i32} : memref<768x768xf16, "hbm_west">
    %wv0 = softhier.hbm_buffer {offset = 58982400 : i32} : memref<768x768xf16, "hbm_west">
    %wo0 = softhier.hbm_buffer {offset = 60162048 : i32} : memref<768x768xf16, "hbm_west">
    %w10 = softhier.hbm_buffer {offset = 61341696 : i32} : memref<768x3072xf16, "hbm_west">
    %w20 = softhier.hbm_buffer {offset = 66060288 : i32} : memref<3072x768xf16, "hbm_west">
    %bq0 = softhier.hbm_buffer {offset = 70778880 : i32} : memref<1x768xf16, "hbm_west">
    %bk0 = softhier.hbm_buffer {offset = 70782976 : i32} : memref<1x768xf16, "hbm_west">
    %bv0 = softhier.hbm_buffer {offset = 70787072 : i32} : memref<1x768xf16, "hbm_west">
    %bo0 = softhier.hbm_buffer {offset = 70791168 : i32} : memref<1x768xf16, "hbm_west">
    %b10 = softhier.hbm_buffer {offset = 70795264 : i32} : memref<1x3072xf16, "hbm_west">
    %b20 = softhier.hbm_buffer {offset = 70803456 : i32} : memref<1x768xf16, "hbm_west">
    %g10 = softhier.hbm_buffer {offset = 70807552 : i32} : memref<1x768xf16, "hbm_west">
    %be10 = softhier.hbm_buffer {offset = 70811648 : i32} : memref<1x768xf16, "hbm_west">
    %g20 = softhier.hbm_buffer {offset = 70815744 : i32} : memref<1x768xf16, "hbm_west">
    %be20 = softhier.hbm_buffer {offset = 70819840 : i32} : memref<1x768xf16, "hbm_west">
    softhier.layernorm %x, %g10, %be10 -> %ln1 {eps = 1.0e-6 : f32, cluster = -1 : i32} : memref<1024x768xf16, "hbm_west">, memref<1x768xf16, "hbm_west">, memref<1x768xf16, "hbm_west"> -> memref<1024x768xf16, "hbm_west">
    softhier.gemm %ln1, %wq0 into %q {fmt = "fp16", tile_m = 256 : i32, tile_n = 256 : i32, tile_k = 256 : i32, pipeline, cluster = -1 : i32} : memref<1024x768xf16, "hbm_west">, memref<768x768xf16, "hbm_west">, memref<1024x768xf16, "hbm_west">
    softhier.add_bias %q, %bq0 -> %q {cluster = -1 : i32} : memref<1024x768xf16, "hbm_west">, memref<1x768xf16, "hbm_west"> -> memref<1024x768xf16, "hbm_west">
    softhier.gemm %ln1, %wk0 into %k {fmt = "fp16", tile_m = 256 : i32, tile_n = 256 : i32, tile_k = 256 : i32, pipeline, cluster = -1 : i32} : memref<1024x768xf16, "hbm_west">, memref<768x768xf16, "hbm_west">, memref<1024x768xf16, "hbm_west">
    softhier.add_bias %k, %bk0 -> %k {cluster = -1 : i32} : memref<1024x768xf16, "hbm_west">, memref<1x768xf16, "hbm_west"> -> memref<1024x768xf16, "hbm_west">
    softhier.gemm %ln1, %wv0 into %v {fmt = "fp16", tile_m = 256 : i32, tile_n = 256 : i32, tile_k = 256 : i32, pipeline, cluster = -1 : i32} : memref<1024x768xf16, "hbm_west">, memref<768x768xf16, "hbm_west">, memref<1024x768xf16, "hbm_west">
    softhier.add_bias %v, %bv0 -> %v {cluster = -1 : i32} : memref<1024x768xf16, "hbm_west">, memref<1x768xf16, "hbm_west"> -> memref<1024x768xf16, "hbm_west">
    softhier.transpose %k -> %kT {cluster = -1 : i32} : memref<1024x768xf16, "hbm_west"> -> memref<768x1024xf16, "hbm_west">
    %q0_0 = softhier.view %q : memref<1024x768xf16, "hbm_west"> -> memref<1024x64xf16, strided<[768, 1], offset: 0>, "hbm_west">
    %kT0_0 = softhier.view %kT : memref<768x1024xf16, "hbm_west"> -> memref<64x1024xf16, strided<[1024, 1], offset: 0>, "hbm_west">
    %s0_0 = softhier.view %sc : memref<12288x1024xf16, "hbm_west"> -> memref<1024x1024xf16, strided<[1024, 1], offset: 0>, "hbm_west">
    %v0_0 = softhier.view %v : memref<1024x768xf16, "hbm_west"> -> memref<1024x64xf16, strided<[768, 1], offset: 0>, "hbm_west">
    %o0_0 = softhier.view %o : memref<1024x768xf16, "hbm_west"> -> memref<1024x64xf16, strided<[768, 1], offset: 0>, "hbm_west">
    softhier.gemm %q0_0, %kT0_0 into %s0_0 {fmt = "fp16", tile_m = 256 : i32, tile_n = 256 : i32, tile_k = 64 : i32, pipeline, cluster = 0 : i32} : memref<1024x64xf16, strided<[768, 1], offset: 0>, "hbm_west">, memref<64x1024xf16, strided<[1024, 1], offset: 0>, "hbm_west">, memref<1024x1024xf16, strided<[1024, 1], offset: 0>, "hbm_west">
    softhier.softmax %s0_0 -> %s0_0 {scale = 0.125 : f32, cluster = 0 : i32} : memref<1024x1024xf16, strided<[1024, 1], offset: 0>, "hbm_west"> -> memref<1024x1024xf16, strided<[1024, 1], offset: 0>, "hbm_west">
    softhier.gemm %s0_0, %v0_0 into %o0_0 {fmt = "fp16", tile_m = 256 : i32, tile_n = 64 : i32, tile_k = 256 : i32, pipeline, cluster = 0 : i32} : memref<1024x1024xf16, strided<[1024, 1], offset: 0>, "hbm_west">, memref<1024x64xf16, strided<[768, 1], offset: 0>, "hbm_west">, memref<1024x64xf16, strided<[768, 1], offset: 0>, "hbm_west">
    %q0_1 = softhier.view %q : memref<1024x768xf16, "hbm_west"> -> memref<1024x64xf16, strided<[768, 1], offset: 64>, "hbm_west">
    %kT0_1 = softhier.view %kT : memref<768x1024xf16, "hbm_west"> -> memref<64x1024xf16, strided<[1024, 1], offset: 65536>, "hbm_west">
    %s0_1 = softhier.view %sc : memref<12288x1024xf16, "hbm_west"> -> memref<1024x1024xf16, strided<[1024, 1], offset: 1048576>, "hbm_west">
    %v0_1 = softhier.view %v : memref<1024x768xf16, "hbm_west"> -> memref<1024x64xf16, strided<[768, 1], offset: 64>, "hbm_west">
    %o0_1 = softhier.view %o : memref<1024x768xf16, "hbm_west"> -> memref<1024x64xf16, strided<[768, 1], offset: 64>, "hbm_west">
    softhier.gemm %q0_1, %kT0_1 into %s0_1 {fmt = "fp16", tile_m = 256 : i32, tile_n = 256 : i32, tile_k = 64 : i32, pipeline, cluster = 1 : i32} : memref<1024x64xf16, strided<[768, 1], offset: 64>, "hbm_west">, memref<64x1024xf16, strided<[1024, 1], offset: 65536>, "hbm_west">, memref<1024x1024xf16, strided<[1024, 1], offset: 1048576>, "hbm_west">
    softhier.softmax %s0_1 -> %s0_1 {scale = 0.125 : f32, cluster = 1 : i32} : memref<1024x1024xf16, strided<[1024, 1], offset: 1048576>, "hbm_west"> -> memref<1024x1024xf16, strided<[1024, 1], offset: 1048576>, "hbm_west">
    softhier.gemm %s0_1, %v0_1 into %o0_1 {fmt = "fp16", tile_m = 256 : i32, tile_n = 64 : i32, tile_k = 256 : i32, pipeline, cluster = 1 : i32} : memref<1024x1024xf16, strided<[1024, 1], offset: 1048576>, "hbm_west">, memref<1024x64xf16, strided<[768, 1], offset: 64>, "hbm_west">, memref<1024x64xf16, strided<[768, 1], offset: 64>, "hbm_west">
    %q0_2 = softhier.view %q : memref<1024x768xf16, "hbm_west"> -> memref<1024x64xf16, strided<[768, 1], offset: 128>, "hbm_west">
    %kT0_2 = softhier.view %kT : memref<768x1024xf16, "hbm_west"> -> memref<64x1024xf16, strided<[1024, 1], offset: 131072>, "hbm_west">
    %s0_2 = softhier.view %sc : memref<12288x1024xf16, "hbm_west"> -> memref<1024x1024xf16, strided<[1024, 1], offset: 2097152>, "hbm_west">
    %v0_2 = softhier.view %v : memref<1024x768xf16, "hbm_west"> -> memref<1024x64xf16, strided<[768, 1], offset: 128>, "hbm_west">
    %o0_2 = softhier.view %o : memref<1024x768xf16, "hbm_west"> -> memref<1024x64xf16, strided<[768, 1], offset: 128>, "hbm_west">
    softhier.gemm %q0_2, %kT0_2 into %s0_2 {fmt = "fp16", tile_m = 256 : i32, tile_n = 256 : i32, tile_k = 64 : i32, pipeline, cluster = 2 : i32} : memref<1024x64xf16, strided<[768, 1], offset: 128>, "hbm_west">, memref<64x1024xf16, strided<[1024, 1], offset: 131072>, "hbm_west">, memref<1024x1024xf16, strided<[1024, 1], offset: 2097152>, "hbm_west">
    softhier.softmax %s0_2 -> %s0_2 {scale = 0.125 : f32, cluster = 2 : i32} : memref<1024x1024xf16, strided<[1024, 1], offset: 2097152>, "hbm_west"> -> memref<1024x1024xf16, strided<[1024, 1], offset: 2097152>, "hbm_west">
    softhier.gemm %s0_2, %v0_2 into %o0_2 {fmt = "fp16", tile_m = 256 : i32, tile_n = 64 : i32, tile_k = 256 : i32, pipeline, cluster = 2 : i32} : memref<1024x1024xf16, strided<[1024, 1], offset: 2097152>, "hbm_west">, memref<1024x64xf16, strided<[768, 1], offset: 128>, "hbm_west">, memref<1024x64xf16, strided<[768, 1], offset: 128>, "hbm_west">
    %q0_3 = softhier.view %q : memref<1024x768xf16, "hbm_west"> -> memref<1024x64xf16, strided<[768, 1], offset: 192>, "hbm_west">
    %kT0_3 = softhier.view %kT : memref<768x1024xf16, "hbm_west"> -> memref<64x1024xf16, strided<[1024, 1], offset: 196608>, "hbm_west">
    %s0_3 = softhier.view %sc : memref<12288x1024xf16, "hbm_west"> -> memref<1024x1024xf16, strided<[1024, 1], offset: 3145728>, "hbm_west">
    %v0_3 = softhier.view %v : memref<1024x768xf16, "hbm_west"> -> memref<1024x64xf16, strided<[768, 1], offset: 192>, "hbm_west">
    %o0_3 = softhier.view %o : memref<1024x768xf16, "hbm_west"> -> memref<1024x64xf16, strided<[768, 1], offset: 192>, "hbm_west">
    softhier.gemm %q0_3, %kT0_3 into %s0_3 {fmt = "fp16", tile_m = 256 : i32, tile_n = 256 : i32, tile_k = 64 : i32, pipeline, cluster = 3 : i32} : memref<1024x64xf16, strided<[768, 1], offset: 192>, "hbm_west">, memref<64x1024xf16, strided<[1024, 1], offset: 196608>, "hbm_west">, memref<1024x1024xf16, strided<[1024, 1], offset: 3145728>, "hbm_west">
    softhier.softmax %s0_3 -> %s0_3 {scale = 0.125 : f32, cluster = 3 : i32} : memref<1024x1024xf16, strided<[1024, 1], offset: 3145728>, "hbm_west"> -> memref<1024x1024xf16, strided<[1024, 1], offset: 3145728>, "hbm_west">
    softhier.gemm %s0_3, %v0_3 into %o0_3 {fmt = "fp16", tile_m = 256 : i32, tile_n = 64 : i32, tile_k = 256 : i32, pipeline, cluster = 3 : i32} : memref<1024x1024xf16, strided<[1024, 1], offset: 3145728>, "hbm_west">, memref<1024x64xf16, strided<[768, 1], offset: 192>, "hbm_west">, memref<1024x64xf16, strided<[768, 1], offset: 192>, "hbm_west">
    %q0_4 = softhier.view %q : memref<1024x768xf16, "hbm_west"> -> memref<1024x64xf16, strided<[768, 1], offset: 256>, "hbm_west">
    %kT0_4 = softhier.view %kT : memref<768x1024xf16, "hbm_west"> -> memref<64x1024xf16, strided<[1024, 1], offset: 262144>, "hbm_west">
    %s0_4 = softhier.view %sc : memref<12288x1024xf16, "hbm_west"> -> memref<1024x1024xf16, strided<[1024, 1], offset: 4194304>, "hbm_west">
    %v0_4 = softhier.view %v : memref<1024x768xf16, "hbm_west"> -> memref<1024x64xf16, strided<[768, 1], offset: 256>, "hbm_west">
    %o0_4 = softhier.view %o : memref<1024x768xf16, "hbm_west"> -> memref<1024x64xf16, strided<[768, 1], offset: 256>, "hbm_west">
    softhier.gemm %q0_4, %kT0_4 into %s0_4 {fmt = "fp16", tile_m = 256 : i32, tile_n = 256 : i32, tile_k = 64 : i32, pipeline, cluster = 4 : i32} : memref<1024x64xf16, strided<[768, 1], offset: 256>, "hbm_west">, memref<64x1024xf16, strided<[1024, 1], offset: 262144>, "hbm_west">, memref<1024x1024xf16, strided<[1024, 1], offset: 4194304>, "hbm_west">
    softhier.softmax %s0_4 -> %s0_4 {scale = 0.125 : f32, cluster = 4 : i32} : memref<1024x1024xf16, strided<[1024, 1], offset: 4194304>, "hbm_west"> -> memref<1024x1024xf16, strided<[1024, 1], offset: 4194304>, "hbm_west">
    softhier.gemm %s0_4, %v0_4 into %o0_4 {fmt = "fp16", tile_m = 256 : i32, tile_n = 64 : i32, tile_k = 256 : i32, pipeline, cluster = 4 : i32} : memref<1024x1024xf16, strided<[1024, 1], offset: 4194304>, "hbm_west">, memref<1024x64xf16, strided<[768, 1], offset: 256>, "hbm_west">, memref<1024x64xf16, strided<[768, 1], offset: 256>, "hbm_west">
    %q0_5 = softhier.view %q : memref<1024x768xf16, "hbm_west"> -> memref<1024x64xf16, strided<[768, 1], offset: 320>, "hbm_west">
    %kT0_5 = softhier.view %kT : memref<768x1024xf16, "hbm_west"> -> memref<64x1024xf16, strided<[1024, 1], offset: 327680>, "hbm_west">
    %s0_5 = softhier.view %sc : memref<12288x1024xf16, "hbm_west"> -> memref<1024x1024xf16, strided<[1024, 1], offset: 5242880>, "hbm_west">
    %v0_5 = softhier.view %v : memref<1024x768xf16, "hbm_west"> -> memref<1024x64xf16, strided<[768, 1], offset: 320>, "hbm_west">
    %o0_5 = softhier.view %o : memref<1024x768xf16, "hbm_west"> -> memref<1024x64xf16, strided<[768, 1], offset: 320>, "hbm_west">
    softhier.gemm %q0_5, %kT0_5 into %s0_5 {fmt = "fp16", tile_m = 256 : i32, tile_n = 256 : i32, tile_k = 64 : i32, pipeline, cluster = 5 : i32} : memref<1024x64xf16, strided<[768, 1], offset: 320>, "hbm_west">, memref<64x1024xf16, strided<[1024, 1], offset: 327680>, "hbm_west">, memref<1024x1024xf16, strided<[1024, 1], offset: 5242880>, "hbm_west">
    softhier.softmax %s0_5 -> %s0_5 {scale = 0.125 : f32, cluster = 5 : i32} : memref<1024x1024xf16, strided<[1024, 1], offset: 5242880>, "hbm_west"> -> memref<1024x1024xf16, strided<[1024, 1], offset: 5242880>, "hbm_west">
    softhier.gemm %s0_5, %v0_5 into %o0_5 {fmt = "fp16", tile_m = 256 : i32, tile_n = 64 : i32, tile_k = 256 : i32, pipeline, cluster = 5 : i32} : memref<1024x1024xf16, strided<[1024, 1], offset: 5242880>, "hbm_west">, memref<1024x64xf16, strided<[768, 1], offset: 320>, "hbm_west">, memref<1024x64xf16, strided<[768, 1], offset: 320>, "hbm_west">
    %q0_6 = softhier.view %q : memref<1024x768xf16, "hbm_west"> -> memref<1024x64xf16, strided<[768, 1], offset: 384>, "hbm_west">
    %kT0_6 = softhier.view %kT : memref<768x1024xf16, "hbm_west"> -> memref<64x1024xf16, strided<[1024, 1], offset: 393216>, "hbm_west">
    %s0_6 = softhier.view %sc : memref<12288x1024xf16, "hbm_west"> -> memref<1024x1024xf16, strided<[1024, 1], offset: 6291456>, "hbm_west">
    %v0_6 = softhier.view %v : memref<1024x768xf16, "hbm_west"> -> memref<1024x64xf16, strided<[768, 1], offset: 384>, "hbm_west">
    %o0_6 = softhier.view %o : memref<1024x768xf16, "hbm_west"> -> memref<1024x64xf16, strided<[768, 1], offset: 384>, "hbm_west">
    softhier.gemm %q0_6, %kT0_6 into %s0_6 {fmt = "fp16", tile_m = 256 : i32, tile_n = 256 : i32, tile_k = 64 : i32, pipeline, cluster = 6 : i32} : memref<1024x64xf16, strided<[768, 1], offset: 384>, "hbm_west">, memref<64x1024xf16, strided<[1024, 1], offset: 393216>, "hbm_west">, memref<1024x1024xf16, strided<[1024, 1], offset: 6291456>, "hbm_west">
    softhier.softmax %s0_6 -> %s0_6 {scale = 0.125 : f32, cluster = 6 : i32} : memref<1024x1024xf16, strided<[1024, 1], offset: 6291456>, "hbm_west"> -> memref<1024x1024xf16, strided<[1024, 1], offset: 6291456>, "hbm_west">
    softhier.gemm %s0_6, %v0_6 into %o0_6 {fmt = "fp16", tile_m = 256 : i32, tile_n = 64 : i32, tile_k = 256 : i32, pipeline, cluster = 6 : i32} : memref<1024x1024xf16, strided<[1024, 1], offset: 6291456>, "hbm_west">, memref<1024x64xf16, strided<[768, 1], offset: 384>, "hbm_west">, memref<1024x64xf16, strided<[768, 1], offset: 384>, "hbm_west">
    %q0_7 = softhier.view %q : memref<1024x768xf16, "hbm_west"> -> memref<1024x64xf16, strided<[768, 1], offset: 448>, "hbm_west">
    %kT0_7 = softhier.view %kT : memref<768x1024xf16, "hbm_west"> -> memref<64x1024xf16, strided<[1024, 1], offset: 458752>, "hbm_west">
    %s0_7 = softhier.view %sc : memref<12288x1024xf16, "hbm_west"> -> memref<1024x1024xf16, strided<[1024, 1], offset: 7340032>, "hbm_west">
    %v0_7 = softhier.view %v : memref<1024x768xf16, "hbm_west"> -> memref<1024x64xf16, strided<[768, 1], offset: 448>, "hbm_west">
    %o0_7 = softhier.view %o : memref<1024x768xf16, "hbm_west"> -> memref<1024x64xf16, strided<[768, 1], offset: 448>, "hbm_west">
    softhier.gemm %q0_7, %kT0_7 into %s0_7 {fmt = "fp16", tile_m = 256 : i32, tile_n = 256 : i32, tile_k = 64 : i32, pipeline, cluster = 7 : i32} : memref<1024x64xf16, strided<[768, 1], offset: 448>, "hbm_west">, memref<64x1024xf16, strided<[1024, 1], offset: 458752>, "hbm_west">, memref<1024x1024xf16, strided<[1024, 1], offset: 7340032>, "hbm_west">
    softhier.softmax %s0_7 -> %s0_7 {scale = 0.125 : f32, cluster = 7 : i32} : memref<1024x1024xf16, strided<[1024, 1], offset: 7340032>, "hbm_west"> -> memref<1024x1024xf16, strided<[1024, 1], offset: 7340032>, "hbm_west">
    softhier.gemm %s0_7, %v0_7 into %o0_7 {fmt = "fp16", tile_m = 256 : i32, tile_n = 64 : i32, tile_k = 256 : i32, pipeline, cluster = 7 : i32} : memref<1024x1024xf16, strided<[1024, 1], offset: 7340032>, "hbm_west">, memref<1024x64xf16, strided<[768, 1], offset: 448>, "hbm_west">, memref<1024x64xf16, strided<[768, 1], offset: 448>, "hbm_west">
    %q0_8 = softhier.view %q : memref<1024x768xf16, "hbm_west"> -> memref<1024x64xf16, strided<[768, 1], offset: 512>, "hbm_west">
    %kT0_8 = softhier.view %kT : memref<768x1024xf16, "hbm_west"> -> memref<64x1024xf16, strided<[1024, 1], offset: 524288>, "hbm_west">
    %s0_8 = softhier.view %sc : memref<12288x1024xf16, "hbm_west"> -> memref<1024x1024xf16, strided<[1024, 1], offset: 8388608>, "hbm_west">
    %v0_8 = softhier.view %v : memref<1024x768xf16, "hbm_west"> -> memref<1024x64xf16, strided<[768, 1], offset: 512>, "hbm_west">
    %o0_8 = softhier.view %o : memref<1024x768xf16, "hbm_west"> -> memref<1024x64xf16, strided<[768, 1], offset: 512>, "hbm_west">
    softhier.gemm %q0_8, %kT0_8 into %s0_8 {fmt = "fp16", tile_m = 256 : i32, tile_n = 256 : i32, tile_k = 64 : i32, pipeline, cluster = 8 : i32} : memref<1024x64xf16, strided<[768, 1], offset: 512>, "hbm_west">, memref<64x1024xf16, strided<[1024, 1], offset: 524288>, "hbm_west">, memref<1024x1024xf16, strided<[1024, 1], offset: 8388608>, "hbm_west">
    softhier.softmax %s0_8 -> %s0_8 {scale = 0.125 : f32, cluster = 8 : i32} : memref<1024x1024xf16, strided<[1024, 1], offset: 8388608>, "hbm_west"> -> memref<1024x1024xf16, strided<[1024, 1], offset: 8388608>, "hbm_west">
    softhier.gemm %s0_8, %v0_8 into %o0_8 {fmt = "fp16", tile_m = 256 : i32, tile_n = 64 : i32, tile_k = 256 : i32, pipeline, cluster = 8 : i32} : memref<1024x1024xf16, strided<[1024, 1], offset: 8388608>, "hbm_west">, memref<1024x64xf16, strided<[768, 1], offset: 512>, "hbm_west">, memref<1024x64xf16, strided<[768, 1], offset: 512>, "hbm_west">
    %q0_9 = softhier.view %q : memref<1024x768xf16, "hbm_west"> -> memref<1024x64xf16, strided<[768, 1], offset: 576>, "hbm_west">
    %kT0_9 = softhier.view %kT : memref<768x1024xf16, "hbm_west"> -> memref<64x1024xf16, strided<[1024, 1], offset: 589824>, "hbm_west">
    %s0_9 = softhier.view %sc : memref<12288x1024xf16, "hbm_west"> -> memref<1024x1024xf16, strided<[1024, 1], offset: 9437184>, "hbm_west">
    %v0_9 = softhier.view %v : memref<1024x768xf16, "hbm_west"> -> memref<1024x64xf16, strided<[768, 1], offset: 576>, "hbm_west">
    %o0_9 = softhier.view %o : memref<1024x768xf16, "hbm_west"> -> memref<1024x64xf16, strided<[768, 1], offset: 576>, "hbm_west">
    softhier.gemm %q0_9, %kT0_9 into %s0_9 {fmt = "fp16", tile_m = 256 : i32, tile_n = 256 : i32, tile_k = 64 : i32, pipeline, cluster = 9 : i32} : memref<1024x64xf16, strided<[768, 1], offset: 576>, "hbm_west">, memref<64x1024xf16, strided<[1024, 1], offset: 589824>, "hbm_west">, memref<1024x1024xf16, strided<[1024, 1], offset: 9437184>, "hbm_west">
    softhier.softmax %s0_9 -> %s0_9 {scale = 0.125 : f32, cluster = 9 : i32} : memref<1024x1024xf16, strided<[1024, 1], offset: 9437184>, "hbm_west"> -> memref<1024x1024xf16, strided<[1024, 1], offset: 9437184>, "hbm_west">
    softhier.gemm %s0_9, %v0_9 into %o0_9 {fmt = "fp16", tile_m = 256 : i32, tile_n = 64 : i32, tile_k = 256 : i32, pipeline, cluster = 9 : i32} : memref<1024x1024xf16, strided<[1024, 1], offset: 9437184>, "hbm_west">, memref<1024x64xf16, strided<[768, 1], offset: 576>, "hbm_west">, memref<1024x64xf16, strided<[768, 1], offset: 576>, "hbm_west">
    %q0_10 = softhier.view %q : memref<1024x768xf16, "hbm_west"> -> memref<1024x64xf16, strided<[768, 1], offset: 640>, "hbm_west">
    %kT0_10 = softhier.view %kT : memref<768x1024xf16, "hbm_west"> -> memref<64x1024xf16, strided<[1024, 1], offset: 655360>, "hbm_west">
    %s0_10 = softhier.view %sc : memref<12288x1024xf16, "hbm_west"> -> memref<1024x1024xf16, strided<[1024, 1], offset: 10485760>, "hbm_west">
    %v0_10 = softhier.view %v : memref<1024x768xf16, "hbm_west"> -> memref<1024x64xf16, strided<[768, 1], offset: 640>, "hbm_west">
    %o0_10 = softhier.view %o : memref<1024x768xf16, "hbm_west"> -> memref<1024x64xf16, strided<[768, 1], offset: 640>, "hbm_west">
    softhier.gemm %q0_10, %kT0_10 into %s0_10 {fmt = "fp16", tile_m = 256 : i32, tile_n = 256 : i32, tile_k = 64 : i32, pipeline, cluster = 10 : i32} : memref<1024x64xf16, strided<[768, 1], offset: 640>, "hbm_west">, memref<64x1024xf16, strided<[1024, 1], offset: 655360>, "hbm_west">, memref<1024x1024xf16, strided<[1024, 1], offset: 10485760>, "hbm_west">
    softhier.softmax %s0_10 -> %s0_10 {scale = 0.125 : f32, cluster = 10 : i32} : memref<1024x1024xf16, strided<[1024, 1], offset: 10485760>, "hbm_west"> -> memref<1024x1024xf16, strided<[1024, 1], offset: 10485760>, "hbm_west">
    softhier.gemm %s0_10, %v0_10 into %o0_10 {fmt = "fp16", tile_m = 256 : i32, tile_n = 64 : i32, tile_k = 256 : i32, pipeline, cluster = 10 : i32} : memref<1024x1024xf16, strided<[1024, 1], offset: 10485760>, "hbm_west">, memref<1024x64xf16, strided<[768, 1], offset: 640>, "hbm_west">, memref<1024x64xf16, strided<[768, 1], offset: 640>, "hbm_west">
    %q0_11 = softhier.view %q : memref<1024x768xf16, "hbm_west"> -> memref<1024x64xf16, strided<[768, 1], offset: 704>, "hbm_west">
    %kT0_11 = softhier.view %kT : memref<768x1024xf16, "hbm_west"> -> memref<64x1024xf16, strided<[1024, 1], offset: 720896>, "hbm_west">
    %s0_11 = softhier.view %sc : memref<12288x1024xf16, "hbm_west"> -> memref<1024x1024xf16, strided<[1024, 1], offset: 11534336>, "hbm_west">
    %v0_11 = softhier.view %v : memref<1024x768xf16, "hbm_west"> -> memref<1024x64xf16, strided<[768, 1], offset: 704>, "hbm_west">
    %o0_11 = softhier.view %o : memref<1024x768xf16, "hbm_west"> -> memref<1024x64xf16, strided<[768, 1], offset: 704>, "hbm_west">
    softhier.gemm %q0_11, %kT0_11 into %s0_11 {fmt = "fp16", tile_m = 256 : i32, tile_n = 256 : i32, tile_k = 64 : i32, pipeline, cluster = 11 : i32} : memref<1024x64xf16, strided<[768, 1], offset: 704>, "hbm_west">, memref<64x1024xf16, strided<[1024, 1], offset: 720896>, "hbm_west">, memref<1024x1024xf16, strided<[1024, 1], offset: 11534336>, "hbm_west">
    softhier.softmax %s0_11 -> %s0_11 {scale = 0.125 : f32, cluster = 11 : i32} : memref<1024x1024xf16, strided<[1024, 1], offset: 11534336>, "hbm_west"> -> memref<1024x1024xf16, strided<[1024, 1], offset: 11534336>, "hbm_west">
    softhier.gemm %s0_11, %v0_11 into %o0_11 {fmt = "fp16", tile_m = 256 : i32, tile_n = 64 : i32, tile_k = 256 : i32, pipeline, cluster = 11 : i32} : memref<1024x1024xf16, strided<[1024, 1], offset: 11534336>, "hbm_west">, memref<1024x64xf16, strided<[768, 1], offset: 704>, "hbm_west">, memref<1024x64xf16, strided<[768, 1], offset: 704>, "hbm_west">
    softhier.group_barrier {grid_x = 4 : i32, grid_y = 4 : i32}
    softhier.gemm %o, %wo0 into %ao {fmt = "fp16", tile_m = 256 : i32, tile_n = 256 : i32, tile_k = 256 : i32, pipeline, cluster = -1 : i32} : memref<1024x768xf16, "hbm_west">, memref<768x768xf16, "hbm_west">, memref<1024x768xf16, "hbm_west">
    softhier.add_bias %ao, %bo0 -> %ao {cluster = -1 : i32} : memref<1024x768xf16, "hbm_west">, memref<1x768xf16, "hbm_west"> -> memref<1024x768xf16, "hbm_west">
    softhier.add %x, %ao -> %h {cluster = -1 : i32} : memref<1024x768xf16, "hbm_west">, memref<1024x768xf16, "hbm_west"> -> memref<1024x768xf16, "hbm_west">
    softhier.layernorm %h, %g20, %be20 -> %ln2 {eps = 1.0e-6 : f32, cluster = -1 : i32} : memref<1024x768xf16, "hbm_west">, memref<1x768xf16, "hbm_west">, memref<1x768xf16, "hbm_west"> -> memref<1024x768xf16, "hbm_west">
    softhier.gemm %ln2, %w10 into %f1 {fmt = "fp16", tile_m = 256 : i32, tile_n = 256 : i32, tile_k = 256 : i32, pipeline, cluster = -1 : i32} : memref<1024x768xf16, "hbm_west">, memref<768x3072xf16, "hbm_west">, memref<1024x3072xf16, "hbm_west">
    softhier.add_bias %f1, %b10 -> %f1 {cluster = -1 : i32} : memref<1024x3072xf16, "hbm_west">, memref<1x3072xf16, "hbm_west"> -> memref<1024x3072xf16, "hbm_west">
    softhier.gelu %f1 -> %g {cluster = -1 : i32} : memref<1024x3072xf16, "hbm_west"> -> memref<1024x3072xf16, "hbm_west">
    softhier.gemm %g, %w20 into %f2 {fmt = "fp16", tile_m = 256 : i32, tile_n = 256 : i32, tile_k = 256 : i32, pipeline, cluster = -1 : i32} : memref<1024x3072xf16, "hbm_west">, memref<3072x768xf16, "hbm_west">, memref<1024x768xf16, "hbm_west">
    softhier.add_bias %f2, %b20 -> %f2 {cluster = -1 : i32} : memref<1024x768xf16, "hbm_west">, memref<1x768xf16, "hbm_west"> -> memref<1024x768xf16, "hbm_west">
    softhier.add %h, %f2 -> %out {cluster = -1 : i32} : memref<1024x768xf16, "hbm_west">, memref<1024x768xf16, "hbm_west"> -> memref<1024x768xf16, "hbm_west">
    func.return
  }
}
