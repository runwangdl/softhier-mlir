// RUN: softhier-translate %s | filecheck %s
// softhier.attention lowers to one sh_attention call: S, D, heads and the four leading dimensions come from
// the (possibly strided) operand types; cluster = -1 deals the heads over all clusters (SH_ALL).
builtin.module {
  func.func @attention() {
    %qkv = softhier.hbm_buffer {offset = 0 : i32}       : memref<256x2304xf16, "hbm_west">
    %o   = softhier.hbm_buffer {offset = 1179648 : i32} : memref<256x768xf16, "hbm_west">
    %q = softhier.view %qkv : memref<256x2304xf16, "hbm_west"> -> memref<256x768xf16, strided<[2304, 1], offset: 0>, "hbm_west">
    %k = softhier.view %qkv : memref<256x2304xf16, "hbm_west"> -> memref<256x768xf16, strided<[2304, 1], offset: 768>, "hbm_west">
    %v = softhier.view %qkv : memref<256x2304xf16, "hbm_west"> -> memref<256x768xf16, strided<[2304, 1], offset: 1536>, "hbm_west">
    softhier.attention %q, %k, %v -> %o {scale = 0.125 : f32, heads = 12 : i32, cluster = -1 : i32}
        : memref<256x768xf16, strided<[2304, 1], offset: 0>, "hbm_west">, memref<256x768xf16, strided<[2304, 1], offset: 768>, "hbm_west">,
          memref<256x768xf16, strided<[2304, 1], offset: 1536>, "hbm_west"> -> memref<256x768xf16, "hbm_west">
    softhier.attention %o, %o, %o -> %o {scale = 0.25 : f32, heads = 24 : i32, cluster = 3 : i32}
        : memref<256x768xf16, "hbm_west">, memref<256x768xf16, "hbm_west">, memref<256x768xf16, "hbm_west"> -> memref<256x768xf16, "hbm_west">
    // q_block (rows per work item) is a policy attribute: it selects sh_attention_q
    softhier.attention %q, %k, %v -> %o {scale = 0.125 : f32, heads = 12 : i32, q_block = 64 : i32, cluster = -1 : i32}
        : memref<256x768xf16, strided<[2304, 1], offset: 0>, "hbm_west">, memref<256x768xf16, strided<[2304, 1], offset: 768>, "hbm_west">,
          memref<256x768xf16, strided<[2304, 1], offset: 1536>, "hbm_west"> -> memref<256x768xf16, "hbm_west">
    func.return
  }
}
// CHECK: sh_attention(hb1, (hb1 + 1536), (hb1 + 3072), hb2, 256, 768, 12, 2304, 2304, 2304, 768, 0.125f, SH_ALL);
// CHECK: sh_attention(hb2, hb2, hb2, hb2, 256, 768, 24, 768, 768, 768, 768, 0.25f, 3);
// CHECK: sh_attention_q(hb1, (hb1 + 1536), (hb1 + 3072), hb2, 256, 768, 12, 2304, 2304, 2304, 768, 0.125f, SH_ALL, 64);
