// A 512x512x512 fp16 GEMM — larger than one 256 tile, so the backend must emit
// a tile loop-nest (2x2 output tiles, 2 K-steps each = 8 RedMule GEMMs) with
// K-accumulation. Verifiable by construction:
//   X = 1.0    (fp16 0x3C00 = 15360)
//   W = 1/512  (fp16 0x1800 = 6144)
//   Z[i,j] = sum_{k=0..511} 1.0 * (1/512) = 1.0
// Each K-tile contributes 256*(1/512) = 0.5; the two must accumulate to 1.0, so
// the check fails unless K-accumulation across tiles is correct.
builtin.module {
  func.func @gemm512() {
    %xh = softhier.hbm_buffer {offset = 0 : i32}       : memref<512x512xf16, "hbm_west">
    %wh = softhier.hbm_buffer {offset = 524288 : i32}  : memref<512x512xf16, "hbm_south">
    %zh = softhier.hbm_buffer {offset = 1048576 : i32} : memref<512x512xf16, "hbm_west">

    softhier.hbm_fill %xh {value_bits = 15360 : i32} : memref<512x512xf16, "hbm_west">
    softhier.hbm_fill %wh {value_bits = 6144 : i32}  : memref<512x512xf16, "hbm_south">

    softhier.gemm %xh, %wh into %zh {fmt = "fp16"}
        : memref<512x512xf16, "hbm_west">, memref<512x512xf16, "hbm_south">, memref<512x512xf16, "hbm_west">

    softhier.hbm_check_const %zh {value_bits = 15360 : i32, tol = 16 : i32} : memref<512x512xf16, "hbm_west">
    func.return
  }
}
