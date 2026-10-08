// RUN: softhier-translate %s | filecheck %s
// softhier.l1_add emits an in-place fp16 dst += src loop.
builtin.module {
  func.func @add_test() {
    %al = softhier.l1_buffer : memref<256x256xf16, "tcdm">
    %bl = softhier.l1_buffer : memref<256x256xf16, "tcdm">
    softhier.l1_fill %al {value_bits = 14336 : i32} : memref<256x256xf16, "tcdm">
    softhier.l1_fill %bl {value_bits = 15360 : i32} : memref<256x256xf16, "tcdm">
    softhier.l1_add %al into %bl : memref<256x256xf16, "tcdm">, memref<256x256xf16, "tcdm">
    softhier.check_const %bl {value_bits = 15872 : i32, tol = 4 : i32} : memref<256x256xf16, "tcdm">
    func.return
  }
}
// CHECK: sh_l1_add_fp16(l1b2, l1b1, 65536)
// CHECK: sh_test_check_const_l1_fp16(l1b2, 65536, 15872u, 4, "ADD_TEST")
