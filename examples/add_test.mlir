// Minimal test for softhier.l1_add:  b = 1.0, a = 0.5, b += a  ->  b = 1.5.
builtin.module {
  func.func @add_test() {
    %al = softhier.l1_buffer : memref<256x256xf16, "tcdm">
    %bl = softhier.l1_buffer : memref<256x256xf16, "tcdm">
    softhier.l1_fill %al {value_bits = 14336 : i32} : memref<256x256xf16, "tcdm">  // 0.5 = 0x3800
    softhier.l1_fill %bl {value_bits = 15360 : i32} : memref<256x256xf16, "tcdm">  // 1.0 = 0x3C00
    softhier.l1_add %al into %bl : memref<256x256xf16, "tcdm">, memref<256x256xf16, "tcdm">
    softhier.check_const %bl {value_bits = 15872 : i32, tol = 4 : i32} : memref<256x256xf16, "tcdm">  // 1.5 = 0x3E00
    func.return
  }
}
