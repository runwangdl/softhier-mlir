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
// CHECK: volatile _Float16 *a{{[0-9]+}} = (volatile _Float16 *)local(0);
// CHECK: d{{[0-9]+}}[i{{[0-9]+}}] += a{{[0-9]+}}[i{{[0-9]+}}];
// CHECK: MLP_PASS
