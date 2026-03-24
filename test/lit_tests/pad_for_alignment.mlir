// RUN: enzymexlamlir-opt --pad-for-alignment %s | FileCheck %s

func.func @test_func_arg(%arg0: tensor<4x760x1533xf32>) -> () {
  return
}

// CHECK: func.func @test_func_arg(%arg0: tensor<4x760x1533xf32>) -> () {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.0> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst : tensor<4x760x1533xf32> to tensor<4x768x1536xf32>
// CHECK-NEXT:     return
// CHECK-NEXT: }
