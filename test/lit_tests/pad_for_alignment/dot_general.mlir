// RUN: enzymexlamlir-opt --pad-for-alignment --allow-unregistered-dialect %s | FileCheck %s

func.func @test_dot_general(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x1533x760xf32>) {
  %0 = stablehlo.dot_general %arg0, %arg1, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<4x760x1533xf32>, tensor<4x1533x760xf32>) -> tensor<4x760x760xf32>
  return
}

// CHECK: func.func @test_dot_general(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x1533x760xf32>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg1, %cst, low = [0, 0, 0], high = [0, 3, 8], interior = [0, 0, 0] : (tensor<4x1533x760xf32>, tensor<f32>) -> tensor<4x1536x768xf32>
// CHECK-NEXT:     %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %1 = stablehlo.pad %arg0, %cst_0, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %2 = stablehlo.dot_general %1, %0, batching_dims = [0] x [0], contracting_dims = [2] x [1] : (tensor<4x768x1536xf32>, tensor<4x1536x768xf32>) -> tensor<4x768x768xf32>
// CHECK-NEXT:     return
// CHECK-NEXT: }
