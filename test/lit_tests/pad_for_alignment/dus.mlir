// RUN: enzymexlamlir-opt --pad-for-alignment --allow-unregistered-dialect %s | FileCheck %s

func.func @test_dus(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>) {
  %cst = stablehlo.constant dense<0> : tensor<i64>
  %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %cst, %cst, %cst : (tensor<4x760x1533xf32>, tensor<4x760x1533xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<4x760x1533xf32>
  return
}

// CHECK: func.func @test_dus(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %1 = stablehlo.pad %arg1, %cst_0, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:     %2 = stablehlo.dynamic_update_slice %0, %1, %c, %c, %c : (tensor<4x768x1536xf32>, tensor<4x768x1536xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     return
// CHECK-NEXT: }

func.func @test_dus_ret(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
  %cst = stablehlo.constant dense<0> : tensor<i64>
  %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %cst, %cst, %cst : (tensor<4x760x1533xf32>, tensor<4x760x1533xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<4x760x1533xf32>
  return %0 : tensor<4x760x1533xf32>
}

// CHECK: func.func @test_dus_ret(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg1, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %1 = stablehlo.pad %arg0, %cst_0, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:     %2 = stablehlo.dynamic_update_slice %1, %0, %c, %c, %c : (tensor<4x768x1536xf32>, tensor<4x768x1536xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %3 = stablehlo.slice %2 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xf32>) -> tensor<4x760x1533xf32>
// CHECK-NEXT:     return %3 : tensor<4x760x1533xf32>
// CHECK-NEXT: }
