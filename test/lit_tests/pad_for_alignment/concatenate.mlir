// RUN: enzymexlamlir-opt --pad-for-alignment --allow-unregistered-dialect %s | FileCheck %s

func.func @test_concatenate_dim_0(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>) {
  %0 = stablehlo.concatenate %arg0, %arg1, dim = 0 : (tensor<4x760x1533xf32>, tensor<4x760x1533xf32>) -> tensor<8x760x1533xf32>
  return
}

// CHECK: func.func @test_concatenate_dim_0(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg1, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %1 = stablehlo.pad %arg0, %cst_0, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %2 = stablehlo.concatenate %1, %0, dim = 0 : (tensor<4x768x1536xf32>, tensor<4x768x1536xf32>) -> tensor<8x768x1536xf32>
// CHECK-NEXT:     %3 = stablehlo.concatenate %arg0, %arg1, dim = 0 : (tensor<4x760x1533xf32>, tensor<4x760x1533xf32>) -> tensor<8x760x1533xf32>
// CHECK-NEXT:     return
// CHECK-NEXT: }

func.func @test_concatenate_dim_1(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>) {
  %0 = stablehlo.concatenate %arg0, %arg1, dim = 1 : (tensor<4x760x1533xf32>, tensor<4x760x1533xf32>) -> tensor<4x1520x1533xf32>
  return
}

// CHECK: func.func @test_concatenate_dim_1(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %1 = stablehlo.pad %arg1, %cst_0, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %2 = stablehlo.slice %0 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xf32>) -> tensor<4x760x1533xf32>
// CHECK-NEXT:     %3 = stablehlo.slice %1 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xf32>) -> tensor<4x760x1533xf32>
// CHECK-NEXT:     %4 = stablehlo.concatenate %2, %3, dim = 1 : (tensor<4x760x1533xf32>, tensor<4x760x1533xf32>) -> tensor<4x1520x1533xf32>
// CHECK-NEXT:     %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %5 = stablehlo.pad %4, %cst_1, low = [0, 0, 0], high = [0, 16, 3], interior = [0, 0, 0] : (tensor<4x1520x1533xf32>, tensor<f32>) -> tensor<4x1536x1536xf32>
// CHECK-NEXT:     return
// CHECK-NEXT: }
