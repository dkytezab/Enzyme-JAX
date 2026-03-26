// RUN: enzymexlamlir-opt --pad-for-alignment --allow-unregistered-dialect %s | FileCheck %s

func.func @test_select(%arg0: tensor<4x760x1533xi1>, %arg1: tensor<4x760x1533xf32>, %arg2: tensor<4x760x1533xf32>) {
  %0 = stablehlo.select %arg0, %arg1, %arg2 : (tensor<4x760x1533xi1>, tensor<4x760x1533xf32>, tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return
}

// CHECK: func.func @test_select(%arg0: tensor<4x760x1533xi1>, %arg1: tensor<4x760x1533xf32>, %arg2: tensor<4x760x1533xf32>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg2, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %c = stablehlo.constant dense<false> : tensor<i1>
// CHECK-NEXT:     %1 = stablehlo.pad %arg0, %c, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xi1>, tensor<i1>) -> tensor<4x768x1536xi1>
// CHECK-NEXT:     %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %2 = stablehlo.pad %arg1, %cst_0, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %3 = stablehlo.select %1, %2, %0 : tensor<4x768x1536xi1>, tensor<4x768x1536xf32>
// CHECK-NEXT:     return
// CHECK-NEXT: }

func.func @test_select_ret(%arg0: tensor<4x760x1533xi1>, %arg1: tensor<4x760x1533xf32>, %arg2: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
  %0 = stablehlo.select %arg0, %arg1, %arg2 : (tensor<4x760x1533xi1>, tensor<4x760x1533xf32>, tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return %0 : tensor<4x760x1533xf32>
}

// CHECK: func.func @test_select_ret(%arg0: tensor<4x760x1533xi1>, %arg1: tensor<4x760x1533xf32>, %arg2: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
// CHECK-NEXT:     %c = stablehlo.constant dense<false> : tensor<i1>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %c, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xi1>, tensor<i1>) -> tensor<4x768x1536xi1>
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %1 = stablehlo.pad %arg1, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %2 = stablehlo.pad %arg2, %cst_0, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %3 = stablehlo.select %0, %1, %2 : tensor<4x768x1536xi1>, tensor<4x768x1536xf32>
// CHECK-NEXT:     %4 = stablehlo.slice %3 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xf32>) -> tensor<4x760x1533xf32>
// CHECK-NEXT:     return %4 : tensor<4x760x1533xf32>
// CHECK-NEXT: }
