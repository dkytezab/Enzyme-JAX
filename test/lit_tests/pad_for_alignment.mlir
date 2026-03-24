// RUN: enzymexlamlir-opt --pad-for-alignment %s | FileCheck %s

func.func @test_arg(%arg0: tensor<4x760x1533xf32>) -> () {
  return
}

// CHECK: func.func @test_arg(%arg0: tensor<4x760x1533xf32>) -> () {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.slice %0 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xf32>) -> tensor<4x760x1533xf32>
// CHECK-NEXT:     return
// CHECK-NEXT: }

func.func @test_arg_ret(%arg0: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
  return %arg0 : tensor<4x760x1533xf32>
}

// CHECK: func.func @test_arg_ret(%arg0: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.slice %0 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xf32>) -> tensor<4x760x1533xf32>
// CHECK-NEXT:     return %arg0 : tensor<4x760x1533xf32>
// CHECK-NEXT: }

func.func @test_constant() -> () {
  %0 = stablehlo.constant dense<0.0> : tensor<4x760x1533xf32>
  return
}

// CHECK: func.func @test_constant() {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<4x760x1533xf32>
// CHECK-NEXT:     %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %cst, %cst_0, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.slice %0 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xf32>) -> tensor<4x760x1533xf32>
// CHECK-NEXT:     return
// CHECK-NEXT: }
