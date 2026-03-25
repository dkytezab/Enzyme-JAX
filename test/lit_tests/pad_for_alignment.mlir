// RUN: enzymexlamlir-opt --pad-for-alignment %s | FileCheck %s

func.func @test_arg(%arg0: tensor<4x760x1533xf32>) -> () {
  return
}

// CHECK: func.func @test_arg(%arg0: tensor<4x760x1533xf32>) -> () {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     return
// CHECK-NEXT: }

func.func @test_arg_ret(%arg0: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
  return %arg0 : tensor<4x760x1533xf32>
}

// CHECK: func.func @test_arg_ret(%arg0: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.slice %0 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xf32>) -> tensor<4x760x1533xf32>
// CHECK-NEXT:     return %1 : tensor<4x760x1533xf32>
// CHECK-NEXT: }

func.func @test_constant() -> () {
  %0 = stablehlo.constant dense<0.0> : tensor<4x760x1533xf32>
  return
}

// CHECK: func.func @test_constant() {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<4x768x1536xf32>
// CHECK-NEXT:     %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<4x760x1533xf32>
// CHECK-NEXT:     return
// CHECK-NEXT: }

func.func @test_constant_ret() -> tensor<4x760x1533xf32> {
  %0 = stablehlo.constant dense<0.0> : tensor<4x760x1533xf32>
  return %0 : tensor<4x760x1533xf32>
}

// CHECK: func.func @test_constant_ret() -> tensor<4x760x1533xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<4x768x1536xf32>
// CHECK-NEXT:     %0 = stablehlo.slice %cst [0:4, 0:760, 0:1533] : (tensor<4x768x1536xf32>) -> tensor<4x760x1533xf32>
// CHECK-NEXT:     %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<4x760x1533xf32>
// CHECK-NEXT:     return %0 : tensor<4x760x1533xf32>
// CHECK-NEXT: }

func.func @test_pad(%arg0: tensor<4x760x1533xf32>) -> () {
  %cst = stablehlo.constant dense<0.0> : tensor<f32>
  %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [1, 0, 1], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<5x760x1534xf32>
  return
}

// CHECK: func.func @test_pad(%arg0: tensor<4x760x1533xf32>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %1 = stablehlo.pad %0, %cst_0, low = [0, 0, 0], high = [1, 0, 0], interior = [0, 0, 0] : (tensor<4x768x1536xf32>, tensor<f32>) -> tensor<5x768x1536xf32>
// CHECK-NEXT:     %2 = stablehlo.pad %arg0, %cst_0, low = [0, 0, 0], high = [1, 0, 1], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<5x760x1534xf32>
// CHECK-NEXT:     return
// CHECK-NEXT: }

func.func @test_pad_ret(%arg0: tensor<4x760x1533xf32>) -> tensor<5x760x1534xf32> {
  %cst = stablehlo.constant dense<0.0> : tensor<f32>
  %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [1, 0, 1], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<5x760x1534xf32>
  return %0 : tensor<5x760x1534xf32>
}

// CHECK-NEXT: func.func @test_pad_ret(%arg0: tensor<4x760x1533xf32>) -> tensor<5x760x1534xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %1 = stablehlo.pad %0, %cst_0, low = [0, 0, 0], high = [1, 0, 0], interior = [0, 0, 0] : (tensor<4x768x1536xf32>, tensor<f32>) -> tensor<5x768x1536xf32>
// CHECK-NEXT:     %2 = stablehlo.slice %1 [0:5, 0:760, 0:1534] : (tensor<5x768x1536xf32>) -> tensor<5x760x1534xf32>
// CHECK-NEXT:     %3 = stablehlo.pad %arg0, %cst_0, low = [0, 0, 0], high = [1, 0, 1], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<5x760x1534xf32>
// CHECK-NEXT:     return %2 : tensor<5x760x1534xf32>
// CHECK-NEXT: }

func.func @test_transpose(%arg0: tensor<4x760x1533xf32>) -> () {
  %0 = stablehlo.transpose %arg0, dims = [0, 1, 2] : (tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return
}

// CHECK: func.func @test_transpose(%arg0: tensor<4x760x1533xf32>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.slice %0 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xf32>) -> tensor<4x760x1533xf32>
// CHECK-NEXT:     %2 = stablehlo.transpose %0, dims = [0, 1, 2] : (tensor<4x768x1536xf32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %3 = stablehlo.transpose %arg0, dims = [0, 1, 2] : (tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
// CHECK-NEXT:     %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %4 = stablehlo.pad %3, %cst_0, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     return
// CHECK-NEXT: }