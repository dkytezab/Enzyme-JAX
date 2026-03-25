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
// CHECK-NEXT:     return
// CHECK-NEXT: }

func.func @test_constant_ret() -> tensor<4x760x1533xf32> {
  %0 = stablehlo.constant dense<0.0> : tensor<4x760x1533xf32>
  return %0 : tensor<4x760x1533xf32>
}

// CHECK: func.func @test_constant_ret() -> tensor<4x760x1533xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<4x768x1536xf32>
// CHECK-NEXT:     %0 = stablehlo.slice %cst [0:4, 0:760, 0:1533] : (tensor<4x768x1536xf32>) -> tensor<4x760x1533xf32>
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
// CHECK-NEXT:     return %2 : tensor<5x760x1534xf32>
// CHECK-NEXT: }

func.func @test_slice_identity(%arg0: tensor<4x760x1533xf32>) -> () {
  %0 = stablehlo.slice %arg0 [0:4, 0:760, 0:1533] : (tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return
}

// CHECK: func.func @test_slice_identity(%arg0: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.slice %0 [0:4, 0:768, 0:1536] : (tensor<4x768x1536xf32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     return
// CHECK-NEXT: }

func.func @test_slice_identity_ret(%arg0: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
  %0 = stablehlo.slice %arg0 [0:4, 0:760, 0:1533] : (tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return %0 : tensor<4x760x1533xf32>
}

// CHECK: func.func @test_slice_identity_ret(%arg0: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.slice %0 [0:4, 0:768, 0:1536] : (tensor<4x768x1536xf32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %2 = stablehlo.slice %1 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xf32>) -> tensor<4x760x1533xf32>
// CHECK-NEXT:     return %2 : tensor<4x760x1533xf32>
// CHECK-NEXT: }

func.func @test_slice_needs_pad_after(%arg0: tensor<4x760x1533xf32>) -> () {
  %0 = stablehlo.slice %arg0 [0:4, 160:760, 0:1533] : (tensor<4x760x1533xf32>) -> tensor<4x600x1533xf32>
  return
}

// CHECK: func.func @test_slice_needs_pad_after(%arg0: tensor<4x760x1533xf32>) -> tensor<4x600x1533xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.slice %0 [0:4, 160:768, 0:1536] : (tensor<4x768x1536xf32>) -> tensor<4x608x1536xf32>
// CHECK-NEXT:     %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %2 = stablehlo.pad %1, %cst_0, low = [0, 0, 0], high = [0, 32, 0], interior = [0, 0, 0] : (tensor<4x608x1536xf32>, tensor<f32>) -> tensor<4x640x1536xf32>
// CHECK-NEXT:     return
// CHECK-NEXT: }

func.func @test_slice_needs_pad_after_ret(%arg0: tensor<4x760x1533xf32>) -> tensor<4x600x1533xf32> {
  %0 = stablehlo.slice %arg0 [0:4, 160:760, 0:1533] : (tensor<4x760x1533xf32>) -> tensor<4x600x1533xf32>
  return %0 : tensor<4x600x1533xf32>
}

// CHECK: func.func @test_slice_needs_pad_after_ret(%arg0: tensor<4x760x1533xf32>) -> tensor<4x600x1533xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.slice %0 [0:4, 160:768, 0:1536] : (tensor<4x768x1536xf32>) -> tensor<4x608x1536xf32>
// CHECK-NEXT:     %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %2 = stablehlo.pad %1, %cst_0, low = [0, 0, 0], high = [0, 32, 0], interior = [0, 0, 0] : (tensor<4x608x1536xf32>, tensor<f32>) -> tensor<4x640x1536xf32>
// CHECK-NEXT:     %3 = stablehlo.slice %2 [0:4, 0:600, 0:1533] : (tensor<4x640x1536xf32>) -> tensor<4x600x1533xf32>
// CHECK-NEXT:     return %3 : tensor<4x600x1533xf32>
// CHECK-NEXT: }

func.func @test_dus(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>) -> () {
  %cst = stablehlo.constant dense<0> : tensor<i64>
  %0 = stablehlo.dynamic_update_slice %arg0, %arg1, %cst, %cst, %cst : (tensor<4x760x1533xf32>, tensor<4x760x1533xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<4x760x1533xf32>
  return
}

// CHECK-NEXT: func.func @test_dus(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>) {
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

func.func @test_select(%arg0: tensor<4x760x1533xi1>, %arg1: tensor<4x760x1533xf32>, %arg2: tensor<4x760x1533xf32>) -> () {
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

func.func @test_concatenate_dim_0(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>) -> () {
  %0 = stablehlo.concatenate %arg0, %arg1, dim = 0 : (tensor<4x760x1533xf32>, tensor<4x760x1533xf32>) -> tensor<8x760x1533xf32>
  return
}

// CHECK-NEXT: func.func @test_concatenate_dim_0(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg1, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %1 = stablehlo.pad %arg0, %cst_0, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %2 = stablehlo.slice %1 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xf32>) -> tensor<4x760x1533xf32>
// CHECK-NEXT:     %3 = stablehlo.slice %0 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xf32>) -> tensor<4x760x1533xf32>
// CHECK-NEXT:     %4 = stablehlo.concatenate %2, %3, dim = 0 : (tensor<4x760x1533xf32>, tensor<4x760x1533xf32>) -> tensor<8x760x1533xf32>
// CHECK-NEXT:     %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %5 = stablehlo.pad %4, %cst_1, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<8x760x1533xf32>, tensor<f32>) -> tensor<8x768x1536xf32>
// CHECK-NEXT:     return
// CHECK-NEXT: }

func.func @test_concatenate_dim_1(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>) -> () {
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

func.func @test_elementwise_abs(%arg0: tensor<4x760x1533xf32>) -> () {
  %0 = stablehlo.abs %arg0 : (tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return
}

// CHECK: func.func @test_elementwise_abs(%arg0: tensor<4x760x1533xf32>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.abs %0 : tensor<4x768x1536xf32>
// CHECK-NEXT:     return
// CHECK-NEXT: }

func.func @test_elementwise_cbrt(%arg0: tensor<4x760x1533xf32>) -> () {
  %0 = stablehlo.cbrt %arg0 : (tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return
}

// CHECK: func.func @test_elementwise_cbrt(%arg0: tensor<4x760x1533xf32>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.cbrt %0 : tensor<4x768x1536xf32>
// CHECK-NEXT:     return
// CHECK-NEXT: }

func.func @test_elementwise_ceil(%arg0: tensor<4x760x1533xf32>) -> () {
  %0 = stablehlo.ceil %arg0 : (tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return
}

// CHECK: func.func @test_elementwise_ceil(%arg0: tensor<4x760x1533xf32>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.ceil %0 : tensor<4x768x1536xf32>
// CHECK-NEXT:     return
// CHECK-NEXT: }

func.func @test_elementwise_convert(%arg0: tensor<4x760x1533xf32>) -> () {
  %0 = stablehlo.convert %arg0 : (tensor<4x760x1533xf32>) -> tensor<4x760x1533xf64>
  return
}

// CHECK: func.func @test_elementwise_cosine(%arg0: tensor<4x760x1533xf32>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.convert %0 : tensor<4x768x1536xf32> -> tensor<4x768x1536xf64>
// CHECK-NEXT:     return
// CHECK-NEXT: }

func.func @test_elementwise_cosine(%arg0: tensor<4x760x1533xf32>) -> () {
  %0 = stablehlo.cosine %arg0 : (tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return
}

// CHECK: func.func @test_elementwise_cosine(%arg0: tensor<4x760x1533xf32>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.cosine %0 : tensor<4x768x1536xf32>
// CHECK-NEXT:     return
// CHECK-NEXT: }

func.func @test_elementwise_count_leading_zeros(%arg0: tensor<4x760x1533xi64>) -> () {
  %0 = stablehlo.count_leading_zeros %arg0 : (tensor<4x760x1533xi64>) -> tensor<4x760x1533xi64>
  return
}

// CHECK: func.func @test_elementwise_count_leading_zeros(%arg0: tensor<4x760x1533xi64>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xi64>, tensor<f32>) -> tensor<4x768x1536xi64>
// CHECK-NEXT:     %1 = stablehlo.count_leading_zeros %0 : tensor<4x768x1536xi64>
// CHECK-NEXT:     return
// CHECK-NEXT: }

func.func @test_elementwise_exponential(%arg0: tensor<4x760x1533xf32>) -> () {
  %0 = stablehlo.exponential %arg0 : (tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return
}

// CHECK: func.func @test_elementwise_exponential(%arg0: tensor<4x760x1533xf32>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.exponential %0 : tensor<4x768x1536xf32>
// CHECK-NEXT:     return
// CHECK-NEXT: }

func.func @test_elementwise_exponential_minus_one(%arg0: tensor<4x760x1533xf32>) -> () {
  %0 = stablehlo.exponential_minus_one %arg0 : (tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return
}

// CHECK: func.func @test_elementwise_exponential_minus_one(%arg0: tensor<4x760x1533xf32>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.exponential_minus_one %0 : tensor<4x768x1536xf32>
// CHECK-NEXT:     return
// CHECK-NEXT: }

func.func @test_elementwise_floor(%arg0: tensor<4x760x1533xf32>) -> () {
  %0 = stablehlo.floor %arg0 : (tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return
}

// CHECK: func.func @test_elementwise_floor(%arg0: tensor<4x760x1533xf32>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.floor %0 : tensor<4x768x1536xf32>
// CHECK-NEXT:     return
// CHECK-NEXT: }

func.func @test_elementwise_imag(%arg0: tensor<4x760x1533xcomplex<f32>>) -> () {
  %0 = stablehlo.imag %arg0 : (tensor<4x760x1533xcomplex<f32>>) -> tensor<4x760x1533xf32>
  return
}

// CHECK: func.func @test_elementwise_imag(%arg0: tensor<4x760x1533xf32>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.imag %0 : tensor<4x768x1536xf32>
// CHECK-NEXT:     return
// CHECK-NEXT: }

func.func @test_elementwise_is_finite(%arg0: tensor<4x760x1533xf32>) -> () {
  %0 = stablehlo.is_finite %arg0 : (tensor<4x760x1533xf32>) -> tensor<4x760x1533xi1>
  return
}

// CHECK: func.func @test_elementwise_is_finite(%arg0: tensor<4x760x1533xf32>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.is_finite %0 : tensor<4x768x1536xf32>
// CHECK-NEXT:     return
// CHECK-NEXT: }

func.func @test_elementwise_log(%arg0: tensor<4x760x1533xf32>) -> () {
  %0 = stablehlo.log %arg0 : (tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return
}

// CHECK: func.func @test_elementwise_log(%arg0: tensor<4x760x1533xf32>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.log %0 : tensor<4x768x1536xf32>
// CHECK-NEXT:     return
// CHECK-NEXT: }

func.func @test_elementwise_log_plus_one(%arg0: tensor<4x760x1533xf32>) -> () {
  %0 = stablehlo.log_plus_one %arg0 : (tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return
}

// CHECK: func.func @test_elementwise_log_plus_one(%arg0: tensor<4x760x1533xf32>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.log_plus_one %0 : tensor<4x768x1536xf32>
// CHECK-NEXT:     return
// CHECK-NEXT: }

func.func @test_elementwise_logistic(%arg0: tensor<4x760x1533xf32>) -> () {
  %0 = stablehlo.logistic %arg0 : (tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return
}

// CHECK: func.func @test_elementwise_logistic(%arg0: tensor<4x760x1533xf32>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.logistic %0 : tensor<4x768x1536xf32>
// CHECK-NEXT:     return
// CHECK-NEXT: }

func.func @test_elementwise_negate(%arg0: tensor<4x760x1533xf32>) -> () {
  %0 = stablehlo.negate %arg0 : (tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return
}

// CHECK: func.func @test_elementwise_negate(%arg0: tensor<4x760x1533xf32>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.negate %0 : tensor<4x768x1536xf32>
// CHECK-NEXT:     return
// CHECK-NEXT: }

func.func @test_elementwise_not(%arg0: tensor<4x760x1533xi1>) -> () {
  %0 = stablehlo.not %arg0 : (tensor<4x760x1533xi1>) -> tensor<4x760x1533xi1>
  return
}

// CHECK: func.func @test_elementwise_not(%arg0: tensor<4x760x1533xf32>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.not %0 : tensor<4x768x1536xf32>
// CHECK-NEXT:     return
// CHECK-NEXT: }

func.func @test_elementwise_popcnt(%arg0: tensor<4x760x1533xi64>) -> () {
  %0 = stablehlo.popcnt %arg0 : (tensor<4x760x1533xi64>) -> tensor<4x760x1533xi64>
  return
}

// CHECK: func.func @test_elementwise_popcnt(%arg0: tensor<4x760x1533xf32>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.popcnt %0 : tensor<4x768x1536xf32>
// CHECK-NEXT:     return
// CHECK-NEXT: }

func.func @test_elementwise_real(%arg0: tensor<4x760x1533xcomplex<f32>>) -> () {
  %0 = stablehlo.real %arg0 : (tensor<4x760x1533xcomplex<f32>>) -> tensor<4x760x1533xf32>
  return
}

// CHECK: func.func @test_elementwise_real(%arg0: tensor<4x760x1533xf32>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.real %0 : tensor<4x768x1536xf32>
// CHECK-NEXT:     return
// CHECK-NEXT: }

func.func @test_elementwise_reduce_precision(%arg0: tensor<4x760x1533xf32>) -> () {
  %0 = stablehlo.reduce_precision %arg0, format = e5m10 : (tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return
}

// CHECK: func.func @test_elementwise_reduce_precision(%arg0: tensor<4x760x1533xf32>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.reduce_precision %0, format = e5m10 : (tensor<4x768x1536xf32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     return
// CHECK-NEXT: }

func.func @test_elementwise_round_nearest_afz(%arg0: tensor<4x760x1533xf32>) -> () {
  %0 = stablehlo.round_nearest_afz %arg0 : (tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return
}

// CHECK: func.func @test_elementwise_round_nearest_afz(%arg0: tensor<4x760x1533xf32>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.round_nearest_afz %0 : tensor<4x768x1536xf32>
// CHECK-NEXT:     return
// CHECK-NEXT: }

func.func @test_elementwise_round_nearest_even(%arg0: tensor<4x760x1533xf32>) -> () {
  %0 = stablehlo.round_nearest_even %arg0 : (tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return
}

// CHECK: func.func @test_elementwise_round_nearest_even(%arg0: tensor<4x760x1533xf32>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.round_nearest_even %0 : tensor<4x768x1536xf32>
// CHECK-NEXT:     return
// CHECK-NEXT: }

func.func @test_elementwise_rsqrt(%arg0: tensor<4x760x1533xf32>) -> () {
  %0 = stablehlo.rsqrt %arg0 : (tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return
}

// CHECK: func.func @test_elementwise_rsqrt(%arg0: tensor<4x760x1533xf32>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.rsqrt %0 : tensor<4x768x1536xf32>
// CHECK-NEXT:     return
// CHECK-NEXT: }

func.func @test_elementwise_sign(%arg0: tensor<4x760x1533xf32>) -> () {
  %0 = stablehlo.sign %arg0 : (tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return
}

// CHECK: func.func @test_elementwise_sign(%arg0: tensor<4x760x1533xf32>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.sign %0 : tensor<4x768x1536xf32>
// CHECK-NEXT:     return
// CHECK-NEXT: }

func.func @test_elementwise_sine(%arg0: tensor<4x760x1533xf32>) -> () {
  %0 = stablehlo.sine %arg0 : (tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return
}

// CHECK: func.func @test_elementwise_sine(%arg0: tensor<4x760x1533xf32>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.sine %0 : tensor<4x768x1536xf32>
// CHECK-NEXT:     return
// CHECK-NEXT: }

func.func @test_elementwise_sqrt(%arg0: tensor<4x760x1533xf32>) -> () {
  %0 = stablehlo.sqrt %arg0 : (tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return
}

// CHECK: func.func @test_elementwise_sqrt(%arg0: tensor<4x760x1533xf32>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.sqrt %0 : tensor<4x768x1536xf32>
// CHECK-NEXT:     return
// CHECK-NEXT: }

func.func @test_elementwise_tan(%arg0: tensor<4x760x1533xf32>) -> () {
  %0 = stablehlo.tan %arg0 : (tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return
}

// CHECK: func.func @test_elementwise_tan(%arg0: tensor<4x760x1533xf32>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.tan %0 : tensor<4x768x1536xf32>
// CHECK-NEXT:     return
// CHECK-NEXT: }

func.func @test_elementwise_tanh(%arg0: tensor<4x760x1533xf32>) -> () {
  %0 = stablehlo.tanh %arg0 : (tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return
}

// CHECK: func.func @test_elementwise_tanh(%arg0: tensor<4x760x1533xf32>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.tanh %0 : tensor<4x768x1536xf32>
// CHECK-NEXT:     return
// CHECK-NEXT: }

func.func @test_elementwise_add(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>) -> () {
  %0 = stablehlo.add %arg0, %arg1 : (tensor<4x760x1533xf32>, tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return
}

// CHECK: func.func @test_elementwise_add(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %1 = stablehlo.pad %arg1, %cst_0, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %2 = stablehlo.add %0, %1 : tensor<4x768x1536xf32>
// CHECK-NEXT:     return
// CHECK-NEXT: }

func.func @test_elementwise_atan2(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>) -> () {
  %0 = stablehlo.atan2 %arg0, %arg1 : (tensor<4x760x1533xf32>, tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return
}

// CHECK: func.func @test_elementwise_add(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %1 = stablehlo.pad %arg1, %cst_0, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %2 = stablehlo.atan2 %0, %1 : tensor<4x768x1536xf32>
// CHECK-NEXT:     return
// CHECK-NEXT: }

func.func @test_elementwise_and(%arg0: tensor<4x760x1533xi1>, %arg1: tensor<4x760x1533xi1>) -> () {
  %0 = stablehlo.and %arg0, %arg1 : (tensor<4x760x1533xi1>, tensor<4x760x1533xi1>) -> tensor<4x760x1533xi1>
  return
}

// CHECK: func.func @test_elementwise_and(%arg0: tensor<4x760x1533xi1>, %arg1: tensor<4x760x1533xi1>) {
// CHECK-NEXT:     %c = stablehlo.constant dense<false> : tensor<i1>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %c, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xi1>, tensor<i1>) -> tensor<4x768x1536xi1>
// CHECK-NEXT:     %c_0 = stablehlo.constant dense<false> : tensor<i1>
// CHECK-NEXT:     %1 = stablehlo.pad %arg1, %c_0, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xi1>, tensor<i1>) -> tensor<4x768x1536xi1>
// CHECK-NEXT:     %2 = stablehlo.and %0, %1 : tensor<4x768x1536xi1>
// CHECK-NEXT:     return
// CHECK-NEXT: }

func.func @test_elementwise_compare(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>) -> () {
  %0 = stablehlo.compare LT, %arg0, %arg1 : (tensor<4x760x1533xf32>, tensor<4x760x1533xf32>) -> tensor<4x760x1533xi1>
  return
}

// CHECK: func.func @test_elementwise_compare(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg1, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %1 = stablehlo.pad %arg0, %cst_0, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %2 = stablehlo.compare LT, %1, %0 : (tensor<4x768x1536xf32>, tensor<4x768x1536xf32>) -> tensor<4x768x1536xi1>
// CHECK-NEXT:     return
// CHECK-NEXT: }

func.func @test_elementwise_complex(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>) -> () {
  %0 = stablehlo.complex %arg0, %arg1 : (tensor<4x760x1533xf32>, tensor<4x760x1533xf32>) -> tensor<4x760x1533xcomplex<f32>>
  return
}

// CHECK: func.func @test_elementwise_complex(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg1, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %1 = stablehlo.pad %arg0, %cst_0, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %2 = stablehlo.complex %1, %0 : tensor<4x768x1536xcomplex<f32>>
// CHECK-NEXT:     return
// CHECK-NEXT: }

func.func @test_elementwise_divide(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>) -> () {
  %0 = stablehlo.divide %arg0, %arg1 : (tensor<4x760x1533xf32>, tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return
}

// CHECK: func.func @test_elementwise_divide(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg1, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %1 = stablehlo.pad %arg0, %cst_0, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %2 = stablehlo.divide %1, %0 : tensor<4x768x1536xf32>
// CHECK-NEXT:     return
// CHECK-NEXT: }

func.func @test_elementwise_maximum(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>) -> () {
  %0 = stablehlo.maximum %arg0, %arg1 : (tensor<4x760x1533xf32>, tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return
}

// CHECK: func.func @test_elementwise_maximum(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg1, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %1 = stablehlo.pad %arg0, %cst_0, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %2 = stablehlo.maximum %1, %0 : tensor<4x768x1536xf32>
// CHECK-NEXT:     return
// CHECK-NEXT: }

func.func @test_elementwise_minimum(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>) -> () {
  %0 = stablehlo.minimum %arg0, %arg1 : (tensor<4x760x1533xf32>, tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return
}

// CHECK: func.func @test_elementwise_minimum(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg1, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %1 = stablehlo.pad %arg0, %cst_0, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %2 = stablehlo.minimum %1, %0 : tensor<4x768x1536xf32>
// CHECK-NEXT:     return
// CHECK-NEXT: }

func.func @test_elementwise_multiply(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>) -> () {
  %0 = stablehlo.multiply %arg0, %arg1 : (tensor<4x760x1533xf32>, tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return
}

// CHECK: func.func @test_elementwise_multiply(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg1, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %1 = stablehlo.pad %arg0, %cst_0, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %2 = stablehlo.multiply %1, %0 : tensor<4x768x1536xf32>
// CHECK-NEXT:     return
// CHECK-NEXT: }

func.func @test_elementwise_or(%arg0: tensor<4x760x1533xi1>, %arg1: tensor<4x760x1533xi1>) -> () {
  %0 = stablehlo.or %arg0, %arg1 : (tensor<4x760x1533xi1>, tensor<4x760x1533xi1>) -> tensor<4x760x1533xi1>
  return
}

// CHECK: func.func @test_elementwise_or(%arg0: tensor<4x760x1533xi1>, %arg1: tensor<4x760x1533xi1>) {
// CHECK-NEXT:     %c = stablehlo.constant dense<false> : tensor<i1>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %c, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xi1>, tensor<i1>) -> tensor<4x768x1536xi1>
// CHECK-NEXT:     %c_0 = stablehlo.constant dense<false> : tensor<i1>
// CHECK-NEXT:     %1 = stablehlo.pad %arg1, %c_0, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xi1>, tensor<i1>) -> tensor<4x768x1536xi1>
// CHECK-NEXT:     %2 = stablehlo.or %0, %1 : tensor<4x768x1536xi1>
// CHECK-NEXT:     return
// CHECK-NEXT: }

func.func @test_elementwise_power(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>) -> () {
  %0 = stablehlo.power %arg0, %arg1 : (tensor<4x760x1533xf32>, tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return
}

// CHECK: func.func @test_elementwise_power(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg1, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %1 = stablehlo.pad %arg0, %cst_0, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %2 = stablehlo.power %1, %0 : tensor<4x768x1536xf32>
// CHECK-NEXT:     return
// CHECK-NEXT: }

func.func @test_elementwise_remainder(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>) -> () {
  %0 = stablehlo.remainder %arg0, %arg1 : (tensor<4x760x1533xf32>, tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return
}

// CHECK: func.func @test_elementwise_remainder(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg1, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %1 = stablehlo.pad %arg0, %cst_0, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %2 = stablehlo.remainder %1, %0 : tensor<4x768x1536xf32>
// CHECK-NEXT:     return
// CHECK-NEXT: }

func.func @test_elementwise_shift_left(%arg0: tensor<4x760x1533xi64>, %arg1: tensor<4x760x1533xi64>) -> () {
  %0 = stablehlo.shift_left %arg0, %arg1 : (tensor<4x760x1533xi64>, tensor<4x760x1533xi64>) -> tensor<4x760x1533xi64>
  return
}

// CHECK: func.func @test_elementwise_shift_left(%arg0: tensor<4x760x1533xi64>, %arg1: tensor<4x760x1533xi64>) {
// CHECK-NEXT:     %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %c, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xi64>, tensor<i64>) -> tensor<4x768x1536xi64>
// CHECK-NEXT:     %c_0 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:     %1 = stablehlo.pad %arg1, %c_0, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xi64>, tensor<i64>) -> tensor<4x768x1536xi64>
// CHECK-NEXT:     %2 = stablehlo.shift_left %0, %1 : tensor<4x768x1536xi64>
// CHECK-NEXT:     return
// CHECK-NEXT: }

func.func @test_elementwise_shift_right_arithmetic(%arg0: tensor<4x760x1533xi64>, %arg1: tensor<4x760x1533xi64>) -> () {
  %0 = stablehlo.shift_right_arithmetic %arg0, %arg1 : (tensor<4x760x1533xi64>, tensor<4x760x1533xi64>) -> tensor<4x760x1533xi64>
  return
}

// CHECK: func.func @test_elementwise_shift_right_arithmetic(%arg0: tensor<4x760x1533xi64>, %arg1: tensor<4x760x1533xi64>) {
// CHECK-NEXT:     %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %c, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xi64>, tensor<i64>) -> tensor<4x768x1536xi64>
// CHECK-NEXT:     %c_0 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:     %1 = stablehlo.pad %arg1, %c_0, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xi64>, tensor<i64>) -> tensor<4x768x1536xi64>
// CHECK-NEXT:     %2 = stablehlo.shift_right_arithmetic %0, %1 : tensor<4x768x1536xi64>
// CHECK-NEXT:     return
// CHECK-NEXT: }

func.func @test_elementwise_shift_right_logical(%arg0: tensor<4x760x1533xi64>, %arg1: tensor<4x760x1533xi64>) -> () {
  %0 = stablehlo.shift_right_logical %arg0, %arg1 : (tensor<4x760x1533xi64>, tensor<4x760x1533xi64>) -> tensor<4x760x1533xi64>
  return
}

// CHECK: func.func @test_elementwise_shift_right_logical(%arg0: tensor<4x760x1533xi64>, %arg1: tensor<4x760x1533xi64>) {
// CHECK-NEXT:     %c = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %c, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xi64>, tensor<i64>) -> tensor<4x768x1536xi64>
// CHECK-NEXT:     %c_0 = stablehlo.constant dense<0> : tensor<i64>
// CHECK-NEXT:     %1 = stablehlo.pad %arg1, %c_0, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xi64>, tensor<i64>) -> tensor<4x768x1536xi64>
// CHECK-NEXT:     %2 = stablehlo.shift_right_logical %0, %1 : tensor<4x768x1536xi64>
// CHECK-NEXT:     return
// CHECK-NEXT: }

func.func @test_elementwise_subtract(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>) -> () {
  %0 = stablehlo.subtract %arg0, %arg1 : (tensor<4x760x1533xf32>, tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return
}

// CHECK: func.func @test_elementwise_subtract(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %1 = stablehlo.pad %arg1, %cst_0, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %2 = stablehlo.subtract %0, %1 : tensor<4x768x1536xf32>
// CHECK-NEXT:     return
// CHECK-NEXT: }

func.func @test_elementwise_xor(%arg0: tensor<4x760x1533xi1>, %arg1: tensor<4x760x1533xi1>) -> () {
  %0 = stablehlo.xor %arg0, %arg1 : (tensor<4x760x1533xi1>, tensor<4x760x1533xi1>) -> tensor<4x760x1533xi1>
  return
}
// CHECK: func.func @test_elementwise_xor(%arg0: tensor<4x760x1533xi1>, %arg1: tensor<4x760x1533xi1>) {
// CHECK-NEXT:     %c = stablehlo.constant dense<false> : tensor<i1>
// CHECK-NEXT:     %0 = stablehlo.pad %arg1, %c, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xi1>, tensor<i1>) -> tensor<4x768x1536xi1>
// CHECK-NEXT:     %c_0 = stablehlo.constant dense<false> : tensor<i1>
// CHECK-NEXT:     %1 = stablehlo.pad %arg0, %c_0, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xi1>, tensor<i1>) -> tensor<4x768x1536xi1>
// CHECK-NEXT:     %2 = stablehlo.xor %1, %0 : tensor<4x768x1536xi1>
// CHECK-NEXT:     return
// CHECK-NEXT: }

func.func @test_elementwise_clamp(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>, %arg2: tensor<4x760x1533xf32>) -> () {
  %0 = stablehlo.clamp %arg0, %arg1, %arg2 : (tensor<4x760x1533xf32>, tensor<4x760x1533xf32>, tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return
}

// CHECK-NEXT: func.func @test_elementwise_clamp(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>, %arg2: tensor<4x760x1533xf32>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %cst_0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %1 = stablehlo.pad %arg1, %cst_0, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %cst_1 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %2 = stablehlo.pad %arg2, %cst_1, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %3 = stablehlo.clamp %0, %1, %2 : tensor<4x768x1536xf32>
// CHECK-NEXT:     return
// CHECK-NEXT: }

func.func @test_dot_general(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x1533x760xf32>) -> () {
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

func.func @test_transpose(%arg0: tensor<4x760x1533xf32>) -> () {
  %0 = stablehlo.transpose %arg0, dims = [0, 1, 2] : (tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return
}

// CHECK: func.func @test_transpose(%arg0: tensor<4x760x1533xf32>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.transpose %0, dims = [0, 1, 2] : (tensor<4x768x1536xf32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     return
// CHECK-NEXT: }

func.func @test_transpose_ret(%arg0: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
  %0 = stablehlo.transpose %arg0, dims = [0, 1, 2] : (tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32>
  return %0 : tensor<4x760x1533xf32>
}

// CHECK: func.func @test_transpose_ret(%arg0: tensor<4x760x1533xf32>) -> tensor<4x760x1533xf32> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.transpose %0, dims = [0, 1, 2] : (tensor<4x768x1536xf32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %2 = stablehlo.slice %1 [0:4, 0:760, 0:1533] : (tensor<4x768x1536xf32>) -> tensor<4x760x1533xf32>
// CHECK-NEXT:     return %2 : tensor<4x760x1533xf32>
// CHECK-NEXT: }

func.func @test_bcast(%arg0: tensor<4x760x1533xf32>) -> () {
  %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2] : (tensor<4x760x1533xf32>) -> tensor<4x760x1533x121xf32>
  return
}

// CHECK: func.func @test_bcast(%arg0: tensor<4x760x1533xf32>) {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-NEXT:     %0 = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT:     %1 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2] : (tensor<4x768x1536xf32>) -> tensor<4x768x1536x128xf32>
// CHECK-NEXT:     %2 = stablehlo.slice %1 [0:4, 0:760, 0:1536, 0:128] : (tensor<4x768x1536x128xf32>) -> tensor<4x760x1536x128xf32>
// CHECK-NEXT:     return
// CHECK-NEXT: }
