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
