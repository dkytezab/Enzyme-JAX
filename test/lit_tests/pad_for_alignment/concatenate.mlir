// RUN: enzymexlamlir-opt --pad-for-alignment --canonicalize --allow-unregistered-dialect -split-input-file %s | FileCheck --dump-input=always %s

func.func @test_concatenate_dim_0(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>) -> tensor<8x760x1533xf32> {
  %0 = stablehlo.concatenate %arg0, %arg1, dim = 0 : (tensor<4x760x1533xf32>, tensor<4x760x1533xf32>) -> tensor<8x760x1533xf32>
  return %0 : tensor<8x760x1533xf32>
}

// CHECK-LABEL: func.func @test_concatenate_dim_0(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>) -> tensor<8x760x1533xf32> {
// CHECK-NEXT: %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG:  %[[pad0:.*]] = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-DAG:  %[[pad1:.*]] = stablehlo.pad %arg1, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-NEXT: %2 = stablehlo.concatenate %[[pad0]], %[[pad1]], dim = 0 : (tensor<4x768x1536xf32>, tensor<4x768x1536xf32>) -> tensor<8x768x1536xf32>
// CHECK-NEXT: %3 = stablehlo.slice %2 [0:8, 0:760, 0:1533] : (tensor<8x768x1536xf32>) -> tensor<8x760x1533xf32>
// CHECK-NEXT: return %3 : tensor<8x760x1533xf32>
// CHECK-NEXT: }

func.func @test_concatenate_dim_1(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>) -> tensor<4x1520x1533xf32> {
  %0 = stablehlo.concatenate %arg0, %arg1, dim = 1 : (tensor<4x760x1533xf32>, tensor<4x760x1533xf32>) -> tensor<4x1520x1533xf32>
  return %0 : tensor<4x1520x1533xf32>
}

// CHECK-LABEL: func.func @test_concatenate_dim_1(%arg0: tensor<4x760x1533xf32>, %arg1: tensor<4x760x1533xf32>) -> tensor<4x1520x1533xf32> {
// CHECK-NEXT: %cst = stablehlo.constant dense<0.000000e+00> : tensor<f32>
// CHECK-DAG: %[[pad0:.*]] = stablehlo.pad %arg0, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-DAG: %[[pad1:.*]] = stablehlo.pad %arg1, %cst, low = [0, 0, 0], high = [0, 8, 3], interior = [0, 0, 0] : (tensor<4x760x1533xf32>, tensor<f32>) -> tensor<4x768x1536xf32>
// CHECK-DAG: %[[slice0:.*]] = stablehlo.slice %[[pad0]] [0:4, 0:760, 0:1536] : (tensor<4x768x1536xf32>) -> tensor<4x760x1536xf32>
// CHECK-DAG: %[[slice1:.*]] = stablehlo.slice %[[pad1]] [0:4, 0:760, 0:1536] : (tensor<4x768x1536xf32>) -> tensor<4x760x1536xf32>
// CHECK-NEXT: %4 = stablehlo.concatenate %[[slice0]], %[[slice1]], dim = 1 : (tensor<4x760x1536xf32>, tensor<4x760x1536xf32>) -> tensor<4x1520x1536xf32>
// CHECK-NEXT: %5 = stablehlo.slice %4 [0:4, 0:1520, 0:1533] : (tensor<4x1520x1536xf32>) -> tensor<4x1520x1533xf32>
// CHECK-NEXT: return %5 : tensor<4x1520x1533xf32>
// CHECK-NEXT: }
