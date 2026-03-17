// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @main(%2850: tensor<4x1518x3056xf64>, %2852: tensor<4x1518x3056xf64>, %2858: tensor<4x1518x3056xf64>) -> tensor<4x1518x3056xf64> {
    %cst_151 = stablehlo.constant dense<0.58333333333333326> : tensor<4x1518x3056xf64>
    %2851 = stablehlo.multiply %2850, %cst_151 : tensor<4x1518x3056xf64>
    %2853 = stablehlo.multiply %2852, %cst_151 : tensor<4x1518x3056xf64>
    %2859 = stablehlo.add %2853, %2858 : tensor<4x1518x3056xf64>
    %2860 = stablehlo.add %2851, %2859 : tensor<4x1518x3056xf64>
    return %2860 : tensor<4x1518x3056xf64>
}

// CHECK: func.func @main(%arg0: tensor<4x1518x3056xf64>, %arg1: tensor<4x1518x3056xf64>, %arg2: tensor<4x1518x3056xf64>) -> tensor<4x1518x3056xf64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.58333333333333326> : tensor<4x1518x3056xf64>
// CHECK-NEXT:     %0 = stablehlo.add %arg0, %arg1 : tensor<4x1518x3056xf64>
// CHECK-NEXT:     %1 = stablehlo.multiply %0, %cst : tensor<4x1518x3056xf64>
// CHECK-NEXT:     %2 = stablehlo.add %1, %arg2 : tensor<4x1518x3056xf64>
// CHECK-NEXT:     return %2 : tensor<4x1518x3056xf64>
// CHECK-NEXT: }
