// RUN: enzymexlamlir-opt --enzyme-hlo-opt %s | FileCheck %s

func.func @opt_seq_short(%2850: tensor<4x1518x3056xf64>, %2852: tensor<4x1518x3056xf64>, %2858: tensor<4x1518x3056xf64>) -> tensor<4x1518x3056xf64> {
    %cst_151 = stablehlo.constant dense<0.58333333333333326> : tensor<4x1518x3056xf64>
    %2851 = stablehlo.multiply %2850, %cst_151 : tensor<4x1518x3056xf64>
    %2853 = stablehlo.multiply %2852, %cst_151 : tensor<4x1518x3056xf64>
    %2859 = stablehlo.add %2853, %2858 : tensor<4x1518x3056xf64>
    %2860 = stablehlo.add %2851, %2859 : tensor<4x1518x3056xf64>
    return %2860 : tensor<4x1518x3056xf64>
}

// CHECK: func.func @opt_seq_short(%arg0: tensor<4x1518x3056xf64>, %arg1: tensor<4x1518x3056xf64>, %arg2: tensor<4x1518x3056xf64>) -> tensor<4x1518x3056xf64> {
// CHECK-NEXT:     %cst = stablehlo.constant dense<0.58333333333333326> : tensor<4x1518x3056xf64>
// CHECK-NEXT:     %0 = stablehlo.add %arg0, %arg1 : tensor<4x1518x3056xf64>
// CHECK-NEXT:     %1 = stablehlo.multiply %cst, %0 : tensor<4x1518x3056xf64>
// CHECK-NEXT:     %2 = stablehlo.add %arg2, %1 : tensor<4x1518x3056xf64>
// CHECK-NEXT:     return %2 : tensor<4x1518x3056xf64>
// CHECK-NEXT: }

func.func @opt_seq_long(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>, %arg2: tensor<4xf64>, %arg3: tensor<4xf64>, %arg4: tensor<4xf64>, %arg5: tensor<4xf64>) -> tensor<4xf64> {
    %0 = stablehlo.multiply %arg0, %arg5 : tensor<4xf64>
    %1 = stablehlo.add %0, %arg2 : tensor<4xf64>
    %2 = stablehlo.add %arg3, %1 : tensor<4xf64>
    %3 = stablehlo.multiply %arg5, %arg1 : tensor<4xf64>
    %4 = stablehlo.add %3, %2 : tensor<4xf64>
    %5 = stablehlo.add %4, %arg4 : tensor<4xf64>
    return %5 : tensor<4xf64>
}

// CHECK: func.func @opt_seq_long(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>, %arg2: tensor<4xf64>, %arg3: tensor<4xf64>, %arg4: tensor<4xf64>, %arg5: tensor<4xf64>) -> tensor<4xf64> {
// CHECK-NEXT:     %0 = stablehlo.add %arg1, %arg0 : tensor<4xf64>
// CHECK-NEXT:     %1 = stablehlo.multiply %arg5, %0 : tensor<4xf64>
// CHECK-NEXT:     %2 = stablehlo.add %arg2, %1 : tensor<4xf64>
// CHECK-NEXT:     %3 = stablehlo.add %arg3, %2 : tensor<4xf64>
// CHECK-NEXT:     %4 = stablehlo.add %3, %arg4 : tensor<4xf64>
// CHECK-NEXT:     return %4 : tensor<4xf64>
// CHECK-NEXT: }

func.func @opt_tree(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>, %arg2: tensor<4xf64>, %arg3: tensor<4xf64>, %arg4: tensor<4xf64>, %arg5: tensor<4xf64>) -> tensor<4xf64> {
    %0 = stablehlo.multiply %arg0, %arg5 : tensor<4xf64>
    %1 = stablehlo.add %0, %arg2 : tensor<4xf64>
    %2 = stablehlo.multiply %arg5, %arg1 : tensor<4xf64>
    %3 = stablehlo.add %2, %arg3 : tensor<4xf64>
    %4 = stablehlo.add %3, %1 : tensor<4xf64>
    %5 = stablehlo.add %4, %arg4 : tensor<4xf64>
    return %5 : tensor<4xf64>
}

// CHECK: func.func @opt_tree(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>, %arg2: tensor<4xf64>, %arg3: tensor<4xf64>, %arg4: tensor<4xf64>, %arg5: tensor<4xf64>) -> tensor<4xf64> {
// CHECK-NEXT:     %0 = stablehlo.add %arg1, %arg0 : tensor<4xf64>
// CHECK-NEXT:     %1 = stablehlo.multiply %arg5, %0 : tensor<4xf64>
// CHECK-NEXT:     %2 = stablehlo.add %arg2, %1 : tensor<4xf64>
// CHECK-NEXT:     %3 = stablehlo.add %arg3, %2 : tensor<4xf64>
// CHECK-NEXT:     %4 = stablehlo.add %3, %arg4 : tensor<4xf64>
// CHECK-NEXT:     return %4 : tensor<4xf64>
// CHECK-NEXT: }

func.func @opt_multi(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>, %arg2: tensor<4xf64>, %arg3: tensor<4xf64>, %arg4: tensor<4xf64>, %arg5: tensor<4xf64>) -> tensor<4xf64> {
    %0 = stablehlo.multiply %arg0, %arg5 : tensor<4xf64>
    %1 = stablehlo.add %0, %arg2 : tensor<4xf64>
    %2 = stablehlo.multiply %arg5, %arg1 : tensor<4xf64>
    %3 = stablehlo.add %2, %arg3 : tensor<4xf64>
    %4 = stablehlo.add %3, %1 : tensor<4xf64>
    %5 = stablehlo.multiply %arg4, %arg5 : tensor<4xf64>
    %6 = stablehlo.add %4, %5 : tensor<4xf64>
    return %6: tensor<4xf64>
}

// CHECK: func.func @opt_multi(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>, %arg2: tensor<4xf64>, %arg3: tensor<4xf64>, %arg4: tensor<4xf64>, %arg5: tensor<4xf64>) -> tensor<4xf64> {
// CHECK-NEXT:     %0 = stablehlo.add %arg1, %arg0 : tensor<4xf64>
// CHECK-NEXT:     %1 = stablehlo.add %0, %arg4 : tensor<4xf64>
// CHECK-NEXT:     %2 = stablehlo.multiply %arg5, %1 : tensor<4xf64>
// CHECK-NEXT:     %3 = stablehlo.add %arg3, %arg2 : tensor<4xf64>
// CHECK-NEXT:     %4 = stablehlo.add %3, %2 : tensor<4xf64>
// CHECK-NEXT:     return %4 : tensor<4xf64>
// CHECK-NEXT: }

func.func @no_opt_benefit(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>, %arg2: tensor<4xf64>) -> tensor<4xf64> {
    %0 = stablehlo.multiply %arg0, %arg2 : tensor<4xf64>
    %1 = stablehlo.multiply %arg2, %arg1 : tensor<4xf64>
    %2 = stablehlo.add %0, %1 : tensor<4xf64>
    return %2 : tensor<4xf64>
}

// CHECK: func.func @no_opt_benefit(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>, %arg2: tensor<4xf64>) -> tensor<4xf64> {
// CHECK-NEXT:     %0 = stablehlo.multiply %arg0, %arg2 : tensor<4xf64>
// CHECK-NEXT:     %1 = stablehlo.multiply %arg2, %arg1 : tensor<4xf64>
// CHECK-NEXT:     %2 = stablehlo.add %0, %1 : tensor<4xf64>
// CHECK-NEXT:     return %2 : tensor<4xf64>
// CHECK-NEXT: }

func.func @no_opt_multiuse_add_inbetween(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>, %arg2: tensor<4xf64>, %arg3: tensor<4xf64>, %arg4: tensor<4xf64>, %arg5: tensor<4xf64>) -> tensor<4xf64> {
    %0 = stablehlo.multiply %arg0, %arg5 : tensor<4xf64>
    %1 = stablehlo.add %0, %arg2 : tensor<4xf64>
    %2 = stablehlo.add %arg3, %1 : tensor<4xf64>
    %3 = stablehlo.multiply %arg5, %arg1 : tensor<4xf64>
    %4 = stablehlo.add %3, %2 : tensor<4xf64>
    %5 = stablehlo.add %4, %arg4 : tensor<4xf64>
    %6 = stablehlo.divide %5, %2 : tensor<4xf64>
    return %6 : tensor<4xf64>
}

// CHECK: func.func @no_opt_multiuse_add_inbetween(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>, %arg2: tensor<4xf64>, %arg3: tensor<4xf64>, %arg4: tensor<4xf64>, %arg5: tensor<4xf64>) -> tensor<4xf64> {
// CHECK-NEXT:     %0 = stablehlo.multiply %arg0, %arg5 : tensor<4xf64>
// CHECK-NEXT:     %1 = stablehlo.add %0, %arg2 : tensor<4xf64>
// CHECK-NEXT:     %2 = stablehlo.add %arg3, %1 : tensor<4xf64>
// CHECK-NEXT:     %3 = stablehlo.multiply %arg5, %arg1 : tensor<4xf64>
// CHECK-NEXT:     %4 = stablehlo.add %3, %2 : tensor<4xf64>
// CHECK-NEXT:     %5 = stablehlo.add %4, %arg4 : tensor<4xf64>
// CHECK-NEXT:     %6 = stablehlo.divide %5, %2 : tensor<4xf64>
// CHECK-NEXT:     return %6 : tensor<4xf64>
// CHECK-NEXT: }

func.func @no_opt_multiuse_mul(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>, %arg2: tensor<4xf64>, %arg3: tensor<4xf64>, %arg4: tensor<4xf64>, %arg5: tensor<4xf64>) -> tensor<4xf64> {
    %0 = stablehlo.multiply %arg0, %arg5 : tensor<4xf64>
    %1 = stablehlo.add %0, %arg2 : tensor<4xf64>
    %2 = stablehlo.add %arg3, %1 : tensor<4xf64>
    %3 = stablehlo.multiply %arg5, %arg1 : tensor<4xf64>
    %4 = stablehlo.add %3, %2 : tensor<4xf64>
    %5 = stablehlo.add %4, %arg4 : tensor<4xf64>
    %6 = stablehlo.divide %5, %0 : tensor<4xf64>
    return %6 : tensor<4xf64>
}

// CHECK: func.func @no_opt_multiuse_mul(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>, %arg2: tensor<4xf64>, %arg3: tensor<4xf64>, %arg4: tensor<4xf64>, %arg5: tensor<4xf64>) -> tensor<4xf64> {
// CHECK-NEXT:     %0 = stablehlo.multiply %arg0, %arg5 : tensor<4xf64>
// CHECK-NEXT:     %1 = stablehlo.add %0, %arg2 : tensor<4xf64>
// CHECK-NEXT:     %2 = stablehlo.add %arg3, %1 : tensor<4xf64>
// CHECK-NEXT:     %3 = stablehlo.multiply %arg5, %arg1 : tensor<4xf64>
// CHECK-NEXT:     %4 = stablehlo.add %3, %2 : tensor<4xf64>
// CHECK-NEXT:     %5 = stablehlo.add %4, %arg4 : tensor<4xf64>
// CHECK-NEXT:     %6 = stablehlo.divide %5, %0 : tensor<4xf64>
// CHECK-NEXT:     return %6 : tensor<4xf64>
// CHECK-NEXT: }

func.func @no_opt_single_mul(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>) -> tensor<4xf64> {
    %0 = stablehlo.multiply %arg0, %arg1 : tensor<4xf64>
    %1 = stablehlo.add %0, %0 : tensor<4xf64>
    return %1 : tensor<4xf64>
}

// CHECK: func.func @no_opt_single_mul(%arg0: tensor<4xf64>, %arg1: tensor<4xf64>) -> tensor<4xf64> {
// CHECK-NEXT:     %0 = stablehlo.multiply %arg0, %arg1 : tensor<4xf64>
// CHECK-NEXT:     %1 = stablehlo.add %0, %0 : tensor<4xf64>
// CHECK-NEXT:     return %1 : tensor<4xf64>
// CHECK-NEXT: }
