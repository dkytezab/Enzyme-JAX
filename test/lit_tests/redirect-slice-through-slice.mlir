// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=redirect_slice_through_slice" --transform-interpreter --enzyme-hlo-remove-transform %s | FileCheck %s

func.func @main(%1227: tensor<20x1536x3056xf32>, %1231: tensor<1x1520x3056xf32>, %1232: tensor<1x1520x3056xf32>) -> (tensor<4x1520x3056xf32>, tensor<20x1536x3056xf32>, tensor<3x1520x3056xf32>, tensor<3x1520x3056xf32>) {
    %c_299 = stablehlo.constant dense<12> : tensor<i32>
    %c_300 = stablehlo.constant dense<7> : tensor<i32>
    %c_302 = stablehlo.constant dense<8> : tensor<i32>
    %c_365 = stablehlo.constant dense<0> : tensor<i32>
    %1237 = stablehlo.dynamic_update_slice %1227, %1231, %c_300, %c_302, %c_365 : (tensor<20x1536x3056xf32>, tensor<1x1520x3056xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<20x1536x3056xf32>
    %1238 = stablehlo.slice %1237 [7:11, 8:1528, 0:3056] : (tensor<20x1536x3056xf32>) -> tensor<4x1520x3056xf32>
    %1245 = stablehlo.dynamic_update_slice %1237, %1232, %c_299, %c_302, %c_365 : (tensor<20x1536x3056xf32>, tensor<1x1520x3056xf32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<20x1536x3056xf32>
    %3556 = stablehlo.slice %1237 [7:10, 8:1528, 0:3056] : (tensor<20x1536x3056xf32>) -> tensor<3x1520x3056xf32>
    %3557 = stablehlo.slice %1245 [10:13, 8:1528, 0:3056] : (tensor<20x1536x3056xf32>) -> tensor<3x1520x3056xf32>
    return %1238, %1245, %3556, %3557 : tensor<4x1520x3056xf32>, tensor<20x1536x3056xf32>, tensor<3x1520x3056xf32>, tensor<3x1520x3056xf32>
}
