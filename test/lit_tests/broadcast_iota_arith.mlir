// RUN: enzymexlamlir-opt --enzyme-hlo-generate-td="patterns=broadcast_iota_simplify" --transform-interpreter --enzyme-hlo-remove-transform -allow-unregistered-dialect --split-input-file %s | FileCheck %s

// Regression test for: broadcasting a range against a reshaped vector
// produces incorrect results when range length >= 128.
//
// The bug was in BroadcastIotaSimplify: when an arithmetic constant sequence
// is broadcast, the iota dimension in the output should be the broadcast
// dimension (where values vary), not the first dimension NOT in
// broadcast_dimensions.
//
// In practice the bug triggers at N >= 128 (when N*K >= max_constant_expansion
// = 1024 elements) because BroadcastInDimSimplify constant-expands the
// broadcast for smaller tensors, masking the bug. Here we test the
// BroadcastIotaSimplify pattern directly with small tensors as minimal
// reproducible examples (MWEs).
//
// MWE corresponding to Julia: Float32.((1:N) .<= reshape(nk, 1, K))
// which produces broadcast_in_dim with dims=[0] for the range.

// Arithmetic sequence [0,1,2,3] broadcast to [4,8] with dims=[0].
// Values vary along output dim 0, so iota must be dim 0 (not dim 1).
func.func @arith_broadcast_dim0() -> tensor<4x8xi32> {
    %c = stablehlo.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [0] : (tensor<4xi32>) -> tensor<4x8xi32>
    return %0 : tensor<4x8xi32>
}

// CHECK:  func.func @arith_broadcast_dim0() -> tensor<4x8xi32> {
// CHECK:    stablehlo.iota dim = 0 : tensor<4x8xi32>

// -----

// Arithmetic sequence [0,1,2,3] broadcast to [8,4] with dims=[1].
// Values vary along output dim 1, so iota must be dim 1 (not dim 0).
func.func @arith_broadcast_dim1() -> tensor<8x4xi32> {
    %c = stablehlo.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [1] : (tensor<4xi32>) -> tensor<8x4xi32>
    return %0 : tensor<8x4xi32>
}

// CHECK:  func.func @arith_broadcast_dim1() -> tensor<8x4xi32> {
// CHECK:    stablehlo.iota dim = 1 : tensor<8x4xi32>

// -----

// Arithmetic sequence [0,1,2,3] broadcast to [5,4,6] with dims=[1].
// Values vary along output dim 1, so iota must be dim 1.
func.func @arith_broadcast_dim1_3d() -> tensor<5x4x6xi32> {
    %c = stablehlo.constant dense<[0, 1, 2, 3]> : tensor<4xi32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [1] : (tensor<4xi32>) -> tensor<5x4x6xi32>
    return %0 : tensor<5x4x6xi32>
}

// CHECK:  func.func @arith_broadcast_dim1_3d() -> tensor<5x4x6xi32> {
// CHECK:    stablehlo.iota dim = 1 : tensor<5x4x6xi32>

// -----

// Arithmetic sequence with non-zero start [1,2,3,4] broadcast to [4,8] dims=[0].
// Should produce: start=1, stride=1 with iota dim 0.
func.func @arith_broadcast_nonzero_start() -> tensor<4x8xi32> {
    %c = stablehlo.constant dense<[1, 2, 3, 4]> : tensor<4xi32>
    %0 = stablehlo.broadcast_in_dim %c, dims = [0] : (tensor<4xi32>) -> tensor<4x8xi32>
    return %0 : tensor<4x8xi32>
}

// CHECK:  func.func @arith_broadcast_nonzero_start() -> tensor<4x8xi32> {
// CHECK:    stablehlo.iota dim = 0 : tensor<4x8xi32>
