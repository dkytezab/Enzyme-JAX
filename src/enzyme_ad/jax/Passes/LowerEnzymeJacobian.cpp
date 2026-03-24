//===- LowerEnzymeJacobian.cpp  ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implements patterns to convert JVP/VJPs originating from an enzyme.jacobian
// to enzyme fwddiff/autodiff calls
//===----------------------------------------------------------------------===//

#include "Enzyme/MLIR/Dialect/Dialect.h"
#include "Enzyme/MLIR/Dialect/Ops.h"
#include "src/enzyme_ad/jax/Passes/Passes.h"
#include "src/enzyme_ad/jax/Utils.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "lower-enzyme-jacobian"

namespace mlir {
namespace enzyme {
#define GEN_PASS_DEF_LOWERENZYMEJACOBIANSTABLEHLO
#include "src/enzyme_ad/jax/Passes/Passes.h.inc"
} // namespace enzyme
} // namespace mlir

using namespace mlir;
using namespace mlir::enzyme;

namespace {

struct DotGeneralLowering : public OpRewritePattern<stablehlo::DotGeneralOp> {
  using OpRewritePattern<stablehlo::DotGeneralOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(stablehlo::DotGeneralOp op,
                                PatternRewriter &rewriter) const override {
    auto *context = rewriter.getContext();
    auto lhsResult = dyn_cast<OpResult>(op.getLhs());
    if (!lhsResult)
      return failure();

    auto jacobian = dyn_cast<enzyme::JacobianOp>(lhsResult.getOwner());
    if (!jacobian)
      return failure();

    auto module = op->getParentOfType<ModuleOp>();
    if (!module)
      return failure();

    auto fn = module.lookupSymbol<func::FuncOp>(jacobian.getFnAttr());
    if (!fn)
      return failure();

    const size_t numArgs = fn.getNumArguments();
    const size_t numResults = fn.getNumResults();
    if (numArgs == 0 || numResults == 0)
      return failure();

    // Assume Jacobian results are laid out by input, then by function result:
    // J(out0, in0), J(out1, in0), ..., J(out0, in1), ...
    const unsigned jacobianResultNumber = lhsResult.getResultNumber();
    if (jacobianResultNumber >= numArgs * numResults)
      return failure();

    const size_t argIdx = jacobianResultNumber / numResults;
    const size_t retIdx = jacobianResultNumber % numResults;

    SmallVector<Attribute> activityAttrs;
    SmallVector<Value> fwddiffInputs;
    activityAttrs.reserve(numArgs);
    fwddiffInputs.reserve(numArgs + 1);

    for (auto [idx, input] : llvm::enumerate(jacobian.getInputs())) {
      if (idx == argIdx) {
        activityAttrs.push_back(ActivityAttr::get(context,
                                                  Activity::enzyme_dup));
        fwddiffInputs.push_back(input);
        fwddiffInputs.push_back(op.getRhs());
      } else {
        activityAttrs.push_back(ActivityAttr::get(context,
                                                  Activity::enzyme_const));
        fwddiffInputs.push_back(input);
      }
    }

    SmallVector<Attribute> retActivityAttrs;
    retActivityAttrs.reserve(numResults);
    for (size_t idx = 0; idx < numResults; ++idx) {
      if (idx == retIdx) {
        retActivityAttrs.push_back(
            ActivityAttr::get(context, Activity::enzyme_dupnoneed));
      } else {
        retActivityAttrs.push_back(
            ActivityAttr::get(context, Activity::enzyme_constnoneed));
      }
    }

    auto selectedResultType = fn.getFunctionType().getResult(retIdx);
    if (selectedResultType != op.getType())
      return failure();

    auto activityAttr = ArrayAttr::get(context, activityAttrs);
    auto retActivityAttr = ArrayAttr::get(context, retActivityAttrs);
    auto fwddiff = rewriter.create<enzyme::ForwardDiffOp>(
        op.getLoc(), TypeRange{selectedResultType}, jacobian.getFnAttr(),
        fwddiffInputs, activityAttr, retActivityAttr,
        rewriter.getI64IntegerAttr(1), jacobian.getStrongZeroAttr());
    rewriter.replaceOp(op, fwddiff.getOutputs());
    return success();
  }
};

struct LowerEnzymeJacobianStableHLO
    : public mlir::enzyme::impl::LowerEnzymeJacobianStableHLOBase<
          LowerEnzymeJacobianStableHLO> {

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<DotGeneralLowering>(context);

    GreedyRewriteConfig config;
    config.enableFolding();
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns),
                                     config))) {
      signalPassFailure();
    }

    // // Verify that all illegal ops have been lowered
    // auto walkResult = getOperation()->walk([&](Operation *op) {
    //   if (isa<enzyme::ConcatOp, enzyme::ExtractOp>(op)) {
    //     op->emitError("Failed to lower enzyme batch operation");
    //     return WalkResult::interrupt();
    //   }
    //   return WalkResult::advance();
    // });
    //
    // if (walkResult.wasInterrupted()) {
    //   signalPassFailure();
    // }
  };
};
} // namespace
