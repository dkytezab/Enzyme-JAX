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

    // either one of lhs or rhs has to be a enzyme.jacobian but not both
    auto lhsOp = op.getLhs().getDefiningOp<enzyme::JacobianOp>();
    auto rhsOp = op.getRhs().getDefiningOp<enzyme::JacobianOp>();

    if (!lhsOp && !rhsOp)
      return failure();

    if (lhsOp && rhsOp)
      return failure();

    // construct autodiff args for fn
    enzyme::JacobianOp jacOp = lhsOp ? lhsOp : rhsOp;
    SymbolTableCollection symbolTable;
    symbolTable.getSymbolTable(jacOp);
    auto fn = cast<FunctionOpInterface>(
        symbolTable.lookupNearestSymbolFrom(jacOp, jacOp.getFnAttr()));
    auto nargs = fn.getNumArguments();
    auto nouts = fn.getNumResults();
    // the jacobian operand
    // TODO: handle indexing for mutable arguments
    // jidx = d(out_idx) / d(in_idx), where jidx = out_idx * nargs + in_idx;
    auto J = cast<OpResult>(lhsOp ? op.getLhs() : op.getRhs());
    auto jidx = J.getResultNumber();

    if (jidx >= nargs * nouts)
      return failure();

    auto diffo_idx = jidx / nargs;
    auto diffin_idx = jidx % nargs;

    if (lhsOp) {
      // JVP -> enzyme.fwddiff transform
      // The resulting fwddiff op will only have in_idx -> enzyme_dup, out_idx
      // -> enzyme_dupnoneed

      SmallVector<Value> in_args;
      SmallVector<ActivityAttr, 2> newInActivityArgs;
      SmallVector<ActivityAttr, 2> newRetActivityArgs;
      for (auto [idx, act] :
           llvm::enumerate(jacOp.getActivity().getAsRange<ActivityAttr>())) {
        Value in = jacOp.getInputs()[idx];

        if (idx != diffin_idx) {
          in_args.push_back(in);
          newInActivityArgs.push_back(
              ActivityAttr::get(rewriter.getContext(), Activity::enzyme_const));
        } else {
          in_args.push_back(in);
          in_args.push_back(op.getRhs());
          newInActivityArgs.push_back(
              ActivityAttr::get(rewriter.getContext(), Activity::enzyme_dup));
        }
      }

      // construct ret_args
      for (auto [idx, ret_act] :
           llvm::enumerate(jacOp.getRetActivity().getAsRange<ActivityAttr>())) {
        if (idx == diffo_idx) {
          newRetActivityArgs.push_back(ActivityAttr::get(
              rewriter.getContext(), Activity::enzyme_dupnoneed));
        } else {
          newRetActivityArgs.push_back(ActivityAttr::get(
              rewriter.getContext(), Activity::enzyme_constnoneed));
        }
      }

      ArrayAttr newInActivity =
          ArrayAttr::get(rewriter.getContext(),
                         llvm::ArrayRef<Attribute>(newInActivityArgs.begin(),
                                                   newInActivityArgs.end()));
      ArrayAttr newRetActivity =
          ArrayAttr::get(rewriter.getContext(),
                         llvm::ArrayRef<Attribute>(newRetActivityArgs.begin(),
                                                   newRetActivityArgs.end()));

      rewriter.replaceOpWithNewOp<ForwardDiffOp>(
          op, op->getResultTypes(), jacOp.getFnAttr(), in_args, newInActivity,
          newRetActivity, nullptr, jacOp.getStrongZeroAttr());
    } else {
      // VJP -> enzyme.autodiff transform
    }

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
