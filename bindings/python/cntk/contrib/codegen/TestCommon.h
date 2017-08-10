#pragma once

#include "Common.h"

template <int inputDimension, int outputDimension>
inline Func VectorByMatrixTimes(Func vec, Func matrix)
{
    Func partial("VectorByMatrixTimesPartial");
    Var vectorizedInnerIndex;
    Var outerIndex;
    RDom k1(0, inputDimension / vectorizationWidth);
    partial(vectorizedInnerIndex, outerIndex) = sum(vec(vectorizedInnerIndex + (k1 * vectorizationWidth)) * matrix(vectorizedInnerIndex + (k1 * vectorizationWidth), outerIndex));

    Func vectorized("VectorByMatrixTimesVectorized");
    Var index;
    RDom k2(0, vectorizationWidth);
    vectorized(index) = sum(partial(k2, index));

    Func residual("VectorByMatrixTimesResidual");
    RDom k3((inputDimension / vectorizationWidth) * vectorizationWidth, inputDimension % vectorizationWidth);
    residual(index) = sum(vec(k3) * matrix(k3, index));

    Func output("VectorByMatrixTimes");
    output(index) = vectorized(index) + residual(index);

    partial.bound(vectorizedInnerIndex, 0, vectorizationWidth);
    partial.compute_at(output, index).vectorize(vectorizedInnerIndex, vectorizationWidth);

    output.bound(index, 0, outputDimension);
    output.compute_root().vectorize(index, vectorizationWidth);

    return output;
}

template <typename T>
inline Func Sigmoid(const Func& input)
{
    Func sigmoidOutput("Sigmoid");
    Var index;
    sigmoidOutput(index) = (T)1 / ((T)1 + fast_exp(-input(index)));
    return sigmoidOutput;
}

inline Func Tanh(const Func& input)
{
    Func tanhOutput("Tanh");
    Var index;
    tanhOutput(index) = tanh(input(index));
    return tanhOutput;
}

template <int From, int To>
inline Func Slice(const Func& input)
{
    Func slice("Slice");
    Var index;
    slice(index) = input(Halide::min(From + index, To - 1));
    return slice;
}

inline Func ElementTimes(const Func& operand1, const Func& operand2)
{
    Var index;
    Func result("ElementTimes");
    result(index) = operand1(index) * operand2(index);
    return result;
}

inline Func Plus(const Func& operand1, const Func& operand2)
{
    Var index;
    Func result("Plus");
    result(index) = operand1(index) + operand2(index);
    return result;
}