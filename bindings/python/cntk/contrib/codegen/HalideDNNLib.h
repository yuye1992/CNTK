#pragma once

#pragma warning(push)
#pragma warning(disable : 4100 4244 4458 4127)
#include "Halide.h"
#pragma warning(pop)

namespace CNTK
{
    const int c_VectorizationWidth = 4;

    template <int inputDimension, int outputDimension>
    inline Halide::Func VectorByMatrixTimes(Halide::Func vec, Halide::Func matrix)
    {
        size_t dimension = inputDimension;
        if (dimension < c_VectorizationWidth)
        {
            // No point in vectorization, the size is too small.
            Halide::Func output("VectorByMatrixTimes");
            Halide::RDom k(0, inputDimension);
            Halide::Var index;
            output(index) = Halide::sum(vec(k) * matrix(k, index));
            return output;
        }

        Halide::Func partial("VectorByMatrixTimesPartial");
        Halide::Var vectorizedInnerIndex;
        Halide::Var outerIndex;
        Halide::RDom k1(0, inputDimension / c_VectorizationWidth);
        partial(vectorizedInnerIndex, outerIndex) = Halide::sum(vec(vectorizedInnerIndex + (k1 * c_VectorizationWidth)) * matrix(vectorizedInnerIndex + (k1 * c_VectorizationWidth), outerIndex));

        Halide::Func vectorized("VectorByMatrixTimesVectorized");
        Halide::Var index;
        Halide::RDom k2(0, c_VectorizationWidth);
        vectorized(index) = Halide::sum(partial(k2, index));

        Halide::Func residual("VectorByMatrixTimesResidual");
        Halide::RDom k3((inputDimension / c_VectorizationWidth) * c_VectorizationWidth, inputDimension % c_VectorizationWidth);
        residual(index) = Halide::sum(vec(k3) * matrix(k3, index));

        Halide::Func output("VectorByMatrixTimes");
        output(index) = vectorized(index) + residual(index);

        partial.bound(vectorizedInnerIndex, 0, c_VectorizationWidth);
        partial.compute_at(output, index).vectorize(vectorizedInnerIndex, c_VectorizationWidth);

        output.bound(index, 0, outputDimension);
        output.compute_root().vectorize(index, c_VectorizationWidth);

        return output;
    }

    template <typename T>
    inline Halide::Func Sigmoid(const Halide::Func& input)
    {
        Halide::Func sigmoidOutput("Sigmoid");
        Halide::Var index;
        sigmoidOutput(index) = (T)1 / ((T)1 + fast_exp(-input(index)));
        return sigmoidOutput;
    }

    inline Halide::Func Tanh(const Halide::Func& input)
    {
        Halide::Func tanhOutput("Tanh");
        Halide::Var index;
        tanhOutput(index) = tanh(input(index));
        return tanhOutput;
    }

    template <int From, int To>
    inline Halide::Func Slice(const Halide::Func& input)
    {
        Halide::Func slice("Slice");
        Halide::Var index;
        slice(index) = input(Halide::min(From + index, To - 1));
        return slice;
    }

    inline Halide::Func ElementTimes(const Halide::Func& operand1, const Halide::Func& operand2)
    {
        Halide::Var index;
        Halide::Func result("ElementTimes");
        result(index) = operand1(index) * operand2(index);
        return result;
    }

    inline Halide::Func Plus(const Halide::Func& operand1, const Halide::Func& operand2)
    {
        Halide::Var index;
        Halide::Func result("Plus");
        result(index) = operand1(index) + operand2(index);
        return result;
    }
}