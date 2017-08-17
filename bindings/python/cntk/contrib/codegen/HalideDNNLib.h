#pragma once

#pragma warning(push)
#pragma warning(disable : 4100 4244 4458 4127)
#include "Halide.h"
#pragma warning(pop)

namespace CNTK
{
    const int c_VectorizationWidth = 4;

    template <int InputDimension, int OutputDimension>
    inline Halide::Func VectorByMatrixTimes(Halide::Func vec, Halide::Func matrix)
    {
        size_t dimension = InputDimension;
        if (dimension < c_VectorizationWidth)
        {
            // No point in vectorization, the size is too small.
            Halide::Func output("VectorByMatrixTimes");
            Halide::RDom k(0, InputDimension);
            Halide::Var index;
            output(index) = Halide::sum(vec(k) * matrix(k, index));
            return output;
        }

        Halide::Func partial("VectorByMatrixTimesPartial");
        Halide::Var vectorizedInnerIndex;
        Halide::Var outerIndex;
        Halide::RDom k1(0, InputDimension / c_VectorizationWidth);
        partial(vectorizedInnerIndex, outerIndex) = Halide::sum(vec(vectorizedInnerIndex + (k1 * c_VectorizationWidth)) * matrix(vectorizedInnerIndex + (k1 * c_VectorizationWidth), outerIndex));

        Halide::Func vectorized("VectorByMatrixTimesVectorized");
        Halide::Var index;
        Halide::RDom k2(0, c_VectorizationWidth);
        vectorized(index) = Halide::sum(partial(k2, index));

        Halide::Func residual("VectorByMatrixTimesResidual");
        Halide::RDom k3((InputDimension / c_VectorizationWidth) * c_VectorizationWidth, InputDimension % c_VectorizationWidth);
        residual(index) = Halide::sum(vec(k3) * matrix(k3, index));

        Halide::Func output("VectorByMatrixTimes");
        output(index) = vectorized(index) + residual(index);

        partial.bound(vectorizedInnerIndex, 0, c_VectorizationWidth);
        partial.compute_at(output, index).vectorize(vectorizedInnerIndex, c_VectorizationWidth);

        output.bound(index, 0, OutputDimension);
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

    // TODO: Probably we should pass a codebook of the following interface
    // Func(bin_value) => gives the unquantized value back.
    // This codebook can be also generated during quantization based on any distribution: symmetric/assymetric/log.
    // Otherwise we have to change the signature for each and every quantization procedure.
    template <typename T, typename QuantizedType, int inputDimension, int outputDimension, int NumReservedBits>
    inline Halide::Func VectorByMatrixTimesQuantized(
        const std::vector<Halide::Func>& vec, const std::vector<Halide::Func>& matrix)
    {
        auto quantized = VectorByMatrixTimes<inputDimension, outputDimension>(vec[1], matrix[1]);

        Halide::Func result("result");
        Halide::Var index;

        Halide::Func inv1;
        inv1() = (T)1.0 / vec[0]();
        Halide::Func inv2;
        inv2() = (T)1.0 / matrix[0]();
        inv1.compute_root();
        inv2.compute_root();

        result(index) = inv1() * inv2() * quantized(index);
        result.bound(index, 0, outputDimension);
        return result;
    }

    template<class Type, class QuantizedType, int InputDimension, int NumReservedBits>
    inline std::vector<Halide::Func> Quantize(Halide::Func vector)
    {
        Halide::Func minMaxVectorized("minMaxVectorized");
        Halide::Var vectorizationIndex("vectorizationIndex");
        Halide::RDom k1(0, InputDimension /c_VectorizationWidth, "vectorizedDom");
        minMaxVectorized(vectorizationIndex) = { std::numeric_limits<Type>::max(), std::numeric_limits<Type>::min() };

        Halide::Expr inputValue = vector(vectorizationIndex + (k1 * c_VectorizationWidth));
        minMaxVectorized(vectorizationIndex) = { Halide::min(minMaxVectorized(vectorizationIndex)[0], inputValue), Halide::max(minMaxVectorized(vectorizationIndex)[1], inputValue) };

        Halide::RDom k2((InputDimension / c_VectorizationWidth) * c_VectorizationWidth, InputDimension % c_VectorizationWidth);
        Halide::Func minMaxTail("minMaxTail");
        minMaxTail() = { Halide::minimum(vector(k2)), Halide::maximum(vector(k2)) };

        Halide::Func minMax("minMax");
        Halide::RDom k3(0, c_VectorizationWidth);
        minMax() = { Halide::min(Halide::minimum(minMaxVectorized(k3)[0]), minMaxTail()[0]), Halide::max(Halide::maximum(minMaxVectorized(k3)[1]), minMaxTail()[1]) };

        Halide::Func absMax("absMax");
        absMax() = Halide::max(-minMax()[0], minMax()[1]) * (1 << NumReservedBits);

        // Quantize, same procedure as in MLP library
        const int numQuantizedTypeBits = sizeof(QuantizedType) * 8;

        // We still need one bit for representing the sign, that's why - 1.
        auto quantizedTypeMaxValue = std::numeric_limits<QuantizedType>::max();

        Halide::Func qStep("qstep");
        qStep() = absMax() / (quantizedTypeMaxValue + 0.5f); // 0.5 is for rounding.

        Halide::Func quantized("quantized");
        Halide::Var index("quantizedIndex");
        // + 1 for the edge case of quantizing -quantizedTypeMaxValue and 0.5 for rounding.
        quantized(index) = Halide::cast(Halide::type_of<QuantizedType>(), Halide::cast(Halide::Int(32), (vector(index) * qStep() + quantizedTypeMaxValue + 1.5f) - (1 + quantizedTypeMaxValue)));

        // Schedule
        minMaxVectorized.bound(vectorizationIndex, 0, c_VectorizationWidth);
        minMaxVectorized.vectorize(vectorizationIndex, c_VectorizationWidth);
        minMaxVectorized.compute_root().update().vectorize(vectorizationIndex, c_VectorizationWidth);
        qStep.compute_root();

        return std::vector<Halide::Func>{ qStep, quantized };
    }

    // Actually for speech models there is no need in using this function, because vectors are the only
    // quantized entities at runtime.
    template<class Type, class QuantizedType, int InputDimension, int OuputDimension, int NumReservedBits>
    inline std::vector<Halide::Func> Quantize(Halide::Func matrix)
    {
        int width = matrix.width();
        int height = matrix.height();
        int totalSize = width * height;

        // Flatten
        Var index("index");
        Func asVector("asVector");
        asVector(index) = matrix(index / OuputDimension, index % OuputDimension);
        asVector.bound(index, 0, InputDimension * OutputDimension);

        Func qStep("qstep"), quantized("quantized"), quantizedMatrix("quantizedMatrix");
        auto result = Quantize<Type, QuantizedType, InputDimension * OutputDimension, NumReservedBits>(
            asVector);

        // Unflatten
        Var x, y;
        quantizedMatrix(x, y) = quantized(x * OutputDimension + y);
        quantizedMatrix.bound(x, 0, InputDimension);
        quantizedMatrix.bound(y, 0, OuputDimension);
        return { qStep, quantizedMatrix };
    }
}