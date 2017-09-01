//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"
#include "Common.h"
#include <random>
#include <string>

// Due to inclusion of windows.h
#undef min
#undef max

#include "XorEvaluator.h"
#include "QuantizedXorEvaluator.h"
#include "LstmSpeechEvaluator.h"
#include "QuantizedLstmSpeechEvaluator.h"

#include "CNTKLibrary.h"

namespace utf = boost::unit_test;

namespace pt = boost::property_tree;

namespace CNTK { namespace Test {

using Halide::ImageParam;

template<class T>
using Buffer = Halide::Buffer<T>;

BOOST_AUTO_TEST_SUITE(CodeGenDeserializationSuite)

BOOST_AUTO_TEST_CASE(XorOperation)
{
    // Halide
    XorEvaluator e;
    e.init("XorEvaluator.json");

    // Quantized
    QuantizedXorEvaluator qe;
    qe.init("XorEvaluator.json");

    Halide::ImageParam features(Halide::type_of<float>(), 1);
    std::vector<float> v = { 0.f, 1.f };
    features.set(Halide::Buffer<float>(v.data(), 2));

    Halide::Buffer<float> result(1);

    e.Evaluate(features, result);
    auto result10 = result(0);

    qe.Evaluate(features, result);
    auto result30 = result(0);

    v[0] = 1.f;
    e.Evaluate(features, result);
    auto result11 = result(0);

    qe.Evaluate(features, result);
    auto result31 = result(0);

    // CNTK
    auto model = Function::Load(L"xor.model", DeviceDescriptor::CPUDevice());
    auto input = model->Arguments();
    auto output = model->Output();

    ValuePtr value = MakeSharedObject<Value>(
        MakeSharedObject<NDArrayView>(DataType::Float, NDShape({ 2 }), v.data(), sizeof(float) * 2, DeviceDescriptor::CPUDevice()));

    std::unordered_map<Variable, ValuePtr> i = { { input.front(), value } };
    std::unordered_map<Variable, ValuePtr> o = { { output, nullptr } };

    v[0] = 0.f;
    model->Forward(i, o, DeviceDescriptor::CPUDevice());
    auto result20 = *(o[output]->Data()->DataBuffer<float>());

    v[0] = 1.f;
    value = MakeSharedObject<Value>(
        MakeSharedObject<NDArrayView>(DataType::Float, NDShape({ 2 }), v.data(), sizeof(float) * 2, DeviceDescriptor::CPUDevice()));
    i = std::unordered_map<Variable, ValuePtr>({ { input.front(), value } });
    o = std::unordered_map<Variable, ValuePtr>({ { output, nullptr } });
    model->Forward(i, o, DeviceDescriptor::CPUDevice());
    auto result21 = *(o[output]->Data()->DataBuffer<float>());

    BOOST_REQUIRE_CLOSE(result10, result20, 0.1);
    BOOST_REQUIRE_CLOSE(result11, result21, 0.1);

    BOOST_REQUIRE_CLOSE_FRACTION(result30, result20, 0.001);
    BOOST_REQUIRE_CLOSE_FRACTION(result31, result21, 0.01);
}

BOOST_AUTO_TEST_CASE(HaliveVsMlp)
{
    // Not a real test yet, a manual check that
    // quantized output looks the same as in MLP.
    const int InputDimension = 80;
    const int OutputDimension = 9404;

    QuantizedLstmSpeechEvaluator qe;
    qe.init("LstmSpeechEvaluator.json");

    // Input
    const int NumberOfFrames = 80;
    std::vector<float> frame = 
    {
        0.336803824f,
        0.390805095f,
        0.511451483f,
        0.427405655f,
        0.0278947111f,
        0.000000000f,
        0.000000000f,
        0.0269803014f,
        0.167097509f,
        0.315470099f,
        0.379547745f,
        0.359533519f,
        0.389804989f,
        0.487798691f,
        0.529781878f,
        0.554115415f,
        0.505439460f,
        0.400578171f,
        0.285043806f,
        0.309477210f,
        0.0319889635f,
        0.000000000f,
        0.000000000f,
        0.000000000f,
        0.000000000f,
        0.000000000f,
        0.000000000f,
        0.000000000f,
        0.000000000f,
        0.000000000f,
        0.000000000f,
        0.000000000f,
        0.000000000f,
        0.000000000f,
        0.000000000f,
        0.000000000f,
        0.000000000f,
        0.000000000f,
        0.000000000f,
        0.000000000f,
        0.000000000f,
        0.000000000f,
        0.000000000f,
        0.000000000f,
        0.000000000f,
        0.000000000f,
        0.406772763f,
        0.792660713f,
        1.05635154f,
        1.27690935f,
        1.40350020f,
        1.38595510f,
        0.950824618f,
        0.000000000f,
        0.000000000f,
        0.0610411242f,
        1.01292658f,
        1.82237661f,
        2.64981270f,
        3.15758562f,
        3.15416884f,
        2.32574773f,
        0.991042674f,
        2.11878896f,
        2.16958523f,
        0.696103752f,
        1.97473192f,
        2.22087717f,
        1.42947054f,
        3.30909586f,
        3.59500527f,
        2.33565593f,
        3.54734874f,
        3.98422337f,
        3.40557742f,
        3.85401320f,
        3.68012881f,
        2.35471392f,
        0.900254428f,
        2.35498857f,
    };

    Halide::ImageParam features(Halide::type_of<float>(), 1);

    features.set(Halide::Buffer<float>(frame.data(), InputDimension));

    std::vector<float> out;
    out.resize(OutputDimension);
    Halide::Buffer<float> result(out.data(), OutputDimension);

    qe.Evaluate(0, features, result);

    for (size_t i = 0; i < out.size(); ++i)
        std::cout << out[i] << std::endl;

}


BOOST_AUTO_TEST_CASE(SpeechLstmModel)
{
    const int InputDimension = 80;
    const int OutputDimension = 9404;

    // Halide
    LstmSpeechEvaluator e;
    e.init("LstmSpeechEvaluator.json");

    QuantizedLstmSpeechEvaluator qe;
    qe.init("LstmSpeechEvaluator.json");

    // CNTK
    auto model = Function::Load(L"lstm_speech.model", DeviceDescriptor::CPUDevice());
    auto input = model->Arguments();
    auto output = model->Output();
    auto inputShape = NDShape({ (size_t)InputDimension });

    // Input
    const int NumberOfFrames = 80;
    std::vector<std::vector<float>> frames;
    frames.resize(NumberOfFrames, std::vector<float>(InputDimension));

    {
        for (int i = 0; i < NumberOfFrames; ++i)
        {
            frames[i].resize(InputDimension);
            for (size_t j = 0; j < frames[i].size(); ++j)
                frames[i][j] = (rand() % 256) / (float)256;
        }
    }

    Halide::ImageParam features(Halide::type_of<float>(), 1);

    auto evaluateFrame = [&](int timestamp, std::vector<float>& f)
    {
        features.set(Halide::Buffer<float>(f.data(), InputDimension));

        std::vector<float> out;
        out.resize(OutputDimension);
        Halide::Buffer<float> result(out.data(), OutputDimension);
        e.Evaluate(timestamp, features, result);

        std::vector<float> output1(OutputDimension, 0);
        for (int j = 0; j < OutputDimension; ++j)
            output1[j] = result(j);

        qe.Evaluate(timestamp, features, result);
        std::vector<float> output2(OutputDimension, 0);
        for (int j = 0; j < OutputDimension; ++j)
            output2[j] = result(j);

        // CNTK evaluation
        auto data = MakeSharedObject<NDArrayView>(DataType::Float, NDShape({ f.size() }), f.data(),
            sizeof(float) * InputDimension, DeviceDescriptor::CPUDevice());
        std::vector<NDArrayViewPtr> d = { data };
        auto value = Value::Create(inputShape, d, { timestamp == 0 ? true : false }, DeviceDescriptor::CPUDevice());

        std::unordered_map<Variable, ValuePtr> i = { { input.front(), value } };
        std::unordered_map<Variable, ValuePtr> o = { { output, nullptr } };

        model->Forward(i, o, DeviceDescriptor::CPUDevice());
        auto result2 = o[output]->Data()->DataBuffer<float>();

        std::vector<float> output3(OutputDimension, 0);
        for (int j = 0; j < OutputDimension; ++j)
            output3[j] = result2[j];

        return std::make_tuple(output1, output2, output3);
    };

    // Perform evaluation
    const int NumberOfIterations = 5;
    for (int i = 0; i < NumberOfIterations; i++)
    {
        int timestamp = 0;
        for (auto& f : frames)
        {
            auto result = evaluateFrame(timestamp++, f);
            for (int j = 0; j < std::get<0>(result).size(); ++j)
            {
                BOOST_REQUIRE_CLOSE_FRACTION(std::get<0>(result)[j], std::get<1>(result)[j], 2.);
                BOOST_REQUIRE_CLOSE(std::get<0>(result)[j], std::get<2>(result)[j], 0.1);
            }
        }
    }
}

class Timer
{
    LARGE_INTEGER m_freq, m_start;

public:
    Timer()
    {
        // count ticks per second
        if (!QueryPerformanceFrequency(&m_freq))
            throw std::exception("Timer: QueryPerformanceFrequency failed!");
    }

    void start()
    {
        if (!QueryPerformanceCounter(&m_start))
            throw std::exception("Timer: QueryPerformanceCounter failed!");
    }

    double stop(bool inMilliseconds = true) const
    {
        LARGE_INTEGER end;
        QueryPerformanceCounter(&end);
        return ((end.QuadPart - m_start.QuadPart) / (double)m_freq.QuadPart) * (inMilliseconds ? 1000 : 1);
    }
};

void MeasurePerf(const std::function<void()>& workload, const std::string& str, int numIterations = 10, bool warmupRun = true)
{
    Timer timer;

    // Warmup run
    if (warmupRun)
        workload();

    std::vector<double> executionTimes(numIterations);
    double averageTime = 0;
    for (int i = 0; i < numIterations; ++i)
    {
        timer.start();
        workload();
        executionTimes[i] = timer.stop();
        averageTime += executionTimes[i];
    }
    averageTime /= (double)numIterations;

    std::sort(executionTimes.begin(), executionTimes.end());

    double stdDev = 0.0;
    std::for_each(executionTimes.begin(), executionTimes.end(), 
        [&stdDev, averageTime, numIterations](double executionTime) { stdDev += (std::abs(executionTime - averageTime)) / numIterations; });

    printf("%s: Min=%.2lf ms, Median=%.2lf ms, Average=%.2lf ms, Max=%.2lf ms, StdDev=%.2lf ms.\n",
        str.c_str(),
        executionTimes[0],
        executionTimes[numIterations / 2],
        averageTime,
        executionTimes[numIterations - 1],
        stdDev);
}

BOOST_AUTO_TEST_CASE(SpeechLstmModelProfiling)
{
    const int InputDimension = 80;
    const int OutputDimension = 9404;

    // Halide
    LstmSpeechEvaluator e;
    e.init("LstmSpeechEvaluator.json");

    std::vector<float> frame;
    frame.resize(InputDimension);

    for (size_t j = 0; j < InputDimension; ++j)
        frame[j] = (rand() % 256) / (float)256;

    Halide::ImageParam features(Halide::type_of<float>(), 1);
    features.set(Halide::Buffer<float>(frame.data(), InputDimension));

    Halide::Buffer<float> result1(OutputDimension);

    auto halide = [&]()
    {
        e.Evaluate(0, features, result1);
    };

    halide();
    halide();
    halide();
}

BOOST_AUTO_TEST_CASE(SpeechLstmModelPerformance)
{
    const int InputDimension = 80;
    const int OutputDimension = 9404;

    // Halide
    LstmSpeechEvaluator e;
    e.init("LstmSpeechEvaluator.json");

    QuantizedLstmSpeechEvaluator qe;
    qe.init("LstmSpeechEvaluator.json");

    std::vector<float> frame;
    frame.resize(InputDimension);

    for (size_t j = 0; j < InputDimension; ++j)
        frame[j] = (rand() % 256) / (float)256;

    Halide::ImageParam features(Halide::type_of<float>(), 1);
    features.set(Halide::Buffer<float>(frame.data(), InputDimension));

    std::vector<float> out;
    out.resize(OutputDimension);
    Halide::Buffer<float> result1(out.data(), OutputDimension);

    auto halide = [&]()
    {
        e.Evaluate(0, features, result1);
    };

    halide();

    std::vector<float> out3;
    out3.resize(OutputDimension);
    Halide::Buffer<float> result3(out3.data(), OutputDimension);

    auto quantized = [&]()
    {
        qe.Evaluate(0, features, result3);
    };

    quantized();

    // CNTK
    auto model = Function::Load(L"lstm_speech.model", DeviceDescriptor::CPUDevice());
    auto input = model->Arguments();
    auto output = model->Output();
    auto inputShape = NDShape({ (size_t)InputDimension });

    auto data = MakeSharedObject<NDArrayView>(DataType::Float, NDShape({ frame.size() }), frame.data(),
            sizeof(float) * InputDimension, DeviceDescriptor::CPUDevice());
    std::vector<NDArrayViewPtr> d = { data };
    auto value = Value::Create(inputShape, d, { true }, DeviceDescriptor::CPUDevice());

    std::unordered_map<Variable, ValuePtr> i = { { input.front(), value } };
    std::unordered_map<Variable, ValuePtr> o = { { output, nullptr } };

    auto cntk = [&]()
    {
        o[output] = nullptr;
        model->Forward(i, o, DeviceDescriptor::CPUDevice());
    };

    cntk();

    auto result2 = o[output]->Data()->DataBuffer<float>();

    for (int j = 0; j < OutputDimension; ++j)
    {
        BOOST_REQUIRE_CLOSE(result1(j), result2[j], 0.1);
        BOOST_REQUIRE_CLOSE_FRACTION(result1(j), result3(j), 1.);
    }

    // All is set up correctly. Run the workloads for perf.
    {
        const int workloads = 5;
        for (int j = 0; j < workloads; j++)
        {
            MeasurePerf(halide,    "Halide workload           " + std::to_string(j), 100);
            MeasurePerf(quantized, "Halide quantized workload " + std::to_string(j), 100);
            MeasurePerf(cntk,      "CNTK workload             " + std::to_string(j), 100);
        }
    }
}

BOOST_AUTO_TEST_CASE(TestVectorQuantization)
{
    std::vector<float> f = { 1.11f, -1.09f };
    auto result = Quantize<float, short>(f, 1);
    BOOST_REQUIRE_CLOSE(result.first[0] * result.second, f[0], 0.01);
    BOOST_REQUIRE_CLOSE(result.first[1] * result.second, f[1], 0.01);
}

BOOST_AUTO_TEST_CASE(TestStorageOrder)
{
    std::vector<float> m;
    for (int i = 0; i < 24; ++i)
        m.push_back((float)i);

    auto bm = Buffer<float>(m.data(), { 4, 3, 2 });

    // First coordinate is column, so it changes the most frequently.
    BOOST_REQUIRE_EQUAL(bm(0, 0, 0), m[0]);
    BOOST_REQUIRE_EQUAL(bm(1, 0, 0), m[1]);
    BOOST_REQUIRE_EQUAL(bm(2, 0, 0), m[2]);
    BOOST_REQUIRE_EQUAL(bm(3, 0, 0), m[3]);
    BOOST_REQUIRE_EQUAL(bm(0, 1, 0), m[4]);
    BOOST_REQUIRE_EQUAL(bm(1, 1, 0), m[5]);
    BOOST_REQUIRE_EQUAL(bm(2, 1, 0), m[6]);
    BOOST_REQUIRE_EQUAL(bm(3, 1, 0), m[7]);
    BOOST_REQUIRE_EQUAL(bm(0, 1, 1), m[16]);
    BOOST_REQUIRE_EQUAL(bm(1, 1, 1), m[17]);
    BOOST_REQUIRE_EQUAL(bm(2, 1, 1), m[18]);
    BOOST_REQUIRE_EQUAL(bm(3, 1, 1), m[19]);
}

BOOST_AUTO_TEST_CASE(TestMatrixQuantization)
{
    std::vector<float> m = { 10.13f, -100.18f, 16.9f, 19.5f, -15.f, -20.f };

    auto r = Quantize<float, short>(m, 1);
    BOOST_REQUIRE_CLOSE(r.first[0] * r.second, m[0], 0.1);
    BOOST_REQUIRE_CLOSE(r.first[1] * r.second, m[1], 0.1);
    BOOST_REQUIRE_CLOSE(r.first[2] * r.second, m[2], 0.1);
    BOOST_REQUIRE_CLOSE(r.first[3] * r.second, m[3], 0.1);
    BOOST_REQUIRE_CLOSE(r.first[4] * r.second, m[4], 0.1);
    BOOST_REQUIRE_CLOSE(r.first[5] * r.second, m[5], 0.1);

    auto bm = Buffer<float>(m.data(), { 3, 2 });
    Halide::Func fbm;
    Halide::Var x, y;
    fbm(x, y) = bm(x, y);

    Halide::Buffer<short> result(3, 2);
    Halide::Buffer<float> step = Halide::Buffer<float>::make_scalar("step");

    auto quantized = Quantize<float, short>(fbm, 2, 3, 1);
    auto p = Halide::Pipeline(quantized);
    p.realize({ result, step });

    BOOST_REQUIRE_CLOSE(step(0), r.second, 0.001);
    BOOST_REQUIRE_EQUAL(result(0, 0), r.first[0]);
    BOOST_REQUIRE_EQUAL(result(1, 0), r.first[1]);
    BOOST_REQUIRE_EQUAL(result(2, 0), r.first[2]);
    BOOST_REQUIRE_EQUAL(result(0, 1), r.first[3]);
    BOOST_REQUIRE_EQUAL(result(1, 1), r.first[4]);
    BOOST_REQUIRE_EQUAL(result(2, 1), r.first[5]);
}

BOOST_AUTO_TEST_CASE(TestQuantizedMatrixMultiplication)
{
    std::vector<float> v = { 1.11f, -1.09f, 8.004f };
    std::vector<float> m = { 10.13f, -100.18f, 16.9f, 19.5f, -15.f, -20.f };
    auto bv = Buffer<float>(v.data(), 3);
    auto bm = Buffer<float>(m.data(), 3, 2);
    Halide::Func fbv;
    Halide::Var index;
    fbv(index) = bv(index);
    Halide::Func fbm;
    Halide::Var x, y;
    fbm(x, y) = bm(x, y);

    auto vq = Quantize<float, short>(fbv, 3, 2);

    // Checking that vector quantization is correct.
    Halide::Buffer<short> vqr(3);
    Halide::Buffer<float> vqs = Halide::Buffer<float>::make_scalar("step");
    Halide::Pipeline(vq).realize({ vqr, vqs });

    BOOST_REQUIRE_CLOSE(vqr(0) * vqs(0), v[0], 0.1);
    BOOST_REQUIRE_CLOSE(vqr(1) * vqs(0), v[1], 0.1);
    BOOST_REQUIRE_CLOSE(vqr(2) * vqs(0), v[2], 0.1);

    auto mq = Quantize<float, short>(fbm, 2, 3, 2);

    // Checking that matrix quantization is correct.
    Halide::Buffer<short> mqr(3, 2);
    Halide::Buffer<float> mqs = Halide::Buffer<float>::make_scalar("step");
    Halide::Pipeline(mq).realize({ mqr, mqs });

    BOOST_REQUIRE_CLOSE(mqr(0, 0) * mqs(0), m[0], 0.1);
    BOOST_REQUIRE_CLOSE(mqr(1, 0) * mqs(0), m[1], 0.1);
    BOOST_REQUIRE_CLOSE(mqr(2, 0) * mqs(0), m[2], 0.1);
    BOOST_REQUIRE_CLOSE(mqr(0, 1) * mqs(0), m[3], 0.1);
    BOOST_REQUIRE_CLOSE(mqr(1, 1) * mqs(0), m[4], 0.1);
    BOOST_REQUIRE_CLOSE(mqr(2, 1) * mqs(0), m[5], 0.1);

    // Actually doing the multiplication
    auto f = MatrixByVectorTimesQuantized(mq, vq, 2, 3);
    auto result = Buffer<float>(2);
    f.realize({ result });

    BOOST_REQUIRE_CLOSE(result(0), 255.7081f, 0.1);
    BOOST_REQUIRE_CLOSE(result(1), -122.085f, 1);
}

BOOST_AUTO_TEST_CASE(TestPlus)
{
    float ca[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    float cb[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

    ImageParam a(Halide::type_of<float>(), 1);
    ImageParam b(Halide::type_of<float>(), 1);

    auto result = Plus(a, b, 10);
    a.set(Buffer<float>((float*)ca, 10));
    b.set(Buffer<float>((float*)cb, 10));

    Buffer<float> output(10);
    result.realize(output);

    for(int i = 0; i < 10; ++i)
        BOOST_REQUIRE_EQUAL(output(i), ca[i] + cb[i]);
}

BOOST_AUTO_TEST_CASE(TestMinus)
{
    float ca[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    float cb[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

    ImageParam a(Halide::type_of<float>(), 1);
    ImageParam b(Halide::type_of<float>(), 1);

    auto result = Minus(a, b, 10);
    a.set(Buffer<float>((float*)ca, 10));
    b.set(Buffer<float>((float*)cb, 10));

    Buffer<float> output(10);
    result.realize(output);

    for (int i = 0; i < 10; ++i)
        BOOST_REQUIRE_EQUAL(output(i), 0);
}

BOOST_AUTO_TEST_CASE(TestElementTimes)
{
    float ca[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    float cb[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

    ImageParam a(Halide::type_of<float>(), 1);
    ImageParam b(Halide::type_of<float>(), 1);

    auto result = ElementTimes(a, b, 10);
    a.set(Buffer<float>((float*)ca, 10));
    b.set(Buffer<float>((float*)cb, 10));

    Buffer<float> output(10);
    result.realize(output);

    for (int i = 0; i < 10; ++i)
        BOOST_REQUIRE_EQUAL(output(i), ca[i] * cb[i]);
}


BOOST_AUTO_TEST_CASE(TestSlice, *utf::tolerance(0.00001))
{
    ImageParam a(Halide::type_of<float>(), 1);
    auto result = Slice(a, 1 ,6);

    float ca[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    a.set(Buffer<float>((float*)ca, 10));

    Buffer<float> output(5);
    result.realize(output);

    for (int i = 0; i < 5; ++i)
    {
        BOOST_REQUIRE_EQUAL(output(i), i + 1);
    }
}

BOOST_AUTO_TEST_CASE(TestSplice, *utf::tolerance(0.00001))
{
    ImageParam a(Halide::type_of<float>(), 1);
    ImageParam b(Halide::type_of<float>(), 1);

    auto result = Splice(a, b, 2, 8);

    float ca[] = { 0, 1 };
    a.set(Buffer<float>((float*)ca, 2));

    float cb[] = { 2, 3, 4, 5, 6, 7, 8, 9 };
    b.set(Buffer<float>((float*)cb, 8));

    Buffer<float> output(10);
    result.realize(output);

    for (int i = 0; i < 10; ++i)
    {
        BOOST_REQUIRE_EQUAL(output(i), i);
    }
}

BOOST_AUTO_TEST_CASE(TestVecMultiply, *utf::tolerance(0.00001))
{
    float ca[] = { 1, 2, 3 };
    float cb[] = { 1, 2, 1, 2, 1, 2, 1, 2, 1 };

    ImageParam a(Halide::type_of<float>(), 1);
    ImageParam b(Halide::type_of<float>(), 2);

    auto result = MatrixByVectorTimes(b, a, 3, 3);
    a.set(Buffer<float>((float*)ca, 3));
    b.set(Buffer<float>((float*)cb, 3, 3));

    Buffer<float> output(3);
    result.realize(output);

    BOOST_REQUIRE_EQUAL(output(0), 8);
    BOOST_REQUIRE_EQUAL(output(1), 10);
    BOOST_REQUIRE_EQUAL(output(2), 8);
}

BOOST_AUTO_TEST_CASE(TestMatrixByVectorProfiling)
{
    const int InputDimension = 256;
    const int OutputDimension = 9404;

    std::vector<float> vec;
    vec.resize(InputDimension);

    for (size_t j = 0; j < vec.size(); ++j)
        vec[j] = (rand() % 256) / (float)256;

    std::vector<float> matrix;
    matrix.resize(InputDimension * OutputDimension);

    for (size_t j = 0; j < matrix.size(); ++j)
        matrix[j] = (rand() % 256) / (float)256;

    Halide::ImageParam features(Halide::type_of<float>(), 1);
    Halide::ImageParam weights(Halide::type_of<float>(), 2);

    Halide::Target t;
    t = Halide::get_jit_target_from_environment()
        .with_feature(Halide::Target::Profile);
    auto exp = MatrixByVectorTimes(weights, features, OutputDimension, InputDimension);

    weights.set(Halide::Buffer<float>(matrix.data(), InputDimension, OutputDimension));
    features.set(Halide::Buffer<float>(vec.data(), InputDimension));
    Halide::Buffer<float> result(OutputDimension);

    auto workload = [&]() { exp.realize({ result }, t); };

    for (int i = 0; i < 10; ++i)
        workload();
}

BOOST_AUTO_TEST_SUITE_END()

}}
