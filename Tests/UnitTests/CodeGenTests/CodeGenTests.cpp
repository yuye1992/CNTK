//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"
#include "Common.h"

#pragma warning(push)
#pragma warning(disable : 4715)
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#pragma warning(pop)

// Due to inclusion of windows.h
#undef min
#undef max

#include "generated/XorEvaluator.h"

#include "CNTKLibrary.h"

namespace utf = boost::unit_test;

namespace pt = boost::property_tree;

namespace CNTK { namespace Test {

using Halide::ImageParam;

template<class T>
using Buffer = Halide::Buffer<T>;

BOOST_AUTO_TEST_SUITE(CodeGenDeserializationSuite)

BOOST_AUTO_TEST_CASE(XorOperation, * utf::tolerance(0.01))
{
    pt::ptree root;
    pt::read_json("XorWeights.json", root);

    auto get_value = [&](const std::string& name)
    {
        std::vector<float> result;
        for (auto& v : root.get_child(name))
            result.push_back(v.second.get_value<float>());
        return result;
    };

    // Halide
    XorEvaluator e;
    e.set_parameter26(get_value("Parameter26"));
    //e.set_parameter25(get_value("Parameter25"));
    e.set_parameter6(get_value("Parameter6"));
    //e.set_parameter5(get_value("Parameter5"));

    Halide::ImageParam features(Halide::type_of<float>(), 1);
    auto f = e.create_eval_graph(features);

    std::vector<float> v = { 0.f, 1.f };
    features.set(Halide::Buffer<float>(v.data(), 2));

    Halide::Target target;
    Halide::Buffer<float> result(1);

    f.realize({ result }, target);
    auto result10 = result(0);

    v[0] = 1.f;
    f.realize({ result }, target);
    auto result11 = result(0);

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

    BOOST_REQUIRE_EQUAL(result10, result20);
    BOOST_REQUIRE_EQUAL(result11, result21);
}

BOOST_AUTO_TEST_CASE(TestPlus, *utf::tolerance(0.00001))
{
    float ca[] = { 1, 2, 3 };
    float cb[] = { 4, 5, 6 };

    ImageParam a(Halide::type_of<float>(), 1);
    ImageParam b(Halide::type_of<float>(), 1);

    auto result = Plus(a, b);
    a.set(Buffer<float>((float*)ca, 3));
    b.set(Buffer<float>((float*)cb, 3));

    Buffer<float> output(3);
    result.realize(output);

    BOOST_REQUIRE_EQUAL(output(0), 5);
    BOOST_REQUIRE_EQUAL(output(1), 7);
    BOOST_REQUIRE_EQUAL(output(2), 9);
}

BOOST_AUTO_TEST_CASE(TestSlice, *utf::tolerance(0.00001))
{
    ImageParam a(Halide::type_of<float>(), 1);
    auto result = Slice<1, 6>(a);

    float ca[] = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    a.set(Buffer<float>((float*)ca, 10));

    Buffer<float> output(5);
    result.realize(output);

    for (int i = 0; i < 5; ++i)
    {
        BOOST_REQUIRE_EQUAL(output(i), i + 1);
    }
}

BOOST_AUTO_TEST_CASE(TestVecMultiply, *utf::tolerance(0.00001))
{
    float ca[] = { 1, 2, 3 };
    float cb[] = { 1, 2, 1, 2, 1, 2, 1, 2, 1 };

    ImageParam a(Halide::type_of<float>(), 1);
    ImageParam b(Halide::type_of<float>(), 2);

    auto result = VectorByMatrixTimes<3, 3>(a, b);
    a.set(Buffer<float>((float*)ca, 3));
    b.set(Buffer<float>((float*)cb, 3, 3));

    Buffer<float> output(3);
    result.realize(output);

    BOOST_REQUIRE_EQUAL(output(0), 8);
    BOOST_REQUIRE_EQUAL(output(1), 10);
    BOOST_REQUIRE_EQUAL(output(2), 8);
}

BOOST_AUTO_TEST_SUITE_END()

}}
