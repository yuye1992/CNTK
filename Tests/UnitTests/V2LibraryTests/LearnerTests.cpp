//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"
#include "CNTKLibrary.h"
#include "Common.h"
#include <string>
#include <random>
#include <initializer_list>

using namespace CNTK;
using namespace std;

namespace CNTK { namespace Test {

static const size_t maxMinibatchSize = 1000;

static const size_t maxNumAxes = 3;
static const size_t maxDimSize = 5;

template <typename ElementType>
void TestUpdate(LearnerPtr& learner, const NDShape& shape, size_t numMinibatches, const DeviceDescriptor& device)
{
    auto seed = (unsigned long) rng();
    unordered_map<Parameter, NDArrayViewPtr> gradientValues;
    for (auto i = 0; i < numMinibatches; i++)
    { 
        for (auto& parameter : learner->Parameters())
        {
            gradientValues[parameter] = NDArrayView::RandomUniform<ElementType>(shape, -1.0, 1.0, seed + i, device);
        }

        learner->Update(gradientValues, 1);
    }
}

template <typename ElementType>
vector<Parameter> CreateParameters(const NDShape& shape, size_t numParameters, const DeviceDescriptor& device)
{
    vector<Parameter> parameters;
    for (int i = 0; i < numParameters; i++)
    {
        parameters.push_back(
            Parameter(NDArrayView::RandomUniform<ElementType>(shape, -1.0, 1.0, i, device), 
                      L"parameter_" + to_wstring(i)));
    }
    return parameters;
}
  
template <typename ElementType>
void TestSGDLearner(size_t numParameters, size_t numMinibatches, const DeviceDescriptor& device)
{
    NDShape shape = CreateShape(rng() % maxNumAxes + 1, maxDimSize);
    auto parameters = CreateParameters<ElementType>(shape, numParameters, device);
    auto learner = SGDLearner(parameters, LearningRateSchedule(RatePerSample(0.4)));
    TestUpdate<ElementType>(learner, shape, numMinibatches, device);
}

template <typename ElementType>
void TestMomentumSGDLearner(size_t numParameters, size_t numMinibatches, bool unitGainMomentum, const DeviceDescriptor& device)
{
    NDShape shape = CreateShape(rng() % maxNumAxes + 1, maxDimSize);
    auto parameters = CreateParameters<ElementType>(shape, numParameters, device);
    LearningRateSchedule learnigRateSchedule = { { Rate(3.0, Rate::UNKNOWN_REFMBSIZE), Rate(2.0, Rate::UNKNOWN_REFMBSIZE), Rate(1.0, Rate::UNKNOWN_REFMBSIZE) }, numMinibatches };
    MomentumSchedule momentumValues = { { { 1, RatePerSample(1.0) }, { 3, RatePerSample(0.1) }, { 10, RatePerSample(0.01) } }, 2 };
    auto learner = MomentumSGDLearner(parameters, learnigRateSchedule, momentumValues, unitGainMomentum);
    TestUpdate<ElementType>(learner, shape, numMinibatches, device);
    FloatingPointCompare(Learner::LearningRatePerMinibatch(learner->LearningRate(), Rate::UNKNOWN_REFMBSIZE), 2.0, "Learner::LearningRate does not match expectation");
}

template <typename ElementType>
void TestNesterovLearner(size_t numParameters, size_t numMinibatches, bool unitGainMomentum, const DeviceDescriptor& device)
{
    NDShape shape = CreateShape(rng() % maxNumAxes + 1, maxDimSize);
    auto parameters = CreateParameters<ElementType>(shape, numParameters, device);
    MomentumSchedule momentumValues = { { { 1, MomentumRateAsTimeConstant(1) }, { 3, MomentumRateAsTimeConstant(5) }, { 10, MomentumRateAsTimeConstant(25) } }, 100 };
    auto learner = NesterovLearner(parameters, LearningRateSchedule( { { 1, Rate(0.5, Rate::UNKNOWN_REFMBSIZE) }, { 10, Rate(0.25, Rate::UNKNOWN_REFMBSIZE) }, { 20, Rate(0.125, Rate::UNKNOWN_REFMBSIZE) } }, 3 ), momentumValues, unitGainMomentum);
    TestUpdate<ElementType>(learner, shape, numMinibatches, device);
}

template <typename ElementType>
void TestAdaGradLearner(size_t numParameters, size_t numMinibatches, const DeviceDescriptor& device)
{
    NDShape shape = CreateShape(rng() % maxNumAxes + 1, maxDimSize);
    auto parameters = CreateParameters<ElementType>(shape, numParameters, device);
    auto learner = AdaGradLearner(parameters, LearningRateSchedule( { RatePerSample(0.5), RatePerSample(0.4), RatePerSample(0.3), RatePerSample(0.2), RatePerSample(0.1) }, 2), true);
    TestUpdate<ElementType>(learner, shape, numMinibatches, device);
}

template <typename ElementType>
void TestFSAdaGradLearner(size_t numParameters, size_t numMinibatches, bool unitGainMomentum, const DeviceDescriptor& device)
{
    NDShape shape = CreateShape(rng() % maxNumAxes + 1, maxDimSize);
    auto parameters = CreateParameters<ElementType>(shape, numParameters, device);
    auto learner = FSAdaGradLearner(parameters, 
                    LearningRateSchedule({ RatePerSample(0.5) }), 
                    MomentumSchedule({ MomentumRateAsTimeConstant(10), MomentumRateAsTimeConstant(100), MomentumRateAsTimeConstant(1000)}), unitGainMomentum);
    TestUpdate<ElementType>(learner, shape, numMinibatches, device);
}

template <typename ElementType>
void TestAdamLearner(size_t numParameters, size_t numMinibatches, bool unitGainMomentum, const DeviceDescriptor& device)
{
    NDShape shape = CreateShape(rng() % maxNumAxes + 1, maxDimSize);
    auto parameters = CreateParameters<ElementType>(shape, numParameters, device);
    auto learner = AdamLearner(parameters, 
                LearningRateSchedule({ RatePerSample(0.5) }), 
                MomentumSchedule({ MomentumRateAsTimeConstant(10), MomentumRateAsTimeConstant(100), MomentumRateAsTimeConstant(1000) }), 
                unitGainMomentum, 
                MomentumSchedule(RatePerSample(0.99)));
    TestUpdate<ElementType>(learner, shape, numMinibatches, device);
}

template <typename ElementType>
void TestAdamaxLearner(size_t numParameters, size_t numMinibatches, bool unitGainMomentum, const DeviceDescriptor& device)
{
    NDShape shape = CreateShape(rng() % maxNumAxes + 1, maxDimSize);
    auto parameters = CreateParameters<ElementType>(shape, numParameters, device);
    auto learner = AdamLearner(parameters, 
                    LearningRateSchedule({ RatePerSample(0.5) }), 
                    MomentumSchedule({ MomentumRateAsTimeConstant(10), MomentumRateAsTimeConstant(100), MomentumRateAsTimeConstant(1000) }), 
                    unitGainMomentum, 
                    MomentumSchedule(RatePerSample(0.99)), 1e-8, true);
    TestUpdate<ElementType>(learner, shape, numMinibatches, device);
}

template <typename ElementType>
void TestRMSPropLearner(size_t numParameters, size_t numMinibatches, const DeviceDescriptor& device)
{
    NDShape shape = CreateShape(rng() % maxNumAxes + 1, maxDimSize);
    auto parameters = CreateParameters<ElementType>(shape, numParameters, device);
    auto learner = RMSPropLearner(parameters, 
            LearningRateSchedule({ { 3, Rate(0.7, Rate::UNKNOWN_REFMBSIZE) }, { 1, Rate(0.2, Rate::UNKNOWN_REFMBSIZE) } }), 0.95, 1.2, 0.7, 10.0, 0.001);
    TestUpdate<ElementType>(learner, shape, numMinibatches, device);
}

template <typename ElementType>
void TestUniversalLearner(size_t numParameters, size_t numMinibatches, const DeviceDescriptor& device)
{
    NDShape shape = CreateShape(rng() % maxNumAxes + 1, maxDimSize);
    auto parameters = CreateParameters<ElementType>(shape, numParameters, device);
    ElementType lr = (ElementType) 0.06125;
    ParameterUpdateFunctor mysgd = [lr](Parameter p, Variable g) -> FunctionPtr 
    { 
        return Assign(p, Minus(p , ElementTimes(Constant::Scalar(lr), g))); 
    };
    auto learner = UniversalLearner(parameters, mysgd);
    TestUpdate<ElementType>(learner, shape, numMinibatches, device);

}

void TestTrainingParametersSchedule()
{
    LearningRateSchedule schedule1 = RatePerSample(0.5);
    assert(schedule1[0] == Rate(0.5, 1));
    assert(schedule1[1] == Rate(0.5, 1));
    assert(schedule1[100] == Rate(0.5, 1));

    LearningRateSchedule schedule2 = { RatePerSample(0.5) };
    assert(schedule2[0] == Rate(0.5, 1));
    assert(schedule2[10] == Rate(0.5, 1));
    assert(schedule2[100] == Rate(0.5, 1));

    LearningRateSchedule schedule3({ RatePerSample(0.5), RatePerSample(0.3), RatePerSample(0.3) });
    assert(schedule3[0] == Rate(0.5, 1));
    assert(schedule3[1] == Rate(0.3, 1));
    assert(schedule3[100] == Rate(0.3, 1));

    LearningRateSchedule schedule4 = { vector<Rate>{ Rate(0.5, Rate::UNKNOWN_REFMBSIZE) }, 10 }; // without vector<> gcc complains that conversion here is ambiguousS
    assert(schedule4[0] == Rate(0.5, Rate::UNKNOWN_REFMBSIZE));
    assert(schedule4[10] == Rate(0.5, Rate::UNKNOWN_REFMBSIZE));
    assert(schedule4[100] == Rate(0.5, Rate::UNKNOWN_REFMBSIZE));

    LearningRateSchedule schedule5 = { { RatePerSample(0.5), RatePerSample(0.3), RatePerSample(0.2) }, 10 };
    assert(schedule5[0] == Rate(0.5, 1));
    assert(schedule5[9] == Rate(0.5, 1));
    assert(schedule5[10] == Rate(0.3, 1));
    assert(schedule5[19] == Rate(0.3, 1));
    assert(schedule5[20] == Rate(0.2, 1));
    assert(schedule5[100] == Rate(0.2, 1));

    MomentumSchedule schedule6 = { { make_pair(1, Rate(0.5, Rate::UNKNOWN_REFMBSIZE)) } }; // without make_pair this is interpreted as a vector of doubles
    assert(schedule6[0] == Rate(0.5, Rate::UNKNOWN_REFMBSIZE));
    assert(schedule6[10] == Rate(0.5, Rate::UNKNOWN_REFMBSIZE));
    assert(schedule6[100] == Rate(0.5, Rate::UNKNOWN_REFMBSIZE));

    LearningRateSchedule schedule7 = { { { 1, Rate(0.5, Rate::UNKNOWN_REFMBSIZE) }, { 1, Rate(0.3, Rate::UNKNOWN_REFMBSIZE) }, { 1, Rate(0.2, Rate::UNKNOWN_REFMBSIZE) } } };
    assert(schedule7[0] == Rate(0.5, Rate::UNKNOWN_REFMBSIZE));
    assert(schedule7[1] == Rate(0.3, Rate::UNKNOWN_REFMBSIZE));
    assert(schedule7[2] == Rate(0.2, Rate::UNKNOWN_REFMBSIZE));
    assert(schedule7[100] == Rate(0.2, Rate::UNKNOWN_REFMBSIZE));

    MomentumSchedule schedule8 = { { { 1, Rate(0.5, Rate::UNKNOWN_REFMBSIZE) }, { 1, Rate(0.3, Rate::UNKNOWN_REFMBSIZE) }, { 1, Rate(0.2, Rate::UNKNOWN_REFMBSIZE) } }, 10 };
    assert(schedule8[0] == Rate(0.5, Rate::UNKNOWN_REFMBSIZE));
    assert(schedule8[9] == Rate(0.5, Rate::UNKNOWN_REFMBSIZE));
    assert(schedule8[10] == Rate(0.3, Rate::UNKNOWN_REFMBSIZE));
    assert(schedule8[19] == Rate(0.3, Rate::UNKNOWN_REFMBSIZE));
    assert(schedule8[20] == Rate(0.2, Rate::UNKNOWN_REFMBSIZE));
    assert(schedule8[100] == Rate(0.2, Rate::UNKNOWN_REFMBSIZE));

    LearningRateSchedule schedule9 = { { { 3, RatePerSample(0.5) }, { 2, RatePerSample(0.3) }, { 1, RatePerSample(0.2) } } };
    assert(schedule9[0] == Rate(0.5, 1));
    assert(schedule9[2] == Rate(0.5, 1));
    assert(schedule9[3] == Rate(0.3, 1));
    assert(schedule9[4] == Rate(0.3, 1));
    assert(schedule9[5] == Rate(0.2, 1));
    assert(schedule9[100] == Rate(0.2, 1));

    MomentumSchedule schedule10 = { { { 3, Rate(0.5, Rate::UNKNOWN_REFMBSIZE) }, { 2, Rate(0.3, Rate::UNKNOWN_REFMBSIZE) }, { 1, Rate(0.2, Rate::UNKNOWN_REFMBSIZE) } }, 10 };
    assert(schedule10[0] == Rate(0.5, Rate::UNKNOWN_REFMBSIZE));
    assert(schedule10[29] == Rate(0.5, Rate::UNKNOWN_REFMBSIZE));
    assert(schedule10[30] == Rate(0.3, Rate::UNKNOWN_REFMBSIZE));
    assert(schedule10[49] == Rate(0.3, Rate::UNKNOWN_REFMBSIZE));
    assert(schedule10[50] == Rate(0.2, Rate::UNKNOWN_REFMBSIZE));
    assert(schedule10[100] == Rate(0.2, Rate::UNKNOWN_REFMBSIZE));

    MomentumSchedule schedule11 = { { MomentumRateAsTimeConstant(0), MomentumRateAsTimeConstant(1), MomentumRateAsTimeConstant(2) }, 10 };
    assert(Learner::ExponetialDecayRateForMinibatch(schedule11[0], 1) == 0.0);
    assert(Learner::ExponetialDecayRateForMinibatch(schedule11[9], 1) == 0.0);
    assert(Learner::ExponetialDecayRateForMinibatch(schedule11[10], 1) == exp(-1.0 / 1.0));
    assert(Learner::ExponetialDecayRateForMinibatch(schedule11[19], 1) == exp(-1.0 / 1.0));
    assert(Learner::ExponetialDecayRateForMinibatch(schedule11[20], 1) == exp(-1.0 / 2.0));
    assert(Learner::ExponetialDecayRateForMinibatch(schedule11[30], 1) == exp(-1.0 / 2.0));

    MomentumSchedule schedule12 = schedule11;
    assert(Learner::ExponetialDecayRateForMinibatch(schedule12[0], 1) == 0.0);
    assert(Learner::ExponetialDecayRateForMinibatch(schedule12[9], 1) == 0.0);
    assert(Learner::ExponetialDecayRateForMinibatch(schedule12[10], 1) == exp(-1.0 / 1.0));
    assert(Learner::ExponetialDecayRateForMinibatch(schedule12[19], 1) == exp(-1.0 / 1.0));
    assert(Learner::ExponetialDecayRateForMinibatch(schedule12[20], 1) == exp(-1.0 / 2.0));
    assert(Learner::ExponetialDecayRateForMinibatch(schedule12[30], 1) == exp(-1.0 / 2.0));
     
    MomentumSchedule schedule13 = MomentumAsTimeConstantSchedule(1);
    assert(Learner::ExponetialDecayRateForMinibatch(schedule13[0], 1) == exp(-1.0 / 1.0));
    assert(Learner::ExponetialDecayRateForMinibatch(schedule13[1], 1) == exp(-1.0 / 1.0));
    assert(Learner::ExponetialDecayRateForMinibatch(schedule13[100], 1) == exp(-1.0 / 1.0));

    MomentumSchedule schedule14 = { { MomentumRateAsTimeConstant(1), MomentumRateAsTimeConstant(2), MomentumRateAsTimeConstant(3) } };
    assert(Learner::ExponetialDecayRateForMinibatch(schedule14[0], 1) == exp(-1.0 / 1.0));
    assert(Learner::ExponetialDecayRateForMinibatch(schedule14[1], 1) == exp(-1.0 / 2.0));
    assert(Learner::ExponetialDecayRateForMinibatch(schedule14[2], 1) == exp(-1.0 / 3.0));
    assert(Learner::ExponetialDecayRateForMinibatch(schedule14[100], 1) == exp(-1.0 / 3.0));
    
    MomentumSchedule schedule15 = { { { 100, MomentumRateAsTimeConstant(7) }, { 10, MomentumRateAsTimeConstant(5) }, { 1, MomentumRateAsTimeConstant(3) } }, 100 };

    auto dict = schedule15.Serialize();

    TrainingParameterSchedule<Rate> schedule16 = TrainingParameterSchedule<Rate>::Deserialize(dict);
    assert(Learner::ExponetialDecayRateForMinibatch(schedule16[0], 1) == exp(-1.0 / 7.0));
    assert(Learner::ExponetialDecayRateForMinibatch(schedule16[9999], 1) == exp(-1.0 / 7.0));
    assert(Learner::ExponetialDecayRateForMinibatch(schedule16[10000], 1) == exp(-1.0 / 5.0));
    assert(Learner::ExponetialDecayRateForMinibatch(schedule16[10999], 1) == exp(-1.0 / 5.0));
    assert(Learner::ExponetialDecayRateForMinibatch(schedule16[11000], 1) == exp(-1.0 / 3.0));
    assert(Learner::ExponetialDecayRateForMinibatch(schedule16[99999], 1) == exp(-1.0 / 3.0));
}

void TestDefaultUnitGainGetterAndSetter()
{
    assert(DefaultUnitGainValue());

    SetDefaultUnitGainValue(false);
    assert(!DefaultUnitGainValue());

    SetDefaultUnitGainValue(true);
    assert(DefaultUnitGainValue());
}

void TestSweepBasedSchedule()
{
    DeviceDescriptor device = DeviceDescriptor::CPUDevice();
    auto schedule = LearningRateSchedule({ RatePerSample(1), RatePerSample(2), RatePerSample(3), RatePerSample(4), RatePerSample(5) }, LearningRateSchedule::FullDataSweep);

    auto weights = Parameter({ 2 }, DataType::Float, 0, device);
    auto learner1 = SGDLearner({ weights }, schedule);
    assert(1 == learner1->LearningRatePerSample());

    
    for (auto i : {2, 3, 4, 5 })
    {
        std::vector<float> gradientValueVector(weights.Shape().TotalSize(), 0);
        auto gradientValue = MakeSharedObject<NDArrayView>(weights.Shape(), gradientValueVector);
        std::unordered_map<Parameter, NDArrayViewPtr> gradients{ { weights, gradientValue } };
        learner1->Update(gradients, 1, true);
        assert(i == learner1->LearningRatePerSample());
    }

    const size_t inputDim = 2;
    const size_t numOutputClasses = 2;
    auto minibatchSource = TextFormatMinibatchSource(L"SimpleDataTest_cntk_text.txt", { { L"features", inputDim }, { L"labels", numOutputClasses } });

    auto sweepSize = 603; // == wc -l SimpleDataTest_cntk_text.txt
    auto minibatchSize = 400; 
    auto featureStreamInfo = minibatchSource->StreamInfo(L"features");
    auto labelStreamInfo = minibatchSource->StreamInfo(L"labels");

    auto input = InputVariable({ inputDim }, DataType::Float, L"features");
    auto labels = InputVariable({ numOutputClasses }, DataType::Float, L"labels");


    auto classifierOutput = FullyConnectedLinearLayer(input, numOutputClasses, device);
    auto trainingLoss = CNTK::CrossEntropyWithSoftmax(classifierOutput, labels, L"lossFunction");
    auto prediction = CNTK::ClassificationError(classifierOutput, labels, L"classificationError");
    auto learner2 = SGDLearner(classifierOutput->Parameters(), schedule);
    auto trainer = CreateTrainer(classifierOutput, trainingLoss, prediction, { learner2 });

    for (auto i = 0; i <= 4000; i += minibatchSize)
    {
        auto sweepIndex1 = i / sweepSize;
        auto minibatchData = minibatchSource->GetNextMinibatch(minibatchSize, device);

        if (minibatchData[featureStreamInfo].sweepEnd != minibatchData[labelStreamInfo].sweepEnd) {
            ReportFailure("TestSweepBasedSchedule failed: "
                "different streams have different end of sweep flag values.");
        }

        auto sweepIndex2 = (i + minibatchSize) / sweepSize;

        if ((sweepIndex1 != sweepIndex2) != minibatchData[labelStreamInfo].sweepEnd) {
            ReportFailure("TestSweepBasedSchedule failed: "
                "end of sweep flag value is different from expected.");
        }
       
        trainer->TrainMinibatch({ { input, minibatchData[featureStreamInfo] }, { labels, minibatchData[labelStreamInfo] } }, device);
        auto expectedLR = std::min((sweepIndex2 + 1), 5);

        if (expectedLR != learner2->LearningRatePerSample()) {
            ReportFailure("TestSweepBasedSchedule failed: "
                "learning rate value is different from expected.");
        }
    }
}

struct LearnerSuiteFixture
{
    LearnerSuiteFixture()
        : unitGain{ true, false }
    {
        srand(1);
        if (ShouldRunOnCpu())
            devices.push_back(DeviceDescriptor::CPUDevice());
        if (ShouldRunOnGpu())
            devices.push_back(DeviceDescriptor::GPUDevice(0));

        numParameters = 1 + rand() % 5;
        numMinibatches = 1 + rand() % 5;
    }
    bool unitGain[2];
    vector<DeviceDescriptor> devices;
    int numParameters;
    int numMinibatches;
};

BOOST_FIXTURE_TEST_SUITE(LearnerSuite, LearnerSuiteFixture)

BOOST_AUTO_TEST_CASE(DefaultUnitGainGetterAndSetter)
{
    TestDefaultUnitGainGetterAndSetter();
}

BOOST_AUTO_TEST_CASE(SweepBasedSchedule)
{
    TestSweepBasedSchedule();
}

BOOST_AUTO_TEST_CASE(TrainingParametersSchedule)
{
    TestTrainingParametersSchedule();
}

BOOST_AUTO_TEST_CASE(CreateAndUpdateSGDLearner)
{
    for (auto& device : devices)
    {
        TestSGDLearner<double>(numParameters, numMinibatches, device);
    }
}

BOOST_AUTO_TEST_CASE(CreateAndUpdateAdaGradLearner)
{
    for (auto& device : devices)
    {
        TestAdaGradLearner<double>(numParameters, numMinibatches, device);
    }
}

BOOST_AUTO_TEST_CASE(CreateAndUpdateRMSPropLearner)
{
    for (auto& device : devices)
    {
        TestRMSPropLearner<float>(numParameters, numMinibatches, device);
    }
}

BOOST_AUTO_TEST_CASE(CreateAndUpdateMomentumLearner)
{
    for (auto& device : devices)
    {
        for (auto gain : unitGain)
        {
            TestMomentumSGDLearner<float>(numParameters, numMinibatches, gain, device);
        }
    }
}

BOOST_AUTO_TEST_CASE(CreateAndUpdateNesterovLearner)
{
    for (auto& device : devices)
    {
        for (auto& gain : unitGain)
        {
            TestNesterovLearner<float>(numParameters, numMinibatches, gain, device);
        }
    }
}

BOOST_AUTO_TEST_CASE(CreateAndUpdateFSAdaGradLearner)
{
    for (auto& device : devices)
    {
        for (auto& gain : unitGain)
        {
            TestFSAdaGradLearner<double>(numParameters, numMinibatches, gain, device);
        }
    }
}

BOOST_AUTO_TEST_CASE(CreateAndUpdateAdamLearner)
{
    for (auto& device : devices)
    {
        for (auto& gain : unitGain)
        {
            TestAdamLearner<float>(numParameters, numMinibatches, gain, device);
            TestAdamLearner<double>(numParameters, numMinibatches, gain, device);
        }
    }
}

BOOST_AUTO_TEST_CASE(CreateAndUpdateUniversalLearner)
{
    for (auto& device : devices)
    {
        TestUniversalLearner<float>(numParameters, numMinibatches, device);
        TestUniversalLearner<double>(numParameters, numMinibatches, device);
    }
}

BOOST_AUTO_TEST_CASE(CreateAndUpdateAdamaxLearner)
{
    for (auto& device : devices)
    {
        for (auto& gain : unitGain)
        {
            TestAdamaxLearner<float>(numParameters, numMinibatches, gain, device);
            TestAdamaxLearner<double>(numParameters, numMinibatches, gain, device);
        }
    }
}

BOOST_AUTO_TEST_CASE(TestResettingLearningRate)
{
    NDShape shape = { 1 };
    auto numSamples = 1; numParameters = 1, numMinibatches = 1;
    DeviceDescriptor device = DeviceDescriptor::CPUDevice();
    auto parameters = CreateParameters<float>(shape, numParameters, device);
    auto learner = SGDLearner(parameters, LearningRateSchedule({ RatePerSample(0.1), RatePerSample(1), RatePerSample(2), RatePerSample(3), RatePerSample(4), RatePerSample(5) }, numSamples));
    BOOST_TEST(learner->LearningRatePerSample() == 0.1);
    for (int i = 1; i < 4; i++)
    {
        TestUpdate<float>(learner, shape, numMinibatches, device);
        BOOST_TEST(learner->LearningRatePerSample() == float(i));
    }

    learner->ResetLearningRate(LearningRateSchedule({ RatePerSample(9), RatePerSample(10), RatePerSample(20), RatePerSample(30), RatePerSample(40), RatePerSample(50) }, numSamples));
    BOOST_TEST(learner->LearningRatePerSample() == 9.0);
    for (int i = 1; i < 4; i++)
    {
        TestUpdate<float>(learner, shape, numMinibatches, device);
        BOOST_TEST(learner->LearningRatePerSample() == float(i*10));
    }
}

BOOST_AUTO_TEST_SUITE_END()

}}
