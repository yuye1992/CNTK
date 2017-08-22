#pragma once
#include "HalideDNNLib.h"

namespace CNTK
{
class XorEvaluator final
{
public:
const std::vector<float> get_parameter26() const { return m_Parameter26; }
void set_parameter26(const std::vector<float>&& v) { m_Parameter26 = std::move(v); };
const std::vector<short> get_parameter25() const { return m_Parameter25; }
void set_parameter25(const std::vector<float>&& v) { auto r = Quantize<float, short>(v, 2); m_Parameter25 = r.first; m_step_Parameter25 = r.second; };
const std::vector<float> get_parameter6() const { return m_Parameter6; }
void set_parameter6(const std::vector<float>&& v) { m_Parameter6 = std::move(v); };
const std::vector<short> get_parameter5() const { return m_Parameter5; }
void set_parameter5(const std::vector<float>&& v) { auto r = Quantize<float, short>(v, 2); m_Parameter5 = r.first; m_step_Parameter5 = r.second; };
Halide::Pipeline create_eval_graph(const Halide::ImageParam& Input3)
 {
 Halide::Var var1, var2; 
 auto b_Parameter26 = Halide::Buffer<float>(m_Parameter26.data(), 1, "Parameter26");
Halide::Func Parameter26("Parameter26"); Parameter26(var1) = b_Parameter26(var1);

auto b_Parameter25 = Halide::Buffer<short>(m_Parameter25.data(), 1, 2, "Parameter25");
Halide::Func f_Parameter25("f_Parameter25"); f_Parameter25(var1, var2) = b_Parameter25(var1, var2);
Halide::Func f_step_Parameter25("f_step_Parameter25"); f_step_Parameter25() = m_step_Parameter25;
std::vector<Halide::Func> Parameter25 { f_Parameter25, f_step_Parameter25 };


auto b_Parameter6 = Halide::Buffer<float>(m_Parameter6.data(), 2, "Parameter6");
Halide::Func Parameter6("Parameter6"); Parameter6(var1) = b_Parameter6(var1);

auto b_Parameter5 = Halide::Buffer<short>(m_Parameter5.data(), 2, 2, "Parameter5");
Halide::Func f_Parameter5("f_Parameter5"); f_Parameter5(var1, var2) = b_Parameter5(var1, var2);
Halide::Func f_step_Parameter5("f_step_Parameter5"); f_step_Parameter5() = m_step_Parameter5;
std::vector<Halide::Func> Parameter5 { f_Parameter5, f_step_Parameter5 };


std::vector<Halide::Func> Input3_Times103_quantize; Input3_Times103_quantize = Quantize<float, short>(Input3, 2, 2);
Halide::Func Times103("Times103"); Times103 = VectorByMatrixTimesQuantized(Input3_Times103_quantize,Parameter5,2,2);
Halide::Func Plus105("Plus105"); Plus105 = Plus(Times103, Parameter6);
Halide::Func Tanh107("Tanh107"); Tanh107 = Tanh(Plus105);
std::vector<Halide::Func> Tanh107_Times121_quantize; Tanh107_Times121_quantize = Quantize<float, short>(Tanh107, 2, 2);
Halide::Func Times121("Times121"); Times121 = VectorByMatrixTimesQuantized(Tanh107_Times121_quantize,Parameter25,2,1);
Halide::Func Plus123("Plus123"); Plus123 = Plus(Times121, Parameter26);
 
 return Halide::Pipeline({ Plus123 }); 
 }

private:
std::vector<float> m_Parameter26;
std::vector<short> m_Parameter25;
float m_step_Parameter25;
std::vector<float> m_Parameter6;
std::vector<short> m_Parameter5;
float m_step_Parameter5;
};
};