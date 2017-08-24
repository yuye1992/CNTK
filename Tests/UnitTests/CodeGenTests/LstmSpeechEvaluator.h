#pragma once
#include <vector>
#include <string>
#include "HalideDNNLib.h"
#pragma warning(push)
#pragma warning(disable : 4715)
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#pragma warning(pop)

namespace CNTK
{
    class LstmSpeechEvaluator final
    {
    public:
        const std::vector<float> get_constant555() const { return m_Constant555; }
        void set_constant555(const std::vector<float>&& v) { m_Constant555 = std::move(v); };
        const std::vector<float> get_parameter550() const { return m_Parameter550; }
        void set_parameter550(const std::vector<float>&& v) { m_Parameter550 = std::move(v); };
        const std::vector<float> get_parameter6() const { return m_Parameter6; }
        void set_parameter6(const std::vector<float>&& v) { m_Parameter6 = std::move(v); };
        const std::vector<float> get_parameter8() const { return m_Parameter8; }
        void set_parameter8(const std::vector<float>&& v) { m_Parameter8 = std::move(v); };
        const std::vector<float> get_parameter10() const { return m_Parameter10; }
        void set_parameter10(const std::vector<float>&& v) { m_Parameter10 = std::move(v); };
        const std::vector<float> get_parameter527() const { return m_Parameter527; }
        void set_parameter527(const std::vector<float>&& v) { m_Parameter527 = std::move(v); };
        const std::vector<float> get_parameter508() const { return m_Parameter508; }
        void set_parameter508(const std::vector<float>&& v) { m_Parameter508 = std::move(v); };
        const std::vector<float> get_parameter495() const { return m_Parameter495; }
        void set_parameter495(const std::vector<float>&& v) { m_Parameter495 = std::move(v); };
        const std::vector<float> get_parameter488() const { return m_Parameter488; }
        void set_parameter488(const std::vector<float>&& v) { m_Parameter488 = std::move(v); };
        const std::vector<float> get_parameter469() const { return m_Parameter469; }
        void set_parameter469(const std::vector<float>&& v) { m_Parameter469 = std::move(v); };
        const std::vector<float> get_constant459() const { return m_Constant459; }
        void set_constant459(const std::vector<float>&& v) { m_Constant459 = std::move(v); };
        const std::vector<float> get_parameter457() const { return m_Parameter457; }
        void set_parameter457(const std::vector<float>&& v) { m_Parameter457 = std::move(v); };
        const std::vector<float> get_parameter446() const { return m_Parameter446; }
        void set_parameter446(const std::vector<float>&& v) { m_Parameter446 = std::move(v); };
        const std::vector<float> get_parameter17() const { return m_Parameter17; }
        void set_parameter17(const std::vector<float>&& v) { m_Parameter17 = std::move(v); };
        const std::vector<float> get_parameter19() const { return m_Parameter19; }
        void set_parameter19(const std::vector<float>&& v) { m_Parameter19 = std::move(v); };
        const std::vector<float> get_constant429() const { return m_Constant429; }
        void set_constant429(const std::vector<float>&& v) { m_Constant429 = std::move(v); };
        const std::vector<float> get_parameter22() const { return m_Parameter22; }
        void set_parameter22(const std::vector<float>&& v) { m_Parameter22 = std::move(v); };
        const std::vector<float> get_parameter411() const { return m_Parameter411; }
        void set_parameter411(const std::vector<float>&& v) { m_Parameter411 = std::move(v); };
        const std::vector<float> get_parameter392() const { return m_Parameter392; }
        void set_parameter392(const std::vector<float>&& v) { m_Parameter392 = std::move(v); };
        const std::vector<float> get_parameter379() const { return m_Parameter379; }
        void set_parameter379(const std::vector<float>&& v) { m_Parameter379 = std::move(v); };
        const std::vector<float> get_parameter372() const { return m_Parameter372; }
        void set_parameter372(const std::vector<float>&& v) { m_Parameter372 = std::move(v); };
        const std::vector<float> get_parameter353() const { return m_Parameter353; }
        void set_parameter353(const std::vector<float>&& v) { m_Parameter353 = std::move(v); };
        const std::vector<float> get_constant343() const { return m_Constant343; }
        void set_constant343(const std::vector<float>&& v) { m_Constant343 = std::move(v); };
        const std::vector<float> get_parameter341() const { return m_Parameter341; }
        void set_parameter341(const std::vector<float>&& v) { m_Parameter341 = std::move(v); };
        const std::vector<float> get_parameter330() const { return m_Parameter330; }
        void set_parameter330(const std::vector<float>&& v) { m_Parameter330 = std::move(v); };
        const std::vector<float> get_parameter29() const { return m_Parameter29; }
        void set_parameter29(const std::vector<float>&& v) { m_Parameter29 = std::move(v); };
        const std::vector<float> get_parameter31() const { return m_Parameter31; }
        void set_parameter31(const std::vector<float>&& v) { m_Parameter31 = std::move(v); };
        const std::vector<float> get_constant313() const { return m_Constant313; }
        void set_constant313(const std::vector<float>&& v) { m_Constant313 = std::move(v); };
        const std::vector<float> get_parameter34() const { return m_Parameter34; }
        void set_parameter34(const std::vector<float>&& v) { m_Parameter34 = std::move(v); };
        const std::vector<float> get_parameter295() const { return m_Parameter295; }
        void set_parameter295(const std::vector<float>&& v) { m_Parameter295 = std::move(v); };
        const std::vector<float> get_parameter276() const { return m_Parameter276; }
        void set_parameter276(const std::vector<float>&& v) { m_Parameter276 = std::move(v); };
        const std::vector<float> get_parameter263() const { return m_Parameter263; }
        void set_parameter263(const std::vector<float>&& v) { m_Parameter263 = std::move(v); };
        const std::vector<float> get_parameter256() const { return m_Parameter256; }
        void set_parameter256(const std::vector<float>&& v) { m_Parameter256 = std::move(v); };
        const std::vector<float> get_parameter237() const { return m_Parameter237; }
        void set_parameter237(const std::vector<float>&& v) { m_Parameter237 = std::move(v); };
        const std::vector<float> get_constant227() const { return m_Constant227; }
        void set_constant227(const std::vector<float>&& v) { m_Constant227 = std::move(v); };
        const std::vector<float> get_parameter225() const { return m_Parameter225; }
        void set_parameter225(const std::vector<float>&& v) { m_Parameter225 = std::move(v); };
        const std::vector<float> get_parameter214() const { return m_Parameter214; }
        void set_parameter214(const std::vector<float>&& v) { m_Parameter214 = std::move(v); };
        const std::vector<float> get_parameter41() const { return m_Parameter41; }
        void set_parameter41(const std::vector<float>&& v) { m_Parameter41 = std::move(v); };
        const std::vector<float> get_parameter43() const { return m_Parameter43; }
        void set_parameter43(const std::vector<float>&& v) { m_Parameter43 = std::move(v); };
        const std::vector<float> get_constant197() const { return m_Constant197; }
        void set_constant197(const std::vector<float>&& v) { m_Constant197 = std::move(v); };
        const std::vector<float> get_parameter46() const { return m_Parameter46; }
        void set_parameter46(const std::vector<float>&& v) { m_Parameter46 = std::move(v); };
        const std::vector<float> get_parameter179() const { return m_Parameter179; }
        void set_parameter179(const std::vector<float>&& v) { m_Parameter179 = std::move(v); };
        const std::vector<float> get_parameter160() const { return m_Parameter160; }
        void set_parameter160(const std::vector<float>&& v) { m_Parameter160 = std::move(v); };
        const std::vector<float> get_parameter147() const { return m_Parameter147; }
        void set_parameter147(const std::vector<float>&& v) { m_Parameter147 = std::move(v); };
        const std::vector<float> get_parameter140() const { return m_Parameter140; }
        void set_parameter140(const std::vector<float>&& v) { m_Parameter140 = std::move(v); };
        const std::vector<float> get_parameter121() const { return m_Parameter121; }
        void set_parameter121(const std::vector<float>&& v) { m_Parameter121 = std::move(v); };
        const std::vector<float> get_constant111() const { return m_Constant111; }
        void set_constant111(const std::vector<float>&& v) { m_Constant111 = std::move(v); };
        const std::vector<float> get_parameter109() const { return m_Parameter109; }
        void set_parameter109(const std::vector<float>&& v) { m_Parameter109 = std::move(v); };
        const std::vector<float> get_parameter98() const { return m_Parameter98; }
        void set_parameter98(const std::vector<float>&& v) { m_Parameter98 = std::move(v); };
        const std::vector<float> get_parameter53() const { return m_Parameter53; }
        void set_parameter53(const std::vector<float>&& v) { m_Parameter53 = std::move(v); };
        const std::vector<float> get_parameter55() const { return m_Parameter55; }
        void set_parameter55(const std::vector<float>&& v) { m_Parameter55 = std::move(v); };
        const std::vector<float> get_constant81() const { return m_Constant81; }
        void set_constant81(const std::vector<float>&& v) { m_Constant81 = std::move(v); };
        const std::vector<float> get_constant62() const { return m_Constant62; }
        void set_constant62(const std::vector<float>&& v) { m_Constant62 = std::move(v); };
        const std::vector<float> get_constant61() const { return m_Constant61; }
        void set_constant61(const std::vector<float>&& v) { m_Constant61 = std::move(v); };
        Halide::Pipeline create_eval_graph(const Halide::ImageParam& features, const Halide::ImageParam& PastValue112, const Halide::ImageParam& PastValue198, const Halide::ImageParam& PastValue228, const Halide::ImageParam& PastValue314, const Halide::ImageParam& PastValue344, const Halide::ImageParam& PastValue430, const Halide::ImageParam& PastValue460, const Halide::ImageParam& PastValue82)
        {
            Halide::Var var1, var2;
            auto b_Constant555 = Halide::Buffer<float>(m_Constant555.data(), 9404, "Constant555");
            Halide::Func Constant555("Constant555"); Constant555(var1) = b_Constant555(var1);

            Halide::Func Log556("Log556"); Log556 = Log(Constant555);
            auto b_Parameter550 = Halide::Buffer<float>(m_Parameter550.data(), 9404, "Parameter550");
            Halide::Func Parameter550("Parameter550"); Parameter550(var1) = b_Parameter550(var1);

            auto b_Parameter6 = Halide::Buffer<float>(m_Parameter6.data(), 9404, 256, "Parameter6");
            Halide::Func Parameter6("Parameter6"); Parameter6(var1, var2) = b_Parameter6(var1, var2);

            auto b_Parameter8 = Halide::Buffer<float>(m_Parameter8.data(), 256, 512, "Parameter8");
            Halide::Func Parameter8("Parameter8"); Parameter8(var1, var2) = b_Parameter8(var1, var2);

            auto b_Parameter10 = Halide::Buffer<float>(m_Parameter10.data(), 512, 1024, "Parameter10");
            Halide::Func Parameter10("Parameter10"); Parameter10(var1, var2) = b_Parameter10(var1, var2);

            auto b_Parameter527 = Halide::Buffer<float>(m_Parameter527.data(), 1024, "Parameter527");
            Halide::Func Parameter527("Parameter527"); Parameter527(var1) = b_Parameter527(var1);

            auto b_Parameter508 = Halide::Buffer<float>(m_Parameter508.data(), 1024, "Parameter508");
            Halide::Func Parameter508("Parameter508"); Parameter508(var1) = b_Parameter508(var1);

            auto b_Parameter495 = Halide::Buffer<float>(m_Parameter495.data(), 1024, "Parameter495");
            Halide::Func Parameter495("Parameter495"); Parameter495(var1) = b_Parameter495(var1);

            auto b_Parameter488 = Halide::Buffer<float>(m_Parameter488.data(), 1024, "Parameter488");
            Halide::Func Parameter488("Parameter488"); Parameter488(var1) = b_Parameter488(var1);

            auto b_Parameter469 = Halide::Buffer<float>(m_Parameter469.data(), 1024, "Parameter469");
            Halide::Func Parameter469("Parameter469"); Parameter469(var1) = b_Parameter469(var1);

            auto b_Constant459 = Halide::Buffer<float>(m_Constant459.data(), 1, "Constant459");
            Halide::Func Constant459("Constant459"); Constant459(var1) = b_Constant459(var1);

            Halide::Func PastValue460_Constant459("PastValue460_Constant459"); PastValue460_Constant459(var1) = Halide::select(m_timestamp(0) < 1, Constant459(0), PastValue460(var1));;
            Halide::Func ElementTimes489("ElementTimes489"); ElementTimes489 = ElementTimes(Parameter488, PastValue460_Constant459);
            auto b_Parameter457 = Halide::Buffer<float>(m_Parameter457.data(), 1024, "Parameter457");
            Halide::Func Parameter457("Parameter457"); Parameter457(var1) = b_Parameter457(var1);

            Halide::Func ElementTimes463("ElementTimes463"); ElementTimes463 = ElementTimes(Parameter457, PastValue460_Constant459);
            auto b_Parameter446 = Halide::Buffer<float>(m_Parameter446.data(), 1024, "Parameter446");
            Halide::Func Parameter446("Parameter446"); Parameter446(var1) = b_Parameter446(var1);

            auto b_Parameter17 = Halide::Buffer<float>(m_Parameter17.data(), 4096, 448, "Parameter17");
            Halide::Func Parameter17("Parameter17"); Parameter17(var1, var2) = b_Parameter17(var1, var2);

            auto b_Parameter19 = Halide::Buffer<float>(m_Parameter19.data(), 448, 1024, "Parameter19");
            Halide::Func Parameter19("Parameter19"); Parameter19(var1, var2) = b_Parameter19(var1, var2);

            auto b_Constant429 = Halide::Buffer<float>(m_Constant429.data(), 1, "Constant429");
            Halide::Func Constant429("Constant429"); Constant429(var1) = b_Constant429(var1);

            Halide::Func PastValue430_Constant429("PastValue430_Constant429"); PastValue430_Constant429(var1) = Halide::select(m_timestamp(0) < 1, Constant429(0), PastValue430(var1));;
            auto b_Parameter22 = Halide::Buffer<float>(m_Parameter22.data(), 512, 1024, "Parameter22");
            Halide::Func Parameter22("Parameter22"); Parameter22(var1, var2) = b_Parameter22(var1, var2);

            auto b_Parameter411 = Halide::Buffer<float>(m_Parameter411.data(), 1024, "Parameter411");
            Halide::Func Parameter411("Parameter411"); Parameter411(var1) = b_Parameter411(var1);

            auto b_Parameter392 = Halide::Buffer<float>(m_Parameter392.data(), 1024, "Parameter392");
            Halide::Func Parameter392("Parameter392"); Parameter392(var1) = b_Parameter392(var1);

            auto b_Parameter379 = Halide::Buffer<float>(m_Parameter379.data(), 1024, "Parameter379");
            Halide::Func Parameter379("Parameter379"); Parameter379(var1) = b_Parameter379(var1);

            auto b_Parameter372 = Halide::Buffer<float>(m_Parameter372.data(), 1024, "Parameter372");
            Halide::Func Parameter372("Parameter372"); Parameter372(var1) = b_Parameter372(var1);

            auto b_Parameter353 = Halide::Buffer<float>(m_Parameter353.data(), 1024, "Parameter353");
            Halide::Func Parameter353("Parameter353"); Parameter353(var1) = b_Parameter353(var1);

            auto b_Constant343 = Halide::Buffer<float>(m_Constant343.data(), 1, "Constant343");
            Halide::Func Constant343("Constant343"); Constant343(var1) = b_Constant343(var1);

            Halide::Func PastValue344_Constant343("PastValue344_Constant343"); PastValue344_Constant343(var1) = Halide::select(m_timestamp(0) < 1, Constant343(0), PastValue344(var1));;
            Halide::Func ElementTimes373("ElementTimes373"); ElementTimes373 = ElementTimes(Parameter372, PastValue344_Constant343);
            auto b_Parameter341 = Halide::Buffer<float>(m_Parameter341.data(), 1024, "Parameter341");
            Halide::Func Parameter341("Parameter341"); Parameter341(var1) = b_Parameter341(var1);

            Halide::Func ElementTimes347("ElementTimes347"); ElementTimes347 = ElementTimes(Parameter341, PastValue344_Constant343);
            auto b_Parameter330 = Halide::Buffer<float>(m_Parameter330.data(), 1024, "Parameter330");
            Halide::Func Parameter330("Parameter330"); Parameter330(var1) = b_Parameter330(var1);

            auto b_Parameter29 = Halide::Buffer<float>(m_Parameter29.data(), 4096, 384, "Parameter29");
            Halide::Func Parameter29("Parameter29"); Parameter29(var1, var2) = b_Parameter29(var1, var2);

            auto b_Parameter31 = Halide::Buffer<float>(m_Parameter31.data(), 384, 1024, "Parameter31");
            Halide::Func Parameter31("Parameter31"); Parameter31(var1, var2) = b_Parameter31(var1, var2);

            auto b_Constant313 = Halide::Buffer<float>(m_Constant313.data(), 1, "Constant313");
            Halide::Func Constant313("Constant313"); Constant313(var1) = b_Constant313(var1);

            Halide::Func PastValue314_Constant313("PastValue314_Constant313"); PastValue314_Constant313(var1) = Halide::select(m_timestamp(0) < 1, Constant313(0), PastValue314(var1));;
            auto b_Parameter34 = Halide::Buffer<float>(m_Parameter34.data(), 512, 1024, "Parameter34");
            Halide::Func Parameter34("Parameter34"); Parameter34(var1, var2) = b_Parameter34(var1, var2);

            auto b_Parameter295 = Halide::Buffer<float>(m_Parameter295.data(), 1024, "Parameter295");
            Halide::Func Parameter295("Parameter295"); Parameter295(var1) = b_Parameter295(var1);

            auto b_Parameter276 = Halide::Buffer<float>(m_Parameter276.data(), 1024, "Parameter276");
            Halide::Func Parameter276("Parameter276"); Parameter276(var1) = b_Parameter276(var1);

            auto b_Parameter263 = Halide::Buffer<float>(m_Parameter263.data(), 1024, "Parameter263");
            Halide::Func Parameter263("Parameter263"); Parameter263(var1) = b_Parameter263(var1);

            auto b_Parameter256 = Halide::Buffer<float>(m_Parameter256.data(), 1024, "Parameter256");
            Halide::Func Parameter256("Parameter256"); Parameter256(var1) = b_Parameter256(var1);

            auto b_Parameter237 = Halide::Buffer<float>(m_Parameter237.data(), 1024, "Parameter237");
            Halide::Func Parameter237("Parameter237"); Parameter237(var1) = b_Parameter237(var1);

            auto b_Constant227 = Halide::Buffer<float>(m_Constant227.data(), 1, "Constant227");
            Halide::Func Constant227("Constant227"); Constant227(var1) = b_Constant227(var1);

            Halide::Func PastValue228_Constant227("PastValue228_Constant227"); PastValue228_Constant227(var1) = Halide::select(m_timestamp(0) < 1, Constant227(0), PastValue228(var1));;
            Halide::Func ElementTimes257("ElementTimes257"); ElementTimes257 = ElementTimes(Parameter256, PastValue228_Constant227);
            auto b_Parameter225 = Halide::Buffer<float>(m_Parameter225.data(), 1024, "Parameter225");
            Halide::Func Parameter225("Parameter225"); Parameter225(var1) = b_Parameter225(var1);

            Halide::Func ElementTimes231("ElementTimes231"); ElementTimes231 = ElementTimes(Parameter225, PastValue228_Constant227);
            auto b_Parameter214 = Halide::Buffer<float>(m_Parameter214.data(), 1024, "Parameter214");
            Halide::Func Parameter214("Parameter214"); Parameter214(var1) = b_Parameter214(var1);

            auto b_Parameter41 = Halide::Buffer<float>(m_Parameter41.data(), 4096, 384, "Parameter41");
            Halide::Func Parameter41("Parameter41"); Parameter41(var1, var2) = b_Parameter41(var1, var2);

            auto b_Parameter43 = Halide::Buffer<float>(m_Parameter43.data(), 384, 1024, "Parameter43");
            Halide::Func Parameter43("Parameter43"); Parameter43(var1, var2) = b_Parameter43(var1, var2);

            auto b_Constant197 = Halide::Buffer<float>(m_Constant197.data(), 1, "Constant197");
            Halide::Func Constant197("Constant197"); Constant197(var1) = b_Constant197(var1);

            Halide::Func PastValue198_Constant197("PastValue198_Constant197"); PastValue198_Constant197(var1) = Halide::select(m_timestamp(0) < 1, Constant197(0), PastValue198(var1));;
            auto b_Parameter46 = Halide::Buffer<float>(m_Parameter46.data(), 512, 1024, "Parameter46");
            Halide::Func Parameter46("Parameter46"); Parameter46(var1, var2) = b_Parameter46(var1, var2);

            auto b_Parameter179 = Halide::Buffer<float>(m_Parameter179.data(), 1024, "Parameter179");
            Halide::Func Parameter179("Parameter179"); Parameter179(var1) = b_Parameter179(var1);

            auto b_Parameter160 = Halide::Buffer<float>(m_Parameter160.data(), 1024, "Parameter160");
            Halide::Func Parameter160("Parameter160"); Parameter160(var1) = b_Parameter160(var1);

            auto b_Parameter147 = Halide::Buffer<float>(m_Parameter147.data(), 1024, "Parameter147");
            Halide::Func Parameter147("Parameter147"); Parameter147(var1) = b_Parameter147(var1);

            auto b_Parameter140 = Halide::Buffer<float>(m_Parameter140.data(), 1024, "Parameter140");
            Halide::Func Parameter140("Parameter140"); Parameter140(var1) = b_Parameter140(var1);

            auto b_Parameter121 = Halide::Buffer<float>(m_Parameter121.data(), 1024, "Parameter121");
            Halide::Func Parameter121("Parameter121"); Parameter121(var1) = b_Parameter121(var1);

            auto b_Constant111 = Halide::Buffer<float>(m_Constant111.data(), 1, "Constant111");
            Halide::Func Constant111("Constant111"); Constant111(var1) = b_Constant111(var1);

            Halide::Func PastValue112_Constant111("PastValue112_Constant111"); PastValue112_Constant111(var1) = Halide::select(m_timestamp(0) < 1, Constant111(0), PastValue112(var1));;
            Halide::Func ElementTimes141("ElementTimes141"); ElementTimes141 = ElementTimes(Parameter140, PastValue112_Constant111);
            auto b_Parameter109 = Halide::Buffer<float>(m_Parameter109.data(), 1024, "Parameter109");
            Halide::Func Parameter109("Parameter109"); Parameter109(var1) = b_Parameter109(var1);

            Halide::Func ElementTimes115("ElementTimes115"); ElementTimes115 = ElementTimes(Parameter109, PastValue112_Constant111);
            auto b_Parameter98 = Halide::Buffer<float>(m_Parameter98.data(), 1024, "Parameter98");
            Halide::Func Parameter98("Parameter98"); Parameter98(var1) = b_Parameter98(var1);

            auto b_Parameter53 = Halide::Buffer<float>(m_Parameter53.data(), 4096, 192, "Parameter53");
            Halide::Func Parameter53("Parameter53"); Parameter53(var1, var2) = b_Parameter53(var1, var2);

            auto b_Parameter55 = Halide::Buffer<float>(m_Parameter55.data(), 192, 592, "Parameter55");
            Halide::Func Parameter55("Parameter55"); Parameter55(var1, var2) = b_Parameter55(var1, var2);

            auto b_Constant81 = Halide::Buffer<float>(m_Constant81.data(), 1, "Constant81");
            Halide::Func Constant81("Constant81"); Constant81(var1) = b_Constant81(var1);

            Halide::Func PastValue82_Constant81("PastValue82_Constant81"); PastValue82_Constant81(var1) = Halide::select(m_timestamp(0) < 1, Constant81(0), PastValue82(var1));;
            auto b_Constant62 = Halide::Buffer<float>(m_Constant62.data(), 80, "Constant62");
            Halide::Func Constant62("Constant62"); Constant62(var1) = b_Constant62(var1);

            auto b_Constant61 = Halide::Buffer<float>(m_Constant61.data(), 80, "Constant61");
            Halide::Func Constant61("Constant61"); Constant61(var1) = b_Constant61(var1);

            Halide::Func Minus66("Minus66"); Minus66 = Minus(features, Constant61);
            Halide::Func ElementTimes69("ElementTimes69"); ElementTimes69 = ElementTimes(Minus66, Constant62);
            Halide::Func Splice85("Splice85"); Splice85 = Splice(ElementTimes69, PastValue82_Constant81, 80, 512);
            Halide::Func Times88("Times88"); Times88 = VectorByMatrixTimes(Splice85, Parameter55, 592, 192);
            Halide::Func Times91("Times91"); Times91 = VectorByMatrixTimes(Times88, Parameter53, 192, 4096);
            Times91.compute_root();
            Halide::Func Slice105("Slice105"); Slice105 = Slice(Times91, 1024, 2048);
            Halide::Func Plus118("Plus118"); Plus118 = Plus(Slice105, ElementTimes115);
            Halide::Func Plus122("Plus122"); Plus122 = Plus(Plus118, Parameter121);
            Halide::Func Sigmoid125("Sigmoid125"); Sigmoid125 = Sigmoid<float>(Plus122);
            Halide::Func ElementTimes128("ElementTimes128"); ElementTimes128 = ElementTimes(Sigmoid125, PastValue112_Constant111);
            Halide::Func Slice136("Slice136"); Slice136 = Slice(Times91, 0, 1024);
            Halide::Func Plus144("Plus144"); Plus144 = Plus(Slice136, ElementTimes141);
            Halide::Func Plus148("Plus148"); Plus148 = Plus(Plus144, Parameter147);
            Halide::Func Sigmoid151("Sigmoid151"); Sigmoid151 = Sigmoid<float>(Plus148);
            Halide::Func Slice157("Slice157"); Slice157 = Slice(Times91, 2048, 3072);
            Halide::Func Plus161("Plus161"); Plus161 = Plus(Slice157, Parameter160);
            Halide::Func Tanh164("Tanh164"); Tanh164 = Tanh(Plus161);
            Halide::Func ElementTimes167("ElementTimes167"); ElementTimes167 = ElementTimes(Sigmoid151, Tanh164);
            Halide::Func Plus170("Plus170"); Plus170 = Plus(ElementTimes128, ElementTimes167);
            Plus170.compute_root();
            Halide::Func ElementTimes173("ElementTimes173"); ElementTimes173 = ElementTimes(Parameter98, Plus170);
            Halide::Func Tanh187("Tanh187"); Tanh187 = Tanh(Plus170);
            Halide::Func Slice94("Slice94"); Slice94 = Slice(Times91, 3072, 4096);
            Halide::Func Plus176("Plus176"); Plus176 = Plus(Slice94, ElementTimes173);
            Halide::Func Plus180("Plus180"); Plus180 = Plus(Plus176, Parameter179);
            Halide::Func Sigmoid183("Sigmoid183"); Sigmoid183 = Sigmoid<float>(Plus180);
            Halide::Func ElementTimes190("ElementTimes190"); ElementTimes190 = ElementTimes(Sigmoid183, Tanh187);
            Halide::Func Times193("Times193"); Times193 = VectorByMatrixTimes(ElementTimes190, Parameter46, 1024, 512);
            Times193.compute_root();
            Halide::Func Splice201("Splice201"); Splice201 = Splice(Times193, PastValue198_Constant197, 512, 512);
            Halide::Func Times204("Times204"); Times204 = VectorByMatrixTimes(Splice201, Parameter43, 1024, 384);
            Halide::Func Times207("Times207"); Times207 = VectorByMatrixTimes(Times204, Parameter41, 384, 4096);
            Times207.compute_root();
            Halide::Func Slice221("Slice221"); Slice221 = Slice(Times207, 1024, 2048);
            Halide::Func Plus234("Plus234"); Plus234 = Plus(Slice221, ElementTimes231);
            Halide::Func Plus238("Plus238"); Plus238 = Plus(Plus234, Parameter237);
            Halide::Func Sigmoid241("Sigmoid241"); Sigmoid241 = Sigmoid<float>(Plus238);
            Halide::Func ElementTimes244("ElementTimes244"); ElementTimes244 = ElementTimes(Sigmoid241, PastValue228_Constant227);
            Halide::Func Slice252("Slice252"); Slice252 = Slice(Times207, 0, 1024);
            Halide::Func Plus260("Plus260"); Plus260 = Plus(Slice252, ElementTimes257);
            Halide::Func Plus264("Plus264"); Plus264 = Plus(Plus260, Parameter263);
            Halide::Func Sigmoid267("Sigmoid267"); Sigmoid267 = Sigmoid<float>(Plus264);
            Halide::Func Slice273("Slice273"); Slice273 = Slice(Times207, 2048, 3072);
            Halide::Func Plus277("Plus277"); Plus277 = Plus(Slice273, Parameter276);
            Halide::Func Tanh280("Tanh280"); Tanh280 = Tanh(Plus277);
            Halide::Func ElementTimes283("ElementTimes283"); ElementTimes283 = ElementTimes(Sigmoid267, Tanh280);
            Halide::Func Plus286("Plus286"); Plus286 = Plus(ElementTimes244, ElementTimes283);
            Plus286.compute_root();
            Halide::Func ElementTimes289("ElementTimes289"); ElementTimes289 = ElementTimes(Parameter214, Plus286);
            Halide::Func Tanh303("Tanh303"); Tanh303 = Tanh(Plus286);
            Halide::Func Slice210("Slice210"); Slice210 = Slice(Times207, 3072, 4096);
            Halide::Func Plus292("Plus292"); Plus292 = Plus(Slice210, ElementTimes289);
            Halide::Func Plus296("Plus296"); Plus296 = Plus(Plus292, Parameter295);
            Halide::Func Sigmoid299("Sigmoid299"); Sigmoid299 = Sigmoid<float>(Plus296);
            Halide::Func ElementTimes306("ElementTimes306"); ElementTimes306 = ElementTimes(Sigmoid299, Tanh303);
            Halide::Func Times309("Times309"); Times309 = VectorByMatrixTimes(ElementTimes306, Parameter34, 1024, 512);
            Times309.compute_root();
            Halide::Func Splice317("Splice317"); Splice317 = Splice(Times309, PastValue314_Constant313, 512, 512);
            Halide::Func Times320("Times320"); Times320 = VectorByMatrixTimes(Splice317, Parameter31, 1024, 384);
            Halide::Func Times323("Times323"); Times323 = VectorByMatrixTimes(Times320, Parameter29, 384, 4096);
            Times323.compute_root();
            Halide::Func Slice337("Slice337"); Slice337 = Slice(Times323, 1024, 2048);
            Halide::Func Plus350("Plus350"); Plus350 = Plus(Slice337, ElementTimes347);
            Halide::Func Plus354("Plus354"); Plus354 = Plus(Plus350, Parameter353);
            Halide::Func Sigmoid357("Sigmoid357"); Sigmoid357 = Sigmoid<float>(Plus354);
            Halide::Func ElementTimes360("ElementTimes360"); ElementTimes360 = ElementTimes(Sigmoid357, PastValue344_Constant343);
            Halide::Func Slice368("Slice368"); Slice368 = Slice(Times323, 0, 1024);
            Halide::Func Plus376("Plus376"); Plus376 = Plus(Slice368, ElementTimes373);
            Halide::Func Plus380("Plus380"); Plus380 = Plus(Plus376, Parameter379);
            Halide::Func Sigmoid383("Sigmoid383"); Sigmoid383 = Sigmoid<float>(Plus380);
            Halide::Func Slice389("Slice389"); Slice389 = Slice(Times323, 2048, 3072);
            Halide::Func Plus393("Plus393"); Plus393 = Plus(Slice389, Parameter392);
            Halide::Func Tanh396("Tanh396"); Tanh396 = Tanh(Plus393);
            Halide::Func ElementTimes399("ElementTimes399"); ElementTimes399 = ElementTimes(Sigmoid383, Tanh396);
            Halide::Func Plus402("Plus402"); Plus402 = Plus(ElementTimes360, ElementTimes399);
            Plus402.compute_root();
            Halide::Func ElementTimes405("ElementTimes405"); ElementTimes405 = ElementTimes(Parameter330, Plus402);
            Halide::Func Tanh419("Tanh419"); Tanh419 = Tanh(Plus402);
            Halide::Func Slice326("Slice326"); Slice326 = Slice(Times323, 3072, 4096);
            Halide::Func Plus408("Plus408"); Plus408 = Plus(Slice326, ElementTimes405);
            Halide::Func Plus412("Plus412"); Plus412 = Plus(Plus408, Parameter411);
            Halide::Func Sigmoid415("Sigmoid415"); Sigmoid415 = Sigmoid<float>(Plus412);
            Halide::Func ElementTimes422("ElementTimes422"); ElementTimes422 = ElementTimes(Sigmoid415, Tanh419);
            Halide::Func Times425("Times425"); Times425 = VectorByMatrixTimes(ElementTimes422, Parameter22, 1024, 512);
            Times425.compute_root();
            Halide::Func Splice433("Splice433"); Splice433 = Splice(Times425, PastValue430_Constant429, 512, 512);
            Halide::Func Times436("Times436"); Times436 = VectorByMatrixTimes(Splice433, Parameter19, 1024, 448);
            Halide::Func Times439("Times439"); Times439 = VectorByMatrixTimes(Times436, Parameter17, 448, 4096);
            Times439.compute_root();
            Halide::Func Slice453("Slice453"); Slice453 = Slice(Times439, 1024, 2048);
            Halide::Func Plus466("Plus466"); Plus466 = Plus(Slice453, ElementTimes463);
            Halide::Func Plus470("Plus470"); Plus470 = Plus(Plus466, Parameter469);
            Halide::Func Sigmoid473("Sigmoid473"); Sigmoid473 = Sigmoid<float>(Plus470);
            Halide::Func ElementTimes476("ElementTimes476"); ElementTimes476 = ElementTimes(Sigmoid473, PastValue460_Constant459);
            Halide::Func Slice484("Slice484"); Slice484 = Slice(Times439, 0, 1024);
            Halide::Func Plus492("Plus492"); Plus492 = Plus(Slice484, ElementTimes489);
            Halide::Func Plus496("Plus496"); Plus496 = Plus(Plus492, Parameter495);
            Halide::Func Sigmoid499("Sigmoid499"); Sigmoid499 = Sigmoid<float>(Plus496);
            Halide::Func Slice505("Slice505"); Slice505 = Slice(Times439, 2048, 3072);
            Halide::Func Plus509("Plus509"); Plus509 = Plus(Slice505, Parameter508);
            Halide::Func Tanh512("Tanh512"); Tanh512 = Tanh(Plus509);
            Halide::Func ElementTimes515("ElementTimes515"); ElementTimes515 = ElementTimes(Sigmoid499, Tanh512);
            Halide::Func Plus518("Plus518"); Plus518 = Plus(ElementTimes476, ElementTimes515);
            Plus518.compute_root();
            Halide::Func ElementTimes521("ElementTimes521"); ElementTimes521 = ElementTimes(Parameter446, Plus518);
            Halide::Func Tanh535("Tanh535"); Tanh535 = Tanh(Plus518);
            Halide::Func Slice442("Slice442"); Slice442 = Slice(Times439, 3072, 4096);
            Halide::Func Plus524("Plus524"); Plus524 = Plus(Slice442, ElementTimes521);
            Halide::Func Plus528("Plus528"); Plus528 = Plus(Plus524, Parameter527);
            Halide::Func Sigmoid531("Sigmoid531"); Sigmoid531 = Sigmoid<float>(Plus528);
            Halide::Func ElementTimes538("ElementTimes538"); ElementTimes538 = ElementTimes(Sigmoid531, Tanh535);
            Halide::Func Times541("Times541"); Times541 = VectorByMatrixTimes(ElementTimes538, Parameter10, 1024, 512);
            Times541.compute_root();
            Halide::Func Times544("Times544"); Times544 = VectorByMatrixTimes(Times541, Parameter8, 512, 256);
            Halide::Func Times547("Times547"); Times547 = VectorByMatrixTimes(Times544, Parameter6, 256, 9404);
            Halide::Func Plus551("Plus551"); Plus551 = Plus(Times547, Parameter550);
            Halide::Func Minus559("Minus559"); Minus559 = Minus(Plus551, Log556);

            return Halide::Pipeline({ Minus559/*ScaledLogLikelihood*/, Plus170/*PastValue112*/, Times309/*PastValue198*/, Plus286/*PastValue228*/, Times425/*PastValue314*/, Plus402/*PastValue344*/, Times541/*PastValue430*/, Plus518/*PastValue460*/, Times193/*PastValue82*/ });
        }


    public:
        void init(const std::string& weightFilePath)
        {
            boost::property_tree::ptree root;
            boost::property_tree::read_json(weightFilePath.c_str(), root);

            auto get_value = [&](const std::string& name)
            {
                std::vector<float> result;
                for (auto& v : root.get_child(name))
                    result.push_back(v.second.get_value<float>());
                return result;
            };
            set_constant555(get_value("Constant555"));
            set_parameter550(get_value("Parameter550"));
            set_parameter6(get_value("Parameter6"));
            set_parameter8(get_value("Parameter8"));
            set_parameter10(get_value("Parameter10"));
            set_parameter527(get_value("Parameter527"));
            set_parameter508(get_value("Parameter508"));
            set_parameter495(get_value("Parameter495"));
            set_parameter488(get_value("Parameter488"));
            set_parameter469(get_value("Parameter469"));
            set_constant459(get_value("Constant459"));
            set_parameter457(get_value("Parameter457"));
            set_parameter446(get_value("Parameter446"));
            set_parameter17(get_value("Parameter17"));
            set_parameter19(get_value("Parameter19"));
            set_constant429(get_value("Constant429"));
            set_parameter22(get_value("Parameter22"));
            set_parameter411(get_value("Parameter411"));
            set_parameter392(get_value("Parameter392"));
            set_parameter379(get_value("Parameter379"));
            set_parameter372(get_value("Parameter372"));
            set_parameter353(get_value("Parameter353"));
            set_constant343(get_value("Constant343"));
            set_parameter341(get_value("Parameter341"));
            set_parameter330(get_value("Parameter330"));
            set_parameter29(get_value("Parameter29"));
            set_parameter31(get_value("Parameter31"));
            set_constant313(get_value("Constant313"));
            set_parameter34(get_value("Parameter34"));
            set_parameter295(get_value("Parameter295"));
            set_parameter276(get_value("Parameter276"));
            set_parameter263(get_value("Parameter263"));
            set_parameter256(get_value("Parameter256"));
            set_parameter237(get_value("Parameter237"));
            set_constant227(get_value("Constant227"));
            set_parameter225(get_value("Parameter225"));
            set_parameter214(get_value("Parameter214"));
            set_parameter41(get_value("Parameter41"));
            set_parameter43(get_value("Parameter43"));
            set_constant197(get_value("Constant197"));
            set_parameter46(get_value("Parameter46"));
            set_parameter179(get_value("Parameter179"));
            set_parameter160(get_value("Parameter160"));
            set_parameter147(get_value("Parameter147"));
            set_parameter140(get_value("Parameter140"));
            set_parameter121(get_value("Parameter121"));
            set_constant111(get_value("Constant111"));
            set_parameter109(get_value("Parameter109"));
            set_parameter98(get_value("Parameter98"));
            set_parameter53(get_value("Parameter53"));
            set_parameter55(get_value("Parameter55"));
            set_constant81(get_value("Constant81"));
            set_constant62(get_value("Constant62"));
            set_constant61(get_value("Constant61"));
            m_bufferPastValue460.resize(2, Halide::Buffer<float>(1024));
            m_bufferPastValue430.resize(2, Halide::Buffer<float>(512));
            m_bufferPastValue344.resize(2, Halide::Buffer<float>(1024));
            m_bufferPastValue314.resize(2, Halide::Buffer<float>(512));
            m_bufferPastValue228.resize(2, Halide::Buffer<float>(1024));
            m_bufferPastValue198.resize(2, Halide::Buffer<float>(512));
            m_bufferPastValue112.resize(2, Halide::Buffer<float>(1024));
            m_bufferPastValue82.resize(2, Halide::Buffer<float>(512));
        }

        void Evaluate(int timestamp, const Halide::ImageParam& features, Halide::Buffer<float>& ScaledLogLikelihood)
        {
            if (!m_graphInitialized)
            {
                m_timestamp.set(m_bufferTimestamp);
                m_graph = create_eval_graph(features, m_PastValue112, m_PastValue198, m_PastValue228, m_PastValue314, m_PastValue344, m_PastValue430, m_PastValue460, m_PastValue82);
                m_graphInitialized = true;
            }
            m_PastValue112.set(m_bufferPastValue112[((timestamp - 1) % m_bufferPastValue112.size())]);
            m_PastValue198.set(m_bufferPastValue198[((timestamp - 1) % m_bufferPastValue198.size())]);
            m_PastValue228.set(m_bufferPastValue228[((timestamp - 1) % m_bufferPastValue228.size())]);
            m_PastValue314.set(m_bufferPastValue314[((timestamp - 1) % m_bufferPastValue314.size())]);
            m_PastValue344.set(m_bufferPastValue344[((timestamp - 1) % m_bufferPastValue344.size())]);
            m_PastValue430.set(m_bufferPastValue430[((timestamp - 1) % m_bufferPastValue430.size())]);
            m_PastValue460.set(m_bufferPastValue460[((timestamp - 1) % m_bufferPastValue460.size())]);
            m_PastValue82.set(m_bufferPastValue82[((timestamp - 1) % m_bufferPastValue82.size())]);
            m_bufferTimestamp(0) = timestamp;
            m_graph.realize({ ScaledLogLikelihood, m_bufferPastValue112[timestamp % m_bufferPastValue112.size()], m_bufferPastValue198[timestamp % m_bufferPastValue198.size()], m_bufferPastValue228[timestamp % m_bufferPastValue228.size()], m_bufferPastValue314[timestamp % m_bufferPastValue314.size()], m_bufferPastValue344[timestamp % m_bufferPastValue344.size()], m_bufferPastValue430[timestamp % m_bufferPastValue430.size()], m_bufferPastValue460[timestamp % m_bufferPastValue460.size()], m_bufferPastValue82[timestamp % m_bufferPastValue82.size()] });
        }

    private:
        std::vector<float> m_Constant555;
        std::vector<float> m_Parameter550;
        std::vector<float> m_Parameter6;
        std::vector<float> m_Parameter8;
        std::vector<float> m_Parameter10;
        std::vector<float> m_Parameter527;
        std::vector<float> m_Parameter508;
        std::vector<float> m_Parameter495;
        std::vector<float> m_Parameter488;
        std::vector<float> m_Parameter469;
        std::vector<float> m_Constant459;
        std::vector<float> m_Parameter457;
        std::vector<float> m_Parameter446;
        std::vector<float> m_Parameter17;
        std::vector<float> m_Parameter19;
        std::vector<float> m_Constant429;
        std::vector<float> m_Parameter22;
        std::vector<float> m_Parameter411;
        std::vector<float> m_Parameter392;
        std::vector<float> m_Parameter379;
        std::vector<float> m_Parameter372;
        std::vector<float> m_Parameter353;
        std::vector<float> m_Constant343;
        std::vector<float> m_Parameter341;
        std::vector<float> m_Parameter330;
        std::vector<float> m_Parameter29;
        std::vector<float> m_Parameter31;
        std::vector<float> m_Constant313;
        std::vector<float> m_Parameter34;
        std::vector<float> m_Parameter295;
        std::vector<float> m_Parameter276;
        std::vector<float> m_Parameter263;
        std::vector<float> m_Parameter256;
        std::vector<float> m_Parameter237;
        std::vector<float> m_Constant227;
        std::vector<float> m_Parameter225;
        std::vector<float> m_Parameter214;
        std::vector<float> m_Parameter41;
        std::vector<float> m_Parameter43;
        std::vector<float> m_Constant197;
        std::vector<float> m_Parameter46;
        std::vector<float> m_Parameter179;
        std::vector<float> m_Parameter160;
        std::vector<float> m_Parameter147;
        std::vector<float> m_Parameter140;
        std::vector<float> m_Parameter121;
        std::vector<float> m_Constant111;
        std::vector<float> m_Parameter109;
        std::vector<float> m_Parameter98;
        std::vector<float> m_Parameter53;
        std::vector<float> m_Parameter55;
        std::vector<float> m_Constant81;
        std::vector<float> m_Constant62;
        std::vector<float> m_Constant61;
        Halide::Pipeline m_graph;
        bool m_graphInitialized{ false };
        Halide::Buffer<int> m_bufferTimestamp{ 1 };
        Halide::ImageParam m_timestamp{ Halide::type_of<int>(), 1 };
        std::vector<Halide::Buffer<float>> m_bufferPastValue112;
        Halide::ImageParam m_PastValue112{ Halide::type_of<float>(), 1 };
        std::vector<Halide::Buffer<float>> m_bufferPastValue198;
        Halide::ImageParam m_PastValue198{ Halide::type_of<float>(), 1 };
        std::vector<Halide::Buffer<float>> m_bufferPastValue228;
        Halide::ImageParam m_PastValue228{ Halide::type_of<float>(), 1 };
        std::vector<Halide::Buffer<float>> m_bufferPastValue314;
        Halide::ImageParam m_PastValue314{ Halide::type_of<float>(), 1 };
        std::vector<Halide::Buffer<float>> m_bufferPastValue344;
        Halide::ImageParam m_PastValue344{ Halide::type_of<float>(), 1 };
        std::vector<Halide::Buffer<float>> m_bufferPastValue430;
        Halide::ImageParam m_PastValue430{ Halide::type_of<float>(), 1 };
        std::vector<Halide::Buffer<float>> m_bufferPastValue460;
        Halide::ImageParam m_PastValue460{ Halide::type_of<float>(), 1 };
        std::vector<Halide::Buffer<float>> m_bufferPastValue82;
        Halide::ImageParam m_PastValue82{ Halide::type_of<float>(), 1 };
    };
};