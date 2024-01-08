#ifndef FIELDS_HPP
#define FIELDS_HPP

#include <Cabana_Core.hpp>

namespace CabanaPD
{
//---------------------------------------------------------------------------//
// Fields.
//---------------------------------------------------------------------------//
namespace Field
{

struct ReferencePosition : Cabana::Field::Position<3>
{
    static std::string label() { return "reference_positions"; }
};

struct Force : Cabana::Field::Vector<double, 3>
{
    static std::string label() { return "forces"; }
};

struct NoFail : Cabana::Field::Scalar<int>
{
    static std::string label() { return "no_fail"; }
};

} // namespace Field
} // namespace CabanaPD

#endif
