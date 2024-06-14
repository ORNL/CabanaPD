/****************************************************************************
 * Copyright (c) 2022 by Oak Ridge National Laboratory                      *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of CabanaPD. CabanaPD is distributed under a           *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

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
