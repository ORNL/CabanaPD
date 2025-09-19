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

#include <CabanaPD.hpp>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

namespace Test
{

// Test construction.
TEST( TEST_CATEGORY, test_force_normal_construct )
{
    double delta = 10.0;
    double radius = 5.0;
    double extend = 1.0;
    double K = 2.0;
    CabanaPD::NormalRepulsionModel contact_model( delta, radius, extend, K );
}

} // end namespace Test
