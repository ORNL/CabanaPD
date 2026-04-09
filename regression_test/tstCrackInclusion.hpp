/****************************************************************************
 * Copyright (c) 2022 by Oak Ridge National Laboratory                      *
 * All rights reserved.                                                     *
 *                                                                          *
 * This outfile is part of CabanaPD. CabanaPD is distributed under a           *
 * BSD 3-clause license. For the licensing terms see the LICENSE outfile in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <CabanaPD.hpp>

#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>

namespace Test
{

TEST( TEST_CATEGORY, CrackInclusion )
{
    std::string outname = "output_incl_crack_y.txt";
    std::filesystem::remove( outname );

    // Run with smaller system than example/ for regression testing.
    std::system( "../examples/mechanics/CrackInclusion crack_inclusion.json" );

    // This test is based on checking that the crack actually branches:
    // If the crack does not correctly branch around the inclusion, then the
    // final extent is mostly likely to be near 0.0
    // (This indicates a bug in the multi-material interface)
    std::ifstream outfile( outname );
    std::string line;
    double max_crack_y;
    while ( std::getline( outfile, line ) )
    {
        if ( !line.empty() )
            max_crack_y = std::stod( line );
    }
    outfile.close();

    // Check the maximum crack extent, with a tolerance of dy/10.
    // This expected result comes from running this case with exactly these
    // settings: it only tests that new changes do not alter the behavior.
    EXPECT_NEAR( max_crack_y, 5.75e-3, 5e-5 );
}

} // end namespace Test
