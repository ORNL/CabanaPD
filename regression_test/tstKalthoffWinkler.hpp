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

#include <cstdlib>
#include <filesystem>
#include <fstream>

namespace Test
{

TEST( TEST_CATEGORY, KalthoffWinkler )
{
    // Remove any existing file first.
    std::string outname = "output_kw_crack_y.txt";
    std::filesystem::remove( outname );

    // Run with smaller system than example/ for regression testing.
    auto sysreturn = std::system(
        "../examples/mechanics/KalthoffWinkler kalthoff_winkler.json" );
    EXPECT_EQ( sysreturn, 0 );

    // Only need to check on rank zero.
    if ( !CabanaPD::print_rank() )
        return;

    // This test is based on checking that the crack grows at the correct angle.
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
    EXPECT_NEAR( max_crack_y, 7.42e-2, 1.1e-4 );
}

} // end namespace Test
