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
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

namespace Test
{

auto polyfit( const std::vector<double> c, const double x )
{
    double result = 0.0;
    for ( double i = 0; i < c.size(); ++i )
    {
        result = result * x + c[i];
    }
    return result;
}

TEST( TEST_CATEGORY, ElasticWave )
{
    // Remove any existing file first.
    std::string outname = "displacement_profile.txt";
    std::filesystem::remove( outname );

    // Run with smaller system than example/ for regression testing.
    auto sysreturn =
        std::system( "../examples/mechanics/ElasticWave elastic_wave.json" );
    EXPECT_EQ( sysreturn, 0 );

    // Coefficients in descending order.
    std::vector<double> coeff{
        1.61854683e+03, 9.92564555e+07, -2.04437979e+03, -1.25377266e+08,
        1.11872348e+03, 6.80635175e+07, -3.47671366e+02, -2.07030221e+07,
        6.75838697e+01, 3.85014862e+06, -8.53079823e+00, -4.46939102e+05,
        7.02713121e-01, 3.15089992e+04, -3.69859638e-02, -1.22783300e+03,
        1.18185545e-03, 2.00620375e+01, -2.07937472e-05, 4.57208084e-03,
        1.66083591e-07, 1.32149495e-03, -3.73328058e-10 };

    // This test compares the final state with an externally fit polynomial
    // using a higher resolution (81 points per dim). The order was chosen to
    // represent the curve well, except the boundary points. This is extremely
    // specific to the initial state and the final time.
    std::ifstream outfile( outname );
    std::string line;
    double total_err = 0.0;
    while ( std::getline( outfile, line ) )
    {
        if ( !line.empty() )
        {
            // Check the displacement profile in x.
            std::istringstream ss( line );
            std::vector<double> values( ( std::istream_iterator<double>( ss ) ),
                                        std::istream_iterator<double>() );

            const double x = values[1];
            // Polynomial fit is bad near the ends and near zero there are large
            // relative errors.
            if ( x > 0.425 || x < -0.425 || ( x < 0.025 && x > -0.025 ) )
                continue;

            const double u = values[2];
            const double u_fit = polyfit( coeff, x );
            const double diff_err = ( u - u_fit );
            const double rel_err = std::abs( diff_err / u_fit );
            // This is a relatively low accuracy check because the polynomial
            // was fit with a higher resolution run.
            EXPECT_LT( rel_err, 0.035 );
            total_err += diff_err * diff_err;
        }
    }
    outfile.close();

    EXPECT_LT( std::sqrt( total_err ), 3e-5 );
}

} // end namespace Test
