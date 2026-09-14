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

template <typename ModelType>
void test_cube_heated_region( ModelType )
{
    // ====================================================
    //            Kokkos spaces
    // ====================================================
    using exec_space = TEST_EXECSPACE;
    using mem_space = TEST_MEMSPACE;
    // ====================================================
    //                   Read inputs
    // ====================================================
    std::string input = "cube_heated_region.json";

    CabanaPD::Inputs inputs( input );

    double rho0 = inputs["density"];
    double E = inputs["elastic_modulus"];
    double nu = 0.25;
    double K = E / ( 3 * ( 1 - 2 * nu ) );
    double alpha = 0.0; // to only do thermomechanics
    double kappa = inputs["thermal_conductivity"];
    double cp = inputs["specific_heat_capacity"];
    double horizon = inputs["horizon"];
    horizon += 1e-10;

    double temp0 = inputs["reference_temperature"];

    // ====================================================
    //                  Discretization
    // ====================================================
    std::array<double, 3> low_corner = inputs["low_corner"];
    std::array<double, 3> high_corner = inputs["high_corner"];

    // ====================================================
    //            Force model type
    // ====================================================
    using model_type = ModelType;
    using thermal_type = CabanaPD::DynamicTemperature;

    // ====================================================
    //                 Particle generation
    // ====================================================
    // Note that individual inputs can be passed instead (see other examples).
    CabanaPD::Particles particles( mem_space{}, model_type{}, thermal_type{} );
    particles.domain( inputs );
    particles.create( exec_space{} );

    // ====================================================
    //            Custom particle initialization
    // ====================================================
    auto rho = particles.sliceDensity();
    auto x = particles.sliceReferencePosition();
    auto temp = particles.sliceTemperature();
    auto type = particles.sliceType();

    auto init_functor = KOKKOS_LAMBDA( const int pid )
    {
        // Density
        rho( pid ) = rho0;
        temp( pid ) = temp0;
    };
    particles.update( exec_space{}, init_functor );

    // ====================================================
    //            Force model
    // ====================================================
    CabanaPD::ForceModel force_model( model_type{}, CabanaPD::NoFracture{},
                                      horizon, K, temp, kappa, cp, alpha,
                                      temp0 );

    // ====================================================
    //                   Create solver
    // ====================================================

    CabanaPD::Solver solver( inputs, particles, force_model );

    // ====================================================
    //            Boundary condition
    // ====================================================
    double edge_length_cube = high_corner[0] - low_corner[0] - 2 * horizon;
    CabanaPD::Region<CabanaPD::RectangularPrism> center_cube(
        low_corner[0] + horizon, high_corner[0] - horizon,
        low_corner[1] + horizon, high_corner[1] - horizon,
        low_corner[2] + horizon, high_corner[2] - horizon );
    CabanaPD::Region<CabanaPD::RectangularPrism> heated_region(
        low_corner[0], low_corner[0] + horizon + 0.1 * ( edge_length_cube ),
        low_corner[1], high_corner[1], low_corner[2], high_corner[2] );
    CabanaPD::Region<CabanaPD::RectangularPrism> constant_temperature_region(
        high_corner[0] - horizon, high_corner[0], low_corner[1], high_corner[1],
        low_corner[2], high_corner[2] );

    // Create BC last to ensure ghost particles are included.
    auto u = solver.particles.sliceDisplacement();
    x = solver.particles.sliceReferencePosition();
    temp = solver.particles.sliceTemperature();
    double power = 1.;
    auto heated_region_func = KOKKOS_LAMBDA( const int pid, const double )
    {
        if ( heated_region.inside( x, pid ) )
        {
            temp( pid ) += power / rho0 / cp;
        }
        if ( constant_temperature_region.inside( x, pid ) )
        {
            temp( pid ) = temp0;
        }
    };
    CabanaPD::BodyTerm bc( heated_region_func, solver.particles.size(), false );
    // ====================================================
    //                   Simulation run
    // ====================================================
    solver.init( bc );
    solver.run( bc );

    // ====================================================
    //            Validation functor
    // ====================================================
    using HostAoSoA =
        Cabana::AoSoA<Cabana::MemberTypes<double[3], double[3], double>,
                      Kokkos::HostSpace>;
    HostAoSoA aosoa_host( "host_aosoa", x.size() );
    auto u_host = Cabana::slice<0>( aosoa_host );
    auto x_host = Cabana::slice<1>( aosoa_host );
    auto temp_host = Cabana::slice<2>( aosoa_host );

    Cabana::deep_copy( u_host, u );
    Cabana::deep_copy( x_host, x );
    Cabana::deep_copy( temp_host, temp );

    // Create region for center of cube. This helps eliminate surface effects
    CabanaPD::Region<CabanaPD::RectangularPrism> center_region(
        low_corner[0], high_corner[0], -0.25 * horizon, +0.25 * horizon,
        -0.25 * horizon, +0.25 * horizon );

    // Primary check on the particle temperature that results from the
    // linear temperature profile in main direction, which is x
    // for ( size_t pid = 0; pid < x.size(); pid++ )
    //{
    //    if ( center_cube.inside( x_host, pid ) )
    //    {
    //        if ( center_region.inside( x_host, pid ) )
    //        {
    //            EXPECT_NEAR( temp_host( pid ),
    //                         x_host( pid, 0 ) / edge_length_cube * delta_temp
    //                         *
    //                             end_time_factor,
    //                         2.e-16 );
    //        }
    //    }
    //}

    // Secondary check on the particle displacement. It should be 0 as the
    // thermal expansion is 0.
    // for ( size_t pid = 0; pid < x.size(); pid++ )
    //{
    //    if ( center_cube.inside( x_host, pid ) )
    //    {
    //        if ( center_region.inside( x_host, pid ) )
    //        {
    //            EXPECT_FLOAT_EQ( u_host( pid, 0 ), 0.0 );
    //            EXPECT_FLOAT_EQ( u_host( pid, 1 ), 0.0 );
    //            EXPECT_FLOAT_EQ( u_host( pid, 2 ), 0.0 );
    //        }
    //    }
    //}
}

TEST( TEST_CATEGORY, test_cube_heated_region_PMB )
{
    test_cube_heated_region( CabanaPD::PMB{} );
}

// TODO: we currently don't have LPS and thermomechanics
// TEST( TEST_CATEGORY, test_cube_heated_region_LPS )
//{
//    test_cube_heated_region( CabanaPD::LPS{} );
//};

} // namespace Test
