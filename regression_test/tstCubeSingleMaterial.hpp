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
TEST( TEST_CATEGORY, test_cube_single_material_PMB )
{
    // ====================================================
    //            Kokkos spaces
    // ====================================================
    using exec_space = TEST_EXECSPACE;
    using mem_space = TEST_MEMSPACE;
    // ====================================================
    //                   Read inputs
    // ====================================================
    std::string input = "cube_single_material_pmb.json";

    CabanaPD::Inputs inputs( input );

    double rho0 = inputs["density"];
    double E = inputs["elastic_modulus"];
    double nu = 0.25;
    double K = E / ( 3 * ( 1 - 2 * nu ) );
    double G0 = inputs["fracture_energy"];
    double final_time = inputs["final_time"];
    double horizon = inputs["horizon"];
    horizon += 1e-10;

    // ====================================================
    //                  Discretization
    // ====================================================
    std::array<double, 3> low_corner = inputs["low_corner"];
    std::array<double, 3> high_corner = inputs["high_corner"];

    // ====================================================
    //            Force model
    // ====================================================
    using model_type = CabanaPD::PMB;
    CabanaPD::ForceModel force_model( model_type{}, horizon, K, G0 );

    // ====================================================
    //                 Particle generation
    // ====================================================
    // Note that individual inputs can be passed instead (see other examples).
    CabanaPD::Particles particles( mem_space{}, model_type{} );
    particles.domain( inputs );
    particles.create( exec_space{} );

    // ====================================================
    //            Custom particle initialization
    // ====================================================
    auto rho = particles.sliceDensity();
    auto x = particles.sliceReferencePosition();
    auto v = particles.sliceVelocity();
    auto f = particles.sliceForce();
    auto type = particles.sliceType();

    auto init_functor = KOKKOS_LAMBDA( const int pid )
    {
        // Density
        rho( pid ) = rho0;
    };
    particles.update( exec_space{}, init_functor );

    // ====================================================
    //                   Create solver
    // ====================================================

    CabanaPD::Solver solver( inputs, particles, force_model );

    // ====================================================
    //            Boundary condition
    // ====================================================
    // Create region for cube.
    std::cout << "low corner: " << low_corner[0] << " " << low_corner[1] << " "
              << low_corner[2] << std::endl;
    std::cout << "high corner: " << high_corner[0] << " " << high_corner[1]
              << " " << high_corner[2] << std::endl;
    CabanaPD::Region<CabanaPD::RectangularPrism> center_cube(
        low_corner[0] + horizon, high_corner[0] - horizon,
        low_corner[1] + horizon, high_corner[1] - horizon,
        low_corner[2] + horizon, high_corner[2] - horizon );
    CabanaPD::Region<CabanaPD::RectangularPrism> full_cube(
        low_corner[0], high_corner[0], low_corner[1], high_corner[1],
        low_corner[2], high_corner[2] );

    double edge_length_cube = high_corner[0] - low_corner[0] - 2 * horizon;

    // Create BC last to ensure ghost particles are included.
    auto u = solver.particles.sliceDisplacement();
    double strain_rate = 0.0001;
    double end_time_factor = 0.5;
    auto disp_func = KOKKOS_LAMBDA( const int pid, const double t )
    {
        double non_dim_time = t / final_time;
        if ( non_dim_time > end_time_factor )
            non_dim_time = end_time_factor;

        if ( !center_cube.inside( x, pid ) )
        {
            // uni-axial strain
            u( pid, 0 ) =
                x( pid, 0 ) / edge_length_cube * strain_rate * non_dim_time;
        }
    };
    auto bc = CabanaPD::createBoundaryCondition(
        disp_func, exec_space{}, solver.particles, false, full_cube );

    // ====================================================
    //                   Simulation run
    // ====================================================
    solver.init( bc );
    solver.run( bc );

    // ====================================================
    //            Validation functor
    // ====================================================
    using HostAoSoA =
        Cabana::AoSoA<Cabana::MemberTypes<double[3], double[3], double[3]>,
                      Kokkos::HostSpace>;
    HostAoSoA aosoa_host( "host_aosoa", particles.size() );
    auto u_host = Cabana::slice<0>( aosoa_host );
    auto f_host = Cabana::slice<1>( aosoa_host );
    auto x_host = Cabana::slice<2>( aosoa_host );

    Cabana::deep_copy( u_host, u );
    Cabana::deep_copy( f_host, f );
    Cabana::deep_copy( x_host, x );

    // Create region for center of cube. This helps eliminate surface effects
    CabanaPD::Region<CabanaPD::RectangularPrism> center_region(
        low_corner[0], high_corner[0], -0.25 * horizon, +0.25 * horizon,
        -0.25 * horizon, +0.25 * horizon );

    // Primary check on the particle displacement that results from the
    // displacement of the clamps check strain in main direction, which is x
    Kokkos::parallel_for(
        "displacement_center_region_x",
        Kokkos::RangePolicy<exec_space>( 0, particles.size() ),
        KOKKOS_LAMBDA( const int pid ) {
            if ( center_cube.inside( x_host, pid ) )
            {
                if ( center_region.inside( x_host, pid ) )
                {
                    EXPECT_NEAR( u_host( pid, 0 ),
                                 x( pid, 0 ) / edge_length_cube * strain_rate *
                                     end_time_factor,
                                 Kokkos::abs( u_host( pid, 0 ) ) * 1e-1 );
                }
            }
        } );

    // Secondary check on the particle displacement that results from the
    // Poisson effect. We only check y check displacement created by nu
    Kokkos::parallel_for(
        "displacement_center_region_y",
        Kokkos::RangePolicy<exec_space>( 0, particles.size() ),
        KOKKOS_LAMBDA( const int pid ) {
            if ( center_cube.inside( x_host, pid ) )
            {
                if ( center_region.inside( x_host, pid ) )
                {
                    EXPECT_NEAR( u_host( pid, 1 ),
                                 ( -nu ) * x( pid, 1 ) / edge_length_cube *
                                     strain_rate * end_time_factor,
                                 Kokkos::abs( nu * u_host( pid, 0 ) ) * 1e-1 );
                }
            }
        } );

    // TODO we probably want to check the stresses.
    // but either we use ADR or we find a good averaging.
};
} // namespace Test
