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
    std::array<int, 3> num_cells = inputs["num_cells"];

    // ====================================================
    //            Force model
    // ====================================================
    using model_type = CabanaPD::PMB;
    CabanaPD::MechanicsModel mechanics_model( model_type{}, horizon, K );
    CabanaPD::FractureModel fracture_model( horizon, K, G0 );
    CabanaPD::ForceModel force_model( mechanics_model, fracture_model );

    // ====================================================
    //                 Particle generation
    // ====================================================
    // Note that individual inputs can be passed instead (see other examples).
    CabanaPD::Particles particles( mem_space{}, model_type{},
                                   CabanaPD::EnergyStressOutput{} );
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

    // Create BC last to ensure ghost particles are included.
    auto u = solver.particles.sliceDisplacement();
    double strain = 0.0001;
    double end_time_factor = 0.5;
    auto disp_func = KOKKOS_LAMBDA( const int pid, const double t )
    {
        double non_dim_time = t / ( final_time * end_time_factor );
        if ( non_dim_time >= 1.0 )
            non_dim_time = 1.0;

        if ( !center_cube.inside( x, pid ) )
        {
            // uni-axial strain
            u( pid, 0 ) = x( pid, 0 ) * strain * non_dim_time;
            // poission contraction
            u( pid, 1 ) = x( pid, 1 ) * -nu * strain * non_dim_time;
            u( pid, 2 ) = x( pid, 2 ) * -nu * strain * non_dim_time;
        }
    };
    auto bc = CabanaPD::createBoundaryCondition(
        disp_func, exec_space{}, solver.particles, false, full_cube );
    // ====================================================
    //                   ADR Integrator
    // ====================================================
    double adrSubDeltaT = 1.0;
    auto particleADRIntegrator =
        CabanaPD::createADRParticleIntegratorWithSimpleMass(
            exec_space{}, f, adrSubDeltaT, horizon,
            ( high_corner[0] - low_corner[0] ) / num_cells[0],
            18.0 * static_cast<double>( K ) /
                ( Kokkos::numbers::pi * horizon * horizon * horizon * horizon ),
            2. );

    // ====================================================
    //                   Simulation run
    // ====================================================
    solver.init( bc );
    particleADRIntegrator.reset( exec_space{}, particles );

    // Do all of the simulation with ADR integration.
    double time = 0.0;
    int numADRSteps = 2;
    double adrDeltaT = final_time / numADRSteps;

    for ( int adrStep = 1; adrStep <= numADRSteps; ++adrStep )
    {
        time = adrDeltaT * adrStep;
        // integrate using the ADR while disabling bond failure. With bond
        // failure active, we can not do large load increments (timesteps) as
        // the particles at the boundary would immediately fail.
        CabanaPD::runUntilConvergedWithExternalIntegrator(
            exec_space{}, solver, particleADRIntegrator, bc, time, true, 1e-11,
            1e-5, 30'000, MPI_COMM_WORLD );

        solver.computeStressTensorField();
        solver.particles.output( adrStep, time, solver.output_reference );

        std::cout << adrStep << "/" << numADRSteps << " " << time
                  << " via ADR integration\n";
    }

    // ====================================================
    //            Validation functor
    // ====================================================
    auto sigma = particles.sliceStress();
    using HostAoSoA_type =
        Cabana::AoSoA<Cabana::MemberTypes<double[3], double[3], double[3]>,
                      Kokkos::HostSpace>;
    HostAoSoA_type aosoa_host( "host_aosoa", particles.size() );
    auto u_host = Cabana::slice<0>( aosoa_host );
    auto f_host = Cabana::slice<1>( aosoa_host );
    auto x_host = Cabana::slice<2>( aosoa_host );

    Cabana::deep_copy( u_host, u );
    Cabana::deep_copy( f_host, f );
    Cabana::deep_copy( x_host, x );

    // Compute stresses
    solver.computeStressTensorField();
    using HostAoSoA_stress_type =
        Cabana::AoSoA<typename decltype( particles )::output_types,
                      Kokkos::HostSpace, 1>;
    HostAoSoA_stress_type aosoa_stress_host( "host_stress-aosoa",
                                             particles.size() );
    auto sigma_host = Cabana::slice<0>( aosoa_stress_host );

    Cabana::deep_copy( sigma_host, sigma );

    // Create region for center of cube. This helps eliminate surface effects
    CabanaPD::Region<CabanaPD::RectangularPrism> center_region(
        low_corner[0], high_corner[0], -0.25 * horizon, +0.25 * horizon,
        -0.25 * horizon, +0.25 * horizon );

    // Primary check on the particle displacement that results from the
    // displacement of the clamps check strain in main direction, which is x
    Kokkos::parallel_for(
        "on-axis_displacement_center_region",
        Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(
            0, particles.size() ),
        KOKKOS_LAMBDA( const int pid ) {
            if ( center_cube.inside( x_host, pid ) )
            {
                if ( center_region.inside( x_host, pid ) )
                {
                    EXPECT_NEAR( u_host( pid, 0 ), x( pid, 0 ) * strain,
                                 Kokkos::abs( x( pid, 0 ) * strain ) * 1e-10 );
                }
            }
        } );

    // Secondary check on the particle displacement that results from the
    // Poisson effect. We only check y check displacement created by nu
    Kokkos::parallel_for(
        "off-axis_displacement_center_region",
        Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(
            0, particles.size() ),
        KOKKOS_LAMBDA( const int pid ) {
            if ( center_cube.inside( x_host, pid ) )
            {
                if ( center_region.inside( x_host, pid ) )
                {
                    EXPECT_NEAR(
                        u_host( pid, 1 ), ( -nu ) * x( pid, 1 ) * strain,
                        Kokkos::abs( -nu * x( pid, 1 ) * strain ) * 1e-10 );
                    EXPECT_NEAR(
                        u_host( pid, 2 ), ( -nu ) * x( pid, 2 ) * strain,
                        Kokkos::abs( -nu * x( pid, 2 ) * strain ) * 1e-10 );
                }
            }
        } );

    // Tertiary check on-axis and off-axis stresses
    Kokkos::parallel_for(
        "stress_center_region",
        Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(
            0, particles.size() ),
        KOKKOS_LAMBDA( const int pid ) {
            if ( center_cube.inside( x_host, pid ) )
            {
                if ( center_region.inside( x_host, pid ) )
                {
                    for ( int i = 0; i < 3; ++i )
                        for ( int j = 0; j < 3; ++j )
                        {
                            if ( i == j && i == 0 )
                            {
                                // on-axis stress (x-axis)
                                EXPECT_NEAR( sigma_host( pid, 0, 0 ),
                                             strain * E,
                                             Kokkos::abs( strain * E ) * 1e-1 );
                            }
                            else
                            {
                                // all other stresses should be 0
                                EXPECT_NEAR( sigma_host( pid, i, j ), 0.0,
                                             1e-1 );
                            }
                        }
                }
            }
        } );
};
} // namespace Test
