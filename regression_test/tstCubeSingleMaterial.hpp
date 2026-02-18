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
    // Create region for both grips.
    CabanaPD::Region<CabanaPD::RectangularPrism> right_grip(
        high_corner[0] - horizon, high_corner[0], low_corner[1], high_corner[1],
        low_corner[2], high_corner[2] );
    CabanaPD::Region<CabanaPD::RectangularPrism> left_grip(
        low_corner[0], low_corner[0] + horizon, low_corner[1], high_corner[1],
        low_corner[2], high_corner[2] );

    // Create BC last to ensure ghost particles are included.
    auto u = solver.particles.sliceDisplacement();
    auto disp_func = KOKKOS_LAMBDA( const int pid, const double t )
    {
        if ( right_grip.inside( x, pid ) )
        {
            u( pid, 0 ) = 1e-0 * t;
            u( pid, 1 ) = 0.0;
            u( pid, 2 ) = 0.0;
        }
        else if ( left_grip.inside( x, pid ) )
        {
            u( pid, 0 ) = 0.0;
            u( pid, 1 ) = 0.0;
            u( pid, 2 ) = 0.0;
        }
    };
    auto bc = CabanaPD::createBoundaryCondition( disp_func, exec_space{},
                                                 solver.particles, false,
                                                 left_grip, right_grip );

    // ====================================================
    //                   Simulation run
    // ====================================================
    solver.init( bc );
    solver.run( bc );

    // ====================================================
    //            Validation functor
    // ====================================================
    // TODO we probably want to check the stresses.
    // but either we use ADR or we find a good averaging.
    // We probably should test both, but I need to think about this some more
    // Also we probably want to check if the poisson ratio is correct. But I am
    // not sure if that works ... maybe just near the centerline
}

} // end namespace Test
