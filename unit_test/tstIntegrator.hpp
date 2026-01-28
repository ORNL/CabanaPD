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

/****************************************************************************
 * Copyright (c) 2018-2021 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#include <algorithm>
#include <vector>

#include <gtest/gtest.h>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <CabanaPD_Integrate.hpp>
#include <CabanaPD_Particles.hpp>
#include <CabanaPD_config.hpp>

namespace Test
{
//---------------------------------------------------------------------------//
void testIntegratorReversibility( int steps )
{
    using exec_space = TEST_EXECSPACE;

    std::array<double, 3> box_min = { -1.0, -1.0, -1.0 };
    std::array<double, 3> box_max = { 1.0, 1.0, 1.0 };
    std::array<int, 3> num_cells = { 10, 10, 10 };

    CabanaPD::Particles particles( TEST_MEMSPACE{}, CabanaPD::PMB{},
                                   CabanaPD::TemperatureIndependent{} );
    particles.domain( box_min, box_max, num_cells, 0 );
    particles.create( exec_space{} );
    auto x = particles.sliceReferencePosition();
    std::size_t num_particle = x.size();

    CabanaPD::VelocityVerlet<> integrator( 0.001 );

    // Keep a copy of initial positions on the host
    using DataTypes = Cabana::MemberTypes<double[3]>;
    using HostAoSoA = Cabana::AoSoA<DataTypes, Kokkos::HostSpace>;
    HostAoSoA x_aosoa_init( "x_init_host", num_particle );
    auto x_init = Cabana::slice<0>( x_aosoa_init );
    Cabana::deep_copy( x_init, x );

    // Integrate one step
    for ( int s = 0; s < steps; ++s )
    {
        integrator.initialHalfStep( exec_space{}, particles );
        integrator.finalHalfStep( exec_space{}, particles );
    }

    // Reverse the system.
    auto v = particles.sliceVelocity();
    Kokkos::RangePolicy<TEST_EXECSPACE> exec_policy( 0, num_particle );
    Kokkos::parallel_for(
        exec_policy, KOKKOS_LAMBDA( const int p ) {
            for ( int d = 0; d < 3; ++d )
                v( p, d ) *= -1.0;
        } );

    // Integrate back
    for ( int s = 0; s < steps; ++s )
    {
        integrator.initialHalfStep( exec_space{}, particles );
        integrator.finalHalfStep( exec_space{}, particles );
    }

    // Make a copy of final results on the host
    HostAoSoA x_aosoa_final( "x_final_host", num_particle );
    auto x_final = Cabana::slice<0>( x_aosoa_final );
    Cabana::deep_copy( x_final, x );

    // Check the results
    x = particles.sliceReferencePosition();
    for ( std::size_t p = 0; p < num_particle; ++p )
        for ( std::size_t d = 0; d < 3; ++d )
            EXPECT_DOUBLE_EQ( x_final( p, d ), x_init( p, d ) );
}

void testIntegratorADR( int steps )
{
    using exec_space = TEST_EXECSPACE;
    constexpr int num_masses = 1;
    double stiffness = 1000;

    Kokkos::View<double[num_masses][3], TEST_EXECSPACE> velocities(
        "testIntegrateADR::velocities" );
    Kokkos::View<double[num_masses][3], TEST_EXECSPACE> displacements(
        "testIntegrateADR::displacements" );
    Kokkos::View<double[num_masses][3], TEST_EXECSPACE> forces(
        "testIntegrateADR::forces" );

    // calculate forces
    auto force_lambda = KOKKOS_LAMBDA( int i )
    {
        forces( i, 0 ) = -stiffness * displacements( i, 0 );
        forces( i, 1 ) = -stiffness * displacements( i, 1 );
        forces( i, 2 ) = -stiffness * displacements( i, 2 );
    };

    // initialize displacements
    Kokkos::parallel_for(
        "testIntegrateADR::initialize_displacements", num_masses,
        KOKKOS_LAMBDA( int i ) {
            displacements( i, 0 ) = -0.3;
            displacements( i, 1 ) = -0.4;
            displacements( i, 2 ) = -0.5;
        } );

    // initialize forces
    Kokkos::parallel_for( "testIntegrateADR::initialize_forces", num_masses,
                          force_lambda );

    double adrDeltaT = 1.0;
    CabanaPD::ADRFictitiousMass adrMass{ adrDeltaT, 1.0, 1.0, stiffness, 5 };
    CabanaPD::ADRInitialVelocity adrInitialVelocity{ forces, adrMass,
                                                     adrDeltaT };
    CabanaPD::ADRIntegrator integrator(
        exec_space{}, adrMass, adrInitialVelocity, num_masses, adrDeltaT );

    integrator.reset( exec_space{} );
    // Integrate one step
    for ( int s = 0; s < steps; ++s )
    {
        integrator.initialStep( exec_space{}, forces );
        Kokkos::parallel_for( "testIntegrateADR::update_forces", num_masses,
                              force_lambda );
        integrator.finalStep( exec_space{}, forces, velocities, displacements );
    }

    // Make a copy of final results on the host
    auto displacements_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, displacements );

    // Check the results
    for ( std::size_t p = 0; p < num_masses; ++p )
    {
        EXPECT_NEAR( displacements_host( p, 0 ), 0.0, 0.01 );
        EXPECT_NEAR( displacements_host( p, 1 ), 0.0, 0.01 );
        EXPECT_NEAR( displacements_host( p, 2 ), 0.0, 0.01 );
    }
}

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, test_integrate_reversibility )
{
    testIntegratorReversibility( 100 );
}

TEST( TEST_CATEGORY, test_integrate_ADR ) { testIntegratorADR( 20 ); }

//---------------------------------------------------------------------------//

} // end namespace Test
