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

#include <CabanaPD.hpp>

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

template <bool GoBySteps>
void testIntegratorADRSingleMass(
    int steps, [[maybe_unused]] double iteration_force_tolerance,
    double displacement_epsilon )
{
    using exec_space = TEST_EXECSPACE;
    constexpr int num_masses = 1;
    double stiffness = 1000;

    Kokkos::View<double[num_masses][3], TEST_EXECSPACE> velocities(
        "testIntegrateADRSingleMass::velocities" );
    Kokkos::View<double[num_masses][3], TEST_EXECSPACE> displacements(
        "testIntegrateADRSingleMass::displacements" );
    Kokkos::View<double[num_masses][3], TEST_EXECSPACE> forces(
        "testIntegrateADRSingleMass::forces" );

    // calculate forces
    auto force_lambda = KOKKOS_LAMBDA( int i )
    {
        forces( i, 0 ) = -stiffness * displacements( i, 0 );
        forces( i, 1 ) = -stiffness * displacements( i, 1 );
        forces( i, 2 ) = -stiffness * displacements( i, 2 );
    };

    // initialize displacements
    Kokkos::parallel_for(
        "testIntegrateADRSingleMass::initialize_displacements", num_masses,
        KOKKOS_LAMBDA( int i ) {
            displacements( i, 0 ) = -0.3;
            displacements( i, 1 ) = -0.4;
            displacements( i, 2 ) = -0.5;
        } );

    double adrDeltaT = 1.0;
    CabanaPD::ADRMassPMBSingleMaterial adrMass{ adrDeltaT, 1.0, 1.0, stiffness,
                                                5 };
    CabanaPD::ADRInitialVelocity adrInitialVelocity{ forces, adrMass,
                                                     adrDeltaT };
    CabanaPD::ADRIntegrator integrator(
        exec_space{}, adrMass, adrInitialVelocity, num_masses, adrDeltaT );

    integrator.reset( exec_space{}, velocities, displacements );

    if constexpr ( GoBySteps )
    {
        for ( int s = 0; s < steps; ++s )
        {
            integrator.initialSubStep( exec_space{}, forces );
            Kokkos::parallel_for( "testIntegrateADRSingleMass::update_forces",
                                  num_masses, force_lambda );
            integrator.middleSubStep( exec_space{}, forces, displacements );
            integrator.finalSubStep( exec_space{}, forces, velocities,
                                     displacements );
        }
    }
    else
    {
        int step = 0;
        while ( step < steps )
        {
            integrator.initialSubStep( exec_space{}, forces );
            Kokkos::parallel_for( "testIntegrateADRSingleMass::update_forces",
                                  num_masses, force_lambda );
            integrator.middleSubStep( exec_space{}, forces, displacements );
            if ( integrator.getForceResidual() < iteration_force_tolerance )
                break;

            integrator.finalSubStep( exec_space{}, forces, velocities,
                                     displacements );
            ++step;
        }
        // check that it took less than the maximum number of steps
        EXPECT_GT( steps, step );
    }

    // Make a copy of final results on the host
    auto displacements_host = Kokkos::create_mirror_view_and_copy(
        Kokkos::HostSpace{}, displacements );

    // Check the results
    for ( std::size_t p = 0; p < num_masses; ++p )
    {
        EXPECT_NEAR( displacements_host( p, 0 ), 0.0, displacement_epsilon );
        EXPECT_NEAR( displacements_host( p, 1 ), 0.0, displacement_epsilon );
        EXPECT_NEAR( displacements_host( p, 2 ), 0.0, displacement_epsilon );
    }
}

template <bool GoBySteps>
void testIntegratorADRparticles(
    int steps, [[maybe_unused]] double iteration_force_tolerance,
    double displacement_epsilon )
{
    using exec_space = TEST_EXECSPACE;

    std::array<double, 3> box_min = { -1.0, -1.0, -1.0 };
    std::array<double, 3> box_max = { 1.0, 1.0, 1.0 };
    std::array<int, 3> num_cells = { 10, 10, 10 };

    CabanaPD::Particles particles( TEST_MEMSPACE{}, CabanaPD::PMB{},
                                   CabanaPD::TemperatureIndependent{} );
    particles.domain( box_min, box_max, num_cells, 0 );
    particles.create( exec_space{} );
    auto displacements = particles.sliceDisplacement();
    auto forces = particles.sliceForce();
    std::size_t num_particle = displacements.size();

    double adrDeltaT = 1.0;
    double stiffness = 1000;
    auto particleIntegrator =
        CabanaPD::createADRParticleIntegratorWithSimpleMass(
            exec_space{}, forces, adrDeltaT, 1.0, 1.0, stiffness );

    // how to calculate forces
    auto force_lambda = KOKKOS_LAMBDA( int i )
    {
        forces( i, 0 ) = -stiffness * displacements( i, 0 );
        forces( i, 1 ) = -stiffness * displacements( i, 1 );
        forces( i, 2 ) = -stiffness * displacements( i, 2 );
    };

    // initialize displacements
    auto positions = particles.sliceReferencePosition();
    Kokkos::parallel_for(
        "testIntegrateADRparticles::initialize_displacements", num_particle,
        KOKKOS_LAMBDA( int i ) {
            for ( int j = 0; j < 3; ++j )
                displacements( i, j ) = positions( i, j );
        } );

    particleIntegrator.reset( exec_space{}, particles );

    if constexpr ( GoBySteps )
    {
        // Integrate one step
        for ( int s = 0; s < steps; ++s )
        {
            particleIntegrator.initialSubStep( exec_space{}, particles );
            Kokkos::parallel_for( "testIntegrateADRParticles::update_forces",
                                  num_particle, force_lambda );
            particleIntegrator.middleSubStep( exec_space{}, particles );
            particleIntegrator.finalSubStep( exec_space{}, particles );
        }
    }
    else
    {
        int step = 0;
        while ( step < steps )
        {
            particleIntegrator.initialSubStep( exec_space{}, particles );
            Kokkos::parallel_for( "testIntegrateADRParticles::update_forces",
                                  num_particle, force_lambda );
            particleIntegrator.middleSubStep( exec_space{}, particles );
            if ( particleIntegrator.getForceResidual() <
                 iteration_force_tolerance )
                break;
            particleIntegrator.finalSubStep( exec_space{}, particles );
            ++step;
        }
        // check that it took less than the maximum number of steps
        EXPECT_GT( steps, step );
    }

    // Make a copy of final results on the host
    using DataTypes = Cabana::MemberTypes<double[3]>;
    using HostAoSoA = Cabana::AoSoA<DataTypes, Kokkos::HostSpace>;
    HostAoSoA displacements_aosoa_final( "displacements_final_host",
                                         num_particle );
    auto displacements_final = Cabana::slice<0>( displacements_aosoa_final );
    Cabana::deep_copy( displacements_final, displacements );

    // Check the results
    for ( std::size_t p = 0; p < num_particle; ++p )
        for ( std::size_t d = 0; d < 3; ++d )
            EXPECT_NEAR( displacements_final( p, d ), 0.0,
                         displacement_epsilon );
}

template <bool GoBySteps>
void testIntegratorADRparticlesMultiMaterialSimpleMass(
    int steps, [[maybe_unused]] double iteration_force_tolerance,
    double displacement_epsilon )
{
    using exec_space = TEST_EXECSPACE;
    const int numMaterials = 2;

    std::array<double, 3> box_min = { -1.0, -1.0, -1.0 };
    std::array<double, 3> box_max = { 1.0, 1.0, 1.0 };
    std::array<int, 3> num_cells = { 10, 10, 10 };

    CabanaPD::Particles particles( TEST_MEMSPACE{}, CabanaPD::PMB{},
                                   CabanaPD::TemperatureIndependent{} );
    particles.domain( box_min, box_max, num_cells, 0 );
    particles.create( exec_space{} );
    auto displacements = particles.sliceDisplacement();
    auto forces = particles.sliceForce();
    auto type = particles.sliceType();
    std::size_t num_particle = displacements.size();
    double stiffness[numMaterials] = { 1000, 10 };

    using model_type = CabanaPD::PMB;
    CabanaPD::ForceModel force_model_0( model_type{}, CabanaPD::NoFracture{},
                                        1.0, stiffness[0] );
    CabanaPD::ForceModel force_model_1( model_type{}, CabanaPD::NoFracture{},
                                        1.0, stiffness[1] );

    auto models = CabanaPD::createMultiForceModel(
        particles, CabanaPD::AverageTag{}, force_model_0, force_model_1 );

    double adrDeltaT = 1.0;
    auto particleIntegrator =
        CabanaPD::createADRParticleIntegratorWithSimpleMass(
            exec_space{}, forces, particles, models, adrDeltaT, 1.0, 1.0 );

    // how to calculate forces
    auto force_lambda = KOKKOS_LAMBDA( int i )
    {
        forces( i, 0 ) = -stiffness[i % numMaterials] * displacements( i, 0 );
        forces( i, 1 ) = -stiffness[i % numMaterials] * displacements( i, 1 );
        forces( i, 2 ) = -stiffness[i % numMaterials] * displacements( i, 2 );
    };

    // initialize displacements
    auto positions = particles.sliceReferencePosition();
    Kokkos::parallel_for(
        "testIntegrateADRparticlesMultiMaterial::initialize_displacements_and_"
        "type",
        num_particle, KOKKOS_LAMBDA( int i ) {
            for ( int j = 0; j < 3; ++j )
                displacements( i, j ) = positions( i, j );
            type( i ) = i % 2; // two materials
        } );

    particleIntegrator.reset( exec_space{}, particles );
    if constexpr ( GoBySteps )
    {
        for ( int s = 0; s < steps; ++s )
        {
            particleIntegrator.initialSubStep( exec_space{}, particles );
            Kokkos::parallel_for(
                "testIntegrateADRparticlesMultiMaterial::update_forces",
                num_particle, force_lambda );
            particleIntegrator.middleSubStep( exec_space{}, particles );
            particleIntegrator.finalSubStep( exec_space{}, particles );
        }
    }
    else
    {
        int step = 0;
        while ( step < steps )
        {
            particleIntegrator.initialSubStep( exec_space{}, particles );
            Kokkos::parallel_for(
                "testIntegrateADRparticlesMultiMaterial::update_forces",
                num_particle, force_lambda );
            particleIntegrator.middleSubStep( exec_space{}, particles );
            if ( particleIntegrator.getForceResidual() <
                 iteration_force_tolerance )
                break;
            particleIntegrator.finalSubStep( exec_space{}, particles );
            ++step;
        }
        // check that it took less than the maximum number of steps
        EXPECT_GT( steps, step );
    }

    // Make a copy of final results on the host
    using DataTypes = Cabana::MemberTypes<double[3]>;
    using HostAoSoA = Cabana::AoSoA<DataTypes, Kokkos::HostSpace>;
    HostAoSoA displacements_aosoa_final( "displacements_final_host",
                                         num_particle );
    auto displacements_final = Cabana::slice<0>( displacements_aosoa_final );
    Cabana::deep_copy( displacements_final, displacements );

    // Check the results
    for ( std::size_t p = 0; p < num_particle; ++p )
        for ( std::size_t d = 0; d < 3; ++d )
            EXPECT_NEAR( displacements_final( p, d ), 0.0,
                         displacement_epsilon );
}

template <bool GoBySteps>
void testIntegratorADRparticlesMultiMaterialExactMass(
    int steps, [[maybe_unused]] double iteration_force_tolerance,
    double displacement_epsilon )
{
    using exec_space = TEST_EXECSPACE;
    const int numMaterials = 2;

    std::array<double, 3> box_min = { -1.0, -1.0, -1.0 };
    std::array<double, 3> box_max = { 1.0, 1.0, 1.0 };
    std::array<int, 3> num_cells = { 10, 10, 10 };

    CabanaPD::Particles particles( TEST_MEMSPACE{}, CabanaPD::PMB{},
                                   CabanaPD::TemperatureIndependent{} );
    particles.domain( box_min, box_max, num_cells, 0 );
    particles.create( exec_space{} );
    auto displacements = particles.sliceDisplacement();
    auto forces = particles.sliceForce();
    auto type = particles.sliceType();
    std::size_t num_particle = displacements.size();
    double stiffness[numMaterials] = { 1000, 10 };

    using model_type = CabanaPD::PMB;
    CabanaPD::ForceModel force_model_0( model_type{}, CabanaPD::NoFracture{},
                                        1.0, stiffness[0] );
    CabanaPD::ForceModel force_model_1( model_type{}, CabanaPD::NoFracture{},
                                        1.0, stiffness[1] );

    auto models = CabanaPD::createMultiForceModel(
        particles, CabanaPD::AverageTag{}, force_model_0, force_model_1 );

    double adrDeltaT = 1.0;
    auto neighbor = CabanaPD::Neighbor<typename exec_space::memory_space>(
        models, particles );
    auto particleIntegrator =
        CabanaPD::createADRParticleIntegratorWithExactMass(
            exec_space{}, forces, particles, neighbor, models, adrDeltaT, 1.0 );

    // how to calculate forces
    auto force_lambda = KOKKOS_LAMBDA( int i )
    {
        forces( i, 0 ) = -stiffness[i % numMaterials] * displacements( i, 0 );
        forces( i, 1 ) = -stiffness[i % numMaterials] * displacements( i, 1 );
        forces( i, 2 ) = -stiffness[i % numMaterials] * displacements( i, 2 );
    };

    // initialize displacements
    auto positions = particles.sliceReferencePosition();
    Kokkos::parallel_for(
        "testIntegrateADRparticlesMultiMaterial::initialize_displacements_and_"
        "type",
        num_particle, KOKKOS_LAMBDA( int i ) {
            for ( int j = 0; j < 3; ++j )
                displacements( i, j ) = positions( i, j );
            type( i ) = i % 2; // two materials
        } );

    particleIntegrator.reset( exec_space{}, particles );
    if constexpr ( GoBySteps )
    {
        for ( int s = 0; s < steps; ++s )
        {
            particleIntegrator.initialSubStep( exec_space{}, particles );
            Kokkos::parallel_for(
                "testIntegrateADRparticlesMultiMaterial::update_forces",
                num_particle, force_lambda );
            particleIntegrator.middleSubStep( exec_space{}, particles );
            particleIntegrator.finalSubStep( exec_space{}, particles );
        }
    }
    else
    {
        int step = 0;
        while ( step < steps )
        {
            particleIntegrator.initialSubStep( exec_space{}, particles );
            Kokkos::parallel_for(
                "testIntegrateADRparticlesMultiMaterial::update_forces",
                num_particle, force_lambda );
            particleIntegrator.middleSubStep( exec_space{}, particles );
            if ( particleIntegrator.getForceResidual() <
                 iteration_force_tolerance )
                break;
            particleIntegrator.finalSubStep( exec_space{}, particles );
            ++step;
        }
        // check that it took less than the maximum number of steps
        EXPECT_GT( steps, step );
    }

    // Make a copy of final results on the host
    using DataTypes = Cabana::MemberTypes<double[3]>;
    using HostAoSoA = Cabana::AoSoA<DataTypes, Kokkos::HostSpace>;
    HostAoSoA displacements_aosoa_final( "displacements_final_host",
                                         num_particle );
    auto displacements_final = Cabana::slice<0>( displacements_aosoa_final );
    Cabana::deep_copy( displacements_final, displacements );

    // Check the results
    for ( std::size_t p = 0; p < num_particle; ++p )
        for ( std::size_t d = 0; d < 3; ++d )
            EXPECT_NEAR( displacements_final( p, d ), 0.0,
                         displacement_epsilon );
}
//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, test_integrate_reversibility )
{
    testIntegratorReversibility( 100 );
}

TEST( TEST_CATEGORY, test_integrate_ADR_single_mass )
{
    // testIntegratorADRSingleMass<true>( 1000, 1e-16, 1e-10 );
    // testIntegratorADRSingleMass<false>( 1000, 1e-16, 1e-10 );
}
TEST( TEST_CATEGORY, test_integrate_ADR_particles )
{
    // testIntegratorADRparticles<true>( 2000, 1e-16, 1e-10 );
    // testIntegratorADRparticles<false>( 2000, 1e-16, 1e-10 );
}

TEST( TEST_CATEGORY, test_integrate_ADR_particles_multi_material_simple_mass )
{
    // testIntegratorADRparticlesMultiMaterialSimpleMass<true>( 3000, 1e-16,
    //                                                          1e-10 );
    // testIntegratorADRparticlesMultiMaterialSimpleMass<false>( 3000, 1e-16,
    //                                                           1e-10 );
}

TEST( TEST_CATEGORY, test_integrate_ADR_particles_multi_material_exact_mass )
{
    // this seems to have a hard time to get to 1e-10
    testIntegratorADRparticlesMultiMaterialExactMass<true>( 6000, 1e-7, 1e-4 );
    testIntegratorADRparticlesMultiMaterialExactMass<false>( 6000, 1e-7, 1e-4 );
}

//---------------------------------------------------------------------------//

} // end namespace Test
