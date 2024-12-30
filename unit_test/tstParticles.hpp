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

#include <gtest/gtest.h>

#include <Cabana_Core.hpp>
#include <Kokkos_Core.hpp>

#include <CabanaPD_Particles.hpp>

namespace Test
{
template <typename ParticlesType>
void checkNumParticles( const ParticlesType& particles,
                        const std::size_t expected_frozen,
                        const std::size_t expected_local,
                        const std::size_t expected_ghost = 0 )
{
    // Check the values.
    auto frozen = particles.numFrozen();
    EXPECT_EQ( frozen, expected_frozen );
    auto local = particles.numLocal();
    EXPECT_EQ( local, expected_local );
    auto ghost = particles.numGhost();
    EXPECT_EQ( ghost, expected_ghost );

    // Check the offsets.
    EXPECT_EQ( particles.frozenOffset(), expected_frozen );
    EXPECT_EQ( particles.localOffset(), expected_frozen + expected_local );
    EXPECT_EQ( particles.referenceOffset(),
               expected_frozen + expected_local + expected_ghost );
}

template <typename ParticlesType>
void checkParticlePositions( const ParticlesType& particles,
                             const std::array<double, 3> box_min,
                             const std::array<double, 3> box_max,
                             const std::size_t start, const std::size_t end )
{
    using HostAoSoA = Cabana::AoSoA<Cabana::MemberTypes<double[3], double>,
                                    Kokkos::HostSpace>;
    HostAoSoA aosoa_host( "host_aosoa", particles.referenceOffset() );
    auto x_host = Cabana::slice<0>( aosoa_host );
    auto x = particles.sliceReferencePosition();
    Cabana::deep_copy( x_host, x );

    // Check the particles were created in the right box.
    for ( std::size_t p = start; p < end; ++p )
        for ( int d = 0; d < 3; ++d )
        {
            EXPECT_GE( x_host( p, d ), box_min[d] );
            EXPECT_LE( x_host( p, d ), box_max[d] );
        }
}

//---------------------------------------------------------------------------//
void testCreateParticles()
{
    using exec_space = TEST_EXECSPACE;

    std::array<double, 3> box_min = { -1.0, -1.0, -1.0 };
    std::array<double, 3> box_max = { 1.0, 1.0, 1.0 };
    std::array<int, 3> num_cells = { 10, 10, 10 };

    // Frozen or all particles first.
    CabanaPD::Particles<TEST_MEMSPACE, CabanaPD::PMB,
                        CabanaPD::TemperatureIndependent>
        particles( exec_space(), box_min, box_max, num_cells, 0 );

    // Check expected values for each block of particles.
    std::size_t expected_local = num_cells[0] * num_cells[1] * num_cells[2];
    std::size_t expected_frozen = 0;

    checkNumParticles( particles, expected_frozen, expected_local );
    checkParticlePositions( particles, box_min, box_max,
                            particles.frozenOffset(), particles.localOffset() );
}

void testCreateFrozenParticles()
{
    using exec_space = TEST_EXECSPACE;

    std::array<double, 3> box_min = { -1.0, -1.0, -1.0 };
    std::array<double, 3> box_max = { 1.0, 1.0, 1.0 };
    std::array<int, 3> num_cells = { 10, 10, 10 };

    // Frozen in bottom half.
    auto init_bottom = KOKKOS_LAMBDA( const int, const double x[3] )
    {
        if ( x[2] < 0.0 )
            return true;
        return false;
    };
    CabanaPD::Particles<TEST_MEMSPACE, CabanaPD::PMB,
                        CabanaPD::TemperatureIndependent>
        particles( exec_space(), box_min, box_max, num_cells, 0, init_bottom, 0,
                   true );

    // Unfrozen in top half.
    auto init_top = KOKKOS_LAMBDA( const int, const double x[3] )
    {
        if ( x[2] > 0.0 )
            return true;
        return false;
    };
    // Create more, starting from the current number of frozen points.
    particles.createParticles( exec_space{}, Cabana::InitUniform{}, init_top,
                               particles.frozenOffset() );

    // Check expected values for each block of particles.
    std::size_t expected_local = num_cells[0] * num_cells[1] * num_cells[2] / 2;
    std::size_t expected_frozen = expected_local;
    checkNumParticles( particles, expected_frozen, expected_local );

    // Check frozen and local separately.
    box_max[2] = 0.0;
    checkParticlePositions( particles, box_min, box_max, 0,
                            particles.frozenOffset() );
    box_min[2] = 0.0;
    box_max[2] = 1.0;
    checkParticlePositions( particles, box_min, box_max,
                            particles.frozenOffset(), particles.localOffset() );
}

void testCreateCustomParticles()
{
    using exec_space = TEST_EXECSPACE;
    using memory_space = TEST_MEMSPACE;

    std::array<double, 3> box_min = { -1.0, -1.0, -1.0 };
    std::array<double, 3> box_max = { 1.0, 1.0, 1.0 };
    std::array<int, 3> num_cells = { 10, 10, 10 };

    std::size_t expected_local = num_cells[0] * num_cells[1] * num_cells[2] / 2;
    Kokkos::View<double* [3], memory_space> position( "custom_position",
                                                      expected_local );
    Kokkos::View<double*, memory_space> volume( "custom_volume",
                                                expected_local );

    // Create every bottom half point in the same location.
    Kokkos::RangePolicy<TEST_EXECSPACE> policy( 0, expected_local );
    Kokkos::parallel_for(
        policy, KOKKOS_LAMBDA( const int p ) {
            for ( int d = 0; d < 3; ++d )
                position( p, d ) = -0.5;
            volume( p ) = 10.2;
        } );

    CabanaPD::Particles<TEST_MEMSPACE, CabanaPD::PMB,
                        CabanaPD::TemperatureIndependent>
        particles( exec_space{}, position, volume, box_min, box_max, num_cells,
                   0, 0, true );

    // Overwrite the previous particles and put them on top instead.
    Kokkos::parallel_for(
        policy, KOKKOS_LAMBDA( const int p ) {
            for ( int d = 0; d < 3; ++d )
                position( p, d ) = 0.5;
            volume( p ) = 3.1;
        } );
    // Create more, starting from the current number of frozen points.
    particles.createParticles( exec_space{}, position, volume,
                               particles.frozenOffset() );

    // Check expected values for each block of particles.
    std::size_t expected_frozen = expected_local;
    checkNumParticles( particles, expected_frozen, expected_local );

    // Check frozen and local separately.
    box_max[2] = 0.0;
    checkParticlePositions( particles, box_min, box_max, 0,
                            particles.frozenOffset() );
    box_min[2] = 0.0;
    box_max[2] = 1.0;
    checkParticlePositions( particles, box_min, box_max,
                            particles.frozenOffset(), particles.localOffset() );
}

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, test_create_local ) { testCreateParticles(); }
TEST( TEST_CATEGORY, test_create_frozen ) { testCreateFrozenParticles(); }
TEST( TEST_CATEGORY, test_create_custom ) { testCreateCustomParticles(); }

//---------------------------------------------------------------------------//

} // end namespace Test
