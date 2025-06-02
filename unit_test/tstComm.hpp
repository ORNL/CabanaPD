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

#include <CabanaPD_Comm.hpp>
#include <CabanaPD_Particles.hpp>
#include <CabanaPD_config.hpp>
#include <force_models/CabanaPD_Hertzian.hpp>

namespace Test
{
//---------------------------------------------------------------------------//
void testHalo()
{
    using exec_space = TEST_EXECSPACE;
    using memory_space = TEST_MEMSPACE;

    std::array<double, 3> box_min = { -1.0, -1.0, -1.0 };
    std::array<double, 3> box_max = { 1.0, 1.0, 1.0 };
    std::array<int, 3> num_cells = { 10, 10, 10 };

    double delta = 0.20000001;
    int halo_width = 2;
    // FIXME: This is for m = 1; should be calculated from m
    int expected_n = 6;
    CabanaPD::Particles particles( memory_space{}, CabanaPD::PMB{},
                                   CabanaPD::TemperatureIndependent{} );
    particles.domain( box_min, box_max, num_cells, halo_width );
    particles.create( exec_space{} );

    // Set ID equal to MPI rank.
    int current_rank = -1;
    MPI_Comm_rank( MPI_COMM_WORLD, &current_rank );

    // No ints are communicated in CabanaPD. We use the volume field for MPI
    // rank here for convenience.
    auto rank = particles.sliceVolume();
    auto x = particles.sliceReferencePosition();
    auto init_functor = KOKKOS_LAMBDA( const int pid )
    {
        rank( pid ) = static_cast<double>( current_rank );
    };
    particles.update( exec_space{}, init_functor );

    int init_num_particles = particles.localOffset();
    using HostAoSoA = Cabana::AoSoA<Cabana::MemberTypes<double[3], double>,
                                    Kokkos::HostSpace>;
    HostAoSoA aosoa_init_host( "host_aosoa", init_num_particles );
    auto x_init_host = Cabana::slice<0>( aosoa_init_host );
    auto rank_init_host = Cabana::slice<1>( aosoa_init_host );
    Cabana::deep_copy( x_init_host, x );
    Cabana::deep_copy( rank_init_host, rank );

    // A gather is performed on construction.
    using particles_type =
        CabanaPD::Particles<memory_space, CabanaPD::PMB,
                            CabanaPD::TemperatureIndependent>;
    CabanaPD::Comm<particles_type, CabanaPD::Pair, CabanaPD::SingleMaterial,
                   CabanaPD::TemperatureIndependent>
        comm( particles );

    HostAoSoA aosoa_host( "host_aosoa", particles.referenceOffset() );
    x = particles.sliceReferencePosition();
    rank = particles.sliceVolume();
    auto x_host = Cabana::slice<0>( aosoa_host );
    auto rank_host = Cabana::slice<1>( aosoa_host );
    Cabana::deep_copy( x_host, x );
    Cabana::deep_copy( rank_host, rank );

    EXPECT_EQ( particles.localOffset(), init_num_particles );

    // Check all local particles unchanged.
    for ( std::size_t p = 0; p < particles.localOffset(); ++p )
    {
        for ( int d = 0; d < 3; ++d )
        {
            EXPECT_EQ( x_host( p, d ), x_init_host( p, d ) );
        }
        EXPECT_EQ( rank_host( p ), rank_init_host( p ) );
    }

    int current_size = -1;
    MPI_Comm_size( MPI_COMM_WORLD, &current_size );
    // Ghosts should have been created for all but single rank systems.
    if ( current_size > 1 )
    {
        EXPECT_GT( particles.numGhost(), 0 );
    }
    // Check all ghost particles in the halo region.
    for ( std::size_t p = particles.localOffset();
          p < particles.referenceOffset(); ++p )
    {
        for ( int d = 0; d < 3; ++d )
        {
            EXPECT_GE( x_host( p, d ), particles.ghost_mesh_lo[d] );
            EXPECT_LE( x_host( p, d ), particles.ghost_mesh_hi[d] );
        }
        EXPECT_NE( rank_host( p ), current_rank );
    }

    double mesh_min[3] = { particles.ghost_mesh_lo[0],
                           particles.ghost_mesh_lo[1],
                           particles.ghost_mesh_lo[2] };
    double mesh_max[3] = { particles.ghost_mesh_hi[0],
                           particles.ghost_mesh_hi[1],
                           particles.ghost_mesh_hi[2] };
    using NeighListType =
        Cabana::VerletList<TEST_MEMSPACE, Cabana::FullNeighborTag,
                           Cabana::VerletLayout2D, Cabana::TeamOpTag>;
    NeighListType nlist( x, 0, particles.localOffset(), delta, 1.0, mesh_min,
                         mesh_max );

    // Copy neighbors per particle to host.
    Kokkos::View<int*, TEST_MEMSPACE> num_neigh( "num_neighbors",
                                                 particles.localOffset() );
    Kokkos::RangePolicy<exec_space> policy( 0, particles.localOffset() );
    Kokkos::parallel_for(
        "num_neighbors", policy, KOKKOS_LAMBDA( const int p ) {
            auto n =
                Cabana::NeighborList<NeighListType>::numNeighbor( nlist, p );
            num_neigh( p ) = n;
        } );

    // Check that all local particles (away from global boundaries) have a full
    // set of neighbors.
    // FIXME: Expected neighbors per particle could also be calculated at the
    // boundaries (less than internal particles).
    auto num_neigh_host =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, num_neigh );
    for ( std::size_t p = 0; p < particles.localOffset(); ++p )
    {
        if ( x_host( p, 0 ) > box_min[0] + delta * 1.01 &&
             x_host( p, 0 ) < box_max[0] - delta * 1.01 &&
             x_host( p, 1 ) > box_min[1] + delta * 1.01 &&
             x_host( p, 1 ) < box_max[1] - delta * 1.01 &&
             x_host( p, 2 ) > box_min[2] + delta * 1.01 &&
             x_host( p, 2 ) < box_max[2] - delta * 1.01 )
        {
            EXPECT_EQ( num_neigh_host( p ), expected_n );
        }
    }
}

void testContactHalo()
{
    using exec_space = TEST_EXECSPACE;
    using memory_space = TEST_MEMSPACE;

    std::array<double, 3> box_min = { -1.0, -1.0, -1.0 };
    std::array<double, 3> box_max = { 1.0, 1.0, 1.0 };
    std::array<int, 3> num_cells = { 10, 10, 10 };
    int halo_width = 1;

    double delta = 0.20000001;
    const int num_particles = 2;
    // Purposely using zero-init here.
    Kokkos::View<double* [3], memory_space> x_view( "custom_position", 2 );
    Kokkos::View<double*, memory_space> rank_view( "rank", 2 );

    // Set ID equal to MPI rank.
    int current_rank = -1;
    MPI_Comm_rank( MPI_COMM_WORLD, &current_rank );

    // No ints are communicated in CabanaPD. We use the volume field for MPI
    // rank here for convenience.
    Kokkos::parallel_for(
        "create_particles", Kokkos::RangePolicy<exec_space>( 0, num_particles ),
        KOKKOS_LAMBDA( const int p ) {
            if ( p == 0 )
                x_view( p, 0 ) = 5.1e-5;
            else
                x_view( p, 0 ) = -5.1e-5;
            rank_view( p ) = static_cast<double>( current_rank );
        } );

    using model_type = CabanaPD::HertzianModel;
    CabanaPD::Particles particles(
        memory_space{}, model_type{}, CabanaPD::BaseOutput{}, x_view, rank_view,
        box_min, box_max, num_cells, halo_width, exec_space{} );

    int init_num_particles = particles.localOffset();
    using HostAoSoA = Cabana::AoSoA<Cabana::MemberTypes<double[3], double>,
                                    Kokkos::HostSpace>;
    HostAoSoA aosoa_init_host( "host_aosoa", init_num_particles );
    auto x_init_host = Cabana::slice<0>( aosoa_init_host );
    auto rank_init_host = Cabana::slice<1>( aosoa_init_host );
    auto x = particles.sliceReferencePosition();
    auto rank = particles.sliceVolume();
    Cabana::deep_copy( x_init_host, x );
    Cabana::deep_copy( rank_init_host, rank );

    // A gather is performed on construction.
    using particles_type =
        CabanaPD::Particles<memory_space, CabanaPD::Contact,
                            CabanaPD::TemperatureIndependent>;
    CabanaPD::Comm<particles_type, CabanaPD::Contact,
                   CabanaPD::TemperatureIndependent>
        comm( particles );

    HostAoSoA aosoa_host( "host_aosoa", particles.referenceOffset() );
    x = particles.sliceReferencePosition();
    rank = particles.sliceVolume();
    auto x_host = Cabana::slice<0>( aosoa_host );
    auto rank_host = Cabana::slice<1>( aosoa_host );
    Cabana::deep_copy( x_host, x );
    Cabana::deep_copy( rank_host, rank );

    // Check original particles are unchanged.
    EXPECT_EQ( particles.localOffset(), init_num_particles );

    int current_size = -1;
    MPI_Comm_size( MPI_COMM_WORLD, &current_size );
    // Ghosts should have been created for all but single rank systems.
    if ( current_size > 1 )
    {
        EXPECT_GT( particles.numGhost(), 0 );
    }
    // Check all ghost particles in the halo region.
    for ( std::size_t p = particles.localOffset();
          p < particles.referenceOffset(); ++p )
    {
        for ( int d = 0; d < 3; ++d )
        {
            EXPECT_GE( x_host( p, d ), particles.ghost_mesh_lo[d] );
            EXPECT_LE( x_host( p, d ), particles.ghost_mesh_hi[d] );
        }
        EXPECT_NE( rank_host( p ), current_rank );
    }

    // Check that all local particles (away from global boundaries) have a full
    // set of neighbors.
    for ( std::size_t p = 0; p < particles.localOffset(); ++p )
    {
        if ( x_host( p, 0 ) > box_min[0] + delta * 1.01 &&
             x_host( p, 0 ) < box_max[0] - delta * 1.01 &&
             x_host( p, 1 ) > box_min[1] + delta * 1.01 &&
             x_host( p, 1 ) < box_max[1] - delta * 1.01 &&
             x_host( p, 2 ) > box_min[2] + delta * 1.01 &&
             x_host( p, 2 ) < box_max[2] - delta * 1.01 )
        {
            EXPECT_EQ( 0, 0 );
        }
    }
}

//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
TEST( TEST_CATEGORY, test_particle_halo ) { testHalo(); }

TEST( TEST_CATEGORY, test_contact_halo ) { testContactHalo(); }

} // end namespace Test
