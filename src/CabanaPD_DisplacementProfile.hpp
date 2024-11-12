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

#ifndef DISPLACEMENTPROFILE_H
#define DISPLACEMENTPROFILE_H

#include <Kokkos_Core.hpp>

#include <Cabana_Core.hpp>

namespace CabanaPD
{

// Given a dimension, returns the other two
auto getDim( const int dim )
{
    Kokkos::Array<int, 2> orthogonal;
    orthogonal[0] = ( dim + 1 ) % 3;
    orthogonal[1] = ( dim + 2 ) % 3;
    return orthogonal;
}

template <typename ParticleType, typename UserFunctor>
void createOutputProfile( MPI_Comm comm, const int num_cell,
                          const int profile_dim, std::string file_name,
                          ParticleType particles, UserFunctor user )
{
    using memory_space = typename ParticleType::memory_space;
    auto profile = Kokkos::View<double* [2], memory_space>(
        Kokkos::ViewAllocateWithoutInitializing( "displacement_profile" ),
        num_cell );
    int mpi_rank;
    MPI_Comm_rank( comm, &mpi_rank );
    Kokkos::View<int*, memory_space> count( "c", 1 );

    auto dims = getDim( profile_dim );
    double dx1 = particles.dx[dims[0]];
    double dx2 = particles.dx[dims[1]];

    auto x = particles.sliceReferencePosition();
    auto measure_profile = KOKKOS_LAMBDA( const int pid )
    {
        if ( x( pid, dims[0] ) < dx1 / 2.0 && x( pid, dims[0] ) > -dx1 / 2.0 &&
             x( pid, dims[1] ) < dx2 / 2.0 && x( pid, dims[1] ) > -dx2 / 2.0 )
        {
            auto c = Kokkos::atomic_fetch_add( &count( 0 ), 1 );
            profile( c, 0 ) = x( pid, profile_dim );
            profile( c, 1 ) = user( pid );
        }
    };
    // TODO: enable ignoring frozen particles.
    Kokkos::RangePolicy<typename memory_space::execution_space> policy(
        0, particles.localOffset() );
    Kokkos::parallel_for( "displacement_profile", policy, measure_profile );
    auto count_host =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, count );
    auto profile_host =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, profile );
    std::fstream fout;

    fout.open( file_name, std::ios::app );
    for ( int p = 0; p < count_host( 0 ); p++ )
    {
        fout << mpi_rank << " " << profile_host( p, 0 ) << " "
             << profile_host( p, 1 ) << std::endl;
    }
}

template <typename ParticleType>
void createDisplacementProfile( MPI_Comm comm, std::string file_name,
                                ParticleType particles, const int num_cell,
                                const int profile_dim,
                                int displacement_dim = -1 )
{
    if ( displacement_dim == -1 )
        displacement_dim = profile_dim;

    auto u = particles.sliceDisplacement();
    auto value = KOKKOS_LAMBDA( const int pid )
    {
        return u( pid, displacement_dim );
    };
    createOutputProfile( comm, num_cell, profile_dim, file_name, particles,
                         value );
}

template <typename ParticleType>
void createDisplacementMagnitudeProfile( MPI_Comm comm, std::string file_name,
                                         ParticleType particles,
                                         const int num_cell,
                                         const int profile_dim )
{
    auto u = particles.sliceDisplacement();
    auto magnitude = KOKKOS_LAMBDA( const int pid )
    {
        return Kokkos::sqrt( u( pid, 0 ) * u( pid, 0 ) +
                             u( pid, 1 ) * u( pid, 1 ) +
                             u( pid, 2 ) * u( pid, 2 ) );
    };
    createOutputProfile( comm, num_cell, profile_dim, file_name, particles,
                         magnitude );
}
} // namespace CabanaPD

#endif
