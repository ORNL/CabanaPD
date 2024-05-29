/****************************************************************************
 * Copyright (c) 2022-2023 by Oak Ridge National Laboratory                 *
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
    double dx1 = particles.dx[dims[0]] / 2.0;
    double dx2 = particles.dx[dims[1]] / 2.0;
    double center1 = particles.local_mesh_lo[dims[0]] +
                     particles.global_mesh_ext[dims[0]] / 2.0;
    double center2 = particles.local_mesh_lo[dims[1]] +
                     particles.global_mesh_ext[dims[1]] / 2.0;

    auto x = particles.sliceReferencePosition();
    auto u = particles.sliceDisplacement();
    // Find points closest to the center line.
    double center_min1;
    double center_min2;
    auto find_profile =
        KOKKOS_LAMBDA( const int pid, double& min1, double& min2 )
    {
        if ( Kokkos::abs( x( pid, dims[0] ) - center1 ) < min1 )
            min1 = x( pid, dims[0] );
        if ( Kokkos::abs( x( pid, dims[1] ) - center2 ) < min2 )
            min2 = x( pid, dims[1] );
    };
    Kokkos::RangePolicy<typename memory_space::execution_space> policy(
        0, x.size() );
    Kokkos::parallel_reduce( "displacement_profile", policy, find_profile,
                             Kokkos::Min<double>( center_min1 ),
                             Kokkos::Min<double>( center_min2 ) );

    // Extract points along that line.
    auto measure_profile = KOKKOS_LAMBDA( const int pid )
    {
        if ( x( pid, dims[0] ) - center_min1 < dx1 &&
             x( pid, dims[0] ) - center_min1 > -dx1 &&
             x( pid, dims[1] ) - center_min2 < dx2 &&
             x( pid, dims[1] ) - center_min2 > -dx2 )
        {
            auto c = Kokkos::atomic_fetch_add( &count( 0 ), 1 );
            profile( c, 0 ) = x( pid, profile_dim );
            profile( c, 1 ) = user( pid );
        }
    };
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
void createDisplacementProfile( MPI_Comm comm, const int num_cell,
                                const int profile_dim, std::string file_name,
                                ParticleType particles )
{
    auto u = particles.sliceDisplacement();
    auto value = KOKKOS_LAMBDA( const int pid )
    {
        return u( pid, profile_dim );
    };
    createOutputProfile( comm, num_cell, profile_dim, file_name, particles,
                         value );
}

template <typename ParticleType>
void createDisplacementMagnitudeProfile( MPI_Comm comm, const int num_cell,
                                         const int profile_dim,
                                         std::string file_name,
                                         ParticleType particles )
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
