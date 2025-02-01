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

#include <CabanaPD_Geometry.hpp>

namespace CabanaPD
{

template <typename ParticleType, typename UserFunctor>
void createOutputProfile( std::string file_name, ParticleType particles,
                          const int profile_dim, UserFunctor user )
{
    using memory_space = typename ParticleType::memory_space;

    std::vector<Region<Line>> line = {
        Region<Line>( profile_dim, particles.dx ) };
    auto sv = createParticleSteeringVector(
        typename memory_space::execution_space{}, particles, line, 0.25 );

    auto profile = Kokkos::View<double* [2], memory_space>(
        Kokkos::ViewAllocateWithoutInitializing( "output_profile" ),
        sv.size() );
    auto indices = sv._view;
    auto x = particles.sliceReferencePosition();
    // FIXME: not in order.
    auto measure_profile = KOKKOS_LAMBDA( const int b )
    {
        auto p = indices( b );
        profile( b, 0 ) = x( p, profile_dim );
        profile( b, 1 ) = user( p );
    };
    Kokkos::RangePolicy<typename memory_space::execution_space> policy(
        0, indices.size() );
    Kokkos::parallel_for( "output_profile", policy, measure_profile );
    auto profile_host =
        Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, profile );

    std::fstream fout;
    fout.open( file_name, std::ios::app );
    auto mpi_rank = particles.rank();
    for ( std::size_t p = 0; p < indices.size(); p++ )
    {
        fout << mpi_rank << " " << profile_host( p, 0 ) << " "
             << profile_host( p, 1 ) << std::endl;
    }
}

template <typename ParticleType>
void createDisplacementProfile( std::string file_name, ParticleType particles,
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
    createOutputProfile( file_name, particles, profile_dim, value );
}

template <typename ParticleType>
void createDisplacementMagnitudeProfile( std::string file_name,
                                         ParticleType particles,
                                         const int profile_dim )
{
    auto u = particles.sliceDisplacement();
    auto magnitude = KOKKOS_LAMBDA( const int pid )
    {
        return Kokkos::sqrt( u( pid, 0 ) * u( pid, 0 ) +
                             u( pid, 1 ) * u( pid, 1 ) +
                             u( pid, 2 ) * u( pid, 2 ) );
    };
    createOutputProfile( file_name, particles, profile_dim, magnitude );
}

} // namespace CabanaPD

#endif
