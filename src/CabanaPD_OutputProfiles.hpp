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

/******************************************************************************
  Scalar time series
******************************************************************************/
struct ForceDisplacementTag
{
};

template <typename MemorySpace, typename OutputType>
class OutputTimeSeries;

template <typename MemorySpace>
class OutputTimeSeries<MemorySpace, ForceDisplacementTag>
{
    using memory_space = MemorySpace;
    using profile_type = Kokkos::View<double* [2], memory_space>;

    using steering_vector_type = ParticleSteeringVector<MemorySpace>;
    steering_vector_type _indices;

    std::string file_name;
    profile_type _profile;
    int index;

  public:
    OutputTimeSeries( std::string name, Inputs inputs,
                      const steering_vector_type indices )
        : _indices( indices )
        , file_name( name )
        , index( 0 )
    {
        double time = inputs["final_time"];
        double dt = inputs["timestep"];
        double steps = inputs["output_frequency"];
        int output_steps = static_cast<int>( time / dt / steps );
        // Purposely using zero-init here.
        _profile = profile_type( "time_output", output_steps );
    }

    template <typename ParticleType>
    void update( const ParticleType& particles )
    {
        auto x = particles.sliceReferencePosition();
        auto f = particles.sliceForce();
        auto u = particles.sliceDisplacement();

        auto profile = _profile;
        auto index_space = _indices._view;
        auto i = index;
        auto step_output = KOKKOS_LAMBDA( const int b )
        {
            auto p = index_space( b );
            profile( i, 0 ) +=
                Kokkos::sqrt( u( p, 0 ) * u( p, 0 ) + u( p, 1 ) * u( p, 0 ) +
                              u( p, 2 ) * u( p, 2 ) );
            profile( i, 1 ) +=
                Kokkos::sqrt( f( p, 0 ) * f( p, 0 ) + f( p, 1 ) * f( p, 0 ) +
                              f( p, 2 ) * f( p, 2 ) );
            std::cout << i << " " << p << " " << profile( i, 0 ) << " "
                      << profile( i, 1 ) << std::endl;
        };
        Kokkos::RangePolicy<typename memory_space::execution_space> policy(
            0, _indices.size() );
        Kokkos::parallel_for( "time_series", policy, step_output );

        index++;
    }

    void print( int rank )
    {
        std::fstream fout;
        fout.open( file_name, std::ios::app );
        auto profile_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, _profile );
        for ( std::size_t t = 0; t < profile_host.extent( 0 ); t++ )
        {
            fout << rank << " " << profile_host( t, 0 ) / _indices.size() << " "
                 << profile_host( t, 1 ) / _indices.size() << std::endl;
        }
    }
};

template <typename ExecSpace, typename ParticleType, typename GeometryType,
          typename OutputType>
auto createOutputTimeSeries( OutputType, std::string name, const Inputs inputs,
                             ExecSpace exec_space,
                             const ParticleType& particles,
                             const GeometryType geom,
                             const double initial_guess = 1.0 )
{
    using memory_space = typename ParticleType::memory_space;
    std::vector<GeometryType> geom_vec = { geom };
    using sv_type = ParticleSteeringVector<memory_space>;
    sv_type indices = createParticleSteeringVector( exec_space, particles,
                                                    geom_vec, initial_guess );
    return OutputTimeSeries<memory_space, OutputType>( name, inputs, indices );
}

} // namespace CabanaPD

#endif
