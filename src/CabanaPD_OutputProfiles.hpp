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
#include <CabanaPD_Output.hpp>

namespace CabanaPD
{

template <typename ParticleType, typename UserFunctor>
void createOutputProfile( std::string file_name, ParticleType particles,
                          const int profile_dim, UserFunctor user )
{
    using memory_space = typename ParticleType::memory_space;
    using exec_space = typename memory_space::execution_space;

    Region<Line> line( profile_dim, particles.dx );
    ParticleSteeringVector sv( exec_space{}, particles, line );

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
    Kokkos::fence();
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
template <typename MemorySpace, typename FunctorType>
class OutputTimeSeries
{
    using memory_space = MemorySpace;
    using profile_type = Kokkos::View<double*, Kokkos::HostSpace>;

    using steering_vector_type = ParticleSteeringVector<MemorySpace>;
    steering_vector_type _indices;

    std::string file_name;
    profile_type _profile;
    FunctorType _output;
    std::size_t index;

  public:
    OutputTimeSeries( std::string name, Inputs inputs,
                      const steering_vector_type indices, FunctorType output )
        : _indices( indices )
        , file_name( name )
        , _output( output )
        , index( 0 )
    {
        double time = inputs["final_time"];
        double dt = inputs["timestep"];
        double freq = inputs["output_frequency"];
        int output_steps = static_cast<int>( time / dt / freq );
        // Purposely using zero-init here.
        _profile = profile_type( "time_output", output_steps );
    }

    KOKKOS_INLINE_FUNCTION
    void operator()( const int b, double& px ) const
    {
        auto p = _indices._view( b );
        px += _output( p );
    }

    void update()
    {
        Kokkos::RangePolicy<typename memory_space::execution_space> policy(
            0, _indices.size() );
        // Reduce into host view.
        Kokkos::parallel_reduce( "time_series", policy, *this,
                                 _profile( index ) );
        index++;
    }

    void print( MPI_Comm comm )
    {
        auto profile_host = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, _profile );
        MPI_Allreduce( MPI_IN_PLACE, profile_host.data(), profile_host.size(),
                       MPI_DOUBLE, MPI_SUM, comm );
        auto num_particles = static_cast<int>( _indices.size() );
        MPI_Allreduce( MPI_IN_PLACE, &num_particles, 1, MPI_INT, MPI_SUM,
                       comm );

        if ( print_rank() )
        {
            std::fstream fout;
            fout.open( file_name, std::ios::app );
            for ( std::size_t t = 0; t < index; t++ )
            {
                fout << std::fixed << std::setprecision( 15 )
                     << profile_host( t ) << "  "
                     << profile_host( t ) / num_particles << std::endl;
            }
        }
    }
};

template <typename FunctorType, typename ExecSpace, typename ParticleType,
          typename GeometryType>
auto createOutputTimeSeries( std::string name, const Inputs inputs,
                             ExecSpace exec_space,
                             const ParticleType& particles, FunctorType user,
                             const GeometryType geom )
{
    ParticleSteeringVector indices( exec_space, particles, geom );
    return OutputTimeSeries( name, inputs, indices, user );
}

} // namespace CabanaPD

#endif
