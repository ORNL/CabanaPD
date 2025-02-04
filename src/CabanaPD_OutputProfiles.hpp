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
                          const int profile_dim, UserFunctor user,
                          const double initial_guess = 0.25 )
{
    using memory_space = typename ParticleType::memory_space;

    std::vector<Region<Line>> line = {
        Region<Line>( profile_dim, particles.dx ) };
    auto sv =
        createParticleSteeringVector( typename memory_space::execution_space{},
                                      particles, line, initial_guess );

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
template <typename MemorySpace, typename FunctorTypeX, typename FunctorTypeY>
class OutputTimeSeries
{
    using memory_space = MemorySpace;
    using profile_type = Kokkos::View<double* [2], memory_space>;

    using steering_vector_type = ParticleSteeringVector<MemorySpace>;
    steering_vector_type _indices;

    std::string file_name;
    profile_type _profile;
    FunctorTypeX _output_x;
    FunctorTypeY _output_y;
    int index;

  public:
    OutputTimeSeries( std::string name, Inputs inputs,
                      const steering_vector_type indices, FunctorTypeX output_x,
                      FunctorTypeY output_y )
        : _indices( indices )
        , file_name( name )
        , _output_x( output_x )
        , _output_y( output_y )
        , index( 0 )
    {
        double time = inputs["final_time"];
        double dt = inputs["timestep"];
        double freq = inputs["output_frequency"];
        int output_steps = static_cast<int>( time / dt / freq );
        // Purposely using zero-init here.
        _profile = profile_type( "time_output", output_steps );
    }

    void operator()( const int b, double& px, double& py ) const
    {
        auto p = _indices._view( b );
        px += _output_x( p );
        py += _output_y( p );
    }

    void update()
    {
        Kokkos::RangePolicy<typename memory_space::execution_space> policy(
            0, _indices.size() );
        Kokkos::parallel_reduce( "time_series", policy, *this,
                                 _profile( index, 0 ), _profile( index, 1 ) );
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

template <typename UserFunctorX, typename UserFunctorY, typename ExecSpace,
          typename ParticleType, typename GeometryType>
auto createOutputTimeSeries( UserFunctorX user_x, UserFunctorY user_y,
                             std::string name, const Inputs inputs,
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
    return OutputTimeSeries( name, inputs, indices, user_x, user_y );
}

struct ForceDisplacementTag
{
};

template <typename FieldType>
struct updateField
{
    FieldType f;

    updateField( const FieldType& field )
        : f( field )
    {
    }

    auto operator()( const int p ) const
    {
        return Kokkos::sqrt( f( p, 0 ) * f( p, 0 ) + f( p, 1 ) * f( p, 1 ) +
                             f( p, 2 ) * f( p, 2 ) );
    }
};

template <typename ExecSpace, typename ParticleType, typename GeometryType>
auto createOutputTimeSeries( ForceDisplacementTag, std::string name,
                             const Inputs inputs, ExecSpace exec_space,
                             const ParticleType& particles,
                             const GeometryType geom,
                             const double initial_guess = 1.0 )
{
    using memory_space = typename ParticleType::memory_space;
    std::vector<GeometryType> geom_vec = { geom };
    using sv_type = ParticleSteeringVector<memory_space>;
    sv_type indices = createParticleSteeringVector( exec_space, particles,
                                                    geom_vec, initial_guess );

    auto f = particles.sliceForce();
    auto u = particles.sliceDisplacement();
    auto update_f = updateField( f );
    auto update_u = updateField( u );
    return OutputTimeSeries( name, inputs, indices, update_u, update_f );
}

} // namespace CabanaPD

#endif
