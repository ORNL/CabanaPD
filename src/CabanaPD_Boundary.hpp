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

#ifndef BOUNDARYCONDITION_H
#define BOUNDARYCONDITION_H

#include <Kokkos_Core.hpp>

#include <Cabana_Core.hpp>
#include <Cajita.hpp>

namespace CabanaPD
{

struct ZeroBoundary
{
};

struct RegionBoundary
{
    double low_x;
    double high_x;
    double low_y;
    double high_y;
    double low_z;
    double high_z;

    RegionBoundary( const double _low_x, const double _high_x,
                    const double _low_y, const double _high_y,
                    const double _low_z, const double _high_z )
        : low_x( _low_x )
        , high_x( _high_x )
        , low_y( _low_y )
        , high_y( _high_y )
        , low_z( _low_z )
        , high_z( _high_z ){};
};

template <class MemorySpace, class BoundaryType>
struct BoundaryIndexSpace;

template <class MemorySpace>
struct BoundaryIndexSpace<MemorySpace, RegionBoundary>
{
    using index_view_type = Kokkos::View<std::size_t*, MemorySpace>;
    index_view_type _index_space;

    template <class ExecSpace, class Particles>
    BoundaryIndexSpace( ExecSpace exec_space, Particles particles,
                        RegionBoundary plane )
    {
        create( exec_space, particles, plane );
    }

    template <class ExecSpace, class Particles>
    void create( ExecSpace, Particles particles, RegionBoundary plane )
    {
        // Guess 10% boundary particles.
        auto index_space =
            index_view_type( "boundary_indices", particles.n_local * 0.1 );
        auto count = index_view_type( "count", 1 );
        auto x = particles.slice_x();
        Kokkos::RangePolicy<ExecSpace> policy( 0, particles.n_local );
        auto index_functor = KOKKOS_LAMBDA( const std::size_t pid )
        {
            if ( x( pid, 0 ) >= plane.low_x && x( pid, 0 ) <= plane.high_x &&
                 x( pid, 1 ) >= plane.low_y && x( pid, 1 ) <= plane.high_y &&
                 x( pid, 2 ) >= plane.low_z && x( pid, 2 ) <= plane.high_z )
            {
                // Resize after count if needed.
                auto c = Kokkos::atomic_fetch_add( &count( 0 ), 1 );
                if ( c < index_space.size() )
                {
                    index_space( c ) = pid;
                }
            }
        };

        Kokkos::parallel_for( "CabanaPD::BC::create", policy, index_functor );
        auto count_host =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, count );
        if ( count_host( 0 ) != index_space.size() )
        {
            Kokkos::resize( index_space, count_host( 0 ) );
        }
        if ( count_host( 0 ) > index_space.size() )
        {
            Kokkos::deep_copy( count, 0 );
            Kokkos::parallel_for( "CabanaPD::BC::create", policy,
                                  index_functor );
        }
        _index_space = index_space;
    }
};

template <class ExecSpace, class Particles, class BoundaryType>
auto createBoundaryIndexSpace( ExecSpace exec_space, Particles particles,
                               BoundaryType plane )
{
    using memory_space = typename Particles::memory_space;
    return BoundaryIndexSpace<memory_space, BoundaryType>( exec_space,
                                                           particles, plane );
}

class ForceBCTag
{
};

template <class BCIndexSpace, class BCTag>
struct BoundaryCondition;

template <class BCIndexSpace>
struct BoundaryCondition<BCIndexSpace, ForceBCTag>
{
    using view_type = typename BCIndexSpace::index_view_type;
    view_type _index_space;

    BoundaryCondition( BCIndexSpace bc_index_space )
        : _index_space( bc_index_space._index_space )
    {
    }

    template <class ExecSpace, class ParticleType>
    void apply( ExecSpace, ParticleType& particles )
    {
        auto f = particles.slice_f();
        auto index_space = _index_space;
        Kokkos::RangePolicy<ExecSpace> policy( 0, index_space.size() );
        Kokkos::parallel_for(
            "CabanaPD::BC::apply", policy, KOKKOS_LAMBDA( const int b ) {
                auto pid = index_space( b );
                for ( int d = 0; d < 3; d++ )
                    f( pid, d ) = 0.0;
            } );
    }
};

template <class ExecSpace, class Particles, class BoundaryType, class BCTag>
auto createBoundaryCondition( ExecSpace exec_space, Particles particles,
                              BoundaryType plane, BCTag )
{
    using memory_space = typename Particles::memory_space;
    using bc_index_type = BoundaryIndexSpace<memory_space, BoundaryType>;
    bc_index_type bc_indices =
        createBoundaryIndexSpace( exec_space, particles, plane );
    return BoundaryCondition<bc_index_type, BCTag>( bc_indices );
}

struct ImpactBCTag
{
};
struct AllTag
{
};

template <class AllTag, class BCTag>
struct BoundaryCondition;

template <>
struct BoundaryCondition<AllTag, ImpactBCTag>
{
    double _R;
    double _v;
    double _dt;
    double _z;

    BoundaryCondition( const double R, const double v, const double dt )
        : _R( R )
        , _v( v )
        , _dt( dt )
        , _z( 0.0 )
    {
    }

    template <class ExecSpace, class ParticleType>
    void apply( ExecSpace, ParticleType& particles )
    {
        _z += _dt * _v;
        auto f = particles.slice_f();
        auto x = particles.slice_x();
        Kokkos::RangePolicy<ExecSpace> policy( 0, x.size() );
        Kokkos::parallel_for(
            "CabanaPD::BC::apply", policy, KOKKOS_LAMBDA( const int p ) {
                double z = 0.0 - x( p, 2 );
                double r = sqrt( x( p, 0 ) * x( p, 0 ) + x( p, 1 ) * x( p, 1 ) +
                                 z * z );
                // if ( r < _R )
                std::cout << ( x( p, 2 ) >= _z ) << " " << _z << " "
                          << x( p, 2 ) << std::endl;
                // if ( x( p, 2 ) >= _z ) //&& x( p, 2 ) < _z + _R * 2.0 )
                f( p, 2 ) += -1.0e17 * ( r - _R ) * ( r - _R );
            } );
    }
};

} // namespace CabanaPD

#endif
