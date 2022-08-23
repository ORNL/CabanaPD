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

// Empty boundary.
struct ZeroBoundary
{
};

// Define a plane or other rectilinear subset of the system as the boundary.
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
    index_view_type _view;
    index_view_type _count;

    template <class ExecSpace, class Particles>
    BoundaryIndexSpace( ExecSpace exec_space, Particles particles,
                        std::vector<RegionBoundary> planes,
                        const double initial_guess )
    {
        _view = index_view_type( "boundary_indices",
                                 particles.n_local * initial_guess );
        _count = index_view_type( "count", 1 );
        Kokkos::deep_copy( _count, 0 );

        for ( RegionBoundary plane : planes )
            update( exec_space, particles, plane );
    }

    template <class ExecSpace, class Particles>
    void update( ExecSpace, Particles particles, RegionBoundary plane )
    {
        auto count_host =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, _count );
        auto init_count = count_host( 0 );

        auto index_space = _view;
        auto count = _count;
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

        Kokkos::parallel_for( "CabanaPD::BC::update", policy, index_functor );
        Kokkos::deep_copy( count_host, _count );
        if ( count_host( 0 ) != index_space.size() )
        {
            Kokkos::resize( index_space, count_host( 0 ) );
        }
        if ( count_host( 0 ) > index_space.size() )
        {
            Kokkos::deep_copy( count, init_count );
            Kokkos::parallel_for( "CabanaPD::BC::update", policy,
                                  index_functor );
        }
    }
};

template <class BoundaryType, class ExecSpace, class Particles>
auto createBoundaryIndexSpace( ExecSpace exec_space, Particles particles,
                               std::vector<RegionBoundary> planes,
                               const double initial_guess )
{
    using memory_space = typename Particles::memory_space;
    return BoundaryIndexSpace<memory_space, BoundaryType>(
        exec_space, particles, planes, initial_guess );
}

struct ForceValueBCTag
{
};
struct ForceUpdateBCTag
{
};

template <class BCIndexSpace, class BCTag>
struct BoundaryCondition;

template <class BCIndexSpace>
struct BoundaryCondition<BCIndexSpace, ForceValueBCTag>
{
    double _value;
    BCIndexSpace _index_space;

    BoundaryCondition( const double value, BCIndexSpace bc_index_space )
        : _value( value )
        , _index_space( bc_index_space )
    {
    }

    template <class ExecSpace, class Particles>
    void update( ExecSpace exec_space, Particles particles,
                 RegionBoundary plane )
    {
        _index_space.update( exec_space, particles, plane );
    }

    template <class ExecSpace, class ParticleType>
    void apply( ExecSpace, ParticleType& particles )
    {
        auto f = particles.slice_f();
        auto x = particles.slice_x();
        auto index_space = _index_space._view;
        Kokkos::RangePolicy<ExecSpace> policy( 0, index_space.size() );
        Kokkos::parallel_for(
            "CabanaPD::BC::apply", policy, KOKKOS_LAMBDA( const int b ) {
                auto pid = index_space( b );
                auto sign = std::abs( x( pid, 0 ) ) / x( pid, 0 );
                std::cout << pid << " " << f( pid, 0 ) << " " << _value << " "
                          << sign << std::endl;
                f( pid, 0 ) += _value * sign;
            } );
    }
};

// Only difference for this tag is the +=...
template <class BCIndexSpace>
struct BoundaryCondition<BCIndexSpace, ForceUpdateBCTag>
{
    double _value;
    BCIndexSpace _index_space;

    BoundaryCondition( const double value, BCIndexSpace bc_index_space )
        : _value( value )
        , _index_space( bc_index_space )
    {
    }

    template <class ExecSpace, class Particles>
    void update( ExecSpace exec_space, Particles particles,
                 RegionBoundary plane )
    {
        _index_space.update( exec_space, particles, plane );
    }

    template <class ExecSpace, class ParticleType>
    void apply( ExecSpace, ParticleType& particles )
    {
        auto f = particles.slice_f();
        auto x = particles.slice_x();
        auto index_space = _index_space._view;
        Kokkos::RangePolicy<ExecSpace> policy( 0, index_space.size() );
        Kokkos::parallel_for(
            "CabanaPD::BC::apply", policy, KOKKOS_LAMBDA( const int b ) {
                auto pid = index_space( b );
                // FIXME don't commit this..
                auto sign = std::abs( x( pid, 0 ) ) / x( pid, 0 );
                std::cout << pid << " " << f( pid, 0 ) << " " << _value << " "
                          << sign << std::endl;
                f( pid, 0 ) += _value * sign;
            } );
    }
};

template <class BoundaryType, class BCTag, class ExecSpace, class Particles>
auto createBoundaryCondition( BCTag, ExecSpace exec_space, Particles particles,
                              std::vector<BoundaryType> planes,
                              const double value,
                              const double initial_guess = 0.1 )
{
    using memory_space = typename Particles::memory_space;
    using bc_index_type = BoundaryIndexSpace<memory_space, BoundaryType>;
    bc_index_type bc_indices = createBoundaryIndexSpace<BoundaryType>(
        exec_space, particles, planes, initial_guess );
    return BoundaryCondition<bc_index_type, BCTag>( value, bc_indices );
}

} // namespace CabanaPD

#endif
