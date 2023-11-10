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
        , high_z( _high_z )
    {
        assert( low_x < high_x );
        assert( low_y < high_y );
        assert( low_z < high_z );
    };
};

template <class MemorySpace, class BoundaryType>
struct BoundaryIndexSpace;

// FIXME: fails for some cases if initial guess is not sufficient.
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

        for ( RegionBoundary plane : planes )
        {
            update( exec_space, particles, plane );
        }
        // Resize after all planes searched.
        auto count_host =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, _count );
        if ( count_host( 0 ) < _view.size() )
        {
            Kokkos::resize( _view, count_host( 0 ) );
        }
    }

    template <class ExecSpace, class Particles>
    void update( ExecSpace, Particles particles, RegionBoundary plane )
    {
        auto count_host =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, _count );
        auto init_count = count_host( 0 );

        auto index_space = _view;
        auto count = _count;
        auto x = particles.sliceReferencePosition();
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
        if ( count_host( 0 ) > index_space.size() )
        {
            Kokkos::resize( index_space, count_host( 0 ) );
            Kokkos::deep_copy( count, init_count );
            Kokkos::parallel_for( "CabanaPD::BC::update", policy,
                                  index_functor );
        }
    }
};

template <class BoundaryType, class ExecSpace, class Particles>
auto createBoundaryIndexSpace( ExecSpace exec_space, Particles particles,
                               std::vector<BoundaryType> planes,
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

struct ForceSymmetric1dBCTag
{
};

template <class BCIndexSpace, class BCTag>
struct BoundaryCondition;

// Reset BC (all dimensions).
template <class BCIndexSpace>
struct BoundaryCondition<BCIndexSpace, ForceValueBCTag>
{
    double _value;
    BCIndexSpace _index_space;
    double _time_start;
    double _time_ramp;
    double _time_end;

    BoundaryCondition( BCIndexSpace bc_index_space, const double value,
                       const double start = 0.0,
                       const double ramp = std::numeric_limits<double>::max(),
                       const double end = std::numeric_limits<double>::max() )
        : _value( value )
        , _index_space( bc_index_space )
        , _time_start( start )
        , _time_ramp( ramp )
        , _time_end( end )
    {
    }

    template <class ExecSpace, class Particles>
    void update( ExecSpace exec_space, Particles particles,
                 RegionBoundary plane )
    {
        _index_space.update( exec_space, particles, plane );
    }

    template <class ExecSpace, class FieldType, class PositionType>
    void apply( ExecSpace, FieldType& f, PositionType&, const double t )
    {
        auto index_space = _index_space._view;
        Kokkos::RangePolicy<ExecSpace> policy( 0, index_space.size() );
        auto value = _value;
        auto start = _time_start;
        auto end = _time_end;
        auto ramp = _time_ramp;
        Kokkos::parallel_for(
            "CabanaPD::BC::apply", policy, KOKKOS_LAMBDA( const int b ) {
                double current = value;
                // Initial linear ramp.
                if ( t < ramp )
                {
                    current = value * ( t - start ) / ( ramp - start );
                }
                if ( t > start && t < end )
                {
                    auto pid = index_space( b );
                    for ( int d = 0; d < 3; d++ )
                        f( pid, d ) = current;
                }
            } );
    }
};

// Increment BC (all dimensions).
template <class BCIndexSpace>
struct BoundaryCondition<BCIndexSpace, ForceUpdateBCTag>
{
    double _value;
    BCIndexSpace _index_space;
    double _time_start;
    double _time_ramp;
    double _time_end;

    BoundaryCondition( BCIndexSpace bc_index_space, const double value,
                       const double start = 0.0,
                       const double ramp = std::numeric_limits<double>::max(),
                       const double end = std::numeric_limits<double>::max() )
        : _value( value )
        , _index_space( bc_index_space )
        , _time_start( start )
        , _time_ramp( ramp )
        , _time_end( end )
    {
        assert( _time_ramp >= _time_start );
        assert( _time_end >= _time_ramp );
        assert( _time_end >= _time_start );
    }

    template <class ExecSpace, class Particles>
    void update( ExecSpace exec_space, Particles particles,
                 RegionBoundary plane )
    {
        _index_space.update( exec_space, particles, plane );
    }

    template <class ExecSpace, class FieldType, class PositionType>
    void apply( ExecSpace, FieldType& f, PositionType&, const double t )
    {
        auto index_space = _index_space._view;
        Kokkos::RangePolicy<ExecSpace> policy( 0, index_space.size() );
        auto value = _value;
        auto start = _time_start;
        auto end = _time_end;
        auto ramp = _time_ramp;
        Kokkos::parallel_for(
            "CabanaPD::BC::apply", policy, KOKKOS_LAMBDA( const int b ) {
                double current = value;
                // Initial linear ramp.
                if ( t < ramp )
                {
                    current = value * ( t - start ) / ( ramp - start );
                }
                if ( t > start && t < end )
                {
                    auto pid = index_space( b );
                    for ( int d = 0; d < 3; d++ )
                        f( pid, d ) += current;
                }
            } );
    }
};

// Symmetric 1d BC applied with opposite sign based on position from the
// midplane in the specified direction.
template <class BCIndexSpace>
struct BoundaryCondition<BCIndexSpace, ForceSymmetric1dBCTag>
{
    BCIndexSpace _index_space;
    double _value_top;
    double _value_bottom;
    double _time_start;
    double _time_ramp;
    double _time_end;

    int _dim;
    double _center;

    BoundaryCondition( BCIndexSpace bc_index_space, const double value,
                       const int dim, const double center,
                       const double start = 0.0,
                       const double ramp = std::numeric_limits<double>::max(),
                       const double end = std::numeric_limits<double>::max() )
        : _index_space( bc_index_space )
        , _value_top( value )
        , _value_bottom( value )
        , _time_start( start )
        , _time_ramp( ramp )
        , _time_end( end )
        , _dim( dim )
        , _center( center )
    {
        assert( _time_ramp >= _time_start );
        assert( _time_end >= _time_ramp );
        assert( _time_end >= _time_start );
        assert( _dim >= 0 );
        assert( _dim < 3 );
    }

    BoundaryCondition( BCIndexSpace bc_index_space, const double value_top,
                       const double value_bottom, const int dim,
                       const double center, const double start = 0.0,
                       const double ramp = std::numeric_limits<double>::max(),
                       const double end = std::numeric_limits<double>::max() )
        : _index_space( bc_index_space )
        , _value_top( value_top )
        , _value_bottom( value_bottom )
        , _time_start( start )
        , _time_ramp( ramp )
        , _time_end( end )
        , _dim( dim )
        , _center( center )
    {
        assert( _time_ramp >= _time_start );
        assert( _time_end >= _time_ramp );
        assert( _time_end >= _time_start );
        assert( _dim >= 0 );
        assert( _dim < 3 );
    }

    template <class ExecSpace, class Particles>
    void update( ExecSpace exec_space, Particles particles,
                 RegionBoundary plane )
    {
        _index_space.update( exec_space, particles, plane );
    }

    template <class ExecSpace, class FieldType, class PositionType>
    void apply( ExecSpace, FieldType& f, PositionType& x, const double t )
    {
        auto index_space = _index_space._view;
        Kokkos::RangePolicy<ExecSpace> policy( 0, index_space.size() );
        auto dim = _dim;
        auto center = _center;
        auto value_top = _value_top;
        auto value_bottom = _value_bottom;
        auto start = _time_start;
        auto end = _time_end;
        auto ramp = _time_ramp;
        Kokkos::parallel_for(
            "CabanaPD::BC::apply", policy, KOKKOS_LAMBDA( const int b ) {
                auto current_top = value_top;
                auto current_bottom = value_bottom;
                // Initial linear ramp.
                if ( t < ramp )
                {
                    auto t_factor = ( t - start ) / ( ramp - start );
                    current_top = value_top * t_factor;
                    current_bottom = value_bottom * t_factor;
                }
                if ( t > start && t < end )
                {
                    auto pid = index_space( b );
                    if ( x( pid, dim ) > center )
                        f( pid, dim ) += current_top;
                    else
                        f( pid, dim ) += current_bottom;
                }
            } );
    }
};

// FIXME: relatively large initial guess for allocation.
template <class BoundaryType, class BCTag, class ExecSpace, class Particles,
          class... Args>
auto createBoundaryCondition(
    BCTag, ExecSpace exec_space, Particles particles,
    std::vector<BoundaryType> planes, const double value,
    const double start = 0.0,
    const double ramp = std::numeric_limits<double>::max(),
    const double end = std::numeric_limits<double>::max(),
    const double initial_guess = 0.5 )
{
    using memory_space = typename Particles::memory_space;
    using bc_index_type = BoundaryIndexSpace<memory_space, BoundaryType>;
    bc_index_type bc_indices = createBoundaryIndexSpace<BoundaryType>(
        exec_space, particles, planes, initial_guess );
    return BoundaryCondition<bc_index_type, BCTag>( bc_indices, value, start,
                                                    ramp, end );
}

// FIXME: relatively large initial guess for allocation.
template <class BoundaryType, class ExecSpace, class Particles>
auto createBoundaryCondition(
    ForceSymmetric1dBCTag, ExecSpace exec_space, Particles particles,
    std::vector<BoundaryType> planes, const double value, const int dim,
    const double center, const double start = 0.0,
    const double ramp = std::numeric_limits<double>::max(),
    const double end = std::numeric_limits<double>::max(),
    const double initial_guess = 0.5 )
{
    using memory_space = typename Particles::memory_space;
    using bc_index_type = BoundaryIndexSpace<memory_space, BoundaryType>;
    bc_index_type bc_indices = createBoundaryIndexSpace(
        exec_space, particles, planes, initial_guess );
    return BoundaryCondition<bc_index_type, ForceSymmetric1dBCTag>(
        bc_indices, value, dim, center, start, ramp, end );
}

// FIXME: relatively large initial guess for allocation.
template <class BoundaryType, class ExecSpace, class Particles>
auto createBoundaryCondition(
    ForceSymmetric1dBCTag, ExecSpace exec_space, Particles particles,
    std::vector<BoundaryType> planes, const double value_top,
    const double value_bottom, const int dim, const double center,
    const double start = 0.0,
    const double ramp = std::numeric_limits<double>::max(),
    const double end = std::numeric_limits<double>::max(),
    const double initial_guess = 0.5 )
{
    using memory_space = typename Particles::memory_space;
    using bc_index_type = BoundaryIndexSpace<memory_space, BoundaryType>;
    bc_index_type bc_indices = createBoundaryIndexSpace(
        exec_space, particles, planes, initial_guess );
    return BoundaryCondition<bc_index_type, ForceSymmetric1dBCTag>(
        bc_indices, value_top, value_bottom, dim, center, start, ramp, end );
}
} // namespace CabanaPD

#endif
