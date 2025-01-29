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

#ifndef GEOMETRY_H
#define GEOMETRY_H

#include <Kokkos_Core.hpp>

#include <Cabana_Core.hpp>

#include <CabanaPD_Timer.hpp>

namespace CabanaPD
{

struct RectangularPrism
{
};
struct Cylinder
{
};

// User-specifed custom boundary. Must use the signature:
//    bool operator()(PositionType&, const int)
template <typename UserFunctor>
struct RegionBoundary
{
    UserFunctor _user_functor;

    RegionBoundary( UserFunctor user )
        : _user_functor( user )
    {
    }

    template <class PositionType>
    KOKKOS_INLINE_FUNCTION bool inside( const PositionType& x,
                                        const int pid ) const
    {
        return user( x, pid );
    }
};

// Define a subset of the system as the boundary with a rectangular prism.
template <>
struct RegionBoundary<RectangularPrism>
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
        , high_z( _high_z ) {};

    template <class PositionType>
    KOKKOS_INLINE_FUNCTION bool inside( const PositionType& x,
                                        const int pid ) const
    {
        return ( x( pid, 0 ) >= low_x && x( pid, 0 ) <= high_x &&
                 x( pid, 1 ) >= low_y && x( pid, 1 ) <= high_y &&
                 x( pid, 2 ) >= low_z && x( pid, 2 ) <= high_z );
    }
};

// Define a subset of the system as the boundary with a cylinder.
template <>
struct RegionBoundary<Cylinder>
{
    double radius_in;
    double radius_out;
    double low_z;
    double high_z;
    double x_center;
    double y_center;

    RegionBoundary( const double _radius_in, const double _radius_out,
                    const double _low_z, const double _high_z,
                    double _x_center = 0.0, double _y_center = 0.0 )
        : radius_in( _radius_in )
        , radius_out( _radius_out )
        , low_z( _low_z )
        , high_z( _high_z )
        , x_center( _x_center )
        , y_center( _y_center ) {};

    template <class PositionType>
    KOKKOS_INLINE_FUNCTION bool inside( const PositionType& x,
                                        const int pid ) const
    {
        double rsq = ( x( pid, 0 ) - x_center ) * ( x( pid, 0 ) - x_center ) +
                     ( x( pid, 1 ) - y_center ) * ( x( pid, 1 ) - y_center );
        return ( rsq >= radius_in * radius_in &&
                 rsq <= radius_out * radius_out && x( pid, 2 ) >= low_z &&
                 x( pid, 2 ) <= high_z );
    }
};

// FIXME: fails for some cases if initial guess is not sufficient.
template <class MemorySpace>
struct BoundaryIndexSpace
{
    using index_view_type = Kokkos::View<std::size_t*, MemorySpace>;
    index_view_type _view;
    index_view_type _count;
    std::size_t particle_count;
    // FIXME: expose this as needed.
    const double initial_guess = 0.5;

    Timer _timer;

    // Construct from region (search for boundary particles).
    template <class ExecSpace, class Particles, class... RegionType>
    BoundaryIndexSpace( ExecSpace exec_space, Particles particles,
                        RegionType... regions )
        : particle_count( particles.referenceOffset() )
    {
        _timer.start();

        _view = index_view_type( "boundary_indices",
                                 particles.localOffset() * initial_guess );
        _count = index_view_type( "count", 1 );

        update( exec_space, particles, regions... );

        // Resize after all regions searched.
        auto count_host =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, _count );
        if ( count_host( 0 ) < _view.size() )
        {
            Kokkos::resize( _view, count_host( 0 ) );
        }

        _timer.stop();
    }

    // Iterate over all regions.
    template <class ExecSpace, class Particles, class... RegionType>
    void update( ExecSpace space, Particles particles, RegionType... region )
    {
        ( update( space, particles, region ), ... );
    }

    // Extract indices from a single region.
    template <class ExecSpace, class Particles, class RegionType>
    void update( ExecSpace, Particles particles, RegionType region )
    {
        particle_count = particles.referenceOffset();

        auto count_host =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, _count );
        auto init_count = count_host( 0 );

        auto index_space = _view;
        auto count = _count;
        auto x = particles.sliceReferencePosition();
        // TODO: configure including frozen particles.
        Kokkos::RangePolicy<ExecSpace> policy( 0, particles.localOffset() );
        auto index_functor = KOKKOS_LAMBDA( const std::size_t pid )
        {
            if ( region.inside( x, pid ) )
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
            Kokkos::fence();
        }
    }

    auto time() { return _timer.time(); };
};

template <class MemorySpace>
struct BoundaryIndexSpaceCustom
{
    using index_view_type = Kokkos::View<std::size_t*, MemorySpace>;
    index_view_type _view;

    // Construct from a View of boundary particles.
    BoundaryIndexSpaceCustom( index_view_type input_view )
        : _view( input_view )
    {
    }
};

template <class ExecSpace, class Particles, class... RegionType>
BoundaryIndexSpace( ExecSpace exec_space, Particles particles,
                    RegionType... regions )
    -> BoundaryIndexSpace<typename Particles::memory_space>;

template <class BoundaryParticles>
BoundaryIndexSpace( BoundaryParticles particles )
    -> BoundaryIndexSpace<typename BoundaryParticles::memory_space>;

template <class BoundaryParticles>
BoundaryIndexSpaceCustom( BoundaryParticles boundary_particles )
    -> BoundaryIndexSpaceCustom<typename BoundaryParticles::memory_space>;

} // namespace CabanaPD

#endif
