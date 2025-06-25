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
struct Line
{
};

// User-specifed custom boundary. Must use the signature:
//    bool operator()(PositionType&, const int)
template <typename UserFunctor>
struct Region
{
    UserFunctor _user_functor;

    Region( UserFunctor user )
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

// Define a subset of the system with a rectangular prism.
template <>
struct Region<RectangularPrism>
{
    double low_x;
    double high_x;
    double low_y;
    double high_y;
    double low_z;
    double high_z;

    Region( const double _low_x, const double _high_x, const double _low_y,
            const double _high_y, const double _low_z, const double _high_z )
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
    }

    template <class ArrayType>
    Region( const ArrayType _low, const ArrayType _high )
        : low_x( _low[0] )
        , high_x( _high[0] )
        , low_y( _low[1] )
        , high_y( _high[1] )
        , low_z( _low[2] )
        , high_z( _high[2] )
    {
        assert( low_x < high_x );
        assert( low_y < high_y );
        assert( low_z < high_z );
    }

    template <class PositionType>
    KOKKOS_INLINE_FUNCTION bool inside( const PositionType& x,
                                        const int pid ) const
    {
        return ( x( pid, 0 ) >= low_x && x( pid, 0 ) <= high_x &&
                 x( pid, 1 ) >= low_y && x( pid, 1 ) <= high_y &&
                 x( pid, 2 ) >= low_z && x( pid, 2 ) <= high_z );
    }
};

// Define a subset of the system with a cylinder.
template <>
struct Region<Cylinder>
{
    double radius_in;
    double radius_out;
    double low_z;
    double high_z;
    double x_center;
    double y_center;

    Region( const double _radius_in, const double _radius_out,
            const double _low_z, const double _high_z, double _x_center = 0.0,
            double _y_center = 0.0 )
        : radius_in( _radius_in )
        , radius_out( _radius_out )
        , low_z( _low_z )
        , high_z( _high_z )
        , x_center( _x_center )
        , y_center( _y_center )
    {
        assert( radius_in < _radius_out );
        assert( low_z < high_z );
    }

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

// Define a subset of the system with a line at the center.
// This is intended to be grid aligned, not a general line within the system.
template <>
struct Region<Line>
{
    Kokkos::Array<double, 2> _half_dx;
    Kokkos::Array<int, 2> _dims;

    template <typename ArrayType>
    Region( const double profile_dim, const ArrayType dx )
    {
        _dims = getOtherDims( profile_dim );
        _half_dx[0] = dx[_dims[0]] / 2.0;
        _half_dx[1] = dx[_dims[1]] / 2.0;
    }

    // Given a dimension, returns the other two
    Kokkos::Array<int, 2> getOtherDims( const int x )
    {
        Kokkos::Array<int, 2> yz;
        yz[0] = ( x + 1 ) % 3;
        yz[1] = ( x + 2 ) % 3;
        return yz;
    }

    template <class PositionType>
    KOKKOS_INLINE_FUNCTION bool inside( const PositionType& x,
                                        const int pid ) const
    {
        return x( pid, _dims[0] ) < _half_dx[0] &&
               x( pid, _dims[0] ) > -_half_dx[0] &&
               x( pid, _dims[1] ) < _half_dx[1] &&
               x( pid, _dims[1] ) > -_half_dx[1];
    }
};

// FIXME: fails for some cases if initial guess is not sufficient.
template <class MemorySpace>
struct ParticleSteeringVector
{
    using memory_space = MemorySpace;
    using index_view_type = Kokkos::View<std::size_t*, MemorySpace>;
    index_view_type _view;
    index_view_type _count;
    std::size_t particle_count;
    // Could expose this parameter as needed.
    const double initial_guess = 0.1;

    Timer _timer;

    // Construct from region (search for boundary particles).
    template <class ExecSpace, class Particles, class... RegionType>
    ParticleSteeringVector( ExecSpace exec_space, Particles particles,
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

    // Construct from a View of boundary particles (custom).
    ParticleSteeringVector( index_view_type input_view )
        : _view( input_view )
    {
    }

    // Iterate over all regions.
    template <class ExecSpace, class Particles, class... RegionType>
    void update( ExecSpace space, Particles particles, RegionType... region )
    {
        ( update( space, particles, region ), ... );
    }

    // Extract indices from a single region.
    template <class ExecSpace, class Particles, class RegionType>
    void update( ExecSpace exec_space, Particles particles, RegionType region )
    {
        particle_count = particles.referenceOffset();

        auto count_host =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, _count );
        auto init_count = count_host( 0 );

        count( exec_space, particles, region );
        // Fence before deep_copy.
        Kokkos::deep_copy( count_host, _count );

        if ( count_host( 0 ) > _view.size() )
        {
            Kokkos::resize( _view, count_host( 0 ) );
            Kokkos::deep_copy( _count, init_count );
            count( exec_space, particles, region );
            Kokkos::fence();
        }
    }

    template <class ExecSpace, class Particles, class RegionType>
    void count( ExecSpace, Particles particles, RegionType region )
    {
        auto count_copy = _count;
        auto index_space = _view;
        auto x = particles.sliceReferencePosition();
        // TODO: configure including frozen particles.
        Kokkos::RangePolicy<ExecSpace> policy( 0, particles.localOffset() );
        auto index_functor = KOKKOS_LAMBDA( const std::size_t pid )
        {
            if ( region.inside( x, pid ) )
            {
                // Resize after count if needed.
                auto c = Kokkos::atomic_fetch_add( &count_copy( 0 ), 1 );
                if ( c < index_space.size() )
                {
                    index_space( c ) = pid;
                }
            }
        };
        Kokkos::parallel_for( "CabanaPD::BC::update", policy, index_functor );
    }

    // Update from a View of boundary particles (custom).
    void update( index_view_type input_view ) { _view = input_view; }

    auto size() { return _view.size(); }

    auto time() { return _timer.time(); }
};

template <class ExecSpace, class Particles, class... RegionType>
ParticleSteeringVector( ExecSpace exec_space, Particles particles,
                        RegionType... regions )
    -> ParticleSteeringVector<typename Particles::memory_space>;

template <class BoundaryParticles>
ParticleSteeringVector( BoundaryParticles particles )
    -> ParticleSteeringVector<typename BoundaryParticles::memory_space>;

} // namespace CabanaPD

#endif
