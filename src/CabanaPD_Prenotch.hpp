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

#ifndef PRENOTCH_H
#define PRENOTCH_H

#include <Kokkos_Core.hpp>

#include <Cabana_Core.hpp>

#include <cassert>

namespace CabanaPD
{

KOKKOS_INLINE_FUNCTION
double dot( Kokkos::Array<double, 3> a, Kokkos::Array<double, 3> b )
{
    double v = 0;
    for ( std::size_t d = 0; d < a.size(); d++ )
        v += ( a[d] ) * ( b[d] );
    return v;
}

KOKKOS_INLINE_FUNCTION
double norm( Kokkos::Array<double, 3> a )
{
    return Kokkos::sqrt( dot( a, a ) );
}

KOKKOS_INLINE_FUNCTION
Kokkos::Array<double, 3> cross( Kokkos::Array<double, 3> a,
                                Kokkos::Array<double, 3> b )
{
    Kokkos::Array<double, 3> c;
    c[0] = a[1] * b[2] - a[2] * b[1];
    c[1] = -( a[0] * b[2] - a[2] * b[0] );
    c[2] = a[0] * b[1] - a[1] * b[0];
    return c;
}

KOKKOS_INLINE_FUNCTION
Kokkos::Array<double, 3> scale( Kokkos::Array<double, 3> a, const double n )
{
    for ( std::size_t d = 0; d < a.size(); d++ )
        a[d] *= n;
    return a;
}

KOKKOS_INLINE_FUNCTION
Kokkos::Array<double, 3> diff( Kokkos::Array<double, 3> a,
                               Kokkos::Array<double, 3> b )
{
    Kokkos::Array<double, 3> c;
    for ( std::size_t d = 0; d < a.size(); d++ )
        c[d] = a[d] - b[d];
    return c;
}

KOKKOS_INLINE_FUNCTION
Kokkos::Array<double, 3> sum( Kokkos::Array<double, 3> a,
                              Kokkos::Array<double, 3> b )
{
    Kokkos::Array<double, 3> c;
    for ( std::size_t d = 0; d < a.size(); d++ )
        c[d] = a[d] + b[d];
    return c;
}

KOKKOS_INLINE_FUNCTION
int linePlaneIntersection( const Kokkos::Array<double, 3> p0,
                           const Kokkos::Array<double, 3> n,
                           const Kokkos::Array<double, 3> l0,
                           const Kokkos::Array<double, 3> l,
                           const double tol = 1e-10 )
{
    if ( Kokkos::abs( dot( l, n ) ) < tol )
    {
        if ( Kokkos::abs( dot( diff( p0, l0 ), n ) ) < tol )
            return 1;
        else
            return 2;
    }
    return 3;
}

KOKKOS_INLINE_FUNCTION
int bondPrenotchIntersection( const Kokkos::Array<double, 3> v1,
                              const Kokkos::Array<double, 3> v2,
                              const Kokkos::Array<double, 3> p0,
                              const Kokkos::Array<double, 3> x_i,
                              const Kokkos::Array<double, 3> x_j,
                              const double tol = 1e-10 )
{
    // Define plane vectors cross product.
    auto cross_v1_v2 = cross( v1, v2 );
    double norm_cross_v1_v2 = norm( cross_v1_v2 );

    // Check if v1 and v2 are parallel.
    assert( Kokkos::abs( norm( cross( v1, v2 ) ) ) > tol );

    // Define plane normal.
    auto n = scale( cross_v1_v2, 1.0 / norm_cross_v1_v2 );

    // Define line.
    auto l0 = x_i;
    auto l = diff( x_j, x_i );

    // Check line-plane intersection.
    int case_flag = linePlaneIntersection( p0, n, l0, l );
    int keep_bond = 1;

    // Case I: full intersection.
    if ( case_flag == 1 )
    {
        double norm2_cross_v1_v2 = norm_cross_v1_v2 * norm_cross_v1_v2;

        double li1 = dot( cross( diff( x_i, p0 ), v2 ), cross_v1_v2 ) /
                     norm2_cross_v1_v2;
        double li2 = -dot( cross( diff( x_i, p0 ), v1 ), cross_v1_v2 ) /
                     norm2_cross_v1_v2;

        double lj1 = dot( cross( diff( x_j, p0 ), v2 ), cross_v1_v2 ) /
                     norm2_cross_v1_v2;
        double lj2 = -dot( cross( diff( x_j, p0 ), v1 ), cross_v1_v2 ) /
                     norm2_cross_v1_v2;

        if ( !( Kokkos::fmin( li1, lj1 ) > 1 + tol ||
                Kokkos::fmax( li1, lj1 ) < -tol ||
                Kokkos::fmin( li2, lj2 ) > 1 + tol ||
                Kokkos::fmax( li2, lj2 ) < -tol ) )
            keep_bond = 0;
    }
    // Case II: no intersection.

    // Case III: single point intersection.
    else if ( case_flag == 3 )
    {
        assert( Kokkos::abs( dot( l, n ) ) > tol );

        // Check if intersection point belongs to the bond.
        auto d = dot( diff( p0, l0 ), n ) / dot( l, n );

        if ( -tol < d && d < 1 + tol )
        {
            // Check if intersection point belongs to the plane.
            auto p = sum( l0, scale( l, d ) );

            double norm2_cross_v1_v2 = norm_cross_v1_v2 * norm_cross_v1_v2;

            double l1 = dot( cross( diff( p, p0 ), v2 ), cross_v1_v2 ) /
                        norm2_cross_v1_v2;
            double l2 = -dot( cross( diff( p, p0 ), v1 ), cross_v1_v2 ) /
                        norm2_cross_v1_v2;

            if ( -tol < Kokkos::fmin( l1, l2 ) &&
                 Kokkos::fmax( l1, l2 ) < 1 + tol )
                // Intersection.
                keep_bond = 0;
        }
    }

    return keep_bond;
}

template <std::size_t NumNotch, std::size_t NumVector = 1>
struct Prenotch
{
    static constexpr std::size_t num_notch = NumNotch;
    static constexpr std::size_t num_vector = NumNotch;
    Kokkos::Array<Kokkos::Array<double, 3>, num_vector> _v1;
    Kokkos::Array<Kokkos::Array<double, 3>, num_vector> _v2;
    Kokkos::Array<Kokkos::Array<double, 3>, num_notch> _p0;
    bool fixed_orientation;

    Timer _timer;

    // Default constructor
    Prenotch() {}

    // Constructor if all pre-notches are oriented the same way (e.g.
    // Kalthoff-Winkler).
    Prenotch( Kokkos::Array<double, 3> v1, Kokkos::Array<double, 3> v2,
              Kokkos::Array<Kokkos::Array<double, 3>, num_notch> p0 )
        : _v1( { v1 } )
        , _v2( { v2 } )
        , _p0( p0 )
    {
        fixed_orientation = true;
    }

    // Constructor for general case of any orientation for any number of
    // pre-notches.
    Prenotch( Kokkos::Array<Kokkos::Array<double, 3>, num_vector> v1,
              Kokkos::Array<Kokkos::Array<double, 3>, num_vector> v2,
              Kokkos::Array<Kokkos::Array<double, 3>, num_notch> p0 )
        : _v1( v1 )
        , _v2( v2 )
        , _p0( p0 )
    {
        static_assert(
            num_vector == num_notch,
            "Number of orientation vectors must match number of pre-notches." );
        fixed_orientation = false;
    }

    template <class ExecSpace, class NeighborView, class Particles,
              class Neighbors>
    void create( ExecSpace, NeighborView& mu, Particles& particles,
                 Neighbors& neighbors )
    {
        _timer.start();

        auto x = particles.sliceReferencePosition();
        // TODO: decide whether to disallow prenotches in frozen particles.
        Kokkos::RangePolicy<ExecSpace> policy( 0, particles.localOffset() );

        for ( std::size_t p = 0; p < _p0.size(); p++ )
        {
            // These will always be different positions.
            auto p0 = _p0[p];
            // These may all have the same orientation or all different
            // orientations.
            auto v1 = getV1( p );
            auto v2 = getV2( p );
            auto notch_functor = KOKKOS_LAMBDA( const int i )
            {
                std::size_t num_neighbors =
                    Cabana::NeighborList<Neighbors>::numNeighbor( neighbors,
                                                                  i );
                Kokkos::Array<double, 3> xi;
                Kokkos::Array<double, 3> xj;
                for ( std::size_t n = 0; n < num_neighbors; n++ )
                {
                    for ( std::size_t d = 0; d < 3; d++ )
                    {
                        xi[d] = x( i, d );
                        std::size_t j =
                            Cabana::NeighborList<Neighbors>::getNeighbor(
                                neighbors, i, n );
                        xj[d] = x( j, d );
                    }
                    int keep_bond =
                        bondPrenotchIntersection( v1, v2, p0, xi, xj );
                    if ( !keep_bond )
                        mu( i, n ) = 0;
                }
            };
            Kokkos::parallel_for( "CabanaPD::Prenotch", policy, notch_functor );
        }
        // outside of for loop since each prenotch can run simultaneously,
        // breaking a bond multiple times without conflict
        Kokkos::fence();
        _timer.stop();
    }
    auto time() { return _timer.time(); };

    auto getV1( const int p )
    {
        if ( fixed_orientation )
            return _v1[0];
        else
            return _v1[p];
    }

    auto getV2( const int p )
    {
        if ( fixed_orientation )
            return _v2[0];
        else
            return _v2[p];
    }
};

} // namespace CabanaPD

#endif
