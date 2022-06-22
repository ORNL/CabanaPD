#ifndef PRENOTCH_H
#define PRENOTCH_H

#include <Kokkos_Core.hpp>

#include <Cabana_Core.hpp>
#include <Cajita.hpp>

#include <cassert>
#include <cmath>

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
double norm( Kokkos::Array<double, 3> a ) { return sqrt( dot( a, a ) ); }

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
int line_plane_intersection( const Kokkos::Array<double, 3> p0,
                             const Kokkos::Array<double, 3> n,
                             const Kokkos::Array<double, 3> l0,
                             const Kokkos::Array<double, 3> l,
                             const double tol = 1e-10 )
{
    if ( abs( dot( l, n ) ) < tol )
    {
        if ( abs( dot( diff( p0, l0 ), n ) ) < tol )
            return 1;
        else
            return 2;
    }
    return 3;
}

KOKKOS_INLINE_FUNCTION
int bond_prenotch_intersection( const Kokkos::Array<double, 3> v1,
                                const Kokkos::Array<double, 3> v2,
                                const Kokkos::Array<double, 3> p0,
                                const Kokkos::Array<double, 3> x_i,
                                const Kokkos::Array<double, 3> x_j,
                                const double tol = 1e-10 )
{
    // Define plane vectors cross product
    auto cross_v1_v2 = cross( v1, v2 );
    double norm_cross_v1_v2 = norm( cross_v1_v2 );

    // Check if v1 and v2 are parallel
    assert( abs( norm( cross( v1, v2 ) ) ) > tol );

    // Define plane normal
    auto n = scale( cross_v1_v2, 1.0 / norm_cross_v1_v2 );

    // Define line
    auto l0 = x_i;
    auto l = diff( x_j, x_i );

    // Check line-plane intersection
    int case_flag = line_plane_intersection( p0, n, l0, l );
    int keep_bond = 1;

    // Case I: full intersection
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

        if ( !( fmin( li1, lj1 ) > 1 + tol || fmax( li1, lj1 ) < -tol ||
                fmin( li2, lj2 ) > 1 + tol || fmax( li2, lj2 ) < -tol ) )
            keep_bond = 0;
    }
    // Case II: no intersection
    // Case III: single point intersection
    else if ( case_flag == 3 )
    {
        assert( abs( dot( l, n ) ) > tol );

        // Check if intersection point belongs to the bond
        auto d = dot( diff( p0, l0 ), n ) / dot( l, n );

        // Check if intersection point belongs to the plane
        if ( -tol < d && d < 1 + tol )
        {
            auto p = sum( l0, scale( l, d ) );

            double norm2_cross_v1_v2 = norm_cross_v1_v2 * norm_cross_v1_v2;

            double l1 = dot( cross( diff( p, p0 ), v2 ), cross_v1_v2 ) /
                        norm2_cross_v1_v2;
            double l2 = -dot( cross( diff( p, p0 ), v1 ), cross_v1_v2 ) /
                        norm2_cross_v1_v2;

            if ( -tol < fmin( l1, l2 ) && fmax( l1, l2 ) < 1 + tol )
                // Intersection
                keep_bond = 0;
        }
    }

    return keep_bond;
}

template <std::size_t NumNotch>
struct Prenotch
{
    static constexpr std::size_t num_notch = NumNotch;
    Kokkos::Array<double, 3> _v1;
    Kokkos::Array<double, 3> _v2;
    Kokkos::Array<Kokkos::Array<double, 3>, num_notch> _p0_list;

    Prenotch() {}

    Prenotch( Kokkos::Array<double, 3> v1, Kokkos::Array<double, 3> v2,
              Kokkos::Array<Kokkos::Array<double, 3>, num_notch> p0_list )
        : _v1( v1 )
        , _v2( v2 )
        , _p0_list( p0_list )
    {
    }

    template <class ExecSpace, class NeighborView, class Particles,
              class Neighbors>
    void create( ExecSpace, NeighborView& mu, Particles& particles,
                 Neighbors& neighbors )
    {
        auto x = particles.slice_x();
        Kokkos::RangePolicy<ExecSpace> policy( 0, particles.n_local );

        auto v1 = _v1;
        auto v2 = _v2;
        for ( std::size_t p = 0; p < _p0_list.size(); p++ )
        {
            auto p0 = _p0_list[p];
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
                        bond_prenotch_intersection( v1, v2, p0, xi, xj );
                    if ( !keep_bond )
                        mu( i, n ) = 0;
                }
            };
            Kokkos::parallel_for( policy, notch_functor, "CabanaPD::Prenotch" );
        }
    }
};

} // namespace CabanaPD

#endif
