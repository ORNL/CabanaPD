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

#ifndef PROPERTIES_H
#define PROPERTIES_H

#include <Kokkos_Core.hpp>

namespace CabanaPD
{
template <typename FunctorType>
struct Property;

template <typename FunctorType1, typename FunctorType2>
struct AverageProperty;

template <typename FunctorType>
struct Property
{
    Property( const FunctorType p )
        : prop( p )
    {
    }

    template <typename PropertyType1, typename PropertyType2>
    Property( const PropertyType1 p1, const PropertyType2 p2 )
        : prop( AverageProperty( p1, p2 ) )
    {
    }

    KOKKOS_FUNCTION
    auto operator()( const int p ) const { return prop( p ); }

    template <typename ParticleType>
    void update( const ParticleType& particles )
    {
        prop.update( particles );
    }

    FunctorType prop;
};

template <typename FunctorType1, typename FunctorType2>
struct AverageProperty : public Property<FunctorType1>
{
    using base_type = Property<FunctorType1>;

    AverageProperty( const Property<FunctorType1> f1,
                     const Property<FunctorType2> f2 )
        : base_type( f1.prop )
        , prop2( f2.prop )
    {
    }

    KOKKOS_FUNCTION
    auto operator()( const int p ) const
    {
        return prop( p ) + prop2( p ) / 2.0;
    }

    template <typename ParticleType>
    void update( const ParticleType& particles )
    {
        prop.update( particles );
        prop2.update( particles );
    }

    using base_type::prop;
    FunctorType2 prop2;
};

struct ConstantProperty
{
    ConstantProperty( const double v )
        : value( v )
    {
    }

    KOKKOS_FUNCTION
    auto operator()( const int ) const { return value; }

    template <typename ParticleType>
    void update( const ParticleType& )
    {
    }

    double value;
};

template <typename ParticleField>
struct PropertyDependence
{
    // Constructor with temperature field.
    PropertyDependence( const ParticleField& f )
        : field( f )
    {
    }

    // Update with new temperature field.

    ParticleField field;
};

template <typename ParticleField>
PropertyDependence( const ParticleField ) -> PropertyDependence<ParticleField>;

template <typename ParticleField>
struct PolynomialDependence : public PropertyDependence<ParticleField>
{
    using base_type = PropertyDependence<ParticleField>;

    // Constructor with polynomial coefficients and temperature field.
    template <typename ArrayType>
    PolynomialDependence( ArrayType array, const ParticleField& f )
        : base_type( f )
        , coeff( array.data() )
    {
    }

    // Return temperature-dependent value.
    KOKKOS_FUNCTION
    auto operator()( const int p ) const
    {
        double val = 0.0;
        const auto f = field( p );
        for ( std::size_t i = 0; i < coeff.size() - 1; i++ )
            val += coeff( i ) * f;

        return val + coeff( 0 );
    }

    Kokkos::View<double*, typename ParticleField::memory_space> coeff;
    using base_type::field;
};

template <typename ArrayType, typename ParticleField>
PolynomialDependence( ArrayType, const ParticleField )
    -> PolynomialDependence<ParticleField>;

template <typename ParticleField>
struct PiecewiseDependence : public PropertyDependence<ParticleField>
{
    using base_type = PropertyDependence<ParticleField>;

    // Constructor with XY values to linearly interpolate and temperature field.
    template <typename ArrayType>
    PiecewiseDependence( ArrayType array_x, ArrayType array_y,
                         const ParticleField& f )
        : base_type( f )
        , x( array_x.data() )
        , y( array_y.data() )
    {
        assert( x.size() == y.size() );
    }

    // Return temperature-dependent value.
    KOKKOS_FUNCTION
    auto operator()( const int p ) const
    {
        const auto f = field( p );
        if ( f < x( 0 ) )
            return y( 0 );

        for ( std::size_t i = 1; i < x.size() - 1; i++ )
        {
            if ( f < x( i ) )
                return ( y( i - 1 ) - y( i ) ) / ( x( i - 1 ) - x( i ) );
        }

        return y( y.size() - 1 );
    }

    Kokkos::View<double*, typename ParticleField::memory_space> x;
    Kokkos::View<double*, typename ParticleField::memory_space> y;
    using base_type::field;
};

template <typename ArrayType, typename ParticleField>
PiecewiseDependence( ArrayType, ArrayType, const ParticleField )
    -> PiecewiseDependence<ParticleField>;

} // namespace CabanaPD

#endif
