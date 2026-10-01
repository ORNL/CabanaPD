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

namespace CabanaPD
{
struct ConstantProperty
{
    ConstantProperty( const double v )
        : value( v )
    {
    }
    template <typename FunctorType>
    ConstantProperty( const FunctorType p1, const FunctorType p2 )
        : value( ( p1( 0 ) + p2( 0 ) / 2.0 ) )
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

template <typename TemperatureType>
struct TemperatureDependentProperty
{
    // Constructor with temperature field.
    TemperatureDependentProperty( const TemperatureType& t )
        : temp( t )
    {
    }

    // Update with new temperature field.
    template <typename ParticleType>
    void update( const ParticleType& particles )
    {
        temp = particles.sliceTemperature();
    }

    TemperatureType temp;
};

template <typename TemperatureType>
TemperatureDependentProperty( const TemperatureType )
    -> TemperatureDependentProperty<TemperatureType>;

template <typename TemperatureType>
struct TemperatureDependentPolynomial
    : public TemperatureDependentProperty<TemperatureType>
{
    using base_type = TemperatureDependentProperty<TemperatureType>;

    // Constructor with polynomial coefficients and temperature field.
    template <typename ArrayType>
    TemperatureDependentPolynomial( ArrayType array, const TemperatureType& t )
        : base_type( t )
        , coeff( array.data() )
    {
    }

    // Return temperature-dependent value.
    KOKKOS_FUNCTION
    auto operator()( const int p ) const
    {
        double val = 0.0;
        const auto t = temp( p );
        for ( std::size_t i = 0; i < coeff.size() - 1; i++ )
            val += coeff( i ) * t;

        return val + coeff( 0 );
    }

    Kokkos::View<double*, typename TemperatureType::memory_space> coeff;
    using base_type::temp;
};

template <typename ArrayType, typename TemperatureType>
TemperatureDependentPolynomial( ArrayType, const TemperatureType )
    -> TemperatureDependentPolynomial<TemperatureType>;

template <typename TemperatureType>
struct TemperatureDependentPiecewise
    : public TemperatureDependentProperty<TemperatureType>
{
    using base_type = TemperatureDependentProperty<TemperatureType>;

    // Constructor with XY values to linearly interpolate and temperature field.
    template <typename ArrayType>
    TemperatureDependentPiecewise( ArrayType array_x, ArrayType array_y,
                                   const TemperatureType& t )
        : base_type( t )
        , x( array_x.data() )
        , y( array_y.data() )
    {
        assert( x.size() == y.size() );
    }

    // Return temperature-dependent value.
    KOKKOS_FUNCTION
    auto operator()( const int p ) const
    {
        const auto t = temp( p );
        if ( t < x( 0 ) )
            return y( 0 );

        for ( std::size_t i = 1; i < x.size() - 1; i++ )
        {
            if ( t < x( i ) )
                return ( y( i - 1 ) - y( i ) ) / ( x( i - 1 ) - x( i ) );
        }

        return y( y.size() - 1 );
    }

    Kokkos::View<double*, typename TemperatureType::memory_space> x;
    Kokkos::View<double*, typename TemperatureType::memory_space> y;
    using base_type::temp;
};

template <typename ArrayType, typename TemperatureType>
TemperatureDependentPiecewise( ArrayType, ArrayType, const TemperatureType )
    -> TemperatureDependentPiecewise<TemperatureType>;

} // namespace CabanaPD

#endif
