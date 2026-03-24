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
struct PolynomialProperty
{
    // Constructor with polynomial coefficients and temperature field.
    template <typename ArrayType>
    PolynomialProperty( ArrayType array, const TemperatureType& t )
        : coeff( array.data() )
        , temp( t )
    {
    }

    // Return temperature-dependent value.
    KOKKOS_FUNCTION
    auto operator()( const int p ) const
    {
        return coeff( 0 ) + coeff( 1 ) * temp( p ) +
               coeff( 2 ) * temp( p ) * temp( p );
    }

    // Update with new temperature field.
    template <typename ParticleType>
    void update( const ParticleType& particles )
    {
        temp = particles.sliceTemperature();
    }

    Kokkos::View<double*, typename TemperatureType::memory_space> coeff;
    TemperatureType temp;
};

template <typename ArrayType, typename TemperatureType>
PolynomialProperty( const ArrayType, const TemperatureType )
    -> PolynomialProperty<TemperatureType>;

} // namespace CabanaPD

#endif
