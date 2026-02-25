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

#endif
