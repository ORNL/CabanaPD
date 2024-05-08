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

#ifndef BODYTERM_H
#define BODYTERM_H

#include <Kokkos_Core.hpp>

#include <Cabana_Core.hpp>

namespace CabanaPD
{

template <class UserFunctor>
struct BodyTerm
{
    UserFunctor _user_functor;

    BodyTerm( UserFunctor user )
        : _user_functor( user )
    {
    }

    // This function interface purposely matches the boundary conditions in
    // order to use the two interchangeably in Solvers.
    template <class ExecSpace, class ParticleType>
    void apply( ExecSpace, ParticleType& particles, const double time )
    {
        Kokkos::RangePolicy<ExecSpace> policy( 0, particles.n_local );
        auto user = _user_functor;
        Kokkos::parallel_for(
            "CabanaPD::BodyTerm::apply", policy,
            KOKKOS_LAMBDA( const int p ) { user( p, time ); } );
    }
};

template <class UserFunctor>
auto createBodyTerm( UserFunctor user_functor )
{
    return BodyTerm<UserFunctor>( user_functor );
}

} // namespace CabanaPD

#endif
