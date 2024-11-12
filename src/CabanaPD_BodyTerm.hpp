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

#ifndef BODYTERM_H
#define BODYTERM_H

#include <Kokkos_Core.hpp>

#include <Cabana_Core.hpp>

#include <CabanaPD_Timer.hpp>

namespace CabanaPD
{

template <class UserFunctor>
struct BodyTerm
{
    UserFunctor _user_functor;
    bool _force_update;

    Timer _timer;

    BodyTerm( UserFunctor user, const bool force )
        : _user_functor( user )
        , _force_update( force )
    {
    }

    // This function interface purposely matches the boundary conditions in
    // order to use the two interchangeably in Solvers.
    template <class ExecSpace, class ParticleType>
    void apply( ExecSpace, ParticleType& particles, const double time )
    {
        _timer.start();
        Kokkos::RangePolicy<ExecSpace> policy( particles.frozenOffset(),
                                               particles.localOffset() );
        auto user = _user_functor;
        Kokkos::parallel_for(
            "CabanaPD::BodyTerm::apply", policy,
            KOKKOS_LAMBDA( const int p ) { user( p, time ); } );
        _timer.stop();
    }

    auto forceUpdate() { return _force_update; }

    auto time() { return _timer.time(); };
    auto timeInit() { return 0.0; };
};

template <class UserFunctor>
auto createBodyTerm( UserFunctor user_functor, const bool force_update )
{
    return BodyTerm<UserFunctor>( user_functor, force_update );
}

} // namespace CabanaPD

#endif
