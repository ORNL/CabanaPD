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

#include <CabanaPD_Boundary.hpp>
#include <CabanaPD_Output.hpp>
#include <CabanaPD_Timer.hpp>

namespace CabanaPD
{

template <class UserFunctor>
struct BodyTerm
{
    UserFunctor _user_functor;
    std::size_t _particle_count;
    bool _force_update;
    bool _update_frozen;

    Timer _timer;

    BodyTerm( UserFunctor user, const std::size_t particle_count,
              const bool force, const bool update_frozen = false )
        : _user_functor( user )
        , _particle_count( particle_count )
        , _force_update( force )
        , _update_frozen( update_frozen )
    {
    }

    // This function interface purposely matches the boundary conditions in
    // order to use the two interchangeably in Solvers.
    template <class ExecSpace, class ParticleType>
    void apply( ExecSpace, ParticleType& particles, const double time )
    {
        checkParticleCount( _particle_count, particles.referenceOffset(),
                            "BodyTerm" );

        _timer.start();
        std::size_t start = particles.frozenOffset();
        if ( _update_frozen )
            start = 0;
        Kokkos::RangePolicy<ExecSpace> policy( start, particles.localOffset() );
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

} // namespace CabanaPD

#endif
