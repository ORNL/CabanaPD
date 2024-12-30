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

/****************************************************************************
 * Copyright (c) 2018-2021 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

//************************************************************************
//  ExaMiniMD v. 1.0
//  Copyright (2018) National Technology & Engineering Solutions of Sandia,
//  LLC (NTESS).
//
//  Under the terms of Contract DE-NA-0003525 with NTESS, the U.S. Government
//  retains certain rights in this software.
//
//  ExaMiniMD is licensed under 3-clause BSD terms of use: Redistribution and
//  use in source and binary forms, with or without modification, are
//  permitted provided that the following conditions are met:
//
//    1. Redistributions of source code must retain the above copyright notice,
//       this list of conditions and the following disclaimer.
//
//    2. Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//    3. Neither the name of the Corporation nor the names of the contributors
//       may be used to endorse or promote products derived from this software
//       without specific prior written permission.
//
//  THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY EXPRESS OR IMPLIED
//  WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
//  MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
//  IN NO EVENT SHALL NTESS OR THE CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
//  INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
//  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
//  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
//  HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
//  STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
//  IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
//  POSSIBILITY OF SUCH DAMAGE.
//
//************************************************************************

#ifndef INTEGRATOR_H
#define INTEGRATOR_H

#include <Kokkos_Core.hpp>

#include <CabanaPD_Particles.hpp>
#include <CabanaPD_Timer.hpp>

namespace CabanaPD
{
template <class ExecutionSpace>
class Integrator
{
    using exec_space = ExecutionSpace;

    double _dt, _half_dt;
    Timer _timer;

  public:
    Integrator( double dt )
        : _dt( dt )
    {
        _half_dt = 0.5 * dt;
    }

    ~Integrator() {}

    template <class ParticlesType>
    void initialHalfStep( ParticlesType& p )
    {
        _timer.start();

        auto u = p.sliceDisplacement();
        auto v = p.sliceVelocity();
        auto f = p.sliceForce();
        auto rho = p.sliceDensity();

        auto dt = _dt;
        auto half_dt = _half_dt;
        auto init_func = KOKKOS_LAMBDA( const int i )
        {
            const double half_dt_m = half_dt / rho( i );
            v( i, 0 ) += half_dt_m * f( i, 0 );
            v( i, 1 ) += half_dt_m * f( i, 1 );
            v( i, 2 ) += half_dt_m * f( i, 2 );
            u( i, 0 ) += dt * v( i, 0 );
            u( i, 1 ) += dt * v( i, 1 );
            u( i, 2 ) += dt * v( i, 2 );
        };
        Kokkos::RangePolicy<exec_space> policy( p.frozenOffset(),
                                                p.localOffset() );
        Kokkos::parallel_for( "CabanaPD::Integrator::Initial", policy,
                              init_func );

        _timer.stop();
    }

    template <class ParticlesType>
    void finalHalfStep( ParticlesType& p )
    {
        _timer.start();

        auto v = p.sliceVelocity();
        auto f = p.sliceForce();
        auto rho = p.sliceDensity();

        auto half_dt = _half_dt;
        auto final_func = KOKKOS_LAMBDA( const int i )
        {
            const double half_dt_m = half_dt / rho( i );
            v( i, 0 ) += half_dt_m * f( i, 0 );
            v( i, 1 ) += half_dt_m * f( i, 1 );
            v( i, 2 ) += half_dt_m * f( i, 2 );
        };
        Kokkos::RangePolicy<exec_space> policy( p.frozenOffset(),
                                                p.localOffset() );
        Kokkos::parallel_for( "CabanaPD::Integrator::Final", policy,
                              final_func );

        _timer.stop();
    }

    double timeInit() { return 0.0; };
    auto time() { return _timer.time(); };
};

} // namespace CabanaPD

#endif
