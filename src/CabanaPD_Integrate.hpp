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

namespace CabanaPD
{

template <class ExecutionSpace>
class Integrator
{
    using exec_space = ExecutionSpace;

    double _dt, _half_dt;

  public:
    Integrator( double dt )
        : _dt( dt )
    {
        _half_dt = 0.5 * dt;
    }

    ~Integrator() {}

    template <class ParticlesType>
    void initial_integrate( ParticlesType& p )
    {
        auto u = p.slice_u();
        auto v = p.slice_v();
        auto f = p.slice_f();
        auto rho = p.slice_rho();

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
        Kokkos::RangePolicy<exec_space> policy( 0, v.size() );
        Kokkos::parallel_for( "CabanaPD::Integrator::Initial", policy,
                              init_func );
    }

    template <class ParticlesType>
    void final_integrate( ParticlesType& p )
    {
        auto v = p.slice_v();
        auto f = p.slice_f();
        auto rho = p.slice_rho();

        auto half_dt = _half_dt;
        auto final_func = KOKKOS_LAMBDA( const int i )
        {
            const double half_dt_m = half_dt / rho( i );
            v( i, 0 ) += half_dt_m * f( i, 0 );
            v( i, 1 ) += half_dt_m * f( i, 1 );
            v( i, 2 ) += half_dt_m * f( i, 2 );
        };
        Kokkos::RangePolicy<exec_space> policy( 0, v.size() );
        Kokkos::parallel_for( "CabanaPD::Integrator::Final", policy,
                              final_func );
    }
};

} // namespace CabanaPD

#endif
