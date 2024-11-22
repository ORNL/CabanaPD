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

#ifndef FORCE_H
#define FORCE_H

#include <cmath>

#include <CabanaPD_ForceModels.hpp>
#include <CabanaPD_Particles.hpp>

namespace CabanaPD
{
/******************************************************************************
  Force helper functions.
******************************************************************************/
template <class PosType>
KOKKOS_INLINE_FUNCTION void
getDistanceComponents( const PosType& x, const PosType& u, const int i,
                       const int j, double& xi, double& r, double& s,
                       double& rx, double& ry, double& rz )
{
    // Get the reference positions and displacements.
    const double xi_x = x( j, 0 ) - x( i, 0 );
    const double eta_u = u( j, 0 ) - u( i, 0 );
    const double xi_y = x( j, 1 ) - x( i, 1 );
    const double eta_v = u( j, 1 ) - u( i, 1 );
    const double xi_z = x( j, 2 ) - x( i, 2 );
    const double eta_w = u( j, 2 ) - u( i, 2 );
    rx = xi_x + eta_u;
    ry = xi_y + eta_v;
    rz = xi_z + eta_w;
    r = sqrt( rx * rx + ry * ry + rz * rz );
    xi = sqrt( xi_x * xi_x + xi_y * xi_y + xi_z * xi_z );
    s = ( r - xi ) / xi;
}

template <class PosType>
KOKKOS_INLINE_FUNCTION void getDistance( const PosType& x, const PosType& u,
                                         const int i, const int j, double& xi,
                                         double& r, double& s )
{
    double rx, ry, rz;
    getDistanceComponents( x, u, i, j, xi, r, s, rx, ry, rz );
}

template <class PosType>
KOKKOS_INLINE_FUNCTION void getLinearizedDistanceComponents(
    const PosType& x, const PosType& u, const int i, const int j, double& xi,
    double& s, double& xi_x, double& xi_y, double& xi_z )
{
    // Get the reference positions and displacements.
    xi_x = x( j, 0 ) - x( i, 0 );
    const double eta_u = u( j, 0 ) - u( i, 0 );
    xi_y = x( j, 1 ) - x( i, 1 );
    const double eta_v = u( j, 1 ) - u( i, 1 );
    xi_z = x( j, 2 ) - x( i, 2 );
    const double eta_w = u( j, 2 ) - u( i, 2 );
    xi = sqrt( xi_x * xi_x + xi_y * xi_y + xi_z * xi_z );
    s = ( xi_x * eta_u + xi_y * eta_v + xi_z * eta_w ) / ( xi * xi );
}

template <class PosType>
KOKKOS_INLINE_FUNCTION void
getLinearizedDistance( const PosType& x, const PosType& u, const int i,
                       const int j, double& xi, double& s )
{
    double xi_x, xi_y, xi_z;
    getLinearizedDistanceComponents( x, u, i, j, xi, s, xi_x, xi_y, xi_z );
}

// Forward declaration.
template <class ExecutionSpace, class ForceType>
class Force;

template <class ExecutionSpace>
class Force<ExecutionSpace, BaseForceModel>
{
  protected:
    bool _half_neigh;

    Timer _timer;
    Timer _energy_timer;

  public:
    Force( const bool half_neigh )
        : _half_neigh( half_neigh )
    {
    }

    template <class ParticleType, class NeighListType, class ParallelType>
    void computeWeightedVolume( ParticleType&, const NeighListType&,
                                const ParallelType ) const
    {
    }
    template <class ParticleType, class NeighListType, class ParallelType>
    void computeDilatation( ParticleType&, const NeighListType&,
                            const ParallelType ) const
    {
    }

    auto time() { return _timer.time(); };
    auto timeEnergy() { return _energy_timer.time(); };
};

/******************************************************************************
  Force free functions.
******************************************************************************/
template <class ForceType, class ParticleType, class NeighListType,
          class ParallelType>
void computeForce( ForceType& force, ParticleType& particles,
                   const NeighListType& neigh_list,
                   const ParallelType& neigh_op_tag )
{
    auto n_local = particles.n_local;
    auto x = particles.sliceReferencePosition();
    auto u = particles.sliceDisplacement();
    auto f = particles.sliceForce();
    auto f_a = particles.sliceForceAtomic();

    // Reset force.
    Cabana::deep_copy( f, 0.0 );

    // if ( half_neigh )
    // Forces must be atomic for half list
    // computeForce_half( f_a, x, u, neigh_list, n_local,
    //                    neigh_op_tag );

    // Forces only atomic if using team threading.
    if ( std::is_same<decltype( neigh_op_tag ), Cabana::TeamOpTag>::value )
        force.computeForceFull( f_a, x, u, particles, neigh_list, n_local,
                                neigh_op_tag );
    else
        force.computeForceFull( f, x, u, particles, neigh_list, n_local,
                                neigh_op_tag );
    Kokkos::fence();
}

template <class ForceType, class ParticleType, class NeighListType,
          class ParallelType>
double computeEnergy( ForceType& force, ParticleType& particles,
                      const NeighListType& neigh_list,
                      const ParallelType& neigh_op_tag )
{
    auto n_local = particles.n_local;
    auto x = particles.sliceReferencePosition();
    auto u = particles.sliceDisplacement();
    auto f = particles.sliceForce();
    auto W = particles.sliceStrainEnergy();
    auto vol = particles.sliceVolume();

    // Reset energy.
    Cabana::deep_copy( W, 0.0 );

    double energy;
    // if ( _half_neigh )
    //    energy = computeEnergy_half( force, x, u, neigh_list,
    //                                  n_local, neigh_op_tag );
    // else
    energy = force.computeEnergyFull( W, x, u, particles, neigh_list, n_local,
                                      neigh_op_tag );
    Kokkos::fence();

    return energy;
}

// Forces with bond breaking.
template <class ForceType, class ParticleType, class NeighListType,
          class NeighborView, class ParallelType>
void computeForce( ForceType& force, ParticleType& particles,
                   const NeighListType& neigh_list, NeighborView& mu,
                   const ParallelType& neigh_op_tag, double multiplier )
{
    auto n_local = particles.n_local;
    auto x = particles.sliceReferencePosition();
    auto u = particles.sliceDisplacement();
    auto f = particles.sliceForce();
    auto f_a = particles.sliceForceAtomic();

    // Reset force.
    Cabana::deep_copy( f, 0.0 );

    // if ( half_neigh )
    // Forces must be atomic for half list
    // computeForce_half( f_a, x, u, neigh_list, n_local,
    //                    neigh_op_tag );

    // Forces only atomic if using team threading.
    if ( std::is_same<decltype( neigh_op_tag ), Cabana::TeamOpTag>::value )
        force.computeForceFull( f_a, x, u, particles, neigh_list, mu,
                                neigh_op_tag, n_local, multiplier );
    else
        force.computeForceFull( f, x, u, particles, neigh_list, mu,
                                neigh_op_tag, n_local, multiplier );
    Kokkos::fence();
}

// Energy and damage.
template <class ForceType, class ParticleType, class NeighListType,
          class NeighborView, class ParallelType>
double computeEnergy( ForceType& force, ParticleType& particles,
                      const NeighListType& neigh_list, NeighborView& mu,
                      const ParallelType& neigh_op_tag )
{
    auto n_local = particles.n_local;
    auto x = particles.sliceReferencePosition();
    auto u = particles.sliceDisplacement();
    auto f = particles.sliceForce();
    auto W = particles.sliceStrainEnergy();
    auto phi = particles.sliceDamage();

    // Reset energy.
    Cabana::deep_copy( W, 0.0 );

    double energy;
    // if ( _half_neigh )
    //    energy = computeEnergy_half( force, x, u, neigh_list,
    //                                  n_local, neigh_op_tag );
    // else
    energy = force.computeEnergyFull( W, x, u, phi, particles, neigh_list, mu,
                                      n_local, neigh_op_tag );
    Kokkos::fence();

    return energy;
}

} // namespace CabanaPD

#endif
