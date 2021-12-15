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

#include <CabanaPD_Output.hpp>

namespace CabanaPD
{

struct PMBModelTag
{
};

struct LinearPMBModelTag
{
};

template <class ExecutionSpace>
class Force
{
  protected:
    bool _half_neigh;
    double _c;
    double _mu;

  public:
    using exec_space = ExecutionSpace;

    Force( const bool half_neigh, const double K, const double delta )
        : _half_neigh( half_neigh )
    {
        _c = 18.0 * K / ( 3.141592653589793 * delta * delta * delta * delta );
        _mu = 1.0;
    }

    template <class ModelTag, class ParticleType, class NeighListType,
              class ParallelType>
    void compute( const ModelTag model_tag, ParticleType &particles,
                  const NeighListType &neigh_list,
                  const ParallelType &neigh_op_tag )
    {
        auto n_local = particles.n_local;
        auto x = particles.slice_x();
        auto u = particles.slice_u();
        auto f = particles.slice_f();
        auto f_a = particles.slice_f_a();
        auto vol = particles.slice_vol();

        // if ( half_neigh )
        // Forces must be atomic for half list
        // compute_force_half( f_a, x, u, neigh_list, n_local,
        //                    neigh_op_tag );

        // Forces only atomic if using team threading
        if ( std::is_same<decltype( neigh_op_tag ), Cabana::TeamOpTag>::value )
            compute_force_full( model_tag, f_a, x, u, vol, neigh_list, n_local,
                                neigh_op_tag );
        else
            compute_force_full( model_tag, f, x, u, vol, neigh_list, n_local,
                                neigh_op_tag );
        Kokkos::fence();
    }

    template <class ModelTag, class ParticleType, class NeighListType,
              class ParallelType>
    double compute_energy( const ModelTag model_tag, ParticleType &particles,
                           const NeighListType &neigh_list,
                           const ParallelType &neigh_op_tag )
    {
        auto n_local = particles.n_local;
        auto x = particles.slice_x();
        auto u = particles.slice_u();
        auto f = particles.slice_f();
        auto W = particles.slice_W();
        auto vol = particles.slice_vol();

        double energy;
        // if ( _half_neigh )
        //    energy = compute_energy_half( model_tag, x, u, neigh_list,
        //                                  n_local, neigh_op_tag );
        // else
        energy = compute_energy_full( model_tag, W, x, u, vol, neigh_list,
                                      n_local, neigh_op_tag );
        Kokkos::fence();

        return energy;
    }

    template <class ForceType, class PosType, class VolType,
              class NeighListType, class ParallelType>
    void compute_force_full( const PMBModelTag, ForceType &f, const PosType &x,
                             const PosType &u, const VolType &vol,
                             const NeighListType &neigh_list, const int n_local,
                             ParallelType &neigh_op_tag )
    {
        auto c = _c;
        // FIXME: will be bond-based
        auto mu = _mu;

        // Cabana::NeighborList<NeighListType> nlist;
        auto force_full = KOKKOS_LAMBDA( const int i, const int j )
        {
            double fx_i = 0.0;
            double fy_i = 0.0;
            double fz_i = 0.0;

            // std::cout << nlist.numNeighbor( neigh_list, i ) << std::endl;

            // Get the reference positions and displacements.
            const double xi_x = x( i, 0 ) - x( j, 0 );
            const double eta_u = u( i, 0 ) - u( j, 0 );
            const double xi_y = x( i, 1 ) - x( j, 1 );
            const double eta_v = u( i, 1 ) - u( j, 1 );
            const double xi_z = x( i, 2 ) - x( j, 2 );
            const double eta_w = u( i, 2 ) - u( j, 2 );
            const double rx = xi_x + eta_u;
            const double ry = xi_y + eta_v;
            const double rz = xi_z + eta_w;
            const double r = sqrt( rx * rx + ry * ry + rz * rz );
            const double xi = sqrt( xi_x * xi_x + xi_y * xi_y + xi_z * xi_z );
            const double s = ( r - xi ) / xi;
            const double coeff = mu * c * s * vol( i );
            fx_i = coeff * rx / r;
            fy_i = coeff * ry / r;
            fz_i = coeff * rz / r;

            f( i, 0 ) += fx_i;
            f( i, 1 ) += fy_i;
            f( i, 2 ) += fz_i;
        };

        Kokkos::RangePolicy<exec_space> policy( 0, n_local );
        Cabana::neighbor_parallel_for(
            policy, force_full, neigh_list, Cabana::FirstNeighborsTag(),
            neigh_op_tag, "CabanaPD::ForcePMB::compute_full" );
    }

    template <class ForceType, class PosType, class VolType,
              class NeighListType, class ParallelType>
    void compute_force_full( const LinearPMBModelTag, ForceType &f,
                             const PosType &x, const PosType &u,
                             const VolType &vol,
                             const NeighListType &neigh_list, const int n_local,
                             ParallelType &neigh_op_tag )
    {
        auto c = _c;
        // FIXME: will be bond-based
        auto mu = _mu;

        // Cabana::NeighborList<NeighListType> nlist;
        auto force_full = KOKKOS_LAMBDA( const int i, const int j )
        {
            double fx_i = 0.0;
            double fy_i = 0.0;
            double fz_i = 0.0;

            // std::cout << nlist.numNeighbor( neigh_list, i ) << std::endl;

            // Get the reference positions and displacements.
            const double xi_x = x( i, 0 ) - x( j, 0 );
            const double eta_u = u( i, 0 ) - u( j, 0 );
            const double xi_y = x( i, 1 ) - x( j, 1 );
            const double eta_v = u( i, 1 ) - u( j, 1 );
            const double xi_z = x( i, 2 ) - x( j, 2 );
            const double eta_w = u( i, 2 ) - u( j, 2 );
            const double xi = sqrt( xi_x * xi_x + xi_y * xi_y + xi_z * xi_z );
            const double linear_s =
                ( xi_x * eta_u + xi_y * eta_v + xi_z * eta_w ) / ( xi * xi );
            const double coeff = mu * c * linear_s * vol( i );
            fx_i = coeff * xi_x / xi;
            fy_i = coeff * xi_y / xi;
            fz_i = coeff * xi_z / xi;

            f( i, 0 ) += fx_i;
            f( i, 1 ) += fy_i;
            f( i, 2 ) += fz_i;
        };

        Kokkos::RangePolicy<exec_space> policy( 0, n_local );
        Cabana::neighbor_parallel_for(
            policy, force_full, neigh_list, Cabana::FirstNeighborsTag(),
            neigh_op_tag, "CabanaPD::ForceLinearPMB::compute_full" );
    }

    template <class PosType, class VolType, class WType, class NeighListType,
              class ParallelType>
    double compute_energy_full( const PMBModelTag, WType &W, const PosType &x,
                                const PosType &u, const VolType &vol,
                                const NeighListType &neigh_list,
                                const int n_local, ParallelType &neigh_op_tag )
    {
        auto c = _c;
        auto energy_full =
            KOKKOS_LAMBDA( const int i, const int j, double &Phi )
        {
            // Get the reference positions and displacements.
            const double xi_x = x( i, 0 ) - x( j, 0 );
            const double eta_u = u( i, 0 ) - u( j, 0 );
            const double xi_y = x( i, 1 ) - x( j, 1 );
            const double eta_v = u( i, 1 ) - u( j, 1 );
            const double xi_z = x( i, 2 ) - x( j, 2 );
            const double eta_w = u( i, 2 ) - u( j, 2 );
            const double rx = xi_x + eta_u;
            const double ry = xi_y + eta_v;
            const double rz = xi_z + eta_w;
            const double r = sqrt( rx * rx + ry * ry + rz * rz );
            const double xi = sqrt( xi_x * xi_x + xi_y * xi_y + xi_z * xi_z );
            const double s = ( r - xi ) / xi;

            // 1/2 from outside the integral; 1/2 from the integrand (pairwise
            // potential).
            double w = 0.25 * c * s * s * xi * vol( i );
            W( i ) += w;
            Phi += w * vol( i );
        };

        double strain_energy = 0.0;
        Kokkos::RangePolicy<exec_space> policy( 0, n_local );
        Cabana::neighbor_parallel_reduce(
            policy, energy_full, neigh_list, Cabana::FirstNeighborsTag(),
            neigh_op_tag, strain_energy,
            "CabanaPD::ForcePMB::compute_energy_full" );

        return strain_energy;
    }

    template <class PosType, class VolType, class WType, class NeighListType,
              class ParallelType>
    double compute_energy_full( const LinearPMBModelTag, WType &W,
                                const PosType &x, const PosType &u,
                                const VolType &vol,
                                const NeighListType &neigh_list,
                                const int n_local, ParallelType &neigh_op_tag )
    {
        auto c = _c;
        auto energy_full =
            KOKKOS_LAMBDA( const int i, const int j, double &Phi )
        {
            // Get the reference positions and displacements.
            const double xi_x = x( i, 0 ) - x( j, 0 );
            const double eta_u = u( i, 0 ) - u( j, 0 );
            const double xi_y = x( i, 1 ) - x( j, 1 );
            const double eta_v = u( i, 1 ) - u( j, 1 );
            const double xi_z = x( i, 2 ) - x( j, 2 );
            const double eta_w = u( i, 2 ) - u( j, 2 );
            const double xi = sqrt( xi_x * xi_x + xi_y * xi_y + xi_z * xi_z );
            const double linear_s =
                ( xi_x * eta_u + xi_y * eta_v + xi_z * eta_w ) / ( xi * xi );

            // 1/2 from outside the integral; 1/2 from the integrand (pairwise
            // potential).
            double w = 0.25 * c * linear_s * linear_s * xi * vol( i );
            W( i ) += w;
            Phi += w * vol( i );
        };

        double strain_energy = 0.0;
        Kokkos::RangePolicy<exec_space> policy( 0, n_local );
        Cabana::neighbor_parallel_reduce(
            policy, energy_full, neigh_list, Cabana::FirstNeighborsTag(),
            neigh_op_tag, strain_energy,
            "CabanaPD::ForceLinearPMB::compute_energy_full" );

        return strain_energy;
    }
};

} // namespace CabanaPD

#endif
