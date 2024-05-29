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

#ifndef FORCE_LPS_H
#define FORCE_LPS_H

#include <cmath>

#include <CabanaPD_Force.hpp>
#include <CabanaPD_Particles.hpp>
#include <CabanaPD_Types.hpp>
#include <force/CabanaPD_ForceModels_LPS.hpp>

namespace CabanaPD
{
template <class ExecutionSpace>
class Force<ExecutionSpace, ForceModel<LPS, Elastic>>
{
  protected:
    bool _half_neigh;
    ForceModel<LPS, Elastic> _model;

    Timer _timer;
    Timer _energy_timer;

  public:
    using exec_space = ExecutionSpace;

    Force( const bool half_neigh, const ForceModel<LPS, Elastic> model )
        : _half_neigh( half_neigh )
        , _model( model )
    {
    }

    template <class ParticleType, class NeighListType, class ParallelType>
    void computeWeightedVolume( ParticleType& particles,
                                const NeighListType& neigh_list,
                                const ParallelType neigh_op_tag )
    {
        _timer.start();

        auto n_local = particles.n_local;
        auto x = particles.sliceReferencePosition();
        auto u = particles.sliceDisplacement();
        const auto vol = particles.sliceVolume();
        auto m = particles.sliceWeightedVolume();
        Cabana::deep_copy( m, 0.0 );
        auto model = _model;

        auto weighted_volume = KOKKOS_LAMBDA( const int i, const int j )
        {
            // Get the reference positions and displacements.
            double xi, r, s;
            getDistance( x, u, i, j, xi, r, s );
            double m_j = model.influence_function( xi ) * xi * xi * vol( j );
            m( i ) += m_j;
        };

        Kokkos::RangePolicy<exec_space> policy( 0, n_local );
        Cabana::neighbor_parallel_for(
            policy, weighted_volume, neigh_list, Cabana::FirstNeighborsTag(),
            neigh_op_tag, "CabanaPD::ForceLPS::computeWeightedVolume" );

        _timer.stop();
    }

    template <class ParticleType, class NeighListType, class ParallelType>
    void computeDilatation( ParticleType& particles,
                            const NeighListType& neigh_list,
                            const ParallelType neigh_op_tag )
    {
        _timer.start();

        auto n_local = particles.n_local;
        const auto x = particles.sliceReferencePosition();
        auto u = particles.sliceDisplacement();
        const auto vol = particles.sliceVolume();
        auto m = particles.sliceWeightedVolume();
        auto theta = particles.sliceDilatation();
        auto model = _model;
        Cabana::deep_copy( theta, 0.0 );

        auto dilatation = KOKKOS_LAMBDA( const int i, const int j )
        {
            // Get the bond distance, displacement, and stretch.
            double xi, r, s;
            getDistance( x, u, i, j, xi, r, s );
            double theta_i =
                model.influence_function( xi ) * s * xi * xi * vol( j );
            theta( i ) += 3.0 * theta_i / m( i );
        };

        Kokkos::RangePolicy<exec_space> policy( 0, n_local );
        Cabana::neighbor_parallel_for(
            policy, dilatation, neigh_list, Cabana::FirstNeighborsTag(),
            neigh_op_tag, "CabanaPD::ForceLPS::computeDilatation" );

        _timer.stop();
    }

    template <class ForceType, class PosType, class ParticleType,
              class NeighListType, class ParallelType>
    void computeForceFull( ForceType& f, const PosType& x, const PosType& u,
                           const ParticleType& particles,
                           const NeighListType& neigh_list, const int n_local,
                           ParallelType& neigh_op_tag )
    {
        _timer.start();

        auto theta_coeff = _model.theta_coeff;
        auto s_coeff = _model.s_coeff;
        auto model = _model;

        const auto vol = particles.sliceVolume();
        auto theta = particles.sliceDilatation();
        auto m = particles.sliceWeightedVolume();
        auto force_full = KOKKOS_LAMBDA( const int i, const int j )
        {
            double fx_i = 0.0;
            double fy_i = 0.0;
            double fz_i = 0.0;

            double xi, r, s;
            double rx, ry, rz;
            getDistanceComponents( x, u, i, j, xi, r, s, rx, ry, rz );
            const double coeff =
                ( theta_coeff * ( theta( i ) / m( i ) + theta( j ) / m( j ) ) +
                  s_coeff * s * ( 1.0 / m( i ) + 1.0 / m( j ) ) ) *
                model.influence_function( xi ) * xi * vol( j );
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
            neigh_op_tag, "CabanaPD::ForceLPS::computeFull" );

        _timer.stop();
    }

    template <class PosType, class WType, class ParticleType,
              class NeighListType, class ParallelType>
    double computeEnergyFull( WType& W, const PosType& x, const PosType& u,
                              const ParticleType& particles,
                              const NeighListType& neigh_list,
                              const int n_local, ParallelType& neigh_op_tag )
    {
        _energy_timer.start();

        auto theta_coeff = _model.theta_coeff;
        auto s_coeff = _model.s_coeff;
        auto model = _model;

        const auto vol = particles.sliceVolume();
        const auto theta = particles.sliceDilatation();
        auto m = particles.sliceWeightedVolume();

        auto energy_full =
            KOKKOS_LAMBDA( const int i, const int j, double& Phi )
        {
            double xi, r, s;
            getDistance( x, u, i, j, xi, r, s );

            double num_neighbors = static_cast<double>(
                Cabana::NeighborList<NeighListType>::numNeighbor( neigh_list,
                                                                  i ) );

            double w = ( 1.0 / num_neighbors ) * 0.5 * theta_coeff / 3.0 *
                           ( theta( i ) * theta( i ) ) +
                       0.5 * ( s_coeff / m( i ) ) *
                           model.influence_function( xi ) * s * s * xi * xi *
                           vol( j );
            W( i ) += w;
            Phi += w * vol( i );
        };

        double strain_energy = 0.0;

        Kokkos::RangePolicy<exec_space> policy( 0, n_local );
        Cabana::neighbor_parallel_reduce(
            policy, energy_full, neigh_list, Cabana::FirstNeighborsTag(),
            neigh_op_tag, strain_energy,
            "CabanaPD::ForceLPS::computeEnergyFull" );

        _energy_timer.stop();
        return strain_energy;
    }

    auto time() { return _timer.time(); };
    auto timeEnergy() { return _energy_timer.time(); };
};

template <class ExecutionSpace>
class Force<ExecutionSpace, ForceModel<LPS, Fracture>>
    : public Force<ExecutionSpace, ForceModel<LPS, Elastic>>
{
  protected:
    using base_type = Force<ExecutionSpace, ForceModel<LPS, Elastic>>;
    using base_type::_half_neigh;
    ForceModel<LPS, Fracture> _model;

    using base_type::_energy_timer;
    using base_type::_timer;

  public:
    using exec_space = ExecutionSpace;

    Force( const bool half_neigh, const ForceModel<LPS, Fracture> model )
        : base_type( half_neigh, model )
        , _model( model )
    {
    }

    template <class ParticleType, class NeighListType, class MuView>
    void computeWeightedVolume( ParticleType& particles,
                                const NeighListType& neigh_list,
                                const MuView& mu )
    {
        _timer.start();

        auto n_local = particles.n_local;
        auto x = particles.sliceReferencePosition();
        auto u = particles.sliceDisplacement();
        const auto vol = particles.sliceVolume();
        auto m = particles.sliceWeightedVolume();
        Cabana::deep_copy( m, 0.0 );
        auto model = _model;

        auto weighted_volume = KOKKOS_LAMBDA( const int i )
        {
            std::size_t num_neighbors =
                Cabana::NeighborList<NeighListType>::numNeighbor( neigh_list,
                                                                  i );
            for ( std::size_t n = 0; n < num_neighbors; n++ )
            {
                std::size_t j =
                    Cabana::NeighborList<NeighListType>::getNeighbor(
                        neigh_list, i, n );

                // Get the reference positions and displacements.
                double xi, r, s;
                getDistance( x, u, i, j, xi, r, s );
                // mu is included to account for bond breaking.
                double m_j = mu( i, n ) * model.influence_function( xi ) * xi *
                             xi * vol( j );
                m( i ) += m_j;
            }
        };

        Kokkos::RangePolicy<exec_space> policy( 0, n_local );
        Kokkos::parallel_for( "CabanaPD::ForceLPSDamage::computeWeightedVolume",
                              policy, weighted_volume );

        _timer.stop();
    }

    template <class ParticleType, class NeighListType, class MuView>
    void computeDilatation( ParticleType& particles,
                            const NeighListType& neigh_list, const MuView& mu )
    {
        _timer.start();

        auto n_local = particles.n_local;
        const auto x = particles.sliceReferencePosition();
        auto u = particles.sliceDisplacement();
        const auto vol = particles.sliceVolume();
        auto m = particles.sliceWeightedVolume();
        auto theta = particles.sliceDilatation();
        auto model = _model;
        Cabana::deep_copy( theta, 0.0 );

        auto dilatation = KOKKOS_LAMBDA( const int i )
        {
            std::size_t num_neighbors =
                Cabana::NeighborList<NeighListType>::numNeighbor( neigh_list,
                                                                  i );
            for ( std::size_t n = 0; n < num_neighbors; n++ )
            {
                std::size_t j =
                    Cabana::NeighborList<NeighListType>::getNeighbor(
                        neigh_list, i, n );

                // Get the bond distance, displacement, and stretch.
                double xi, r, s;
                getDistance( x, u, i, j, xi, r, s );
                // mu is included to account for bond breaking.
                double theta_i = mu( i, n ) * model.influence_function( xi ) *
                                 s * xi * xi * vol( j );
                // Check if all bonds are broken (m=0) to avoid dividing by
                // zero. Alternatively, one could check if this bond mu(i,n) is
                // broken, because m=0 only occurs when all bonds are broken.
                if ( m( i ) > 0 )
                    theta( i ) += 3.0 * theta_i / m( i );
            }
        };

        Kokkos::RangePolicy<exec_space> policy( 0, n_local );
        Kokkos::parallel_for( "CabanaPD::ForceLPSDamage::computeDilatation",
                              policy, dilatation );

        _timer.stop();
    }

    template <class ForceType, class PosType, class ParticleType,
              class NeighListType, class MuView, class ParallelType>
    void computeForceFull( ForceType& f, const PosType& x, const PosType& u,
                           const ParticleType& particles,
                           const NeighListType& neigh_list, MuView& mu,
                           const int n_local, ParallelType& )
    {
        _timer.start();

        auto break_coeff = _model.bond_break_coeff;
        auto theta_coeff = _model.theta_coeff;
        auto s_coeff = _model.s_coeff;
        auto model = _model;

        const auto vol = particles.sliceVolume();
        auto theta = particles.sliceDilatation();
        auto m = particles.sliceWeightedVolume();
        const auto nofail = particles.sliceNoFail();
        auto force_full = KOKKOS_LAMBDA( const int i )
        {
            std::size_t num_neighbors =
                Cabana::NeighborList<NeighListType>::numNeighbor( neigh_list,
                                                                  i );
            for ( std::size_t n = 0; n < num_neighbors; n++ )
            {
                double fx_i = 0.0;
                double fy_i = 0.0;
                double fz_i = 0.0;

                std::size_t j =
                    Cabana::NeighborList<NeighListType>::getNeighbor(
                        neigh_list, i, n );

                // Get the reference positions and displacements.
                double xi, r, s;
                double rx, ry, rz;
                getDistanceComponents( x, u, i, j, xi, r, s, rx, ry, rz );

                // Break if beyond critical stretch unless in no-fail zone.
                if ( r * r >= break_coeff * xi * xi && !nofail( i ) &&
                     !nofail( i ) )
                {
                    mu( i, n ) = 0;
                }
                // Check if this bond is broken (mu=0) to ensure m(i) and m(j)
                // are both >0 (m=0 only occurs when all bonds are broken) to
                // avoid dividing by zero.
                else if ( mu( i, n ) > 0 )
                {
                    const double coeff =
                        ( theta_coeff *
                              ( theta( i ) / m( i ) + theta( j ) / m( j ) ) +
                          s_coeff * s * ( 1.0 / m( i ) + 1.0 / m( j ) ) ) *
                        model.influence_function( xi ) * xi * vol( j );
                    double muij = mu( i, n );
                    fx_i = muij * coeff * rx / r;
                    fy_i = muij * coeff * ry / r;
                    fz_i = muij * coeff * rz / r;

                    f( i, 0 ) += fx_i;
                    f( i, 1 ) += fy_i;
                    f( i, 2 ) += fz_i;
                }
            }
        };

        Kokkos::RangePolicy<exec_space> policy( 0, n_local );
        Kokkos::parallel_for( "CabanaPD::ForceLPSDamage::computeFull", policy,
                              force_full );

        _timer.stop();
    }

    template <class PosType, class WType, class DamageType, class ParticleType,
              class NeighListType, class MuView, class ParallelType>
    double computeEnergyFull( WType& W, const PosType& x, const PosType& u,
                              DamageType& phi, const ParticleType& particles,
                              const NeighListType& neigh_list, MuView& mu,
                              const int n_local, ParallelType& )
    {
        _energy_timer.start();

        auto theta_coeff = _model.theta_coeff;
        auto s_coeff = _model.s_coeff;
        auto model = _model;

        const auto vol = particles.sliceVolume();
        const auto theta = particles.sliceDilatation();
        auto m = particles.sliceWeightedVolume();

        auto energy_full = KOKKOS_LAMBDA( const int i, double& Phi )
        {
            std::size_t num_neighbors =
                Cabana::NeighborList<NeighListType>::numNeighbor( neigh_list,
                                                                  i );
            double num_bonds = 0.0;
            for ( std::size_t n = 0; n < num_neighbors; n++ )
                num_bonds += static_cast<double>( mu( i, n ) );

            double phi_i = 0.0;
            double vol_H_i = 0.0;
            for ( std::size_t n = 0; n < num_neighbors; n++ )
            {
                std::size_t j =
                    Cabana::NeighborList<NeighListType>::getNeighbor(
                        neigh_list, i, n );
                // Get the bond distance, displacement, and stretch.
                double xi, r, s;
                getDistance( x, u, i, j, xi, r, s );

                double w =
                    mu( i, n ) * ( ( 1.0 / num_bonds ) * 0.5 * theta_coeff /
                                       3.0 * ( theta( i ) * theta( i ) ) +
                                   0.5 * ( s_coeff / m( i ) ) *
                                       model.influence_function( xi ) * s * s *
                                       xi * xi * vol( j ) );
                W( i ) += w;

                phi_i += mu( i, n ) * vol( j );
                vol_H_i += vol( j );
            }
            Phi += W( i ) * vol( i );
            phi( i ) = 1 - phi_i / vol_H_i;
        };

        double strain_energy = 0.0;
        Kokkos::RangePolicy<exec_space> policy( 0, n_local );
        Kokkos::parallel_reduce( "CabanaPD::ForceLPSDamage::computeEnergyFull",
                                 policy, energy_full, strain_energy );

        _energy_timer.stop();
        return strain_energy;
    }
};

template <class ExecutionSpace>
class Force<ExecutionSpace, ForceModel<LinearLPS, Elastic>>
    : public Force<ExecutionSpace, ForceModel<LPS, Elastic>>
{
  protected:
    using base_type = Force<ExecutionSpace, ForceModel<LPS, Elastic>>;
    using base_type::_half_neigh;
    ForceModel<LinearLPS, Elastic> _model;

    using base_type::_energy_timer;
    using base_type::_timer;

  public:
    using exec_space = ExecutionSpace;

    Force( const bool half_neigh, const ForceModel<LinearLPS, Elastic> model )
        : base_type( half_neigh, model )
        , _model( model )
    {
    }

    template <class ForceType, class PosType, class ParticleType,
              class NeighListType, class ParallelType>
    void computeForceFull( ForceType& f, const PosType& x, const PosType& u,
                           const ParticleType& particles,
                           const NeighListType& neigh_list, const int n_local,
                           ParallelType& neigh_op_tag )
    {
        _timer.start();

        auto theta_coeff = _model.theta_coeff;
        auto s_coeff = _model.s_coeff;
        auto model = _model;

        const auto vol = particles.sliceVolume();
        auto linear_theta = particles.sliceDilatation();
        // Using weighted volume from base LPS class.
        auto m = particles.sliceWeightedVolume();

        auto force_full = KOKKOS_LAMBDA( const int i, const int j )
        {
            double fx_i = 0.0;
            double fy_i = 0.0;
            double fz_i = 0.0;

            // Get the bond distance and linearized stretch.
            double xi, linear_s;
            double xi_x, xi_y, xi_z;
            getLinearizedDistanceComponents( x, u, i, j, xi, linear_s, xi_x,
                                             xi_y, xi_z );

            const double coeff =
                ( theta_coeff * ( linear_theta( i ) / m( i ) +
                                  linear_theta( j ) / m( j ) ) +
                  s_coeff * linear_s * ( 1.0 / m( i ) + 1.0 / m( j ) ) ) *
                model.influence_function( xi ) * xi * vol( j );
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
            neigh_op_tag, "CabanaPD::ForceLPS::computeFull" );

        _timer.stop();
    }

    template <class PosType, class WType, class ParticleType,
              class NeighListType, class ParallelType>
    double computeEnergyFull( WType& W, const PosType& x, const PosType& u,
                              const ParticleType& particles,
                              const NeighListType& neigh_list,
                              const int n_local, ParallelType& neigh_op_tag )
    {
        _energy_timer.start();

        auto theta_coeff = _model.theta_coeff;
        auto s_coeff = _model.s_coeff;
        auto model = _model;

        const auto vol = particles.sliceVolume();
        const auto linear_theta = particles.sliceDilatation();
        auto m = particles.sliceWeightedVolume();

        auto energy_full =
            KOKKOS_LAMBDA( const int i, const int j, double& Phi )
        {
            // Do we need to recompute linear_theta_i?

            double xi, linear_s;
            getLinearizedDistance( x, u, i, j, xi, linear_s );

            double num_neighbors = static_cast<double>(
                Cabana::NeighborList<NeighListType>::numNeighbor( neigh_list,
                                                                  i ) );

            double w = ( 1.0 / num_neighbors ) * 0.5 * theta_coeff / 3.0 *
                           ( linear_theta( i ) * linear_theta( i ) ) +
                       0.5 * ( s_coeff / m( i ) ) *
                           model.influence_function( xi ) * linear_s *
                           linear_s * xi * xi * vol( j );
            W( i ) += w;
            Phi += w * vol( i );
        };

        double strain_energy = 0.0;

        Kokkos::RangePolicy<exec_space> policy( 0, n_local );
        Cabana::neighbor_parallel_reduce(
            policy, energy_full, neigh_list, Cabana::FirstNeighborsTag(),
            neigh_op_tag, strain_energy,
            "CabanaPD::ForceLPS::computeEnergyFull" );

        _energy_timer.stop();
        return strain_energy;
    }
};

} // namespace CabanaPD

#endif
