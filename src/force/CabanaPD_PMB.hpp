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

#ifndef FORCE_PMB_H
#define FORCE_PMB_H

#include <cmath>

#include <CabanaPD_Force.hpp>
#include <CabanaPD_ForceModels.hpp>
#include <CabanaPD_Particles.hpp>
#include <CabanaPD_Types.hpp>
#include <force_models/CabanaPD_PMB.hpp>

namespace CabanaPD
{
template <class MemorySpace, class ModelType>
class Force<MemorySpace, ModelType, PMB, NoFracture>
    : public BaseForce<MemorySpace>
{
  public:
    // Using the default exec_space.
    using exec_space = typename MemorySpace::execution_space;
    using model_type = ModelType;
    using base_type = BaseForce<MemorySpace>;
    using neighbor_list_type = typename base_type::neighbor_list_type;
    using base_type::_neigh_list;

  protected:
    using base_type::_half_neigh;
    model_type _model;

    using base_type::_energy_timer;
    using base_type::_stress_timer;
    using base_type::_timer;
    using base_type::_total_strain_energy;

  public:
    template <class ParticleType>
    Force( const bool half_neigh, const ParticleType& particles,
           const model_type model )
        : base_type( half_neigh, model.cutoff(), particles )
        , _model( model )
    {
    }

    template <class ForceType, class PosType, class ParticleType,
              class ParallelType>
    void computeForceFull( ForceType& f, const PosType& x, const PosType& u,
                           const ParticleType& particles,
                           ParallelType& neigh_op_tag )
    {
        _timer.start();

        auto model = _model;
        const auto vol = particles.sliceVolume();

        auto force_full = KOKKOS_LAMBDA( const int i, const int j )
        {
            double fx_i = 0.0;
            double fy_i = 0.0;
            double fz_i = 0.0;

            double xi, r, s;
            double rx, ry, rz;
            getDistance( x, u, i, j, xi, r, s, rx, ry, rz );

            s = model( ThermalStretchTag{}, i, j, s );

            const double coeff = model( ForceCoeffTag{}, i, j, s, vol( j ) );
            fx_i = coeff * rx / r;
            fy_i = coeff * ry / r;
            fz_i = coeff * rz / r;

            f( i, 0 ) += fx_i;
            f( i, 1 ) += fy_i;
            f( i, 2 ) += fz_i;
        };

        Kokkos::RangePolicy<exec_space> policy( particles.frozenOffset(),
                                                particles.localOffset() );
        Cabana::neighbor_parallel_for(
            policy, force_full, _neigh_list, Cabana::FirstNeighborsTag(),
            neigh_op_tag, "CabanaPD::ForcePMB::computeFull" );
        Kokkos::fence();
        _timer.stop();
    }

    template <class PosType, class WType, class ParticleType,
              class ParallelType>
    void computeEnergyFull( WType& W, const PosType& x, const PosType& u,
                            const ParticleType& particles,
                            ParallelType& neigh_op_tag )
    {
        _energy_timer.start();

        auto model = _model;
        const auto vol = particles.sliceVolume();

        auto energy_full =
            KOKKOS_LAMBDA( const int i, const int j, double& Phi )
        {
            // Get the bond distance, displacement, and stretch.
            double xi, r, s;
            getDistance( x, u, i, j, xi, r, s );

            s = model( ThermalStretchTag{}, i, j, s );

            double w = model( EnergyTag{}, i, j, s, xi, vol( j ) );
            W( i ) += w;
            Phi += w * vol( i );
        };

        Kokkos::RangePolicy<exec_space> policy( particles.frozenOffset(),
                                                particles.localOffset() );
        Cabana::neighbor_parallel_reduce(
            policy, energy_full, _neigh_list, Cabana::FirstNeighborsTag(),
            neigh_op_tag, _total_strain_energy,
            "CabanaPD::ForcePMB::computeEnergyFull" );
        Kokkos::fence();
        _energy_timer.stop();
    }

    template <class ParticleType, class ParallelType>
    void computeStressFull( ParticleType& particles,
                            ParallelType& neigh_op_tag )
    {
        _stress_timer.start();

        auto model = _model;
        const auto x = particles.sliceReferencePosition();
        const auto u = particles.sliceDisplacement();
        const auto vol = particles.sliceVolume();
        const auto f = particles.sliceForce();
        auto stress = particles.sliceStress();

        auto stress_full = KOKKOS_LAMBDA( const int i, const int j )
        {
            // Get the bond distance, displacement, and stretch.
            double xi, r, s;
            double rx, ry, rz;
            double xi_x, xi_y, xi_z;
            getDistance( x, u, i, j, xi, r, s, rx, ry, rz, xi_x, xi_y, xi_z );

            s = model( ThermalStretchTag{}, i, j, s );

            const double coeff =
                0.5 * model( ForceCoeffTag{}, i, j, s, vol( j ) );
            const double fx_i = coeff * rx / r;
            const double fy_i = coeff * ry / r;
            const double fz_i = coeff * rz / r;

            stress( i, 0, 0 ) += fx_i * xi_x;
            stress( i, 1, 1 ) += fy_i * xi_y;
            stress( i, 2, 2 ) += fz_i * xi_z;

            stress( i, 0, 1 ) += fx_i * xi_y;
            stress( i, 1, 0 ) += fy_i * xi_x;

            stress( i, 0, 2 ) += fx_i * xi_z;
            stress( i, 2, 0 ) += fz_i * xi_x;

            stress( i, 1, 2 ) += fy_i * xi_z;
            stress( i, 2, 1 ) += fz_i * xi_y;
        };

        Kokkos::RangePolicy<exec_space> policy( particles.frozenOffset(),
                                                particles.localOffset() );
        Cabana::neighbor_parallel_for(
            policy, stress_full, _neigh_list, Cabana::FirstNeighborsTag(),
            neigh_op_tag, "CabanaPD::ForcePMB::computeStressFull" );
        Kokkos::fence();
        _stress_timer.stop();
    }
};

template <class MemorySpace, class ModelType>
class Force<MemorySpace, ModelType, PMB, Fracture>
    : public BaseForce<MemorySpace>, public BaseFracture<MemorySpace>
{
  public:
    // Using the default exec_space.
    using exec_space = typename MemorySpace::execution_space;
    using model_type = ModelType;
    using base_type = BaseForce<MemorySpace>;
    using neighbor_list_type = typename base_type::neighbor_list_type;
    using base_type::_neigh_list;

  protected:
    using base_type::_half_neigh;
    using fracture_type = BaseFracture<MemorySpace>;
    using fracture_type::_mu;
    model_type _model;

    using base_type::_energy_timer;
    using base_type::_stress_timer;
    using base_type::_timer;
    using base_type::_total_strain_energy;
    double _total_damage;

  public:
    template <class ParticleType>
    Force( const bool half_neigh, const ParticleType& particles,
           const model_type model )
        : base_type( half_neigh, model.cutoff(), particles )
        , fracture_type( particles.localOffset(),
                         base_type::getMaxLocalNeighbors() )
        , _model( model )
    {
        // Needed only for models which store per-bond information.
        _model.updateBonds( particles.localOffset(),
                            base_type::getMaxLocalNeighbors() );
    }

    template <class ExecSpace, class ParticleType, class PrenotchType>
    void prenotch( ExecSpace exec_space, const ParticleType& particles,
                   PrenotchType& prenotch )
    {
        fracture_type::prenotch( exec_space, particles, prenotch, _neigh_list );
    }

    template <class ForceType, class PosType, class ParticleType,
              class ParallelType>
    void computeForceFull( ForceType& f, const PosType& x, const PosType& u,
                           const ParticleType& particles, ParallelType& )
    {
        _timer.start();

        auto model = _model;
        auto neigh_list = _neigh_list;
        auto mu = _mu;
        const auto vol = particles.sliceVolume();
        const auto nofail = particles.sliceNoFail();

        auto force_full = KOKKOS_LAMBDA( const int i )
        {
            std::size_t num_neighbors =
                Cabana::NeighborList<neighbor_list_type>::numNeighbor(
                    neigh_list, i );
            for ( std::size_t n = 0; n < num_neighbors; n++ )
            {
                double fx_i = 0.0;
                double fy_i = 0.0;
                double fz_i = 0.0;

                std::size_t j =
                    Cabana::NeighborList<neighbor_list_type>::getNeighbor(
                        neigh_list, i, n );

                // Get the reference positions and displacements.
                double xi, r, s;
                double rx, ry, rz;
                getDistance( x, u, i, j, xi, r, s, rx, ry, rz );

                s = model( ThermalStretchTag{}, i, j, s );

                // Break if beyond critical stretch unless in no-fail zone.
                if ( model( CriticalStretchTag{}, i, j, r, xi ) &&
                     !nofail( i ) && !nofail( j ) )
                {
                    mu( i, n ) = 0;
                }
                // Else if statement is only for performance.
                else if ( mu( i, n ) > 0 )
                {
                    const double coeff =
                        model( ForceCoeffTag{}, i, j, s, vol( j ), n );

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

        Kokkos::RangePolicy<exec_space> policy( particles.frozenOffset(),
                                                particles.localOffset() );
        Kokkos::parallel_for( "CabanaPD::ForcePMBDamage::computeFull", policy,
                              force_full );
        Kokkos::fence();
        _timer.stop();
    }

    template <class PosType, class WType, class ParticleType,
              class ParallelType>
    void computeEnergyFull( WType& W, const PosType& x, const PosType& u,
                            ParticleType& particles, ParallelType& )
    {
        _energy_timer.start();

        auto model = _model;
        auto neigh_list = _neigh_list;
        auto mu = _mu;
        const auto vol = particles.sliceVolume();
        auto phi = particles.sliceDamage();

        auto energy_full = KOKKOS_LAMBDA( const int i, double& Phi, double& D )
        {
            std::size_t num_neighbors =
                Cabana::NeighborList<neighbor_list_type>::numNeighbor(
                    neigh_list, i );
            double phi_i = 0.0;
            double vol_H_i = 0.0;
            for ( std::size_t n = 0; n < num_neighbors; n++ )
            {
                std::size_t j =
                    Cabana::NeighborList<neighbor_list_type>::getNeighbor(
                        neigh_list, i, n );
                // Get the bond distance, displacement, and stretch.
                double xi, r, s;
                getDistance( x, u, i, j, xi, r, s );

                s = model( ThermalStretchTag{}, i, j, s );

                double w =
                    mu( i, n ) * model( EnergyTag{}, i, j, s, xi, vol( j ), n );
                W( i ) += w;

                phi_i += mu( i, n ) * vol( j );
                vol_H_i += vol( j );
            }
            Phi += W( i ) * vol( i );
            phi( i ) = 1 - phi_i / vol_H_i;
            D += phi( i );
        };

        Kokkos::RangePolicy<exec_space> policy( particles.frozenOffset(),
                                                particles.localOffset() );
        Kokkos::parallel_reduce( "CabanaPD::ForcePMBDamage::computeEnergyFull",
                                 policy, energy_full, _total_strain_energy,
                                 _total_damage );
        Kokkos::fence();
        _energy_timer.stop();
    }

    template <class ParticleType, class ParallelType>
    void computeStressFull( ParticleType& particles, ParallelType& )
    {
        _stress_timer.start();

        auto model = _model;
        auto neigh_list = _neigh_list;
        const auto x = particles.sliceReferencePosition();
        const auto u = particles.sliceDisplacement();
        const auto vol = particles.sliceVolume();
        const auto f = particles.sliceForce();
        auto stress = particles.sliceStress();
        auto mu = _mu;

        auto stress_full = KOKKOS_LAMBDA( const int i )
        {
            std::size_t num_neighbors =
                Cabana::NeighborList<neighbor_list_type>::numNeighbor(
                    neigh_list, i );
            for ( std::size_t n = 0; n < num_neighbors; n++ )
            {
                std::size_t j =
                    Cabana::NeighborList<neighbor_list_type>::getNeighbor(
                        neigh_list, i, n );
                // Get the bond distance, displacement, and stretch.
                double xi, r, s;
                double rx, ry, rz;
                double xi_x, xi_y, xi_z;
                getDistance( x, u, i, j, xi, r, s, rx, ry, rz, xi_x, xi_y,
                             xi_z );

                s = model( ThermalStretchTag{}, i, j, s );

                const double coeff =
                    0.5 * model( ForceCoeffTag{}, i, j, s, vol( j ), n );
                const double muij = mu( i, n );
                const double fx_i = muij * coeff * rx / r;
                const double fy_i = muij * coeff * ry / r;
                const double fz_i = muij * coeff * rz / r;

                stress( i, 0, 0 ) += fx_i * xi_x;
                stress( i, 1, 1 ) += fy_i * xi_y;
                stress( i, 2, 2 ) += fz_i * xi_z;

                stress( i, 0, 1 ) += fx_i * xi_y;
                stress( i, 1, 0 ) += fy_i * xi_x;

                stress( i, 0, 2 ) += fx_i * xi_z;
                stress( i, 2, 0 ) += fz_i * xi_x;

                stress( i, 1, 2 ) += fy_i * xi_z;
                stress( i, 2, 1 ) += fz_i * xi_y;
            }
        };

        Kokkos::RangePolicy<exec_space> policy( particles.frozenOffset(),
                                                particles.localOffset() );
        Kokkos::parallel_for( "CabanaPD::ForcePMBDamage::computeStressFull",
                              policy, stress_full );
        Kokkos::fence();
        _stress_timer.stop();
    }

    auto totalDamage() { return _total_damage; }
};

template <class MemorySpace, class ModelType>
class Force<MemorySpace, ModelType, LinearPMB, NoFracture>
    : public BaseForce<MemorySpace>
{
  public:
    // Using the default exec_space.
    using exec_space = typename MemorySpace::execution_space;
    using model_type = ModelType;
    using base_type = BaseForce<MemorySpace>;
    using neighbor_list_type = typename base_type::neighbor_list_type;
    using base_type::_neigh_list;

  protected:
    using base_type::_half_neigh;
    model_type _model;

    using base_type::_energy_timer;
    using base_type::_stress_timer;
    using base_type::_timer;
    using base_type::_total_strain_energy;

  public:
    template <class ParticleType>
    Force( const bool half_neigh, const ParticleType& particles,
           const model_type model )
        : base_type( half_neigh, model.cutoff(), particles )
        , _model( model )
    {
    }

    template <class ForceType, class PosType, class ParticleType,
              class ParallelType>
    void computeForceFull( ForceType& f, const PosType& x, const PosType& u,
                           ParticleType& particles, ParallelType& neigh_op_tag )
    {
        _timer.start();

        auto model = _model;
        const auto vol = particles.sliceVolume();

        auto force_full = KOKKOS_LAMBDA( const int i, const int j )
        {
            double fx_i = 0.0;
            double fy_i = 0.0;
            double fz_i = 0.0;

            // Get the bond distance, displacement, and linearized stretch.
            double xi, linear_s;
            double xi_x, xi_y, xi_z;
            getLinearizedDistance( x, u, i, j, xi, linear_s, xi_x, xi_y, xi_z );

            linear_s = model( ThermalStretchTag{}, i, j, linear_s );

            const double coeff =
                model( ForceCoeffTag{}, i, j, linear_s, vol( j ) );
            fx_i = coeff * xi_x / xi;
            fy_i = coeff * xi_y / xi;
            fz_i = coeff * xi_z / xi;

            f( i, 0 ) += fx_i;
            f( i, 1 ) += fy_i;
            f( i, 2 ) += fz_i;
        };

        Kokkos::RangePolicy<exec_space> policy( particles.frozenOffset(),
                                                particles.localOffset() );
        Cabana::neighbor_parallel_for(
            policy, force_full, _neigh_list, Cabana::FirstNeighborsTag(),
            neigh_op_tag, "CabanaPD::ForceLinearPMB::computeFull" );
        Kokkos::fence();
        _timer.stop();
    }

    template <class PosType, class WType, class ParticleType,
              class ParallelType>
    void computeEnergyFull( WType& W, const PosType& x, const PosType& u,
                            ParticleType& particles,
                            ParallelType& neigh_op_tag )
    {
        _energy_timer.start();

        auto model = _model;
        const auto vol = particles.sliceVolume();

        auto energy_full =
            KOKKOS_LAMBDA( const int i, const int j, double& Phi )
        {
            // Get the bond distance, displacement, and linearized stretch.
            double xi, linear_s;
            getLinearizedDistance( x, u, i, j, xi, linear_s );

            linear_s = model( ThermalStretchTag{}, i, j, linear_s );

            double w = model( EnergyTag{}, i, j, linear_s, xi, vol( j ) );
            W( i ) += w;
            Phi += w * vol( i );
        };

        Kokkos::RangePolicy<exec_space> policy( particles.frozenOffset(),
                                                particles.localOffset() );
        Cabana::neighbor_parallel_reduce(
            policy, energy_full, _neigh_list, Cabana::FirstNeighborsTag(),
            neigh_op_tag, _total_strain_energy,
            "CabanaPD::ForceLinearPMB::computeEnergyFull" );
        Kokkos::fence();
        _energy_timer.stop();
    }

    template <class ParticleType, class ParallelType>
    void computeStressFull( ParticleType& particles,
                            ParallelType& neigh_op_tag )
    {
        _stress_timer.start();

        auto model = _model;
        auto neigh_list = _neigh_list;
        const auto x = particles.sliceReferencePosition();
        const auto u = particles.sliceDisplacement();
        const auto vol = particles.sliceVolume();
        auto stress = particles.sliceStress();

        auto stress_full = KOKKOS_LAMBDA( const int i, const int j )
        {
            // Get the linearized components
            double xi, linear_s;
            double xi_x, xi_y, xi_z;
            getLinearizedDistance( x, u, i, j, xi, linear_s, xi_x, xi_y, xi_z );

            linear_s = model( ThermalStretchTag{}, i, j, linear_s );

            const double coeff =
                0.5 * model( ForceCoeffTag{}, i, j, linear_s, vol( j ) );
            const double fx_i = coeff * xi_x / xi;
            const double fy_i = coeff * xi_y / xi;
            const double fz_i = coeff * xi_z / xi;

            // Update stress tensor components
            stress( i, 0, 0 ) += fx_i * xi_x;
            stress( i, 1, 1 ) += fy_i * xi_y;
            stress( i, 2, 2 ) += fz_i * xi_z;

            stress( i, 0, 1 ) += fx_i * xi_y;
            stress( i, 1, 0 ) += fy_i * xi_x;

            stress( i, 0, 2 ) += fx_i * xi_z;
            stress( i, 2, 0 ) += fz_i * xi_x;

            stress( i, 1, 2 ) += fy_i * xi_z;
            stress( i, 2, 1 ) += fz_i * xi_y;
        };

        Kokkos::RangePolicy<exec_space> policy( particles.frozenOffset(),
                                                particles.localOffset() );
        Cabana::neighbor_parallel_for(
            policy, stress_full, _neigh_list, Cabana::FirstNeighborsTag(),
            neigh_op_tag, "CabanaPD::ForceLinearPMB::computeStressFull" );
        Kokkos::fence();
        _stress_timer.stop();
    }
};

} // namespace CabanaPD

#endif
