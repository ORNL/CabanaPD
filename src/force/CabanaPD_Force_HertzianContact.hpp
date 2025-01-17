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

#ifndef CONTACT_HERTZIAN_H
#define CONTACT_HERTZIAN_H

#include <Kokkos_Core.hpp>

#include <CabanaPD_Force.hpp>
#include <CabanaPD_Input.hpp>
#include <CabanaPD_Output.hpp>
#include <force/CabanaPD_Force_Contact.hpp>
#include <force/CabanaPD_HertzianContact.hpp>

namespace CabanaPD
{
/******************************************************************************
  Normal repulsion forces
******************************************************************************/
template <class MemorySpace>
class Force<MemorySpace, HertzianModel> : public BaseForceContact<MemorySpace>
{
  public:
    using base_type = BaseForceContact<MemorySpace>;
    using neighbor_list_type = typename base_type::neighbor_list_type;

    template <class ParticleType>
    Force( const bool half_neigh, const ParticleType& particles,
           const HertzianModel model )
        : base_type( half_neigh, particles, model )
        , _model( model )
    {
    }

    template <class ForceType, class PosType, class ParticleType,
              class ParallelType>
    void computeForceFull( ForceType& fc, const PosType& x, const PosType& u,
                           const ParticleType& particles,
                           const double max_displacement,
                           ParallelType& neigh_op_tag )
    {
        auto radius = _model.radius;
        auto Es = _model.Es;
        auto Rs = _model.Rs;
        auto beta = _model.beta;
        const int n_frozen = particles.frozenOffset();
        const int n_local = particles.localOffset();

        const double coeff_h_n = 4.0 / 3.0 * Es * Kokkos::sqrt( Rs );
        const double coeff_h_d = -2.0 * Kokkos::sqrt( 5.0 / 6.0 ) * beta;

        const auto vol = particles.sliceVolume();
        const auto rho = particles.sliceDensity();
        const auto vel = particles.sliceVelocity();

        base_type::update( particles, max_displacement );

        auto contact_full = KOKKOS_LAMBDA( const int i, const int j )
        {
            double fcx_i = 0.0;
            double fcy_i = 0.0;
            double fcz_i = 0.0;

            double xi, r, s;
            double rx, ry, rz;
            getDistanceComponents( x, u, i, j, xi, r, s, rx, ry, rz );

            // Contact "overlap"
            const double delta_n = ( r - 2.0 * radius );

            // Hertz normal force coefficient
            double coeff = 0.0;
            double Sn = 0.0;
            if ( delta_n < 0.0 )
            {
                coeff = Kokkos::min(
                    0.0, -coeff_h_n *
                             Kokkos::pow( Kokkos::abs( delta_n ), 3.0 / 2.0 ) );
                Sn = 2.0 * Es * Kokkos::sqrt( Rs * Kokkos::abs( delta_n ) );
            }

            coeff /= vol( i );

            fcx_i = coeff * rx / r;
            fcy_i = coeff * ry / r;
            fcz_i = coeff * rz / r;

            fc( i, 0 ) += fcx_i;
            fc( i, 1 ) += fcy_i;
            fc( i, 2 ) += fcz_i;

            // Hertz normal force damping component
            double vx, vy, vz, vn;
            getRelativeNormalVelocityComponents( vel, i, j, rx, ry, rz, r, vx,
                                                 vy, vz, vn );

            double ms = ( rho( i ) * vol( i ) ) / 2.0;
            double fnd = coeff_h_d * Kokkos::sqrt( Sn * ms ) * vn / vol( i );

            fcx_i = fnd * rx / r;
            fcy_i = fnd * ry / r;
            fcz_i = fnd * rz / r;

            fc( i, 0 ) += fcx_i;
            fc( i, 1 ) += fcy_i;
            fc( i, 2 ) += fcz_i;
        };

        _timer.start();

        // FIXME: using default space for now.
        using exec_space = typename MemorySpace::execution_space;
        Kokkos::RangePolicy<exec_space> policy( n_frozen, n_local );
        Cabana::neighbor_parallel_for(
            policy, contact_full, _neigh_list, Cabana::FirstNeighborsTag(),
            neigh_op_tag, "CabanaPD::Contact::compute_full" );

        _timer.stop();
    }

    // FIXME: implement energy
    template <class PosType, class WType, class ParticleType,
              class ParallelType>
    double computeEnergyFull( WType&, const PosType&, const PosType&,
                              ParticleType&, ParallelType& )
    {
        return 0.0;
    }

  protected:
    HertzianModel _model;
    using base_type::_half_neigh;
    using base_type::_neigh_list;
    using base_type::_timer;
};

} // namespace CabanaPD

#endif
