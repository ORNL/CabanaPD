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

#include <cmath>

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
class Force<MemorySpace, HertzianModel>
    : public Force<MemorySpace, BaseForceModel>
{
  public:
    using base_type = Force<MemorySpace, BaseForceModel>;
    using neighbor_list_type = typename base_type::neighbor_list_type;

    template <class ParticleType>
    Force( const bool half_neigh, const ParticleType& particles,
           const HertzianModel model )
        : base_type( half_neigh, model.Rc, particles.sliceCurrentPosition(),
                     particles.n_local, particles.ghost_mesh_lo,
                     particles.ghost_mesh_hi )
        , _model( model )
    {
        for ( int d = 0; d < particles.dim; d++ )
        {
            mesh_min[d] = particles.ghost_mesh_lo[d];
            mesh_max[d] = particles.ghost_mesh_hi[d];
        }
    }

    template <class ForceType, class PosType, class ParticleType,
              class ParallelType>
    void computeForceFull( ForceType& fc, const PosType& x, const PosType& u,
                           const ParticleType& particles, const int n_local,
                           ParallelType& neigh_op_tag )
    {
        auto Rc = _model.Rc;
        auto radius = _model.radius;
        auto Es = _model.Es;
        auto Rs = _model.Rs;
        auto beta = _model.beta;

        const double coeff_h_n = 4.0 / 3.0 * Es * std::sqrt( Rs );
        const double coeff_h_d = -2.0 * sqrt( 5.0 / 6.0 ) * beta;

        const auto vol = particles.sliceVolume();
        const auto rho = particles.sliceDensity();
        const auto y = particles.sliceCurrentPosition();
        const auto vel = particles.sliceVelocity();

        _neigh_timer.start();
        _neigh_list.build( y, 0, n_local, Rc, 1.0, mesh_min, mesh_max );
        _neigh_timer.stop();

        auto contact_full = KOKKOS_LAMBDA( const int i, const int j )
        {
            using Kokkos::abs;
            using Kokkos::min;
            using Kokkos::pow;
            using Kokkos::sqrt;

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
                coeff =
                    min( 0.0, -coeff_h_n * pow( abs( delta_n ), 3.0 / 2.0 ) );
                Sn = 2.0 * Es * sqrt( Rs * abs( delta_n ) );
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
            double fnd = coeff_h_d * sqrt( Sn * ms ) * vn / vol( i );

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
        Kokkos::RangePolicy<exec_space> policy( 0, n_local );
        Cabana::neighbor_parallel_for(
            policy, contact_full, _neigh_list, Cabana::FirstNeighborsTag(),
            neigh_op_tag, "CabanaPD::Contact::compute_full" );

        _timer.stop();
    }

    // FIXME: implement energy
    template <class PosType, class WType, class ParticleType,
              class ParallelType>
    double computeEnergyFull( WType&, const PosType&, const PosType&,
                              ParticleType&, const int, ParallelType& )
    {
        return 0.0;
    }

  protected:
    HertzianModel _model;
    using base_type::_half_neigh;
    using base_type::_neigh_list;
    using base_type::_timer;
    Timer _neigh_timer;

    double mesh_max[3];
    double mesh_min[3];
};

} // namespace CabanaPD

#endif
