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
#include <force_models/CabanaPD_Hertzian.hpp>

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
        const int n_frozen = particles.frozenOffset();
        const int n_local = particles.localOffset();

        auto model = _model;
        const auto vol = particles.sliceVolume();
        const auto rho = particles.sliceDensity();
        const auto vel = particles.sliceVelocity();

        base_type::update( particles, max_displacement );

        auto contact_full = KOKKOS_LAMBDA( const int i, const int j )
        {
            double xi, r, s;
            double rx, ry, rz;
            getDistanceComponents( x, u, i, j, xi, r, s, rx, ry, rz );

            // Hertz normal force damping component
            double vx, vy, vz, vn;
            getRelativeNormalVelocityComponents( vel, i, j, rx, ry, rz, r, vx,
                                                 vy, vz, vn );

            const double coeff = model.forceCoeff( r, vn, vol( i ), rho( i ) );
            fc( i, 0 ) += coeff * rx / r;
            fc( i, 1 ) += coeff * ry / r;
            fc( i, 2 ) += coeff * rz / r;
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
    using base_type::_neigh_timer;
    using base_type::_timer;
};

} // namespace CabanaPD

#endif
