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

#ifndef CONTACT_H
#define CONTACT_H

#include <cmath>

#include <CabanaPD_Force.hpp>
#include <CabanaPD_Input.hpp>
#include <CabanaPD_Output.hpp>

namespace CabanaPD
{
/******************************************************************************
  Normal repulsion forces
******************************************************************************/
template <class MemorySpace>
class Force<MemorySpace, NormalRepulsionModel>
    : public Force<MemorySpace, BaseForceModel>
{
  public:
    using base_type = Force<MemorySpace, BaseForceModel>;
    using neighbor_list_type = typename base_type::neighbor_list_type;

    template <class ParticleType>
    Force( const bool half_neigh, const ParticleType& particles,
           const NormalRepulsionModel model )
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
        auto delta = _model.delta;
        auto Rc = _model.Rc;
        auto c = _model.c;
        const auto vol = particles.sliceVolume();
        const auto y = particles.sliceCurrentPosition();

        _neigh_timer.start();
        _neigh_list.build( y, 0, n_local, Rc, 1.0, mesh_min, mesh_max );
        _neigh_timer.stop();

        auto contact_full = KOKKOS_LAMBDA( const int i, const int j )
        {
            double fcx_i = 0.0;
            double fcy_i = 0.0;
            double fcz_i = 0.0;

            double xi, r, s;
            double rx, ry, rz;
            getDistanceComponents( x, u, i, j, xi, r, s, rx, ry, rz );

            // Contact "stretch"
            const double sc = ( r - Rc ) / delta;

            // Normal repulsion uses a 15 factor compared to the PMB force
            const double coeff = 15 * c * sc * vol( j );
            fcx_i = coeff * rx / r;
            fcy_i = coeff * ry / r;
            fcz_i = coeff * rz / r;

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
    NormalRepulsionModel _model;
    using base_type::_half_neigh;
    using base_type::_neigh_list;
    using base_type::_timer;
    Timer _neigh_timer;

    double mesh_max[3];
    double mesh_min[3];
};

} // namespace CabanaPD

#endif
