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
#include <force_models/CabanaPD_Contact.hpp>

namespace CabanaPD
{
/******************************************************************************
Contact helper functions
******************************************************************************/
template <class VelType>
KOKKOS_INLINE_FUNCTION void getRelativeNormalVelocityComponents(
    const VelType& vel, const int i, const int j, const double rx,
    const double ry, const double rz, const double r, double& vx, double& vy,
    double& vz, double& vn )
{
    vx = vel( i, 0 ) - vel( j, 0 );
    vy = vel( i, 1 ) - vel( j, 1 );
    vz = vel( i, 2 ) - vel( j, 2 );

    vn = vx * rx + vy * ry + vz * rz;
    vn /= r;
};

// Contact forces base class.
template <class MemorySpace>
class BaseForceContact : public Force<MemorySpace, BaseForceModel>
{
  public:
    using base_type = Force<MemorySpace, BaseForceModel>;
    using neighbor_list_type = typename base_type::neighbor_list_type;

    // NOTE: using 2x radius to find neighbors when particles first touch.
    template <class ParticleType, class ModelType>
    BaseForceContact( const bool half_neigh, const ParticleType& particles,
                      const ModelType model )
        : base_type( half_neigh, 2.0 * model.radius + model.radius_extend,
                     particles.sliceCurrentPosition(), particles.frozenOffset(),
                     particles.localOffset(), particles.ghost_mesh_lo,
                     particles.ghost_mesh_hi )
        , radius( 2.0 * model.radius )
        , radius_extend( model.radius_extend )
    {
        for ( int d = 0; d < particles.dim; d++ )
        {
            mesh_min[d] = particles.ghost_mesh_lo[d];
            mesh_max[d] = particles.ghost_mesh_hi[d];
        }
    }

    // Only rebuild neighbor list as needed.
    template <class ParticleType>
    void update( const ParticleType& particles, const double max_displacement,
                 const bool require_update = false )
    {
        if ( max_displacement > radius_extend || require_update )
        {
            _neigh_timer.start();
            const auto y = particles.sliceCurrentPosition();
            _neigh_list.build( y, particles.frozenOffset(),
                               particles.localOffset(), radius + radius_extend,
                               1.0, mesh_min, mesh_max );
            // Reset neighbor update displacement.
            const auto u = particles.sliceDisplacement();
            auto u_neigh = particles.sliceDisplacementNeighborBuild();
            Cabana::deep_copy( u_neigh, u );
            _neigh_timer.stop();
        }
    }

    auto timeNeighbor() { return _neigh_timer.time(); };

  protected:
    double radius;
    double radius_extend;
    Timer _neigh_timer;

    using base_type::_half_neigh;
    using base_type::_neigh_list;
    double mesh_max[3];
    double mesh_min[3];
};

/******************************************************************************
  Normal repulsion forces
******************************************************************************/
template <class MemorySpace>
class Force<MemorySpace, NormalRepulsionModel>
    : public BaseForceContact<MemorySpace>
{
  public:
    using base_type = BaseForceContact<MemorySpace>;
    using neighbor_list_type = typename base_type::neighbor_list_type;

    template <class ParticleType>
    Force( const bool half_neigh, const ParticleType& particles,
           const NormalRepulsionModel model )
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
        auto model = _model;
        const auto vol = particles.sliceVolume();
        const auto y = particles.sliceCurrentPosition();
        const int n_frozen = particles.frozenOffset();
        const int n_local = particles.localOffset();

        base_type::update( particles, max_displacement );

        auto contact_full = KOKKOS_LAMBDA( const int i, const int j )
        {
            double fcx_i = 0.0;
            double fcy_i = 0.0;
            double fcz_i = 0.0;

            double xi, r, s;
            double rx, ry, rz;
            getDistanceComponents( x, u, i, j, xi, r, s, rx, ry, rz );

            const double coeff = model.forceCoeff( r, vol( j ) );
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
                              ParticleType&, const int, ParallelType& )
    {
        return 0.0;
    }

  protected:
    NormalRepulsionModel _model;
    using base_type::_half_neigh;
    using base_type::_neigh_list;
    using base_type::_neigh_timer;
    using base_type::_timer;

    double mesh_max[3];
    double mesh_min[3];
};

} // namespace CabanaPD

#endif
