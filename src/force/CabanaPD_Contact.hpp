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
#include <force_models/CabanaPD_Hertzian.hpp>
#include <force_models/CabanaPD_HertzianJKR.hpp>

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
    vx = vel( j, 0 ) - vel( i, 0 );
    vy = vel( j, 1 ) - vel( i, 1 );
    vz = vel( j, 2 ) - vel( i, 2 );

    vn = vx * rx + vy * ry + vz * rz;
    vn /= r;
};

// Contact forces base class.
template <class MemorySpace>
class BaseForceContact : public BaseForce<MemorySpace>
{
  public:
    using base_type = BaseForce<MemorySpace>;
    using neighbor_list_type = typename base_type::neighbor_list_type;

    // NOTE: using 2x radius to find neighbors when particles first touch.
    template <class ParticleType, class ModelType>
    BaseForceContact( const bool half_neigh, const ParticleType& particles,
                      const ModelType model )
        : base_type( half_neigh, 2.0 * model.radius + model.radius_extend,
                     particles.sliceCurrentPosition(), particles.frozenOffset(),
                     particles.localOffset(), particles.ghost_mesh_lo,
                     particles.ghost_mesh_hi )
        , search_radius( 2.0 * model.radius + model.radius_extend )
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
    void update( const ParticleType& particles,
                 const bool require_update = false )
    {
        double max_displacement = particles.getMaxDisplacement();
        if ( max_displacement > radius_extend || require_update )
        {
            _neigh_timer.start();
            const auto y = particles.sliceCurrentPosition();
            _neigh_list.build( y, particles.frozenOffset(),
                               particles.localOffset(), search_radius, 1.0,
                               mesh_min, mesh_max );
            // Reset neighbor update displacement.
            const auto u = particles.sliceDisplacement();
            auto u_neigh = particles.sliceDisplacementNeighborBuild();
            // This is not a deep_copy because they are likely different sizes.
            Kokkos::RangePolicy<typename ParticleType::execution_space> policy(
                0, u_neigh.size() );
            Kokkos::parallel_for(
                "CabanaPD::Contact::update", policy,
                KOKKOS_LAMBDA( const int p ) {
                    for ( int d = 0; d < 3; d++ )
                        u_neigh( p, d ) = u( p, d );
                } );
            Kokkos::fence();
            _neigh_timer.stop();
        }
    }

    auto timeNeighbor() { return _neigh_timer.time(); };

  protected:
    double search_radius;
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
template <class MemorySpace, class ModelType>
class Force<MemorySpace, ModelType, NormalRepulsionModel, NoFracture>
    : public BaseForceContact<MemorySpace>
{
  public:
    using base_type = BaseForceContact<MemorySpace>;
    using neighbor_list_type = typename base_type::neighbor_list_type;

    template <class ParticleType>
    Force( const bool half_neigh, const ParticleType& particles,
           const ModelType model )
        : base_type( half_neigh, particles, model )
        , _model( model )
    {
    }

    template <class ForceType, class PosType, class ParticleType,
              class ParallelType>
    void computeForceFull( ForceType& fc, const PosType& x, const PosType& u,
                           const ParticleType& particles,
                           ParallelType& neigh_op_tag )
    {
        auto model = _model;
        const auto vol = particles.sliceVolume();
        const auto y = particles.sliceCurrentPosition();
        const int n_frozen = particles.frozenOffset();
        const int n_local = particles.localOffset();

        base_type::update( particles, particles.getMaxDisplacement() );

        auto contact_full = KOKKOS_LAMBDA( const int i, const int j )
        {
            double fcx_i = 0.0;
            double fcy_i = 0.0;
            double fcz_i = 0.0;

            double xi, r, s;
            double rx, ry, rz;
            getDistance( x, u, i, j, xi, r, s, rx, ry, rz );

            if ( r < model.radius )
            {
                const double coeff = model.forceCoeff( r, vol( j ) );
                fcx_i = coeff * rx / r;
                fcy_i = coeff * ry / r;
                fcz_i = coeff * rz / r;
            }
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
        Kokkos::fence();
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
    ModelType _model;
    using base_type::_half_neigh;
    using base_type::_neigh_list;
    using base_type::_neigh_timer;
    using base_type::_timer;

    double mesh_max[3];
    double mesh_min[3];
};

/******************************************************************************
  Hertzian contact forces
******************************************************************************/
template <class MemorySpace, class ModelType>
class Force<MemorySpace, ModelType, HertzianModel, NoFracture>
    : public BaseForceContact<MemorySpace>
{
  public:
    using base_type = BaseForceContact<MemorySpace>;
    using neighbor_list_type = typename base_type::neighbor_list_type;

    template <class ParticleType>
    Force( const bool half_neigh, const ParticleType& particles,
           const ModelType model )
        : base_type( half_neigh, particles, model )
        , _model( model )
    {
    }

    template <class ForceType, class PosType, class ParticleType,
              class ParallelType>
    void computeForceFull( ForceType& fc, const PosType& x, const PosType& u,
                           const ParticleType& particles,
                           ParallelType& neigh_op_tag )
    {
        const int n_frozen = particles.frozenOffset();
        const int n_local = particles.localOffset();

        auto model = _model;
        const auto vol = particles.sliceVolume();
        const auto rho = particles.sliceDensity();
        const auto vel = particles.sliceVelocity();

        base_type::update( particles, particles.getMaxDisplacement() );

        auto contact_full = KOKKOS_LAMBDA( const int i, const int j )
        {
            double xi, r, s;
            double rx, ry, rz;
            getDistance( x, u, i, j, xi, r, s, rx, ry, rz );

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
        Kokkos::fence();
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
    ModelType _model;
    using base_type::_half_neigh;
    using base_type::_neigh_list;
    using base_type::_neigh_timer;
    using base_type::_timer;
};

} // namespace CabanaPD

#endif
