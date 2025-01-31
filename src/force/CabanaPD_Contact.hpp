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

/******************************************************************************
  Normal repulsion forces
******************************************************************************/
template <class MemorySpace, class ModelType>
class Force<MemorySpace, ModelType, NormalRepulsionModel, NoFracture>
    : public BaseForce<MemorySpace>
{
  public:
    using base_type = BaseForce<MemorySpace>;

    Force( const NormalRepulsionModel model )
        : _model( model )
    {
    }

    template <class ForceType, class PosType, class ParticleType,
              class NeighborType>
    void computeForceFull( ForceType& fc, const PosType& x, const PosType& u,
                           ParticleType& particles, NeighborType& neighbor )
    {
        auto model = _model;
        const auto vol = particles.sliceVolume();

        neighbor.update( particles, model.cutoff(), model.extend() );

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
        neighbor.iterate( exec_space{}, contact_full, particles,
                          "CabanaPD::Contact::compute_full" );
        Kokkos::fence();

        _timer.stop();
    }

    // FIXME: implement energy
    template <class PosType, class WType, class ParticleType,
              class NeighborType>
    void computeEnergyFull( WType&, const PosType&, const PosType&,
                            ParticleType&, const int, NeighborType& )
    {
    }

  protected:
    ModelType _model;
    using base_type::_timer;
};

/******************************************************************************
  Hertzian contact forces
******************************************************************************/
template <class MemorySpace, class ModelType>
class Force<MemorySpace, ModelType, HertzianModel, NoFracture>
    : public BaseForce<MemorySpace>
{
  public:
    using base_type = BaseForce<MemorySpace>;

    Force( const ModelType model )
        : _model( model )
    {
    }

    template <class ForceType, class PosType, class ParticleType,
              class NeighborType>
    void computeForceFull( ForceType& fc, const PosType& x, const PosType& u,
                           const ParticleType& particles,
                           NeighborType& neighbor )
    {
        auto model = _model;
        const auto vol = particles.sliceVolume();
        const auto rho = particles.sliceDensity();
        const auto vel = particles.sliceVelocity();

        neighbor.update( particles, model.cutoff(), model.extend() );

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
        neighbor.iterate( exec_space{}, contact_full, particles,
                          "CabanaPD::Contact::compute_full" );
        Kokkos::fence();

        _timer.stop();
    }

    // FIXME: implement energy
    template <class PosType, class WType, class ParticleType,
              class ParallelType>
    void computeEnergyFull( WType&, const PosType&, const PosType&,
                            ParticleType&, ParallelType& )
    {
    }

  protected:
    ModelType _model;
    using base_type::_timer;
    using base_type::_total_strain_energy;
};

} // namespace CabanaPD

#endif
