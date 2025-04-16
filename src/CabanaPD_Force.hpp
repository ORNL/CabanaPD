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

#ifndef FORCE_H
#define FORCE_H

#include <Kokkos_Core.hpp>

#include <CabanaPD_ForceModels.hpp>
#include <CabanaPD_Neighbor.hpp>
#include <CabanaPD_Particles.hpp>

namespace CabanaPD
{
/******************************************************************************
  Force helper functions.
******************************************************************************/
template <class PosType>
KOKKOS_INLINE_FUNCTION void
getDistance( const PosType& x, const PosType& u, const int i, const int j,
             double& xi, double& r, double& s, double& rx, double& ry,
             double& rz, double& xi_x, double& xi_y, double& xi_z )
{
    // Get the reference positions and displacements.
    xi_x = x( j, 0 ) - x( i, 0 );
    const double eta_u = u( j, 0 ) - u( i, 0 );
    xi_y = x( j, 1 ) - x( i, 1 );
    const double eta_v = u( j, 1 ) - u( i, 1 );
    xi_z = x( j, 2 ) - x( i, 2 );
    const double eta_w = u( j, 2 ) - u( i, 2 );
    rx = xi_x + eta_u;
    ry = xi_y + eta_v;
    rz = xi_z + eta_w;
    r = Kokkos::sqrt( rx * rx + ry * ry + rz * rz );
    xi = Kokkos::sqrt( xi_x * xi_x + xi_y * xi_y + xi_z * xi_z );
    s = ( r - xi ) / xi;
}

template <class PosType>
KOKKOS_INLINE_FUNCTION void getDistance( const PosType& x, const PosType& u,
                                         const int i, const int j, double& xi,
                                         double& r, double& s, double& rx,
                                         double& ry, double& rz )
{
    double xi_x, xi_y, xi_z;
    getDistance( x, u, i, j, xi, r, s, rx, ry, rz, xi_x, xi_y, xi_z );
}

template <class PosType>
KOKKOS_INLINE_FUNCTION void getDistance( const PosType& x, const PosType& u,
                                         const int i, const int j, double& xi,
                                         double& r, double& s )
{
    double rx, ry, rz;
    getDistance( x, u, i, j, xi, r, s, rx, ry, rz );
}

template <class PosType>
KOKKOS_INLINE_FUNCTION void
getLinearizedDistance( const PosType& x, const PosType& u, const int i,
                       const int j, double& xi, double& s, double& xi_x,
                       double& xi_y, double& xi_z )
{
    // Get the reference positions and displacements.
    xi_x = x( j, 0 ) - x( i, 0 );
    const double eta_u = u( j, 0 ) - u( i, 0 );
    xi_y = x( j, 1 ) - x( i, 1 );
    const double eta_v = u( j, 1 ) - u( i, 1 );
    xi_z = x( j, 2 ) - x( i, 2 );
    const double eta_w = u( j, 2 ) - u( i, 2 );
    xi = Kokkos::sqrt( xi_x * xi_x + xi_y * xi_y + xi_z * xi_z );
    s = ( xi_x * eta_u + xi_y * eta_v + xi_z * eta_w ) / ( xi * xi );
}

template <class PosType>
KOKKOS_INLINE_FUNCTION void
getLinearizedDistance( const PosType& x, const PosType& u, const int i,
                       const int j, double& xi, double& s )
{
    double xi_x, xi_y, xi_z;
    getLinearizedDistance( x, u, i, j, xi, s, xi_x, xi_y, xi_z );
}

// Forward declaration.
template <class MemorySpace, class ForceModelType, class ModelType,
          class FractureType = NoFracture, class DensityType = StaticDensity>
class Force;

template <class MemorySpace>
class BaseForce
{
  protected:
    Timer _timer;
    Timer _energy_timer;
    Timer _stress_timer;

  public:
    // Default to no-op.
    template <class ParticleType, class NeighborType>
    void computeWeightedVolume( ParticleType&, const NeighborType ) const
    {
    }
    template <class ParticleType, class NeighborType>
    void computeDilatation( ParticleType&, const NeighborType ) const
    {
    }

    auto time() { return _timer.time(); };
    auto timeEnergy() { return _energy_timer.time(); };
};

/******************************************************************************
  Dilatation.
******************************************************************************/
template <class MemorySpace, class ModelType, class FractureType>
class Dilatation;

template <class MemorySpace, class ModelType>
class Dilatation<MemorySpace, ModelType, NoFracture>
{
  protected:
    using model_type = ModelType;
    model_type _model;

    Timer _timer;

  public:
    // Using the default exec_space.
    using exec_space = typename MemorySpace::execution_space;

    Dilatation( const model_type model )
        : _model( model )
    {
    }

    template <class ParticleType, class NeighborType>
    void computeWeightedVolume( ParticleType& particles,
                                const NeighborType& neighbor )
    {
        _timer.start();

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
            m( i ) += model.weightedVolume( xi, vol( j ) );
        };
        neighbor.iterate( exec_space{}, weighted_volume, particles,
                          "CabanaPD::Dilatation::computeWeightedVolume" );
        _timer.stop();
    }

    template <class ParticleType, class NeighborType>
    void computeDilatation( ParticleType& particles,
                            const NeighborType neighbor )
    {
        _timer.start();

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
            theta( i ) += model.dilatation( i, s, xi, vol( j ), m( i ) );
        };

        neighbor.iterate( exec_space{}, dilatation, particles,
                          "CabanaPD::Dilatation::compute" );

        _timer.stop();
    }

    auto time() { return _timer.time(); };
};

template <class MemorySpace, class ModelType>
class Dilatation<MemorySpace, ModelType, Fracture>
{
  protected:
    using model_type = ModelType;
    model_type _model;

    Timer _timer;

  public:
    // Using the default exec_space.
    using exec_space = typename MemorySpace::execution_space;

    Dilatation( const model_type model )
        : _model( model )
    {
    }

    template <class ParticleType, class NeighborType>
    void computeWeightedVolume( ParticleType& particles,
                                const NeighborType& neighbor )
    {
        _timer.start();

        using neighbor_list_type = typename NeighborType::list_type;
        auto neigh_list = neighbor.list();
        const auto mu = neighbor.brokenBonds();

        auto x = particles.sliceReferencePosition();
        auto u = particles.sliceDisplacement();
        const auto vol = particles.sliceVolume();
        auto m = particles.sliceWeightedVolume();
        Cabana::deep_copy( m, 0.0 );
        auto model = _model;

        auto weighted_volume = KOKKOS_LAMBDA( const int i )
        {
            std::size_t num_neighbors =
                Cabana::NeighborList<neighbor_list_type>::numNeighbor(
                    neigh_list, i );
            for ( std::size_t n = 0; n < num_neighbors; n++ )
            {
                std::size_t j =
                    Cabana::NeighborList<neighbor_list_type>::getNeighbor(
                        neigh_list, i, n );

                // Get the reference positions and displacements.
                double xi, r, s;
                getDistance( x, u, i, j, xi, r, s );
                // mu is included to account for bond breaking.
                m( i ) += mu( i, n ) * model.weightedVolume( xi, vol( j ) );
            }
        };

        neighbor.iterateLinear( exec_space{}, weighted_volume, particles,
                                "CabanaPD::Dilatation::computeWeightedVolume" );

        _timer.stop();
    }

    template <class ParticleType, class NeighborType>
    void computeDilatation( ParticleType& particles,
                            const NeighborType& neighbor )
    {
        _timer.start();
        using neighbor_list_type = typename NeighborType::list_type;
        auto neigh_list = neighbor.list();
        const auto mu = neighbor.brokenBonds();

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
                Cabana::NeighborList<neighbor_list_type>::numNeighbor(
                    neigh_list, i );
            for ( std::size_t n = 0; n < num_neighbors; n++ )
            {
                std::size_t j =
                    Cabana::NeighborList<neighbor_list_type>::getNeighbor(
                        neigh_list, i, n );

                // Get the bond distance, displacement, and stretch.
                double xi, r, s;
                getDistance( x, u, i, j, xi, r, s );

                // Check if all bonds are broken (m=0) to avoid dividing by
                // zero. Alternatively, one could check if this bond mu(i,n) is
                // broken, because m=0 only occurs when all bonds are broken.
                // mu is still included to account for individual bond breaking.
                if ( m( i ) > 0 )
                    theta( i ) += mu( i, n ) * model.dilatation(
                                                   i, s, xi, vol( j ), m( i ) );
            }
        };

        neighbor.iterateLinear( exec_space{}, dilatation, particles,
                                "CabanaPD::Dilatation::compute" );

        _timer.stop();
    }
};

/******************************************************************************
  Force free functions.
******************************************************************************/
template <class ForceType, class ParticleType, class NeighborType>
void computeForce( ForceType& force, ParticleType& particles,
                   NeighborType& neighbor, const bool reset = true )
{
    auto x = particles.sliceReferencePosition();
    auto u = particles.sliceDisplacement();
    auto f = particles.sliceForce();
    auto f_a = particles.sliceForceAtomic();

    // Reset force.
    if ( reset )
        Cabana::deep_copy( f, 0.0 );

    // if ( half_neigh )
    // Forces must be atomic for half list
    // computeForce_half( f_a, x, u,
    //                    neigh_op_tag );

    // Forces only atomic if using team threading.
    if constexpr ( std::is_same<typename NeighborType::Tag,
                                Cabana::TeamOpTag>::value )
        force.computeForceFull( f_a, x, u, particles, neighbor );
    else
        force.computeForceFull( f, x, u, particles, neighbor );
    Kokkos::fence();
}

template <class ForceType, class ParticleType, class NeighborType>
double computeEnergy( ForceType& force, ParticleType& particles,
                      const NeighborType& neighbor )
{
    double energy = 0.0;
    if constexpr ( is_energy_output<typename ParticleType::output_type>::value )
    {
        auto x = particles.sliceReferencePosition();
        auto u = particles.sliceDisplacement();
        auto f = particles.sliceForce();
        auto W = particles.sliceStrainEnergy();
        auto vol = particles.sliceVolume();

        // Reset energy.
        Cabana::deep_copy( W, 0.0 );

        // if ( _half_neigh )
        //    energy = computeEnergy_half( force, x, u,
        //                                   neigh_op_tag );
        // else
        energy = force.computeEnergyFull( W, x, u, particles, neighbor );
        Kokkos::fence();
    }
    return energy;
}

template <class ForceType, class ParticleType, class ParallelType>
void computeStress( ForceType& force, ParticleType& particles,
                    const ParallelType& neigh_op_tag )
{
    if constexpr ( is_stress_output<typename ParticleType::output_type>::value )
    {
        auto stress = particles.sliceStress();

        // Reset stress.
        Cabana::deep_copy( stress, 0.0 );

        force.computeStressFull( particles, neigh_op_tag );
        Kokkos::fence();
    }
}

} // namespace CabanaPD

#endif
