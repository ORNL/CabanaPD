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

#ifndef FORCE_LPS_H
#define FORCE_LPS_H

#include <cmath>
#include <type_traits>
#include <utility>

#include <CabanaPD_Force.hpp>
#include <CabanaPD_ForceModels.hpp>
#include <CabanaPD_Particles.hpp>
#include <CabanaPD_Types.hpp>
#include <force_models/CabanaPD_LPS.hpp>

namespace CabanaPD
{
namespace detail
{
template <class T, class = void>
struct has_n_version_force_coeff : std::false_type
{
};

template <class T>
struct has_n_version_force_coeff<
    T, std::void_t<decltype( std::declval<const T&>()(
           ForceCoeffTag{}, SingleMaterial{}, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0,
           0.0, 0.0 ) )>> : std::true_type
{
};

template <class T, class = void>
struct has_n_version_energy : std::false_type
{
};

template <class T>
struct has_n_version_energy<
    T, std::void_t<decltype( std::declval<const T&>()(
           EnergyTag{}, SingleMaterial{}, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0,
           0.0 ) )>> : std::true_type
{
};
} // namespace detail

template <class MemorySpace>
class Force<MemorySpace, LPS, NoFracture> : public BaseForce<MemorySpace>
{
  protected:
    using base_type = BaseForce<MemorySpace>;
    using base_type::_energy_timer;
    using base_type::_stress_timer;
    using base_type::_timer;
    using base_type::_total_strain_energy;

  public:
    Force() = default;

    // Using the default exec_space.
    using exec_space = typename MemorySpace::execution_space;

    template <class ModelType, class ParticleType, class NeighborType>
    void computeWeightedVolume( const ModelType& model, ParticleType& particles,
                                const NeighborType& neighbor )
    {
        _timer.start();

        auto x = particles.sliceReferencePosition();
        auto u = particles.sliceDisplacement();
        const auto vol = particles.sliceVolume();
        auto m = particles.sliceWeightedVolume();
        Cabana::deep_copy( m, 0.0 );

        auto weighted_volume = KOKKOS_LAMBDA( const int i, const int j )
        {
            // Get the reference positions and displacements.
            double xi, r, s;
            getDistance( x, u, i, j, xi, r, s );
            m( i ) += model( WeightedVolumeTag{}, i, j, xi, vol( j ) );
        };
        neighbor.iterate( exec_space{}, weighted_volume, particles,
                          "CabanaPD::Dilatation::computeWeightedVolume" );
        Kokkos::fence();
        _timer.stop();
    }

    template <class ModelType, class ParticleType, class NeighborType>
    void computeDilatation( const ModelType& model, ParticleType& particles,
                            const NeighborType neighbor )
    {
        _timer.start();

        const auto x = particles.sliceReferencePosition();
        auto u = particles.sliceDisplacement();
        const auto vol = particles.sliceVolume();
        auto m = particles.sliceWeightedVolume();
        auto theta = particles.sliceDilatation();
        Cabana::deep_copy( theta, 0.0 );

        auto dilatation = KOKKOS_LAMBDA( const int i, const int j )
        {
            // Get the bond distance, displacement, and stretch.
            double xi, r, s;
            getDistance( x, u, i, j, xi, r, s );
            theta( i ) +=
                model( DilatationTag{}, i, j, s, xi, vol( j ), m( i ) );
        };

        neighbor.iterate( exec_space{}, dilatation, particles,
                          "CabanaPD::Dilatation::compute" );
        Kokkos::fence();

        _timer.stop();
    }

    template <class ModelType, class ForceType, class PosType,
              class ParticleType, class NeighborType>
    void computeForceFull( const ModelType& model, ForceType& f,
                           const PosType& x, const PosType& u,
                           const ParticleType& particles,
                           NeighborType& neighbor )
    {
        _timer.start();

        const auto vol = particles.sliceVolume();
        auto theta = particles.sliceDilatation();
        auto m = particles.sliceWeightedVolume();

        if constexpr ( detail::has_n_version_force_coeff<ModelType>::value )
        {
            using neighbor_list_type = typename NeighborType::list_type;
            const auto neigh_list = neighbor.list();

            auto force_full = KOKKOS_LAMBDA( const int i )
            {
                std::size_t num_neighbors =
                    Cabana::NeighborList<neighbor_list_type>::numNeighbor(
                        neigh_list, i );
                // Note: Cabana::VerletList internal storage is not directly accessible.
                // numNeighbor() should return the actual count stored for particle i.
                // If numNeighbor() exceeds the allocated neighbor list storage, getNeighbor()
                // will fail or return invalid data, which is a bug that must be caught.
                for ( std::size_t n = 0; n < num_neighbors; n++ )
                {
                    double fx_i = 0.0;
                    double fy_i = 0.0;
                    double fz_i = 0.0;

                    std::size_t j =
                        Cabana::NeighborList<neighbor_list_type>::getNeighbor(
                            neigh_list, i, n );

                    double xi, r, s;
                    double rx, ry, rz;
                    getDistance( x, u, i, j, xi, r, s, rx, ry, rz );

                    const double coeff = model( ForceCoeffTag{}, SingleMaterial{},
                                                i, j, n, s, xi, vol( j ), m( i ),
                                                m( j ), theta( i ), theta( j ) );
                    fx_i = coeff * rx / r;
                    fy_i = coeff * ry / r;
                    fz_i = coeff * rz / r;

                    f( i, 0 ) += fx_i;
                    f( i, 1 ) += fy_i;
                    f( i, 2 ) += fz_i;
                }
            };

            neighbor.iterateLinear( exec_space{}, force_full, particles,
                                    "CabanaPD::ForceLPS::computeFull" );
        }
        else
        {
            auto force_full = KOKKOS_LAMBDA( const int i, const int j )
            {
                double fx_i = 0.0;
                double fy_i = 0.0;
                double fz_i = 0.0;

                double xi, r, s;
                double rx, ry, rz;
                getDistance( x, u, i, j, xi, r, s, rx, ry, rz );

                const double coeff =
                    model( ForceCoeffTag{}, SingleMaterial{}, i, j, s, xi, vol( j ),
                           m( i ), m( j ), theta( i ), theta( j ) );
                fx_i = coeff * rx / r;
                fy_i = coeff * ry / r;
                fz_i = coeff * rz / r;

                f( i, 0 ) += fx_i;
                f( i, 1 ) += fy_i;
                f( i, 2 ) += fz_i;
            };

            neighbor.iterate( exec_space{}, force_full, particles,
                              "CabanaPD::ForceLPS::computeFull" );
        }
        Kokkos::fence();
        _timer.stop();
    }

    template <class ModelType, class PosType, class WType, class ParticleType,
              class NeighborType>
    void computeEnergyFull( const ModelType& model, WType& W, const PosType& x,
                            const PosType& u, const ParticleType& particles,
                            NeighborType& neighbor )
    {
        _energy_timer.start();

        const auto vol = particles.sliceVolume();
        const auto theta = particles.sliceDilatation();
        auto m = particles.sliceWeightedVolume();
        using neighbor_list_type = typename NeighborType::list_type;
        const auto neigh_list = neighbor.list();

        if constexpr ( detail::has_n_version_energy<ModelType>::value )
        {
            auto energy_full = KOKKOS_LAMBDA( const int i, double& Phi )
            {
                std::size_t num_neighbors =
                    Cabana::NeighborList<neighbor_list_type>::numNeighbor(
                        neigh_list, i );
                // Note: Cabana::VerletList internal storage is not directly accessible.
                // numNeighbor() should return the actual count stored for particle i.
                // If numNeighbor() exceeds the allocated neighbor list storage, getNeighbor()
                // will fail or return invalid data, which is a bug that must be caught.
                double num_bonds = static_cast<double>( num_neighbors );

                for ( std::size_t n = 0; n < num_neighbors; n++ )
                {
                    std::size_t j =
                        Cabana::NeighborList<neighbor_list_type>::getNeighbor(
                            neigh_list, i, n );

                    double xi, r, s;
                    getDistance( x, u, i, j, xi, r, s );

                    double w = model( EnergyTag{}, SingleMaterial{}, i, j, n, s,
                                      xi, vol( j ), m( i ), theta( i ), num_bonds );
                    W( i ) += w;
                }
                Phi += W( i ) * vol( i );
            };

            neighbor.reduceLinear( exec_space{}, energy_full, particles,
                                  "CabanaPD::ForceLPS::computeEnergyFull",
                                  _total_strain_energy );
        }
        else
        {
            auto energy_full =
                KOKKOS_LAMBDA( const int i, const int j, double& Phi )
            {
                double xi, r, s;
                getDistance( x, u, i, j, xi, r, s );

                double num_neighbors = static_cast<double>(
                    Cabana::NeighborList<neighbor_list_type>::numNeighbor(
                        neigh_list, i ) );

                double w = model( EnergyTag{}, SingleMaterial{}, i, j, s, xi,
                                  vol( j ), m( i ), theta( i ), num_neighbors );
                W( i ) += w;
                Phi += w * vol( i );
            };

            neighbor.reduce( exec_space{}, energy_full, particles,
                             "CabanaPD::ForceLPS::computeEnergyFull",
                             _total_strain_energy );
        }
        Kokkos::fence();

        _energy_timer.stop();
    }

    template <class ModelType, class ParticleType, class NeighborType>
    void computeStressFull( const ModelType& model, ParticleType& particles,
                            NeighborType& neighbor )
    {
        _stress_timer.start();

        const auto x = particles.sliceReferencePosition();
        const auto u = particles.sliceDisplacement();
        const auto vol = particles.sliceVolume();
        const auto theta = particles.sliceDilatation();
        const auto m = particles.sliceWeightedVolume();
        auto stress = particles.sliceStress();

        if constexpr ( detail::has_n_version_force_coeff<ModelType>::value )
        {
            using neighbor_list_type = typename NeighborType::list_type;
            const auto neigh_list = neighbor.list();

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

                    double xi, r, s;
                    double rx, ry, rz;
                    double xi_x, xi_y, xi_z;
                    getDistance( x, u, i, j, xi, r, s, rx, ry, rz, xi_x, xi_y,
                                 xi_z );

                    double coeff = model( ForceCoeffTag{}, SingleMaterial{}, i,
                                           j, n, s, xi, vol( j ), m( i ), m( j ),
                                           theta( i ), theta( j ) );
                    coeff *= 0.5;
                    const double fx_i = coeff * rx / r;
                    const double fy_i = coeff * ry / r;
                    const double fz_i = coeff * rz / r;

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
                }
            };
            neighbor.iterateLinear( exec_space{}, stress_full, particles,
                                    "CabanaPD::ForceLPS::computeStressFull" );
        }
        else
        {
            auto stress_full = KOKKOS_LAMBDA( const int i, const int j )
            {
                double xi, r, s;
                double rx, ry, rz;
                double xi_x, xi_y, xi_z;
                getDistance( x, u, i, j, xi, r, s, rx, ry, rz, xi_x, xi_y, xi_z );

                double coeff =
                    model( ForceCoeffTag{}, SingleMaterial{}, i, j, s, xi, vol( j ),
                           m( i ), m( j ), theta( i ), theta( j ) );
                coeff *= 0.5;
                const double fx_i = coeff * rx / r;
                const double fy_i = coeff * ry / r;
                const double fz_i = coeff * rz / r;

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
            neighbor.iterate( exec_space{}, stress_full, particles,
                              "CabanaPD::ForceLPS::computeStressFull" );
        }
        Kokkos::fence();

        _stress_timer.stop();
    }
};

template <class MemorySpace>
class Force<MemorySpace, LPS, Fracture> : public BaseForce<MemorySpace>
{
  protected:
    using base_type = BaseForce<MemorySpace>;
    using base_type::_energy_timer;
    using base_type::_stress_timer;
    using base_type::_timer;
    using base_type::_total_strain_energy;
    double _total_damage;

  public:
    Force() = default;

    // Using the default exec_space.
    using exec_space = typename MemorySpace::execution_space;

    template <class ModelType, class ParticleType, class NeighborType>
    void computeWeightedVolume( const ModelType& model, ParticleType& particles,
                                const NeighborType& neighbor )
    {
        _timer.start();

        using neighbor_list_type = typename NeighborType::list_type;
        const auto neigh_list = neighbor.list();
        const auto mu = neighbor.brokenBonds();

        auto x = particles.sliceReferencePosition();
        auto u = particles.sliceDisplacement();
        const auto vol = particles.sliceVolume();
        auto m = particles.sliceWeightedVolume();
        Cabana::deep_copy( m, 0.0 );

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
                m( i ) += mu( i, n ) *
                          model( WeightedVolumeTag{}, i, j, xi, vol( j ) );
            }
        };

        neighbor.iterateLinear(
            exec_space{}, weighted_volume, particles,
            "CabanaPD::ForceLPSDamage::computeWeightedVolume" );
        Kokkos::fence();

        _timer.stop();
    }

    template <class ModelType, class ParticleType, class NeighborType>
    void computeDilatation( const ModelType& model, ParticleType& particles,
                            const NeighborType& neighbor )
    {
        _timer.start();
        using neighbor_list_type = typename NeighborType::list_type;
        const auto neigh_list = neighbor.list();
        const auto mu = neighbor.brokenBonds();

        const auto x = particles.sliceReferencePosition();
        auto u = particles.sliceDisplacement();
        const auto vol = particles.sliceVolume();
        auto m = particles.sliceWeightedVolume();
        auto theta = particles.sliceDilatation();

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
                    theta( i ) += mu( i, n ) * model( DilatationTag{}, i, j, s,
                                                      xi, vol( j ), m( i ) );
            }
        };

        neighbor.iterateLinear( exec_space{}, dilatation, particles,
                                "CabanaPD::ForceLPSDamage::computeDilatation" );
        Kokkos::fence();

        _timer.stop();
    }

    template <class ModelType, class ForceType, class PosType,
              class ParticleType, class NeighborType>
    void computeForceFull( const ModelType& model, ForceType& f,
                           const PosType& x, const PosType& u,
                           const ParticleType& particles,
                           NeighborType& neighbor )
    {
        _timer.start();
        using neighbor_list_type = typename NeighborType::list_type;
        const auto neigh_list = neighbor.list();
        const auto mu = neighbor.brokenBonds();

        const auto vol = particles.sliceVolume();
        auto theta = particles.sliceDilatation();
        auto m = particles.sliceWeightedVolume();
        const auto nofail = particles.sliceNoFail();
        auto force_full = KOKKOS_LAMBDA( const int i )
        {
            std::size_t num_neighbors =
                Cabana::NeighborList<neighbor_list_type>::numNeighbor(
                    neigh_list, i );
            // Abort check: num_neighbors must not exceed mu.extent(1).
            // mu is allocated with global max-neighbor count to match neighbor list capacity.
            // If numNeighbor() returns a value > mu.extent(1), the neighbor list storage
            // is insufficient, which is a bug that must abort (not silently clamp).
            const std::size_t mu_extent1 = static_cast<std::size_t>( mu.extent( 1 ) );
            if ( num_neighbors > mu_extent1 )
            {
                Kokkos::abort( "FATAL: num_neighbors > mu.extent(1) in computeForceFull (fracture). "
                               "num_neighbors must not exceed neighbor list storage capacity." );
            }
            for ( std::size_t n = 0; n < num_neighbors; n++ )
            {
                double fx_i = 0.0;
                double fy_i = 0.0;
                double fz_i = 0.0;

                std::size_t j =
                    Cabana::NeighborList<neighbor_list_type>::getNeighbor(
                        neigh_list, i, n );
                if ( j >= static_cast<std::size_t>( x.extent( 0 ) ) )
                {
                    Kokkos::printf( "FATAL: getNeighbor returned out-of-range j in computeForceFull (fracture). "
                                    "i=%d n=%zu j=%zu x.extent(0)=%zu\n", i, n, j,
                                    static_cast<std::size_t>( x.extent( 0 ) ) );
                    Kokkos::abort( "FATAL: getNeighbor returned out-of-range j" );
                }

                // Get the reference positions and displacements.
                double xi, r, s;
                double rx, ry, rz;
                getDistance( x, u, i, j, xi, r, s, rx, ry, rz );

                // Break if beyond critical stretch unless in no-fail zone.
                if ( model( CriticalStretchTag{}, i, j, r, xi ) &&
                     !nofail( i ) && !nofail( i ) )
                {
                    mu( i, n ) = 0;
                }
                // Check if this bond is broken (mu=0) to ensure m(i) and m(j)
                // are both >0 (m=0 only occurs when all bonds are broken) to
                // avoid dividing by zero.
                else if ( mu( i, n ) > 0 )
                {
                    double coeff = 0.0;
                    if constexpr ( detail::has_n_version_force_coeff<ModelType>::value )
                    {
                        coeff = model( ForceCoeffTag{}, SingleMaterial{}, i, j, static_cast<int>(n),
                                     s, xi, vol( j ), m( i ), m( j ), theta( i ), theta( j ) );
                    }
                    else
                    {
                        coeff = model( ForceCoeffTag{}, SingleMaterial{}, i, j,
                                     s, xi, vol( j ), m( i ), m( j ), theta( i ), theta( j ) );
                    }
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

        neighbor.iterateLinear( exec_space{}, force_full, particles,
                                "CabanaPD::ForceLPSDamage::computeFull" );
        Kokkos::fence();

        _timer.stop();
    }

    template <class ModelType, class PosType, class WType, class ParticleType,
              class NeighborType>
    void computeEnergyFull( const ModelType& model, WType& W, const PosType& x,
                            const PosType& u, ParticleType& particles,
                            const NeighborType& neighbor )
    {
        _energy_timer.start();
        using neighbor_list_type = typename NeighborType::list_type;
        const auto neigh_list = neighbor.list();
        const auto mu = neighbor.brokenBonds();

        const auto vol = particles.sliceVolume();
        const auto theta = particles.sliceDilatation();
        auto m = particles.sliceWeightedVolume();
        auto phi = particles.sliceDamage();

        auto energy_full = KOKKOS_LAMBDA( const int i, double& Phi, double& D )
        {
            std::size_t num_neighbors =
                Cabana::NeighborList<neighbor_list_type>::numNeighbor(
                    neigh_list, i );
            // Abort check: num_neighbors must not exceed mu.extent(1).
            // mu is allocated with global max-neighbor count to match neighbor list capacity.
            // If numNeighbor() returns a value > mu.extent(1), the neighbor list storage
            // is insufficient, which is a bug that must abort (not silently clamp).
            const std::size_t mu_extent1 = static_cast<std::size_t>( mu.extent( 1 ) );
            if ( num_neighbors > mu_extent1 )
            {
                Kokkos::abort( "FATAL: num_neighbors > mu.extent(1) in computeEnergyFull (fracture). "
                               "num_neighbors must not exceed neighbor list storage capacity." );
            }
            double num_bonds = 0.0;
            for ( std::size_t n = 0; n < num_neighbors; n++ )
                num_bonds += static_cast<double>( mu( i, n ) );

            double phi_i = 0.0;
            double vol_H_i = 0.0;
            for ( std::size_t n = 0; n < num_neighbors; n++ )
            {
                std::size_t j =
                    Cabana::NeighborList<neighbor_list_type>::getNeighbor(
                        neigh_list, i, n );
                if ( j >= static_cast<std::size_t>( x.extent( 0 ) ) )
                {
                    Kokkos::printf( "FATAL: getNeighbor returned out-of-range j in computeEnergyFull (fracture). "
                                    "i=%d n=%zu j=%zu x.extent(0)=%zu\n", i, n, j,
                                    static_cast<std::size_t>( x.extent( 0 ) ) );
                    Kokkos::abort( "FATAL: getNeighbor returned out-of-range j" );
                }
                // Get the bond distance, displacement, and stretch.
                double xi, r, s;
                getDistance( x, u, i, j, xi, r, s );

                double w_model = 0.0;
                if constexpr ( detail::has_n_version_energy<ModelType>::value )
                {
                    w_model = model( EnergyTag{}, SingleMaterial{}, i, j, static_cast<int>(n),
                                   s, xi, vol( j ), m( i ), theta( i ), num_bonds );
                }
                else
                {
                    w_model = model( EnergyTag{}, SingleMaterial{}, i, j,
                                   s, xi, vol( j ), m( i ), theta( i ), num_bonds );
                }
                double w = mu( i, n ) * w_model;
                W( i ) += w;

                phi_i += mu( i, n ) * vol( j );
                vol_H_i += vol( j );
            }
            Phi += W( i ) * vol( i );
            phi( i ) = 1 - phi_i / vol_H_i;
            D += phi( i );
        };

        neighbor.reduceLinear( exec_space{}, energy_full, particles,
                               "CabanaPD::ForceLPSDamage::computeEnergyFull",
                               _total_strain_energy, _total_damage );
        Kokkos::fence();

        _energy_timer.stop();
    }

    template <class ModelType, class ParticleType, class NeighborType>
    void computeStressFull( const ModelType& model, ParticleType& particles,
                            NeighborType& neighbor )
    {
        _stress_timer.start();
        using neighbor_list_type = typename NeighborType::list_type;
        const auto neigh_list = neighbor.list();
        const auto mu = neighbor.brokenBonds();

        const auto x = particles.sliceReferencePosition();
        const auto u = particles.sliceDisplacement();
        const auto vol = particles.sliceVolume();
        const auto theta = particles.sliceDilatation();
        const auto m = particles.sliceWeightedVolume();
        auto stress = particles.sliceStress();

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

                if ( mu( i, n ) > 0 ) // Only compute stress for unbroken bonds
                {
                    double xi, r, s;
                    double rx, ry, rz;
                    double xi_x, xi_y, xi_z;
                    getDistance( x, u, i, j, xi, r, s, rx, ry, rz, xi_x, xi_y,
                                 xi_z );

                    double coeff = 0.0;
                    if constexpr ( detail::has_n_version_force_coeff<ModelType>::value )
                    {
                        coeff = model( ForceCoeffTag{}, SingleMaterial{}, i, j, static_cast<int>(n),
                                     s, xi, vol( j ), m( i ), m( j ), theta( i ), theta( j ) );
                    }
                    else
                    {
                        coeff = model( ForceCoeffTag{}, SingleMaterial{}, i, j,
                                     s, xi, vol( j ), m( i ), m( j ), theta( i ), theta( j ) );
                    }
                    coeff *= 0.5;
                    const double muij = mu( i, n );
                    const double fx_i = muij * coeff * rx / r;
                    const double fy_i = muij * coeff * ry / r;
                    const double fz_i = muij * coeff * rz / r;

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
                }
            }
        };
        neighbor.iterateLinear(
            exec_space{}, stress_full, particles,
            "CabanaPD::ForceLPSFracture::computeStressFull" );
        Kokkos::fence();

        _stress_timer.stop();
    }

    auto totalDamage() { return _total_damage; };
};

template <class MemorySpace>
class Force<MemorySpace, LinearLPS, NoFracture>
    : public Force<MemorySpace, LPS, NoFracture>
{
  protected:
    using base_type = Force<MemorySpace, LPS, NoFracture>;
    using base_type::_energy_timer;
    using base_type::_stress_timer;
    using base_type::_timer;
    using base_type::_total_strain_energy;

  public:
    Force() = default;

    // Using the default exec_space.
    using exec_space = typename MemorySpace::execution_space;

    template <class ModelType, class ForceType, class PosType,
              class ParticleType, class NeighborType>
    void computeForceFull( const ModelType& model, ForceType& f,
                           const PosType& x, const PosType& u,
                           const ParticleType& particles,
                           NeighborType& neighbor )
    {
        _timer.start();

        const auto vol = particles.sliceVolume();
        auto theta = particles.sliceDilatation();
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
            getLinearizedDistance( x, u, i, j, xi, linear_s, xi_x, xi_y, xi_z );

            const double coeff =
                model( ForceCoeffTag{}, SingleMaterial{}, i, j, linear_s, xi,
                       vol( j ), m( i ), m( j ), theta( i ), theta( j ) );
            fx_i = coeff * xi_x / xi;
            fy_i = coeff * xi_y / xi;
            fz_i = coeff * xi_z / xi;

            f( i, 0 ) += fx_i;
            f( i, 1 ) += fy_i;
            f( i, 2 ) += fz_i;
        };

        neighbor.iterate( exec_space{}, force_full, particles,
                          "CabanaPD::ForceLPS::computeFull" );
        Kokkos::fence();
        _timer.stop();
    }

    template <class ModelType, class PosType, class WType, class ParticleType,
              class NeighborType>
    void computeEnergyFull( const ModelType& model, WType& W, const PosType& x,
                            const PosType& u, const ParticleType& particles,
                            NeighborType& neighbor )
    {
        _energy_timer.start();

        using neighbor_list_type = typename NeighborType::list_type;
        const auto neigh_list = neighbor.list();

        const auto vol = particles.sliceVolume();
        const auto theta = particles.sliceDilatation();
        auto m = particles.sliceWeightedVolume();

        auto energy_full =
            KOKKOS_LAMBDA( const int i, const int j, double& Phi )
        {
            // Do we need to recompute linear_theta_i?
            double xi, linear_s;
            getLinearizedDistance( x, u, i, j, xi, linear_s );

            double num_neighbors = static_cast<double>(
                Cabana::NeighborList<neighbor_list_type>::numNeighbor(
                    neigh_list, i ) );

            double w = model( EnergyTag{}, SingleMaterial{}, i, j, linear_s, xi,
                              vol( j ), m( i ), theta( i ), num_neighbors );
            W( i ) += w;
            Phi += w * vol( i );
        };

        neighbor.reduce( exec_space{}, energy_full, particles,
                         "CabanaPD::ForceLPS::computeEnergyFull",
                         _total_strain_energy );
        Kokkos::fence();

        _energy_timer.stop();
    }

    template <class ModelType, class ParticleType, class NeighborType>
    void computeStressFull( const ModelType& model, ParticleType& particles,
                            NeighborType& neighbor )
    {
        _stress_timer.start();

        const auto x = particles.sliceReferencePosition();
        const auto u = particles.sliceDisplacement();
        const auto vol = particles.sliceVolume();
        const auto theta = particles.sliceDilatation();
        const auto m = particles.sliceWeightedVolume();
        auto stress = particles.sliceStress();

        auto stress_full = KOKKOS_LAMBDA( const int i, const int j )
        {
            // Get the linearized components
            double xi, linear_s;
            double xi_x, xi_y, xi_z;
            getLinearizedDistance( x, u, i, j, xi, linear_s, xi_x, xi_y, xi_z );

            double coeff =
                model( ForceCoeffTag{}, SingleMaterial{}, i, j, linear_s, xi,
                       vol( j ), m( i ), m( j ), theta( i ), theta( j ) );
            coeff *= 0.5;
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
        neighbor.iterate( exec_space{}, stress_full, particles,
                          "CabanaPD::ForceLPS::computeStressFull" );
        Kokkos::fence();

        _stress_timer.stop();
    }
};

} // namespace CabanaPD

#endif
