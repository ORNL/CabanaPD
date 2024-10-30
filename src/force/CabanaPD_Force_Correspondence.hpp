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

#ifndef FORCE_CORR_H
#define FORCE_CORR_H

#include <cmath>

#include <CabanaPD_Force.hpp>
#include <CabanaPD_Particles.hpp>
#include <CabanaPD_Types.hpp>
#include <force/CabanaPD_ForceModels_Correspondence.hpp>

namespace CabanaPD
{
template <class ExecutionSpace>
class Force<ExecutionSpace, ForceModel<Correspondence, Elastic>>
    : public Force<ExecutionSpace, BaseForceModel>
{
  protected:
    using base_type = Force<ExecutionSpace, BaseForceModel>;
    using base_type::_half_neigh;
    ForceModel<Correspondence, Elastic> _model;

    using base_type::_energy_timer;
    using base_type::_timer;

  public:
    using exec_space = ExecutionSpace;

    Force( const bool half_neigh,
           const ForceModel<Correspondence, Elastic> model )
        : base_type( half_neigh )
        , _model( model )
    {
    }

    template <class ParticleType, class NeighListType, class ParallelType>
    void computeShapeTensor( ParticleType& particles,
                             const NeighListType& neigh_list,
                             const ParallelType neigh_op_tag )
    {
        _timer.start();

        auto n_local = particles.n_local;
        auto x = particles.sliceReferencePosition();
        auto u = particles.sliceDisplacement();
        const auto vol = particles.sliceVolume();
        auto K = particles.sliceShapeTensor();
        Cabana::deep_copy( K, 0.0 );
        auto model = _model;

        auto shape = KOKKOS_LAMBDA( const int i, const int j )
        {
            // Get the reference positions and displacements.
            double xi, r, s;
            getDistance( x, u, i, j, xi, r, s );
            double K_j = model.influence_function( xi ) * xi * xi * vol( j );
            K( i ) += K_j;
        };
        Kokkos::RangePolicy<exec_space> policy( 0, n_local );
        Cabana::neighbor_parallel_for(
            policy, shape, neigh_list, Cabana::FirstNeighborsTag(),
            neigh_op_tag, "CabanaPD::ForceCorrespondence::computeShapeTensor" );
        Kokkos::fence();

        auto invert = KOKKOS_LAMBDA( const int i )
        {
            auto k = K( i );
            auto det =
                ( k[0][0] * k[1][1] * k[2][2] - k[0][0] * k[1][2] * k[1][2] ) -
                ( k[0][1] * k[0][1] * k[2][2] - k[0][1] * k[0][2] * k[1][2] ) +
                ( k[0][2] * k[0][1] * k[1][2] - k[0][2] * k[0][2] * k[1][1] );

            K( i, 0, 0 ) = ( -k[1][2] * k[1][2] + k[1][1] * k[2] k[2] ) / det;
            K( i, 1, 1 ) = ( -k[0][2] * k[0][2] + k[0][0] * k[2] k[2] ) / det;
            K( i, 2, 2 ) = ( -k[0][1] * k[0][1] + k[0][0] * k[1] k[1] ) / det;
        };
        Kokkos::parallel_for(
            "CabanaPD::ForceCorrespondence::invertShapeTensor", policy,
            invert );

        _timer.stop();
    }

    template <class ParticleType, class NeighListType, class ParallelType>
    void computeDeformationGradient( ParticleType& particles,
                                     const NeighListType& neigh_list,
                                     const ParallelType neigh_op_tag )
    {
        _timer.start();

        auto n_local = particles.n_local;
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
            double theta_i =
                model.influence_function( xi ) * s * xi * xi * vol( j );
            theta( i ) += 3.0 * theta_i / m( i );
        };

        Kokkos::RangePolicy<exec_space> policy( 0, n_local );
        Cabana::neighbor_parallel_for(
            policy, dilatation, neigh_list, Cabana::FirstNeighborsTag(),
            neigh_op_tag, "CabanaPD::ForceCorrespondence::computeDilatation" );

        _timer.stop();
    }

    template <class ForceType, class PosType, class ParticleType,
              class NeighListType, class ParallelType>
    void computeForceFull( ForceType& f, const PosType&, const PosType& u,
                           const ParticleType& particles,
                           const NeighListType& neigh_list, const int n_local,
                           ParallelType& neigh_op_tag )
    {
        _timer.start();

        auto theta_coeff = _model.theta_coeff;
        auto s_coeff = _model.s_coeff;
        auto model = _model;

        const auto vol = particles.sliceVolume();
        const auto Kinv = particles.sliceShapeTensor();
        const auto y = particles.sliceCurrentPosition();
        auto force_full = KOKKOS_LAMBDA( const int i, const int j )
        {
            double fx_i = 0.0;
            double fy_i = 0.0;
            double fz_i = 0.0;

            double xi, r, s;
            double rx, ry, rz;
            getDistanceComponents( y, u, i, j, xi, r, s, rx, ry, rz );
            const double coeff = model.influence_function( xi ) *
                                 model.getStress( F, i ) * Kinv * vol( j );
            fx_i = coeff * rx / r;
            fy_i = coeff * ry / r;
            fz_i = coeff * rz / r;

            f( i, 0 ) += fx_i;
            f( i, 1 ) += fy_i;
            f( i, 2 ) += fz_i;
        };

        Kokkos::RangePolicy<exec_space> policy( 0, n_local );
        Cabana::neighbor_parallel_for(
            policy, force_full, neigh_list, Cabana::FirstNeighborsTag(),
            neigh_op_tag, "CabanaPD::ForceCorrespondence::computeFull" );

        _timer.stop();
    }

    template <class PosType, class WType, class ParticleType,
              class NeighListType, class ParallelType>
    double computeEnergyFull( WType& W, const PosType& x, const PosType& u,
                              const ParticleType& particles,
                              const NeighListType& neigh_list,
                              const int n_local, ParallelType& neigh_op_tag )
    {
        _energy_timer.start();

        auto theta_coeff = _model.theta_coeff;
        auto s_coeff = _model.s_coeff;
        auto model = _model;

        const auto vol = particles.sliceVolume();
        const auto theta = particles.sliceDilatation();
        auto m = particles.sliceWeightedVolume();

        auto energy_full =
            KOKKOS_LAMBDA( const int i, const int j, double& Phi )
        {
            double xi, r, s;
            getDistance( x, u, i, j, xi, r, s );

            double num_neighbors = static_cast<double>(
                Cabana::NeighborList<NeighListType>::numNeighbor( neigh_list,
                                                                  i ) );

            double w = ( 1.0 / num_neighbors ) * 0.5 * theta_coeff / 3.0 *
                           ( theta( i ) * theta( i ) ) +
                       0.5 * ( s_coeff / m( i ) ) *
                           model.influence_function( xi ) * s * s * xi * xi *
                           vol( j );
            W( i ) += w;
            Phi += w * vol( i );
        };

        double strain_energy = 0.0;

        Kokkos::RangePolicy<exec_space> policy( 0, n_local );
        Cabana::neighbor_parallel_reduce(
            policy, energy_full, neigh_list, Cabana::FirstNeighborsTag(),
            neigh_op_tag, strain_energy,
            "CabanaPD::ForceCorrespondence::computeEnergyFull" );

        _energy_timer.stop();
        return strain_energy;
    }
};

} // namespace CabanaPD

#endif
