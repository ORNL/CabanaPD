/****************************************************************************
 * Copyright (c) 2022-2023 by Oak Ridge National Laboratory                 *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of CabanaPD. CabanaPD is distributed under a           *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef HEATTRANSFER_H
#define HEATTRANSFER_H

#include <CabanaPD_Timer.hpp>

namespace CabanaPD
{

// Peridynamic heat transfer with forward-Euler time integration.
// Inherits only because this is a similar neighbor-based kernel.
template <class ExecutionSpace, class ModelType>
class HeatTransfer : public Force<ExecutionSpace, BaseForceModel>
{
  protected:
    using base_type = Force<ExecutionSpace, BaseForceModel>;
    using base_type::_half_neigh;
    using base_type::_timer;
    Timer _euler_timer = base_type::_energy_timer;
    ModelType _model;

  public:
    HeatTransfer( const bool half_neigh, const ModelType model )
        : base_type( half_neigh )
        , _model( model )
    {
    }

    template <class TemperatureType, class PosType, class ParticleType,
              class NeighListType, class ParallelType>
    void
    computeHeatTransferFull( TemperatureType& prev_temp, const PosType& x,
                             const PosType& u, const ParticleType& particles,
                             const NeighListType& neigh_list, const int n_local,
                             ParallelType& neigh_op_tag )
    {
        _timer.start();

        auto model = _model;
        const auto vol = particles.sliceVolume();
        const auto temp = particles.sliceTemperature();

        auto temp_func = KOKKOS_LAMBDA( const int i, const int j )
        {
            double xi, r, s;
            getDistance( x, u, i, j, xi, r, s );

            const double coeff = model.thermal_coeff *
                                 model.conductivity_function( xi ) / model.cp;
            prev_temp( i ) +=
                coeff * ( temp( j ) - temp( i ) ) / xi / xi * vol( j );
        };

        Kokkos::RangePolicy<ExecutionSpace> policy( 0, n_local );
        Cabana::neighbor_parallel_for(
            policy, temp_func, neigh_list, Cabana::FirstNeighborsTag(),
            neigh_op_tag, "CabanaPD::HeatTransfer::computeFull" );

        _timer.stop();
    }

    template <class ParticleType>
    void stepEuler( const ParticleType& particles, const double dt )
    {
        _euler_timer.start();
        const auto rho = particles.sliceDensity();
        const auto prev_temp = particles.slicePreviousTemperature();
        auto temp = particles.sliceTemperature();
        auto n_local = particles.n_local;
        auto euler_func = KOKKOS_LAMBDA( const int i )
        {
            temp( i ) += dt / rho( i ) * prev_temp( i );
        };
        Kokkos::RangePolicy<ExecutionSpace> policy( 0, n_local );
        Kokkos::parallel_for( "CabanaPD::HeatTransfer::Euler", policy,
                              euler_func );
        _euler_timer.stop();
    }
};

// Heat transfer free function.
template <class HeatTransferType, class ParticleType, class NeighListType,
          class ParallelType>
void computeHeatTransfer( HeatTransferType& heat_transfer,
                          ParticleType& particles,
                          const NeighListType& neigh_list,
                          const ParallelType& neigh_op_tag, const double dt )
{
    auto n_local = particles.n_local;
    auto x = particles.sliceReferencePosition();
    auto u = particles.sliceDisplacement();
    auto temp = particles.slicePreviousTemperature();
    auto temp_a = particles.slicePreviousTemperatureAtomic();

    // Reset previous temperature.
    Cabana::deep_copy( temp, 0.0 );

    // Temperature only needs to be atomic if using team threading.
    if ( std::is_same<decltype( neigh_op_tag ), Cabana::TeamOpTag>::value )
        heat_transfer.computeHeatTransferFull(
            temp_a, x, u, particles, neigh_list, n_local, neigh_op_tag );
    else
        heat_transfer.computeHeatTransferFull(
            temp, x, u, particles, neigh_list, n_local, neigh_op_tag );
    Kokkos::fence();

    heat_transfer.stepEuler( particles, dt );
    Kokkos::fence();
}

} // namespace CabanaPD

#endif
