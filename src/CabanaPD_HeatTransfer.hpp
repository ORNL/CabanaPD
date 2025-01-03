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
template <class MemorySpace, class ModelType>
class HeatTransfer : public Force<MemorySpace, BaseForceModel>
{
  protected:
    using base_type = Force<MemorySpace, BaseForceModel>;
    using base_type::_half_neigh;
    using base_type::_timer;

    Timer _euler_timer = base_type::_energy_timer;
    ModelType _model;
    // Using the default exec_space.
    using exec_space = typename MemorySpace::execution_space;

  public:
    using base_type::_neigh_list;
    using model_type = ModelType;
    static_assert(
        std::is_same_v<typename model_type::fracture_type, NoFracture> );

    // Running with mechanics as well; no reason to rebuild neighbors.
    template <class NeighborType>
    HeatTransfer( const bool half_neigh, const NeighborType& neighbors,
                  const ModelType model )
        : base_type( half_neigh, neighbors )
        , _model( model )
    {
    }

    template <class TemperatureType, class PosType, class ParticleType,
              class ParallelType>
    void computeHeatTransferFull( TemperatureType& conduction, const PosType& x,
                                  const PosType& u,
                                  const ParticleType& particles,
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

            const double coeff = model.microconductivity_function( xi );
            conduction( i ) +=
                coeff * ( temp( j ) - temp( i ) ) / xi / xi * vol( j );
        };

        Kokkos::RangePolicy<exec_space> policy( particles.frozenOffset(),
                                                particles.localOffset() );
        Cabana::neighbor_parallel_for(
            policy, temp_func, _neigh_list, Cabana::FirstNeighborsTag(),
            neigh_op_tag, "CabanaPD::HeatTransfer::computeFull" );

        _timer.stop();
    }

    template <class ParticleType>
    void forwardEuler( const ParticleType& particles, const double dt )
    {
        _euler_timer.start();
        auto model = _model;
        const auto rho = particles.sliceDensity();
        const auto conduction = particles.sliceTemperatureConduction();
        auto temp = particles.sliceTemperature();
        auto euler_func = KOKKOS_LAMBDA( const int i )
        {
            temp( i ) += dt / rho( i ) / model.cp * conduction( i );
        };
        Kokkos::RangePolicy<exec_space> policy( particles.frozenOffset(),
                                                particles.localOffset() );
        Kokkos::parallel_for( "CabanaPD::HeatTransfer::forwardEuler", policy,
                              euler_func );
        _euler_timer.stop();
    }
};

// Heat transfer free function.
template <class HeatTransferType, class ParticleType, class ParallelType>
void computeHeatTransfer( HeatTransferType& heat_transfer,
                          ParticleType& particles,
                          const ParallelType& neigh_op_tag, const double dt )
{
    auto x = particles.sliceReferencePosition();
    auto u = particles.sliceDisplacement();
    auto conduction = particles.sliceTemperatureConduction();
    auto conduction_a = particles.sliceTemperatureConductionAtomic();

    // Reset temperature conduction.
    Cabana::deep_copy( conduction, 0.0 );

    // Temperature only needs to be atomic if using team threading.
    if ( std::is_same<decltype( neigh_op_tag ), Cabana::TeamOpTag>::value )
        heat_transfer.computeHeatTransferFull( conduction_a, x, u, particles,
                                               neigh_op_tag );
    else
        heat_transfer.computeHeatTransferFull( conduction, x, u, particles,
                                               neigh_op_tag );
    Kokkos::fence();

    heat_transfer.forwardEuler( particles, dt );
    Kokkos::fence();
}

} // namespace CabanaPD

#endif
