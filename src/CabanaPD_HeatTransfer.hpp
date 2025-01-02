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

#include <CabanaPD_Force.hpp>
#include <CabanaPD_Timer.hpp>
#include <CabanaPD_Types.hpp>

namespace CabanaPD
{

template <class MemorySpace, class ModelType>
class HeatTransfer;

// Peridynamic heat transfer with forward-Euler time integration.
// Inherits only because this is a similar neighbor-based kernel.
template <class MemorySpace, class... ModelParams>
class HeatTransfer<MemorySpace,
                   ForceModel<PMB, Elastic, DynamicTemperature, ModelParams...>>
    : public Force<MemorySpace, BaseForceModel>
{
  protected:
    using base_type = Force<MemorySpace, BaseForceModel>;
    using base_type::_half_neigh;
    using base_type::_timer;
    Timer _euler_timer = base_type::_energy_timer;
    using model_type =
        ForceModel<PMB, Elastic, DynamicTemperature, ModelParams...>;
    model_type _model;

    // Using the default exec_space.
    using exec_space = typename MemorySpace::execution_space;

  public:
    using base_type::_neigh_list;
    static_assert(
        std::is_same_v<typename model_type::thermal_type, DynamicTemperature> );

    // Running with mechanics as well; no reason to rebuild neighbors.
    template <class NeighborType>
    HeatTransfer( const bool half_neigh, const NeighborType& neighbors,
                  const model_type model )
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

template <class MemorySpace, class... ModelParams>
class HeatTransfer<
    MemorySpace, ForceModel<PMB, Fracture, DynamicTemperature, ModelParams...>>
    : public HeatTransfer<
          MemorySpace,
          ForceModel<PMB, Elastic, DynamicTemperature, ModelParams...>>
{
  protected:
    using base_type =
        HeatTransfer<MemorySpace, ForceModel<PMB, Elastic, DynamicTemperature,
                                             ModelParams...>>;
    using exec_space = typename base_type::exec_space;
    using base_type::_euler_timer;
    using base_type::_half_neigh;
    using base_type::_timer;
    using model_type =
        ForceModel<PMB, Fracture, DynamicTemperature, ModelParams...>;
    model_type _model;

  public:
    using neighbor_list_type = typename base_type::neighbor_list_type;
    using base_type::_neigh_list;

    // This is necessary because of the indirect model inheritance.
    // The intent is that these HeatTransfer classes merge as more details are
    // merged into the respective models.
    template <class NeighborType>
    HeatTransfer( const bool half_neigh, const NeighborType& neighbors,
                  const model_type model )
        : base_type( half_neigh, neighbors,
                     typename base_type::model_type(
                         model.delta, model.K, model.temperature, model.kappa,
                         model.cp, model.alpha, model.temp0,
                         model.constant_microconductivity ) )
        , _model( model )
    {
    }

    template <class TemperatureType, class PosType, class ParticleType,
              class MuView, class ParallelType>
    void computeHeatTransferFull( TemperatureType& conduction, const PosType& x,
                                  const PosType& u,
                                  const ParticleType& particles,
                                  const MuView& mu, const int n_local,
                                  ParallelType& )
    {
        _timer.start();

        auto model = _model;
        const auto neigh_list = _neigh_list;
        const auto vol = particles.sliceVolume();
        const auto temp = particles.sliceTemperature();

        auto temp_func = KOKKOS_LAMBDA( const int i )
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

                model.thermalStretch( s, i, j );

                // Only include unbroken bonds.
                if ( mu( i, n ) > 0 )
                {
                    const double coeff = model.microconductivity_function( xi );
                    conduction( i ) +=
                        coeff * ( temp( j ) - temp( i ) ) / xi / xi * vol( j );
                }
            }
        };

        Kokkos::RangePolicy<exec_space> policy( 0, n_local );
        Kokkos::parallel_for( "CabanaPD::HeatTransfer::computeFull", policy,
                              temp_func );
        _timer.stop();
    }
};

// Heat transfer free functions.
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

template <class HeatTransferType, class ParticleType, class MuView,
          class ParallelType>
void computeHeatTransfer( HeatTransferType& heat_transfer,
                          ParticleType& particles, const MuView& mu,
                          const ParallelType& neigh_op_tag, const double dt )
{
    auto n_local = particles.n_local;
    auto x = particles.sliceReferencePosition();
    auto u = particles.sliceDisplacement();
    auto conduction = particles.sliceTemperatureConduction();
    auto conduction_a = particles.sliceTemperatureConductionAtomic();

    // Reset temperature conduction.
    Cabana::deep_copy( conduction, 0.0 );

    // Temperature only needs to be atomic if using team threading.
    if ( std::is_same<decltype( neigh_op_tag ), Cabana::TeamOpTag>::value )
        heat_transfer.computeHeatTransferFull( conduction_a, x, u, particles,
                                               mu, n_local, neigh_op_tag );
    else
        heat_transfer.computeHeatTransferFull( conduction, x, u, particles, mu,
                                               n_local, neigh_op_tag );
    Kokkos::fence();

    heat_transfer.forwardEuler( particles, dt );
    Kokkos::fence();
}

} // namespace CabanaPD

#endif
