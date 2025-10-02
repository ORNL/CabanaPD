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
#include <CabanaPD_Neighbor.hpp>
#include <CabanaPD_Timer.hpp>
#include <CabanaPD_Types.hpp>

namespace CabanaPD
{

template <class MemorySpace, class ModelType, class FractureType>
class HeatTransfer;

// Peridynamic heat transfer with forward-Euler time integration.
// Inherits only because this is a similar neighbor-based kernel.
template <class MemorySpace, class ModelType>
class HeatTransfer<MemorySpace, ModelType, NoFracture>
{
  public:
    // Using the default exec_space.
    using exec_space = typename MemorySpace::execution_space;
    using model_type = ModelType;

  protected:
    Timer _timer;
    Timer _euler_timer;
    model_type _model;

  public:
    // Running with mechanics as well; no reason to rebuild neighbors.
    HeatTransfer( const model_type model )
        : _model( model )
    {
    }

    template <class TemperatureType, class PosType, class ParticleType,
              class NeighborType>
    void computeHeatTransferFull( TemperatureType& conduction, const PosType& x,
                                  const PosType& u,
                                  const ParticleType& particles,
                                  NeighborType& neighbor )
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

        neighbor.iterate( exec_space{}, temp_func, particles,
                          "CabanaPD::HeatTransfer::computeFull" );
        Kokkos::fence();
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
        Kokkos::fence();
        _euler_timer.stop();
    }
};

template <class MemorySpace, class ModelType>
class HeatTransfer<MemorySpace, ModelType, Fracture>
    : public HeatTransfer<MemorySpace, ModelType, NoFracture>
{
  public:
    // Using the default exec_space.
    using exec_space = typename MemorySpace::execution_space;
    using model_type = ModelType;
    using base_type = HeatTransfer<MemorySpace, ModelType, NoFracture>;

  protected:
    using base_type::_model;

    using base_type::_euler_timer;
    using base_type::_timer;

  public:
    // Explicit base model construction is necessary because of the indirect
    // model inheritance. This could be avoided with a BaseHeatTransfer object.
    HeatTransfer( const model_type model )
        : base_type( model )
    {
    }

    template <class TemperatureType, class PosType, class ParticleType,
              class NeighborType>
    void computeHeatTransferFull( TemperatureType& conduction, const PosType& x,
                                  const PosType& u,
                                  const ParticleType& particles,
                                  NeighborType& neighbor )
    {
        _timer.start();
        using neighbor_list_type = typename NeighborType::list_type;
        auto neigh_list = neighbor.list();
        const auto mu = neighbor.brokenBonds();

        auto model = _model;
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

                // Only include unbroken bonds.
                if ( mu( i, n ) > 0 )
                {
                    const double coeff = model.microconductivity_function( xi );
                    conduction( i ) +=
                        coeff * ( temp( j ) - temp( i ) ) / xi / xi * vol( j );
                }
            }
        };

        neighbor.iterateLinear( exec_space{}, temp_func, particles,
                                "CabanaPD::HeatTransfer::computeFull" );
        Kokkos::fence();
        _timer.stop();
    }
};

// Heat transfer free functions.
template <class HeatTransferType, class ParticleType, class NeighborType>
void computeHeatTransfer( HeatTransferType& heat_transfer,
                          ParticleType& particles, const NeighborType& neighbor,
                          const double dt )
{
    auto x = particles.sliceReferencePosition();
    auto u = particles.sliceDisplacement();
    auto conduction = particles.sliceTemperatureConduction();
    auto conduction_a = particles.sliceTemperatureConductionAtomic();

    // Reset temperature conduction.
    Cabana::deep_copy( conduction, 0.0 );

    // Temperature only needs to be atomic if using team threading.
    if ( std::is_same<typename NeighborType::Tag, Cabana::TeamOpTag>::value )
        heat_transfer.computeHeatTransferFull( conduction_a, x, u, particles,
                                               neighbor );
    else
        heat_transfer.computeHeatTransferFull( conduction, x, u, particles,
                                               neighbor );
    Kokkos::fence();

    heat_transfer.forwardEuler( particles, dt );
}

} // namespace CabanaPD

#endif
