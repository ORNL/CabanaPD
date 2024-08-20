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

#ifndef FORCE_MODELS_H
#define FORCE_MODELS_H

#include <CabanaPD_Types.hpp>

namespace CabanaPD
{

template <typename MemorySpace>
struct BaseForceModel;

template <>
struct BaseForceModel<SingleSpecies>
{
    using species_type = SingleSpecies;
    double delta;

    BaseForceModel(){};
    BaseForceModel( const double _delta )
        : delta( _delta ){};

    auto horizon( const int ) { return delta; }
    auto maxHorizon() { return delta; }

    // No-op for temperature.
    KOKKOS_INLINE_FUNCTION
    void thermalStretch( double&, const int, const int ) const {}
};

template <typename MemorySpace>
struct BaseForceModel
{
    using species_type = MultiSpecies;

    // Only allow one memory space.
    using memory_space = MemorySpace;
    using view_type_1d = Kokkos::View<double*, memory_space>;
    view_type_1d delta;
    double max_delta;
    std::size_t num_types;

    using view_type_2d = Kokkos::View<double**, memory_space>;

    BaseForceModel( const double _delta )
        : delta( view_type_1d( "delta", 1 ) )
        , num_types( 1 )
    {
        Kokkos::deep_copy( delta, _delta );
    }

    template <typename ArrayType>
    BaseForceModel( const ArrayType& _delta )
        : delta( view_type_1d( "delta", _delta.size() ) )
        , num_types( _delta.size() )
    {
        setParameters( _delta );
    }

    template <typename ArrayType>
    void setParameters( const ArrayType& _delta )
    {
        max_delta = 0;
        auto delta_copy = delta;
        auto init_func = KOKKOS_LAMBDA( const int i, double& max )
        {
            delta_copy( i ) = _delta[i];
            if ( delta_copy( i ) > max )
                max = delta_copy( i );
        };
        using exec_space = typename memory_space::execution_space;
        Kokkos::RangePolicy<exec_space> policy( 0, num_types );
        Kokkos::parallel_reduce( "CabanaPD::Model::Init", policy, init_func,
                                 max_delta );
    }

    auto horizon( const int i ) { return delta( i ); }
    auto maxHorizon() { return max_delta; }

    // No-op for temperature.
    KOKKOS_INLINE_FUNCTION
    void thermalStretch( double&, const int, const int ) const {}
};

template <typename TemperatureType, typename SpeciesType>
struct BaseTemperatureModel;

template <typename TemperatureType>
struct BaseTemperatureModel<TemperatureType, SingleSpecies>
{
    using species_type = SingleSpecies;
    using memory_space = typename TemperatureType::memory_space;
    double alpha;
    double temp0;

    // Temperature field
    TemperatureType temperature;

    BaseTemperatureModel( const TemperatureType _temp, const double _alpha,
                          const double _temp0 )
        : alpha( _alpha )
        , temp0( _temp0 )
        , temperature( _temp ){};

    void update( const TemperatureType _temp ) { temperature = _temp; }

    // Update stretch with temperature effects.
    KOKKOS_INLINE_FUNCTION
    void thermalStretch( double& s, const int i, const int j ) const
    {
        double temp_avg = 0.5 * ( temperature( i ) + temperature( j ) ) - temp0;
        s -= alpha * temp_avg;
    }
};

template <typename TemperatureType, typename ParticleType>
struct BaseTemperatureModel
{
    using species_type = MultiSpecies;
    using memory_space = typename TemperatureType::memory_space;
    using view_type_1d = Kokkos::View<double*, memory_space>;
    view_type_1d alpha;
    view_type_1d temp0;

    // Particle fields
    TemperatureType temperature;
    ParticleType type;

    template <typename ArrayType>
    BaseTemperatureModel( const TemperatureType& _temp, const ArrayType& _alpha,
                          const ArrayType& _temp0, const ParticleType& _type )
        : alpha( view_type_1d( "delta", _alpha.size() ) )
        , temp0( view_type_1d( "delta", _temp0.size() ) )
        , temperature( _temp )
        , type( _type )
    {
        setParameters( _alpha, _temp0 );
    }

    template <typename ArrayType>
    void setParameters( const ArrayType& _alpha, const ArrayType& _temp0 )
    {
        auto alpha_copy = alpha;
        auto temp0_copy = temp0;
        auto init_func = KOKKOS_LAMBDA( const int i )
        {
            alpha_copy( i ) = _alpha[i];
            temp0_copy( i ) = _temp0[i];
        };
        using exec_space = typename memory_space::execution_space;
        Kokkos::RangePolicy<exec_space> policy( 0, alpha.size() );
        Kokkos::parallel_for( "CabanaPD::Model::Init", policy, init_func );
    }

    void update( const TemperatureType& _temp, const ParticleType& _type )
    {
        temperature = _temp;
        type = _type;
    }

    // Update stretch with temperature effects.
    KOKKOS_INLINE_FUNCTION
    void thermalStretch( double& s, const int i, const int j ) const
    {
        double temp_avg =
            0.5 * ( temperature( i ) + temperature( j ) ) - temp0( type( i ) );
        s -= alpha( type( i ) ) * temp_avg;
    }
};

template <typename ModelType, typename DamageType,
          typename ThermalType = TemperatureIndependent, typename... DataTypes>
struct ForceModel;

} // namespace CabanaPD

#endif
