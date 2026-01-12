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

#include <CabanaPD_Constants.hpp>
#include <CabanaPD_Types.hpp>

namespace CabanaPD
{
struct ForceCoeffTag
{
};
struct EnergyTag
{
};
struct CriticalStretchTag
{
};
struct ThermalStretchTag
{
};
struct WeightedVolumeTag
{
};
struct DilatationTag
{
};
struct InfluenceFunctionTag
{
};

struct BaseForceModel
{
    // Is there a field that needs to be updated later?
    using needs_update = std::false_type;
    using material_type = SingleMaterial;
    double force_horizon;
    double K;

    BaseForceModel( const double _force_horizon, const double _K )
        : force_horizon( _force_horizon )
        , K( _K )
    {
    }

    // FIXME: use the first model cutoff for now.
    template <typename ModelType1, typename ModelType2>
    BaseForceModel( const ModelType1& model1, const ModelType2& model2 )
    {
        force_horizon = model1.force_horizon;
        K = ( model1.K + model2.K ) / 2.0;
    }

    auto cutoff() const { return force_horizon; }
    auto extend() const { return 0.0; }

    // Only needed for models which store bond properties.
    void updateBonds( const int, const int ) {}
};

struct BaseNoFractureModel
{
    using fracture_type = NoFracture;

    // This should only be used in multi-material models in which the current
    // model does not support failure.
    KOKKOS_INLINE_FUNCTION
    bool operator()( CriticalStretchTag, const int, const int, const double,
                     const double ) const
    {
        return false;
    }
};

struct BaseFractureModel
{
    using fracture_type = Fracture;

    double G0;
    double s0;
    double bond_break_coeff;

    BaseFractureModel( const double _force_horizon, const double _K,
                       const double _G0, const int influence_type = 1 )
        : G0( _G0 )
    {
        s0 = Kokkos::sqrt( 5.0 * G0 / 9.0 / _K / _force_horizon ); // 1/xi
        if ( influence_type == 0 )
            s0 = Kokkos::sqrt( 8.0 * G0 / 15.0 / _K / _force_horizon ); // 1

        bond_break_coeff = ( 1.0 + s0 ) * ( 1.0 + s0 );
    };

    // Constructor to work with plasticity.
    BaseFractureModel( const double _G0, const double _s0 )
        : G0( _G0 )
        , s0( _s0 )
    {
        bond_break_coeff = ( 1.0 + s0 ) * ( 1.0 + s0 );
    }

    // Average from existing models.
    template <typename ModelType1, typename ModelType2>
    BaseFractureModel( const ModelType1& model1, const ModelType2& model2 )
    {
        G0 = ( model1.G0 + model2.G0 ) / 2.0;
        s0 = Kokkos::sqrt( ( model1.s0 * model1.s0 * model1.K +
                             model2.s0 * model2.s0 * model2.K ) /
                           ( model1.K + model2.K ) );
        bond_break_coeff = ( 1.0 + s0 ) * ( 1.0 + s0 );
    }

    KOKKOS_INLINE_FUNCTION
    bool operator()( CriticalStretchTag, const int, const int, const double r,
                     const double xi ) const
    {
        return r * r >= bond_break_coeff * xi * xi;
    }
};

template <class MemorySpace>
class BasePlasticity
{
  protected:
    using memory_space = MemorySpace;
    using neighbor_view = typename Kokkos::View<double**, memory_space>;
    neighbor_view _s_p;

  public:
    // Must update later because number of neighbors not known at construction.
    void updateBonds( const int num_local, const int max_neighbors )
    {
        Kokkos::realloc( _s_p, num_local, max_neighbors );
        Kokkos::deep_copy( _s_p, 0.0 );
    }
};

struct ConstantProperty
{
    ConstantProperty( const double v )
        : value( v )
    {
    }

    ConstantProperty( const double v1, const double v2 )
        : value( ( v1 + v2 ) / 2.0 )
    {
    }

    auto operator()( const double ) const { return value; }

    double value;
};

template <typename ThermalType, typename... TemperatureType>
struct ThermalModel;

template <typename TemperatureType, typename FunctorType>
struct ThermalModel<TemperatureDependent, FunctorType, TemperatureType>
{
    using thermal_type = TemperatureDependent;
    using needs_update = std::true_type;

    FunctorType alpha;
    double temp0;

    // Temperature field
    TemperatureType temperature;

    explicit ThermalModel( const TemperatureType _temp, const double _alpha,
                           const double _temp0 )
        : ThermalModel( _temp, ConstantProperty( _alpha ), _temp0 )
    {
    }

    ThermalModel( const TemperatureType _temp, const FunctorType _alpha,
                  const double _temp0 )
        : alpha( _alpha )
        , temp0( _temp0 )
        , temperature( _temp )
    {
    }

    // FIXME: use the first model temperature for now.
    template <typename ModelType1, typename ModelType2>
    ThermalModel( const ModelType1& model1, const ModelType2& model2 )
        : alpha( model1.alpha( 0 ), model2.alpha( 0 ) )
    {
        static_assert( std::is_same_v<decltype( model1.temperature ),
                                      decltype( model2.temperature )>,
                       "ThermalModel: Both models must have same "
                       "TemperatureType" );
        temperature = model1.temperature;
        temp0 = ( model1.temp0 + model2.temp0 ) / 2.0;
    }

    template <typename ParticleType>
    void update( const ParticleType& particles )
    {
        temperature = particles.sliceTemperature();
    }

    // Update stretch with temperature effects.
    KOKKOS_INLINE_FUNCTION
    double operator()( ThermalStretchTag, const int i, const int j,
                       const double s ) const
    {
        double temp_avg = 0.5 * ( temperature( i ) + temperature( j ) ) - temp0;
        return s - ( alpha( temp_avg ) * temp_avg );
    }
};

template <typename TemperatureType>
ThermalModel( const TemperatureType, const double, const double )
    -> ThermalModel<TemperatureDependent, ConstantProperty, TemperatureType>;

template <typename TemperatureType, typename FunctorType>
ThermalModel( const TemperatureType, const FunctorType, const double )
    -> ThermalModel<TemperatureDependent, FunctorType, TemperatureType>;

template <typename TemperatureType>
struct ThermalFractureModel
    : public BaseFractureModel,
      ThermalModel<TemperatureDependent, ConstantProperty, TemperatureType>
{
    using base_fracture_type = BaseFractureModel;
    using base_temperature_type =
        ThermalModel<TemperatureDependent, ConstantProperty, TemperatureType>;
    using typename base_fracture_type::fracture_type;
    using typename base_temperature_type::thermal_type;

    // Does not use the base bond_break_coeff.
    using base_fracture_type::s0;
    using base_temperature_type::alpha;
    using base_temperature_type::temp0;
    using base_temperature_type::temperature;

    // Does not use the base critical stretch.
    using base_temperature_type::operator();

    ThermalFractureModel( const double _force_horizon, const double _K,
                          const double _G0, const TemperatureType _temp,
                          const double _alpha, const double _temp0,
                          const int influence_type = 1 )
        : base_fracture_type( _force_horizon, _K, _G0, influence_type )
        , base_temperature_type( _temp, _alpha, _temp0 )
    {
    }

    // FIXME: use the first model horizon and microconductivity for now.
    template <typename ModelType1, typename ModelType2>
    ThermalFractureModel( const ModelType1& model1, const ModelType2& model2 )
        : base_fracture_type( model1, model2 )
        , base_temperature_type( model1, model2 )
    {
    }

    KOKKOS_INLINE_FUNCTION
    bool operator()( CriticalStretchTag, const int i, const int j,
                     const double r, const double xi ) const
    {
        double temp_avg = 0.5 * ( temperature( i ) + temperature( j ) ) - temp0;
        double bond_break_coeff = ( 1.0 + s0 + alpha( temp_avg ) * temp_avg ) *
                                  ( 1.0 + s0 + alpha( temp_avg ) * temp_avg );
        return r * r >= bond_break_coeff * xi * xi;
    }
};

// This class stores temperature parameters needed for heat transfer, but not
// the temperature itself (stored instead in the static temperature class
// above).
struct BaseDynamicTemperatureModel
{
    using thermal_type = DynamicTemperature;
    using needs_update = std::true_type;

    double thermal_horizon;

    double thermal_coeff;
    double kappa;
    double cp;
    bool constant_microconductivity;

    BaseDynamicTemperatureModel( const double _thermal_horizon,
                                 const double _kappa, const double _cp,
                                 const bool _constant_microconductivity = true )
    {
        thermal_horizon = _thermal_horizon;
        kappa = _kappa;
        cp = _cp;
        const double d3 =
            _thermal_horizon * _thermal_horizon * _thermal_horizon;
        thermal_coeff = 9.0 / 2.0 * _kappa / pi / d3;
        constant_microconductivity = _constant_microconductivity;
    }

    // FIXME: use the first model horizon and microconductivity for now.
    template <typename ModelType1, typename ModelType2>
    BaseDynamicTemperatureModel( const ModelType1& model1,
                                 const ModelType2& model2 )
    {
        constant_microconductivity = model1.constant_microconductivity;
        thermal_horizon = model1.thermal_horizon;
        thermal_coeff = ( model1.thermal_coeff + model2.thermal_coeff ) / 2.0;
        kappa = ( model1.kappa + model2.kappa ) / 2.0;
        cp = ( model1.cp + model2.cp ) / 2.0;
    }

    KOKKOS_INLINE_FUNCTION double microconductivity_function( double r ) const
    {
        if ( constant_microconductivity )
            return thermal_coeff;
        else
            return 4.0 * thermal_coeff * ( 1.0 - r / thermal_horizon );
    }
};

template <typename PeridynamicsModelType, typename MechanicsModelType = Elastic,
          typename DamageType = Fracture, typename... DataTypes>
struct ForceModel;

template <typename ForceType, typename ThermalType>
struct ThermalForceModel : public ForceType, ThermalType
{
    using base_type = ForceType;
    using base_temperature_type = ThermalType;

    using base_type::operator();
    using base_temperature_type::operator();
    using typename base_temperature_type::needs_update;
    using typename base_temperature_type::thermal_type;
    using typename base_type::fracture_type;

    ThermalForceModel( ForceType force, ThermalType thermal )
        : base_type( force )
        , base_temperature_type( thermal )
    {
    }

    // Constructor to average from existing models.
    template <typename ModelType1, typename ModelType2>
    ThermalForceModel( const ModelType1& model1, const ModelType2& model2 )
        : base_type( model1, model2 )
        , base_temperature_type( model1, model2 )
    {
    }
};

} // namespace CabanaPD

#endif
