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

template <typename PDModelType, typename MechanicsType, typename... DataTypes>
struct MechanicsModel;

template <typename T>
struct is_mechanics_model : public std::false_type
{
};
template <typename PDModelType, typename MechanicsType, typename... DataTypes>
struct is_mechanics_model<
    MechanicsModel<PDModelType, MechanicsType, DataTypes...>>
    : public std::true_type
{
};

struct BaseForceModel
{
    // Is there a field that needs to be updated later?
    using needs_update = std::false_type;
    using material_type = SingleMaterial;
    using thermal_tag = TemperatureIndependent;
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

/******************************************************************************
  Fracture models.
******************************************************************************/
template <typename FractureType>
struct FractureModel;

template <>
struct FractureModel<NoFracture>
{
    using fracture_tag = NoFracture;

    // This should only be used in multi-material models in which the current
    // model does not support failure.
    KOKKOS_INLINE_FUNCTION
    bool operator()( CriticalStretchTag, const int, const int, const double,
                     const double ) const
    {
        return false;
    }
};

template <>
struct FractureModel<CriticalStretch>
{
    using fracture_tag = Fracture;

    double G0;
    double s0;
    double bond_break_coeff;
    int influence_type;

    FractureModel( const double _force_horizon, const double _K,
                   const double _G0, const int influence = 1 )
        : G0( _G0 )
        , influence_type( influence )
    {
        s0 = Kokkos::sqrt( 5.0 * G0 / 9.0 / _K / _force_horizon ); // 1/xi
        if ( influence_type == 0 )
            s0 = Kokkos::sqrt( 8.0 * G0 / 15.0 / _K / _force_horizon ); // 1

        bond_break_coeff = ( 1.0 + s0 ) * ( 1.0 + s0 );
    };

    // Constructor to work with plasticity.
    FractureModel( const double _G0, const double _s0 )
        : G0( _G0 )
        , s0( _s0 )
    {
        bond_break_coeff = ( 1.0 + s0 ) * ( 1.0 + s0 );
    }

    // Average from existing models.
    template <typename ModelType1, typename ModelType2>
    FractureModel( const ModelType1& model1, const ModelType2& model2 )
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

FractureModel( const double _force_horizon, const double _K, const double _G0,
               const int influence = 1 )
    ->FractureModel<CriticalStretch>;

/******************************************************************************
Force models forward declaration.
******************************************************************************/
template <typename PeridynamicsModelType, typename MechanicsModelType = Elastic,
          typename DamageType = Fracture,
          typename ThermalType = TemperatureIndependent, typename... DataTypes>
struct ForceModel;

/******************************************************************************
  Temperature-dependent models.
******************************************************************************/
template <typename ThermalType, typename TemperatureType,
          typename FractureType = NoFracture>
struct ThermalModel;

template <>
struct ThermalModel<TemperatureIndependent, TemperatureIndependent, NoFracture>
{
    KOKKOS_FUNCTION
    double operator()( ThermalStretchTag, const int, const int,
                       const double s ) const
    {
        return s;
    }
};

template <typename TemperatureType>
struct ThermalModel<TemperatureDependent, TemperatureType, NoFracture>
{
    using thermal_tag = TemperatureDependent;
    using needs_update = std::true_type;
    using fracture_tag = NoFracture;

    double alpha;
    double temp0;

    // Temperature field
    TemperatureType temperature;

    ThermalModel( const TemperatureType _temp, const double _alpha,
                  const double _temp0 )
        : alpha( _alpha )
        , temp0( _temp0 )
        , temperature( _temp )
    {
    }

    // FIXME: use the first model temperature for now.
    template <typename ModelType1, typename ModelType2>
    ThermalModel( const ModelType1& model1, const ModelType2& model2 )
    {
        static_assert( std::is_same_v<decltype( model1.temperature ),
                                      decltype( model2.temperature )>,
                       "ThermalModel: Both models must have same "
                       "TemperatureType" );
        temperature = model1.temperature;
        alpha = ( model1.alpha + model2.alpha ) / 2.0;
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
        return s - ( alpha * temp_avg );
    }
};

template <typename TemperatureType>
ThermalModel( const TemperatureType, const double, const double )
    -> ThermalModel<TemperatureDependent, TemperatureType, NoFracture>;

template <typename TemperatureType>
struct ThermalModel<TemperatureDependent, TemperatureType, CriticalStretch>
    : public FractureModel<CriticalStretch>,
      public ThermalModel<TemperatureDependent, TemperatureType, NoFracture>
{
    using base_fracture_type = FractureModel<CriticalStretch>;
    using base_temperature_type =
        ThermalModel<TemperatureDependent, TemperatureType, NoFracture>;
    using typename base_fracture_type::fracture_tag;

    // Does not use the base bond_break_coeff.
    using base_fracture_type::s0;
    using base_temperature_type::alpha;
    using base_temperature_type::temp0;
    using base_temperature_type::temperature;

    // Does not use the base critical stretch.
    using base_temperature_type::operator();

    ThermalModel( const base_fracture_type fracture,
                  const TemperatureType _temp, const double _alpha,
                  const double _temp0 )
        : base_fracture_type( fracture )
        , base_temperature_type( _temp, _alpha, _temp0 )
    {
    }

    template <typename ModelType1, typename ModelType2>
    ThermalModel( const ModelType1& model1, const ModelType2& model2 )
        : base_fracture_type( model1, model2 )
        , base_temperature_type( model1, model2 )
    {
    }

    KOKKOS_INLINE_FUNCTION
    bool operator()( CriticalStretchTag, const int i, const int j,
                     const double r, const double xi ) const
    {
        double temp_avg = 0.5 * ( temperature( i ) + temperature( j ) ) - temp0;
        double bond_break_coeff =
            ( 1.0 + s0 + alpha * temp_avg ) * ( 1.0 + s0 + alpha * temp_avg );
        return r * r >= bond_break_coeff * xi * xi;
    }
};

template <typename ForceModelType, typename TemperatureType>
ThermalModel( ForceModelType, const TemperatureType, const double,
              const double )
    -> ThermalModel<TemperatureDependent, TemperatureType, CriticalStretch>;

// This class stores temperature parameters needed for heat transfer, but not
// the temperature itself (stored instead in the static temperature class
// above).
struct BaseDynamicTemperatureModel
{
    using thermal_tag = DynamicTemperature;

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

template <typename TemperatureType>
struct ThermalModel<DynamicTemperature, TemperatureType, NoFracture>
    : public BaseDynamicTemperatureModel,
      ThermalModel<TemperatureDependent, TemperatureType, NoFracture>
{
    using thermal_tag = TemperatureDependent;
    using base_heattransfer_type = BaseDynamicTemperatureModel;
    using base_temperature_type =
        ThermalModel<TemperatureDependent, TemperatureType, NoFracture>;
    using typename base_temperature_type::fracture_tag;

    ThermalModel( const TemperatureType _temp, const double _thermal_horizon,
                  const double _alpha, const double _kappa, const double _cp,
                  const double _temp0,
                  const bool _constant_microconductivity = true )
        : base_heattransfer_type( _thermal_horizon, _kappa, _cp,
                                  _constant_microconductivity )
        , base_temperature_type( _temp, _alpha, _temp0 )
    {
    }

    template <typename ModelType1, typename ModelType2>
    ThermalModel( const ModelType1& model1, const ModelType2& model2 )
        : base_heattransfer_type( model1, model2 )
        , base_temperature_type( model1, model2 )
    {
    }
};

template <typename TemperatureType>
ThermalModel( const TemperatureType, const double, const double, const double,
              const double, const double, const bool = true )
    -> ThermalModel<DynamicTemperature, TemperatureType, NoFracture>;

template <typename TemperatureType>
struct ThermalModel<DynamicTemperature, TemperatureType, CriticalStretch>
    : public BaseDynamicTemperatureModel,
      ThermalModel<TemperatureDependent, TemperatureType, CriticalStretch>
{
    using thermal_tag = TemperatureDependent;
    using base_heattransfer_type = BaseDynamicTemperatureModel;
    using base_temperature_type =
        ThermalModel<TemperatureDependent, TemperatureType, CriticalStretch>;

    template <typename ForceModelType>
    ThermalModel( ForceModelType force_model, const TemperatureType _temp,
                  const double _thermal_horizon, const double _alpha,
                  const double _kappa, const double _cp, const double _temp0,
                  const bool _constant_microconductivity = true )
        : base_heattransfer_type( _thermal_horizon, _kappa, _cp,
                                  _constant_microconductivity )
        , base_temperature_type( force_model, _temp, _alpha, _temp0 )

    {
    }

    template <typename ModelType1, typename ModelType2>
    ThermalModel( const ModelType1& model1, const ModelType2& model2 )
        : base_heattransfer_type( model1, model2 )
        , base_temperature_type( model1, model2 )
    {
    }
};

template <typename ForceModelType, typename TemperatureType>
ThermalModel( ForceModelType, const TemperatureType, const double, const double,
              const double, const double, const double, const bool = true )
    -> ThermalModel<DynamicTemperature, TemperatureType, CriticalStretch>;

namespace Experimental
{
template <typename MechanicsType, typename FractureType>
struct ForceModel;

template <typename T>
struct is_force_model : public std::false_type
{
};
template <typename MechanicsType, typename FractureType>
struct is_force_model<ForceModel<MechanicsType, FractureType>>
    : public std::true_type
{
};

template <typename MechanicsType,
          typename FractureType = FractureModel<NoFracture>>
struct ForceModel
    : public MechanicsType,
      FractureType,
      ThermalModel<TemperatureIndependent, TemperatureIndependent, NoFracture>
{
    using MechanicsType::operator();
    using FractureType::operator();
    using typename FractureType::fracture_tag;
    using thermal_type = ThermalModel<TemperatureIndependent,
                                      TemperatureIndependent, NoFracture>;
    using thermal_type::operator();

    ForceModel( MechanicsType mechanics, FractureType thermal )
        : MechanicsType( mechanics )
        , FractureType( thermal )
        , thermal_type()
    {
    }

    ForceModel( MechanicsType mechanics )
        : MechanicsType( mechanics )
        , FractureType()
        , thermal_type()
    {
    }

    // Constructor to average from existing models.
    template <typename ModelType1, typename ModelType2>
    ForceModel(
        const ModelType1& model1, const ModelType2& model2,
        typename std::enable_if_t<is_force_model<ModelType1>::value &&
                                  is_force_model<ModelType2>::value> = 0 )
        : MechanicsType( model1, model2 )
        , FractureType( model1, model2 )
    {
    }
};

template <typename MechanicsType, typename ThermalType>
struct ThermalForceModel;

template <typename MechanicsType, typename ThermalType>
struct is_force_model<ThermalForceModel<MechanicsType, ThermalType>>
    : public std::true_type
{
};

template <typename MechanicsType, typename ThermalType>
struct ThermalForceModel : public MechanicsType, ThermalType
{
    using MechanicsType::operator();
    using ThermalType::operator();
    using typename ThermalType::needs_update;
    using typename ThermalType::thermal_tag;

    ThermalForceModel( MechanicsType mechanics, ThermalType thermal )
        : MechanicsType( mechanics )
        , ThermalType( thermal )
    {
    }

    // Constructor to average from existing models.
    template <typename ModelType1, typename ModelType2>
    ThermalForceModel(
        const ModelType1& model1, const ModelType2& model2,
        typename std::enable_if_t<is_force_model<ModelType1>::value &&
                                  is_force_model<ModelType2>::value> = 0 )
        : MechanicsType( model1, model2 )
        , ThermalType( model1, model2 )
    {
    }
};

template <typename MechanicsType, typename PhysicsType>
ForceModel( MechanicsType mechanics, PhysicsType thermal )
    -> ForceModel<MechanicsType, PhysicsType>;

} // namespace Experimental
} // namespace CabanaPD

#endif
