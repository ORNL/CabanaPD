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
#include <CabanaPD_Properties.hpp>
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

/******************************************************************************
  Forward declarations.
******************************************************************************/
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

/******************************************************************************
  Base models.
******************************************************************************/
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

template <typename T>
struct is_fracture_model : public std::false_type
{
};
template <typename FractureType>
struct is_fracture_model<FractureModel<FractureType>> : public std::true_type
{
};

template <>
struct FractureModel<NoFracture>
{
    using fracture_tag = NoFracture;

    FractureModel() = default;

    // Average from existing models.
    template <typename ModelType1, typename ModelType2>
    FractureModel(
        const ModelType1&, const ModelType2&,
        typename std::enable_if_t<( is_force_model<ModelType1>::value &&
                                    is_force_model<ModelType2>::value ),
                                  int>* = 0 )
    {
    }

    // This should only be used in multi-material models in which the
    // current model does not support failure.
    KOKKOS_INLINE_FUNCTION
    bool operator()( CriticalStretchTag, const int, const int, const double,
                     const double ) const
    {
        return false;
    }

    KOKKOS_FUNCTION
    auto criticalStretch() const { return DBL_MAX; }

    KOKKOS_FUNCTION
    auto fractureEnergy() const { return DBL_MAX; }
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
    FractureModel(
        const ModelType1& model1, const ModelType2& model2,
        typename std::enable_if_t<( is_force_model<ModelType1>::value &&
                                    is_force_model<ModelType2>::value ),
                                  int>* = 0 )
    {
        G0 = ( model1.fractureEnergy() + model2.fractureEnergy() ) / 2.0;
        const double s0_1 = model1.criticalStretch();
        const double s0_2 = model2.criticalStretch();
        s0 = Kokkos::sqrt( ( s0_1 * s0_1 * model1.K + s0_2 * s0_2 * model2.K ) /
                           ( model1.K + model2.K ) );
        bond_break_coeff = ( 1.0 + s0 ) * ( 1.0 + s0 );
    }

    KOKKOS_INLINE_FUNCTION
    bool operator()( CriticalStretchTag, const int, const int, const double r,
                     const double xi ) const
    {
        return r * r >= bond_break_coeff * xi * xi;
    }

    KOKKOS_FUNCTION
    auto criticalStretch() const { return s0; }

    KOKKOS_FUNCTION
    auto fractureEnergy() const { return G0; }

    KOKKOS_FUNCTION
    auto influenceType() const { return influence_type; }
};

FractureModel( const double _force_horizon, const double _K, const double _G0,
               const int influence = 1 )
    ->FractureModel<CriticalStretch>;

/******************************************************************************
  Temperature-dependent models.
******************************************************************************/
template <typename ThermalType, typename TemperatureType, typename FunctorType,
          typename FractureType = NoFracture>
struct ThermalModel;

template <typename T>
struct is_thermal_model : public std::false_type
{
};
template <typename ThermalType, typename TemperatureType, typename FractureType>
struct is_thermal_model<
    ThermalModel<ThermalType, TemperatureType, FractureType>>
    : public std::true_type
{
};

// Placeholder template arguments.
template <>
struct ThermalModel<TemperatureIndependent, TemperatureIndependent,
                    TemperatureIndependent, NoFracture>
{
    using thermal_tag = TemperatureIndependent;

    KOKKOS_FUNCTION
    double operator()( ThermalStretchTag, const int, const int,
                       const double s ) const
    {
        return s;
    }
};

template <typename TemperatureType, typename FunctorType>
struct ThermalModel<TemperatureDependent, TemperatureType, FunctorType,
                    NoFracture>
{
    using thermal_tag = TemperatureDependent;
    using needs_update = std::true_type;
    using fracture_tag = NoFracture;

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
    // FIXME: does not work for both constant and custom alpha yet
    template <typename ModelType1, typename ModelType2>
    ThermalModel( const ModelType1& model1, const ModelType2& model2 )
        : alpha( ConstantProperty( model1.alpha, model2.alpha ) )
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
        alpha.update( particles );
    }

    // Update stretch with temperature effects.
    KOKKOS_INLINE_FUNCTION
    double operator()( ThermalStretchTag, const int i, const int j,
                       const double s ) const
    {
        const double temp_avg =
            0.5 * ( temperature( i ) + temperature( j ) ) - temp0;
        const double alpha_avg = 0.5 * ( alpha( i ) + alpha( j ) );
        return s - ( alpha_avg * temp_avg );
    }
};

template <typename TemperatureType>
ThermalModel( const TemperatureType, const double, const double )
    -> ThermalModel<TemperatureDependent, TemperatureType, ConstantProperty,
                    NoFracture>;

template <typename TemperatureType, typename FunctorType>
ThermalModel( const TemperatureType, const FunctorType, const double )
    -> ThermalModel<TemperatureDependent, TemperatureType, FunctorType,
                    NoFracture>;

template <typename TemperatureType, typename FunctorType>
struct ThermalModel<TemperatureDependent, TemperatureType, FunctorType,
                    CriticalStretch>
    : public FractureModel<CriticalStretch>,
      public ThermalModel<TemperatureDependent, TemperatureType, FunctorType,
                          NoFracture>
{
    using base_type =
        ThermalModel<TemperatureDependent, TemperatureType, NoFracture>;
    using base_type::operator();
    using fracture_tag = Fracture;

    double s0;
    using base_type::alpha;
    using base_type::temp0;
    using base_type::temperature;

    ThermalModel( const double _s0, const TemperatureType _temp,
                  const double _alpha, const double _temp0 )
        : ThermalModel( _s0, _temp, ConstantProperty( _alpha ), _temp0 )
    {
    }

    ThermalModel( const double _s0, const TemperatureType _temp,
                  const FunctorType _alpha, const double _temp0 )
        : base_type( _temp, _alpha, _temp0 )
        , s0( _s0 )
    {
    }

    template <typename ModelType1, typename ModelType2>
    ThermalModel( const ModelType1& model1, const ModelType2& model2 )
        : base_type( model1, model2 )
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

template <typename TemperatureType>
ThermalModel( const double, const TemperatureType, const double, const double )
    -> ThermalModel<TemperatureDependent, TemperatureType, ConstantProperty,
                    CriticalStretch>;

template <typename TemperatureType, typename FunctorType>
ThermalModel( const double, const TemperatureType, const FunctorType,
              const double )
    -> ThermalModel<TemperatureDependent, TemperatureType, FunctorType,
                    CriticalStretch>;

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

template <typename TemperatureType, typename FunctoryType>
struct ThermalModel<DynamicTemperature, TemperatureType, FunctoryType,
                    NoFracture>
    : public BaseDynamicTemperatureModel,
      ThermalModel<TemperatureDependent, TemperatureType, FunctoryType,
                   NoFracture>
{
    using base_heattransfer_type = BaseDynamicTemperatureModel;
    using typename base_heattransfer_type::thermal_tag;
    using base_type = ThermalModel<TemperatureDependent, TemperatureType,
                                   NoFracture, FunctoryType>;
    using typename base_type::fracture_tag;
    using base_type::operator();

    ThermalModel( const double _thermal_horizon, const TemperatureType _temp,
                  const double _alpha, const double _kappa, const double _cp,
                  const double _temp0,
                  const bool _constant_microconductivity = true )
        : base_heattransfer_type( _thermal_horizon, _kappa, _cp,
                                  _constant_microconductivity )
        , base_type( _temp, _alpha, _temp0 )
    {
    }

    template <typename ModelType1, typename ModelType2>
    ThermalModel( const ModelType1& model1, const ModelType2& model2 )
        : base_heattransfer_type( model1, model2 )
        , base_type( model1, model2 )
    {
    }
};

template <typename TemperatureType>
ThermalModel( const double, const TemperatureType, const double, const double,
              const double, const double, const bool = true )
    -> ThermalModel<DynamicTemperature, TemperatureType, ConstantProperty,
                    NoFracture>;

template <typename TemperatureType, typename FunctorType>
ThermalModel( const TemperatureType, const double, const FunctorType,
              const double, const double, const double, const bool = true )
    -> ThermalModel<DynamicTemperature, TemperatureType, FunctorType,
                    NoFracture>;

template <typename TemperatureType, typename FunctorType>
struct ThermalModel<DynamicTemperature, TemperatureType, FunctorType,
                    CriticalStretch>
    : public BaseDynamicTemperatureModel,
      ThermalModel<TemperatureDependent, TemperatureType, FunctorType,
                   CriticalStretch>
{
    using base_heattransfer_type = BaseDynamicTemperatureModel;
    using typename base_heattransfer_type::thermal_tag;
    using base_type = ThermalModel<TemperatureDependent, TemperatureType,
                                   CriticalStretch, FunctoryType>;
    using base_type::operator();

    ThermalModel( const double _thermal_horizon, const double _s0,
                  const TemperatureType _temp, const double _alpha,
                  const double _kappa, const double _cp, const double _temp0,
                  const bool _constant_microconductivity = true )
        : base_heattransfer_type( _thermal_horizon, _kappa, _cp,
                                  _constant_microconductivity )
        , base_type( _s0, _temp, _alpha, _temp0 )

    {
    }

    template <typename ModelType1, typename ModelType2>
    ThermalModel( const ModelType1& model1, const ModelType2& model2 )
        : base_heattransfer_type( model1, model2 )
        , base_type( model1, model2 )
    {
    }
};

template <typename TemperatureType>
ThermalModel( const double, const TemperatureType, const double, const double,
              const double, const double, const double, const bool = true )
    -> ThermalModel<DynamicTemperature, TemperatureType, ConstantProperty,
                    CriticalStretch>;

/******************************************************************************
Force models.
******************************************************************************/
template <typename MechanicsType,
          typename FractureType = FractureModel<NoFracture>>
struct ForceModel : public MechanicsType,
                    FractureType,
                    ThermalModel<TemperatureIndependent, TemperatureIndependent,
                                 TemperatureIndependent, NoFracture>
{
    using MechanicsType::operator();
    using MechanicsType::influenceType;
    using FractureType::operator();
    using thermal_tag = TemperatureIndependent;
    using typename FractureType::fracture_tag;
    using thermal_type =
        ThermalModel<TemperatureIndependent, TemperatureIndependent,
                     TemperatureIndependent, NoFracture>;
    using thermal_type::operator();

    ForceModel(
        MechanicsType mechanics, FractureType fracture,
        typename std::enable_if_t<( is_mechanics_model<MechanicsType>::value &&
                                    is_fracture_model<FractureType>::value ),
                                  int>* = 0 )
        : MechanicsType( mechanics )
        , FractureType( fracture )
        , thermal_type()
    {
    }

    ForceModel( MechanicsType mechanics )
        : ForceModel( mechanics, FractureType() )
    {
    }

    // Constructor to average from existing models.
    template <typename ModelType1, typename ModelType2>
    ForceModel( const ModelType1& model1, const ModelType2& model2,
                typename std::enable_if_t<( is_force_model<ModelType1>::value &&
                                            is_force_model<ModelType2>::value ),
                                          int>* = 0 )
        : MechanicsType( model1, model2 )
        , FractureType( model1, model2 )
    {
    }
};

template <typename MechanicsType, typename PhysicsType>
ForceModel( MechanicsType mechanics, PhysicsType thermal )
    -> ForceModel<MechanicsType, PhysicsType>;

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

    ThermalForceModel(
        MechanicsType mechanics, ThermalType thermal,
        typename std::enable_if_t<( is_mechanics_model<MechanicsType>::value &&
                                    is_thermal_model<ThermalType>::value ),
                                  int>* = 0 )
        : MechanicsType( mechanics )
        , ThermalType( thermal )
    {
    }

    // Constructor to average from existing models.
    template <typename ModelType1, typename ModelType2>
    ThermalForceModel(
        const ModelType1& model1, const ModelType2& model2,
        typename std::enable_if_t<( is_force_model<ModelType1>::value &&
                                    is_force_model<ModelType2>::value ),
                                  int>* = 0 )
        : MechanicsType( model1, model2 )
        , ThermalType( model1, model2 )
    {
    }
};

} // namespace CabanaPD

#endif
