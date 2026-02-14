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

#ifndef FORCE_MODELS_PMB_H
#define FORCE_MODELS_PMB_H

#include <Kokkos_Core.hpp>

#include <CabanaPD_Constants.hpp>
#include <CabanaPD_ForceModels.hpp>
#include <CabanaPD_Types.hpp>
#include <force_models/CabanaPD_Contact.hpp>

namespace CabanaPD
{

template <typename MechanicsModelType, typename... DataTypes>
struct BaseForceModelPMB;

template <>
struct BaseForceModelPMB<Elastic> : public BaseForceModel
{
    using base_type = BaseForceModel;
    using mechanics_type = Elastic;

    // Tags for creating particle fields and dispatch to force iteration.
    using model_tag = PMB;
    using force_tag = PMB;

    using base_type::force_horizon;
    using base_type::K;
    double c;

    BaseForceModelPMB( PMB, NoFracture, const double force_horizon,
                       const double _K )
        : base_type( force_horizon, _K )
    {
        init();
    }

    BaseForceModelPMB( PMB, Elastic, NoFracture, const double force_horizon,
                       const double _K )
        : base_type( force_horizon, _K )
    {
        init();
    }

    void init()
    {
        c = 18.0 * K /
            ( pi * force_horizon * force_horizon * force_horizon *
              force_horizon );
    }

    // Constructor to average from existing models.
    template <typename ModelType1, typename ModelType2>
    BaseForceModelPMB( const ModelType1& model1, const ModelType2& model2 )
        : base_type( model1, model2 )
    {
        c = ( model1.c + model2.c ) / 2.0;
    }

    KOKKOS_INLINE_FUNCTION
    auto operator()( ForceCoeffTag, const int, const int, const double s,
                     const double vol, const int = -1 ) const
    {
        return c * s * vol;
    }

    KOKKOS_INLINE_FUNCTION
    auto operator()( EnergyTag, const int, const int, const double s,
                     const double xi, const double vol, const int = -1 ) const
    {
        // 0.25 factor is due to 1/2 from outside the integral and 1/2 from
        // the integrand (pairwise potential).
        return 0.25 * c * s * s * xi * vol;
    }

    // For multi material.
    KOKKOS_INLINE_FUNCTION
    auto operator()( DilatationTag, const int, const int, const double,
                     const double, const double, const double ) const
    {
        return 0.0;
    }
    auto getYieldStretch() const { return DBL_MAX; }
    auto getPlasticStretch( const int, const int ) const { return 0.0; }
};

template <typename MemorySpace>
struct BaseForceModelPMB<ElasticPerfectlyPlastic, MemorySpace>
    : public BaseForceModelPMB<Elastic>, public BasePlasticity<MemorySpace>
{
    using base_type = BaseForceModelPMB<Elastic>;
    using base_plasticity_type = BasePlasticity<MemorySpace>;

    using mechanics_type = ElasticPerfectlyPlastic;

    using base_plasticity_type::_s_p;
    using base_type::c;
    double s_Y;

    using base_plasticity_type::updateBonds;

    BaseForceModelPMB( PMB model, mechanics_type, MemorySpace,
                       const double force_horizon, const double K,
                       const double sigma_y )
        : base_type( model, NoFracture{}, force_horizon, K )
        , base_plasticity_type()
        , s_Y( sigma_y / 3.0 / K )
    {
    }

    // Constructor to average from existing models.
    template <typename ModelType1, typename ModelType2>
    BaseForceModelPMB( const ModelType1& model1, const ModelType2& model2 )
        : base_type( model1, model2 )
        , base_plasticity_type()
    {
        s_Y = ( model1.getYieldStretch() + model2.getYieldStretch() ) / 2.0;
    }

    auto getYieldStretch() const { return s_Y; }

    auto updatePlasticStretch( const int i, const double s, const int n ) const
    {
        // Update bond plastic stretch.
        auto s_p = _s_p( i, n );
        // Yield in tension.
        if ( s >= s_p + s_Y )
            _s_p( i, n ) = s - s_Y;
        // Yield in compression.
        else if ( s <= s_p - s_Y )
            _s_p( i, n ) = s + s_Y;
        // else: Elastic (in between), do not modify.

        return _s_p( i, n );
    }

    auto getPlasticStretch( const int i, const int n ) const
    {
        return _s_p( i, n );
    }

    KOKKOS_INLINE_FUNCTION
    auto operator()( ForceCoeffTag, const int i, const int, const double s,
                     const double vol, const int n ) const
    {
        const double s_p = updatePlasticStretch( i, s, n );
        return c * ( s - s_p ) * vol;
    }

    // This energy calculation is only valid for pure tension or pure
    // compression.
    KOKKOS_INLINE_FUNCTION
    auto operator()( EnergyTag, const int i, const int, const double s,
                     const double xi, const double vol, const int n ) const
    {
        auto s_p = _s_p( i, n );
        double stretch_term;
        // Yield in tension.
        if ( s >= s_p + s_Y )
            stretch_term = s_Y * ( 2.0 * s - s_Y );
        // Yield in compression.
        else if ( s <= s_p - s_Y )
            stretch_term = s_Y * ( -2.0 * s - s_Y );
        else
            // Elastic (in between).
            stretch_term = s * s;

        // 0.25 factor is due to 1/2 from outside the integral and 1/2 from
        // the integrand (pairwise potential).
        return 0.25 * c * stretch_term * xi * vol;
    }
};

template <>
struct ForceModel<PMB, Elastic, NoFracture, TemperatureIndependent>
    : public BaseForceModelPMB<Elastic>,
      BaseNoFractureModel,
      BaseTemperatureModel<TemperatureIndependent>
{
    using base_type = BaseForceModelPMB<Elastic>;
    using base_fracture_type = BaseNoFractureModel;
    using base_temperature_type = BaseTemperatureModel<TemperatureIndependent>;

    using base_type::base_type;
    using base_type::force_horizon;
    using base_type::operator();
    using base_fracture_type::operator();
    using base_temperature_type::operator();
};

template <>
struct ForceModel<PMB, Elastic, Fracture, TemperatureIndependent>
    : public BaseForceModelPMB<Elastic>,
      BaseFractureModel,
      BaseTemperatureModel<TemperatureIndependent>
{
    using base_type = BaseForceModelPMB<Elastic>;
    using base_fracture_type = BaseFractureModel;
    using base_temperature_type = BaseTemperatureModel<TemperatureIndependent>;

    using base_type::operator();
    using base_fracture_type::operator();
    using base_temperature_type::operator();

    ForceModel( PMB model, const double force_horizon, const double K,
                const double G0, const int influence_type = 1 )
        : base_type( model, NoFracture{}, force_horizon, K )
        , base_fracture_type( force_horizon, K, G0, influence_type )
    {
    }

    ForceModel( PMB model, Elastic elastic, const double force_horizon,
                const double K, const double G0, const int influence_type = 1 )
        : base_type( model, elastic, NoFracture{}, force_horizon, K )
        , base_fracture_type( force_horizon, K, G0, influence_type )
    {
    }

    ForceModel( PMB model, Fracture, const double force_horizon, const double K,
                const double G0, const int influence_type = 1 )
        : base_type( model, Elastic{}, NoFracture{}, force_horizon, K )
        , base_fracture_type( force_horizon, K, G0, influence_type )
    {
    }

    ForceModel( PMB model, Elastic, Fracture, const double force_horizon,
                const double K, const double G0, const int influence_type = 1 )
        : base_type( model, NoFracture{}, force_horizon, K )
        , base_fracture_type( force_horizon, K, G0, influence_type )
    {
    }

    // Constructor to work with plasticity.
    ForceModel( PMB model, Elastic elastic, const double force_horizon,
                const double K, const double G0, const double s0 )
        : base_type( model, elastic, NoFracture{}, force_horizon, K )
        , base_fracture_type( G0, s0 )
    {
    }

    // Constructor to average from existing models.
    template <typename ModelType1, typename ModelType2>
    ForceModel( const ModelType1& model1, const ModelType2& model2 )
        : base_type( model1, model2 )
        , base_fracture_type( model1, model2 )
    {
    }
};

template <typename MemorySpace>
struct ForceModel<PMB, ElasticPerfectlyPlastic, Fracture,
                  TemperatureIndependent, MemorySpace>
    : public BaseForceModelPMB<ElasticPerfectlyPlastic, MemorySpace>,
      public BaseFractureModel,
      public BaseTemperatureModel<TemperatureIndependent>
{
    using base_type = BaseForceModelPMB<ElasticPerfectlyPlastic, MemorySpace>;
    using base_fracture_type = BaseFractureModel;
    using base_temperature_type = BaseTemperatureModel<TemperatureIndependent>;

    using base_type::operator();
    using base_fracture_type::operator();
    using base_temperature_type::operator();
    using base_type::updateBonds;

    ForceModel( PMB model, ElasticPerfectlyPlastic mechanics, MemorySpace space,
                const double force_horizon, const double K, const double G0,
                const double sigma_y )
        : base_type( model, mechanics, space, force_horizon, K, sigma_y )
        , base_fracture_type(
              G0,
              // s0
              ( 5.0 * G0 / sigma_y / force_horizon + sigma_y / K ) / 6.0 )
        , base_temperature_type()
    {
    }

    // Constructor to average from existing models.
    template <typename ModelType1, typename ModelType2>
    ForceModel( const ModelType1& model1, const ModelType2& model2 )
        : base_type( model1, model2 )
        , base_fracture_type( model1, model2 )
    {
    }
};

template <typename ModelType>
ForceModel( ModelType, Elastic, NoFracture, const double force_horizon,
            const double K ) -> ForceModel<ModelType, Elastic, NoFracture>;

template <typename ModelType>
ForceModel( ModelType, Elastic, Fracture, const double force_horizon,
            const double K, const double G0 ) -> ForceModel<ModelType, Elastic>;

// Default to fracture.
template <typename ModelType>
ForceModel( ModelType, Elastic, const double force_horizon, const double K,
            const double G0 ) -> ForceModel<ModelType, Elastic>;

template <typename ModelType>
ForceModel( ModelType, NoFracture, const double force_horizon, const double K )
    -> ForceModel<ModelType, Elastic, NoFracture>;

// Default to elastic.
template <typename ModelType>
ForceModel( ModelType, const double force_horizon, const double K,
            const double G0 ) -> ForceModel<ModelType>;

template <typename ModelType, typename MemorySpace>
ForceModel( ModelType, ElasticPerfectlyPlastic, MemorySpace,
            const double force_horizon, const double K, const double G0,
            const double sigma_y )
    -> ForceModel<ModelType, ElasticPerfectlyPlastic, Fracture,
                  TemperatureIndependent, MemorySpace>;

template <typename TemperatureType>
struct ForceModel<PMB, Elastic, NoFracture, TemperatureDependent,
                  TemperatureType>
    : public BaseForceModelPMB<Elastic>,
      BaseNoFractureModel,
      BaseTemperatureModel<TemperatureDependent, TemperatureType>
{
    using base_type = BaseForceModelPMB<Elastic>;
    using base_temperature_type =
        BaseTemperatureModel<TemperatureDependent, TemperatureType>;

    using base_type::operator();
    using base_temperature_type::operator();
    using typename base_temperature_type::needs_update;

    ForceModel( PMB model, NoFracture fracture, const double _force_horizon,
                const double _K, const TemperatureType _temp,
                const double _alpha, const double _temp0 = 0.0 )
        : base_type( model, fracture, _force_horizon, _K )
        , base_temperature_type( _temp, _alpha, _temp0 )
    {
    }

    // Constructor to average from existing models.
    template <typename ModelType1, typename ModelType2>
    ForceModel( const ModelType1& model1, const ModelType2& model2 )
        : base_type( model1, model2 )
        , base_temperature_type( model1, model2 )
    {
    }
};

template <typename TemperatureType>
struct ForceModel<PMB, Elastic, Fracture, TemperatureDependent, TemperatureType>
    : public BaseForceModelPMB<Elastic>, ThermalFractureModel<TemperatureType>
{
    using base_type = BaseForceModelPMB<Elastic>;
    using base_temperature_type = ThermalFractureModel<TemperatureType>;

    using base_type::operator();
    using base_temperature_type::operator();
    using typename base_temperature_type::needs_update;

    ForceModel( PMB model, const double _horizon, const double _K,
                const double _G0, const TemperatureType _temp,
                const double _alpha, const double _temp0 = 0.0 )
        : base_type( model, NoFracture{}, _horizon, _K )
        , base_temperature_type( _horizon, _K, _G0, _temp, _alpha, _temp0 )
    {
    }

    // Constructor to average from existing models.
    template <typename ModelType1, typename ModelType2>
    ForceModel( const ModelType1& model1, const ModelType2& model2 )
        : base_type( model1, model2 )
        , base_temperature_type( model1, model2 )
    {
    }
};

template <typename TemperatureType>
struct ForceModel<PMB, ElasticPerfectlyPlastic, Fracture, TemperatureDependent,
                  TemperatureType>
    : public BaseForceModelPMB<ElasticPerfectlyPlastic,
                               typename TemperatureType::memory_space>,
      public ThermalFractureModel<TemperatureType>
{
    using base_type = BaseForceModelPMB<ElasticPerfectlyPlastic,
                                        typename TemperatureType::memory_space>;
    using base_temperature_type = ThermalFractureModel<TemperatureType>;

    using base_type::operator();
    using base_temperature_type::operator();
    using typename base_temperature_type::needs_update;

    ForceModel( PMB model, ElasticPerfectlyPlastic mechanics,
                const double _horizon, const double _K, const double _G0,
                const double sigma_y, const TemperatureType _temp,
                const double _alpha, const double _temp0 = 0.0 )
        : base_type( model, mechanics, typename TemperatureType::memory_space{},
                     _horizon, _K, sigma_y )
        , base_temperature_type( _horizon, _K, _G0, _temp, _alpha, _temp0 )
    {
    }

    // Constructor to average from existing models.
    template <typename ModelType1, typename ModelType2>
    ForceModel( const ModelType1& model1, const ModelType2& model2 )
        : base_type( model1, model2 )
        , base_temperature_type( model1, model2 )
    {
    }
};

template <typename ModelType, typename TemperatureType>
ForceModel( ModelType, NoFracture, const double horizon, const double K,
            const TemperatureType& temp, const double alpha,
            const double temp0 = 0.0 )
    -> ForceModel<ModelType, Elastic, NoFracture, TemperatureDependent,
                  TemperatureType>;

template <typename ModelType, typename TemperatureType>
ForceModel( ModelType, const double horizon, const double K, const double _G0,
            const TemperatureType& temp, const double alpha,
            const double temp0 = 0.0 )
    -> ForceModel<ModelType, Elastic, Fracture, TemperatureDependent,
                  TemperatureType>;

template <typename ModelType, typename TemperatureType>
ForceModel( ModelType, ElasticPerfectlyPlastic, const double horizon,
            const double K, const double _G0, const double sigma_y,
            const TemperatureType& temp, const double alpha,
            const double temp0 = 0.0 )
    -> ForceModel<ModelType, ElasticPerfectlyPlastic, Fracture,
                  TemperatureDependent, TemperatureType>;

template <typename TemperatureType>
struct ForceModel<PMB, Elastic, NoFracture, DynamicTemperature, TemperatureType>
    : public BaseForceModelPMB<Elastic>,
      BaseNoFractureModel,
      BaseTemperatureModel<TemperatureDependent, TemperatureType>,
      BaseDynamicTemperatureModel
{
    using base_type = BaseForceModelPMB<Elastic>;
    using base_temperature_type =
        BaseTemperatureModel<TemperatureDependent, TemperatureType>;
    using base_heat_transfer_type = BaseDynamicTemperatureModel;

    // Necessary to distinguish between TemperatureDependent
    using thermal_type = DynamicTemperature;

    using base_type::operator();
    using base_temperature_type::operator();
    using base_heat_transfer_type::operator();
    using typename base_temperature_type::needs_update;

    ForceModel( PMB model, NoFracture fracture, const double _horizon,
                const double _K, const TemperatureType& _temp,
                const double _kappa, const double _cp, const double _alpha,
                const double _temp0 = 0.0,
                const bool _constant_microconductivity = true )
        : base_type( model, fracture, _horizon, _K )
        , base_temperature_type( _temp, _alpha, _temp0 )
        , base_heat_transfer_type( _horizon, _kappa, _cp,
                                   _constant_microconductivity )
    {
    }

    // Constructor to average from existing models.
    template <typename ModelType1, typename ModelType2>
    ForceModel( const ModelType1& model1, const ModelType2& model2 )
        : base_type( model1, model2 )
        , base_temperature_type( model1, model2 )
    {
    }
};

template <typename ModelType, typename TemperatureType>
ForceModel( ModelType, NoFracture, const double horizon, const double K,
            const TemperatureType& temp, const double kappa, const double cp,
            const double alpha, const double temp0 = 0.0,
            const bool constant_microconductivity = true )
    -> ForceModel<ModelType, Elastic, NoFracture, DynamicTemperature,
                  TemperatureType>;

template <typename ModelType, typename TemperatureType>
struct ForceModel<ModelType, Elastic, Fracture, DynamicTemperature,
                  TemperatureType> : public BaseForceModelPMB<Elastic>,
                                     ThermalFractureModel<TemperatureType>,
                                     BaseDynamicTemperatureModel
{
    using base_type = BaseForceModelPMB<Elastic>;
    using base_temperature_type = ThermalFractureModel<TemperatureType>;
    using base_heat_transfer_type = BaseDynamicTemperatureModel;

    // Necessary to distinguish between TemperatureDependent
    using thermal_type = DynamicTemperature;

    using base_type::operator();
    using base_temperature_type::operator();
    using base_heat_transfer_type::operator();
    using typename base_temperature_type::needs_update;

    ForceModel( PMB model, const double _horizon, const double _K,
                const double _G0, const TemperatureType _temp,
                const double _kappa, const double _cp, const double _alpha,
                const double _temp0 = 0.0,
                const bool _constant_microconductivity = true )
        : base_type( model, NoFracture{}, _horizon, _K )
        , base_temperature_type( _horizon, _K, _G0, _temp, _alpha, _temp0 )
        , base_heat_transfer_type( _horizon, _kappa, _cp,
                                   _constant_microconductivity )
    {
    }

    // Constructor to average from existing models.
    template <typename ModelType1, typename ModelType2>
    ForceModel( const ModelType1& model1, const ModelType2& model2 )
        : base_type( model1, model2 )
        , base_temperature_type( model1, model2 )
        , base_heat_transfer_type( model1, model2 )
    {
    }
};

template <typename ModelType, typename TemperatureType>
ForceModel( ModelType, const double horizon, const double K, const double G0,
            const TemperatureType& temp, const double kappa, const double cp,
            const double alpha, const double temp0 = 0.0,
            const bool constant_microconductivity = true )
    -> ForceModel<ModelType, Elastic, Fracture, DynamicTemperature,
                  TemperatureType>;

template <typename TemperatureType>
struct ForceModel<PMB, ElasticPerfectlyPlastic, Fracture, DynamicTemperature,
                  TemperatureType>
    : public BaseForceModelPMB<ElasticPerfectlyPlastic,
                               typename TemperatureType::memory_space>,
      public ThermalFractureModel<TemperatureType>,
      BaseDynamicTemperatureModel
{
    using base_type = BaseForceModelPMB<ElasticPerfectlyPlastic,
                                        typename TemperatureType::memory_space>;
    using base_temperature_type = ThermalFractureModel<TemperatureType>;
    using base_heat_transfer_type = BaseDynamicTemperatureModel;

    // Necessary to distinguish between TemperatureDependent
    using thermal_type = DynamicTemperature;

    using base_type::operator();
    using base_temperature_type::operator();
    using base_heat_transfer_type::operator();
    using typename base_temperature_type::needs_update;

    ForceModel( PMB model, ElasticPerfectlyPlastic mechanics,
                const double _horizon, const double _K, const double _G0,
                const double sigma_y, const TemperatureType _temp,
                const double _kappa, const double _cp, const double _alpha,
                const double _temp0 = 0.0,
                const bool _constant_microconductivity = true )
        : base_type( model, mechanics, typename TemperatureType::memory_space{},
                     _horizon, _K, sigma_y )
        , base_temperature_type( _horizon, _K, _G0, _temp, _alpha, _temp0 )
        , base_heat_transfer_type( _horizon, _kappa, _cp,
                                   _constant_microconductivity )
    {
    }

    // Constructor to average from existing models.
    template <typename ModelType1, typename ModelType2>
    ForceModel( const ModelType1& model1, const ModelType2& model2 )
        : base_type( model1, model2 )
        , base_temperature_type( model1, model2 )
        , base_heat_transfer_type( model1, model2 )
    {
    }
};

template <typename ModelType, typename TemperatureType>
ForceModel( ModelType, ElasticPerfectlyPlastic, const double horizon,
            const double K, const double G0, const double sigma_y,
            const TemperatureType& temp, const double kappa, const double cp,
            const double alpha, const double temp0 = 0.0,
            const bool constant_microconductivity = true )
    -> ForceModel<ModelType, ElasticPerfectlyPlastic, Fracture,
                  DynamicTemperature, TemperatureType>;

/******************************************************************************
 Linear PMB.
******************************************************************************/
template <>
struct ForceModel<LinearPMB, Elastic, NoFracture, TemperatureIndependent>
    : public ForceModel<PMB, Elastic, NoFracture, TemperatureIndependent>
{
    using base_type =
        ForceModel<PMB, Elastic, NoFracture, TemperatureIndependent>;
    // Tag to dispatch to force iteration.
    using force_tag = LinearPMB;

    using base_type::base_type;
    using base_type::operator();

    template <typename... Args>
    ForceModel( LinearPMB, Args... args )
        : base_type( typename base_type::model_tag{}, args... )
    {
    }
};

template <>
struct ForceModel<LinearPMB, Elastic, Fracture, TemperatureIndependent>
    : public ForceModel<PMB, Elastic, Fracture, TemperatureIndependent>
{
    using base_type =
        ForceModel<PMB, Elastic, Fracture, TemperatureIndependent>;

    // Tag to dispatch to force iteration.
    using force_tag = LinearPMB;

    using base_type::operator();

    template <typename... Args>
    ForceModel( LinearPMB, Args... args )
        : base_type( typename base_type::model_tag{},
                     std::forward<Args>( args )... )
    {
    }
};

template <typename MemorySpace>
struct ForceModel<LinearPMB, ElasticPerfectlyPlastic, Fracture,
                  TemperatureIndependent, MemorySpace>
    : public ForceModel<PMB, ElasticPerfectlyPlastic, Fracture,
                        TemperatureIndependent, MemorySpace>
{
    using base_type = ForceModel<PMB, ElasticPerfectlyPlastic, Fracture,
                                 TemperatureIndependent, MemorySpace>;

    // Tag to dispatch to force iteration.
    using force_tag = LinearPMB;

    using base_type::operator();

    template <typename... Args>
    ForceModel( LinearPMB, Args... args )
        : base_type( typename base_type::base_model{},
                     std::forward<Args>( args )... )
    {
    }
};

template <typename MechanicsType, typename ThermalType, typename... FieldTypes>
struct ForceModel<LinearPMB, MechanicsType, Fracture, ThermalType,
                  FieldTypes...>
    : public ForceModel<PMB, MechanicsType, Fracture, ThermalType,
                        FieldTypes...>
{
    using base_type =
        ForceModel<PMB, MechanicsType, Fracture, ThermalType, FieldTypes...>;

    // Tag to dispatch to force iteration.
    using force_tag = LinearPMB;

    using base_type::operator();

    template <typename... Args>
    ForceModel( LinearPMB, Args... args )
        : base_type( typename base_type::base_model{},
                     std::forward<Args>( args )... )
    {
    }
};

template <typename PDType, typename MechanicsType, typename ThermalType,
          typename FractureType, typename TemperatureType, typename DensityType,
          typename CurrentDensityType>
struct ForceDensityModel;

// Force models with evolving density.
template <typename TemperatureType, typename DensityType,
          typename CurrentDensityType>
struct ForceDensityModel<PMB, ElasticPerfectlyPlastic, Fracture,
                         TemperatureDependent, TemperatureType, DensityType,
                         CurrentDensityType>
    : public ForceModel<PMB, ElasticPerfectlyPlastic, Fracture,
                        TemperatureDependent, TemperatureType>
{
    using base_type = ForceModel<PMB, ElasticPerfectlyPlastic, Fracture,
                                 TemperatureDependent, TemperatureType>;
    using fracture_type = typename base_type::fracture_type;
    using mechanics_type = typename base_type::mechanics_type;
    using thermal_type = typename base_type::thermal_type;
    using density_type = DynamicDensity;

    using base_type::operator();

    // Does not use base_type::c because it is density dependent.
    using base_type::_s_p;
    using base_type::K;
    double coeff;

    double rho0;
    DensityType rho;
    CurrentDensityType rho_current;

    NormalRepulsionModel contact;

    ForceDensityModel( PMB model, ElasticPerfectlyPlastic mechanics,
                       const DensityType& _rho,
                       const CurrentDensityType& _rho_c, const double delta,
                       const double K, const double G0, const double sigma_y,
                       const double _rho0, const double contact_radius,
                       const TemperatureType _temp, const double alpha,
                       const double temp0 = 0.0 )
        : base_type( model, mechanics, delta, K, G0, sigma_y, _temp, alpha,
                     temp0 )
        , rho0( _rho0 )
        , rho( _rho )
        , rho_current( _rho_c )
        , contact( delta, contact_radius, 0.0, K )
    {
        coeff = 18.0 / pi / delta / delta / delta / delta;
    }

    // Constructor to average from existing models.
    template <typename ModelType1, typename ModelType2>
    ForceDensityModel( const ModelType1& model1, const ModelType2& model2 )
        : base_type( model1, model2 )
        , rho0( model1.rho0 )
        , rho( model1.rho )
        , rho_current( model1.rho_current )
        , contact( model1.contact )
    {
        coeff = model1.coeff;
    }

    KOKKOS_INLINE_FUNCTION
    auto currentC( const int i ) const
    {
        // Initial relative density for relative density factor (HARDCODED)
        double D0 = 0.69;
        // Relative density:
        double D = rho_current( i ) / rho0;
        // Relative density factor from C. Van Nguyen et al., Journal of
        // Materials Processing Technology 226 (2015): 134-145.
        double RD = Kokkos::pow( ( D - D0 ) / ( 1.0 - D0 ), 1.46 * ( 1 - D0 ) );

        // Young's modulus from C. Van Nguyen et al., Journal of Materials
        // Processing Technology 226 (2015): 134-145.
        // double T = temperature( i );
        // double E = ( 199510 - 65.63 * T - 0.0276 * T * T - 1.754E-6 * T * T *
        // T ) * RD;
        //
        // Bulk modulus
        // double nu = 0.25; // Use bond-based model
        // double K = E / ( 3 * ( 1 - 2 * nu ) );
        //
        // return coeff * K;

        return coeff * K * RD;
    }

    KOKKOS_INLINE_FUNCTION
    auto operator()( ForceCoeffTag, const int i, const int, const double s,
                     const double vol, const int n ) const
    {
        // Update c given updated density.
        auto c_current = currentC( i );
        // Plastic stretch is already updated.
        auto s_p = base_type::getPlasticStretch( i, n );
        return c_current * ( s - s_p ) * vol;
    }

    // Update plastic dilatation.
    KOKKOS_INLINE_FUNCTION
    auto operator()( DilatationTag, const int i, const int, const double s,
                     const double xi, const double vol, const double,
                     const int n ) const
    {
        // Update plastic stretch and then plastic dilatation.
        const auto s_p = base_type::updatePlasticStretch( i, s, n );
        return coeff / 6.0 * s_p * xi * vol;
    }

    // Density update using plastic dilatation.
    KOKKOS_INLINE_FUNCTION auto operator()( DensityTag, const int i,
                                            const double theta_i ) const
    {
        // Update density using plastic dilatation.
        // Note that this assumes zero initial plastic dilatation.
        rho_current( i ) = Kokkos::min( rho( i ) * Kokkos::exp( -theta_i ),
                                        rho0 ); // exp(theta_i - theta_i_0)
    }

    template <typename ParticleType>
    void update( const ParticleType& particles )
    {
        base_type::update( particles );
        rho = particles.sliceDensity();
        rho_current = particles.sliceCurrentDensity();
    }
};

template <typename DensityType, typename CurrentDensityType,
          typename TemperatureType>
ForceDensityModel( PMB, ElasticPerfectlyPlastic, DensityType rho,
                   const CurrentDensityType& rho_c, const double delta,
                   const double K, const double G0, const double sigma_y,
                   const double rho0, const double contact_radius,
                   TemperatureType temp, const double _alpha,
                   const double _temp0 )
    -> ForceDensityModel<PMB, ElasticPerfectlyPlastic, Fracture,
                         TemperatureDependent, TemperatureType, DensityType,
                         CurrentDensityType>;

// Force models with evolving density with heat transfer.
template <typename TemperatureType, typename DensityType,
          typename CurrentDensityType>
struct ForceDensityModel<PMB, ElasticPerfectlyPlastic, Fracture,
                         DynamicTemperature, TemperatureType, DensityType,
                         CurrentDensityType>
    : public BaseDynamicTemperatureModel,
      public ForceDensityModel<PMB, ElasticPerfectlyPlastic, Fracture,
                               TemperatureDependent, TemperatureType,
                               DensityType, CurrentDensityType>
{
    using base_type = ForceDensityModel<PMB, ElasticPerfectlyPlastic, Fracture,
                                        TemperatureDependent, TemperatureType,
                                        DensityType, CurrentDensityType>;
    using base_temperature_type = BaseDynamicTemperatureModel;

    using fracture_type = typename base_type::fracture_type;
    using mechanics_type = typename base_type::mechanics_type;
    using density_type = typename base_type::density_type;
    using thermal_type = DynamicTemperature;

    using base_type::operator();
    using base_temperature_type::operator();

    ForceDensityModel( PMB model, ElasticPerfectlyPlastic mechanics,
                       const DensityType& _rho,
                       const CurrentDensityType& _rho_c, const double _delta,
                       const double _K, const double _G0, const double _sigma_y,
                       const double _rho0, const double contact_radius,
                       const TemperatureType _temp, const double _kappa,
                       const double _cp, const double _alpha,
                       const double _temp0 = 0.0,
                       const bool _constant_microconductivity = true )
        : base_temperature_type( _delta, _kappa, _cp,
                                 _constant_microconductivity )
        , base_type( model, mechanics, _rho, _rho_c, _delta, _K, _G0, _sigma_y,
                     _rho0, contact_radius, _temp, _alpha, _temp0 )
    {
    }

    // Constructor to average from existing models.
    template <typename ModelType1, typename ModelType2>
    ForceDensityModel( const ModelType1& model1, const ModelType2& model2 )
        : base_temperature_type( model1, model2 )
        , base_type( model1, model2 )
    {
    }
};

template <typename DensityType, typename CurrentDensityType,
          typename TemperatureType>
ForceDensityModel( PMB, ElasticPerfectlyPlastic, DensityType rho,
                   const CurrentDensityType& rho_c, const double delta,
                   const double K, const double G0, const double sigma_y,
                   const double rho0, const double contact_radius,
                   TemperatureType temp, const double _kappa, const double _cp,
                   const double _alpha, const double _temp0 )
    -> ForceDensityModel<PMB, ElasticPerfectlyPlastic, Fracture,
                         DynamicTemperature, TemperatureType, DensityType,
                         CurrentDensityType>;

} // namespace CabanaPD

#endif
