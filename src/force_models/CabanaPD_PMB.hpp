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

namespace CabanaPD
{

template <typename MechanicsModelType, typename AnisotropyType,
          typename... DataTypes>
struct BaseForceModelPMB;

template <>
struct BaseForceModelPMB<Elastic, Isotropic> : public BaseForceModel
{
    using base_type = BaseForceModel;
    using model_type = PMB;
    using base_model = PMB;
    using mechanics_type = Elastic;

    using base_type::delta;
    using base_type::K;
    double _c;

    BaseForceModelPMB( PMB, NoFracture, const double delta, const double _K )
        : base_type( delta, _K )
    {
        init();
    }

    BaseForceModelPMB( PMB, Elastic, NoFracture, const double delta,
                       const double _K )
        : base_type( delta, _K )
    {
        init();
    }

    void init() { _c = 18.0 * K / ( pi * delta * delta * delta * delta ); }

    // Constructor to average from existing models.
    template <typename ModelType1, typename ModelType2>
    BaseForceModelPMB( const ModelType1& model1, const ModelType2& model2 )
        : base_type( model1, model2 )
    {
        _c = ( model1.c() + model2.c() ) / 2.0;
    }

    auto c() const { return _c; }

    KOKKOS_INLINE_FUNCTION
    auto operator()( ForceCoeffTag, const int, const int, const double s,
                     const double vol, const int = -1 ) const
    {
        return c() * s * vol;
    }

    KOKKOS_INLINE_FUNCTION
    auto operator()( EnergyTag, const int, const int, const double s,
                     const double xi, const double vol, const int = -1 ) const
    {
        // 0.25 factor is due to 1/2 from outside the integral and 1/2 from
        // the integrand (pairwise potential).
        return 0.25 * c() * s * s * xi * vol;
    }
};

template <>
struct BaseForceModelPMB<Elastic, Anisotropic>
    : public BaseForceModelPMB<Elastic, Isotropic>
{
    using base_type = BaseForceModelPMB<Elastic, Isotropic>;

    using base_type::delta;
    using base_type::K;

    BaseForceModelPMB( PMB model, NoFracture fracture, const double delta,
                       const double _K )
        : base_type( model, fracture, delta, _K )
    {
        init();
    }

    BaseForceModelPMB( PMB model, Elastic, NoFracture fracture,
                       const double delta, const double _K )
        : base_type( model, fracture, delta, _K )
    {
        init();
    }

    void init() {}

    // Constructor to average from existing models.
    template <typename ModelType1, typename ModelType2>
    BaseForceModelPMB( const ModelType1& model1, const ModelType2& model2 )
        : base_type( model1, model2 )
    {
    }

    auto c() const { return add_orientation_dependence_here; }
};

template <typename MemorySpace, typename AnisotropyType>
struct BaseForceModelPMB<ElasticPerfectlyPlastic, AnisotropyType, MemorySpace>
    : public BaseForceModelPMB<Elastic, AnisotropyType>,
      public BasePlasticity<MemorySpace>
{
    using base_type = BaseForceModelPMB<Elastic, AnisotropyType>;
    using base_plasticity_type = BasePlasticity<MemorySpace>;

    using mechanics_type = ElasticPerfectlyPlastic;

    using base_plasticity_type::_s_p;
    using base_type::c;
    double s_Y;

    using base_plasticity_type::updateBonds;

    BaseForceModelPMB( PMB model, mechanics_type, MemorySpace,
                       const double delta, const double K,
                       const double sigma_y )
        : base_type( model, NoFracture{}, delta, K )
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
        s_Y = ( model1.s_Y + model2.s_Y ) / 2.0;
    }

    KOKKOS_INLINE_FUNCTION
    auto operator()( ForceCoeffTag, const int i, const int, const double s,
                     const double vol, const int n ) const
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

        // Must extract again if in the plastic regime.
        s_p = _s_p( i, n );
        return c() * ( s - s_p ) * vol;
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
        return 0.25 * c() * stretch_term * xi * vol;
    }
};

template <typename AnisotropyType>
struct ForceModel<PMB, Elastic, AnisotropyType, NoFracture,
                  TemperatureIndependent>
    : public BaseForceModelPMB<Elastic, AnisotropyType>,
      BaseNoFractureModel,
      BaseTemperatureModel<TemperatureIndependent>
{
    using base_type = BaseForceModelPMB<Elastic, AnisotropyType>;
    using base_fracture_type = BaseNoFractureModel;
    using base_temperature_type = BaseTemperatureModel<TemperatureIndependent>;

    using base_type::base_type;
    using base_type::delta;
    using base_type::operator();
    using base_fracture_type::operator();
    using base_temperature_type::operator();
};

template <typename AnisotropyType>
struct ForceModel<PMB, Elastic, AnisotropyType, Fracture,
                  TemperatureIndependent>
    : public BaseForceModelPMB<Elastic, AnisotropyType>,
      BaseFractureModel,
      BaseTemperatureModel<TemperatureIndependent>
{
    using base_type = BaseForceModelPMB<Elastic, AnisotropyType>;
    using base_fracture_type = BaseFractureModel;
    using base_temperature_type = BaseTemperatureModel<TemperatureIndependent>;

    using base_type::operator();
    using base_fracture_type::operator();
    using base_temperature_type::operator();

    ForceModel( PMB model, const double delta, const double K, const double G0,
                const int influence_type = 1 )
        : base_type( model, NoFracture{}, delta, K )
        , base_fracture_type( delta, K, G0, influence_type )
    {
    }

    ForceModel( PMB model, Elastic elastic, const double delta, const double K,
                const double G0, const int influence_type = 1 )
        : base_type( model, elastic, NoFracture{}, delta, K )
        , base_fracture_type( delta, K, G0, influence_type )
    {
    }

    ForceModel( PMB model, Fracture, const double delta, const double K,
                const double G0, const int influence_type = 1 )
        : base_type( model, Elastic{}, NoFracture{}, delta, K )
        , base_fracture_type( delta, K, G0, influence_type )
    {
    }

    ForceModel( PMB model, Elastic, Fracture, const double delta,
                const double K, const double G0, const int influence_type = 1 )
        : base_type( model, NoFracture{}, delta, K )
        , base_fracture_type( delta, K, G0, influence_type )
    {
    }

    // Constructor to work with plasticity.
    ForceModel( PMB model, Elastic elastic, const double delta, const double K,
                const double G0, const double s0 )
        : base_type( model, elastic, NoFracture{}, delta, K )
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

template <typename AnisotropyType, typename MemorySpace>
struct ForceModel<PMB, ElasticPerfectlyPlastic, AnisotropyType, Fracture,
                  TemperatureIndependent, MemorySpace>
    : public BaseForceModelPMB<ElasticPerfectlyPlastic, AnisotropyType,
                               MemorySpace>,
      public BaseFractureModel,
      public BaseTemperatureModel<TemperatureIndependent>
{
    using base_type =
        BaseForceModelPMB<ElasticPerfectlyPlastic, AnisotropyType, MemorySpace>;
    using base_fracture_type = BaseFractureModel;
    using base_temperature_type = BaseTemperatureModel<TemperatureIndependent>;

    using base_type::operator();
    using base_fracture_type::operator();
    using base_temperature_type::operator();
    using base_type::updateBonds;

    ForceModel( PMB model, ElasticPerfectlyPlastic mechanics, MemorySpace space,
                const double delta, const double K, const double G0,
                const double sigma_y )
        : base_type( model, mechanics, space, delta, K, sigma_y )
        , base_fracture_type( G0,
                              // s0
                              ( 5.0 * G0 / sigma_y / delta + sigma_y / K ) /
                                  6.0 )
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
ForceModel( ModelType, Elastic, NoFracture, const double delta, const double K )
    -> ForceModel<ModelType, Elastic, Isotropic, NoFracture>;

template <typename ModelType>
ForceModel( ModelType, Elastic, Fracture, const double delta, const double K,
            const double G0 ) -> ForceModel<ModelType, Elastic>;

// Default to fracture.
template <typename ModelType>
ForceModel( ModelType, Elastic, const double delta, const double K,
            const double G0 ) -> ForceModel<ModelType, Elastic>;

template <typename ModelType>
ForceModel( ModelType, NoFracture, const double delta, const double K )
    -> ForceModel<ModelType, Elastic, Isotropic, NoFracture>;

// Default to elastic.
template <typename ModelType>
ForceModel( ModelType, const double delta, const double K, const double G0 )
    -> ForceModel<ModelType>;

template <typename ModelType, typename MemorySpace>
ForceModel( ModelType, ElasticPerfectlyPlastic, MemorySpace, const double delta,
            const double K, const double G0, const double sigma_y )
    -> ForceModel<ModelType, ElasticPerfectlyPlastic, Isotropic, Fracture,
                  TemperatureIndependent, MemorySpace>;

template <typename AnisotropyType, typename TemperatureType>
struct ForceModel<PMB, Elastic, AnisotropyType, NoFracture,
                  TemperatureDependent, TemperatureType>
    : public BaseForceModelPMB<Elastic, AnisotropyType>,
      BaseNoFractureModel,
      BaseTemperatureModel<TemperatureDependent, TemperatureType>
{
    using base_type = BaseForceModelPMB<Elastic, AnisotropyType>;
    using base_temperature_type =
        BaseTemperatureModel<TemperatureDependent, TemperatureType>;

    using base_type::operator();
    using base_temperature_type::operator();

    ForceModel( PMB model, NoFracture fracture, const double _delta,
                const double _K, const TemperatureType _temp,
                const double _alpha, const double _temp0 = 0.0 )
        : base_type( model, fracture, _delta, _K )
        , base_temperature_type( _temp, _alpha, _temp0 )
    {
    }
};

template <typename AnisotropyType, typename TemperatureType>
struct ForceModel<PMB, Elastic, AnisotropyType, Fracture, TemperatureDependent,
                  TemperatureType>
    : public BaseForceModelPMB<Elastic, AnisotropyType>,
      ThermalFractureModel<TemperatureType>
{
    using base_type = BaseForceModelPMB<Elastic, AnisotropyType>;
    using base_temperature_type = ThermalFractureModel<TemperatureType>;

    using base_type::operator();
    using base_temperature_type::operator();

    ForceModel( PMB model, const double _delta, const double _K,
                const double _G0, const TemperatureType _temp,
                const double _alpha, const double _temp0 = 0.0 )
        : base_type( model, NoFracture{}, _delta, _K )
        , base_temperature_type( _delta, _K, _G0, _temp, _alpha, _temp0 )
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

template <typename AnisotropyType, typename TemperatureType>
struct ForceModel<PMB, ElasticPerfectlyPlastic, AnisotropyType, Fracture,
                  TemperatureDependent, TemperatureType>
    : public BaseForceModelPMB<ElasticPerfectlyPlastic, AnisotropyType,
                               typename TemperatureType::memory_space>,
      public ThermalFractureModel<TemperatureType>
{
    using base_type = BaseForceModelPMB<ElasticPerfectlyPlastic, AnisotropyType,
                                        typename TemperatureType::memory_space>;
    using base_temperature_type = ThermalFractureModel<TemperatureType>;

    using base_type::operator();
    using base_temperature_type::operator();

    ForceModel( PMB model, ElasticPerfectlyPlastic mechanics,
                const double _delta, const double _K, const double _G0,
                const double sigma_y, const TemperatureType _temp,
                const double _alpha, const double _temp0 = 0.0 )
        : base_type( model, mechanics, typename TemperatureType::memory_space{},
                     _delta, _K, sigma_y )
        , base_temperature_type( _delta, _K, _G0, _temp, _alpha, _temp0 )
    {
    }
};

template <typename ModelType, typename TemperatureType>
ForceModel( ModelType, NoFracture, const double delta, const double K,
            const TemperatureType& temp, const double alpha,
            const double temp0 = 0.0 )
    -> ForceModel<ModelType, Elastic, Isotropic, NoFracture,
                  TemperatureDependent, TemperatureType>;

template <typename ModelType, typename TemperatureType>
ForceModel( ModelType, const double delta, const double K, const double _G0,
            const TemperatureType& temp, const double alpha,
            const double temp0 = 0.0 )
    -> ForceModel<ModelType, Elastic, Isotropic, Fracture, TemperatureDependent,
                  TemperatureType>;

template <typename ModelType, typename TemperatureType>
ForceModel( ModelType, ElasticPerfectlyPlastic, const double delta,
            const double K, const double _G0, const double sigma_y,
            const TemperatureType& temp, const double alpha,
            const double temp0 = 0.0 )
    -> ForceModel<ModelType, ElasticPerfectlyPlastic, Isotropic, Fracture,
                  TemperatureDependent, TemperatureType>;

template <typename AnisotropyType, typename TemperatureType>
struct ForceModel<PMB, Elastic, AnisotropyType, NoFracture, DynamicTemperature,
                  TemperatureType>
    : public BaseForceModelPMB<Elastic, AnisotropyType>,
      BaseNoFractureModel,
      BaseTemperatureModel<TemperatureDependent, TemperatureType>,
      BaseDynamicTemperatureModel
{
    using base_type = BaseForceModelPMB<Elastic, AnisotropyType>;
    using base_temperature_type =
        BaseTemperatureModel<TemperatureDependent, TemperatureType>;
    using base_heat_transfer_type = BaseDynamicTemperatureModel;

    // Necessary to distinguish between TemperatureDependent
    using thermal_type = DynamicTemperature;

    using base_type::operator();
    using base_temperature_type::operator();

    ForceModel( PMB model, NoFracture fracture, const double _delta,
                const double _K, const TemperatureType& _temp,
                const double _kappa, const double _cp, const double _alpha,
                const double _temp0 = 0.0,
                const bool _constant_microconductivity = true )
        : base_type( model, fracture, _delta, _K )
        , base_temperature_type( _temp, _alpha, _temp0 )
        , base_heat_transfer_type( _delta, _kappa, _cp,
                                   _constant_microconductivity )
    {
    }
};

template <typename ModelType, typename TemperatureType>
ForceModel( ModelType, NoFracture, const double delta, const double K,
            const TemperatureType& temp, const double kappa, const double cp,
            const double alpha, const double temp0 = 0.0,
            const bool constant_microconductivity = true )
    -> ForceModel<ModelType, Elastic, Isotropic, NoFracture, DynamicTemperature,
                  TemperatureType>;

template <typename ModelType, typename AnisotropyType, typename TemperatureType>
struct ForceModel<ModelType, Elastic, AnisotropyType, Fracture,
                  DynamicTemperature, TemperatureType>
    : public BaseForceModelPMB<Elastic, AnisotropyType>,
      ThermalFractureModel<TemperatureType>,
      BaseDynamicTemperatureModel
{
    using base_type = BaseForceModelPMB<Elastic, AnisotropyType>;
    using base_temperature_type = ThermalFractureModel<TemperatureType>;
    using base_heat_transfer_type = BaseDynamicTemperatureModel;

    // Necessary to distinguish between TemperatureDependent
    using thermal_type = DynamicTemperature;

    using base_type::operator();
    using base_temperature_type::operator();

    ForceModel( PMB model, const double _delta, const double _K,
                const double _G0, const TemperatureType _temp,
                const double _kappa, const double _cp, const double _alpha,
                const double _temp0 = 0.0,
                const bool _constant_microconductivity = true )
        : base_type( model, NoFracture{}, _delta, _K )
        , base_temperature_type( _delta, _K, _G0, _temp, _alpha, _temp0 )
        , base_heat_transfer_type( _delta, _kappa, _cp,
                                   _constant_microconductivity )
    {
    }
};

template <typename ModelType, typename TemperatureType>
ForceModel( ModelType, const double delta, const double K, const double G0,
            const TemperatureType& temp, const double kappa, const double cp,
            const double alpha, const double temp0 = 0.0,
            const bool constant_microconductivity = true )
    -> ForceModel<ModelType, Elastic, Isotropic, Fracture, DynamicTemperature,
                  TemperatureType>;

template <typename AnisotropyType, typename TemperatureType>
struct ForceModel<PMB, ElasticPerfectlyPlastic, AnisotropyType, Fracture,
                  DynamicTemperature, TemperatureType>
    : public BaseForceModelPMB<ElasticPerfectlyPlastic, AnisotropyType,
                               typename TemperatureType::memory_space>,
      public ThermalFractureModel<TemperatureType>,
      BaseDynamicTemperatureModel
{
    using base_type = BaseForceModelPMB<ElasticPerfectlyPlastic, AnisotropyType,
                                        typename TemperatureType::memory_space>;
    using base_temperature_type = ThermalFractureModel<TemperatureType>;
    using base_heat_transfer_type = BaseDynamicTemperatureModel;

    // Necessary to distinguish between TemperatureDependent
    using thermal_type = DynamicTemperature;

    using base_type::operator();
    using base_temperature_type::operator();

    ForceModel( PMB model, ElasticPerfectlyPlastic mechanics,
                const double _delta, const double _K, const double _G0,
                const double sigma_y, const TemperatureType _temp,
                const double _kappa, const double _cp, const double _alpha,
                const double _temp0 = 0.0,
                const bool _constant_microconductivity = true )
        : base_type( model, mechanics, typename TemperatureType::memory_space{},
                     _delta, _K, sigma_y )
        , base_temperature_type( _delta, _K, _G0, _temp, _alpha, _temp0 )
        , base_heat_transfer_type( _delta, _kappa, _cp,
                                   _constant_microconductivity )
    {
    }
};

template <typename ModelType, typename TemperatureType>
ForceModel( ModelType, ElasticPerfectlyPlastic, const double delta,
            const double K, const double G0, const double sigma_y,
            const TemperatureType& temp, const double kappa, const double cp,
            const double alpha, const double temp0 = 0.0,
            const bool constant_microconductivity = true )
    -> ForceModel<ModelType, ElasticPerfectlyPlastic, Isotropic, Fracture,
                  DynamicTemperature, TemperatureType>;

/******************************************************************************
 Linear PMB.
******************************************************************************/
template <typename AnisotropyType>
struct ForceModel<LinearPMB, Elastic, AnisotropyType, NoFracture,
                  TemperatureIndependent>
    : public ForceModel<PMB, Elastic, AnisotropyType, NoFracture,
                        TemperatureIndependent>
{
    using base_type = ForceModel<PMB, Elastic, AnisotropyType, NoFracture,
                                 TemperatureIndependent>;
    using model_type = LinearPMB;

    using base_type::base_type;
    using base_type::operator();

    template <typename... Args>
    ForceModel( LinearPMB, Args... args )
        : base_type( typename base_type::base_model{}, args... )
    {
    }
};

template <typename AnisotropyType>
struct ForceModel<LinearPMB, Elastic, AnisotropyType, Fracture,
                  TemperatureIndependent>
    : public ForceModel<PMB, Elastic, AnisotropyType, Fracture,
                        TemperatureIndependent>
{
    using base_type = ForceModel<PMB, Elastic, AnisotropyType, Fracture,
                                 TemperatureIndependent>;

    using model_type = LinearPMB;

    using base_type::operator();

    template <typename... Args>
    ForceModel( LinearPMB, Args... args )
        : base_type( typename base_type::base_model{},
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

    using model_type = LinearPMB;

    using base_type::operator();

    template <typename... Args>
    ForceModel( LinearPMB, Args... args )
        : base_type( typename base_type::base_model{},
                     std::forward<Args>( args )... )
    {
    }
};

template <typename MechanicsType, typename AnisotropyType, typename ThermalType,
          typename... FieldTypes>
struct ForceModel<LinearPMB, MechanicsType, AnisotropyType, Fracture,
                  ThermalType, FieldTypes...>
    : public ForceModel<PMB, MechanicsType, AnisotropyType, Fracture,
                        ThermalType, FieldTypes...>
{
    using base_type = ForceModel<PMB, MechanicsType, AnisotropyType, Fracture,
                                 ThermalType, FieldTypes...>;

    using model_type = LinearPMB;

    using base_type::operator();

    template <typename... Args>
    ForceModel( LinearPMB, Args... args )
        : base_type( typename base_type::base_model{},
                     std::forward<Args>( args )... )
    {
    }
};

} // namespace CabanaPD

#endif
