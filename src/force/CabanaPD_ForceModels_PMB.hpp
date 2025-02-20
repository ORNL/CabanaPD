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

template <typename MechanicsModelType>
struct BaseForceModelPMB;

template <>
struct BaseForceModelPMB<Elastic> : public BaseForceModel
{
    using base_type = BaseForceModel;
    using base_model = PMB;
    using model_type = PMB;
    using fracture_type = NoFracture;

    double c;
    double K;

    BaseForceModelPMB( const double delta, const double _K )
        : base_type( delta )
        , K( _K )
    {
        c = 18.0 * K / ( pi * delta * delta * delta * delta );
    }

    // Average from existing models.
    template <typename ModelType1, typename ModelType2>
    BaseForceModelPMB( const ModelType1& model1, const ModelType2& model2 )
        : base_type( model1.delta )
    {
        K = ( model1.K + model2.K ) / 2.0;
        c = ( model1.c + model2.c ) / 2.0;
    }

    KOKKOS_INLINE_FUNCTION
    auto operator()( ForceCoeffTag, const int, const int, const double s,
                     const double vol ) const
    {
        return c * s * vol;
    }

    KOKKOS_INLINE_FUNCTION
    auto operator()( EnergyTag, const int, const int, const double s,
                     const double xi, const double vol ) const
    {
        // 0.25 factor is due to 1/2 from outside the integral and 1/2 from
        // the integrand (pairwise potential).
        return 0.25 * c * s * s * xi * vol;
    }
};

template <>
struct ForceModel<PMB, Elastic, NoFracture, TemperatureIndependent>
    : public BaseForceModelPMB<Elastic>,
      BaseTemperatureModel<TemperatureIndependent>
{
    using base_type = BaseForceModelPMB<Elastic>;
    using base_temperature_type = BaseTemperatureModel<TemperatureIndependent>;
    using fracture_type = NoFracture;
    using thermal_type = TemperatureIndependent;
    using material_type = typename base_type::material_type;

    using base_type::base_type;
    using base_type::delta;
    using base_type::operator();
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

    using model_type = typename base_type::model_type;
    using base_model = typename base_type::base_model;
    using fracture_type = Fracture;
    using thermal_type = base_temperature_type::thermal_type;

    using base_fracture_type::bond_break_coeff;
    using base_fracture_type::G0;
    using base_fracture_type::s0;
    using base_type::c;
    using base_type::delta;
    using base_type::K;

    using base_type::operator();
    using base_fracture_type::operator();
    using base_temperature_type::operator();

    ForceModel( const double _delta, const double _K, const double _G0 )
        : base_type( _delta, _K )
        , base_fracture_type( _delta, _K, _G0 )
    {
    }

    // Average from existing models.
    template <typename ModelType1, typename ModelType2>
    ForceModel( const ModelType1& model1, const ModelType2& model2 )
        : base_type( model1, model2 )
        , base_fracture_type( model1, model2 )
    {
    }
};

template <>
struct ForceModel<LinearPMB, Elastic, NoFracture, TemperatureIndependent>
    : public ForceModel<PMB, Elastic, NoFracture, TemperatureIndependent>
{
    using base_type =
        ForceModel<PMB, Elastic, NoFracture, TemperatureIndependent>;

    using model_type = LinearPMB;
    using base_model = typename base_type::base_model;
    using fracture_type = typename base_type::fracture_type;
    using thermal_type = typename base_temperature_type::thermal_type;

    using base_type::base_type;
    using base_type::operator();

    using base_type::c;
    using base_type::delta;
    using base_type::K;
};

template <>
struct ForceModel<LinearPMB, Elastic, Fracture, TemperatureIndependent>
    : public ForceModel<PMB, Elastic, Fracture, TemperatureIndependent>
{
    using base_type =
        ForceModel<PMB, Elastic, Fracture, TemperatureIndependent>;

    using model_type = LinearPMB;
    using base_model = typename base_type::base_model;
    using fracture_type = typename base_type::fracture_type;
    using thermal_type = typename base_type::thermal_type;

    using base_type::base_type;
    using base_type::operator();

    using base_type::c;
    using base_type::delta;
    using base_type::K;

    using base_type::bond_break_coeff;
    using base_type::G0;
    using base_type::s0;
};

template <typename TemperatureType>
struct ForceModel<PMB, Elastic, NoFracture, TemperatureDependent,
                  TemperatureType>
    : public BaseForceModelPMB<Elastic>,
      BaseTemperatureModel<TemperatureDependent, TemperatureType>
{
    using base_type = BaseForceModelPMB<Elastic>;
    using base_temperature_type =
        BaseTemperatureModel<TemperatureDependent, TemperatureType>;

    using model_type = PMB;
    using base_model = PMB;
    using fracture_type = NoFracture;
    using thermal_type = TemperatureDependent;

    using base_type::c;
    using base_type::delta;
    using base_type::K;

    // Thermal parameters
    using base_temperature_type::alpha;
    using base_temperature_type::temp0;

    using base_type::operator();
    using base_temperature_type::operator();

    ForceModel( const double _delta, const double _K,
                const TemperatureType _temp, const double _alpha,
                const double _temp0 = 0.0 )
        : base_type( _delta, _K )
        , base_temperature_type( _temp, _alpha, _temp0 )
    {
    }
};

// Default to Fracture.
template <typename ParticleType>
auto createForceModel( PMB model, ParticleType particles, const double delta,
                       const double K, const double alpha, const double temp0 )
{
    return createForceModel( model, Fracture{}, particles, delta, K, alpha,
                             temp0 );
}

template <typename ParticleType>
auto createForceModel( PMB, NoFracture, ParticleType particles,
                       const double delta, const double K, const double alpha,
                       const double temp0 )
{
    auto temp = particles.sliceTemperature();
    using temp_type = decltype( temp );
    return ForceModel<PMB, Elastic, NoFracture, TemperatureDependent,
                      temp_type>( delta, K, temp, alpha, temp0 );
}

template <typename TemperatureType>
struct ForceModel<PMB, Elastic, Fracture, TemperatureDependent, TemperatureType>
    : public BaseForceModelPMB<Elastic>, ThermalFractureModel<TemperatureType>
{
    using base_type = BaseForceModelPMB<Elastic>;
    using base_temperature_type = ThermalFractureModel<TemperatureType>;

    using model_type = PMB;
    using base_model = typename base_type::base_model;
    using fracture_type = Fracture;
    using thermal_type = TemperatureDependent;

    using base_type::c;
    using base_type::delta;
    using base_type::K;

    using base_temperature_type::G0;
    using base_temperature_type::s0;

    // Thermal parameters
    using base_temperature_type::alpha;
    using base_temperature_type::temp0;
    using base_temperature_type::temperature;

    using base_type::operator();
    using base_temperature_type::operator();

    ForceModel( const double _delta, const double _K, const double _G0,
                const TemperatureType _temp, const double _alpha,
                const double _temp0 = 0.0 )
        : base_type( _delta, _K )
        , base_temperature_type( _delta, _K, _G0, _temp, _alpha, _temp0 )
    {
    }

    // Average from existing models.
    ForceModel( const ForceModel& model1, const ForceModel& model2 )
        : base_type( model1, model2 )
        , base_temperature_type( model1, model2 )
    {
    }
};

template <typename ParticleType>
auto createForceModel( PMB, Fracture, ParticleType particles,
                       const double delta, const double K, const double G0,
                       const double alpha, const double temp0 )
{
    auto temp = particles.sliceTemperature();
    using temp_type = decltype( temp );
    return ForceModel<PMB, Elastic, Fracture, TemperatureDependent, temp_type>(
        delta, K, G0, temp, alpha, temp0 );
}

template <typename TemperatureType>
struct ForceModel<PMB, Elastic, NoFracture, DynamicTemperature, TemperatureType>
    : public BaseForceModelPMB<Elastic>,
      BaseTemperatureModel<TemperatureDependent, TemperatureType>,
      BaseDynamicTemperatureModel
{
    using base_type = BaseForceModelPMB<Elastic>;
    using base_temperature_type =
        BaseTemperatureModel<TemperatureDependent, TemperatureType>;
    using base_heat_transfer_type = BaseDynamicTemperatureModel;

    using model_type = PMB;
    using base_model = PMB;
    using fracture_type = NoFracture;
    using thermal_type = DynamicTemperature;

    using base_type::c;
    using base_type::delta;
    using base_type::K;

    // Thermal parameters
    using base_heat_transfer_type::cp;
    using base_heat_transfer_type::kappa;
    using base_heat_transfer_type::thermal_coeff;
    using base_temperature_type::alpha;
    using base_temperature_type::temp0;
    using base_temperature_type::temperature;

    using base_type::operator();
    using base_temperature_type::operator();

    ForceModel( const double _delta, const double _K,
                const TemperatureType _temp, const double _kappa,
                const double _cp, const double _alpha,
                const double _temp0 = 0.0,
                const bool _constant_microconductivity = true )
        : base_type( _delta, _K )
        , base_temperature_type( _temp, _alpha, _temp0 )
        , base_heat_transfer_type( _delta, _kappa, _cp,
                                   _constant_microconductivity )
    {
    }
};

template <typename ParticleType>
auto createForceModel( PMB, NoFracture, ParticleType particles,
                       const double delta, const double K, const double kappa,
                       const double cp, const double alpha, const double temp0,
                       const bool constant_microconductivity = true )
{
    auto temp = particles.sliceTemperature();
    using temp_type = decltype( temp );
    return ForceModel<PMB, Elastic, NoFracture, DynamicTemperature, temp_type>(
        delta, K, temp, kappa, cp, alpha, temp0, constant_microconductivity );
}

template <typename TemperatureType>
struct ForceModel<PMB, Elastic, Fracture, DynamicTemperature, TemperatureType>
    : public BaseForceModelPMB<Elastic>,
      ThermalFractureModel<TemperatureType>,
      BaseDynamicTemperatureModel
{
    using base_type = BaseForceModelPMB<Elastic>;
    using base_temperature_type = ThermalFractureModel<TemperatureType>;
    using base_heat_transfer_type = BaseDynamicTemperatureModel;

    using model_type = PMB;
    using base_model = typename base_type::base_model;
    using fracture_type = Fracture;
    using thermal_type = DynamicTemperature;

    using base_type::c;
    using base_type::delta;
    using base_type::K;

    using base_temperature_type::G0;
    using base_temperature_type::s0;

    // Thermal parameters
    using base_heat_transfer_type::cp;
    using base_heat_transfer_type::kappa;
    using base_heat_transfer_type::thermal_coeff;
    using base_temperature_type::alpha;
    using base_temperature_type::temp0;
    using base_temperature_type::temperature;

    using base_type::operator();
    using base_temperature_type::operator();

    ForceModel( const double _delta, const double _K, const double _G0,
                const TemperatureType _temp, const double _kappa,
                const double _cp, const double _alpha,
                const double _temp0 = 0.0,
                const bool _constant_microconductivity = true )
        : base_type( _delta, _K )
        , base_temperature_type( _delta, _K, _G0, _temp, _alpha, _temp0 )
        , base_heat_transfer_type( _delta, _kappa, _cp,
                                   _constant_microconductivity )
    {
    }
};

template <typename ParticleType>
auto createForceModel( PMB, Fracture, ParticleType particles,
                       const double delta, const double K, const double G0,
                       const double kappa, const double cp, const double alpha,
                       const double temp0,
                       const bool constant_microconductivity = true )
{
    auto temp = particles.sliceTemperature();
    using temp_type = decltype( temp );
    return ForceModel<PMB, Elastic, Fracture, DynamicTemperature, temp_type>(
        delta, K, G0, temp, kappa, cp, alpha, temp0,
        constant_microconductivity );
}

} // namespace CabanaPD

#endif
