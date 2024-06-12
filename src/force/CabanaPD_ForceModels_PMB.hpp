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

#ifndef FORCE_MODELS_PMB_H
#define FORCE_MODELS_PMB_H

#include <CabanaPD_ForceModels.hpp>
#include <CabanaPD_Types.hpp>

namespace CabanaPD
{
template <>
struct ForceModel<PMB, Elastic, TemperatureIndependent> : public BaseForceModel
{
    using base_type = BaseForceModel;
    using base_model = PMB;
    using fracture_type = Elastic;
    using thermal_type = TemperatureIndependent;

    using base_type::delta;

    double c;
    double K;

    ForceModel(){};
    ForceModel( const double delta, const double K )
        : base_type( delta )
    {
        set_param( delta, K );
    }

    ForceModel( const ForceModel& model )
        : base_type( model )
    {
        c = model.c;
        K = model.K;
    }

    void set_param( const double _delta, const double _K )
    {
        delta = _delta;
        K = _K;
        c = 18.0 * K / ( 3.141592653589793 * delta * delta * delta * delta );
    }
};

template <>
struct ForceModel<PMB, Fracture, TemperatureIndependent>
    : public ForceModel<PMB, Elastic, TemperatureIndependent>
{
    using base_type = ForceModel<PMB, Elastic>;
    using base_model = typename base_type::base_model;
    using fracture_type = Fracture;
    using thermal_type = base_type::thermal_type;

    using base_type::c;
    using base_type::delta;
    using base_type::K;
    double G0;
    double s0;
    double bond_break_coeff;

    ForceModel() {}
    ForceModel( const double delta, const double K, const double G0 )
        : base_type( delta, K )
    {
        set_param( delta, K, G0 );
    }

    ForceModel( const ForceModel& model )
        : base_type( model )
    {
        G0 = model.G0;
        s0 = model.s0;
        bond_break_coeff = model.bond_break_coeff;
    }

    void set_param( const double _delta, const double _K, const double _G0 )
    {
        base_type::set_param( _delta, _K );
        G0 = _G0;
        s0 = sqrt( 5.0 * G0 / 9.0 / K / delta );
        bond_break_coeff = ( 1.0 + s0 ) * ( 1.0 + s0 );
    }
};

template <>
struct ForceModel<LinearPMB, Elastic, TemperatureIndependent>
    : public ForceModel<PMB, Elastic, TemperatureIndependent>
{
    using base_type = ForceModel<PMB, Elastic>;
    using base_model = typename base_type::base_model;
    using fracture_type = typename base_type::fracture_type;
    using thermal_type = base_type::thermal_type;

    using base_type::base_type;

    using base_type::c;
    using base_type::delta;
    using base_type::K;
};

template <>
struct ForceModel<LinearPMB, Fracture, TemperatureIndependent>
    : public ForceModel<PMB, Fracture, TemperatureIndependent>
{
    using base_type = ForceModel<PMB, Fracture>;
    using base_model = typename base_type::base_model;
    using fracture_type = typename base_type::fracture_type;
    using thermal_type = base_type::thermal_type;

    using base_type::base_type;

    using base_type::c;
    using base_type::delta;
    using base_type::K;

    using base_type::bond_break_coeff;
    using base_type::G0;
    using base_type::s0;
};

template <typename TemperatureType>
struct ForceModel<PMB, Elastic, TemperatureDependent, TemperatureType>
    : public ForceModel<PMB, Elastic, TemperatureIndependent>,
      BaseTemperatureModel<TemperatureType>
{
    using base_type = ForceModel<PMB, Elastic, TemperatureIndependent>;
    using base_temperature_type = BaseTemperatureModel<TemperatureType>;
    using base_model = PMB;
    using fracture_type = Elastic;
    using thermal_type = TemperatureDependent;

    using base_type::c;
    using base_type::delta;
    using base_type::K;

    // Thermal parameters
    using base_temperature_type::alpha;
    using base_temperature_type::temp0;

    // Explicitly use the temperature-dependent stretch.
    using base_temperature_type::thermalStretch;

    // ForceModel(){};
    ForceModel( const double _delta, const double _K,
                const TemperatureType _temp, const double _alpha,
                const double _temp0 = 0.0 )
        : base_type( _delta, _K )
        , base_temperature_type( _temp, _alpha, _temp0 )
    {
        set_param( _delta, _K, _alpha, _temp0 );
    }

    void set_param( const double _delta, const double _K, const double _alpha,
                    const double _temp0 )
    {
        base_type::set_param( _delta, _K );
        base_temperature_type::set_param( _alpha, _temp0 );
    }
};

template <typename ModelType, typename DamageType, typename ThermalType,
          typename ParticleType>
auto createForceModel( ParticleType particles, const double delta,
                       const double K, const double alpha, const double temp0 )
{
    auto temp = particles.sliceTemperature();
    using temp_type = decltype( temp );
    return ForceModel<ModelType, DamageType, ThermalType, temp_type>(
        delta, K, temp, alpha, temp0 );
}
} // namespace CabanaPD

#endif
