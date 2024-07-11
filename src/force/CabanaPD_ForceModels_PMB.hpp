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
template <>
struct ForceModel<PMB, Elastic, NoFracture, TemperatureIndependent>
    : public BaseForceModel
{
    using base_type = BaseForceModel;
    using base_model = PMB;
    using fracture_type = NoFracture;
    using thermal_type = TemperatureIndependent;

    using base_type::delta;

    double c;
    double K;

    ForceModel( const double delta, const double _K )
        : base_type( delta )
        , K( _K )
    {
        c = 18.0 * K / ( pi * delta * delta * delta * delta );
    }
};

template <>
struct ForceModel<PMB, Elastic, Fracture, TemperatureIndependent>
    : public ForceModel<PMB, Elastic, NoFracture, TemperatureIndependent>
{
    using base_type = ForceModel<PMB, Elastic, NoFracture>;
    using base_model = typename base_type::base_model;
    using fracture_type = Fracture;
    using thermal_type = base_type::thermal_type;

    using base_type::c;
    using base_type::delta;
    using base_type::K;
    double G0;
    double s0;
    double bond_break_coeff;

    ForceModel( const double delta, const double K, const double _G0 )
        : base_type( delta, K )
        , G0( _G0 )
    {
        s0 = Kokkos::sqrt( 5.0 * G0 / 9.0 / K / delta );
        bond_break_coeff = ( 1.0 + s0 ) * ( 1.0 + s0 );
    }

    KOKKOS_INLINE_FUNCTION
    bool criticalStretch( const int, const int, const double r,
                          const double xi ) const
    {
        return r * r >= bond_break_coeff * xi * xi;
    }
};

template <>
struct ForceModel<LinearPMB, Elastic, NoFracture, TemperatureIndependent>
    : public ForceModel<PMB, Elastic, NoFracture, TemperatureIndependent>
{
    using base_type = ForceModel<PMB, Elastic, NoFracture>;
    using base_model = typename base_type::base_model;
    using fracture_type = typename base_type::fracture_type;
    using thermal_type = base_type::thermal_type;

    using base_type::base_type;

    using base_type::c;
    using base_type::delta;
    using base_type::K;
};

template <>
struct ForceModel<LinearPMB, Elastic, Fracture, TemperatureIndependent>
    : public ForceModel<PMB, Elastic, Fracture, TemperatureIndependent>
{
    using base_type = ForceModel<PMB>;
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
struct ForceModel<PMB, Elastic, NoFracture, TemperatureDependent,
                  TemperatureType>
    : public ForceModel<PMB, Elastic, NoFracture, TemperatureIndependent>,
      BaseTemperatureModel<TemperatureType>
{
    using base_type =
        ForceModel<PMB, Elastic, NoFracture, TemperatureIndependent>;
    using base_temperature_type = BaseTemperatureModel<TemperatureType>;
    using base_model = PMB;
    using fracture_type = NoFracture;
    using thermal_type = TemperatureDependent;

    using base_type::c;
    using base_type::delta;
    using base_type::K;

    // Thermal parameters
    using base_temperature_type::alpha;
    using base_temperature_type::temp0;

    // Explicitly use the temperature-dependent stretch.
    using base_temperature_type::thermalStretch;

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
    : public ForceModel<PMB, Elastic, Fracture, TemperatureIndependent>,
      BaseTemperatureModel<TemperatureType>
{
    using base_type =
        ForceModel<PMB, Elastic, Fracture, TemperatureIndependent>;
    using base_temperature_type = BaseTemperatureModel<TemperatureType>;
    using base_model = typename base_type::base_model;
    using fracture_type = typename base_type::fracture_type;
    using thermal_type = TemperatureDependent;

    using base_type::c;
    using base_type::delta;
    using base_type::K;

    // Does not use the base bond_break_coeff.
    using base_type::G0;
    using base_type::s0;

    // Thermal parameters
    using base_temperature_type::alpha;
    using base_temperature_type::temp0;
    using base_temperature_type::temperature;

    // Explicitly use the temperature-dependent stretch.
    using base_temperature_type::thermalStretch;

    ForceModel( const double _delta, const double _K, const double _G0,
                const TemperatureType _temp, const double _alpha,
                const double _temp0 = 0.0 )
        : base_type( _delta, _K, _G0 )
        , base_temperature_type( _temp, _alpha, _temp0 )
    {
    }

    KOKKOS_INLINE_FUNCTION
    bool criticalStretch( const int i, const int j, const double r,
                          const double xi ) const
    {
        double temp_avg = 0.5 * ( temperature( i ) + temperature( j ) ) - temp0;
        double bond_break_coeff =
            ( 1.0 + s0 + alpha * temp_avg ) * ( 1.0 + s0 + alpha * temp_avg );
        return r * r >= bond_break_coeff * xi * xi;
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
    : public ForceModel<PMB, Elastic, NoFracture, TemperatureDependent,
                        TemperatureType>,
      BaseDynamicTemperatureModel
{
    using base_type = ForceModel<PMB, Elastic, NoFracture, TemperatureDependent,
                                 TemperatureType>;
    using base_temperature_type = BaseDynamicTemperatureModel;
    using base_model = PMB;
    using fracture_type = NoFracture;
    using thermal_type = DynamicTemperature;

    using base_type::c;
    using base_type::delta;
    using base_type::K;

    // Thermal parameters
    using base_temperature_type::cp;
    using base_temperature_type::kappa;
    using base_temperature_type::thermal_coeff;
    using base_type::alpha;
    using base_type::temp0;
    using base_type::temperature;

    // Explicitly use the temperature-dependent stretch.
    using base_type::thermalStretch;

    ForceModel( const double _delta, const double _K,
                const TemperatureType _temp, const double _kappa,
                const double _cp, const double _alpha,
                const double _temp0 = 0.0,
                const bool _constant_microconductivity = true )
        : base_type( _delta, _K, _temp, _alpha, _temp0 )
        , base_temperature_type( _delta, _kappa, _cp,
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
struct ForceModel<PMB, Fracture, DynamicTemperature, TemperatureType>
    : public ForceModel<PMB, Fracture, TemperatureDependent, TemperatureType>,
      BaseDynamicTemperatureModel
{
    using base_type =
        ForceModel<PMB, Fracture, TemperatureDependent, TemperatureType>;
    using base_temperature_type = BaseDynamicTemperatureModel;
    using base_model = typename base_type::base_model;
    using fracture_type = typename base_type::fracture_type;
    using thermal_type = DynamicTemperature;

    using base_type::c;
    using base_type::delta;
    using base_type::K;

    // Does not use the base bond_break_coeff.
    using base_type::G0;
    using base_type::s0;

    // Thermal parameters
    using base_temperature_type::cp;
    using base_temperature_type::kappa;
    using base_temperature_type::thermal_coeff;
    using base_type::alpha;
    using base_type::temp0;
    using base_type::temperature;

    // Explicitly use the temperature-dependent stretch.
    using base_type::thermalStretch;

    ForceModel( const double _delta, const double _K, const double _G0,
                const TemperatureType _temp, const double _kappa,
                const double _cp, const double _alpha,
                const double _temp0 = 0.0,
                const bool _constant_microconductivity = true )
        : base_type( _delta, _K, _G0, _temp, _alpha, _temp0 )
        , base_temperature_type( _delta, _kappa, _cp,
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
