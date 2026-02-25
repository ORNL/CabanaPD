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
struct MechanicsModel<PMB, Elastic> : public BaseForceModel
{
    using base_type = BaseForceModel;
    using mechanics_type = Elastic;

    // Tags for creating particle fields and dispatch to force iteration.
    using model_tag = PMB;
    using force_tag = PMB;

    using base_type::force_horizon;
    using base_type::K;
    double c;

    MechanicsModel( PMB, const double force_horizon, const double _K )
        : base_type( force_horizon, _K )
    {
        c = 18.0 * K /
            ( pi * force_horizon * force_horizon * force_horizon *
              force_horizon );
    }

    // Constructor to average from existing models.
    template <typename ModelType1, typename ModelType2>
    MechanicsModel(
        const ModelType1& model1, const ModelType2& model2,
        typename std::enable_if_t<is_mechanics_model<ModelType1>::value &&
                                  is_mechanics_model<ModelType2>::value> = 0 )
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
};

template <typename ModelType>
MechanicsModel( ModelType, const double force_horizon, const double K )
    -> MechanicsModel<ModelType, Elastic>;

template <typename MemorySpace>
struct MechanicsModel<PMB, ElasticPerfectlyPlastic, MemorySpace>
    : public MechanicsModel<PMB, Elastic>, public BasePlasticity<MemorySpace>
{
    using base_type = MechanicsModel<PMB, Elastic>;
    using base_plasticity_type = BasePlasticity<MemorySpace>;

    using mechanics_type = ElasticPerfectlyPlastic;

    using base_plasticity_type::_s_p;
    using base_type::c;
    double s_Y;

    using base_plasticity_type::updateBonds;

    MechanicsModel( PMB tag, ElasticPerfectlyPlastic, MemorySpace,
                    const double force_horizon, const double K,
                    const double sigma_y )
        : base_type( tag, force_horizon, K )
        , base_plasticity_type()
        , s_Y( sigma_y / 3.0 / K )
    {
    }

    // Constructor to average from existing models.
    template <typename ModelType1, typename ModelType2>
    MechanicsModel( const ModelType1& model1, const ModelType2& model2 )
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
struct MechanicsModel<LinearPMB, Elastic> : public MechanicsModel<LPS, Elastic>
{
    using base_type = MechanicsModel<LPS, Elastic>;
    // Tag to dispatch to force iteration.
    using force_tag = LinearPMB;

    using base_type::base_type;
    using base_type::operator();

    template <typename... Args>
    MechanicsModel( LinearPMB, Args&&... args )
        : base_type( typename base_type::model_tag{},
                     std::forward<Args>( args )... )
    {
    }
};

template <typename ModelType, typename MemorySpace>
MechanicsModel( ModelType, ElasticPerfectlyPlastic, MemorySpace,
                const double force_horizon, const double K,
                const double sigma_y )
    -> MechanicsModel<ModelType, ElasticPerfectlyPlastic, MemorySpace>;

// Backwards compatibility wrappers.
template <>
struct ForceModel<PMB, Elastic, NoFracture>
    : public Experimental::ForceModel<MechanicsModel<PMB, Elastic>,
                                      FractureModel<NoFracture>>
{
    using base_type = Experimental::ForceModel<MechanicsModel<PMB, Elastic>,
                                               FractureModel<NoFracture>>;
    using base_type::operator();

    ForceModel( PMB tag, NoFracture, const double force_horizon,
                const double K )
        : base_type( MechanicsModel( tag, force_horizon, K ), FractureModel() )
    {
    }

    ForceModel( PMB tag, Elastic, NoFracture fracture,
                const double force_horizon, const double K )
        : ForceModel( tag, fracture, force_horizon, K )
    {
    }

    template <typename ModelType1, typename ModelType2>
    ForceModel( const ModelType1& model1, const ModelType2& model2 )
        : base_type( model1, model2 )
    {
    }
};

template <>
struct ForceModel<PMB, Elastic, Fracture>
    : public Experimental::ForceModel<MechanicsModel<PMB, Elastic>,
                                      FractureModel<CriticalStretch>>
{
    using base_type = Experimental::ForceModel<MechanicsModel<PMB, Elastic>,
                                               FractureModel<CriticalStretch>>;
    using base_type::operator();

    ForceModel( PMB tag, const double force_horizon, const double K,
                const double G0, const int influence_type = 1 )
        : base_type( MechanicsModel( tag, force_horizon, K ),
                     FractureModel( force_horizon, K, G0, influence_type ) )
    {
    }

    ForceModel( PMB model, Elastic, const double force_horizon, const double K,
                const double G0, const int influence_type = 1 )
        : ForceModel( model, force_horizon, K, G0, influence_type )
    {
    }

    ForceModel( PMB model, Fracture, const double force_horizon, const double K,
                const double G0, const int influence_type = 1 )
        : ForceModel( model, force_horizon, K, G0, influence_type )
    {
    }

    ForceModel( PMB model, Elastic, Fracture, const double force_horizon,
                const double K, const double G0, const int influence_type = 1 )
        : ForceModel( model, force_horizon, K, G0, influence_type )
    {
    }

    // Constructor to work with plasticity.
    ForceModel( PMB tag, Elastic, const double force_horizon, const double K,
                const double G0, const double s0 )
        : base_type( MechanicsModel( tag, force_horizon, K ),
                     FractureModel( G0, s0 ) )
    {
    }

    template <typename ModelType1, typename ModelType2>
    ForceModel( const ModelType1& model1, const ModelType2& model2 )
        : base_type( model1, model2 )
    {
    }
};

template <typename MemorySpace>
struct ForceModel<PMB, ElasticPerfectlyPlastic, Fracture, MemorySpace>
    : public Experimental::ForceModel<
          MechanicsModel<PMB, ElasticPerfectlyPlastic, MemorySpace>,
          FractureModel<CriticalStretch>>
{
    using mechanics_type =
        MechanicsModel<PMB, ElasticPerfectlyPlastic, MemorySpace>;
    using fracture_type = FractureModel<CriticalStretch>;
    using base_type = Experimental::ForceModel<mechanics_type, fracture_type>;
    using base_type::operator();

    ForceModel( PMB tag, ElasticPerfectlyPlastic epp, MemorySpace space,
                const double force_horizon, const double K, const double G0,
                const double sigma_y )
        : base_type(
              mechanics_type( tag, epp, space, force_horizon, K, sigma_y ),
              fracture_type(
                  G0, // G0, s0
                  ( 5.0 * G0 / sigma_y / force_horizon + sigma_y / K ) / 6.0 ) )
    {
    }

    ForceModel( PMB model, ElasticPerfectlyPlastic mechanics, Fracture,
                MemorySpace space, const double force_horizon, const double K,
                const double G0, const double sigma_y )
        : ForceModel( model, mechanics, space, force_horizon, K, G0, sigma_y )
    {
    }

    template <typename ModelType1, typename ModelType2>
    ForceModel( const ModelType1& model1, const ModelType2& model2 )
        : base_type( model1, model2 )
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
    -> ForceModel<ModelType, ElasticPerfectlyPlastic, Fracture, MemorySpace>;

/******************************************************************************
 Temperature-dependent backwards compatibility wrappers.
******************************************************************************/
template <typename TemperatureType>
struct ForceModel<PMB, Elastic, NoFracture, TemperatureDependent,
                  TemperatureType>
    : public Experimental::ThermalForceModel<
          MechanicsModel<PMB, Elastic>,
          ThermalModel<TemperatureDependent, TemperatureType, ConstantProperty>>
{
    using mechanics_type = MechanicsModel<PMB, Elastic>;
    using thermal_type =
        ThermalModel<TemperatureDependent, TemperatureType, ConstantProperty>;
    using base_type =
        Experimental::ThermalForceModel<mechanics_type, thermal_type>;
    using typename thermal_type::thermal_tag;

    ForceModel( PMB tag, NoFracture, const double _force_horizon,
                const double _K, const TemperatureType _temp,
                const double _alpha, const double _temp0 = 0.0 )
        : base_type( mechanics_type( tag, _force_horizon, _K ),
                     thermal_type( _temp, _alpha, _temp0 ) )
    {
    }

    template <typename ModelType1, typename ModelType2>
    ForceModel( const ModelType1& model1, const ModelType2& model2 )
        : base_type( model1, model2 )
    {
    }
};

template <typename TemperatureType>
struct ForceModel<PMB, Elastic, Fracture, TemperatureDependent, TemperatureType>
    : public Experimental::ThermalForceModel<
          MechanicsModel<PMB, Elastic>,
          ThermalModel<TemperatureDependent, TemperatureType, ConstantProperty,
                       CriticalStretch>>
{
    using mechanics_type = MechanicsModel<PMB, Elastic>;
    using thermal_type = ThermalModel<TemperatureDependent, TemperatureType,
                                      ConstantProperty, CriticalStretch>;
    using fracture_type = FractureModel<CriticalStretch>;
    using base_type =
        Experimental::ThermalForceModel<mechanics_type, thermal_type>;

    ForceModel( PMB tag, const double _horizon, const double _K,
                const double _G0, const TemperatureType _temp,
                const double _alpha, const double _temp0 = 0.0 )
        : base_type( mechanics_type( tag, _horizon, _K ),
                     thermal_type( fracture_type( _horizon, _K, _G0 ), _temp,
                                   _alpha, _temp0 ) )
    {
    }

    template <typename ModelType1, typename ModelType2>
    ForceModel( const ModelType1& model1, const ModelType2& model2 )
        : base_type( model1, model2 )
    {
    }
};

template <typename TemperatureType>
struct ForceModel<PMB, ElasticPerfectlyPlastic, Fracture, TemperatureDependent,
                  TemperatureType>
    : public Experimental::ThermalForceModel<
          MechanicsModel<PMB, ElasticPerfectlyPlastic,
                         typename TemperatureType::memory_space>,
          ThermalModel<TemperatureDependent, TemperatureType, ConstantProperty,
                       CriticalStretch>>
{
    using mechanics_type =
        MechanicsModel<PMB, ElasticPerfectlyPlastic,
                       typename TemperatureType::memory_space>;
    using thermal_type = ThermalModel<TemperatureDependent, TemperatureType,
                                      ConstantProperty, CriticalStretch>;
    using base_type =
        Experimental::ThermalForceModel<mechanics_type, thermal_type>;

    ForceModel( PMB tag, ElasticPerfectlyPlastic epp, const double _horizon,
                const double _K, const double _G0, const double sigma_y,
                const TemperatureType _temp, const double _alpha,
                const double _temp0 = 0.0 )
        : base_type( mechanics_type( tag, epp,
                                     typename TemperatureType::memory_space{},
                                     _horizon, _K, sigma_y ),
                     thermal_type( FractureModel( _horizon, _K, _G0 ), _temp,
                                   _alpha, _temp0 ) )
    {
    }

    template <typename ModelType1, typename ModelType2>
    ForceModel( const ModelType1& model1, const ModelType2& model2 )
        : base_type( model1, model2 )
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
    : public Experimental::ThermalForceModel<
          MechanicsModel<PMB, Elastic>,
          ThermalModel<DynamicTemperature, TemperatureType, ConstantProperty,
                       NoFracture>>
{
    using mechanics_type = MechanicsModel<PMB, Elastic>;
    using thermal_type = ThermalModel<DynamicTemperature, TemperatureType,
                                      ConstantProperty, NoFracture>;
    using base_type =
        Experimental::ThermalForceModel<mechanics_type, thermal_type>;

    ForceModel( PMB tag, NoFracture, const double _horizon, const double _K,
                const TemperatureType& _temp, const double _kappa,
                const double _cp, const double _alpha,
                const double _temp0 = 0.0,
                const bool _constant_microconductivity = true )
        : base_type( mechanics_type( tag, _horizon, _K ),
                     thermal_type( _temp, _kappa, _cp, _alpha, _temp0,
                                   _constant_microconductivity ) )
    {
    }

    template <typename ModelType1, typename ModelType2>
    ForceModel( const ModelType1& model1, const ModelType2& model2 )
        : base_type( model1, model2 )
    {
    }
};

template <typename ModelType, typename TemperatureType>
struct ForceModel<ModelType, Elastic, Fracture, DynamicTemperature,
                  TemperatureType>
    : public Experimental::ThermalForceModel<
          MechanicsModel<PMB, Elastic>,
          ThermalModel<DynamicTemperature, TemperatureType, ConstantProperty,
                       CriticalStretch>>
{
    using mechanics_type = MechanicsModel<PMB, Elastic>;
    using thermal_type = ThermalModel<DynamicTemperature, TemperatureType,
                                      ConstantProperty, CriticalStretch>;
    using base_type =
        Experimental::ThermalForceModel<mechanics_type, thermal_type>;

    ForceModel( PMB tag, const double _horizon, const double _K,
                const double _G0, const TemperatureType _temp,
                const double _kappa, const double _cp, const double _alpha,
                const double _temp0 = 0.0,
                const bool _constant_microconductivity = true )
        : base_type( mechanics_type( tag, _horizon, _K ),
                     thermal_type( FractureModel( _horizon, _K, _G0 ), _temp,
                                   _kappa, _cp, _alpha, _temp0,
                                   _constant_microconductivity ) )
    {
    }

    template <typename ModelType1, typename ModelType2>
    ForceModel( const ModelType1& model1, const ModelType2& model2 )
        : base_type( model1, model2 )
    {
    }
};

template <typename TemperatureType>
struct ForceModel<PMB, ElasticPerfectlyPlastic, Fracture, DynamicTemperature,
                  TemperatureType>
    : public Experimental::ThermalForceModel<
          MechanicsModel<PMB, ElasticPerfectlyPlastic,
                         typename TemperatureType::memory_space>,
          ThermalModel<DynamicTemperature, TemperatureType, ConstantProperty,
                       CriticalStretch>>

{
    using mechanics_type =
        MechanicsModel<PMB, ElasticPerfectlyPlastic,
                       typename TemperatureType::memory_space>;
    using thermal_type = ThermalModel<DynamicTemperature, TemperatureType,
                                      ConstantProperty, CriticalStretch>;
    using base_type =
        Experimental::ThermalForceModel<mechanics_type, thermal_type>;

    ForceModel( PMB tag, ElasticPerfectlyPlastic epp, const double _horizon,
                const double _K, const double _G0, const double sigma_y,
                const TemperatureType _temp, const double _kappa,
                const double _cp, const double _alpha,
                const double _temp0 = 0.0,
                const bool _constant_microconductivity = true )
        : base_type( mechanics_type( tag, epp,
                                     typename TemperatureType::memory_space{},
                                     _horizon, _K, sigma_y ),
                     thermal_type( FractureModel( _horizon, _K, _G0 ), _temp,
                                   _kappa, _cp, _alpha, _temp0,
                                   _constant_microconductivity ) )
    {
    }

    template <typename ModelType1, typename ModelType2>
    ForceModel( const ModelType1& model1, const ModelType2& model2 )
        : base_type( model1, model2 )
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
ForceModel( ModelType, const double horizon, const double K, const double G0,
            const TemperatureType& temp, const double kappa, const double cp,
            const double alpha, const double temp0 = 0.0,
            const bool constant_microconductivity = true )
    -> ForceModel<ModelType, Elastic, Fracture, DynamicTemperature,
                  TemperatureType>;

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
template <typename MechanicsType, typename FractureType, typename ThermalType,
          typename... FieldTypes>
struct ForceModel<LinearPMB, MechanicsType, FractureType, ThermalType,
                  FieldTypes...>
    : public ForceModel<PMB, MechanicsType, FractureType, ThermalType,
                        FieldTypes...>
{
    using base_type = ForceModel<PMB, MechanicsType, FractureType, ThermalType,
                                 FieldTypes...>;
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

} // namespace CabanaPD

#endif
