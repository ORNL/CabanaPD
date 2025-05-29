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
{
    using model_type = PMB;
    using base_model = PMB;
    using fracture_type = NoFracture;
    using thermal_type = TemperatureIndependent;
    using density_type = StaticDensity;

    double delta;
    double c;
    double K;

    ForceModel( PMB, NoFracture, const double _delta, const double _K )
        : delta( _delta )
        , K( _K )
    {
        init();
    }

    ForceModel( PMB, Elastic, NoFracture, const double _delta, const double _K )
        : delta( _delta )
        , K( _K )
    {
        init();
    }

    void init() { c = 18.0 * K / ( pi * delta * delta * delta * delta ); }

    auto cutoff() const { return delta; }

    // Only needed for models which store bond properties.
    void updateBonds( const int, const int ) {}

    // No-op for temperature.
    KOKKOS_INLINE_FUNCTION
    void thermalStretch( double&, const int, const int ) const {}

    KOKKOS_INLINE_FUNCTION
    auto forceCoeff( const int, const int, const double s,
                     const double vol ) const
    {
        return c * s * vol;
    }

    KOKKOS_INLINE_FUNCTION
    auto energy( const int, const int, const double s, const double xi,
                 const double vol ) const
    {
        // 0.25 factor is due to 1/2 from outside the integral and 1/2 from
        // the integrand (pairwise potential).
        return 0.25 * c * s * s * xi * vol;
    }
};

template <>
struct ForceModel<PMB, Elastic, Fracture, TemperatureIndependent>
    : public ForceModel<PMB, Elastic, NoFracture, TemperatureIndependent>
{
    using base_type = ForceModel<PMB, Elastic, NoFracture>;
    using base_type::model_type;
    using base_model = typename base_type::base_model;
    using fracture_type = Fracture;
    using mechanics_type = Elastic;
    using thermal_type = base_type::thermal_type;

    using base_type::c;
    using base_type::delta;
    using base_type::K;
    double G0;
    double s0;
    double bond_break_coeff;

    ForceModel( PMB model, const double delta, const double K,
                const double _G0 )
        : base_type( model, NoFracture{}, delta, K )
        , G0( _G0 )
    {
        init();
    }

    ForceModel( PMB model, Elastic elastic, const double delta, const double K,
                const double _G0 )
        : base_type( model, elastic, NoFracture{}, delta, K )
        , G0( _G0 )
    {
        init();
    }

    void init()
    {
        s0 = Kokkos::sqrt( 5.0 * G0 / 9.0 / K / delta );
        bond_break_coeff = ( 1.0 + s0 ) * ( 1.0 + s0 );
    }

    // Constructor to work with plasticity.
    ForceModel( PMB model, Elastic elastic, const double delta, const double K,
                const double _G0, const double _s0 )
        : base_type( model, elastic, NoFracture{}, delta, K )
        , G0( _G0 )
        , s0( _s0 )
    {
        bond_break_coeff = ( 1.0 + s0 ) * ( 1.0 + s0 );
    }

    KOKKOS_INLINE_FUNCTION
    bool criticalStretch( const int, const int, const double r,
                          const double xi ) const
    {
        return r * r >= bond_break_coeff * xi * xi;
    }
};

template <typename MemorySpace>
struct ForceModel<PMB, ElasticPerfectlyPlastic, Fracture,
                  TemperatureIndependent, MemorySpace>
    : public ForceModel<PMB, Elastic, Fracture, TemperatureIndependent>,
      public BasePlasticity<MemorySpace>
{
    using base_type = ForceModel<PMB, Elastic>;
    using base_plasticity_type = BasePlasticity<MemorySpace>;
    using base_type::model_type;
    using base_model = typename base_type::base_model;
    using fracture_type = Fracture;
    using mechanics_type = ElasticPerfectlyPlastic;
    using thermal_type = base_type::thermal_type;

    using base_type::bond_break_coeff;
    using base_type::c;
    using base_type::delta;
    using base_type::G0;
    using base_type::K;
    using base_type::s0;

    using base_plasticity_type::_s_p;
    double s_Y;

    using base_plasticity_type::updateBonds;

    ForceModel( PMB model, mechanics_type, MemorySpace, const double delta,
                const double K, const double G0, const double sigma_y )
        : base_type( model, Elastic{}, delta, K, G0,
                     // s0
                     ( 5.0 * G0 / sigma_y / delta + sigma_y / K ) / 6.0 )
        , base_plasticity_type()
        , s_Y( sigma_y / 3.0 / K )
    {
    }

    // FIXME: avoiding multiple inheritance.
    KOKKOS_INLINE_FUNCTION
    auto forceCoeff( const int i, const int n, const double s,
                     const double vol ) const
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
    auto energy( const int i, const int n, const double s, const double xi,
                 const double vol ) const
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
struct ForceModel<LinearPMB, Elastic, NoFracture, TemperatureIndependent>
    : public ForceModel<PMB, Elastic, NoFracture, TemperatureIndependent>
{
    using base_type = ForceModel<PMB, Elastic, NoFracture>;
    using model_type = LinearPMB;
    using base_model = typename base_type::base_model;
    using fracture_type = typename base_type::fracture_type;
    using mechanics_type = Elastic;
    using thermal_type = base_type::thermal_type;

    template <typename... Args>
    ForceModel( LinearPMB, Args... args )
        : base_type( base_model{}, args... )
    {
    }

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
    using mechanics_type = Elastic;
    using thermal_type = base_type::thermal_type;

    template <typename... Args>
    ForceModel( LinearPMB, Args... args )
        : base_type( base_model{}, std::forward<Args>( args )... )
    {
    }

    using base_type::c;
    using base_type::delta;
    using base_type::K;

    using base_type::bond_break_coeff;
    using base_type::G0;
    using base_type::s0;
};

template <typename ModelType>
ForceModel( ModelType, Elastic, NoFracture, const double delta, const double K )
    -> ForceModel<ModelType, Elastic, NoFracture>;

// Default to fracture.
template <typename ModelType>
ForceModel( ModelType, Elastic, const double delta, const double K,
            const double G0 ) -> ForceModel<ModelType, Elastic>;

template <typename ModelType>
ForceModel( ModelType, NoFracture, const double delta, const double K )
    -> ForceModel<ModelType, Elastic, NoFracture>;

// Default to elastic.
template <typename ModelType>
ForceModel( ModelType, const double delta, const double K, const double G0 )
    -> ForceModel<ModelType>;

template <typename ModelType, typename MemorySpace>
ForceModel( ModelType, ElasticPerfectlyPlastic, MemorySpace, const double delta,
            const double K, const double G0, const double sigma_y )
    -> ForceModel<ModelType, ElasticPerfectlyPlastic, Fracture,
                  TemperatureIndependent, MemorySpace>;

template <typename TemperatureType>
struct ForceModel<PMB, Elastic, NoFracture, TemperatureDependent,
                  TemperatureType>
    : public BaseTemperatureModel<TemperatureType>,
      public ForceModel<PMB, Elastic, NoFracture, TemperatureIndependent>
{
    using base_type =
        ForceModel<PMB, Elastic, NoFracture, TemperatureIndependent>;
    using base_temperature_type = BaseTemperatureModel<TemperatureType>;
    using base_type::model_type;
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

    ForceModel( PMB model, NoFracture fracture, const double _delta,
                const double _K, const TemperatureType _temp,
                const double _alpha, const double _temp0 = 0.0 )
        : base_temperature_type( _temp, _alpha, _temp0 )
        , base_type( model, fracture, _delta, _K )
    {
    }
};

template <typename TemperatureType>
struct ForceModel<PMB, Elastic, Fracture, TemperatureDependent, TemperatureType>
    : public BaseTemperatureModel<TemperatureType>,
      public ForceModel<PMB, Elastic, Fracture, TemperatureIndependent>

{
    using base_type =
        ForceModel<PMB, Elastic, Fracture, TemperatureIndependent>;
    using base_temperature_type = BaseTemperatureModel<TemperatureType>;
    using base_type::model_type;
    using base_model = typename base_type::base_model;
    using fracture_type = typename base_type::fracture_type;
    using mechanics_type = Elastic;
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

    ForceModel( PMB model, const double _delta, const double _K,
                const double _G0, const TemperatureType _temp,
                const double _alpha, const double _temp0 = 0.0 )
        : base_temperature_type( _temp, _alpha, _temp0 )
        , base_type( model, _delta, _K, _G0 )
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

template <typename TemperatureType>
struct ForceModel<PMB, ElasticPerfectlyPlastic, Fracture, TemperatureDependent,
                  TemperatureType>
    : public ForceModel<PMB, ElasticPerfectlyPlastic, Fracture,
                        TemperatureIndependent,
                        typename TemperatureType::memory_space>,
      BaseTemperatureModel<TemperatureType>
{
    using memory_space = typename TemperatureType::memory_space;
    using base_type = ForceModel<PMB, ElasticPerfectlyPlastic, Fracture,
                                 TemperatureIndependent, memory_space>;
    using base_temperature_type = BaseTemperatureModel<TemperatureType>;
    using typename base_type::model_type;
    using base_model = typename base_type::base_model;
    using fracture_type = typename base_type::fracture_type;
    using mechanics_type = ElasticPerfectlyPlastic;
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

    ForceModel( PMB model, ElasticPerfectlyPlastic mechanics,
                const double _delta, const double _K, const double _G0,
                const double _sigma_y, const TemperatureType _temp,
                const double _alpha, const double _temp0 = 0.0 )
        : base_type( model, mechanics, memory_space{}, _delta, _K, _G0,
                     _sigma_y )
        , base_temperature_type( _temp, _alpha, _temp0 )
    {
    }

    // This is copied from the base temperature.
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

template <typename ModelType, typename TemperatureType>
ForceModel( ModelType, NoFracture, const double delta, const double K,
            const TemperatureType& temp, const double alpha,
            const double temp0 = 0.0 )
    -> ForceModel<ModelType, Elastic, NoFracture, TemperatureDependent,
                  TemperatureType>;

template <typename ModelType, typename TemperatureType>
ForceModel( ModelType, const double delta, const double K, const double _G0,
            const TemperatureType& temp, const double alpha,
            const double temp0 = 0.0 )
    -> ForceModel<ModelType, Elastic, Fracture, TemperatureDependent,
                  TemperatureType>;

template <typename TemperatureType>
struct ForceModel<PMB, Elastic, NoFracture, DynamicTemperature, TemperatureType>
    : public BaseDynamicTemperatureModel,
      public ForceModel<PMB, Elastic, NoFracture, TemperatureDependent,
                        TemperatureType>
{
    using base_type = ForceModel<PMB, Elastic, NoFracture, TemperatureDependent,
                                 TemperatureType>;
    using base_temperature_type = BaseDynamicTemperatureModel;
    using typename base_type::model_type;
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

    ForceModel( PMB model, NoFracture fracture, const double _delta,
                const double _K, const TemperatureType& _temp,
                const double _kappa, const double _cp, const double _alpha,
                const double _temp0 = 0.0,
                const bool _constant_microconductivity = true )
        : base_temperature_type( _delta, _kappa, _cp,
                                 _constant_microconductivity )
        , base_type( model, fracture, _delta, _K, _temp, _alpha, _temp0 )

    {
    }
};

template <typename ModelType, typename TemperatureType>
ForceModel( ModelType, NoFracture, const double delta, const double K,
            const TemperatureType& temp, const double kappa, const double cp,
            const double alpha, const double temp0 = 0.0,
            const bool constant_microconductivity = true )
    -> ForceModel<ModelType, Elastic, NoFracture, DynamicTemperature,
                  TemperatureType>;

template <typename ModelType, typename TemperatureType>
struct ForceModel<ModelType, Elastic, Fracture, DynamicTemperature,
                  TemperatureType>
    : public BaseDynamicTemperatureModel,
      public ForceModel<ModelType, Elastic, Fracture, TemperatureDependent,
                        TemperatureType>

{
    using base_type = ForceModel<PMB, Elastic, Fracture, TemperatureDependent,
                                 TemperatureType>;
    using base_temperature_type = BaseDynamicTemperatureModel;
    using typename base_type::model_type;
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

    ForceModel( PMB model, const double _delta, const double _K,
                const double _G0, const TemperatureType _temp,
                const double _kappa, const double _cp, const double _alpha,
                const double _temp0 = 0.0,
                const bool _constant_microconductivity = true )
        : base_temperature_type( _delta, _kappa, _cp,
                                 _constant_microconductivity )
        , base_type( model, _delta, _K, _G0, _temp, _alpha, _temp0 )
    {
    }
};

template <typename ModelType, typename TemperatureType>
ForceModel( ModelType, const double delta, const double K, const double G0,
            const TemperatureType& temp, const double kappa, const double cp,
            const double alpha, const double temp0 = 0.0,
            const bool constant_microconductivity = true )
    -> ForceModel<ModelType, Elastic, Fracture, DynamicTemperature,
                  TemperatureType>;

// Force models with evolving density.
template <typename TemperatureType, typename DensityType,
          typename CurrentDensityType>
struct ForceDensityModel<PMB, ElasticPerfectlyPlastic, Fracture,
                         TemperatureDependent, TemperatureType, DensityType,
                         CurrentDensityType>
    : public ForceModel<PMB, ElasticPerfectlyPlastic, Fracture,
                        TemperatureDependent, TemperatureType>,
      ForceModel<LPS, Elastic, Fracture, TemperatureIndependent>
{
    using base_type = ForceModel<PMB, ElasticPerfectlyPlastic, Fracture,
                                 TemperatureDependent, TemperatureType>;
    using lps_base_type =
        ForceModel<LPS, Elastic, Fracture, TemperatureIndependent>;
    using typename base_type::model_type;
    using base_model = typename base_type::base_model;
    using fracture_type = typename base_type::fracture_type;
    using mechanics_type = typename base_type::mechanics_type;
    using thermal_type = typename base_type::thermal_type;
    using density_type = DynamicDensity;

    // Does not use base_type::c because it is state dependent.
    using base_type::_s_p;
    using base_type::bond_break_coeff;
    using base_type::delta;
    using base_type::G0;
    using base_type::K;
    using base_type::s0;
    using base_type::s_Y;
    double coeff;

    double rho0;
    DensityType rho;
    CurrentDensityType rho_current;

    // Define which base functions to use (do not use LPS).
    using base_type::cutoff;
    using base_type::energy;
    using base_type::thermalStretch;
    using base_type::updateBonds;

    // Thermal parameters
    using base_type::alpha;
    using base_type::temp0;
    using base_type::temperature;

    ForceDensityModel( PMB model, ElasticPerfectlyPlastic mechanics,
                       const DensityType& _rho,
                       const CurrentDensityType& _rho_c, const double delta,
                       const double K, const double G0, const double sigma_y,
                       const double _rho0, const TemperatureType _temp,
                       const double alpha, const double temp0 = 0.0 )
        : base_type( model, mechanics, delta, K, G0, sigma_y, _temp, alpha,
                     temp0 )
        , lps_base_type( LPS{}, Fracture{}, delta, K, ( 3.0 / 5.0 * K ), G0 )
        , rho0( _rho0 )
        , rho( _rho )
        , rho_current( _rho_c )
    {
        coeff = 3.0 / pi / delta / delta / delta / delta;
    }

    KOKKOS_INLINE_FUNCTION
    auto currentC( const int i ) const
    {
        // FIXME: form of density dependence.
        return 6.0 * coeff * K * rho_current( i ) / rho0;
    }

    KOKKOS_INLINE_FUNCTION
    auto forceCoeff( const int i, const int, const double s,
                     const double vol ) const
    {
        auto c_current = currentC( i ) * rho_current( i ) / rho0;
        return c_current * s * vol;
    }

    // Update plastic dilatation.
    KOKKOS_INLINE_FUNCTION auto dilatation( const int, const double s,
                                            const double xi, const double vol,
                                            const double ) const
    {
        return coeff * s * xi * vol;
    }

    // Density update using plastic dilatation.
    KOKKOS_INLINE_FUNCTION void updateDensity( const int i,
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
                   const double rho0, TemperatureType temp, const double _alpha,
                   const double _temp0 )
    -> ForceDensityModel<PMB, ElasticPerfectlyPlastic, Fracture,
                         TemperatureDependent, TemperatureType, DensityType,
                         CurrentDensityType>;

} // namespace CabanaPD

#endif
