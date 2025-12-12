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

    KOKKOS_FUNCTION
    auto c() const { return _c; }

    KOKKOS_INLINE_FUNCTION
    auto operator()( ForceCoeffTag, const int, const int, const double s,
                     const double vol, const double, const double, const double,
                     const double, const int = -1 ) const
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
struct BaseForceModelPMB<Elastic, TransverselyIsotropic>
    : public BaseForceModelPMB<Elastic, Isotropic>
{
    using base_type = BaseForceModelPMB<Elastic, Isotropic>;

    using base_type::delta;
    double C11;
    double C13;
    double C33;
    double A1111;
    double A1133;
    double A3333;

    using base_type::operator();

    BaseForceModelPMB( PMB model, NoFracture fracture,
                       TransverselyIsotropic aniso, const double delta,
                       const double _C11, const double _C13, const double _C33 )
        : BaseForceModelPMB( model, Elastic{}, fracture, aniso, delta, _C11,
                             _C13, _C33 )
    {
    }

    BaseForceModelPMB( PMB model, Elastic, NoFracture fracture,
                       TransverselyIsotropic, const double delta,
                       const double _C11, const double _C13, const double _C33 )
        : base_type( model, fracture, delta, 1.0 / 3.0 * ( _C11 + 2.0 * _C13 ) )
        , C11( _C11 )
        , C13( _C13 )
        , C33( _C33 )
    {
        A1111 = 75.0 / 4.0 * C11 - 75.0 / 2.0 * C13 + 15.0 / 4.0 * C33;
        A1133 = -25.0 / 3.0 * C11 + 115.0 / 2.0 * C13 + 5.0 / 4.0 * C33;
        A3333 = 10.0 * C11 - 90.0 * C13 + 30.0 * C33;
    }

    // Constructor to average from existing models.
    template <typename ModelType1, typename ModelType2>
    BaseForceModelPMB( const ModelType1& model1, const ModelType2& model2 )
        : base_type( model1, model2 )
    {
    }

    KOKKOS_FUNCTION
    auto lambda( const double xi, const double xi1, const double xi2,
                 const double xi3 ) const
    {
        return ( A1111 * ( Kokkos::pow( xi1, 2.0 ) + Kokkos::pow( xi2, 2.0 ) ) *
                     ( Kokkos::pow( xi1, 2.0 ) + Kokkos::pow( xi2, 2.0 ) ) +
                 6.0 * A1133 *
                     ( Kokkos::pow( xi1, 2.0 ) + Kokkos::pow( xi2, 2.0 ) ) *
                     Kokkos::pow( xi3, 2.0 ) +
                 A3333 * Kokkos::pow( xi3, 4.0 ) ) /
               ( pi * Kokkos::pow( delta, 4.0 ) * Kokkos::pow( xi, 4.0 ) );
    }

    KOKKOS_INLINE_FUNCTION
    auto operator()( ForceCoeffTag, const int, const int, const double s,
                     const double vol, const double xi, const double xi_x,
                     const double xi_y, const double xi_z,
                     const int = -1 ) const
    {
        return lambda( xi, xi_x, xi_y, xi_z ) * s * vol;
    }
};

template <>
struct BaseForceModelPMB<Elastic, Cubic> : public BaseForceModel
{
    using base_type = BaseForceModel;
    using model_type = PMB;
    using base_model = PMB;
    using mechanics_type = Elastic;

    using base_type::delta;
    double C11;
    double C12;
    double A1111;
    double A1122;

    BaseForceModelPMB( PMB model, NoFracture fracture, Cubic cubic,
                       const double delta, const double _C11,
                       const double _C12 )
        : BaseForceModelPMB( model, Elastic{}, fracture, cubic, delta, _C11,
                             _C12 )
    {
    }

    BaseForceModelPMB( PMB, Elastic, NoFracture, Cubic, const double delta,
                       const double _C11, const double _C12 )
        : base_type( delta, 1.0 / 3.0 * ( _C11 + 2.0 * _C12 ) )
        , C11( _C11 )
        , C12( _C12 )
    {
        A1111 = 75.0 / 2.0 * C11 - 165.0 / 2.0 * C12;
        A1122 = -55.0 / 4.0 * C11 + 205.0 / 4.0 * C12;
    }

    // Constructor to average from existing models.
    template <typename ModelType1, typename ModelType2>
    BaseForceModelPMB( const ModelType1& model1, const ModelType2& model2 )
        : base_type( model1, model2 )
    {
    }

    KOKKOS_FUNCTION
    auto lambda( const double xi, const double xi1, const double xi2,
                 const double xi3 ) const
    {
        return ( A1111 * ( Kokkos::pow( xi1, 4.0 ) + Kokkos::pow( xi2, 4.0 ) +
                           Kokkos::pow( xi3, 4.0 ) ) +
                 6.0 * A1122 *
                     ( Kokkos::pow( xi1, 2.0 ) * Kokkos::pow( xi2, 2.0 ) +
                       Kokkos::pow( xi1, 2.0 ) * Kokkos::pow( xi3, 2.0 ) ) +
                 Kokkos::pow( xi2, 2.0 ) * Kokkos::pow( xi3, 2.0 ) ) /
               ( pi * Kokkos::pow( delta, 4.0 ) * Kokkos::pow( xi, 4.0 ) );
    }

    KOKKOS_INLINE_FUNCTION
    auto operator()( ForceCoeffTag, const int, const int, const double s,
                     const double vol, const double xi, const double xi_x,
                     const double xi_y, const double xi_z,
                     const int = -1 ) const
    {
        return lambda( xi, xi_x, xi_y, xi_z ) * s * vol;
    }

    KOKKOS_INLINE_FUNCTION
    auto operator()( EnergyTag, const int, const int, const double,
                     const double, const double, const int = -1 ) const
    {
        // TODO: implement cubic energy.
        return 0.0;
    }
};

template <typename MemorySpace>
struct BaseForceModelPMB<ElasticPerfectlyPlastic, Isotropic, MemorySpace>
    : public BaseForceModelPMB<Elastic, Isotropic>,
      public BasePlasticity<MemorySpace>
{
    using base_type = BaseForceModelPMB<Elastic, Isotropic>;
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
                     const double vol, const double, const double, const double,
                     const double, const int n ) const
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

template <typename MemorySpace>
struct BaseForceModelPMB<ElasticPerfectlyPlastic, TransverselyIsotropic,
                         MemorySpace>
    : public BaseForceModelPMB<Elastic, TransverselyIsotropic>,
      public BasePlasticity<MemorySpace>
{
    using base_type = BaseForceModelPMB<Elastic, TransverselyIsotropic>;
    using base_plasticity_type = BasePlasticity<MemorySpace>;

    using mechanics_type = ElasticPerfectlyPlastic;

    using base_plasticity_type::_s_p;
    Kokkos::Array<double, 2> s_Y;

    using base_plasticity_type::updateBonds;

    BaseForceModelPMB( PMB model, mechanics_type, MemorySpace,
                       const double delta, const double _C11, const double _C13,
                       const double _C33,
                       const Kokkos::Array<double, 2> sigma_y )
        : base_type( model, NoFracture{}, TransverselyIsotropic{}, delta, _C11,
                     _C13, _C33 )
        , base_plasticity_type()
        , s_Y( sigma_y )
    {
        for ( std::size_t d = 0; d < s_Y.size(); d++ )
            s_Y[d] /= ( 3.0 * base_type::K );
    }

    // Constructor to average from existing models.
    template <typename ModelType1, typename ModelType2>
    BaseForceModelPMB( const ModelType1& model1, const ModelType2& model2 )
        : base_type( model1, model2 )
        , base_plasticity_type()
    {
        for ( std::size_t d = 0; d < s_Y.size(); d++ )
            s_Y[d] = ( model1.s_Y[d] + model2.s_Y[d] ) / 2.0;
    }

    KOKKOS_INLINE_FUNCTION
    auto operator()( ForceCoeffTag, const int i, const int, const double s,
                     const double vol, const double xi, const double xi_x,
                     const double xi_y, const double xi_z, const int n ) const
    {
        // Compute orientation-dependent yield stretch
        auto s_Y_o = s_Y[0] + ( s_Y[1] - s_Y[0] ) * xi_z * xi_z / ( xi * xi );

        auto s_p = _s_p( i, n );
        // Yield in tension.
        if ( s >= s_p + s_Y_o )
            _s_p( i, n ) = s - s_Y_o;
        // Yield in compression.
        else if ( s <= s_p - s_Y_o )
            _s_p( i, n ) = s + s_Y_o;
        // else: Elastic (in between), do not modify.

        // Must extract again if in the plastic regime.
        s_p = _s_p( i, n );
        return lambda( xi, xi_x, xi_y, xi_z ) * ( s - s_p ) * vol;
    }

    KOKKOS_INLINE_FUNCTION
    auto operator()( EnergyTag, const int, const int, const double,
                     const double, const double, const int = -1 ) const
    {
        // TODO: implement energy.
        return 0.0;
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
      BaseFractureModel<AnisotropyType>,
      BaseTemperatureModel<TemperatureIndependent>
{
    using base_type = BaseForceModelPMB<Elastic, AnisotropyType>;
    using base_fracture_type = BaseFractureModel<AnisotropyType>;
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

template <typename MemorySpace>
struct ForceModel<PMB, ElasticPerfectlyPlastic, Isotropic, Fracture,
                  TemperatureIndependent, MemorySpace>
    : public BaseForceModelPMB<ElasticPerfectlyPlastic, Isotropic, MemorySpace>,
      public BaseFractureModel<Isotropic>,
      public BaseTemperatureModel<TemperatureIndependent>
{
    using base_type =
        BaseForceModelPMB<ElasticPerfectlyPlastic, Isotropic, MemorySpace>;
    using base_fracture_type = BaseFractureModel<Isotropic>;
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

template <typename ModelType>
ForceModel( ModelType, Elastic, NoFracture, Cubic, const double delta,
            const double, const double )
    -> ForceModel<ModelType, Elastic, Cubic, NoFracture>;

template <typename ModelType>
ForceModel( ModelType, NoFracture, Cubic, const double delta, const double,
            const double ) -> ForceModel<ModelType, Elastic, Cubic, NoFracture>;

template <typename ModelType>
ForceModel( ModelType, Elastic, NoFracture, TransverselyIsotropic,
            const double delta, const double, const double, const double )
    -> ForceModel<ModelType, Elastic, TransverselyIsotropic, NoFracture>;

template <typename ModelType>
ForceModel( ModelType, NoFracture, TransverselyIsotropic, const double delta,
            const double, const double, const double )
    -> ForceModel<ModelType, Elastic, TransverselyIsotropic, NoFracture>;

template <typename MemorySpace>
struct ForceModel<PMB, ElasticPerfectlyPlastic, TransverselyIsotropic, Fracture,
                  TemperatureIndependent, MemorySpace>
    : public BaseForceModelPMB<ElasticPerfectlyPlastic, TransverselyIsotropic,
                               MemorySpace>,
      public BaseFractureModel<TransverselyIsotropic>,
      public BaseTemperatureModel<TemperatureIndependent>
{
    using base_type = BaseForceModelPMB<ElasticPerfectlyPlastic,
                                        TransverselyIsotropic, MemorySpace>;
    using base_fracture_type = BaseFractureModel<TransverselyIsotropic>;
    using base_temperature_type = BaseTemperatureModel<TemperatureIndependent>;

    using base_type::operator();
    using base_fracture_type::operator();
    using base_temperature_type::operator();
    using base_type::updateBonds;

    ForceModel( PMB model, ElasticPerfectlyPlastic mechanics,
                TransverselyIsotropic, MemorySpace space, const double delta,
                const double C11, const double C13, const double C33,
                const Kokkos::Array<double, 2> sigma_y,
                const Kokkos::Array<double, 2> s0 )
        : base_type( model, mechanics, space, delta, C11, C13, C33, sigma_y )
        , base_fracture_type( s0 )
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

template <typename ModelType, typename MemorySpace>
ForceModel( ModelType, ElasticPerfectlyPlastic, TransverselyIsotropic,
            MemorySpace, const double, const double, const double, const double,
            const Kokkos::Array<double, 2>, const Kokkos::Array<double, 2> )
    -> ForceModel<ModelType, ElasticPerfectlyPlastic, TransverselyIsotropic,
                  Fracture, TemperatureIndependent, MemorySpace>;

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

template <typename TemperatureType>
struct ForceModel<PMB, Elastic, Isotropic, Fracture, TemperatureDependent,
                  TemperatureType>
    : public BaseForceModelPMB<Elastic, Isotropic>,
      ThermalFractureModel<Isotropic, TemperatureType>
{
    using base_type = BaseForceModelPMB<Elastic, Isotropic>;
    using base_temperature_type =
        ThermalFractureModel<Isotropic, TemperatureType>;

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

template <typename TemperatureType>
struct ForceModel<PMB, ElasticPerfectlyPlastic, Isotropic, Fracture,
                  TemperatureDependent, TemperatureType>
    : public BaseForceModelPMB<ElasticPerfectlyPlastic, Isotropic,
                               typename TemperatureType::memory_space>,
      public ThermalFractureModel<Isotropic, TemperatureType>
{
    using base_type = BaseForceModelPMB<ElasticPerfectlyPlastic, Isotropic,
                                        typename TemperatureType::memory_space>;
    using base_temperature_type =
        ThermalFractureModel<Isotropic, TemperatureType>;

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

template <typename ModelType, typename TemperatureType>
struct ForceModel<ModelType, Elastic, Isotropic, Fracture, DynamicTemperature,
                  TemperatureType>
    : public BaseForceModelPMB<Elastic, Isotropic>,
      ThermalFractureModel<Isotropic, TemperatureType>,
      BaseDynamicTemperatureModel
{
    using base_type = BaseForceModelPMB<Elastic, Isotropic>;
    using base_temperature_type =
        ThermalFractureModel<Isotropic, TemperatureType>;
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

template <typename TemperatureType>
struct ForceModel<PMB, ElasticPerfectlyPlastic, Isotropic, Fracture,
                  DynamicTemperature, TemperatureType>
    : public BaseForceModelPMB<ElasticPerfectlyPlastic, Isotropic,
                               typename TemperatureType::memory_space>,
      public ThermalFractureModel<Isotropic, TemperatureType>,
      BaseDynamicTemperatureModel
{
    using base_type = BaseForceModelPMB<ElasticPerfectlyPlastic, Isotropic,
                                        typename TemperatureType::memory_space>;
    using base_temperature_type =
        ThermalFractureModel<Isotropic, TemperatureType>;
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
