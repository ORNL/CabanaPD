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

#include <CabanaPD_Constants.hpp>
#include <CabanaPD_ForceModels.hpp>
#include <CabanaPD_Types.hpp>

namespace CabanaPD
{

template <>
struct ForceModel<PMB, Elastic, TemperatureIndependent>
    : public BaseForceModel<>
{
    using base_type = BaseForceModel<>;
    using species_type = typename base_type::species_type;
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
        setParameters( delta, K );
    }

    void setParameters( const double _delta, const double _K )
    {
        delta = _delta;
        K = _K;
        c = micromodulus();
    }

    KOKKOS_INLINE_FUNCTION
    double micromodulus()
    {
        return 18.0 * K / ( pi * delta * delta * delta * delta );
    }

    KOKKOS_INLINE_FUNCTION
    auto micromodulus( const int, const int ) const { return c; }
};

template <typename ParticleType>
struct ForceModel<PMB, Elastic, TemperatureIndependent, ParticleType>
    : public BaseForceModel<typename ParticleType::memory_space>
{
    using memory_space = typename ParticleType::memory_space;
    using base_type = BaseForceModel<memory_space>;
    using species_type = typename base_type::species_type;
    using base_model = PMB;
    using fracture_type = Elastic;
    using thermal_type = TemperatureIndependent;

    using base_type::delta;
    using base_type::num_types;
    using view_type_1d = typename base_type::view_type_1d;
    using view_type_2d = typename base_type::view_type_2d;
    view_type_2d c;
    view_type_1d K;
    ParticleType type;

    template <typename ArrayType>
    ForceModel( const ArrayType& delta, const ArrayType& _K,
                ParticleType _type )
        : base_type( delta )
        , c( view_type_2d( "micromodulus", num_types, num_types ) )
        , K( view_type_1d( "bulk_modulus", num_types ) )
        , type( _type )
    {
        setParameters( _K );
    }

    template <typename ArrayType>
    void setParameters( const ArrayType& _K )
    {
        // Initialize self interaction parameters.
        auto init_self_func = KOKKOS_CLASS_LAMBDA( const int i )
        {
            K( i ) = _K[i];
            c( i, i ) = micromodulus( i );
        };
        using exec_space = typename memory_space::execution_space;
        Kokkos::RangePolicy<exec_space> policy( 0, num_types );
        Kokkos::parallel_for( "CabanaPD::Model::Init", policy, init_self_func );
        Kokkos::fence();

        // Initialize cross-terms.
        auto init_cross_func = KOKKOS_CLASS_LAMBDA( const int i )
        {
            for ( std::size_t j = i; j < num_types; j++ )
                c( i, j ) = ( micromodulus( i ) + micromodulus( j ) ) / 2.0;
        };
        Kokkos::parallel_for( "CabanaPD::Model::Init", policy,
                              init_cross_func );
    }

    KOKKOS_INLINE_FUNCTION
    auto micromodulus( const int i ) const
    {
        auto d = delta( i );
        return 18.0 * K( i ) / ( pi * d * d * d * d );
    }

    KOKKOS_INLINE_FUNCTION
    auto micromodulus( const int i, const int j ) const
    {
        return c( type( i ), type( j ) );
    }

    void update( const ParticleType _type ) { type = _type; }
};

template <typename ParticleType>
auto createForceModel( PMB, Elastic, TemperatureIndependent,
                       ParticleType particles, const double delta,
                       const double K )
{
    auto type = particles.sliceType();
    using type_type = decltype( type );
    return ForceModel<PMB, Elastic, TemperatureIndependent, type_type>(
        delta, K, type );
}

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
        setParameters( G0 );
    }

    void setParameters( const double _G0 )
    {
        G0 = _G0;
        s0 = sqrt( 5.0 * G0 / 9.0 / K / delta );
        bond_break_coeff = ( 1.0 + s0 ) * ( 1.0 + s0 );
    }

    KOKKOS_INLINE_FUNCTION
    bool criticalStretch( const int, const int, const double r,
                          const double xi ) const
    {
        return r * r >= bond_break_coeff * xi * xi;
    }
};

template <typename ParticleType>
struct ForceModel<PMB, Fracture, TemperatureIndependent, ParticleType>
    : public ForceModel<PMB, Elastic, TemperatureIndependent, ParticleType>
{
    using base_type =
        ForceModel<PMB, Elastic, TemperatureIndependent, ParticleType>;
    using memory_space = typename base_type::memory_space;
    using base_model = typename base_type::base_model;
    using species_type = typename base_type::species_type;
    using fracture_type = Fracture;
    using thermal_type = typename base_type::thermal_type;

    using base_type::c;
    using base_type::delta;
    using base_type::K;
    using base_type::num_types;
    using base_type::type;

    using view_type_1d = typename base_type::view_type_1d;
    using view_type_2d = typename base_type::view_type_2d;
    view_type_1d G0;
    view_type_2d s0;
    view_type_2d bond_break_coeff;

    template <typename ArrayType>
    ForceModel( const ArrayType& delta, const ArrayType& K,
                const ArrayType& _G0, const ParticleType& _type )
        : base_type( delta, K, _type )
        , G0( view_type_1d( "fracture_energy", num_types ) )
        , s0( view_type_2d( "critical_stretch", num_types, num_types ) )
        , bond_break_coeff(
              view_type_2d( "break_coeff", num_types, num_types ) )
    {
        setParameters( _G0 );
    }

    template <typename ArrayType>
    void setParameters( const ArrayType& _G0 )
    {
        // Initialize self interaction parameters.
        auto init_self_func = KOKKOS_CLASS_LAMBDA( const int i )
        {
            G0( i ) = _G0[i];
            s0( i, i ) = criticalStretch( i );
            bond_break_coeff( i, i ) =
                ( 1.0 + s0( i, i ) ) * ( 1.0 + s0( i, i ) );
        };
        using exec_space = typename memory_space::execution_space;
        Kokkos::RangePolicy<exec_space> policy( 0, num_types );
        Kokkos::parallel_for( "CabanaPD::Model::Init", policy, init_self_func );
        Kokkos::fence();

        // Initialize cross-terms.
        auto init_cross_func = KOKKOS_CLASS_LAMBDA( const int i )
        {
            for ( std::size_t j = i; j < num_types; j++ )
            {
                s0( i, j ) = criticalStretch( i, j );
                bond_break_coeff( i, j ) =
                    ( 1.0 + s0( i, j ) ) * ( 1.0 + s0( i, j ) );
            }
        };
        Kokkos::parallel_for( "CabanaPD::Model::Init", policy,
                              init_cross_func );
    }

    KOKKOS_INLINE_FUNCTION
    auto criticalStretch( const int type_i ) const
    {
        return sqrt( 5.0 * G0( type_i ) / 9.0 / K( type_i ) / delta( type_i ) );
    }

    KOKKOS_INLINE_FUNCTION
    auto criticalStretch( const int type_i, const int type_j ) const
    {
        auto s0_i = s0( type_i, type_i );
        auto s0_j = s0( type_j, type_j );
        auto c_i = c( type_i, type_i );
        auto c_j = c( type_j, type_j );
        return Kokkos::sqrt( ( s0_i * s0_i * c_i + s0_j * s0_j * c_j ) /
                             ( c_i + c_j ) );
    }

    KOKKOS_INLINE_FUNCTION
    bool criticalStretch( const int i, const int j, const double r,
                          const double xi ) const
    {
        // if ( type( j ) != 0 )
        //     std::cout << bond_break_coeff.size() << " " << type( i ) << " "
        //               << type( j ) << "\n";
        //<< type.size() << " " << i << " " << j << "\n";
        return r * r >= bond_break_coeff( type( i ), type( j ) ) * xi * xi;
    }
};

template <typename ParticleType, typename ArrayType>
auto createForceModel( PMB, Fracture, TemperatureIndependent,
                       ParticleType particles, const ArrayType& delta,
                       const ArrayType& K, const ArrayType& G0 )
{
    auto type = particles.sliceType();
    using type_type = decltype( type );
    return ForceModel<PMB, Fracture, TemperatureIndependent, type_type>(
        delta, K, G0, type );
}

template <class... ModelParams>
struct ForceModel<LinearPMB, Elastic, TemperatureIndependent, ModelParams...>
    : public ForceModel<PMB, Elastic, TemperatureIndependent, ModelParams...>
{
    using base_type =
        ForceModel<PMB, Elastic, TemperatureIndependent, ModelParams...>;
    using base_model = typename base_type::base_model;
    using species_type = typename base_type::species_type;
    using fracture_type = typename base_type::fracture_type;
    using thermal_type = typename base_type::thermal_type;

    using base_type::base_type;

    using base_type::c;
    using base_type::delta;
    using base_type::K;
};

template <typename... ModelParams>
struct ForceModel<LinearPMB, Fracture, TemperatureIndependent, ModelParams...>
    : public ForceModel<PMB, Fracture, TemperatureIndependent, ModelParams...>
{
    using base_type =
        ForceModel<PMB, Fracture, TemperatureIndependent, ModelParams...>;
    using base_model = typename base_type::base_model;
    using species_type = typename base_type::species_type;
    using fracture_type = typename base_type::fracture_type;
    using thermal_type = typename base_type::thermal_type;

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
    using memory_space = typename TemperatureType::memory_space;
    using base_type = ForceModel<PMB, Elastic, TemperatureIndependent>;
    using species_type = typename base_type::species_type;
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
    }
};

template <typename ParticleType>
auto createForceModel( PMB, Elastic, TemperatureDependent,
                       ParticleType particles, const double delta,
                       const double K, const double alpha, const double temp0 )
{
    auto temp = particles.sliceTemperature();
    using temp_type = decltype( temp );
    return ForceModel<PMB, Elastic, TemperatureDependent, temp_type>(
        delta, K, temp, alpha, temp0 );
}

template <typename TemperatureType>
struct ForceModel<PMB, Fracture, TemperatureDependent, TemperatureType>
    : public ForceModel<PMB, Fracture, TemperatureIndependent>,
      BaseTemperatureModel<TemperatureType>
{
    using memory_space = typename TemperatureType::memory_space;
    using base_type = ForceModel<PMB, Fracture, TemperatureIndependent>;
    using species_type = typename base_type::species_type;
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

    // ForceModel(){};
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
    return ForceModel<PMB, Fracture, TemperatureDependent, temp_type>(
        delta, K, G0, temp, alpha, temp0 );
}

} // namespace CabanaPD

#endif
