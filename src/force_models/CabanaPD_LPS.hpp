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

#ifndef FORCE_MODELS_LPS_H
#define FORCE_MODELS_LPS_H

#include <Kokkos_Core.hpp>

#include <CabanaPD_ForceModels.hpp>
#include <CabanaPD_Output.hpp>
#include <CabanaPD_Types.hpp>

namespace CabanaPD
{
template <typename MechanicsModelType>
struct BaseForceModelLPS;

template <>
struct BaseForceModelLPS<Elastic> : public BaseForceModel
{
    using base_type = BaseForceModel;
    using model_type = LPS;
    using base_model = LPS;

    using base_type::delta;

    int influence_type;
    InfluenceFunctionTag influence_tag;

    using base_type::K;
    double G;
    // Store coefficients for multi-material systems.
    // TODO: this currently only supports bi-material systems.
    Kokkos::Array<double, 2> theta_coeff;
    Kokkos::Array<double, 2> s_coeff;

    BaseForceModelLPS( LPS, NoFracture, const double _delta, const double _K,
                       const double _G, const int _influence = 0 )
        : base_type( _delta, _K )
        , influence_type( _influence )
        , G( _G )
    {
        init();
    }

    BaseForceModelLPS( LPS, Elastic, NoFracture, const double _delta,
                       const double _K, const double _G,
                       const int _influence = 0 )
        : base_type( _delta, _K )
        , influence_type( _influence )
        , G( _G )
    {
        init();
    }

    // Constructor to average from existing models.
    template <typename ModelType1, typename ModelType2>
    BaseForceModelLPS( const ModelType1& model1, const ModelType2& model2 )
        : base_type( model1, model2 )
    {
        G = ( model1.G + model2.G ) / 2.0;
        theta_coeff[0] = 3.0 * model1.K - 5.0 * model1.G;
        s_coeff[0] = 15.0 * model1.G;
        theta_coeff[1] = 3.0 * model2.K - 5.0 * model2.G;
        s_coeff[1] = 15.0 * model2.G;

        influence_type = model1.influence_type;
        if ( model2.influence_type != model1.influence_type )
            log_err( std::cout,
                     "Influence function type for each model must match for "
                     "multi-material systems" );
    }

    void init()
    {
        theta_coeff[0] = 3.0 * K - 5.0 * G;
        s_coeff[0] = 15.0 * G;
        // Set extra coefficients for multi-material.
        theta_coeff[1] = theta_coeff[0];
        s_coeff[1] = s_coeff[0];

        if ( influence_type > 1 || influence_type < 0 )
            log_err( std::cout, "Influence function type must be 0 or 1." );
    }

    KOKKOS_INLINE_FUNCTION auto operator()( InfluenceFunctionTag,
                                            double xi ) const
    {
        if ( influence_type == 1 )
            return 1.0 / xi;
        else
            return 1.0;
    }

    KOKKOS_INLINE_FUNCTION auto operator()( InfluenceFunctionTag, const int,
                                            const int, double xi ) const
    {
        return ( *this )( influence_tag, xi );
    }

    KOKKOS_INLINE_FUNCTION
    auto operator()( WeightedVolumeTag, const int, const int, const double xi,
                     const double vol ) const
    {
        auto influence = ( *this )( influence_tag, xi );
        return influence * xi * xi * vol;
    }

    KOKKOS_INLINE_FUNCTION auto operator()( DilatationTag, const int, const int,
                                            const double s, const double xi,
                                            const double vol,
                                            const double m_i ) const
    {
        auto influence = ( *this )( influence_tag, xi );
        double theta_i = influence * s * xi * xi * vol;
        return 3.0 * theta_i / m_i;
    }

    // In this case, we know that we only have one material so we use the first.
    // This must be separate from the multi-material interface because no type
    // information was passed here.
    KOKKOS_INLINE_FUNCTION auto
    operator()( ForceCoeffTag, SingleMaterial, const int, const int,
                const double s, const double xi, const double vol,
                const double m_i, const double m_j, const double theta_i,
                const double theta_j ) const
    {
        auto influence = ( *this )( influence_tag, xi );

        return ( theta_coeff[0] * ( theta_i / m_i + theta_j / m_j ) +
                 s_coeff[0] * s * ( 1.0 / m_i + 1.0 / m_j ) ) *
               influence * xi * vol;
    }

    // In this case we may have any combination of material types. These
    // coefficients may still be the same for some interaction pairs.
    KOKKOS_INLINE_FUNCTION auto
    operator()( ForceCoeffTag, MultiMaterial, const int type_i,
                const int type_j, const double s, const double xi,
                const double vol, const double m_i, const double m_j,
                const double theta_i, const double theta_j ) const
    {
        KOKKOS_ASSERT( type_i < 2 );
        KOKKOS_ASSERT( type_j < 2 );
        auto influence = ( *this )( influence_tag, xi );
        double theta_coeff_i = theta_coeff[type_i];
        double theta_coeff_j = theta_coeff[type_j];
        double s_coeff_i = s_coeff[type_i];
        double s_coeff_j = s_coeff[type_j];

        return ( theta_coeff_i * theta_i / m_i + theta_coeff_j * theta_j / m_j +
                 s * ( s_coeff_i / m_i + s_coeff_j / m_j ) ) *
               influence * xi * vol;
    }

    // In this case, we know that we only have one material so we use the first.
    // This must be separate from the multi-material interface because no type
    // information was passed here.
    KOKKOS_INLINE_FUNCTION
    auto operator()( EnergyTag, SingleMaterial, const int, const int,
                     const double s, const double xi, const double vol,
                     const double m_i, const double theta_i,
                     const double num_bonds ) const
    {
        auto influence = ( *this )( influence_tag, xi );
        return 1.0 / num_bonds * 0.5 * theta_coeff[0] / 3.0 *
                   ( theta_i * theta_i ) +
               0.5 * ( s_coeff[0] / m_i ) * influence * s * s * xi * xi * vol;
    }

    // In this case we may have any combination of material types. These
    // coefficients may still be the same for some interaction pairs.
    KOKKOS_INLINE_FUNCTION
    auto operator()( EnergyTag, MultiMaterial, const int type_i, const int,
                     const double s, const double xi, const double vol,
                     const double m_i, const double theta_i,
                     const double num_bonds ) const
    {
        auto influence = ( *this )( influence_tag, xi );

        KOKKOS_ASSERT( type_i < 2 );
        double theta_coeff_i = theta_coeff[type_i];
        double s_coeff_i = s_coeff[type_i];
        return 1.0 / num_bonds * 0.5 * theta_coeff_i / 3.0 *
                   ( theta_i * theta_i ) +
               0.5 * ( s_coeff_i / m_i ) * influence * s * s * xi * xi * vol;
    }
};

template <>
struct ForceModel<LPS, Elastic, NoFracture, TemperatureIndependent>
    : public BaseForceModelLPS<Elastic>,
      BaseNoFractureModel,
      BaseTemperatureModel<TemperatureIndependent>

{
    using base_type = BaseForceModelLPS<Elastic>;
    using base_fracture_type = BaseNoFractureModel;
    using base_temperature_type = BaseTemperatureModel<TemperatureIndependent>;
    using fracture_type = NoFracture;
    using thermal_type = typename base_temperature_type::thermal_type;

    using base_type::base_type;
    using base_type::operator();
    using base_fracture_type::operator();
    using base_temperature_type::operator();

    using base_type::influence_type;
};

template <>
struct ForceModel<LPS, Elastic, Fracture, TemperatureIndependent>
    : public BaseForceModelLPS<Elastic>,
      BaseFractureModel,
      BaseTemperatureModel<TemperatureIndependent>
{
    using base_type = BaseForceModelLPS<Elastic>;
    using base_fracture_type = BaseFractureModel;
    using base_temperature_type = BaseTemperatureModel<TemperatureIndependent>;

    using fracture_type = Fracture;
    using thermal_type = base_temperature_type::thermal_type;

    using base_fracture_type::bond_break_coeff;
    using base_fracture_type::G0;
    using base_fracture_type::s0;
    using base_type::base_type;
    using base_type::delta;
    using base_type::influence_type;
    using base_type::K;

    using base_type::operator();
    using base_fracture_type::operator();
    using base_temperature_type::operator();

    ForceModel( LPS model, const double _delta, const double _K,
                const double _G, const double _G0, const int _influence = 0 )
        : base_type( model, NoFracture{}, _delta, _K, _G, _influence )
        , base_fracture_type( _delta, _K, _G0, _influence )
    {
        init();
    }

    ForceModel( LPS model, Fracture, const double _delta, const double _K,
                const double _G, const double _G0, const int _influence = 0 )
        : base_type( model, NoFracture{}, _delta, _K, _G, _influence )
        , base_fracture_type( _delta, _K, _G0, _influence )
    {
        init();
    }

    ForceModel( LPS model, Elastic elastic, Fracture, const double _delta,
                const double _K, const double _G, const double _G0,
                const int _influence = 0 )
        : base_type( model, elastic, NoFracture{}, _delta, _K, _G, _influence )
        , base_fracture_type( _delta, _K, _G0, _influence )
    {
        init();
    }

    // Constructor to average from existing models.
    template <typename ModelType1, typename ModelType2>
    ForceModel( const ModelType1& model1, const ModelType2& model2 )
        : base_type( model1, model2 )
        , base_fracture_type( model1, model2 )
    {
    }

    void init()
    {
        if ( influence_type == 1 )
        {
            s0 = Kokkos::sqrt( 5.0 * G0 / 9.0 / K / delta ); // 1/xi
        }
        else
        {
            s0 = Kokkos::sqrt( 8.0 * G0 / 15.0 / K / delta ); // 1
        }
        bond_break_coeff = ( 1.0 + s0 ) * ( 1.0 + s0 );
    }
};

template <>
struct ForceModel<LinearLPS, Elastic, NoFracture, TemperatureIndependent>
    : public ForceModel<LPS, Elastic, NoFracture, TemperatureIndependent>
{
    using base_type =
        ForceModel<LPS, Elastic, NoFracture, TemperatureIndependent>;
    using model_type = LinearLPS;

    template <typename... Args>
    ForceModel( LinearLPS, Args&&... args )
        : base_type( base_model{}, std::forward<Args>( args )... )
    {
    }

    using base_type::base_type;
    using base_type::operator();
};

template <>
struct ForceModel<LinearLPS, Elastic, Fracture, TemperatureIndependent>
    : public ForceModel<LPS, Elastic, Fracture, TemperatureIndependent>
{
    using base_type =
        ForceModel<LPS, Elastic, Fracture, TemperatureIndependent>;

    using model_type = LinearLPS;

    template <typename... Args>
    ForceModel( LinearLPS, Args&&... args )
        : base_type( base_model{}, std::forward<Args>( args )... )
    {
    }

    using base_type::base_type;
    using base_type::operator();
};

template <typename ModelType>
ForceModel( ModelType, Elastic, NoFracture, const double delta, const double K,
            const double G, const int influence = 0 )
    -> ForceModel<ModelType, Elastic, NoFracture>;

template <typename ModelType>
ForceModel( ModelType, NoFracture, const double delta, const double K,
            const double G, const int influence = 0 )
    -> ForceModel<ModelType, Elastic, NoFracture>;

template <typename ModelType>
ForceModel( ModelType, Elastic, const double delta, const double K,
            const double G, const int influence = 0 )
    -> ForceModel<ModelType, Elastic>;

template <typename ModelType>
ForceModel( ModelType, const double _delta, const double _K, const double _G,
            const double _G0, const int _influence = 0 )
    -> ForceModel<ModelType>;

} // namespace CabanaPD

#endif
