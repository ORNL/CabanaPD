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
#include <CabanaPD_Types.hpp>

namespace CabanaPD
{
template <typename MechanicsModelType>
struct BaseForceModelLPS;

template <>
struct BaseForceModelLPS<Elastic> : public BaseForceModel
{
    using base_type = BaseForceModel;
    using base_model = LPS;

    using base_type::delta;

    int influence_type;

    double K;
    double G;
    double theta_coeff;
    double s_coeff;

    BaseForceModelLPS( LPS, NoFracture, const double _delta, const double _K,
                       const double _G, const int _influence = 0 )
        : base_type( _delta )
        , influence_type( _influence )
        , K( _K )
        , G( _G )
    {
        init();
    }

    BaseForceModelLPS( LPS, Elastic, NoFracture, const double _delta,
                       const double _K, const double _G,
                       const int _influence = 0 )
        : base_type( _delta )
        , influence_type( _influence )
        , K( _K )
        , G( _G )
    {
        init();
    }

    void init()
    {
        theta_coeff = 3.0 * K - 5.0 * G;
        s_coeff = 15.0 * G;
    }

    KOKKOS_INLINE_FUNCTION double influenceFunction( double xi ) const
    {
        if ( influence_type == 1 )
            return 1.0 / xi;
        else
            return 1.0;
    }

    KOKKOS_INLINE_FUNCTION auto weightedVolume( const double xi,
                                                const double vol ) const
    {
        return influenceFunction( xi ) * xi * xi * vol;
    }

    KOKKOS_INLINE_FUNCTION auto dilatation( const double s, const double xi,
                                            const double vol,
                                            const double m_i ) const
    {
        double theta_i = influenceFunction( xi ) * s * xi * xi * vol;
        return 3.0 * theta_i / m_i;
    }

    KOKKOS_INLINE_FUNCTION auto operator()( ForceCoeffTag, const int, const int,
                                            const double s, const double xi,
                                            const double vol, const double m_i,
                                            const double m_j,
                                            const double theta_i,
                                            const double theta_j ) const
    {
        return ( theta_coeff * ( theta_i / m_i + theta_j / m_j ) +
                 s_coeff * s * ( 1.0 / m_i + 1.0 / m_j ) ) *
               influenceFunction( xi ) * xi * vol;
    }

    KOKKOS_INLINE_FUNCTION
    auto operator()( EnergyTag, const int, const int, const double s,
                     const double xi, const double vol, const double m_i,
                     const double theta_i, const double num_bonds ) const
    {
        return 1.0 / num_bonds * 0.5 * theta_coeff / 3.0 *
                   ( theta_i * theta_i ) +
               0.5 * ( s_coeff / m_i ) * influenceFunction( xi ) * s * s * xi *
                   xi * vol;
    }
};

template <>
struct ForceModel<LPS, Elastic, NoFracture, TemperatureIndependent>
    : public BaseForceModelLPS<Elastic>,
      BaseTemperatureModel<TemperatureIndependent>

{
    using base_type = BaseForceModelLPS<Elastic>;
    using base_temperature_type = BaseTemperatureModel<TemperatureIndependent>;
    using fracture_type = NoFracture;
    using thermal_type = typename base_temperature_type::thermal_type;

    using base_type::base_type;
    using base_type::operator();
    using base_temperature_type::operator();
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

    using model_type = LPS;
    using base_model = typename base_type::base_model;
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
    using base_temperature_type = typename base_type::base_temperature_type;
    using base_model = typename base_type::base_model;
    using fracture_type = typename base_type::fracture_type;

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
    using base_model = typename base_type::base_model;
    using fracture_type = typename base_type::fracture_type;
    using thermal_type = base_type::thermal_type;

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
