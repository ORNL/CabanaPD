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
template <typename FunctorK, typename FunctorG>
struct MechanicsModel<LPS, Elastic, FunctorK, FunctorG> : public BaseForceModel
{
    using base_type = BaseForceModel;
    using thermal_type = TemperatureIndependent;
    // Tags for creating particle fields and dispatch to force iteration.
    using model_tag = LPS;
    using force_tag = LPS;

    using base_type::force_horizon;

    int influence_type;
    InfluenceFunctionTag influence_tag;

    PropertyWrapper<FunctorK> K;
    PropertyWrapper<FunctorG> G;

    MechanicsModel( LPS tag, Elastic mech, const double _force_horizon,
                    const double _K, const double _G, const int _influence = 0 )
        : MechanicsModel( tag, mech, _force_horizon, ConstantProperty( _K ),
                          ConstantProperty( _G ), _influence )
    {
    }

    MechanicsModel( LPS tag, const double _force_horizon, const double _K,
                    const double _G, const int _influence = 0 )
        : MechanicsModel( tag, Elastic{}, _force_horizon,
                          ConstantProperty( _K ), ConstantProperty( _G ),
                          _influence )
    {
    }

    MechanicsModel( LPS tag, const double _force_horizon, const FunctorK _K,
                    const FunctorG _G, const int _influence = 0 )
        : MechanicsModel( tag, Elastic{}, _force_horizon, _K, _G, _influence )
    {
    }

    MechanicsModel( LPS, Elastic, const double _force_horizon,
                    const FunctorK _K, const FunctorG _G,
                    const int _influence = 0 )
        : base_type( _force_horizon )
        , influence_type( _influence )
        , K( PropertyWrapper( _K ) )
        , G( PropertyWrapper( _G ) )
    {
        if ( influence_type > 1 || influence_type < 0 )
            log_err( std::cout, "Influence function type must be 0 or 1." );
    }

    // Constructor to average from existing models.
    template <typename ModelType1, typename ModelType2>
    MechanicsModel( const ModelType1& model1, const ModelType2& model2 )
        : base_type( model1, model2 )
        , K( AveragePropertyWrapper( model1.K, model2.K ) )
        , G( AveragePropertyWrapper( model1.G, model2.G ) )
    {
        influence_type = model1.influenceType();
        if ( model2.influenceType() != model1.influenceType() )
            log_err( std::cout,
                     "Influence function type for each model must match for "
                     "multi-material systems" );
    }

    KOKKOS_FUNCTION
    auto influenceType() const { return influence_type; }

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

    KOKKOS_INLINE_FUNCTION auto operator()( ForceCoeffTag, const int i,
                                            const int j, const double s,
                                            const double xi, const double vol,
                                            const double m_i, const double m_j,
                                            const double theta_i,
                                            const double theta_j ) const
    {
        auto influence = ( *this )( influence_tag, xi );
        double theta_coeff_i = 3.0 * K( i ) - 5.0 * G( i );
        double theta_coeff_j = 3.0 * K( j ) - 5.0 * G( j );
        double s_coeff_i = 15.0 * G( i );
        double s_coeff_j = 15.0 * G( j );

        return ( theta_coeff_i * theta_i / m_i + theta_coeff_j * theta_j / m_j +
                 s * ( s_coeff_i / m_i + s_coeff_j / m_j ) ) *
               influence * xi * vol;
    }

    KOKKOS_INLINE_FUNCTION
    auto operator()( EnergyTag, const int i, const int, const double s,
                     const double xi, const double vol, const double m_i,
                     const double theta_i, const double num_bonds ) const
    {
        auto influence = ( *this )( influence_tag, xi );

        double theta_coeff_i = 3.0 * K( i ) - 5.0 * G( i );
        double s_coeff_i = 15.0 * G( i );
        return 1.0 / num_bonds * 0.5 * theta_coeff_i / 3.0 *
                   ( theta_i * theta_i ) +
               0.5 * ( s_coeff_i / m_i ) * influence * s * s * xi * xi * vol;
    }
};

template <typename FunctorK, typename FunctorG>
struct MechanicsModel<LinearLPS, Elastic, FunctorK, FunctorG>
    : public MechanicsModel<LPS, Elastic, FunctorK, FunctorG>
{
    using base_type = MechanicsModel<LPS, Elastic, FunctorK, FunctorG>;
    // Tag to dispatch to force iteration.
    using force_tag = LinearLPS;

    using base_type::base_type;
    using base_type::operator();

    template <typename... Args>
    MechanicsModel( LinearLPS, Args&&... args )
        : base_type( typename base_type::model_tag{},
                     std::forward<Args>( args )... )
    {
    }
};

template <typename ModelType>
MechanicsModel( ModelType, const double force_horizon, const double K,
                const double G, const int _influence = 0 )
    -> MechanicsModel<ModelType, Elastic, ConstantProperty, ConstantProperty>;
template <typename ModelType>
MechanicsModel( ModelType, Elastic, const double force_horizon, const double K,
                const double G, const int _influence = 0 )
    -> MechanicsModel<ModelType, Elastic, ConstantProperty, ConstantProperty>;

template <typename ModelType, typename FunctorK, typename FunctorG>
MechanicsModel( ModelType, const double force_horizon, const FunctorK K,
                const FunctorG G, const int _influence = 0 )
    -> MechanicsModel<ModelType, Elastic, FunctorK, FunctorG>;
template <typename ModelType, typename FunctorK, typename FunctorG>
MechanicsModel( ModelType, Elastic, const double force_horizon,
                const FunctorK K, const FunctorG G, const int _influence = 0 )
    -> MechanicsModel<ModelType, Elastic, FunctorK, FunctorG>;

} // namespace CabanaPD

#endif
