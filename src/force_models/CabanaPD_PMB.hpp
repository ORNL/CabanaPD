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

template <typename FunctorType>
struct MechanicsModel<PMB, Elastic, FunctorType> : public BaseForceModel
{
    using base_type = BaseForceModel;
    using mechanics_type = Elastic;

    // Tags for creating particle fields and dispatch to force iteration.
    using model_tag = PMB;
    using force_tag = PMB;

    using base_type::force_horizon;
    FunctorType K;
    double c_factor;

    MechanicsModel( PMB tag, Elastic mech, const double force_horizon,
                    const double _K )
        : MechanicsModel( tag, mech, force_horizon, ConstantProperty( _K ) )
    {
    }

    MechanicsModel( PMB tag, const double force_horizon, const double _K )
        : MechanicsModel( tag, force_horizon, ConstantProperty( _K ) )
    {
    }

    MechanicsModel( PMB tag, const double force_horizon, const FunctorType _K )
        : MechanicsModel( tag, Elastic{}, force_horizon, _K )
    {
    }

    MechanicsModel( PMB, Elastic, const double force_horizon,
                    const FunctorType _K )
        : base_type( force_horizon )
        , K( _K )
    {
        c_factor = 18.0 / ( pi * force_horizon * force_horizon * force_horizon *
                            force_horizon );
    }

    // Constructor to average from existing models.
    // FIXME: use the first model functional dependence for now.
    template <typename ModelType1, typename ModelType2>
    MechanicsModel( const ModelType1& model1, const ModelType2& model2 )
        : base_type( model1, model2 )
        , K( model1.K )
    {
    }

    KOKKOS_FUNCTION
    auto c( const int i ) const { return K( i ) * c_factor; }

    KOKKOS_FUNCTION
    int influenceType() const { return 1; }

    KOKKOS_INLINE_FUNCTION
    auto operator()( ForceCoeffTag, const int i, const int, const double s,
                     const double vol, const int = -1 ) const
    {
        return c( i ) * s * vol;
    }

    KOKKOS_INLINE_FUNCTION
    auto operator()( EnergyTag, const int i, const int, const double s,
                     const double xi, const double vol, const int = -1 ) const
    {
        // 0.25 factor is due to 1/2 from outside the integral and 1/2 from
        // the integrand (pairwise potential).
        return 0.25 * c( i ) * s * s * xi * vol;
    }
};

template <typename ModelType>
MechanicsModel( ModelType, Elastic, const double force_horizon, const double K )
    -> MechanicsModel<ModelType, Elastic, ConstantProperty>;

template <typename ModelType>
MechanicsModel( ModelType, const double force_horizon, const double K )
    -> MechanicsModel<ModelType, Elastic, ConstantProperty>;

template <typename ModelType, typename FunctorType>
MechanicsModel( ModelType, Elastic, const double force_horizon,
                const FunctorType K )
    -> MechanicsModel<ModelType, Elastic, FunctorType>;

template <typename MemorySpace, typename FunctorModulus, typename FunctorYield>
struct MechanicsModel<PMB, ElasticPerfectlyPlastic, FunctorModulus,
                      FunctorYield, MemorySpace>
    : public MechanicsModel<PMB, Elastic, FunctorModulus>,
      public BasePlasticity<MemorySpace>
{
    using base_type = MechanicsModel<PMB, Elastic, FunctorModulus>;
    using base_plasticity_type = BasePlasticity<MemorySpace>;

    using mechanics_type = ElasticPerfectlyPlastic;

    using base_plasticity_type::_s_p;
    FunctorYield s_Y;

    using base_plasticity_type::updateBonds;

    MechanicsModel( PMB tag, ElasticPerfectlyPlastic, MemorySpace,
                    const double force_horizon, const double K,
                    const double sigma_y )
        : base_type( tag, force_horizon, K )
        , base_plasticity_type()
        , s_Y( ConstantProperty( sigma_y / 3.0 / K ) )
    {
    }

    MechanicsModel( PMB tag, ElasticPerfectlyPlastic, MemorySpace,
                    const double force_horizon, const FunctorModulus K,
                    const FunctorYield _s_Y )
        : base_type( tag, force_horizon, K )
        , base_plasticity_type()
        , s_Y( _s_Y )
    {
    }

    // Constructor to average from existing models.
    // FIXME: use the first model functional dependence for now.
    template <typename ModelType1, typename ModelType2>
    MechanicsModel( const ModelType1& model1, const ModelType2& model2 )
        : base_type( model1, model2 )
        , base_plasticity_type()
        , s_Y( model1.s_Y )
    {
    }

    KOKKOS_INLINE_FUNCTION
    auto operator()( ForceCoeffTag, const int i, const int, const double s,
                     const double vol, const int n ) const
    {
        // Update bond plastic stretch.
        auto s_p = _s_p( i, n );
        // Yield in tension.
        if ( s >= s_p + s_Y( i ) )
            _s_p( i, n ) = s - s_Y( i );
        // Yield in compression.
        else if ( s <= s_p - s_Y( i ) )
            _s_p( i, n ) = s + s_Y( i );
        // else: Elastic (in between), do not modify.

        // Must extract again if in the plastic regime.
        s_p = _s_p( i, n );
        return base_type::c( i ) * ( s - s_p ) * vol;
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
        if ( s >= s_p + s_Y( i ) )
            stretch_term = s_Y( i ) * ( 2.0 * s - s_Y( i ) );
        // Yield in compression.
        else if ( s <= s_p - s_Y( i ) )
            stretch_term = s_Y( i ) * ( -2.0 * s - s_Y( i ) );
        else
            // Elastic (in between).
            stretch_term = s * s;

        // 0.25 factor is due to 1/2 from outside the integral and 1/2 from
        // the integrand (pairwise potential).
        return 0.25 * base_type::c( i ) * stretch_term * xi * vol;
    }
};

template <typename FunctorType>
struct MechanicsModel<LinearPMB, Elastic, FunctorType>
    : public MechanicsModel<PMB, Elastic, FunctorType>
{
    using base_type = MechanicsModel<PMB, Elastic, FunctorType>;
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
    -> MechanicsModel<ModelType, ElasticPerfectlyPlastic, ConstantProperty,
                      ConstantProperty, MemorySpace>;

template <typename ModelType, typename FunctorModulus, typename FunctorYield,
          typename MemorySpace>
MechanicsModel( ModelType, ElasticPerfectlyPlastic, MemorySpace,
                const double force_horizon, const FunctorModulus K,
                const FunctorYield sigma_y )
    -> MechanicsModel<ModelType, ElasticPerfectlyPlastic, FunctorModulus,
                      FunctorYield, MemorySpace>;

} // namespace CabanaPD

#endif
