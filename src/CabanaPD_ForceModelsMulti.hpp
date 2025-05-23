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

#ifndef FORCE_MODELS_MULTI_H
#define FORCE_MODELS_MULTI_H

#include <CabanaPD_Output.hpp>

namespace CabanaPD
{

/******************************************************************************
  Multi-material models
******************************************************************************/
struct AverageTag
{
};

// Wrap multiple models in a single object.
// TODO: this currently only supports bi-material systems.
template <typename MaterialType, typename ModelType1, typename ModelType2,
          typename ModelType12>
struct ForceModels
{
    using material_type = MultiMaterial;

    using first_model = ModelType1;
    using model_type = typename first_model::model_type;
    using base_model = typename first_model::base_model;
    using thermal_type = typename first_model::thermal_type;
    using fracture_type = typename first_model::fracture_type;

    static_assert( ( std::is_same<typename ModelType1::thermal_type,
                                  TemperatureIndependent>::value ||
                     std::is_same<typename ModelType2::thermal_type,
                                  TemperatureIndependent>::value ||
                     std::is_same<typename ModelType12::thermal_type,
                                  TemperatureIndependent>::value ),
                   "Thermomechanics does not yet support multiple materials!" );

    ForceModels( MaterialType t, const ModelType1 m1, ModelType2 m2,
                 ModelType12 m12 )
        : delta( 0.0 )
        , type( t )
        , model1( m1 )
        , model2( m2 )
        , model12( m12 )
    {
        setHorizon();
    }

    void setHorizon()
    {
        maxDelta( model1 );
        maxDelta( model2 );
        maxDelta( model12 );
    }

    template <typename Model>
    auto maxDelta( Model m )
    {
        if ( m.delta > delta )
        {
            delta = m.delta;
            // Enforce equal cutoff for now.
            if ( m.delta != delta )
                log_err( std::cout, "Horizon for each model must match for "
                                    "multi-material systems." );
        }
    }

    auto cutoff() const { return delta; }

    auto bulkModulus() const { return model1.K; }
    auto bulkModulus( const int i ) const
    {
        if ( i == 0 )
            return model1.K;
        else if ( i == 1 )
            return model2.K;
        else if ( i == 2 )
            return model12.K;
    }
    // FIXME
    auto shearModulus() const
    {
        if constexpr ( std::is_same<typename ModelType1::base_model,
                                    LPS>::value )
            return model1.G;
        else
            return 0.0;
    }

    void updateBonds( const int num_local, const int max_neighbors )
    {
        model1.updateBonds( num_local, max_neighbors );
        model2.updateBonds( num_local, max_neighbors );
        model12.updateBonds( num_local, max_neighbors );
    }

    KOKKOS_INLINE_FUNCTION int getIndex( const int i, const int j ) const
    {
        const int type_i = type( i );
        const int type_j = type( j );
        // FIXME: only for binary.
        if ( type_i == type_j )
            return type_i;
        else
            return 2;
    }

    // Extract model index and hide template indexing.
    template <typename Tag, typename... Args>
    KOKKOS_INLINE_FUNCTION auto operator()( Tag tag, const int i, const int j,
                                            Args... args ) const
    {
        auto t = getIndex( i, j );
        // Call individual model.
        if ( t == 0 )
            return model1( tag, i, j, std::forward<Args>( args )... );
        else if ( t == 1 )
            return model2( tag, i, j, std::forward<Args>( args )... );
        else if ( t == 2 )
            return model12( tag, i, j, std::forward<Args>( args )... );
        else
            Kokkos::abort( "Invalid model index." );
    }

    auto horizon( const int ) { return delta; }
    auto maxHorizon() { return delta; }

    void update( const MaterialType _type ) { type = _type; }

    double delta;
    MaterialType type;
    ModelType1 model1;
    ModelType2 model2;
    ModelType12 model12;
};

template <typename ParticleType, typename ModelType1, typename ModelType2,
          typename ModelType12>
auto createMultiForceModel( ParticleType particles, ModelType1 m1,
                            ModelType2 m2, ModelType12 m12 )
{
    auto type = particles.sliceType();
    return ForceModels( type, m1, m2, m12 );
}

template <typename ParticleType, typename ModelType1, typename ModelType2>
auto createMultiForceModel( ParticleType particles, AverageTag, ModelType1 m1,
                            ModelType2 m2 )
{
    ModelType1 m12( m1, m2 );

    auto type = particles.sliceType();
    return ForceModels( type, m1, m2, m12 );
}

} // namespace CabanaPD

#endif
