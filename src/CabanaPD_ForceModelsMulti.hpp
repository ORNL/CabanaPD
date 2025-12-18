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
    using force_tag = typename first_model::force_tag;
    using model_tag = typename first_model::model_tag;
    using thermal_type = typename first_model::thermal_type;
    using fracture_type = typename first_model::fracture_type;

    ForceModels( MaterialType t, const ModelType1 m1, ModelType2 m2,
                 ModelType12 m12 )
        : type( t )
        , model1( m1 )
        , model2( m2 )
        , model12( m12 )
    {
        setForceHorizon();
    }

    auto cutoff() const { return force_horizon; }

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

    // This is only for LPS force/energy, currently the only cases that require
    // type information. When running models individually, the SingleMaterial
    // tag is used in the model directly; here it is replaced with the
    // MultiMaterial tag instead.
    template <typename Tag, typename... Args>
    KOKKOS_INLINE_FUNCTION auto operator()( Tag tag, SingleMaterial,
                                            const int i, const int j,
                                            Args... args ) const
    {
        const int type_i = type( i );
        const int type_j = type( j );

        auto t = getIndex( i, j );
        MultiMaterial mtag;
        // Call individual model.
        if ( t == 0 )
            return model1( tag, mtag, type_i, type_j,
                           std::forward<Args>( args )... );
        else if ( t == 1 )
            return model2( tag, mtag, type_i, type_j,
                           std::forward<Args>( args )... );
        else if ( t == 2 )
            return model12( tag, mtag, type_i, type_j,
                            std::forward<Args>( args )... );
        else
            Kokkos::abort( "Invalid model index." );
    }

    template <typename ParticleType>
    void update( const ParticleType& particles )
    {
        type = particles.sliceType();

        // Update any individual model particle fields.
        model1.update( particles );
        model2.update( particles );
        model12.update( particles );
    }

    double force_horizon;
    MaterialType type;
    ModelType1 model1;
    ModelType2 model2;
    ModelType12 model12;

  protected:
    void setForceHorizon()
    {
        // Enforce equal cutoff for now.
        force_horizon = model1.force_horizon;
        checkHorizon( model2 );
        checkHorizon( model12 );
    }

    template <typename Model>
    auto checkHorizon( Model m, const double tol = 1e-10 )
    {
        if ( std::abs( m.force_horizon - force_horizon ) > tol )
            log_err( std::cout, "Horizon for each model must match for "
                                "multi-material systems." );
    }
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
