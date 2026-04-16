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

#include <CabanaPD_Indexing.hpp>
#include <CabanaPD_Output.hpp>

namespace CabanaPD
{

namespace Impl
{
template <std::size_t current, typename FunctorType, typename ParameterPackType,
          typename... ARGS,
          std::enable_if_t<current<ParameterPackType::size - 1, int> = 0>
              KOKKOS_INLINE_FUNCTION auto
                  recursion_functor_for_index_in_pack_with_args(
                      FunctorType const& functor, std::size_t index,
                      ParameterPackType& parameterPack, ARGS... args )
{
    if ( index == current )
    {
        return functor( Cabana::get<current>( parameterPack ), args... );
    }
    else
    {
        return recursion_functor_for_index_in_pack_with_args<current + 1>(
            functor, index, parameterPack, args... );
    }
}

template <std::size_t current, typename FunctorType, typename ParameterPackType,
          typename... ARGS,
          std::enable_if_t<current == ParameterPackType::size - 1, int> = 0>
KOKKOS_INLINE_FUNCTION auto recursion_functor_for_index_in_pack_with_args(
    FunctorType const& functor, std::size_t index,
    ParameterPackType& parameterPack, ARGS... args )
{
    if ( index == current )
    {
        return functor( Cabana::get<current>( parameterPack ), args... );
    }
    else
    {
        Kokkos::abort( "Requested index not contained in ParameterPack" );
        return functor( Cabana::get<current>( parameterPack ), args... );
    }
}

template <typename FunctorType, typename ParameterPackType, typename... ARGS>
KOKKOS_INLINE_FUNCTION auto run_functor_for_index_in_pack_with_args(
    FunctorType const& functor, std::size_t index,
    ParameterPackType& parameterPack, ARGS... args )
{
    return recursion_functor_for_index_in_pack_with_args<0>(
        functor, index, parameterPack, args... );
}

struct IdentityFunctor
{
    template <typename Model, typename... ARGS>
    KOKKOS_FUNCTION auto operator()( Model& model, ARGS... args ) const
    {
        return model( args... );
    }
};

template <typename IfType, typename ElseType, bool Condition>
struct IfElseType
{
    static_assert( !Condition, "IfElseType: We should never end up here" );
    static_assert( Condition, "IfElseType: We should never end up here" );
};

template <typename IfType, typename ElseType>
struct IfElseType<IfType, ElseType, true>
{
    using type = IfType;
};

template <typename IfType, typename ElseType>
struct IfElseType<IfType, ElseType, false>
{
    using type = ElseType;
};

template <typename ParameterPackType, typename Sequence>
struct CheckTemperatureDependence
{
    static_assert( !std::is_same_v<Sequence, Sequence>,
                   "CheckTemperatureDependence: We should never end up here" );
};

template <typename ParameterPackType, std::size_t... Indices>
struct CheckTemperatureDependence<ParameterPackType,
                                  std::index_sequence<Indices...>>
{
    using type =
        typename IfElseType<TemperatureDependent, TemperatureIndependent,
                            std::disjunction_v<std::is_same<
                                typename ParameterPackType::template value_type<
                                    Indices>::thermal_type,
                                TemperatureDependent>...>>::type;
};

template <typename ParameterPackType, typename Sequence>
struct CheckFractureModel
{
    static_assert( !std::is_same_v<Sequence, Sequence>,
                   "CheckFractureModel: We should never end up here" );
};

template <typename ParameterPackType, std::size_t... Indices>
struct CheckFractureModel<ParameterPackType, std::index_sequence<Indices...>>
{
    using type =
        typename IfElseType<Fracture, NoFracture,
                            std::disjunction_v<std::is_same<
                                typename ParameterPackType::template value_type<
                                    Indices>::fracture_type,
                                Fracture>...>>::type;
};

template <typename FractureType, typename ParameterPackType, unsigned Index,
          bool Condition>
struct FirstModelWithFractureTypeImpl
{
    static_assert(
        !Condition,
        "FirstModelWithFractureTypeImpl: We should never end up here" );
    static_assert(
        Condition,
        "FirstModelWithFractureTypeImpl: We should never end up here" );
};

template <typename FractureType, typename ParameterPackType, unsigned Index>
struct FirstModelWithFractureTypeImpl<FractureType, ParameterPackType, Index,
                                      true>
{
    using type = typename ParameterPackType::template value_type<Index>;
};

template <typename FractureType, typename ParameterPackType, unsigned Index>
struct FirstModelWithFractureTypeImpl<FractureType, ParameterPackType, Index,
                                      false>
{
    using type = typename FirstModelWithFractureTypeImpl<
        FractureType, ParameterPackType, Index + 1,
        std::is_same_v<typename ParameterPackType::template value_type<
                           Index + 1>::fracture_type,
                       FractureType>>::type;
};

template <typename FractureType, typename ParameterPackType>
struct FirstModelWithFractureType
{
    using type = typename FirstModelWithFractureTypeImpl<
        FractureType, ParameterPackType, 0,
        std::is_same_v<
            typename ParameterPackType::template value_type<0>::fracture_type,
            FractureType>>::type;
};

template <typename BaseModelPackType, typename IndexingType, size_t Index>
auto generateSingleModelCombination( BaseModelPackType const& baseModels )
{
    using type = typename BaseModelPackType::template value_type<
        IndexingType::template getInverseIndexPair<Index>().first>;

    auto firstModel =
        Cabana::get<IndexingType::template getInverseIndexPair<Index>().first>(
            baseModels );

    auto secondModel =
        Cabana::get<IndexingType::template getInverseIndexPair<Index>().second>(
            baseModels );
    return type{ firstModel, secondModel };
}

template <typename BaseModelPackType, typename IndexingType, size_t... Indices>
auto generateAllModelCombinationsForIndexing(
    BaseModelPackType const& baseModels, IndexingType,
    std::index_sequence<Indices...> )
{
    return Cabana::makeParameterPack(
        generateSingleModelCombination<BaseModelPackType, IndexingType,
                                       Indices>( baseModels )... );
}

// Wrap multiple models in a single object.
template <typename MaterialType, typename Indexing, typename ParameterPackType,
          typename OutsideRangeFunctorType, typename Sequence>
struct ForceModelsImpl
{
    static_assert( !std::is_same_v<Sequence, Sequence>,
                   "ForceModelsImpl: We should never end up here" );
};

template <typename MaterialType, typename Indexing, typename ParameterPackType,
          typename OutsideRangeFunctorType, std::size_t... Indices>
struct ForceModelsImpl<MaterialType, Indexing, ParameterPackType,
                       OutsideRangeFunctorType, std::index_sequence<Indices...>>
{
    using material_type = MultiMaterial;
    using needs_update = std::true_type;

    static_assert(
        (std::conjunction_v<
            std::is_same<typename ParameterPackType::template value_type<
                             Indices>::model_tag,
                         typename ParameterPackType::template value_type<
                             0>::model_tag>...>),
        "All models need the same base model" );

    // TODO this might be softened in the future
    static_assert(
        (std::conjunction_v<
            std::is_same<typename ParameterPackType::template value_type<
                             Indices>::force_tag,
                         typename ParameterPackType::template value_type<
                             0>::force_tag>...>),
        "All forces need the same base type" );

    using model_tag =
        typename ParameterPackType::template value_type<0>::model_tag;

    static_assert( is_symmetric<model_tag>::value ==
                       is_symmetric<Indexing>::value,
                   "Symmetry of model and indexing must match" );

    using fracture_type = typename CheckFractureModel<
        ParameterPackType,
        std::make_index_sequence<ParameterPackType::size>>::type;
    using thermal_type = typename CheckTemperatureDependence<
        ParameterPackType,
        std::make_index_sequence<ParameterPackType::size>>::type;

    using force_tag =
        typename FirstModelWithFractureType<fracture_type,
                                            ParameterPackType>::type::force_tag;

    ForceModelsImpl( MaterialType t, Indexing const& i,
                     ParameterPackType const& m,
                     OutsideRangeFunctorType const& o )
        : type( t )
        , indexing( i )
        , models( m )
        , outsideRangeFunctor( o )
    {
        setForceHorizon();
    }

    auto cutoff() const { return force_horizon; }

    void updateBonds( const int num_local, const int max_neighbors )
    {
        ( Cabana::get<Indices>( models ).updateBonds( num_local,
                                                      max_neighbors ),
          ... );
    }

    KOKKOS_INLINE_FUNCTION int getIndex( const int i, const int j ) const
    {
        return indexing( type( i ), type( j ) );
    }

    // Extract model index and hide template indexing.
    template <typename Tag, typename... Args>
    KOKKOS_INLINE_FUNCTION auto operator()( Tag tag, const int i, const int j,
                                            Args... args ) const
    {
        using commonReturnType = typename std::invoke_result_t<
            typename ParameterPackType::template value_type<0>, Tag, const int,
            const int, Args...>;

        auto t = getIndex( i, j );
        // Call individual model.
        // if inside the pack
        if ( static_cast<unsigned>( t ) < ParameterPackType::size )
            return run_functor_for_index_in_pack_with_args(
                IdentityFunctor{}, t, models, tag, i, j, args... );
        else
            return outsideRangeFunctor.template operator()<commonReturnType>(
                tag, i, j, args... );
    }

    template <typename ParticleType>
    void update( const ParticleType& particles )
    {
        type = particles.sliceType();

        // Update any individual model particle fields.
        (
            [&]
            {
                if constexpr ( std::is_same_v<typename ParameterPackType::
                                                  template value_type<
                                                      Indices>::needs_update,
                                              std::true_type> )
                    Cabana::get<Indices>( models ).update( particles );
            }(),
            ... );
    }

    double force_horizon;
    MaterialType type;
    Indexing indexing;
    ParameterPackType models;
    OutsideRangeFunctorType outsideRangeFunctor;

  protected:
    void setForceHorizon()
    {
        // Enforce equal cutoff for now.
        const double tol = 1e-10;
        force_horizon = Cabana::get<1>( models ).force_horizon;

        if ( ( ( std::abs( Cabana::get<Indices>( models ).force_horizon -
                           force_horizon ) > tol ) ||
               ... ) )
        {
            log_err( std::cout, "Horizon for each model must match for "
                                "multi-material systems." );
        }
    }
};
} // namespace Impl

/******************************************************************************
  Multi-material models
******************************************************************************/
struct AbortIfOutsideRangeFunctor
{
    template <typename ReturnType, typename... Args>
    KOKKOS_FUNCTION ReturnType operator()( Args... ) const
    {
        Kokkos::abort( "Functor outside of range requested" );
        return ReturnType{};
    }
};

template <typename MaterialType, typename IndexingType,
          typename ParameterPackType,
          typename OutsideRangeFunctorType = AbortIfOutsideRangeFunctor>
struct ForceModels : CabanaPD::Impl::ForceModelsImpl<
                         MaterialType, IndexingType, ParameterPackType,
                         OutsideRangeFunctorType,
                         std::make_index_sequence<ParameterPackType::size>>
{
    ForceModels( MaterialType t, IndexingType i, ParameterPackType const& m,
                 OutsideRangeFunctorType const& o = OutsideRangeFunctorType() )
        : CabanaPD::Impl::ForceModelsImpl<
              MaterialType, IndexingType, ParameterPackType,
              OutsideRangeFunctorType,
              std::make_index_sequence<ParameterPackType::size>>( t, i, m, o )
    {
    }
};

struct AverageTag
{
};

template <typename ParticleType, typename IndexingType, typename... ModelTypes>
auto createMultiForceModel( ParticleType particles, IndexingType indexing,
                            ModelTypes... m )
{
    static_assert(
        is_Indexing<IndexingType>,
        "Indexing requires is_Indexing trait to be usable in ForceModels" );
    auto type = particles.sliceType();
    return ForceModels( type, indexing, Cabana::makeParameterPack( m... ) );
}

template <typename ParticleType, typename... ModelTypes,
          std::enable_if_t<std::conjunction_v<
                               is_symmetric<typename ModelTypes::model_tag>...>,
                           int> = 0>
auto createMultiForceModel( ParticleType particles, AverageTag,
                            ModelTypes... m )
{
    using IndexingType = DiagonalIndexing<sizeof...( ModelTypes )>;
    IndexingType indexing;
    auto type = particles.sliceType();
    auto baseModels = Cabana::makeParameterPack( m... );
    return ForceModels(
        type, indexing,
        CabanaPD::Impl::generateAllModelCombinationsForIndexing(
            baseModels, indexing,
            std::make_index_sequence<IndexingType::NumTotalModels>{} ) );
}

template <typename ParticleType, typename... ModelTypes,
          std::enable_if_t<!std::conjunction_v<
                               is_symmetric<typename ModelTypes::model_tag>...>,
                           int> = 0>
auto createMultiForceModel( ParticleType particles, AverageTag,
                            ModelTypes... m )
{
    using IndexingType = FullIndexing<sizeof...( ModelTypes )>;
    IndexingType indexing;
    auto type = particles.sliceType();
    auto baseModels = Cabana::makeParameterPack( m... );
    return ForceModels(
        type, indexing,
        CabanaPD::Impl::generateAllModelCombinationsForIndexing(
            baseModels, indexing,
            std::make_index_sequence<IndexingType::NumTotalModels>{} ) );
}

} // namespace CabanaPD

#endif
