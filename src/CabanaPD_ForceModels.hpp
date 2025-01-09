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

#ifndef FORCE_MODELS_H
#define FORCE_MODELS_H

#include <CabanaPD_Constants.hpp>
#include <CabanaPD_Types.hpp>

namespace CabanaPD
{
struct ForceCoeffTag
{
};
struct EnergyTag
{
};
struct CriticalStretchTag
{
};
struct ThermalStretchTag
{
};

struct BaseForceModel
{
    using material_type = SingleMaterial;
    double delta;

    BaseForceModel( const double _delta )
        : delta( _delta ){};

    // FIXME: use the first model cutoff for now.
    template <typename ModelType1, typename ModelType2>
    BaseForceModel( const ModelType1& model1, const ModelType2& )
    {
        delta = model1.delta;
    }

    auto cutoff() const { return delta; }

    // Only needed for models which store bond properties.
    void updateBonds( const int, const int ) {}
};

struct BaseNoFractureModel
{
    using fracture_type = NoFracture;
};

struct BaseFractureModel
{
    using fracture_type = Fracture;

    double G0;
    double s0;
    double bond_break_coeff;

    BaseFractureModel( const double _delta, const double _K, const double _G0,
                       const int influence_type = 1 )
        : G0( _G0 )
    {
        s0 = Kokkos::sqrt( 5.0 * G0 / 9.0 / _K / _delta ); // 1/xi
        if ( influence_type == 0 )
            s0 = Kokkos::sqrt( 8.0 * G0 / 15.0 / _K / _delta ); // 1

        bond_break_coeff = ( 1.0 + s0 ) * ( 1.0 + s0 );
    };

    // Constructor to work with plasticity.
    BaseFractureModel( const double _G0, const double _s0 )
        : G0( _G0 )
        , s0( _s0 )
    {
        bond_break_coeff = ( 1.0 + s0 ) * ( 1.0 + s0 );
    }

    // Average from existing models.
    template <typename ModelType1, typename ModelType2>
    BaseFractureModel( const ModelType1& model1, const ModelType2& model2 )
    {
        G0 = ( model1.G0 + model2.G0 ) / 2.0;
        s0 = Kokkos::sqrt( ( model1.s0 * model1.s0 * model1.K +
                             model2.s0 * model2.s0 * model2.K ) /
                           ( model1.K + model2.K ) );
        bond_break_coeff = ( 1.0 + s0 ) * ( 1.0 + s0 );
    }

    KOKKOS_INLINE_FUNCTION
    bool operator()( CriticalStretchTag, const int, const int, const double r,
                     const double xi ) const
    {
        return r * r >= bond_break_coeff * xi * xi;
    }
};

template <typename ThermalType, typename... TemperatureType>
struct BaseTemperatureModel;

template <>
struct BaseTemperatureModel<TemperatureIndependent>
{
    using thermal_type = TemperatureIndependent;

    // No-op for temperature.
    KOKKOS_INLINE_FUNCTION
    double operator()( ThermalStretchTag, const int, const int,
                       const double s ) const
    {
        return s;
    }
};

template <class MemorySpace>
class BasePlasticity
{
  protected:
    using memory_space = MemorySpace;
    using NeighborView = typename Kokkos::View<double**, memory_space>;
    NeighborView _s_p;

  public:
    // Must update later because number of neighbors not known at construction.
    void updateBonds( const int num_local, const int max_neighbors )
    {
        Kokkos::realloc( _s_p, num_local, max_neighbors );
        Kokkos::deep_copy( _s_p, 0.0 );
    }
};

struct AverageTag
{
};

// Wrap multiple models in a single object.
// TODO: this currently only supports bi-material systems.
template <typename MaterialType, typename... ModelType>
struct ForceModels
{
    using material_type = MultiMaterial;

    using tuple_type = std::tuple<ModelType...>;
    using first_model = typename std::tuple_element<0, tuple_type>::type;
    using model_type = typename first_model::model_type;
    using base_model = typename first_model::base_model;
    using thermal_type = typename first_model::thermal_type;
    using fracture_type = typename first_model::fracture_type;

    ForceModels( MaterialType t, const ModelType... m )
        : delta( 0.0 )
        , type( t )
        , models( std::make_tuple( m... ) )
    {
        setHorizon();
    }

    ForceModels( MaterialType t, const tuple_type m )
        : delta( 0.0 )
        , type( t )
        , models( m )
    {
        setHorizon();
    }

    void setHorizon()
    {
        std::apply( [this]( auto&&... m ) { ( this->maxDelta( m ), ... ); },
                    models );
    }

    template <typename Model>
    auto maxDelta( Model m )
    {
        if ( m.delta > delta )
            delta = m.delta;
    }

    auto cutoff() const { return delta; }

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
        return apply<0>( t, tag, i, j, std::forward<Args>( args )... );
    }

    template <std::size_t N, typename Tag, typename... Args>
    KOKKOS_INLINE_FUNCTION auto apply( const int t, Tag tag,
                                       Args... args ) const
    {
        // Call individual model.
        if ( N == t )
            return std::get<N>( models )( tag, std::forward<Args>( args )... );

        // Recurse to find the right index.
        if constexpr ( N + 1 < std::tuple_size_v<tuple_type> )
            return apply<N + 1>( t, tag, std::forward<Args>( args )... );
        else
            Kokkos::abort( "Invalid model index." );
    }

    auto horizon( const int ) { return delta; }
    auto maxHorizon() { return delta; }

    void update( const MaterialType _type ) { type = _type; }

    double delta;
    MaterialType type;
    tuple_type models;
};

template <typename ParticleType, typename... ModelType>
auto createMultiForceModel( ParticleType particles, ModelType... models )
{
    static_assert( std::tuple_size_v<std::tuple<ModelType...>> == 3,
                   "Only binary material systems supported." );

    auto type = particles.sliceType();
    using material_type = decltype( type );
    return ForceModels<material_type, ModelType...>( type, models... );
}

template <typename ParticleType, typename... ModelType>
auto createMultiForceModel( ParticleType particles, AverageTag,
                            ModelType... models )
{
    static_assert( std::tuple_size_v<std::tuple<ModelType...>> == 2,
                   "Only binary material systems supported." );

    auto tuple = std::make_tuple( models... );
    auto m1 = std::get<0>( tuple );
    auto m2 = std::get<1>( tuple );

    using first_model =
        typename std::tuple_element<0, std::tuple<ModelType...>>::type;
    auto m12 = std::make_tuple( first_model( m1, m2 ) );
    auto all_models = std::tuple_cat( tuple, m12 );

    auto type = particles.sliceType();
    using material_type = decltype( type );
    return ForceModels<material_type, ModelType..., first_model>( type,
                                                                  all_models );
}

template <typename TemperatureType>
struct BaseTemperatureModel<TemperatureDependent, TemperatureType>
{
    using thermal_type = TemperatureDependent;

    double alpha;
    double temp0;

    // Temperature field
    TemperatureType temperature;

    BaseTemperatureModel( const TemperatureType _temp, const double _alpha,
                          const double _temp0 )
        : alpha( _alpha )
        , temp0( _temp0 )
        , temperature( _temp ){};

    // Average from existing models.
    BaseTemperatureModel( const BaseTemperatureModel& model1,
                          const BaseTemperatureModel& model2 )
    {
        // FIXME: correct averaging
        alpha = ( model1.alpha + model2.alpha ) / 2.0;
        temp0 = ( model1.temp0 + model2.temp0 ) / 2.0;
    }

    void update( const TemperatureType _temp ) { temperature = _temp; }

    // Update stretch with temperature effects.
    KOKKOS_INLINE_FUNCTION
    double operator()( ThermalStretchTag, const int i, const int j,
                       const double s ) const
    {
        double temp_avg = 0.5 * ( temperature( i ) + temperature( j ) ) - temp0;
        return s - ( alpha * temp_avg );
    }
};

template <typename TemperatureType>
struct ThermalFractureModel
    : public BaseFractureModel,
      BaseTemperatureModel<TemperatureDependent, TemperatureType>
{
    using base_fracture_type = BaseFractureModel;
    using base_temperature_type =
        BaseTemperatureModel<TemperatureDependent, TemperatureType>;
    using typename base_fracture_type::fracture_type;
    using typename base_temperature_type::thermal_type;

    // Does not use the base bond_break_coeff.
    using base_fracture_type::s0;
    using base_temperature_type::alpha;
    using base_temperature_type::temp0;
    using base_temperature_type::temperature;

    // Does not use the base critical stretch.
    using base_temperature_type::operator();

    ThermalFractureModel( const double _delta, const double _K,
                          const double _G0, const TemperatureType _temp,
                          const double _alpha, const double _temp0,
                          const int influence_type = 1 )
        : base_fracture_type( _delta, _K, _G0, influence_type )
        , base_temperature_type( _temp, _alpha, _temp0 ){};

    KOKKOS_INLINE_FUNCTION
    bool operator()( CriticalStretchTag, const int i, const int j,
                     const double r, const double xi ) const
    {
        double temp_avg = 0.5 * ( temperature( i ) + temperature( j ) ) - temp0;
        double bond_break_coeff =
            ( 1.0 + s0 + alpha * temp_avg ) * ( 1.0 + s0 + alpha * temp_avg );
        return r * r >= bond_break_coeff * xi * xi;
    }
};

// This class stores temperature parameters needed for heat transfer, but not
// the temperature itself (stored instead in the static temperature class
// above).
struct BaseDynamicTemperatureModel
{
    using thermal_type = DynamicTemperature;

    double delta;

    double thermal_coeff;
    double kappa;
    double cp;
    bool constant_microconductivity;

    BaseDynamicTemperatureModel( const double _delta, const double _kappa,
                                 const double _cp,
                                 const bool _constant_microconductivity = true )
    {
        delta = _delta;
        kappa = _kappa;
        cp = _cp;
        const double d3 = _delta * _delta * _delta;
        thermal_coeff = 9.0 / 2.0 * _kappa / pi / d3;
        constant_microconductivity = _constant_microconductivity;
    }

    // Average from existing models.
    BaseDynamicTemperatureModel( const BaseDynamicTemperatureModel& model1,
                                 const BaseDynamicTemperatureModel& model2 )
    {
        delta = ( model1.delta + model2.delta ) / 2.0;
        kappa = ( model1.kappa + model2.kappa ) / 2.0;
        cp = ( model1.cp + model2.cp ) / 2.0;
        constant_microconductivity = model1.constant_microconductivity;
    }

    KOKKOS_INLINE_FUNCTION double microconductivity_function( double r ) const
    {
        if ( constant_microconductivity )
            return thermal_coeff;
        else
            return 4.0 * thermal_coeff * ( 1.0 - r / delta );
    }
};

template <typename PeridynamicsModelType, typename MechanicsModelType = Elastic,
          typename DamageType = Fracture,
          typename ThermalType = TemperatureIndependent, typename... DataTypes>
struct ForceModel;

} // namespace CabanaPD

#endif
