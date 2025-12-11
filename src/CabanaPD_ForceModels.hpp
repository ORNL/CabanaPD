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
struct WeightedVolumeTag
{
};
struct DilatationTag
{
};
struct InfluenceFunctionTag
{
};

struct BaseForceModel
{
    using material_type = SingleMaterial;
    double force_horizon;
    double K;

    BaseForceModel( const double _force_horizon, const double _K )
        : force_horizon( _force_horizon )
        , K( _K ){};

    // FIXME: use the first model cutoff for now.
    template <typename ModelType1, typename ModelType2>
    BaseForceModel( const ModelType1& model1, const ModelType2& model2 )
    {
        force_horizon = model1.force_horizon;
        K = ( model1.K + model2.K ) / 2.0;
    }

    auto cutoff() const { return force_horizon; }
    auto extend() const { return 0.0; }

    // Only needed for models which store bond properties.
    void updateBonds( const int, const int ) {}
};

struct BaseNoFractureModel
{
    using fracture_type = NoFracture;

    // This should only be used in multi-material models in which the current
    // model does not support failure.
    KOKKOS_INLINE_FUNCTION
    bool operator()( CriticalStretchTag, const int, const int, const double,
                     const double ) const
    {
        return false;
    }
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
    using neighbor_view = typename Kokkos::View<double**, memory_space>;
    neighbor_view _s_p;

  public:
    // Must update later because number of neighbors not known at construction.
    void updateBonds( const int num_local, const int max_neighbors )
    {
        Kokkos::realloc( _s_p, num_local, max_neighbors );
        Kokkos::deep_copy( _s_p, 0.0 );
    }
};

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

    double thermal_horizon;

    double thermal_coeff;
    double kappa;
    double cp;
    bool constant_microconductivity;

    BaseDynamicTemperatureModel( const double _thermal_horizon,
                                 const double _kappa, const double _cp,
                                 const bool _constant_microconductivity = true )
    {
        thermal_horizon = _thermal_horizon;
        kappa = _kappa;
        cp = _cp;
        const double d3 =
            _thermal_horizon * _thermal_horizon * _thermal_horizon;
        thermal_coeff = 9.0 / 2.0 * _kappa / pi / d3;
        constant_microconductivity = _constant_microconductivity;
    }

    KOKKOS_INLINE_FUNCTION double microconductivity_function( double r ) const
    {
        if ( constant_microconductivity )
            return thermal_coeff;
        else
            return 4.0 * thermal_coeff * ( 1.0 - r / thermal_horizon );
    }
};

template <typename PeridynamicsModelType, typename MechanicsModelType = Elastic,
          typename DamageType = Fracture,
          typename ThermalType = TemperatureIndependent, typename... DataTypes>
struct ForceModel;

} // namespace CabanaPD

#endif
