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

#ifndef TYPES_H
#define TYPES_H

#include <type_traits>

#include <Cabana_Grid.hpp>

namespace CabanaPD
{
// Fracture tags.
struct NoFracture
{
};
struct Fracture
{
};
template <class>
struct is_fracture : public std::false_type
{
};
template <>
struct is_fracture<Fracture> : public std::true_type
{
};

// Mechanics tags.
struct Elastic
{
};
struct ElasticPerfectlyPlastic
{
};

// Model category tags.
struct Pair
{
};
struct State
{
};

// Thermal tags.
struct TemperatureIndependent
{
    using base_type = TemperatureIndependent;
};
struct TemperatureDependent
{
    using base_type = TemperatureDependent;
};
struct DynamicTemperature : public TemperatureDependent
{
    using base_type = TemperatureDependent;
};

//! Static type checkers.
template <class>
struct is_temperature_dependent : public std::false_type
{
};
template <>
struct is_temperature_dependent<TemperatureDependent> : public std::true_type
{
};
template <>
struct is_temperature_dependent<DynamicTemperature> : public std::true_type
{
};
template <class>
struct is_heat_transfer : public std::false_type
{
};
template <>
struct is_heat_transfer<DynamicTemperature> : public std::true_type
{
};
template <class>
struct is_temperature : public std::false_type
{
};
template <>
struct is_temperature<TemperatureIndependent> : public std::true_type
{
};
template <>
struct is_temperature<TemperatureDependent> : public std::true_type
{
};
template <>
struct is_temperature<DynamicTemperature> : public std::true_type
{
};

// Force model tags.
struct PMB
{
    using base_type = Pair;
    using base_model = PMB;
};
struct LinearPMB
{
    using base_type = Pair;
    using base_model = PMB;
};
struct LPS
{
    using base_type = State;
    using base_model = LPS;
};
struct LinearLPS
{
    using base_type = State;
    using base_model = LPS;
};

// Contact and DEM (contact without PD) tags.
struct Contact
{
    using base_type = Pair;
};
struct NoContact
{
    using base_model = std::false_type;
    using model_type = std::false_type;
    using thermal_type = TemperatureIndependent;
    using fracture_type = NoFracture;
};
template <class, class SFINAE = void>
struct is_contact : public std::false_type
{
};
template <typename ModelType>
struct is_contact<
    ModelType,
    typename std::enable_if<(
        std::is_same<typename ModelType::base_model, Contact>::value )>::type>
    : public std::true_type
{
};

template <class, class, class SFINAE = void>
struct either_contact
{
    using base_type = NoContact;
};
template <class Model1, class Model2>
struct either_contact<
    Model1, Model2,
    typename std::enable_if<( is_contact<Model1>::value ||
                              is_contact<Model2>::value )>::type>
{
    using base_type = Contact;
};

// Output tags.
struct BaseOutput
{
};
struct EnergyOutput
{
};
struct EnergyStressOutput
{
};

template <class>
struct is_output : public std::false_type
{
};
template <>
struct is_output<BaseOutput> : public std::true_type
{
};
template <>
struct is_output<EnergyOutput> : public std::true_type
{
};
template <>
struct is_output<EnergyStressOutput> : public std::true_type
{
};

template <class>
struct is_energy_output : public std::false_type
{
};
template <>
struct is_energy_output<EnergyOutput> : public std::true_type
{
};
template <>
struct is_energy_output<EnergyStressOutput> : public std::true_type
{
};

template <class>
struct is_stress_output : public std::false_type
{
};
template <>
struct is_stress_output<EnergyStressOutput> : public std::true_type
{
};

// Particle init types.
template <class>
struct is_particle_init : public std::false_type
{
};
template <>
struct is_particle_init<Cabana::InitUniform> : public std::true_type
{
};
template <>
struct is_particle_init<Cabana::InitRandom> : public std::true_type
{
};

} // namespace CabanaPD
#endif
