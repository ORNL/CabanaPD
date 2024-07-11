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
template <>
struct ForceModel<LPS, Elastic, NoFracture> : public BaseForceModel
{
    using base_type = BaseForceModel;
    using base_model = LPS;
    using fracture_type = NoFracture;
    using thermal_type = TemperatureIndependent;

    using base_type::delta;

    int influence_type;

    double K;
    double G;
    double theta_coeff;
    double s_coeff;

    ForceModel( const double _delta, const double _K, const double _G,
                const int _influence = 0 )
        : base_type( _delta )
        , influence_type( _influence )
        , K( _K )
        , G( _G )
    {
        theta_coeff = 3.0 * K - 5.0 * G;
        s_coeff = 15.0 * G;
    }

    KOKKOS_INLINE_FUNCTION double influence_function( double xi ) const
    {
        if ( influence_type == 1 )
            return 1.0 / xi;
        else
            return 1.0;
    }
};

template <>
struct ForceModel<LPS, Elastic, Fracture>
    : public ForceModel<LPS, Elastic, NoFracture>
{
    using base_type = ForceModel<LPS, Elastic, NoFracture>;
    using base_model = typename base_type::base_model;
    using fracture_type = Fracture;
    using thermal_type = base_type::thermal_type;

    using base_type::delta;
    using base_type::G;
    using base_type::influence_type;
    using base_type::K;
    using base_type::s_coeff;
    using base_type::theta_coeff;
    double G0;
    double s0;
    double bond_break_coeff;

    ForceModel( const double _delta, const double _K, const double _G,
                const double _G0, const int _influence = 0 )
        : base_type( _delta, _K, _G, _influence )
        , G0( _G0 )
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
struct ForceModel<LinearLPS, Elastic, NoFracture>
    : public ForceModel<LPS, Elastic, NoFracture>
{
    using base_type = ForceModel<LPS, Elastic, NoFracture>;
    using base_model = typename base_type::base_model;
    using fracture_type = typename base_type::fracture_type;
    using thermal_type = base_type::thermal_type;

    using base_type::base_type;

    using base_type::delta;
    using base_type::G;
    using base_type::influence_type;
    using base_type::K;
    using base_type::s_coeff;
    using base_type::theta_coeff;
};

template <>
struct ForceModel<LinearLPS, Elastic, Fracture>
    : public ForceModel<LPS, Elastic, Fracture>
{
    using base_type = ForceModel<LPS, Elastic, Fracture>;
    using base_model = typename base_type::base_model;
    using fracture_type = typename base_type::fracture_type;
    using thermal_type = base_type::thermal_type;

    using base_type::base_type;

    using base_type::delta;
    using base_type::G;
    using base_type::influence_type;
    using base_type::K;
    using base_type::s_coeff;
    using base_type::theta_coeff;

    using base_type::bond_break_coeff;
    using base_type::G0;
    using base_type::s0;
};

} // namespace CabanaPD

#endif
