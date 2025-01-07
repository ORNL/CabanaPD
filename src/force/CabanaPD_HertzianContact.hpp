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

#ifndef CONTACTMODEL_HERTZIAN_H
#define CONTACTMODEL_HERTZIAN_H

#include <Kokkos_Core.hpp>

#include <CabanaPD_Force.hpp>
#include <CabanaPD_Input.hpp>
#include <CabanaPD_Output.hpp>

namespace CabanaPD
{
struct HertzianModel : public ContactModel
{
    // FIXME: This is for use as the primary force model.
    using base_model = PMB;
    using fracture_type = NoFracture;
    using thermal_type = TemperatureIndependent;

    using ContactModel::Rc; // Contact horizon (should be > 2*radius)

    double nu;     // Poisson's ratio
    double radius; // Actual radius
    double Rs;     // Equivalent radius
    double Es;     // Equivalent Young's modulus
    double e;      // Coefficient of restitution
    double beta;   // Damping coefficient

    HertzianModel( const double _Rc, const double _radius, const double _nu,
                   const double _E, const double _e )
        : ContactModel( 1.0, _Rc )
    {
        set_param( _radius, _nu, _E, _e );
    }

    void set_param( const double _radius, const double _nu, const double _E,
                    const double _e )
    {
        nu = _nu;
        radius = _radius;
        Rs = 0.5 * radius;
        Es = _E / ( 2.0 * Kokkos::pow( 1.0 - nu, 2.0 ) );
        e = _e;
        double ln_e = Kokkos::log( e );
        beta = -ln_e / Kokkos::sqrt( Kokkos::pow( ln_e, 2.0 ) +
                                     Kokkos::pow( pi, 2.0 ) );
    }
};

template <>
struct is_contact<HertzianModel> : public std::true_type
{
};

} // namespace CabanaPD

#endif
