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
    using base_type = ContactModel;
    using base_model = base_type::base_model;
    using model_type = HertzianModel;
    using fracture_type = NoFracture;
    using thermal_type = TemperatureIndependent;

    using base_type::radius;
    double nu;   // Poisson's ratio
    double Rs;   // Equivalent radius
    double Es;   // Equivalent Young's modulus
    double e;    // Coefficient of restitution
    double beta; // Damping coefficient
    double coeff_h_n;
    double coeff_h_d;

    HertzianModel() {}
    HertzianModel( const double _radius, const double _extend, const double _nu,
                   const double _E, const double _e )
        : base_type( _radius, _extend )
        , nu( _nu )
    {
        Rs = 0.5 * radius;
        Es = _E / ( 2.0 * ( 1.0 - Kokkos::pow( nu, 2.0 ) ) );
        e = _e;
        double ln_e = Kokkos::log( e );
        beta = -ln_e / Kokkos::sqrt( Kokkos::pow( ln_e, 2.0 ) +
                                     Kokkos::pow( pi, 2.0 ) );

        // Derived constants.
        coeff_h_n = 4.0 / 3.0 * Es * Kokkos::sqrt( Rs );
        coeff_h_d = -2.0 * Kokkos::sqrt( 5.0 / 6.0 ) * beta;
    }

    KOKKOS_INLINE_FUNCTION
    auto normalForce( const double r, const double vn, const double vol,
                      const double rho ) const
    {
        // Contact "overlap"
        const double delta_n = ( r - 2.0 * radius );

        // Hertz normal force coefficient
        double coeff = 0.0;
        if ( delta_n < 0.0 )
        {
            coeff = Kokkos::min(
                0.0,
                -coeff_h_n * Kokkos::pow( Kokkos::abs( delta_n ), 3.0 / 2.0 ) );
        }
        coeff /= vol;

        // Damping force coefficient
        double Sn = 0.0;
        if ( delta_n < 0.0 )
            Sn = 2.0 * Es * Kokkos::sqrt( Rs * Kokkos::abs( delta_n ) );
        double ms = ( rho * vol ) / 2.0;
        coeff -= coeff_h_d * Kokkos::sqrt( Sn * ms ) * vn / vol;
        return coeff;
    }
    KOKKOS_INLINE_FUNCTION
    auto tangentialForce( const double r, const double vn, const double vol,
                          const double rho ) const
    {
    }
};

} // namespace CabanaPD

#endif
