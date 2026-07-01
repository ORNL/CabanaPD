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

#ifndef CONTACTMODEL_HERTZIANJKR_H
#define CONTACTMODEL_HERTZIANJKR_H

#include <Kokkos_Core.hpp>

#include <CabanaPD_Force.hpp>
#include <CabanaPD_Input.hpp>
#include <CabanaPD_Output.hpp>

namespace CabanaPD
{
struct HertzianJKRModel : public ContactModel
{
    using base_type = ContactModel;
    using fracture_tag = NoFracture;
    using thermal_tag = TemperatureIndependent;
    // Tag to dispatch to force iteration.
    using force_tag = HertzianModel;

    using base_type::radius;
    double nu; // Poisson's ratio
    double E;  // Young's modulus
    double mu; // Coloumb friction coefficient

    double Rs; // Equivalent radius
    double Es; // Equivalent Young's modulus
    double Gs; // Equivalent shear modulus

    double e;    // Coefficient of restitution
    double beta; // Damping coefficient
    double coeff_h_n;
    double coeff_h_d;

    double Gamma;      // Surface energy
    double a0;         // Equilibrium contact radius
    double delta_tear; // Maximum separation distance to break contact
    double fc;         // Maximum cohesion (pull-off) force

    HertzianJKRModel() {}
    HertzianJKRModel( const double _radius, const double _extend,
                      const double _nu, const double _E, const double _e,
                      const double _gamma, const double _mu )
        : base_type( _radius, _extend )
        , nu( _nu )
        , E( _E )
        , mu( _mu )
    {
        Rs = 0.5 * radius;
        Es = _E / ( 2.0 * ( 1.0 - Kokkos::pow( nu, 2.0 ) ) );
        Gs = _E / ( 4.0 * ( 2 - nu ) * ( 1 + nu ) );
        e = _e;
        double ln_e = Kokkos::log( e );
        beta = -ln_e / Kokkos::sqrt( Kokkos::pow( ln_e, 2.0 ) +
                                     Kokkos::pow( pi, 2.0 ) );

        // Derived constants.
        coeff_h_n = 4.0 / 3.0 * Es / Rs;
        coeff_h_d = -2.0 * Kokkos::sqrt( 5.0 / 6.0 ) * beta;

        // JKR cohesion model
        Gamma = 2.0 * _gamma;
        a0 = Kokkos::pow( 9.0 / 2.0 * Gamma * pi / Es * Kokkos::pow( Rs, 2.0 ),
                          1.0 / 3.0 );

        delta_tear = -1.0 / 2.0 * Kokkos::pow( 6.0, -1.0 / 3.0 ) *
                     Kokkos::pow( a0, 2.0 ) / Rs;

        fc = 3.0 / 2.0 * pi * Rs * Gamma;
    }

    KOKKOS_INLINE_FUNCTION
    auto normalForce( const double r, const double vn, const double vol,
                      const double rho ) const
    {
        // Contact "overlap"
        const double delta_n = ( 2.0 * radius - r );

        // HertzJKR normal force coefficient
        double coeff = 0.0;
        if ( delta_n >= delta_tear )
        {
            auto a = patchRadius( delta_n );
            auto a3 = Kokkos::pow( a, 3.0 );
            coeff =
                -1.0 * ( coeff_h_n * a3 -
                         Kokkos::pow( 8.0 * Gamma * pi * Es * a3, 1.0 / 2.0 ) );

            // Damping force coefficient
            double Sn = 2.0 * Es * a;
            double ms = ( rho * vol ) / 2.0;

            coeff -= coeff_h_d * Kokkos::sqrt( Sn * ms ) * vn;

            if ( delta_n <= 0.0 && vn <= 0.0 )
            {
                coeff = 0.0;
            }
        }
        coeff /= vol;

        return coeff;
    }
    KOKKOS_INLINE_FUNCTION
    void tangentialForce( const double vtx, const double vty, const double vtz,
                          const double fn, double& ftx, double& fty,
                          double& ftz ) const
    {
        // Coloumb sliding (kinetic) friction
        auto mu_c = [&]( const double v )
        { return Kokkos::abs( mu * Kokkos::tanh( 10 * v ) ); };

        auto vt = Kokkos::hypot( vtx, vty, vtz );
        auto fna = Kokkos::abs( fn );

        ftx = ( vt > 0 ) ? mu_c( vtx ) * fna * vtx / vt : 0.0;
        fty = ( vt > 0 ) ? mu_c( vty ) * fna * vty / vt : 0.0;
        ftz = ( vt > 0 ) ? mu_c( vtz ) * fna * vtz / vt : 0.0;
    }

    // The normal force for JKR theory is given in terms of the contact
    // patch radius, (a), which obeys a different relationship with the
    // normal contact overlap, (delta_n), than from ordinary Hertz theory
    // alone.
    //
    // We need to calculate (a) from the more readily available (delta_n)
    // Here we use the approach from PFC:
    // https://docs.itascacg.com/pfc700/common/contactmodel/jkr/doc/manual/cmjkr.html
    // which is itself taken from E. Parteli (2014).

    KOKKOS_INLINE_FUNCTION
    double patchRadius( const double delta_n ) const
    {
        double c0 = Kokkos::pow( Rs * delta_n, 2.0 );
        double c1 = -4.0 * ( 1.0 - Kokkos::pow( nu, 2.0 ) ) * pi * Gamma *
                    Kokkos::pow( Rs, 2.0 ) / E;
        double c2 = -2.0 * Rs * delta_n;
        double P = -Kokkos::pow( c2, 2.0 ) / 12.0 - c0;
        double Q = -Kokkos::pow( c2, 3.0 ) / 108.0 + c0 * c2 / 3.0 -
                   Kokkos::pow( c1, 2.0 ) / 8.0;
        double U = Kokkos::pow(
            -Q / 2.0 + Kokkos::pow(
                           Kokkos::max( 0.0, Kokkos::pow( Q, 2.0 ) / 4.0 +
                                                 Kokkos::pow( P, 3.0 ) / 27.0 ),
                           1.0 / 2.0 ),
            1.0 / 3.0 );

        double s = 0.0;
        if ( P != 0.0 )
        {
            s = -5.0 / 6.0 * c2 + U - P / ( 3.0 * U );
        }
        else
        {
            s = -5.0 / 6.0 * c2 - Kokkos::pow( Q, 1.0 / 3.0 );
        }
        double w = Kokkos::pow( Kokkos::max( 0.0, c2 + 2.0 * s ), 1.0 / 2.0 );
        double lambda = c1 / ( 2.0 * w + 1e-14 );

        // The final value for (a)
        return 1.0 / 2.0 *
               ( w +
                 Kokkos::pow( Kokkos::max( 0.0, Kokkos::pow( w, 2.0 ) -
                                                    4.0 * ( c2 + s + lambda ) ),
                              1.0 / 2.0 ) );
    }
};

} // namespace CabanaPD

#endif
