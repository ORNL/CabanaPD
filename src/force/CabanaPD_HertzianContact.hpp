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

#ifndef CONTACTMODELS_H
#define CONTACTMODELS_H

#include <cmath>

#include <CabanaPD_Force.hpp>
#include <CabanaPD_Input.hpp>
#include <CabanaPD_Output.hpp>

namespace CabanaPD
{
/******************************************************************************
  Contact model
******************************************************************************/
struct ContactModel
{
    double delta;
    double Rc;

    ContactModel() {};
    // PD horizon
    // Contact radius
    ContactModel( const double _delta, const double _Rc )
        : delta( _delta )
        , Rc( _Rc ) {};
};

/* Normal repulsion */

struct HertzianModel : public ContactModel
{
    // FIXME: This is for use as the primary force model.
    using base_model = PMB;
    using fracture_type = Elastic;
    using thermal_type = TemperatureIndependent;

    using ContactModel::Rc;

    double nu;   // Poisson's ratio
    double E_s;  // Equivalent Young's modulus
    double R_s;  // Equivalent radius
    double e;    // Coefficient of restitution
    double beta; // Damping coefficient
    double Sn;   // Normal stiffness

    HertzianModel() {};
    HertzianModel( const double _Rc, const double _nu, const double _E,
                   const double _e )
        : ContactModel( 1.0, Rc )
    {
        set_param( _Rc, _nu, _E, _e );
    }

    void set_param( const double _Rc, const double _nu, const double _E,
                    const double _e )
    {
        R_s = 0.5 * _Rc;
        nu = _nu;
        E_s = _E / ( 2.0 * std::pow( 1.0 - nu, 2.0 ) );
        e = _e;
        double ln_e = std::log( e );
        beta = -ln_e / std::sqrt( std::pow( ln_e, 2.0 ) + std::pow( pi, 2.0 ) );
    }
};

template <>
struct is_contact<HertzianModel> : public std::true_type
{
};

} // namespace CabanaPD

#endif
