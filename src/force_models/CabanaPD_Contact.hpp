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
    using base_model = Contact;

    // Contact neighbor search radius.
    double radius;
    // Extend neighbor search radius to reuse lists.
    double radius_extend;

    ContactModel() {}

    // PD horizon
    // Contact radius
    ContactModel( const double _radius, const double _radius_extend )
        : radius( _radius )
        , radius_extend( _radius_extend ){};
};

/* Normal repulsion */
struct NormalRepulsionModel : public ContactModel
{
    using base_type = ContactModel;
    using base_model = base_type::base_model;
    using model_type = NormalRepulsionModel;
    using fracture_type = NoFracture;
    using thermal_type = TemperatureIndependent;

    double delta;
    using ContactModel::radius;
    using ContactModel::radius_extend;

    double c;
    double K;

    NormalRepulsionModel() {}
    NormalRepulsionModel( const double _delta, const double radius,
                          const double radius_extend, const double _K )
        : ContactModel( radius, radius_extend )
        , delta( _delta )
        , K( _K )
    {
        K = _K;
        // This could inherit from PMB (same c)
        c = 18.0 * K / ( pi * delta * delta * delta * delta );
    }

    KOKKOS_INLINE_FUNCTION
    auto forceCoeff( const double r, const double vol ) const
    {
        // Contact "stretch"
        const double sc = ( r - radius ) / delta;
        // Normal repulsion uses a 15 factor compared to the PMB force
        return 15.0 * c * sc * vol;
    }
};

} // namespace CabanaPD

#endif
