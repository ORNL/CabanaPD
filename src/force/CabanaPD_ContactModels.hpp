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
    // PD neighbor search radius.
    double delta;
    // Contact neighbor search radius.
    double Rc;
    // Extend neighbor search radius to reuse lists.
    double Rc_extend;

    // PD horizon
    // Contact radius
    ContactModel( const double _delta, const double _Rc,
                  const double _Rc_extend )
        : delta( _delta )
        , Rc( _Rc )
        , Rc_extend( _Rc_extend ){};
};

/* Normal repulsion */
struct NormalRepulsionModel : public ContactModel
{
    // FIXME: This is for use as the primary force model.
    using base_model = PMB;
    using fracture_type = NoFracture;
    using thermal_type = TemperatureIndependent;

    using ContactModel::delta;
    using ContactModel::Rc;
    using ContactModel::Rc_extend;

    double c;
    double K;

    NormalRepulsionModel( const double delta, const double Rc,
                          const double Rc_extend, const double _K )
        : ContactModel( delta, Rc, Rc_extend )
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
        const double sc = ( r - Rc ) / delta;
        // Normal repulsion uses a 15 factor compared to the PMB force
        return 15.0 * c * sc * vol;
    }
};

template <>
struct is_contact<NormalRepulsionModel> : public std::true_type
{
};

} // namespace CabanaPD

#endif
