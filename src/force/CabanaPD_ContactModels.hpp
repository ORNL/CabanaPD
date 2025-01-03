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

    ContactModel(){};
    // PD horizon
    // Contact radius
    ContactModel( const double _delta, const double _Rc )
        : delta( _delta )
        , Rc( _Rc ){};
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

    double c;
    double K;

    NormalRepulsionModel(){};
    NormalRepulsionModel( const double delta, const double Rc, const double _K )
        : ContactModel( delta, Rc )
        , K( _K )
    {
        set_param( delta, Rc, K );
    }

    void set_param( const double _delta, const double _Rc, const double _K )
    {
        delta = _delta;
        Rc = _Rc;
        K = _K;
        // This could inherit from PMB (same c)
        c = 18.0 * K / ( pi * delta * delta * delta * delta );
    }
};

template <>
struct is_contact<NormalRepulsionModel> : public std::true_type
{
};

} // namespace CabanaPD

#endif
