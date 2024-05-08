/****************************************************************************
 * Copyright (c) 2022-2023 by Oak Ridge National Laboratory                 *
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

namespace CabanaPD
{
struct BaseForceModel
{
    double delta;

    BaseForceModel(){};
    BaseForceModel( const double _delta )
        : delta( _delta ){};

    BaseForceModel( const BaseForceModel& model ) { delta = model.delta; }
};

template <typename ModelType, typename DamageType>
struct ForceModel;

} // namespace CabanaPD

#endif
