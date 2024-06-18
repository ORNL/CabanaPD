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

namespace CabanaPD
{
// Mechanics types.
struct Elastic
{
};
struct Fracture
{
};

// Thermal types.
struct TemperatureIndependent
{
};
struct TemperatureDependent
{
};
struct DynamicTemperature : public TemperatureDependent
{
    using base_type = TemperatureDependent;
};

// Model types.
struct PMB
{
};
struct LinearPMB
{
};
struct LPS
{
};
struct LinearLPS
{
};

} // namespace CabanaPD
#endif
