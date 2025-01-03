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

#ifndef CONSTANTS_H
#define CONSTANTS_H

#include "Kokkos_Core.hpp"

namespace CabanaPD
{

constexpr double pi = Kokkos::numbers::pi_v<double>;

} // namespace CabanaPD
#endif
