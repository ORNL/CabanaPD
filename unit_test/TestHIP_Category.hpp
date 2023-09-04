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

/****************************************************************************
 * Copyright (c) 2018-2021 by the Cabana authors                            *
 * All rights reserved.                                                     *
 *                                                                          *
 * This file is part of the Cabana library. Cabana is distributed under a   *
 * BSD 3-clause license. For the licensing terms see the LICENSE file in    *
 * the top-level directory.                                                 *
 *                                                                          *
 * SPDX-License-Identifier: BSD-3-Clause                                    *
 ****************************************************************************/

#ifndef TEST_HIP_CATEGORY_HPP
#define TEST_HIP_CATEGORY_HPP

#define TEST_CATEGORY hip
#define TEST_EXECSPACE Kokkos::Experimental::HIP
#define TEST_MEMSPACE Kokkos::Experimental::HIPSpace
#define TEST_DEVICE                                                            \
    Kokkos::Device<Kokkos::Experimental::HIP, Kokkos::Experimental::HIPSpace>

#endif
