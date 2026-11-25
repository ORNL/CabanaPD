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

#include <gtest/gtest.h>

#include <CabanaPD_Force.hpp>

namespace Test
{

TEST( TEST_CATEGORY, test_distance )
{
    const unsigned num_theta_steps = 100;
    const unsigned num_phi_steps = 100;
    const unsigned num_eta_steps = 100;
    double xi_norm = 1e-6;

    Kokkos::View<double[2][3], Kokkos::HostSpace> x( "position" );
    Kokkos::View<double[2][3], Kokkos::HostSpace> u( "displacement" );

    // have one point at the origin and one point at one xi_norm away
    for ( unsigned i = 0; i < 3; ++i )
    {
        x( 0, i ) = 0.0;
        u( 0, i ) = 0.0;
    }

    for ( unsigned eta_step = 0; eta_step < num_theta_steps; ++eta_step )
    {
        double eta_norm =
            1e-12 * xi_norm * (double)eta_step / (double)num_eta_steps;
        for ( unsigned phi_step = 0; phi_step < num_phi_steps; ++phi_step )
        {
            for ( unsigned theta_step = 0; theta_step < num_theta_steps;
                  theta_step++ )
            {
                double theta =
                    2.0 * M_PI * double( eta_step ) / double( num_theta_steps );
                double phi =
                    M_PI * double( phi_step ) / double( num_phi_steps );
                x( 1, 0 ) = xi_norm * sin( phi ) * cos( theta );
                x( 1, 1 ) = xi_norm * sin( phi ) * sin( theta );
                x( 1, 2 ) = xi_norm * cos( phi );

                // this way they both point into the same direction
                u( 1, 0 ) = eta_norm * sin( phi ) * cos( theta );
                u( 1, 1 ) = eta_norm * sin( phi ) * sin( theta );
                u( 1, 2 ) = eta_norm * cos( phi );

                double xi;
                double r;
                double s;
                double rx, ry, rz;
                double xi_x, xi_y, xi_z;

                CabanaPD::getDistance(
                    // input variables
                    x, u, 0, 1,
                    // output variables
                    xi, r, s, rx, ry, rz, xi_x, xi_y, xi_z );
                EXPECT_NEAR( s, eta_norm / xi_norm,
                             eta_norm / xi_norm * 1e-10 );
            }
        }
    }
}
} // end namespace Test
