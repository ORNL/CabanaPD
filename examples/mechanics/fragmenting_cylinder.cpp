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

#include <fstream>
#include <iostream>

#include "mpi.h"

#include <Kokkos_Core.hpp>

#include <CabanaPD.hpp>

// Simulate an expanding cylinder resulting in fragmentation.
void fragmentingCylinderExample( const std::string filename )
{
    // ====================================================
    //               Choose Kokkos spaces
    // ====================================================
    using exec_space = Kokkos::DefaultExecutionSpace;
    using memory_space = typename exec_space::memory_space;

    // ====================================================
    //                   Read inputs
    // ====================================================
    CabanaPD::Inputs inputs( filename );

    // ====================================================
    //                Material parameters
    // ====================================================
    double rho0 = inputs["density"];
    double K = inputs["bulk_modulus"];
    // double G = inputs["shear_modulus"]; // Only for LPS.
    double sc = inputs["critical_stretch"];
    double delta = inputs["horizon"];
    delta += 1e-10;
    // For PMB or LPS with influence_type == 1
    double G0 = 9 * K * delta * ( sc * sc ) / 5;
    // For LPS with influence_type == 0 (default)
    // double G0 = 15 * K * delta * ( sc * sc ) / 8;

    // ====================================================
    //                  Discretization
    // ====================================================
    std::array<double, 3> low_corner = inputs["low_corner"];
    std::array<double, 3> high_corner = inputs["high_corner"];
    std::array<int, 3> num_cells = inputs["num_cells"];
    int m = std::floor( delta /
                        ( ( high_corner[0] - low_corner[0] ) / num_cells[0] ) );
    int halo_width = m + 1; // Just to be safe.

    // ====================================================
    //                    Force model
    // ====================================================
    using model_type = CabanaPD::PMB;
    CabanaPD::ForceModel force_model( model_type{}, delta, K, G0 );

    // ====================================================
    //    Custom particle generation and initialization
    // ====================================================
    double x_center = 0.5 * ( low_corner[0] + high_corner[0] );
    double y_center = 0.5 * ( low_corner[1] + high_corner[1] );
    double z_center = 0.5 * ( low_corner[2] + high_corner[2] );
    double Rout = inputs["cylinder_outer_radius"];
    double Rin = inputs["cylinder_inner_radius"];
    double H = inputs["cylinder_height"];

    // Do not create particles outside given cylindrical region
    auto init_op = KOKKOS_LAMBDA( const int, const double x[3] )
    {
        double rsq = ( x[0] - x_center ) * ( x[0] - x_center ) +
                     ( x[1] - y_center ) * ( x[1] - y_center );
        if ( rsq < Rin * Rin || rsq > Rout * Rout ||
             x[2] > z_center + 0.5 * H || x[2] < z_center - 0.5 * H )
            return false;
        return true;
    };

    // ====================================================
    //  Simulation run with contact physics
    // ====================================================
    if ( inputs["use_contact"] )
    {
        using contact_type = CabanaPD::NormalRepulsionModel;
        CabanaPD::Particles particles( memory_space{}, contact_type{} );
        particles.domain( low_corner, high_corner, num_cells, halo_width );
        particles.create( exec_space{}, Cabana::InitRandom{}, init_op );

        auto rho = particles.sliceDensity();
        auto x = particles.sliceReferencePosition();
        auto v = particles.sliceVelocity();
        auto f = particles.sliceForce();
        auto dx = particles.dx;

        double vrmax = inputs["max_radial_velocity"];
        double vrmin = inputs["min_radial_velocity"];
        double vzmax = inputs["max_vertical_velocity"];
        double zmin = z_center - 0.5 * H;

        auto init_functor = KOKKOS_LAMBDA( const int pid )
        {
            // Density
            rho( pid ) = rho0;

            // Velocity
            double zfactor = ( ( x( pid, 2 ) - zmin ) / ( 0.5 * H ) ) - 1;
            double vr = vrmax - vrmin * zfactor * zfactor;
            v( pid, 0 ) =
                vr * Kokkos::cos( Kokkos::atan2( x( pid, 1 ), x( pid, 0 ) ) );
            v( pid, 1 ) =
                vr * Kokkos::sin( Kokkos::atan2( x( pid, 1 ), x( pid, 0 ) ) );
            v( pid, 2 ) = vzmax * zfactor;
        };
        particles.update( exec_space{}, init_functor );

        // Use contact radius and extension relative to particle spacing.
        double r_c = inputs["contact_horizon_factor"];
        double r_extend = inputs["contact_horizon_extend_factor"];
        // NOTE: dx/2 is when particles first touch.
        r_c *= dx[0] / 2.0;
        r_extend *= dx[0];

        contact_type contact_model( delta, r_c, r_extend, K );

        CabanaPD::Solver solver( inputs, particles, force_model,
                                 contact_model );
        solver.init();
        solver.run();
    }
    // ====================================================
    //  Simulation run without contact
    // ====================================================
    else
    {
        CabanaPD::Particles particles( memory_space{}, model_type{} );
        particles.domain( low_corner, high_corner, num_cells, halo_width );
        particles.create( exec_space{}, Cabana::InitRandom{}, init_op );

        auto rho = particles.sliceDensity();
        auto x = particles.sliceReferencePosition();
        auto v = particles.sliceVelocity();
        auto f = particles.sliceForce();

        double vrmax = inputs["max_radial_velocity"];
        double vrmin = inputs["min_radial_velocity"];
        double vzmax = inputs["max_vertical_velocity"];
        double zmin = z_center - 0.5 * H;

        auto init_functor = KOKKOS_LAMBDA( const int pid )
        {
            // Density
            rho( pid ) = rho0;

            // Velocity
            double zfactor = ( ( x( pid, 2 ) - zmin ) / ( 0.5 * H ) ) - 1;
            double vr = vrmax - vrmin * zfactor * zfactor;
            v( pid, 0 ) =
                vr * Kokkos::cos( Kokkos::atan2( x( pid, 1 ), x( pid, 0 ) ) );
            v( pid, 1 ) =
                vr * Kokkos::sin( Kokkos::atan2( x( pid, 1 ), x( pid, 0 ) ) );
            v( pid, 2 ) = vzmax * zfactor;
        };
        particles.update( exec_space{}, init_functor );

        CabanaPD::Solver solver( inputs, particles, force_model );
        solver.init();
        solver.run();
    }
}

// Initialize MPI+Kokkos.
int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    fragmentingCylinderExample( argv[1] );

    Kokkos::finalize();
    MPI_Finalize();
}
