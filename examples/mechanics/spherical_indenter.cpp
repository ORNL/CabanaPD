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

#include <fstream>
#include <iostream>

#include "mpi.h"

#include <Kokkos_Core.hpp>

#include <CabanaPD.hpp>

// Simulate a spherical indenter on a plate.
void sphericalIndenterExample( const std::string filename )
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
    double E = inputs["elastic_modulus"];
    double nu = 0.25; // Use bond-based model
    double K = E / ( 3 * ( 1 - 2 * nu ) );
    double G0 = inputs["fracture_energy"];
    double horizon = inputs["horizon"];
    horizon += 1e-10;

    // ====================================================
    //                  Discretization
    // ====================================================
    std::array<double, 3> low_corner = inputs["low_corner"];
    std::array<double, 3> high_corner = inputs["high_corner"];

    // ====================================================
    //                    Force model
    // ====================================================
    using model_type = CabanaPD::PMB;
    // using contact_type = CabanaPD::NormalRepulsionModel;
    CabanaPD::ForceModel force_model( model_type{}, horizon, K, G0 );

    // ====================================================
    //                 Particle generation
    // ====================================================
    CabanaPD::Particles particles( memory_space{}, model_type{} );
    // CabanaPD::Particles particles( memory_space{}, contact_type{} );

    // Note that individual inputs can be passed instead (see other examples).
    particles.domain( inputs );
    particles.create( exec_space{} );

    // ====================================================
    //               Boundary conditions planes
    // ====================================================
    double dx = particles.dx[0];
    double dy = particles.dx[1];
    double dz = particles.dx[2];
    double R = inputs["indenter_radius"];
    double x_center = 0.5 * ( low_corner[0] + high_corner[0] );
    double y_center = 0.5 * ( low_corner[1] + high_corner[1] );

    // Define cylindrical region to apply pressure.
    CabanaPD::Region<CabanaPD::Cylinder> pressure_region(
        0.0, R, high_corner[2] - dz, high_corner[2] + dz, x_center, y_center );

    // Define boundary condition regions to constrain displacements.
    CabanaPD::Region<CabanaPD::RectangularPrism> low_x_plane(
        low_corner[0] - dx, low_corner[0] + dx, low_corner[1], high_corner[1],
        low_corner[2], high_corner[2] );
    CabanaPD::Region<CabanaPD::RectangularPrism> high_x_plane(
        high_corner[0] - dx, high_corner[0] + dx, low_corner[1], high_corner[1],
        low_corner[2], high_corner[2] );
    CabanaPD::Region<CabanaPD::RectangularPrism> low_y_plane(
        low_corner[0], high_corner[0], low_corner[1] - dy, low_corner[1] + dy,
        low_corner[2], high_corner[2] );
    CabanaPD::Region<CabanaPD::RectangularPrism> high_y_plane(
        low_corner[0], high_corner[0], high_corner[1] - dy, high_corner[1] + dy,
        low_corner[2], high_corner[2] );

    // ====================================================
    //            Custom particle initialization
    // ====================================================
    auto rho = particles.sliceDensity();
    auto nofail = particles.sliceNoFail();
    auto init_functor = KOKKOS_LAMBDA( const int pid )
    {
        // Density
        rho( pid ) = rho0;
        nofail( pid ) = 1;
    };
    particles.update( exec_space{}, init_functor );

    // ====================================================
    //                   Create solver
    // ====================================================
    CabanaPD::Solver solver( inputs, particles, force_model );

    /*
    // Use contact radius and extension relative to particle spacing.
    double r_c = inputs["contact_horizon_factor"];
    double r_extend = inputs["contact_horizon_extend_factor"];
    // NOTE: dx/2 is when particles first touch.
    r_c *= dx / 2.0;
    r_extend *= dx;

    contact_type contact_model( horizon, r_c, r_extend, K );
    CabanaPD::Solver solver( inputs, particles, force_model,
                                 contact_model );
    */

    // ====================================================
    //                Boundary conditions
    // ====================================================
    // Create BC last to ensure ghost particles are included.
    auto x = solver.particles.sliceReferencePosition();
    auto u = solver.particles.sliceDisplacement();
    double v0 = inputs["indenter_velocity"];

    // z-coordinate of top layer of particles
    double z_top = high_corner[2] - 0.5 * dz;
    // Initial z-coordinate of the indenter center
    double z0_indenter = z_top + R;

    auto disp_func = KOKKOS_LAMBDA( const int pid, const double t )
    {
        // z-coordinate of the indenter center
        double z_indenter = z0_indenter - v0 * t;

        // Check if indenter touches the plate (top layer of particles)
        if ( z_indenter - R < z_top )
        {
            double r_indenter_sq;

            if ( z_indenter > z_top )
            {
                // Current radius of indenter squared
                r_indenter_sq =
                    R * R - ( z_indenter - z_top ) * ( z_indenter - z_top );
            }
            else
            {
                // Current radius of indenter squared
                r_indenter_sq = R * R;
            }

            // Distance squared of particle to center on XY plane (assumes
            // indenter is centered on the XY plane)
            double r_sq =
                ( x( pid, 0 ) - x_center ) * ( x( pid, 0 ) - x_center ) +
                ( x( pid, 1 ) - y_center ) * ( x( pid, 1 ) - y_center );

            // Indenter displacement
            if ( r_sq <= r_indenter_sq )
                u( pid, 2 ) =
                    ( z_indenter - std::sqrt( R * R - r_sq ) ) - x( pid, 2 );
        }

        // Edge constraints
        if ( low_x_plane.inside( x, pid ) || high_x_plane.inside( x, pid ) )
        {
            u( pid, 0 ) = 0;
            u( pid, 1 ) = 0;
            u( pid, 2 ) = 0;
        }

        if ( low_y_plane.inside( x, pid ) || high_y_plane.inside( x, pid ) )
        {
            u( pid, 0 ) = 0;
            u( pid, 1 ) = 0;
            u( pid, 2 ) = 0;
        }
    };

    auto bc = createBoundaryCondition(
        disp_func, exec_space{}, solver.particles, true, pressure_region,
        low_x_plane, high_x_plane, low_y_plane, high_y_plane );

    //========================================
    //            OUTPUTS
    //========================================
    auto f = solver.particles.sliceForce();

    // Output force in x-direction.
    auto force_func_x = KOKKOS_LAMBDA( const int p, const double )
    {
        return f( p, 0 ) * dx * dy * dz;
    };
    auto output_fx = CabanaPD::createOutputTimeSeries(
        "output_force_x.txt", inputs, exec_space{}, solver.particles,
        force_func_x, pressure_region );

    // Output force in y-direction.
    auto force_func_y = KOKKOS_LAMBDA( const int p, const double )
    {
        return f( p, 1 ) * dx * dy * dz;
    };
    auto output_fy = CabanaPD::createOutputTimeSeries(
        "output_force_y.txt", inputs, exec_space{}, solver.particles,
        force_func_y, pressure_region );

    // Output force in z-direction.
    auto force_func_z = KOKKOS_LAMBDA( const int p, const double )
    {
        return f( p, 2 ) * dx * dy * dz;
    };
    auto output_fz = CabanaPD::createOutputTimeSeries(
        "output_force_z.txt", inputs, exec_space{}, solver.particles,
        force_func_z, pressure_region );

    // ====================================================
    //                   Simulation run
    // ====================================================
    solver.init( bc );

    solver.updateRegion( 0.0, output_fx, output_fy, output_fz );

    solver.run( bc, output_fx, output_fy, output_fz );
}

// Initialize MPI+Kokkos.
int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    sphericalIndenterExample( argv[1] );

    Kokkos::finalize();
    MPI_Finalize();
}
