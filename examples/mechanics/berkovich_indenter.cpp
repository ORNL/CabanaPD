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

// Simulate a Berkovich indenter on a plate.
void berkovichIndenterExample( const std::string filename )
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
    double alpha = inputs["indenter_angle"];
    // Convert angle to radians.
    alpha *= CabanaPD::pi / 180.0;
    // Compute tan( alpha ) to use multiple times.
    double tan_alpha = Kokkos::tan( alpha );

    double v0 = inputs["indenter_velocity"];
    double tfinal = inputs["final_time"];
    // Height
    double H = v0 * tfinal;
    // Base radius
    double R = H * tan_alpha;

    double x_center = 0.5 * ( low_corner[0] + high_corner[0] );
    double y_center = 0.5 * ( low_corner[1] + high_corner[1] );

    // TODO: Adapt region to pyramid base shape for force calculation.
    // Define boundary condition regions to constrain displacements.
    CabanaPD::Region<CabanaPD::RectangularPrism> pressure_region(
        low_corner[0], high_corner[0], low_corner[1], high_corner[1],
        low_corner[2], high_corner[2] );

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
    // double v0 = inputs["indenter_velocity"];

    // z-coordinate of top layer of particles
    double z_top = high_corner[2] - 0.5 * dz;
    // Initial z-coordinate of the indenter tip
    double z0_indenter = z_top;

    auto disp_func = KOKKOS_LAMBDA( const int pid, const double t )
    {
        // z-coordinate of the indenter tip
        double z_indenter = z0_indenter - v0 * t;

        // Check if indenter tip touches the plate (top layer of particles)
        if ( z_indenter < z_top )
        {
            // Height of indenter
            double H = z_top - z_indenter;

            // Edge length of base equilateral triangle
            double L = 2.0 * std::sqrt( 3.0 ) * tan_alpha * H;

            // Inradius of base equilateral triangle
            double R = std::sqrt( 3.0 ) * L / 6.0;

            // Indenter displacement
            double x_i = x( pid, 0 );
            double y_i = x( pid, 1 );

            if ( y_i > -R && y_i <= std::sqrt( 3.0 ) * x_i + 2 * R &&
                 y_i <= -std::sqrt( 3.0 ) * x_i + 2 * R )
            {
                // Find distance to edges.
                double d1 = std::abs( y_i + R );
                double d2 =
                    0.5 * std::abs( -std::sqrt( 3.0 ) * x_i + y_i - 2 * R );
                double d3 =
                    0.5 * std::abs( std::sqrt( 3.0 ) * x_i + y_i - 2 * R );

                // Find distance to nearest edge.
                double d = std::min( { d1, d2, d3 } );

                // Find inradius of triangle for which particle is on its
                // perimenter
                double r = R - d;

                // Find edge length of triangle for which particle is on its
                // perimenter
                double l = 6.0 * r / std::sqrt( 3.0 );

                // Indenter displacement
                u( pid, 2 ) = z_indenter +
                              l / ( 2.0 * std::sqrt( 3.0 ) * tan_alpha ) -
                              x( pid, 2 );
            }
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
    auto force_func_x = KOKKOS_LAMBDA( const int p )
    {
        return f( p, 0 ) * dx * dy * dz;
    };
    auto output_fx = CabanaPD::createOutputTimeSeries(
        "output_force_x.txt", inputs, exec_space{}, solver.particles,
        force_func_x, pressure_region );

    // Output force in y-direction.
    auto force_func_y = KOKKOS_LAMBDA( const int p )
    {
        return f( p, 1 ) * dx * dy * dz;
    };
    auto output_fy = CabanaPD::createOutputTimeSeries(
        "output_force_y.txt", inputs, exec_space{}, solver.particles,
        force_func_y, pressure_region );

    // Output force in z-direction.
    auto force_func_z = KOKKOS_LAMBDA( const int p )
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

    solver.updateRegion( output_fx, output_fy, output_fz );

    solver.run( bc, output_fx, output_fy, output_fz );
}

// Initialize MPI+Kokkos.
int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    berkovichIndenterExample( argv[1] );

    Kokkos::finalize();
    MPI_Finalize();
}
