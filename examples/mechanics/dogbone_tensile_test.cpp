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

// Simulate ASTM D638 type I dogbone tensile test.
void dogboneTensileTestExample( const std::string filename )
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
    double sigma_y = inputs["yield_stress"];
    double delta = inputs["horizon"];
    delta += 1e-10;

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
    //                  Force model type
    // ====================================================
    using model_type = CabanaPD::PMB;
    using mechanics_type = CabanaPD::ElasticPerfectlyPlastic;

    // ====================================================
    //    Custom particle generation and initialization
    // ====================================================
    double G = inputs["gage_length"];
    double D = inputs["distance_between_grips"];
    double W = inputs["width_narrow_section"];
    double R = inputs["fillet_radius"];

    // x- and y-coordinates of center of domain
    double midx = 0.5 * ( low_corner[0] + high_corner[0] );
    double midy = 0.5 * ( low_corner[1] + high_corner[1] );

    // Do not create particles outside dogbone tensile test specimen region.
    auto init_op = KOKKOS_LAMBDA( const int, const double x[3] )
    {
        // Filler radius squared
        double Rsq = R * R;

        // Bottom-left fillet circle center
        double xc_bl = midx - 0.5 * G;
        double yc_bl = midy - 0.5 * W - R;

        // Bottom-right fillet circle center
        double xc_br = midx + 0.5 * G;
        double yc_br = yc_bl;

        // Top-left fillet circle center
        double xc_tl = xc_bl;
        double yc_tl = midy + 0.5 * W + R;

        // Top-right fillet circle center
        double xc_tr = xc_br;
        double yc_tr = yc_tl;

        // Gage section
        if ( Kokkos::abs( x[0] - midx ) < 0.5 * G &&
             Kokkos::abs( x[1] - midy ) > 0.5 * W )
        {
            return false;
        }
        // Bottom-left fillet
        else if ( Kokkos::abs( x[0] - xc_bl ) * Kokkos::abs( x[0] - xc_bl ) +
                      Kokkos::abs( x[1] - yc_bl ) *
                          Kokkos::abs( x[1] - yc_bl ) <
                  Rsq )
        {
            return false;
        }
        // Bottom-right fillet
        else if ( Kokkos::abs( x[0] - xc_br ) * Kokkos::abs( x[0] - xc_br ) +
                      Kokkos::abs( x[1] - yc_br ) *
                          Kokkos::abs( x[1] - yc_br ) <
                  Rsq )
        {
            return false;
        }
        // Top-left fillet
        else if ( Kokkos::abs( x[0] - xc_tl ) * Kokkos::abs( x[0] - xc_tl ) +
                      Kokkos::abs( x[1] - yc_tl ) *
                          Kokkos::abs( x[1] - yc_tl ) <
                  Rsq )
        {
            return false;
        }
        // Top-right fillet
        else if ( Kokkos::abs( x[0] - xc_tr ) * Kokkos::abs( x[0] - xc_tr ) +
                      Kokkos::abs( x[1] - yc_tr ) *
                          Kokkos::abs( x[1] - yc_tr ) <
                  Rsq )
        {
            return false;
        }
        else
        {
            return true;
        }
    };

    CabanaPD::Particles particles( memory_space{}, model_type{} );
    particles.domain( low_corner, high_corner, num_cells, halo_width );
    particles.create( exec_space{}, Cabana::InitRandom{}, init_op );

    auto rho = particles.sliceDensity();

    auto init_functor = KOKKOS_LAMBDA( const int pid )
    {
        // Density
        rho( pid ) = rho0;
    };
    particles.update( exec_space{}, init_functor );

    // ====================================================
    //                    Force model
    // ====================================================
    CabanaPD::ForceModel force_model( model_type{}, mechanics_type{},
                                      memory_space{}, delta, K, G0, sigma_y );

    // ====================================================
    //                   Create solver
    // ====================================================
    CabanaPD::Solver solver( inputs, particles, force_model );

    // ====================================================
    //                  Boundary conditions
    // ====================================================
    // Grip velocity
    double v0 = inputs["grip_velocity"];

    // Create region for both grips.
    CabanaPD::Region<CabanaPD::RectangularPrism> right_grip(
        midx + 0.5 * D, high_corner[0], low_corner[1], high_corner[1],
        low_corner[2], high_corner[2] );
    CabanaPD::Region<CabanaPD::RectangularPrism> left_grip(
        low_corner[0], midx - 0.5 * D, low_corner[1], high_corner[1],
        low_corner[2], high_corner[2] );

    // Create BC last to ensure ghost particles are included.
    auto x = solver.particles.sliceReferencePosition();
    auto u = solver.particles.sliceDisplacement();
    auto disp_func = KOKKOS_LAMBDA( const int pid, const double t )
    {
        if ( right_grip.inside( x, pid ) )
        {
            u( pid, 0 ) = v0 * t;
            u( pid, 1 ) = 0.0;
            u( pid, 2 ) = 0.0;
        }
        else if ( left_grip.inside( x, pid ) )
        {
            u( pid, 0 ) = 0.0;
            u( pid, 1 ) = 0.0;
            u( pid, 2 ) = 0.0;
        }
    };
    auto bc = CabanaPD::createBoundaryCondition( disp_func, exec_space{},
                                                 solver.particles, false,
                                                 left_grip, right_grip );

    // ====================================================
    //                      Outputs
    // ====================================================
    auto dx = solver.particles.dx[0];
    auto dy = solver.particles.dx[1];
    auto dz = solver.particles.dx[2];
    auto f = solver.particles.sliceForce();

    // Generate force outputs for right grip to compute stress.
    // Output force on right grip in x-direction.
    auto force_func_x = KOKKOS_LAMBDA( const int p )
    {
        return f( p, 0 ) * dx * dy * dz;
    };
    auto output_fx = CabanaPD::createOutputTimeSeries(
        "output_force_x.txt", inputs, exec_space{}, solver.particles,
        force_func_x, right_grip );

    // Output force on right grip in y-direction.
    auto force_func_y = KOKKOS_LAMBDA( const int p )
    {
        return f( p, 1 ) * dx * dy * dz;
    };
    auto output_fy = CabanaPD::createOutputTimeSeries(
        "output_force_y.txt", inputs, exec_space{}, solver.particles,
        force_func_y, right_grip );

    // Output force on right grip in z-direction.
    auto force_func_z = KOKKOS_LAMBDA( const int p )
    {
        return f( p, 2 ) * dx * dy * dz;
    };
    auto output_fz = CabanaPD::createOutputTimeSeries(
        "output_force_z.txt", inputs, exec_space{}, solver.particles,
        force_func_z, right_grip );

    // Generate position outputs for gage to compute strain.
    auto y = solver.particles.sliceCurrentPosition();
    auto pos_func = KOKKOS_LAMBDA( const int p ) { return y( p, 0 ); };
    double dG = 0.1 * G;

    // Output x-position of left side of gage.
    double x_gl = midx - 0.5 * G;
    CabanaPD::Region<CabanaPD::RectangularPrism> left_pos(
        x_gl, x_gl + dG, low_corner[1], high_corner[1], low_corner[2],
        high_corner[2] );
    auto output_yl = CabanaPD::createOutputTimeSeries(
        "output_left_position.txt", inputs, exec_space{}, solver.particles,
        pos_func, left_pos );

    // Output x-position of right side of gage.
    double x_gr = midx + 0.5 * G;
    CabanaPD::Region<CabanaPD::RectangularPrism> right_pos(
        x_gr - dG, x_gr, low_corner[1], high_corner[1], low_corner[2],
        high_corner[2] );
    auto output_yr = CabanaPD::createOutputTimeSeries(
        "output_right_position.txt", inputs, exec_space{}, solver.particles,
        pos_func, right_pos );

    // ====================================================
    //                   Simulation run
    // ====================================================
    solver.init( bc );
    solver.run( bc, output_fx, output_fy, output_fz, output_yl, output_yr );
}

// Initialize MPI+Kokkos.
int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    dogboneTensileTestExample( argv[1] );

    Kokkos::finalize();
    MPI_Finalize();
}
