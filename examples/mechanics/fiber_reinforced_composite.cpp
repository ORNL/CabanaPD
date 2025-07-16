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

// Two-ply unidirectional fiber-reinforced composite laminate with 0-degree and
// 90-degree plies subjected to displacement boundary conditions.
void fiberReinforcedCompositeExample( const std::string filename )
{
    // ====================================================
    //             Use default Kokkos spaces
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
    // Matrix material
    double rho0_m = inputs["density"][0];
    double E_m = inputs["elastic_modulus"][0];
    double G_m = inputs["shear_modulus_matrix"]; // Only for LPS.
    double K_m = E_m * G_m / ( 3.0 * ( 3.0 * G_m - E_m ) );
    double G0_m = inputs["fracture_energy"][0];

    // Fiber material
    double rho0_f = inputs["density"][1];
    double E_f = inputs["elastic_modulus"][1];
    double nu_f = 1.0 / 4.0;
    double K_f = E_f / ( 3.0 * ( 1.0 - 2.0 * nu_f ) );
    // double G_f = E_f / ( 2.0 * ( 1.0 + nu_f ) ); // Only for LPS.
    double G0_f = inputs["fracture_energy"][1];

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
    //                    Force model
    // ====================================================
    using model_type = CabanaPD::PMB;
    // using model_type = CabanaPD::LPS;

    // Matrix material
    CabanaPD::ForceModel force_model_matrix( model_type{}, delta, K_m, G0_m );
    // CabanaPD::ForceModel force_model_matrix( model_type{}, delta, K_m, G_m,
    // G0_m );

    // Fiber material
    CabanaPD::ForceModel force_model_fiber( model_type{}, delta, K_f, G0_f );
    // CabanaPD::ForceModel force_model_fiber( model_type{}, delta, K_f, G_f,
    // G0_f );

    // ====================================================
    //                 Particle generation
    // ====================================================
    CabanaPD::Particles particles( memory_space{}, model_type{}, low_corner,
                                   high_corner, num_cells, halo_width,
                                   exec_space{} );

    // ====================================================
    //            Custom particle initialization
    // ====================================================
    auto rho = particles.sliceDensity();
    auto x = particles.sliceReferencePosition();
    auto type = particles.sliceType();

    // Number of fibers per dimension
    int Nf = inputs["fibers_per_dimension"];
    // Check if Nf is even
    if ( Nf % 2 != 0 )
    {
        throw std::runtime_error( "Error: Nf is odd. It must be even." );
    }

    // Fiber radius
    double Df = inputs["fiber_diameter"];
    double Rf = 0.5 * Df;

    // System sizes
    std::array<double, 3> system_size = inputs["system_size"];
    double Lx = system_size[0];
    double Ly = system_size[1];
    double Lz = system_size[2];

    // Fiber grid spacings
    double dxf = Lx / Nf;
    double dyf = Ly / Nf;
    double dzf = Lz / Nf;

    // Check fibers do not overlap
    if ( Df > dxf || Df > dyf || Df > dzf )
    {
        throw std::runtime_error( "Error: Fiber diameter is too large for "
                                  "given number of fibers per dimension." );
    }

    // Domain center coordinates
    double Zc = 0.5 * ( low_corner[2] + high_corner[2] );
    auto init_functor = KOKKOS_LAMBDA( const int pid )
    {
        // Particle position
        double xi = x( pid, 0 );
        double yi = x( pid, 1 );
        double zi = x( pid, 2 );

        // --------------------------------------------
        // Bottom ply: fiber orientation in x-direction
        // --------------------------------------------
        if ( x( pid, 2 ) < Zc )
        {
            // Find nearest fiber grid center point in YZ plane.
            double Iyf = Kokkos::floor( yi / dyf );
            double Izf = Kokkos::floor( zi / dzf );
            double YI = 0.5 * dyf + dyf * Iyf;
            double ZI = 0.5 * dzf + dzf * Izf;

            // Check if point belongs to fiber
            if ( ( yi - YI ) * ( yi - YI ) + ( zi - ZI ) * ( zi - ZI ) <
                 Rf * Rf + 1e-8 )
            {
                // Material type: 1 = fiber (default is 0 = matrix)
                type( pid ) = 1;
                // Density (fiber)
                rho( pid ) = rho0_f;
            }
            else
            {
                // Density (matrix)
                rho( pid ) = rho0_m;
            }
        }
        // --------------------------------------------
        // Top ply: fiber orientation in y-direction
        // --------------------------------------------
        else
        {
            // Find nearest fiber grid center point in XZ plane.
            double Ixf = Kokkos::floor( xi / dxf );
            double Izf = Kokkos::floor( zi / dzf );
            double XI = 0.5 * dxf + dxf * Ixf;
            double ZI = 0.5 * dzf + dzf * Izf;

            // Check if point belongs to fiber
            if ( ( xi - XI ) * ( xi - XI ) + ( zi - ZI ) * ( zi - ZI ) <
                 Rf * Rf + 1e-8 )
            {
                // Material type: 1 = fiber (default is 0 = matrix)
                type( pid ) = 1;
                // Density (fiber)
                rho( pid ) = rho0_f;
            }
            else
            {
                // Density (matrix)
                rho( pid ) = rho0_m;
            }
        }
    };
    particles.updateParticles( exec_space{}, init_functor );

    // ====================================================
    //                   Create solver
    // ====================================================
    auto models = CabanaPD::createMultiForceModel(
        particles, CabanaPD::AverageTag{}, force_model_matrix,
        force_model_fiber );
    CabanaPD::Solver solver( inputs, particles, models );

    // ====================================================
    //                  Boundary conditions
    // ====================================================
    // Grip velocity
    double v0 = inputs["velocity_bc"];

    // Create region for boundary conditions
    CabanaPD::Region<CabanaPD::RectangularPrism> plane1(
        low_corner[0] - 2.0 * delta, low_corner[0] + 2.0 * delta, low_corner[1],
        high_corner[1], low_corner[2], high_corner[2] );
    CabanaPD::Region<CabanaPD::RectangularPrism> plane2(
        high_corner[0] - 2.0 * delta, high_corner[0] + 2.0 * delta,
        low_corner[1], high_corner[1], low_corner[2], high_corner[2] );

    // Create BC last to ensure ghost particles are included.
    x = solver.particles.sliceReferencePosition();
    auto u = solver.particles.sliceDisplacement();
    auto disp_func = KOKKOS_LAMBDA( const int pid, const double t )
    {
        if ( plane1.inside( x, pid ) )
        {
            u( pid, 0 ) = 0.0;
            u( pid, 1 ) = 0.0;
            u( pid, 2 ) = 0.0;
        }
        else if ( plane2.inside( x, pid ) )
        {
            u( pid, 0 ) = v0 * t;
            u( pid, 1 ) = 0.0;
            u( pid, 2 ) = 0.0;
        }
    };
    auto bc = CabanaPD::createBoundaryCondition(
        disp_func, exec_space{}, solver.particles, false, plane1, plane2 );

    // ====================================================
    //                   Simulation run
    // ====================================================
    solver.init( bc );
    solver.run( bc );
}

// Initialize MPI+Kokkos.
int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    fiberReinforcedCompositeExample( argv[1] );

    Kokkos::finalize();
    MPI_Finalize();
}
