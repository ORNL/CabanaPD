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

// Simulate crack initiation and propagation in a ceramic plate under thermal
// shock caused by water quenching.
void thermalCrackExample( const std::string filename )
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
    //            Material and problem parameters
    // ====================================================
    // Material parameters
    double rho0 = inputs["density"];
    double E = inputs["elastic_modulus"];
    double nu = 0.25;
    double K = E / ( 3 * ( 1 - 2 * nu ) );
    double G0 = inputs["fracture_energy"];
    double delta = inputs["horizon"];
    delta += 1e-10;
    double alpha = inputs["thermal_expansion_coeff"];

    // Problem parameters
    double temp0 = inputs["reference_temperature"];
    double temp_w = inputs["background_temperature"];
    double t_ramp = inputs["surface_temperature_ramp_time"];

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
    //                Force model type
    // ====================================================
    using model_type = CabanaPD::PMB;
    using thermal_type = CabanaPD::TemperatureDependent;

    // ====================================================
    //                 Particle generation
    // ====================================================
    // Does not set displacements, velocities, etc.
    CabanaPD::Particles particles( memory_space{}, model_type{}, thermal_type{},
                                   low_corner, high_corner, num_cells,
                                   halo_width, exec_space{} );

    // ====================================================
    //            Custom particle initialization
    // ====================================================
    auto rho = particles.sliceDensity();
    auto init_functor = KOKKOS_LAMBDA( const int pid ) { rho( pid ) = rho0; };
    particles.updateParticles( exec_space{}, init_functor );

    // ====================================================
    //                    Force model
    // ====================================================
    auto force_model =
        CabanaPD::createForceModel( model_type{}, CabanaPD::Fracture{},
                                    particles, delta, K, G0, alpha, temp0 );

    // ====================================================
    //                   Create solver
    // ====================================================

    CabanaPD::Solver solver( inputs, particles, force_model );

    // --------------------------------------------
    //                Thermal shock
    // --------------------------------------------
    auto x = particles.sliceReferencePosition();
    auto temp = particles.sliceTemperature();

    // Plate limits
    double X0 = low_corner[0];
    double Xn = high_corner[0];
    double Y0 = low_corner[1];
    double Yn = high_corner[1];
    // This is purposely delayed until after solver init so that ghosted
    // particles are correctly taken into account for lambda capture here.
    auto temp_func = KOKKOS_LAMBDA( const int pid, const double t )
    {
        // Define a time-dependent surface temperature:
        // An inverted triangular pulse over a 2*t_ramp period starting at temp0
        // and linearly decreasing to temp_w within t_ramp, then linearly
        // increasing back to temp0, and finally staying constant at temp0
        double temp_infinity;
        if ( t <= t_ramp )
        {
            // Increasing pulse
            temp_infinity = temp0 - ( temp0 - temp_w ) * ( t / t_ramp );
        }
        else if ( t < 2 * t_ramp )
        {
            // Decreasing pulse
            temp_infinity =
                temp_w + ( temp0 - temp_w ) * ( t - t_ramp ) / t_ramp;
        }
        else
        {
            // Constant value
            temp_infinity = temp0;
        }

        // Rescale x and y particle position values
        double xi = ( 2.0 * x( pid, 0 ) - ( X0 + Xn ) ) / ( Xn - X0 );
        double eta = ( 2.0 * x( pid, 1 ) - ( Y0 + Yn ) ) / ( Yn - Y0 );

        // Define profile powers in x- and y-directions
        double sx = 1.0 / 50.0;
        double sy = 1.0 / 10.0;

        // Define profiles in x- and y-direcions
        double fx = 1.0 - Kokkos::pow( Kokkos::abs( xi ), 1.0 / sx );
        double fy = 1.0 - Kokkos::pow( Kokkos::abs( eta ), 1.0 / sy );

        // Compute particle temperature
        temp( pid ) = temp_infinity + ( temp0 - temp_infinity ) * fx * fy;
    };
    CabanaPD::BodyTerm body_term( temp_func, particles.size(), false );

    // ====================================================
    //                   Simulation run
    // ====================================================
    solver.init( body_term );
    solver.run( body_term );
}

// Initialize MPI+Kokkos.
int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    thermalCrackExample( argv[1] );

    Kokkos::finalize();
    MPI_Finalize();
}
