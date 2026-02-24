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

// Simulate thermally-induced deformation in a rectangular plate.
void thermalDeformationExample( const std::string filename )
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
    double horizon = inputs["horizon"];
    horizon += 1e-10;
    double alpha = inputs["thermal_expansion_coeff"];

    // Problem parameters
    double temp0 = inputs["reference_temperature"];

    // ====================================================
    //                  Discretization
    // ====================================================
    std::array<double, 3> low_corner = inputs["low_corner"];
    std::array<double, 3> high_corner = inputs["high_corner"];
    std::array<int, 3> num_cells = inputs["num_cells"];
    int m = std::floor( horizon /
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
    CabanaPD::Particles particles( memory_space{}, model_type{},
                                   thermal_type{} );
    particles.domain( low_corner, high_corner, num_cells, halo_width );
    particles.create( exec_space{} );

    // ====================================================
    //            Custom particle initialization
    // ====================================================
    auto rho = particles.sliceDensity();
    auto init_functor = KOKKOS_LAMBDA( const int pid ) { rho( pid ) = rho0; };
    particles.update( exec_space{}, init_functor );

    // ====================================================
    //                    Force model
    // ====================================================
    auto temp = particles.sliceTemperature();
    CabanaPD::MechanicsModel mechanics_model( model_type{}, horizon, K );
    CabanaPD::ThermalModel thermal_model( temp, alpha, temp0 );
    CabanaPD::Experimental::ThermalForceModel force_model( mechanics_model,
                                                           thermal_model );

    // ====================================================
    //                   Create solver
    // ====================================================
    CabanaPD::Solver solver( inputs, particles, force_model );

    // ====================================================
    //                   Imposed field
    // ====================================================
    auto x = solver.particles.sliceReferencePosition();
    temp = solver.particles.sliceTemperature();
    const double low_corner_y = low_corner[1];
    // This is purposely delayed until after solver init so that ghosted
    // particles are correctly taken into account for lambda capture here.
    auto temp_func = KOKKOS_LAMBDA( const int pid, const double t )
    {
        temp( pid ) = temp0 + 5000.0 * ( x( pid, 1 ) - low_corner_y ) * t;
    };
    CabanaPD::BodyTerm body_term( temp_func, solver.particles.size(), false );

    // ====================================================
    //                   Simulation run
    // ====================================================
    solver.init( body_term );
    solver.run( body_term );

    // ====================================================
    //                      Outputs
    // ====================================================
    // Output y-displacement along the x-axis
    CabanaPD::createDisplacementProfile( "ydisplacement_xaxis_profile.txt",
                                         solver.particles, 0, 1 );

    // Output y-displacement along the y-axis
    CabanaPD::createDisplacementProfile( "ydisplacement_yaxis_profile.txt",
                                         solver.particles, 1, 1 );

    // Output displacement magnitude along the x-axis
    CabanaPD::createDisplacementMagnitudeProfile(
        "displacement_magnitude_xaxis_profile.txt", solver.particles, 0 );

    // Output displacement magnitude along the y-axis
    CabanaPD::createDisplacementMagnitudeProfile(
        "displacement_magnitude_yaxis_profile.txt", solver.particles, 1 );
}

// Initialize MPI+Kokkos.
int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    thermalDeformationExample( argv[1] );

    Kokkos::finalize();
    MPI_Finalize();
}
