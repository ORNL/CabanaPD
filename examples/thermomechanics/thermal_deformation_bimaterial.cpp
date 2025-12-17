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
    std::array<double, 2> alpha = inputs["thermal_expansion_coeff"];

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
    auto x = particles.sliceReferencePosition();
    auto type = particles.sliceType();
    double half_height = ( high_corner[2] - low_corner[2] ) / 2.0;
    auto init_functor = KOKKOS_LAMBDA( const int pid )
    {
        rho( pid ) = rho0;
        if ( x( pid, 2 ) <= half_height )
            type( pid ) = 0;
        else
            type( pid ) = 1;
    };
    particles.update( exec_space{}, init_functor );

    // ====================================================
    //                    Force model
    // ====================================================
    auto temp = particles.sliceTemperature();
    CabanaPD::ForceModel force_model_upper( model_type{},
                                            CabanaPD::NoFracture{}, horizon, K,
                                            temp, alpha[0], temp0 );
    CabanaPD::ForceModel force_model_lower( model_type{},
                                            CabanaPD::NoFracture{}, horizon, K,
                                            temp, alpha[1], temp0 );

    auto models =
        CabanaPD::createMultiForceModel( particles, CabanaPD::AverageTag{},
                                         force_model_upper, force_model_lower );
    // ====================================================
    //                   Create solver
    // ====================================================
    CabanaPD::Solver solver( inputs, particles, models );

    // ====================================================
    //                   Imposed field
    // ====================================================
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
    CabanaPD::createDisplacementProfile( "zdisplacement_xaxis_profile.txt",
                                         solver.particles, 0, 2 );

    // Output y-displacement along the y-axis
    CabanaPD::createDisplacementProfile( "zdisplacement_yaxis_profile.txt",
                                         solver.particles, 1, 2 );

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
