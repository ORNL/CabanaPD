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

// Simulate heat transfer in a pseudo-1d cube.
void thermalDeformationHeatTransferExample( const std::string filename )
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
    double delta = inputs["horizon"];
    delta += 1e-10;
    double alpha = inputs["thermal_expansion_coeff"];
    double kappa = inputs["thermal_conductivity"];
    double cp = inputs["specific_heat_capacity"];

    // Problem parameters
    double temp0 = inputs["reference_temperature"];

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
    using thermal_type = CabanaPD::DynamicTemperature;

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
    auto temp = particles.sliceTemperature();
    auto init_functor = KOKKOS_LAMBDA( const int pid )
    {
        // Density
        rho( pid ) = rho0;
        // Temperature
        temp( pid ) = temp0;
    };
    particles.updateParticles( exec_space{}, init_functor );

    // ====================================================
    //                    Force model
    // ====================================================
    auto force_model = CabanaPD::createForceModel(
        model_type{}, CabanaPD::NoFracture{}, particles, delta, K, kappa, cp,
        alpha, temp0 );

    // ====================================================
    //                   Create solver
    // ====================================================
    auto cabana_pd =
        CabanaPD::createSolver<memory_space>( inputs, particles, force_model );

    // ====================================================
    //                   Boundary condition
    // ====================================================
    // Temperature profile imposed on top and bottom surfaces
    double dy = particles.dx[1];
    using plane_type = CabanaPD::RegionBoundary<CabanaPD::RectangularPrism>;

    // Top surface
    plane_type plane1( low_corner[0], high_corner[0], high_corner[1] - dy,
                       high_corner[1] + dy, low_corner[2], high_corner[2] );

    // Bottom surface
    plane_type plane2( low_corner[0], high_corner[0], low_corner[1] - dy,
                       low_corner[1] + dy, low_corner[2], high_corner[2] );

    // This is purposely delayed until after solver init so that ghosted
    // particles are correctly taken into account for lambda capture here.
    temp = particles.sliceTemperature();
    auto temp_bc = KOKKOS_LAMBDA( const int pid, const double )
    {
        temp( pid ) = 0.0;
    };

    auto bc = CabanaPD::createBoundaryCondition(
        temp_bc, exec_space{}, particles, false, plane1, plane2 );

    // ====================================================
    //                   Simulation run
    // ====================================================
    cabana_pd->init( bc );
    cabana_pd->run( bc );

    // ====================================================
    //                      Outputs
    // ====================================================
    // Output temperature along the y-axis
    int profile_dim = 1;
    auto value = KOKKOS_LAMBDA( const int pid ) { return temp( pid ); };
    std::string file_name = "temperature_yaxis_profile.txt";
    createOutputProfile( MPI_COMM_WORLD, num_cells[1], profile_dim, file_name,
                         particles, value );
}

// Initialize MPI+Kokkos.
int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    thermalDeformationHeatTransferExample( argv[1] );

    Kokkos::finalize();
    MPI_Finalize();
}
