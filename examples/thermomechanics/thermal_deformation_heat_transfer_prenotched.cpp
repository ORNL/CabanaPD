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

// Simulate heat transfer in a pre-notched pseudo-1d cube.
void thermalDeformationHeatTransferPrenotchedExample(
    const std::string filename )
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
    //                    Pre-notches
    // ====================================================
    // Number of pre-notches
    constexpr int Npn = 2;

    double height = inputs["system_size"][0];
    double width = inputs["system_size"][1];
    double thickness = inputs["system_size"][2];
    double dy = particles.dx[1];

    // Initialize pre-notch arrays
    Kokkos::Array<Kokkos::Array<double, 3>, Npn> notch_positions;
    Kokkos::Array<Kokkos::Array<double, 3>, Npn> notch_v1;
    Kokkos::Array<Kokkos::Array<double, 3>, Npn> notch_v2;

    // We intentionally translate the pre-notches vertically by 0.5 * dy
    // to prevent particles from sitting exactly on the pre-notches.

    // Pre-notch 1
    Kokkos::Array<double, 3> p01 = { low_corner[0] + 0.25 * height,
                                     low_corner[1] + 0.25 * height + 0.5 * dy,
                                     low_corner[2] };
    notch_positions[0] = p01;
    Kokkos::Array<double, 3> v11 = { 0.5 * height, 0.5 * width, 0 };
    notch_v1[0] = v11;
    Kokkos::Array<double, 3> v2 = { 0, 0, thickness };
    notch_v2[0] = v2;

    // Pre-notch 2
    Kokkos::Array<double, 3> p02 = { low_corner[0] + 0.25 * height,
                                     high_corner[1] - 0.25 * height + 0.5 * dy,
                                     low_corner[2] };
    notch_positions[1] = p02;
    Kokkos::Array<double, 3> v12 = { 0.5 * height, -0.5 * width, 0 };
    notch_v1[1] = v12;
    notch_v2[1] = v2;

    CabanaPD::Prenotch<Npn> prenotch( notch_v1, notch_v2, notch_positions );

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
    CabanaPD::ForceModel force_model( model_type{}, delta, K, G0, temp, kappa,
                                      cp, alpha, temp0 );

    // ====================================================
    //                   Create solver
    // ====================================================
    CabanaPD::Solver solver( inputs, particles, force_model );

    // ====================================================
    //                   Boundary condition
    // ====================================================
    // Temperature profile imposed on top surface
    using plane_type = CabanaPD::Region<CabanaPD::RectangularPrism>;

    // Top surface
    plane_type plane( low_corner[0], high_corner[0], high_corner[1] - dy,
                      high_corner[1] + dy, low_corner[2], high_corner[2] );

    // This is purposely delayed until after solver init so that ghosted
    // particles are correctly taken into account for lambda capture here.
    temp = solver.particles.sliceTemperature();
    auto temp_bc = KOKKOS_LAMBDA( const int pid, const double )
    {
        temp( pid ) = 0.0;
    };

    auto bc = CabanaPD::createBoundaryCondition(
        temp_bc, exec_space{}, solver.particles, false, plane );

    // ====================================================
    //                   Simulation run
    // ====================================================
    solver.init( bc, prenotch );
    solver.run( bc );

    // ====================================================
    //                      Outputs
    // ====================================================
    // Output temperature along the y-axis
    int profile_dim = 1;
    auto value = KOKKOS_LAMBDA( const int pid ) { return temp( pid ); };
    std::string file_name = "temperature_yaxis_profile.txt";
    createOutputProfile( MPI_COMM_WORLD, num_cells[1], profile_dim, file_name,
                         solver.particles, value );
}

// Initialize MPI+Kokkos.
int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    thermalDeformationHeatTransferPrenotchedExample( argv[1] );

    Kokkos::finalize();
    MPI_Finalize();
}
