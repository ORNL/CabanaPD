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
void thermalDeformationHeatTransferExample( const std::string filename )
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
    //            Material and problem parameters
    // ====================================================
    // Material parameters
    double rho0 = inputs["density"];
    double E = inputs["elastic_modulus"];
    double nu = 0.25;
    double K = E / ( 3 * ( 1 - 2 * nu ) );
    double delta = inputs["horizon"];
    double alpha = inputs["thermal_expansion_coeff"];
    double kappa = inputs["thermal_conductivity"];
    double cp = inputs["specific_heat_capacity"];

    // Problem parameters
    double temp0 = inputs["reference_temperature"];

    // ====================================================
    //                  Discretization
    // ====================================================
    // FIXME: set halo width based on delta
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
    auto particles =
        std::make_shared<CabanaPD::Particles<memory_space, model_type,
                                             typename thermal_type::base_type>>(
            exec_space(), low_corner, high_corner, num_cells, halo_width );

    // ====================================================
    //            Custom particle initialization
    // ====================================================
    auto rho = particles->sliceDensity();
    auto temp = particles->sliceTemperature();
    auto init_functor = KOKKOS_LAMBDA( const int pid )
    {
        // Density
        rho( pid ) = rho0;
        // Temperature
        temp( pid ) = temp0;
    };
    particles->updateParticles( exec_space{}, init_functor );

    // ====================================================
    //                    Force model
    // ====================================================
    auto force_model = CabanaPD::createForceModel(
        model_type{}, CabanaPD::Elastic{}, *particles, delta, K, kappa, cp,
        alpha, temp0 );

    // ====================================================
    //                   Create solver
    // ====================================================
    auto cabana_pd = CabanaPD::createSolverElastic<memory_space>(
        inputs, particles, force_model );

    // ====================================================
    //                   Boundary condition
    // ====================================================
    // EXAMPLE 1: Temperature profile imposed over entire domain
    using plane_type = CabanaPD::RegionBoundary<CabanaPD::RectangularPrism>;
    /*
        plane_type plane( low_corner[0], high_corner[0],
                                        low_corner[1], high_corner[1],
                                        low_corner[2], high_corner[2] );

        std::vector<plane_type> planes = { plane };
    */

    // EXAMPLE 2: Temperature profile imposed on top, bottom, left, and right
    // surfaces
    double dx = particles->dx[0];
    double dy = particles->dx[1];
    double dz = particles->dx[2];

    // Left surface: x-direction
    plane_type plane1( low_corner[0] - dx, low_corner[0] + dx, low_corner[1],
                       high_corner[1], low_corner[2], high_corner[2] );

    // Right surface: x-direction
    plane_type plane2( high_corner[0] - dx, high_corner[0] + dx, low_corner[1],
                       high_corner[1], low_corner[2], high_corner[2] );

    // Top surface: y-direction
    plane_type plane3( low_corner[0], high_corner[0], high_corner[1] - dy,
                       high_corner[1] + dy, low_corner[2], high_corner[2] );

    // Bottom surface: y-direction
    plane_type plane4( low_corner[0], high_corner[0], low_corner[1] - dy,
                       low_corner[1] + dy, low_corner[2], high_corner[2] );

    // Front surface: z-direction
    plane_type plane5( low_corner[0], high_corner[0], low_corner[1],
                       high_corner[1], low_corner[2] - dz, low_corner[2] + dz );

    // Back surface: z-direction
    plane_type plane6( low_corner[0], high_corner[0], low_corner[1],
                       high_corner[1], high_corner[2] - dz,
                       high_corner[2] + dz );

    // std::vector<plane_type> planes = { plane1, plane2, plane3, plane4 };
    // std::vector<plane_type> planes = { plane1, plane2, plane3, plane4,
    // plane5, plane6 }; std::vector<plane_type> planes = { plane1, plane2 };
    std::vector<plane_type> planes = { plane3, plane4 };
    /*
        // EXAMPLE 3: Temperature profile imposed on top, bottom, left, and
       right
        // nonlocal boundaries (width delta) Top surface
        plane_type plane1(
            low_corner[0], high_corner[0], high_corner[1] - delta,
            high_corner[1] + delta, low_corner[2], high_corner[2] );

        // Bottom surface
        plane_type plane2(
            low_corner[0], high_corner[0], low_corner[1] - delta,
            low_corner[1] + delta, low_corner[2], high_corner[2] );

        // Left surface
        plane_type plane3(
            low_corner[0] - delta, low_corner[0] + delta, low_corner[1],
            high_corner[1], low_corner[2], high_corner[2] );

        // Right surface
        plane_type plane4(
            high_corner[0] - delta, high_corner[0] + delta, low_corner[1],
            high_corner[1], low_corner[2], high_corner[2] );

        std::vector<plane_type> planes = { plane1, plane2, plane3,
                                                         plane4 };
    */

    auto x = particles->sliceReferencePosition();
    // Need to reslice to include ghosted particles on the boundary.
    temp = particles->sliceTemperature();
    const double low_corner_y = low_corner[1];
    // This is purposely delayed until after solver init so that ghosted
    // particles are correctly taken into account for lambda capture here.
    auto temp_func = KOKKOS_LAMBDA( const int pid, const double t )
    {
        // temp( pid ) = temp0 + 5000.0 * ( x( pid, 1 ) - low_corner_y ) * t;
        temp( pid ) = 0.0;
    };
    auto bc = CabanaPD::createBoundaryCondition(
        temp_func, exec_space{}, *particles, planes, false, 1.0 );

    // ====================================================
    //                   Simulation run
    // ====================================================
    cabana_pd->init( bc );
    cabana_pd->run( bc );

    // ====================================================
    //                      Outputs
    // ====================================================

    // Output temperature along the x-axis
    auto temp_output = particles->sliceTemperature();
    auto value = KOKKOS_LAMBDA( const int pid ) { return temp_output( pid ); };

    int profile_dim = 0;
    std::string file_name = "temperature_xaxis_profile.txt";
    createOutputProfile( MPI_COMM_WORLD, num_cells[0], profile_dim, file_name,
                         *particles, value );

    /*
    // Output y-displacement along the x-axis
    createDisplacementProfile( MPI_COMM_WORLD,
                               "ydisplacement_xaxis_profile.txt", *particles,
                               num_cells[0], 0, 1 );

    // Output y-displacement along the y-axis
    createDisplacementProfile( MPI_COMM_WORLD,
                               "ydisplacement_yaxis_profile.txt", *particles,
                               num_cells[1], 1, 1 );

    // Output displacement magnitude along the x-axis
    createDisplacementMagnitudeProfile(
        MPI_COMM_WORLD, "displacement_magnitude_xaxis_profile.txt", *particles,
        num_cells[0], 0 );

    // Output displacement magnitude along the y-axis
    createDisplacementMagnitudeProfile(
        MPI_COMM_WORLD, "displacement_magnitude_yaxis_profile.txt", *particles,
        num_cells[1], 1 );

    */
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
