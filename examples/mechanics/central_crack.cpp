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

// Simulate a square plate with a central crack under tension.
void centralCrackExample( const std::string filename )
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
    double rho0 = inputs["density"];
    double E = inputs["elastic_modulus"];
    double nu = 0.25; // Use bond-based model
    double K = E / ( 3 * ( 1 - 2 * nu ) );
    double G0 = inputs["fracture_energy"];
    double delta = inputs["horizon"];
    delta += 1e-10;

    // ====================================================
    //                  Discretization
    // ====================================================
    // Set halo width based on delta
    std::array<double, 3> low_corner = inputs["low_corner"];
    std::array<double, 3> high_corner = inputs["high_corner"];

    // ====================================================
    //                    Central Crack
    // ====================================================
    double plate_width = inputs["system_size"][1];  // Plate width
    double plate_length = inputs["system_size"][0]; // Plate length
    double thickness = inputs["system_size"][2];    // Plate thickness
    double crack_length = 0.01;                     // Crack length is 10 mm

    // Assume the plate is symmetric, ranging from -0.025 to +0.025
    double plate_center_x = 0.0;  // Center is at 0 for symmetric coordinates
    double plate_center_y = 0.0;  // Center is at 0 for symmetric coordinates

    // Define the crack start and end points for a horizontal crack
    Kokkos::Array<double, 3> crack_start = { 
        plate_center_x - (crack_length / 2.0), // Center the crack along the x-axis
        plate_center_y,                        // Center the crack along the y-axis
        0.0                                   // Assume the crack lies in the z=0 plane
    };
    Kokkos::Array<double, 3> crack_end = { 
        plate_center_x + (crack_length / 2.0), // Center the crack along the x-axis
        plate_center_y,                        // Center the crack along the y-axis
        0.0                                   // Assume the crack lies in the z=0 plane
    };

    // Compute the vector defining the crack
    Kokkos::Array<double, 3> crack_vector = { 
        crack_end[0] - crack_start[0], 
        crack_end[1] - crack_start[1], 
        crack_end[2] - crack_start[2] 
    };

    // Define the crack using Prenotch
    Kokkos::Array<Kokkos::Array<double, 3>, 2> notch_positions = { crack_start, crack_end };
    CabanaPD::Prenotch<2> prenotch( crack_vector, { 0.0, 0.0, thickness }, notch_positions );

    // ====================================================
    //                    Force model
    // ====================================================
    using model_type = CabanaPD::ForceModel<CabanaPD::PMB, CabanaPD::Fracture>;
    model_type force_model( delta, K, G0 );

    // ====================================================
    //                 Particle generation
    // ====================================================
    auto particles = CabanaPD::createParticles<memory_space, model_type>(
        exec_space{}, inputs, CabanaPD::EnergyStressOutput{} );

    // ====================================================
    //                Boundary conditions planes
    // ====================================================
    double dy = particles->dx[1];
    CabanaPD::RegionBoundary<CabanaPD::RectangularPrism> plane1(
        low_corner[0], high_corner[0], low_corner[1] - dy, low_corner[1] + dy,
        low_corner[2], high_corner[2] );
    CabanaPD::RegionBoundary<CabanaPD::RectangularPrism> plane2(
        low_corner[0], high_corner[0], high_corner[1] - dy, high_corner[1] + dy,
        low_corner[2], high_corner[2] );
    std::vector<CabanaPD::RegionBoundary<CabanaPD::RectangularPrism>> planes = {
        plane1, plane2 };

    // ====================================================
    //            Custom particle initialization
    // ====================================================
    auto rho = particles->sliceDensity();
    auto x = particles->sliceReferencePosition();
    auto v = particles->sliceVelocity();
    auto f = particles->sliceForce();
    auto nofail = particles->sliceNoFail();

    auto init_functor = KOKKOS_LAMBDA( const int pid )
    {
        // Density
        rho( pid ) = rho0;
        // No-fail zone
        if ( x( pid, 1 ) <= plane1.low_y + delta + 1e-10 ||
             x( pid, 1 ) >= plane2.high_y - delta - 1e-10 )
            nofail( pid ) = 1;
    };
    particles->updateParticles( exec_space{}, init_functor );

    // ====================================================
    //                   Create solver
    // ====================================================
    auto cabana_pd = CabanaPD::createSolverFracture<memory_space>(
        inputs, particles, force_model, prenotch );

    // ====================================================
    //                Boundary conditions
    // ====================================================
    // Create BC last to ensure ghost particles are included.
    double sigma0 = inputs["traction"];
    double b0 = sigma0 / dy;
    f = particles->sliceForce();
    x = particles->sliceReferencePosition();
    // Create a symmetric force BC in the y-direction.
    auto bc_op = KOKKOS_LAMBDA( const int pid, const double )
    {
        auto ypos = x( pid, 1 );
        auto sign = (ypos > (low_corner[1] + high_corner[1]) / 2.0) ? 1.0 : -1.0;
        f( pid, 1 ) += b0 * sign;
    };
    auto bc = createBoundaryCondition( bc_op, exec_space{}, *particles, planes,
                                       true );

    // ====================================================
    //                   Simulation run
    // ====================================================
    cabana_pd->init();
    cabana_pd->run( bc );
}

// Initialize MPI+Kokkos.
int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    centralCrackExample( argv[1] );

    Kokkos::finalize();
    MPI_Finalize();
}
