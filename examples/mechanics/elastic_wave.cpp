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

// Simulate elastic wave propagation from an initial displacement field.
void elasticWaveExample( const std::string filename )
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
    auto K = inputs["bulk_modulus"];
    double G = inputs["shear_modulus"];
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
    using model_type =
        CabanaPD::ForceModel<CabanaPD::LinearLPS, CabanaPD::Elastic>;
    model_type force_model( delta, K, G );

    // ====================================================
    //                 Particle generation
    // ====================================================
    auto particles = CabanaPD::createParticles<memory_space, model_type>(
        exec_space(), low_corner, high_corner, num_cells, halo_width );

    // ====================================================
    //            Custom particle initialization
    // ====================================================
    auto rho = particles->sliceDensity();
    auto x = particles->sliceReferencePosition();
    auto u = particles->sliceDisplacement();
    auto v = particles->sliceVelocity();

    auto init_functor = KOKKOS_LAMBDA( const int pid )
    {
        // Density
        rho( pid ) = rho0;

        // Initial conditions: displacements and velocities
        double a = 0.001;
        double r0 = 0.25;
        double l = 0.07;
        double norm =
            std::sqrt( x( pid, 0 ) * x( pid, 0 ) + x( pid, 1 ) * x( pid, 1 ) +
                       x( pid, 2 ) * x( pid, 2 ) );
        double diff = norm - r0;
        double arg = diff * diff / l / l;
        for ( int d = 0; d < 3; d++ )
        {
            double comp = 0.0;
            if ( norm > 0.0 )
                comp = x( pid, d ) / norm;
            u( pid, d ) = a * std::exp( -arg ) * comp;
            v( pid, d ) = 0.0;
        }
    };
    particles->updateParticles( exec_space{}, init_functor );

    // ====================================================
    //                   Create solver
    // ====================================================
    auto cabana_pd = CabanaPD::createSolverElastic<memory_space>(
        inputs, particles, force_model );

    // ====================================================
    //                   Simulation run
    // ====================================================
    cabana_pd->init();
    cabana_pd->run();

    // ====================================================
    //                      Outputs
    // ====================================================
    // Output displacement along the x-axis
    createDisplacementProfile( MPI_COMM_WORLD, num_cells[0], 0,
                               "displacement_profile.txt", *particles );
}

// Initialize MPI+Kokkos.
int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    elasticWaveExample( argv[1] );

    Kokkos::finalize();
    MPI_Finalize();
}
