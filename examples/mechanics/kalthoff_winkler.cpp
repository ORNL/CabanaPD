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

// Simulate the Kalthoff-Winkler experiment of crack propagation in
// a pre-notched steel plate due to impact.
void kalthoffWinklerExample( const std::string filename )
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
    double nu = 1.0 / 3.0;
    double K = E / ( 3.0 * ( 1.0 - 2.0 * nu ) );
    double G0 = inputs["fracture_energy"];
    // double G = E / ( 2.0 * ( 1.0 + nu ) ); // Only for LPS.
    double delta = inputs["horizon"];
    delta += 1e-10;

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
    //                   Pre-notches
    // ====================================================
    std::array<double, 3> system_size = inputs["system_size"];
    double height = system_size[0];
    double width = system_size[1];
    double thickness = system_size[2];
    double L_prenotch = height / 2.0;
    double y_prenotch1 = -width / 8.0;
    double y_prenotch2 = width / 8.0;
    double low_x = low_corner[0];
    double low_z = low_corner[2];
    Kokkos::Array<double, 3> p01 = { low_x, y_prenotch1, low_z };
    Kokkos::Array<double, 3> p02 = { low_x, y_prenotch2, low_z };
    Kokkos::Array<double, 3> v1 = { L_prenotch, 0, 0 };
    Kokkos::Array<double, 3> v2 = { 0, 0, thickness };
    Kokkos::Array<Kokkos::Array<double, 3>, 2> notch_positions = { p01, p02 };
    CabanaPD::Prenotch<2> prenotch( v1, v2, notch_positions );

    // ====================================================
    //                    Force model
    // ====================================================
    using model_type = CabanaPD::ForceModel<CabanaPD::PMB, CabanaPD::Fracture>;
    model_type force_model( delta, K, G0 );
    // using model_type =
    //     CabanaPD::ForceModel<CabanaPD::LPS, CabanaPD::Fracture>;
    // model_type force_model( delta, K, G, G0 );

    // ====================================================
    //                 Particle generation
    // ====================================================
    // Does not set displacements, velocities, etc.
    auto particles = CabanaPD::createParticles<memory_space, model_type>(
        exec_space(), low_corner, high_corner, num_cells, halo_width );

    // ====================================================
    //            Custom particle initialization
    // ====================================================
    auto rho = particles->sliceDensity();
    auto x = particles->sliceReferencePosition();
    auto v = particles->sliceVelocity();
    auto f = particles->sliceForce();

    double dx = particles->dx[0];
    double v0 = inputs["impactor_velocity"];
    auto init_functor = KOKKOS_LAMBDA( const int pid )
    {
        // Density
        rho( pid ) = rho0;
        // x velocity between the pre-notches
        if ( x( pid, 1 ) > y_prenotch1 && x( pid, 1 ) < y_prenotch2 &&
             x( pid, 0 ) < -0.5 * height + dx )
            v( pid, 0 ) = v0;
    };
    particles->updateParticles( exec_space{}, init_functor );

    // ====================================================
    //                   Create solver
    // ====================================================
    auto cabana_pd = CabanaPD::createSolverDHPD<memory_space>(
        inputs, particles, force_model, prenotch );

    // ====================================================
    //                Boundary conditions
    // ====================================================
    // Create BC last to ensure ghost particles are included.
    double x_bc = -0.5 * height;
    CabanaPD::RegionBoundary<CabanaPD::RectangularPrism> plane(
        x_bc - dx, x_bc + dx, y_prenotch1 - 0.25 * dx, y_prenotch2 + 0.25 * dx,
        -thickness, thickness );
    auto bc = createBoundaryCondition( CabanaPD::ForceValueBCTag{}, 0.0,
                                       exec_space{}, *particles, plane );

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

    kalthoffWinklerExample( argv[1] );

    Kokkos::finalize();
    MPI_Finalize();
}
