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

int main( int argc, char* argv[] )
{

    MPI_Init( &argc, &argv );

    {
        Kokkos::ScopeGuard scope_guard( argc, argv );

        // FIXME: change backend at compile time for now.
        using exec_space = Kokkos::DefaultExecutionSpace;
        using memory_space = typename exec_space::memory_space;

        // Plate dimension)
        double height = 0.1;      // [m] (100 mm)
        double width = 0.2;       // [m] (200 mm)
        double thickness = 0.009; // [m] (  9 mm)

        // Domain
        std::array<int, 3> num_cell = { 41, 81, 4 };
        std::array<double, 3> low_corner = { -0.5 * height, -0.5 * width,
                                             -0.5 * thickness };
        std::array<double, 3> high_corner = { 0.5 * height, 0.5 * width,
                                              0.5 * thickness };
        double t_final = 140e-6;
        double dt = 0.2e-6;

        // Material constants
        double E = 191e+9;                           // [Pa]
        double nu = 1.0 / 3.0;                       // unitless
        double K = E / ( 3.0 * ( 1.0 - 2.0 * nu ) ); // [Pa]
        double rho0 = 8000;                          // [kg/m^3]
        double G0 = 42408;                           // [J/m^2]
        double G = E / ( 2.0 * ( 1.0 + nu ) );

        double v0 = 16;              // [m/sec] (Half impactor's velocity)
        double L_prenotch = 0.05;    // [m] (50 mm)
        double y_prenotch1 = -0.025; // [m] (-25 mm)
        double y_prenotch2 = 0.025;  // [m] ( 25 mm)
        Kokkos::Array<double, 3> p01 = { low_corner[0], y_prenotch1,
                                         low_corner[2] };
        Kokkos::Array<double, 3> p02 = { low_corner[0], y_prenotch2,
                                         low_corner[2] };
        Kokkos::Array<double, 3> v1 = { L_prenotch, 0, 0 };
        Kokkos::Array<double, 3> v2 = { 0, 0, thickness };
        Kokkos::Array<Kokkos::Array<double, 3>, 2> notch_positions = { p01,
                                                                       p02 };
        CabanaPD::Prenotch<2> prenotch( v1, v2, notch_positions );

        double delta = 0.0075;
        int m = std::floor(
            delta / ( ( high_corner[0] - low_corner[0] ) / num_cell[0] ) );
        int halo_width = m + 1; // Just to be safe.

        // Choose force model type.
        // CabanaPD::PMBDamageModel force_model( delta, K, G0 );
        CabanaPD::LPSDamageModel force_model( delta, K, G, G0 );
        CabanaPD::Inputs inputs( num_cell, low_corner, high_corner, t_final,
                                 dt );
        inputs.read_args( argc, argv );

        // Create particles from mesh.
        // Does not set displacements, velocities, etc.
        auto particles = std::make_shared<CabanaPD::Particles<memory_space>>(
            exec_space(), inputs.low_corner, inputs.high_corner,
            inputs.num_cells, halo_width );

        // Define particle initialization.
        auto x = particles->slice_x();
        auto v = particles->slice_v();
        auto f = particles->slice_f();
        auto rho = particles->slice_rho();

        double dx = particles->dx;

        double x_bc = -0.5 * height;
        CabanaPD::RegionBoundary plane(
            x_bc - dx, x_bc + dx * 1.25, y_prenotch1 - dx * 0.25,
            y_prenotch2 + dx * 0.25, -thickness, thickness );

        auto bc = createBoundaryCondition( exec_space{}, *particles, plane,
                                           CabanaPD::ForceBCTag{} );

        auto init_functor = KOKKOS_LAMBDA( const int pid )
        {
            rho( pid ) = rho0;
            // Set the x velocity inside the pre-notches.
            if ( x( pid, 1 ) > y_prenotch1 && x( pid, 1 ) < y_prenotch2 &&
                 x( pid, 0 ) < -0.5 * height + dx )
                v( pid, 0 ) = v0;
        };
        particles->update_particles( exec_space{}, init_functor );

        // FIXME: use createSolver to switch backend at runtime.
        using device_type = Kokkos::Device<exec_space, memory_space>;
        auto cabana_pd = CabanaPD::createSolverFracture<device_type>(
            inputs, particles, force_model, bc, prenotch );
        cabana_pd->init_force();
        cabana_pd->run();
    }

    MPI_Finalize();
}
