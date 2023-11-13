/****************************************************************************
 * Copyright (c) 2022-2023 by Oak Ridge National Laboratory                 *
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

        CabanaPD::Inputs inputs( argv[1] );

        // Material constants
        double E = inputs["elastic_modulus"]["value"];
        double nu = 1.0 / 3.0;
        double K = E / ( 3.0 * ( 1.0 - 2.0 * nu ) );
        double rho0 = inputs["density"]["value"];
        double G0 = inputs["fracture_energy"]["value"];
        // double G = E / ( 2.0 * ( 1.0 + nu ) ); // Only for LPS.

        // PD horizon
        double delta = inputs["horizon"]["value"];
        delta += 1e-10;

        // Impactor velocity
        double v0 = inputs["impactor_velocity"]["value"];

        // FIXME: set halo width based on delta
        std::array<double, 3> low_corner = inputs["low_corner"]["value"];
        std::array<double, 3> high_corner = inputs["high_corner"]["value"];
        std::array<int, 3> num_cells = inputs["num_cells"]["value"];
        int m = std::floor(
            delta / ( ( high_corner[0] - low_corner[0] ) / num_cells[0] ) );
        int halo_width = m + 1; // Just to be safe.

        // Prenotches
        std::array<double, 3> system_size = inputs["system_size"]["value"];
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
        Kokkos::Array<Kokkos::Array<double, 3>, 2> notch_positions = { p01,
                                                                       p02 };
        CabanaPD::Prenotch<2> prenotch( v1, v2, notch_positions );

        // Choose force model type.
        using model_type =
            CabanaPD::ForceModel<CabanaPD::PMB, CabanaPD::Fracture>;
        model_type force_model( delta, K, G0 );
        // using model_type =
        //     CabanaPD::ForceModel<CabanaPD::LPS, CabanaPD::Fracture>;
        // model_type force_model( delta, K, G, G0 );

        // Create particles from mesh.
        // Does not set displacements, velocities, etc.
        auto particles = std::make_shared<
            CabanaPD::Particles<memory_space, typename model_type::base_model>>(
            exec_space(), low_corner, high_corner, num_cells, halo_width );

        // Define particle initialization.
        auto x = particles->sliceReferencePosition();
        auto v = particles->sliceVelocity();
        auto f = particles->sliceForce();
        auto rho = particles->sliceDensity();

        double dx = particles->dx[0];
        double x_bc = -0.5 * height;
        CabanaPD::RegionBoundary plane(
            x_bc - dx, x_bc + dx * 1.25, y_prenotch1 - dx * 0.25,
            y_prenotch2 + dx * 0.25, -thickness, thickness );
        std::vector<CabanaPD::RegionBoundary> planes = { plane };
        auto bc =
            createBoundaryCondition( CabanaPD::ForceValueBCTag{}, exec_space{},
                                     *particles, planes, 0.0 );

        auto init_functor = KOKKOS_LAMBDA( const int pid )
        {
            rho( pid ) = rho0;
            // Set the x velocity inside the pre-notches.
            if ( x( pid, 1 ) > y_prenotch1 && x( pid, 1 ) < y_prenotch2 &&
                 x( pid, 0 ) < -0.5 * height + dx )
                v( pid, 0 ) = v0;
        };
        particles->updateParticles( exec_space{}, init_functor );

        auto cabana_pd = CabanaPD::createSolverFracture<memory_space>(
            inputs, particles, force_model, bc, prenotch );
        cabana_pd->init_force();
        cabana_pd->run();
    }

    MPI_Finalize();
}
