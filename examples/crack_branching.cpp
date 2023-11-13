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
        double nu = 0.25;
        double K = E / ( 3 * ( 1 - 2 * nu ) );
        double rho0 = inputs["density"]["value"];
        double G0 = inputs["fracture_energy"]["value"];

        // PD horizon
        double delta = inputs["horizon"]["value"];
        delta += 1e-10;

        // FIXME: set halo width based on delta
        std::array<double, 3> low_corner = inputs["low_corner"]["value"];
        std::array<double, 3> high_corner = inputs["high_corner"]["value"];
        std::array<int, 3> num_cells = inputs["num_cells"]["value"];
        int m = std::floor(
            delta / ( ( high_corner[0] - low_corner[0] ) / num_cells[0] ) );
        int halo_width = m + 1; // Just to be safe.

        // Prenotch
        double height = inputs["system_size"]["value"][0];
        double thickness = inputs["system_size"]["value"][2];
        double L_prenotch = height / 2.0;
        double y_prenotch1 = 0.0;
        Kokkos::Array<double, 3> p01 = { low_corner[0], y_prenotch1,
                                         low_corner[2] };
        Kokkos::Array<double, 3> v1 = { L_prenotch, 0, 0 };
        Kokkos::Array<double, 3> v2 = { 0, 0, thickness };
        Kokkos::Array<Kokkos::Array<double, 3>, 1> notch_positions = { p01 };
        CabanaPD::Prenotch<1> prenotch( v1, v2, notch_positions );

        // Choose force model type.
        using model_type =
            CabanaPD::ForceModel<CabanaPD::PMB, CabanaPD::Fracture>;
        model_type force_model( delta, K, G0 );

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
        auto nofail = particles->sliceNoFail();

        // Relying on uniform grid here
        double dy = particles->dx[1];
        double sigma0 = inputs["traction"]["value"];
        double b0 = sigma0 / dy;

        CabanaPD::RegionBoundary plane1(
            low_corner[0], high_corner[0], low_corner[1] - dy,
            high_corner[1] + dy, low_corner[2], high_corner[2] );
        CabanaPD::RegionBoundary plane2(
            low_corner[0], high_corner[0], low_corner[1] - dy,
            high_corner[1] + dy, low_corner[2], high_corner[2] );
        std::vector<CabanaPD::RegionBoundary> planes = { plane1, plane2 };
        auto bc =
            createBoundaryCondition( CabanaPD::ForceCrackBranchBCTag{},
                                     exec_space{}, *particles, planes, b0 );

        auto init_functor = KOKKOS_LAMBDA( const int pid )
        {
            rho( pid ) = rho0;
            // Set the no-fail zone.
            if ( x( pid, 1 ) <= plane1.low_y + delta + 1e-10 ||
                 x( pid, 1 ) >= plane2.high_y - delta - 1e-10 )
                nofail( pid ) = 1;
        };
        particles->updateParticles( exec_space{}, init_functor );

        // FIXME: use createSolver to switch backend at runtime.
        auto cabana_pd = CabanaPD::createSolverFracture<memory_space>(
            inputs, particles, force_model, bc, prenotch );
        cabana_pd->init_force();
        cabana_pd->run();
    }

    MPI_Finalize();
}
