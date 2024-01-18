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
        double E = inputs["elastic_modulus"];
        double rho0 = inputs["density"];
        double nu = 0.25;                      // unitless
        double K = E / ( 3 * ( 1 - 2 * nu ) ); // [Pa]
        double delta = inputs["horizon"];

        double alpha = inputs["thermal_coeff"]; // [1/oC]
        // Reference temperature
        // double temp0 = 0.0;

        std::array<double, 3> low_corner = inputs["low_corner"];
        std::array<double, 3> high_corner = inputs["high_corner"];
        std::array<int, 3> num_cells = inputs["num_cells"];
        int m = std::floor(
            delta / ( ( high_corner[0] - low_corner[0] ) / num_cells[0] ) );
        int halo_width = m + 1; // Just to be safe.

        // Choose force model type.
        using model_type =
            CabanaPD::ForceModel<CabanaPD::PMB, CabanaPD::Elastic>;
        // model_type force_model( delta, K );
        model_type force_model( delta, K, alpha );
        // using model_type =
        //     CabanaPD::ForceModel<CabanaPD::LinearLPS, CabanaPD::Elastic>;
        // model_type force_model( delta, K, G );

        // Create particles from mesh.
        // Does not set displacements, velocities, etc.
        auto particles = std::make_shared<
            CabanaPD::Particles<memory_space, typename model_type::base_model>>(
            low_corner, high_corner, num_cells, halo_width );
        // Do not create particles in the center.
        double x_center = 0.0;
        double y_center = -0.0005;
        double radius = inputs["radius"];
        auto init_op = KOKKOS_LAMBDA( const int, const double x[3] )
        {
            if ( ( ( x[0] - x_center ) * ( x[0] - x_center ) +
                   ( x[1] - y_center ) * ( x[1] - y_center ) ) <
                 radius * radius )
                return false;
            return true;
        };
        particles->createParticles( exec_space(), init_op );

        // Define particle initialization.
        auto x = particles->sliceReferencePosition();
        auto u = particles->sliceDisplacement();
        auto f = particles->sliceForce();
        auto v = particles->sliceVelocity();
        auto rho = particles->sliceDensity();
        // auto temp = particles->sliceTemperature();

        // Domain to apply b.c.
        CabanaPD::RegionBoundary domain1( low_corner[0], high_corner[0],
                                          low_corner[1], high_corner[1],
                                          low_corner[2], high_corner[2] );
        std::vector<CabanaPD::RegionBoundary> domain = { domain1 };

        auto bc = createBoundaryCondition( CabanaPD::TempBCTag{}, 5000.0,
                                           exec_space{}, *particles, domain );

        auto init_functor = KOKKOS_LAMBDA( const int pid )
        {
            // temp( pid ) = 5000 * x( pid, 1 ) * t_final;
            rho( pid ) = rho0;
        };
        particles->updateParticles( exec_space{}, init_functor );

        auto cabana_pd = CabanaPD::createSolverElastic<memory_space>(
            inputs, particles, force_model, bc );
        cabana_pd->init_force();
        cabana_pd->run();

        double num_cell_x = num_cells[0];
        auto profile = Kokkos::View<double* [2], memory_space>(
            Kokkos::ViewAllocateWithoutInitializing( "displacement_profile" ),
            num_cell_x );
        int mpi_rank;
        MPI_Comm_rank( MPI_COMM_WORLD, &mpi_rank );
        Kokkos::View<int*, memory_space> count( "c", 1 );
    }

    MPI_Finalize();
}
