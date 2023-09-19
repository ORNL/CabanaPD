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

        std::array<int, 3> num_cell = { 41, 41, 41 };
        std::array<double, 3> low_corner = { -0.5, -0.5, -0.5 };
        std::array<double, 3> high_corner = { 0.5, 0.5, 0.5 };
        double t_final = 0.6;
        double dt = 0.01;
        int output_frequency = 5;
        double K = 1.0;
        double G = 0.5;
        double delta = 0.075;
        int m = std::floor(
            delta / ( ( high_corner[0] - low_corner[0] ) / num_cell[0] ) );
        int halo_width = m + 1; // Just to be safe.

        // Choose force model type.
        // using model_type =
        //    CabanaPD::ForceModel<CabanaPD::PMB, CabanaPD::Elastic>;
        // model_type force_model( delta, K );
        using model_type =
            CabanaPD::ForceModel<CabanaPD::LinearLPS, CabanaPD::Elastic>;
        model_type force_model( delta, K, G );

        CabanaPD::Inputs inputs( num_cell, low_corner, high_corner, t_final, dt,
                                 output_frequency );
        inputs.read_args( argc, argv );

        // Create particles from mesh.
        // Does not set displacements, velocities, etc.
        // FIXME: use createSolver to switch backend at runtime.
        using device_type = Kokkos::Device<exec_space, memory_space>;
        auto particles = std::make_shared<
            CabanaPD::Particles<device_type, typename model_type::base_model>>(
            exec_space(), inputs.low_corner, inputs.high_corner,
            inputs.num_cells, halo_width );

        // Define particle initialization.
        auto x = particles->sliceRefPosition();
        auto u = particles->sliceDisplacement();
        auto v = particles->sliceVelocity();
        auto rho = particles->sliceDensity();

        auto init_functor = KOKKOS_LAMBDA( const int pid )
        {
            double a = 0.001;
            double r0 = 0.25;
            double l = 0.07;
            double norm = std::sqrt( x( pid, 0 ) * x( pid, 0 ) +
                                     x( pid, 1 ) * x( pid, 1 ) +
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
            rho( pid ) = 100.0;
        };
        particles->updateParticles( exec_space{}, init_functor );

        auto cabana_pd = CabanaPD::createSolverElastic<device_type>(
            inputs, particles, force_model );
        cabana_pd->init_force();
        cabana_pd->run();

        x = particles->sliceRefPosition();
        u = particles->sliceDisplacement();
        double num_cell_x = inputs.num_cells[0];
        auto profile = Kokkos::View<double* [2], memory_space>(
            Kokkos::ViewAllocateWithoutInitializing( "displacement_profile" ),
            num_cell_x );
        double length = ( high_corner[0] - low_corner[0] );
        int mpi_rank;
        MPI_Comm_rank( MPI_COMM_WORLD, &mpi_rank );
        Kokkos::View<int*, memory_space> count( "c", 1 );
        auto measure_profile = KOKKOS_LAMBDA( const int pid )
        {
            double dx = length / num_cell_x;
            if ( x( pid, 1 ) < dx / 2.0 && x( pid, 1 ) > -dx / 2.0 &&
                 x( pid, 2 ) < dx / 2.0 && x( pid, 2 ) > -dx / 2.0 )
            {
                auto c = Kokkos::atomic_fetch_add( &count( 0 ), 1 );
                profile( c, 0 ) = x( pid, 0 );
                profile( c, 1 ) = u( pid, 0 );
            }
        };
        Kokkos::RangePolicy<exec_space> policy( 0, x.size() );
        Kokkos::parallel_for( "displacement_profile", policy, measure_profile );
        auto count_host =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, count );
        auto profile_host =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, profile );
        std::fstream fout;
        std::string file_name = "displacement_profile.txt";
        fout.open( file_name, std::ios::app );
        for ( int p = 0; p < count_host( 0 ); p++ )
        {
            fout << mpi_rank << " " << profile_host( p, 0 ) << " "
                 << profile_host( p, 1 ) << std::endl;
        }
    }

    MPI_Finalize();
}
