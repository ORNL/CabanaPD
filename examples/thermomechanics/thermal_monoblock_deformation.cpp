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
        // ====================================================
        //                  Setup Kokkos
        // ====================================================
        Kokkos::ScopeGuard scope_guard( argc, argv );

        using exec_space = Kokkos::DefaultExecutionSpace;
        using memory_space = typename exec_space::memory_space;

        // ====================================================
        //                   Read inputs
        // ====================================================
        CabanaPD::Inputs inputs( argv[1] );

        // ====================================================
        //                Material parameters
        // ====================================================
        double rho0 = inputs["density"];
        double E = inputs["elastic_modulus"];
        double nu = 0.25;
        double K = E / ( 3 * ( 1 - 2 * nu ) );
        double delta = inputs["horizon"];
        double alpha = inputs["thermal_coefficient"];
        double temp_ref = inputs["reference_temperature"];
        // Reference temperature
        // double temp0 = 0.0;

        // ====================================================
        //                  Discretization
        // ====================================================
        std::array<double, 3> low_corner = inputs["low_corner"];
        std::array<double, 3> high_corner = inputs["high_corner"];
        std::array<int, 3> num_cells = inputs["num_cells"];
        int m = std::floor(
            delta / ( ( high_corner[0] - low_corner[0] ) / num_cells[0] ) );
        int halo_width = m + 1; // Just to be safe.

        // ====================================================
        //                    Force model
        // ====================================================
        using model_type =
            CabanaPD::ForceModel<CabanaPD::PMB, CabanaPD::Elastic>;
        // model_type force_model( delta, K );
        // model_type force_model( delta, K, alpha );
        model_type force_model( delta, K, alpha, temp_ref );
        // using model_type =
        //     CabanaPD::ForceModel<CabanaPD::LinearLPS, CabanaPD::Elastic>;
        // model_type force_model( delta, K, G );

        // ====================================================
        //                 Particle generation
        // ====================================================
        // Does not set displacements, velocities, etc.
        auto particles = std::make_shared<
            CabanaPD::Particles<memory_space, typename model_type::base_model>>(
            exec_space(), low_corner, high_corner, num_cells, halo_width );

        // Do not create particles within given cylindrical region
        auto x = particles->sliceReferencePosition();
        double x_center = inputs["cylindrical_hole"][0];
        double y_center = inputs["cylindrical_hole"][1];
        double radius = inputs["cylindrical_hole"][2];
        auto init_op = KOKKOS_LAMBDA( const int, const double x[3] )
        {
            if ( ( ( x[0] - x_center ) * ( x[0] - x_center ) +
                   ( x[1] - y_center ) * ( x[1] - y_center ) ) <
                 radius * radius )
                return false;
            return true;
        };

        particles->createParticles( exec_space(), init_op );

        // ====================================================
        //                Boundary conditions
        // ====================================================
        auto temp = particles->sliceTemperature();
        // Reslice after updating size.
        x = particles->sliceReferencePosition();
        auto temp_func = KOKKOS_LAMBDA( const int pid, const double t )
        {
            // temp( pid ) = 5000.0 * ( x( pid, 1 ) - ( -0.014 ) ) * t;
            //  temp( pid ) = 5000.0 * ( x( pid, 1 ) - low_corner[1] ) * t;
            temp( pid ) = 2.0e+6 * ( x( pid, 1 ) - low_corner[1] ) * t;
        };
        auto body_term = CabanaPD::createBodyTerm( temp_func );

        // ====================================================
        //            Custom particle initialization
        // ====================================================
        auto rho = particles->sliceDensity();
        // auto x = particles->sliceReferencePosition();
        auto u = particles->sliceDisplacement();
        auto v = particles->sliceVelocity();
        auto f = particles->sliceForce();
        // auto temp = particles->sliceTemperature();

        auto init_functor = KOKKOS_LAMBDA( const int pid )
        {
            rho( pid ) = rho0;
        };
        particles->updateParticles( exec_space{}, init_functor );

        // ====================================================
        //                   Simulation run
        // ====================================================
        auto cabana_pd = CabanaPD::createSolverElastic<memory_space>(
            inputs, particles, force_model, body_term );
        cabana_pd->init_force();
        cabana_pd->run();

        // ====================================================
        //                      Outputs
        // ====================================================

        // ------------------------------------
        // Displacement profiles in x-direction
        // ------------------------------------

        double num_cell_x = num_cells[0];
        auto profile_x = Kokkos::View<double* [3], memory_space>(
            Kokkos::ViewAllocateWithoutInitializing( "displacement_profile" ),
            num_cell_x );
        int mpi_rank;
        MPI_Comm_rank( MPI_COMM_WORLD, &mpi_rank );
        Kokkos::View<int*, memory_space> count_x( "c", 1 );

        double dy = particles->dx[1];
        double dz = particles->dx[2];

        auto measure_profile = KOKKOS_LAMBDA( const int pid )
        {
            if ( x( pid, 1 ) < dy / 2.0 && x( pid, 1 ) > -dy / 2.0 &&
                 x( pid, 2 ) < dz / 2.0 && x( pid, 2 ) > -dz / 2.0 )
            {
                auto c = Kokkos::atomic_fetch_add( &count_x( 0 ), 1 );
                profile_x( c, 0 ) = x( pid, 0 );
                profile_x( c, 1 ) = u( pid, 1 );
                profile_x( c, 2 ) = std::sqrt( u( pid, 0 ) * u( pid, 0 ) +
                                               u( pid, 1 ) * u( pid, 1 ) +
                                               u( pid, 2 ) * u( pid, 2 ) );
            }
        };
        Kokkos::RangePolicy<exec_space> policy( 0, x.size() );
        Kokkos::parallel_for( "displacement_profile", policy, measure_profile );
        auto count_host_x =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, count_x );
        auto profile_host_x = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, profile_x );
        std::fstream fout_x;

        std::string file_name_x = "displacement_profile_x_direction.txt";
        fout_x.open( file_name_x, std::ios::app );
        for ( int p = 0; p < count_host_x( 0 ); p++ )
        {
            fout_x << mpi_rank << " " << profile_host_x( p, 0 ) << " "
                   << profile_host_x( p, 1 ) << " " << profile_host_x( p, 2 )
                   << std::endl;
        }

        // ------------------------------------
        // Displacement profiles in y-direction
        // ------------------------------------

        double num_cell_y = num_cells[1];
        auto profile_y = Kokkos::View<double* [3], memory_space>(
            Kokkos::ViewAllocateWithoutInitializing( "displacement_profile" ),
            num_cell_y );
        MPI_Comm_rank( MPI_COMM_WORLD, &mpi_rank );
        Kokkos::View<int*, memory_space> count_y( "c", 1 );

        double dx = particles->dx[0];

        auto measure_profile_y = KOKKOS_LAMBDA( const int pid )
        {
            if ( x( pid, 0 ) < dx / 2.0 && x( pid, 0 ) > -dx / 2.0 &&
                 x( pid, 2 ) < dz / 2.0 && x( pid, 2 ) > -dz / 2.0 )
            {
                auto c = Kokkos::atomic_fetch_add( &count_y( 0 ), 1 );
                profile_y( c, 0 ) = x( pid, 1 );
                profile_y( c, 1 ) = u( pid, 1 );
                profile_y( c, 2 ) = std::sqrt( u( pid, 0 ) * u( pid, 0 ) +
                                               u( pid, 1 ) * u( pid, 1 ) +
                                               u( pid, 2 ) * u( pid, 2 ) );
            }
        };
        Kokkos::RangePolicy<exec_space> policy_y( 0, x.size() );
        Kokkos::parallel_for( "displacement_profile", policy_y,
                              measure_profile_y );
        auto count_host_y =
            Kokkos::create_mirror_view_and_copy( Kokkos::HostSpace{}, count_y );
        auto profile_host_y = Kokkos::create_mirror_view_and_copy(
            Kokkos::HostSpace{}, profile_y );
        std::fstream fout_y;

        std::string file_name_y = "displacement_profile_y_direction.txt";
        fout_y.open( file_name_y, std::ios::app );
        for ( int p = 0; p < count_host_y( 0 ); p++ )
        {
            fout_y << mpi_rank << " " << profile_host_y( p, 0 ) << " "
                   << profile_host_y( p, 1 ) << " " << profile_host_y( p, 2 )
                   << std::endl;
        }
    }

    MPI_Finalize();
}
