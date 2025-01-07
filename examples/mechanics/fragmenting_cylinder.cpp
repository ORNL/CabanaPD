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

// Simulate an expanding cylinder resulting in fragmentation
void fragmentingCylinderExample( const std::string filename )
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
    double K = inputs["bulk_modulus"];
    // double G = inputs["shear_modulus"]; // Only for LPS.
    double sc = inputs["critical_stretch"];
    double delta = inputs["horizon"];
    delta += 1e-10;
    // For PMB or LPS with influence_type == 1
    double G0 = 9 * K * delta * ( sc * sc ) / 5;
    // For LPS with influence_type == 0 (default)
    // double G0 = 15 * K * delta * ( sc * sc ) / 8;

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
    //                    Force model
    // ====================================================
    using model_type = CabanaPD::ForceModel<CabanaPD::PMB>;
    model_type force_model( delta, K, G0 );
    // using model_type =
    //      CabanaPD::ForceModel<CabanaPD::LPS, CabanaPD::Fracture>;
    // model_type force_model( delta, K, G, G0 );

    // ====================================================
    //    Custom particle generation and initialization
    // ====================================================
    double x_center = 0.5 * ( low_corner[0] + high_corner[0] );
    double y_center = 0.5 * ( low_corner[1] + high_corner[1] );
    double Rout = inputs["cylinder_outer_radius"];
    double Rin = inputs["cylinder_inner_radius"];

    // Do not create particles outside given cylindrical region
    auto init_op = KOKKOS_LAMBDA( const int, const double x[3] )
    {
        double rsq = ( x[0] - x_center ) * ( x[0] - x_center ) +
                     ( x[1] - y_center ) * ( x[1] - y_center );
        if ( rsq < Rin * Rin || rsq > Rout * Rout || x[2] > 0.05 ||
             x[2] < -0.05 )
            return false;
        return true;
    };

    auto particles = CabanaPD::createParticles<memory_space, model_type>(
        exec_space(), low_corner, high_corner, num_cells, halo_width, init_op );

    auto rho = particles->sliceDensity();
    auto x = particles->sliceReferencePosition();
    auto v = particles->sliceVelocity();
    auto f = particles->sliceForce();

    double vrmax = inputs["max_radial_velocity"];
    double vrmin = inputs["min_radial_velocity"];
    double vzmax = inputs["max_vertical_velocity"];
    double zmin = low_corner[2];

    auto dx = particles->dx;
    double height = inputs["system_size"][2];
    double factor = inputs["grid_perturbation_factor"];

    using pool_type = Kokkos::Random_XorShift64_Pool<exec_space>;
    using random_type = Kokkos::Random_XorShift64<exec_space>;
    pool_type pool;
    int seed = 456854;
    pool.init( seed, particles->localOffset() );
    auto init_functor = KOKKOS_LAMBDA( const int pid )
    {
        // Density
        rho( pid ) = rho0;

        // Perturb particle positions
        auto gen = pool.get_state();
        for ( std::size_t d = 0; d < 3; d++ )
        {
            auto rand =
                Kokkos::rand<random_type, double>::draw( gen, 0.0, 1.0 );
            x( pid, d ) += ( 2.0 * rand - 1.0 ) * factor * dx[d];
        }
        pool.free_state( gen );

        // Velocity
        double zfactor = ( ( x( pid, 2 ) - zmin ) / ( 0.5 * height ) ) - 1;
        double vr = vrmax - vrmin * zfactor * zfactor;
        v( pid, 0 ) =
            vr * Kokkos::cos( Kokkos::atan2( x( pid, 1 ), x( pid, 0 ) ) );
        v( pid, 1 ) =
            vr * Kokkos::sin( Kokkos::atan2( x( pid, 1 ), x( pid, 0 ) ) );
        v( pid, 2 ) = vzmax * zfactor;
    };
    particles->updateParticles( exec_space{}, init_functor );

    // ====================================================
    //  Simulation run with contact physics
    // ====================================================
    if ( inputs["use_contact"] )
    {
        double r_c = inputs["contact_horizon_factor"];
        r_c *= dx[0];
        CabanaPD::NormalRepulsionModel contact_model( delta, r_c, K );

        auto cabana_pd = CabanaPD::createSolverFracture<memory_space>(
            inputs, particles, force_model, contact_model );
        cabana_pd->init();
        cabana_pd->run();
    }
    // ====================================================
    //  Simulation run without contact
    // ====================================================
    else
    {
        auto cabana_pd = CabanaPD::createSolverFracture<memory_space>(
            inputs, particles, force_model );
        cabana_pd->init();
        cabana_pd->run();
    }
}

// Initialize MPI+Kokkos.
int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    fragmentingCylinderExample( argv[1] );

    Kokkos::finalize();
    MPI_Finalize();
}
