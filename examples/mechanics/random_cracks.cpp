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

// Simulate crack propagation from multiple pre-notches.
void randomCracksExample( const std::string filename )
{
    // ====================================================
    //               Choose Kokkos spaces
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
    std::array<double, 3> low_corner = inputs["low_corner"];
    std::array<double, 3> high_corner = inputs["high_corner"];

    // ====================================================
    //                    Pre-notches
    // ====================================================
    // Number of pre-notches
    constexpr int Npn = 200;

    double minl = inputs["minimum_prenotch_length"];
    double maxl = inputs["maximum_prenotch_length"];
    double thickness = high_corner[2] - low_corner[2];

    // Initialize pre-notch arrays
    Kokkos::Array<Kokkos::Array<double, 3>, Npn> notch_positions;
    Kokkos::Array<Kokkos::Array<double, 3>, Npn> notch_v1;
    Kokkos::Array<Kokkos::Array<double, 3>, Npn> notch_v2;

    // Changing this seed will re-randomize the cracks.
    std::size_t seed = 44758454;
    // Random number generator
    std::mt19937 gen( seed );
    std::uniform_real_distribution<> dis( 0.0, 1.0 );

    // Loop over pre-notches
    for ( int n = 0; n < Npn; n++ )
    {
        // Random numbers for pre-notch position
        double random_number_x = dis( gen );
        double random_number_y = dis( gen );

        // Random coordinates of one endpoint of the pre-notch
        // Note: the addition and subtraction of "maxl" ensures the prenotch
        // does not extend outside the domain
        double Xc1 = ( low_corner[0] + maxl ) +
                     ( ( high_corner[0] - maxl ) - ( low_corner[0] + maxl ) ) *
                         random_number_x;
        double Yc1 = ( low_corner[1] + maxl ) +
                     ( ( high_corner[1] - maxl ) - ( low_corner[1] + maxl ) ) *
                         random_number_y;
        Kokkos::Array<double, 3> p0 = { Xc1, Yc1, low_corner[2] };

        // Assign pre-notch position
        notch_positions[n] = p0;

        // Random pre-notch length on XY-plane
        double random_number_l = dis( gen );
        double l = minl + ( maxl - minl ) * random_number_l;

        // Random pre-notch orientation on XY-plane
        double random_number_theta = dis( gen );
        double theta = CabanaPD::pi * random_number_theta;

        // Pre-notch v1 vector on XY-plane
        Kokkos::Array<double, 3> v1 = { l * cos( theta ), l * sin( theta ), 0 };
        notch_v1[n] = v1;

        // Random number for y-component of v2 vector: the angle of v2 in the
        // YZ-plane is between -45 and 45 deg.
        double random_number_v2_y = dis( gen );

        // Pre-notch v2 vector on YZ-plane
        Kokkos::Array<double, 3> v2 = {
            0, ( -1.0 + 2.0 * random_number_v2_y ) * thickness, thickness };
        notch_v2[n] = v2;
    }

    CabanaPD::Prenotch<Npn> prenotch( notch_v1, notch_v2, notch_positions );

    // ====================================================
    //                    Force model
    // ====================================================
    using model_type = CabanaPD::PMB;
    CabanaPD::ForceModel<model_type> force_model( delta, K, G0 );

    // ====================================================
    //                 Particle generation
    // ====================================================
    // Note that individual inputs can be passed instead (see other examples).
    CabanaPD::Particles particles( memory_space{}, model_type{}, inputs,
                                   exec_space{} );

    // ====================================================
    //                Boundary conditions planes
    // ====================================================
    double dy = particles.dx[1];
    CabanaPD::RegionBoundary<CabanaPD::RectangularPrism> plane1(
        low_corner[0], high_corner[0], low_corner[1] - dy, low_corner[1] + dy,
        low_corner[2], high_corner[2] );
    CabanaPD::RegionBoundary<CabanaPD::RectangularPrism> plane2(
        low_corner[0], high_corner[0], high_corner[1] - dy, high_corner[1] + dy,
        low_corner[2], high_corner[2] );

    // ====================================================
    //            Custom particle initialization
    // ====================================================
    auto rho = particles.sliceDensity();
    auto x = particles.sliceReferencePosition();
    auto v = particles.sliceVelocity();
    auto f = particles.sliceForce();
    auto nofail = particles.sliceNoFail();

    auto init_functor = KOKKOS_LAMBDA( const int pid )
    {
        // Density
        rho( pid ) = rho0;
        // No-fail zone
        if ( x( pid, 1 ) <= plane1.low_y + delta + 1e-10 ||
             x( pid, 1 ) >= plane2.high_y - delta - 1e-10 )
            nofail( pid ) = 1;
    };
    particles.updateParticles( exec_space{}, init_functor );

    // ====================================================
    //                   Create solver
    // ====================================================
    auto cabana_pd =
        CabanaPD::createSolver<memory_space>( inputs, particles, force_model );

    // ====================================================
    //                Boundary conditions
    // ====================================================
    // Create BC last to ensure ghost particles are included.
    double sigma0 = inputs["traction"];
    double b0 = sigma0 / dy;
    f = particles.sliceForce();
    x = particles.sliceReferencePosition();
    // Create a symmetric force BC in the y-direction.
    auto bc_op = KOKKOS_LAMBDA( const int pid, const double )
    {
        auto ypos = x( pid, 1 );
        auto sign = std::abs( ypos ) / ypos;
        f( pid, 1 ) += b0 * sign;
    };
    auto bc = createBoundaryCondition( bc_op, exec_space{}, particles, true,
                                       plane1, plane2 );

    // ====================================================
    //                   Simulation run
    // ====================================================
    cabana_pd->init( bc, prenotch );
    cabana_pd->run( bc );
}

// Initialize MPI+Kokkos.
int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    randomCracksExample( argv[1] );

    Kokkos::finalize();
    MPI_Finalize();
}
