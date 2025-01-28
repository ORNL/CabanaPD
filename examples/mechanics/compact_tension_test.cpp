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

// Simulate crack propagation in a compact tension test.
void compactTensionTestExample( const std::string filename )
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
    std::array<int, 3> num_cells = inputs["num_cells"];
    int m = std::floor( delta /
                        ( ( high_corner[0] - low_corner[0] ) / num_cells[0] ) );
    int halo_width = m + 1; // Just to be safe.

    // ====================================================
    //                    Pre-notch
    // ====================================================
    double height = inputs["system_size"][0];
    double thickness = inputs["system_size"][2];
    double L_prenotch = height / 2.0;
    double y_prenotch = 0.0;
    Kokkos::Array<double, 3> p0 = { low_corner[0], y_prenotch, low_corner[2] };
    Kokkos::Array<double, 3> v1 = { L_prenotch, 0, 0 };
    Kokkos::Array<double, 3> v2 = { 0, 0, thickness };
    Kokkos::Array<Kokkos::Array<double, 3>, 1> notch_position = { p0 };
    CabanaPD::Prenotch<1> prenotch( v1, v2, notch_position );

    // ====================================================
    //                    Force model
    // ====================================================
    using model_type = CabanaPD::PMB;
    using thermal_type = CabanaPD::TemperatureIndependent;
    using mechanics_type = CabanaPD::ElasticPerfectlyPlastic;

    // ====================================================
    //                 Particle generation
    // ====================================================
    auto particles =
        CabanaPD::createParticles<memory_space, model_type, thermal_type>(
            exec_space{}, low_corner, high_corner, num_cells, halo_width );

    auto force_model = CabanaPD::createForceModel(
        model_type{}, mechanics_type{}, *particles, delta, K, G0 );

    // ====================================================
    //    Custom particle generation and initialization
    // ====================================================
    // Rectangular prism containing the full specimen: original geometry
    CabanaPD::RegionBoundary<CabanaPD::RectangularPrism> plane(
        low_corner[0], high_corner[0], low_corner[1], high_corner[1],
        low_corner[2], high_corner[2] );
    std::vector<CabanaPD::RegionBoundary<CabanaPD::RectangularPrism>> planes = {
        plane };

    // Geometric parameters of specimen
    double L = inputs["system_size"][1];
    double W = L / 1.25;
    double a = 0.45 * W;

    // Grid spacing in y-direction
    double dy = particles->dx[1];

    // Remove particles from original geometry
    auto init_op = KOKKOS_LAMBDA( const int, const double x[3] )
    {
        // Thin rectangle
        if ( x[0] < low_corner[1] + 0.25 * W + a &&
             Kokkos::abs( x[1] ) < 0.5 * dy )
        {
            return false;
        }
        // Thick rectangle
        else if ( x[0] < low_corner[1] + 0.25 * W &&
                  Kokkos::abs( x[1] ) < 26e-4 )
        {
            return false;
        }
        else
        {
            return true;
        }
    };

    particles->createParticles( exec_space(), init_op );

    auto rho = particles->sliceDensity();
    auto x = particles->sliceReferencePosition();
    auto v = particles->sliceVelocity();
    auto nofail = particles->sliceNoFail();

    // Pin radius
    double R = 4e-3;
    // Pin center coordinates (top)
    double x_pin = low_corner[0] + 0.25 * W;
    double y_pin = 0.37 * W;
    // Pin velocity magnitude
    double v0 = inputs["pin_velocity"];

    auto init_functor = KOKKOS_LAMBDA( const int pid )
    {
        // Density
        rho( pid ) = rho0;

        auto xpos = x( pid, 0 );
        auto ypos = x( pid, 1 );
        auto distsq =
            ( xpos - x_pin ) * ( xpos - x_pin ) +
            ( Kokkos::abs( ypos ) - y_pin ) * ( Kokkos::abs( ypos ) - y_pin );
        auto sign = Kokkos::abs( ypos ) / ypos;

        // pins' y-velocity
        if ( distsq < R * R )
            v( pid, 1 ) = sign * v0;

        // No-fail zone
        if ( distsq < ( 2 * R ) * ( 2 * R ) )
            nofail( pid ) = 1;
    };
    particles->updateParticles( exec_space{}, init_functor );

    // ====================================================
    //                   Create solver
    // ====================================================
    auto cabana_pd =
        CabanaPD::createSolver<memory_space>( inputs, particles, force_model );

    // ====================================================
    //                Boundary conditions
    // ====================================================

    // Create BC last to ensure ghost particles are included.
    auto f = particles->sliceForce();
    // Create a symmetric force BC in the y-direction.
    auto bc_op = KOKKOS_LAMBDA( const int pid, const double )
    {
        auto xpos = x( pid, 0 );
        auto ypos = x( pid, 1 );
        if ( ( xpos - x_pin ) * ( xpos - x_pin ) +
                 ( Kokkos::abs( ypos ) - y_pin ) *
                     ( Kokkos::abs( ypos ) - y_pin ) <
             R * R )
            f( pid, 1 ) = 0;
    };
    auto bc = createBoundaryCondition( bc_op, exec_space{}, *particles, planes,
                                       true );

    // ====================================================
    //                   Simulation run
    // ====================================================
    cabana_pd->init( prenotch );
    cabana_pd->run( bc );
}

// Initialize MPI+Kokkos.
int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    compactTensionTestExample( argv[1] );

    Kokkos::finalize();
    MPI_Finalize();
}
