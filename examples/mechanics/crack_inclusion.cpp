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

// Simulate a crack interacting with an inclusion.
void crackInclusionExample( const std::string filename )
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
    // Plate
    double rho0 = inputs["density"][0];
    double E = inputs["elastic_modulus"][0];
    double nu = inputs["Poisson's_ratio"][0];
    double K = E / ( 3 * ( 1 - 2 * nu ) );
    double G = E / ( 2 * ( 1 + nu ) );
    double G0 = inputs["fracture_energy"][0];

    // Inclusion
    double rho0_I = inputs["density"][1];
    double E_I = inputs["elastic_modulus"][1];
    double nu_I = inputs["Poisson's_ratio"][1];
    double K_I = E_I / ( 3 * ( 1 - 2 * nu_I ) );
    double G_I = E_I / ( 2 * ( 1 + nu_I ) );
    double G0_I = inputs["fracture_energy"][1];

    double horizon = inputs["horizon"];
    horizon += 1e-10;

    // ====================================================
    //                  Discretization
    // ====================================================
    std::array<double, 3> low_corner = inputs["low_corner"];
    std::array<double, 3> high_corner = inputs["high_corner"];

    // ====================================================
    //                    Pre-notch
    // ====================================================
    double height = inputs["system_size"][0];
    double thickness = inputs["system_size"][2];
    double L_prenotch = height / 2.0;
    double y_prenotch = 0.5 * ( low_corner[1] + high_corner[1] );
    Kokkos::Array<double, 3> p01 = { low_corner[0], y_prenotch, low_corner[2] };
    Kokkos::Array<double, 3> v1 = { L_prenotch, 0, 0 };
    Kokkos::Array<double, 3> v2 = { 0, 0, thickness };
    Kokkos::Array<Kokkos::Array<double, 3>, 1> notch_positions = { p01 };
    CabanaPD::Prenotch<1> prenotch( v1, v2, notch_positions );

    // ====================================================
    //                   Force models
    // ====================================================
    using model_type = CabanaPD::LPS;

    // Plate material
    CabanaPD::ForceModel force_model_plate( model_type{}, horizon, K, G, G0 );

    // Inclusion material
    CabanaPD::ForceModel force_model_inclusion( model_type{}, horizon, K_I, G_I,
                                                G0_I );

    // ====================================================
    //                 Particle generation
    // ====================================================
    // Note that individual inputs can be passed instead (see other examples).
    CabanaPD::Particles particles( memory_space{}, model_type{} );
    particles.domain( inputs );
    particles.create( exec_space{} );

    // ====================================================
    //                Boundary conditions planes
    // ====================================================
    double dy = particles.dx[1];
    CabanaPD::Region<CabanaPD::RectangularPrism> plane1(
        low_corner[0], high_corner[0], low_corner[1] - dy, low_corner[1] + dy,
        low_corner[2], high_corner[2] );
    CabanaPD::Region<CabanaPD::RectangularPrism> plane2(
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
    auto type = particles.sliceType();

    double R = inputs["inclusion_radius"];
    double xI = inputs["inclusion_center"][0];
    double yI = inputs["inclusion_center"][1];
    CabanaPD::Region<CabanaPD::Cylinder> inclusion( 0.0, R, low_corner[2],
                                                    high_corner[2], xI, yI );

    auto init_functor = KOKKOS_LAMBDA( const int pid )
    {
        // Density
        rho( pid ) = rho0;
        // No-fail zone
        if ( x( pid, 1 ) <= plane1.low[1] + horizon + 1e-10 ||
             x( pid, 1 ) >= plane2.high[1] - horizon - 1e-10 )
            nofail( pid ) = 1;
        // Inclusion material
        if ( inclusion.inside( x, pid ) )
        {
            type( pid ) = 1;
            rho( pid ) = rho0_I;
        }
        // Plate material
        else
        {
            rho( pid ) = rho0;
        }
    };
    particles.update( exec_space{}, init_functor );

    // ====================================================
    //                   Create solver
    // ====================================================
    auto models = CabanaPD::createMultiForceModel(
        particles, CabanaPD::AverageTag{}, force_model_plate,
        force_model_inclusion );
    CabanaPD::Solver solver( inputs, particles, models );

    // ====================================================
    //                Boundary conditions
    // ====================================================
    // Create BC last to ensure ghost particles are included.
    double sigma0 = inputs["traction"];
    double b0 = sigma0 / dy;
    f = solver.particles.sliceForce();
    x = solver.particles.sliceReferencePosition();
    // Create a symmetric force BC in the y-direction.
    auto bc_op = KOKKOS_LAMBDA( const int pid, const double )
    {
        auto ypos = x( pid, 1 );
        auto sign = std::abs( ypos ) / ypos;
        f( pid, 1 ) += b0 * sign;
    };
    auto bc = createBoundaryCondition( bc_op, exec_space{}, solver.particles,
                                       true, plane1, plane2 );

    // ====================================================
    //                   Simulation run
    // ====================================================
    solver.init( bc, prenotch );
    solver.run( bc );
}

// Initialize MPI+Kokkos.
int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    crackInclusionExample( argv[1] );

    Kokkos::finalize();
    MPI_Finalize();
}
