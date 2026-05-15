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

// Simulate a square plate under tension with a circular hole at its center.
void plateWithHoleExample( const std::string filename )
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
    double nu = 1.0 / 3.0;
    double K = E / ( 3 * ( 1 - 2 * nu ) );
    double G0 = inputs["fracture_energy"];
    // double G = E / ( 2.0 * ( 1.0 + nu ) ); // Only for LPS.
    double horizon = inputs["horizon"];
    horizon += 1e-10;

    // ====================================================
    //                  Discretization
    // ====================================================
    std::array<double, 3> low_corner = inputs["low_corner"];
    std::array<double, 3> high_corner = inputs["high_corner"];
    std::array<int, 3> num_cells = inputs["num_cells"];
    int m = std::floor( horizon /
                        ( ( high_corner[0] - low_corner[0] ) / num_cells[0] ) );
    int halo_width = m + 1; // Just to be safe.

    // ====================================================
    //                    Force model
    // ====================================================
    using model_type = CabanaPD::PMB;
    CabanaPD::ForceModel force_model( model_type{}, horizon, K, G0 );

    // ====================================================
    //    Custom particle generation and initialization
    // ====================================================
    double x_center = 0.5 * ( low_corner[0] + high_corner[0] );
    double y_center = 0.5 * ( low_corner[1] + high_corner[1] );
    double R = inputs["hole_radius"];
    CabanaPD::Region<CabanaPD::Cylinder> cylinder(
        0.0, R, low_corner[2], high_corner[2], x_center, y_center );

    // Do not create particles inside given cylindrical region
    auto init_op = KOKKOS_LAMBDA( const int, const double x[3] )
    {
        if ( cylinder.inside( x ) )
            return false;
        return true;
    };

    CabanaPD::Particles particles( memory_space{}, model_type{},
                                   CabanaPD::EnergyStressOutput{} );
    particles.domain( low_corner, high_corner, num_cells, halo_width );
    particles.create( exec_space{}, Cabana::InitUniform{}, init_op );

    // ====================================================
    //                Boundary conditions planes
    // ====================================================
    double dx = particles.dx[1];
    CabanaPD::Region<CabanaPD::RectangularPrism> plane1(
        low_corner[0] - dx, low_corner[0] + dx, low_corner[1], high_corner[1],
        low_corner[2], high_corner[2] );
    CabanaPD::Region<CabanaPD::RectangularPrism> plane2(
        high_corner[0] - dx, high_corner[0] + dx, low_corner[1], high_corner[1],
        low_corner[2], high_corner[2] );

    // ====================================================
    //            Custom particle initialization
    // ====================================================
    auto rho = particles.sliceDensity();
    auto x = particles.sliceReferencePosition();
    auto f = particles.sliceForce();
    auto nofail = particles.sliceNoFail();

    auto init_functor = KOKKOS_LAMBDA( const int pid )
    {
        // Density
        rho( pid ) = rho0;
        // No-fail zone
        if ( x( pid, 0 ) <= plane1.low[0] + horizon + 1e-10 ||
             x( pid, 0 ) >= plane2.high[0] - horizon - 1e-10 )
            nofail( pid ) = 1;
    };
    particles.update( exec_space{}, init_functor );

    // ====================================================
    //                   Create solver
    // ====================================================
    CabanaPD::Solver solver( inputs, particles, force_model );

    // ====================================================
    //                Boundary conditions
    // ====================================================
    // Create BC last to ensure ghost particles are included.
    double sigma0 = inputs["traction"];
    double b0 = sigma0 / dx;
    f = solver.particles.sliceForce();
    x = solver.particles.sliceReferencePosition();
    // Create a symmetric force BC in the x-direction.
    auto bc_op = KOKKOS_LAMBDA( const int pid, const double )
    {
        auto xpos = x( pid, 0 );
        auto sign = std::abs( xpos ) / xpos;
        f( pid, 0 ) += b0 * sign;
    };
    auto bc = createBoundaryCondition( bc_op, exec_space{}, solver.particles,
                                       true, plane1, plane2 );

    // ====================================================
    //                   Simulation run
    // ====================================================
    solver.init( bc );
    solver.run( bc );
}

// Initialize MPI+Kokkos.
int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    plateWithHoleExample( argv[1] );

    Kokkos::finalize();
    MPI_Finalize();
}
