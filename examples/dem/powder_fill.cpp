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

// Simulate powder settling.
void powderSettlingExample( const std::string filename )
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
    double vol0 = inputs["volume"];
    double radius = inputs["radius"];
    double radius_extend = inputs["radius_extend"];
    double nu = inputs["poisson_ratio"];
    double E = inputs["elastic_modulus"];
    double e = inputs["restitution"];

    // ====================================================
    //                  Discretization
    // ====================================================
    std::array<double, 3> low_corner = inputs["low_corner"];
    std::array<double, 3> high_corner = inputs["high_corner"];
    std::array<int, 3> num_cells = inputs["num_cells"];
    int halo_width = 1;

    // ====================================================
    //                    Force model
    // ====================================================
    using model_type = CabanaPD::HertzianModel;
    model_type contact_model( radius, radius_extend, nu, E, e );

    // ====================================================
    //            Custom particle initialization
    // ====================================================
    double diameter = inputs["cylinder_diameter"];
    double cylinder_radius = 0.5 * diameter;
    double wall_thickness = inputs["wall_thickness"];
    double H = inputs["system_size"][2];
    double min_height = inputs["min_height"];
    CabanaPD::Region<CabanaPD::Cylinder> cylinder( 0.0, cylinder_radius,
                                                   -0.5 * H, 0.5 * H );
    double powder_radius = cylinder_radius - wall_thickness;
    CabanaPD::Region<CabanaPD::Cylinder> powder_cylinder(
        0.0, powder_radius, min_height, 0.5 * H - wall_thickness );
    CabanaPD::Region<CabanaPD::Cylinder> open_cylinder(
        0.0, powder_radius, -0.5 * H + wall_thickness, 0.5 * H );

    CabanaPD::Particles particles( memory_space{}, model_type{},
                                   CabanaPD::BaseOutput{} );
    // Create container.
    auto create_container = KOKKOS_LAMBDA( const int, const double x[3] )
    {
        // Overall cylindrical region.
        if ( !cylinder.inside( x ) )
            return false;

        // Powder region.
        if ( open_cylinder.inside( x ) )
            return false;

        return true;
    };
    // Container particles should be frozen, never updated.
    particles.domain( low_corner, high_corner, num_cells, halo_width );
    particles.create( exec_space{}, create_container, 0, true );

    // Create powder.
    auto create_powder = KOKKOS_LAMBDA( const int, const double x[3] )
    {
        // Only create particles inside cylinder.
        if ( powder_cylinder.inside( x ) )
            return true;

        return false;
    };
    particles.create( exec_space{}, Cabana::InitRandom{}, create_powder,
                      particles.numFrozen() );

    // Set density/volumes.
    auto rho = particles.sliceDensity();
    auto vol = particles.sliceVolume();
    auto init_functor = KOKKOS_LAMBDA( const int pid )
    {
        rho( pid ) = rho0;
        vol( pid ) = vol0;
    };
    particles.update( exec_space{}, init_functor );

    // ====================================================
    //                   Create solver
    // ====================================================
    CabanaPD::Solver solver( inputs, particles, contact_model );

    // ====================================================
    //                   Simulation init
    // ====================================================
    solver.init();

    // Use a force magnitude threshold to remove particles that are too close.
    // TODO: The force magnitude should be based on the maximum desired overlap
    // according to the properties of the contact model
    solver.remove( 1e6 );

    // ====================================================
    //                   Boundary condition
    // ====================================================
    auto f = solver.particles.sliceForce();
    rho = solver.particles.sliceDensity();
    auto body_functor = KOKKOS_LAMBDA( const int pid, const double )
    {
        f( pid, 2 ) -= 9.8 * rho( pid );
    };
    CabanaPD::BodyTerm gravity( body_functor, solver.particles.size(), true );

    // ====================================================
    //                   Simulation run
    // ====================================================
    solver.run( gravity );
}

// Initialize MPI+Kokkos.
int main( int argc, char* argv[] )
{
    MPI_Init( &argc, &argv );
    Kokkos::initialize( argc, argv );

    powderSettlingExample( argv[1] );

    Kokkos::finalize();
    MPI_Finalize();
}
