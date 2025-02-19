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

    // Create container.
    auto create_container = KOKKOS_LAMBDA( const int, const double x[3] )
    {
        double rsq = x[0] * x[0] + x[1] * x[1];

        // Convert domain block into cylinder
        if ( rsq > cylinder_radius * cylinder_radius )
            return false;
        // Leave remaining bottom wall particles and remove particles inside
        // cylinder
        if ( x[2] > low_corner[2] + wall_thickness &&
             rsq < ( cylinder_radius - wall_thickness ) *
                       ( cylinder_radius - wall_thickness ) )
            return false;

        return true;
    };
    // Container particles should be frozen, never updated.
    CabanaPD::Particles particles( memory_space{}, model_type{},
                                   CabanaPD::BaseOutput{}, low_corner,
                                   high_corner, num_cells, halo_width,
                                   create_container, exec_space{}, true );

    // Create powder.
    double min_height = inputs["min_height"];
    auto create_powder = KOKKOS_LAMBDA( const int, const double x[3] )
    {
        double rsq = x[0] * x[0] + x[1] * x[1];

        // Only create particles inside cylinder.
        if ( x[2] > min_height &&
             rsq < ( cylinder_radius - wall_thickness ) *
                       ( cylinder_radius - wall_thickness ) )
            return true;

        return false;
    };
    particles.createParticles( exec_space(), Cabana::InitRandom{},
                               create_powder, particles.numFrozen() );

    // Set density/volumes.
    auto rho = particles.sliceDensity();
    auto vol = particles.sliceVolume();
    auto init_functor = KOKKOS_LAMBDA( const int pid )
    {
        rho( pid ) = rho0;
        vol( pid ) = vol0;
    };
    particles.updateParticles( exec_space{}, init_functor );

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
    solver.updateNeighbors();
    solver.run( gravity );

    // ====================================================
    // Interpolate for consolidation
    // ====================================================
    // Do this first to get the coarser grid.
    std::array<int, 3> coarse_cells{ num_cells[0] / 2, num_cells[1] / 2,
                                     num_cells[2] / 2 };
    CabanaPD::Particles pd_particles(
        memory_space{}, model_type{}, CabanaPD::BaseOutput{}, low_corner,
        high_corner, coarse_cells, halo_width, create_powder, exec_space{} );
    // FIXME: missing container.

    CabanaPD::interpolate( solver.particles, pd_particles );

    // Output
    std::string name = "particles_pd";
    pd_particles.output( name, 0, 0.0, true );
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
