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
    double vol0 = inputs["volume"];
    double radius = inputs["radius"];
    double extend = inputs["radius_extended"];
    double nu = inputs["poisson_ratio"];
    double E = inputs["elastic_modulus"];
    double e = inputs["restitution"];

    // ====================================================
    //                  Discretization
    // ====================================================
    std::array<double, 3> low_corner = inputs["low_corner"];
    std::array<double, 3> high_corner = inputs["high_corner"];
    std::array<int, 3> num_cells = inputs["num_cells"];

    // ====================================================
    //                    Force model
    // ====================================================
    using model_type = CabanaPD::HertzianModel;
    model_type contact_model( radius, extend, radius, nu, E, e );

    // ====================================================
    //            Custom particle initialization
    // ====================================================
    double d_out = inputs["outer_cylinder_diameter"];
    double d_in = inputs["inner_cylinder_diameter"];
    double Rout = 0.5 * d_out;
    double Rin = 0.5 * d_in;
    double Wall_th = inputs["wall_thickness"];

    // Create container.
    auto create_container = KOKKOS_LAMBDA( const int, const double x[3] )
    {
        double rsq = x[0] * x[0] + x[1] * x[1];

        // Convert domain block into hollow cylinder
        if ( rsq > Rout * Rout || rsq < ( Rin - Wall_th ) * ( Rin - Wall_th ) )
            return false;
        // Leave remaining bottom wall particles and remove particles in between
        // inner and outer cylinder
        if ( x[2] > low_corner[2] + Wall_th && rsq > Rin * Rin &&
             rsq < ( Rout - Wall_th ) * ( Rout - Wall_th ) )
            return false;

        return true;
    };
    // Container particles should be frozen, never updated.
    auto particles = CabanaPD::createParticles<memory_space, model_type>(
									 exec_space(), low_corner, high_corner, num_cells, 1, 
        CabanaPD::BaseOutput{}, create_container, 0, true );

    // Create powder.
    auto create_powder = KOKKOS_LAMBDA( const int, const double x[3] )
    {
        double rsq = x[0] * x[0] + x[1] * x[1];

        // Only create particles in between inner and outer cylinder.
        if ( x[2] > low_corner[2] + Wall_th && rsq > ( Rin ) * ( Rin ) &&
             rsq < ( Rout - Wall_th ) * ( Rout - Wall_th ) )
            return true;

        return false;
    };
    particles->createParticles( exec_space(), Cabana::InitRandom{},
                                create_powder, particles->numFrozen() );

    // Set density/volumes.
    auto rho = particles->sliceDensity();
    auto vol = particles->sliceVolume();
    auto init_functor = KOKKOS_LAMBDA( const int pid )
    {
        rho( pid ) = rho0;
        vol( pid ) = vol0;
    };
    particles->updateParticles( exec_space{}, init_functor );

    // ====================================================
    //                   Create solver
    // ====================================================
    auto cabana_pd = CabanaPD::createSolverNoFracture<memory_space>(
        inputs, particles, contact_model );

    // ====================================================
    //                   Simulation init
    // ====================================================
    cabana_pd->init();

    // Remove any points that are too close.
    Kokkos::View<int*, memory_space> keep( "keep_points",
                                           particles->numLocal() );
    Kokkos::deep_copy( keep, 1 );
    int num_keep;
    auto num_frozen = particles->numFrozen();
    auto f = particles->sliceForce();
    auto remove_functor = KOKKOS_LAMBDA( const int pid, int& k )
    {
        auto f_mag = Kokkos::hypot( f( pid, 0 ), f( pid, 1 ), f( pid, 2 ) );
        if ( f_mag > 1e8 )
            keep( pid - num_frozen ) = 0;
        else
            k++;
    };
    Kokkos::RangePolicy<exec_space> policy( num_frozen,
                                            particles->localOffset() );
    Kokkos::parallel_reduce( "remove", policy, remove_functor,
                             Kokkos::Sum<int>( num_keep ) );
    cabana_pd->particles->remove( num_keep, keep );
    // FIXME: Will need to rebuild ghosts.

    // ====================================================
    //                   Boundary condition
    // ====================================================
    f = cabana_pd->particles->sliceForce();
    rho = cabana_pd->particles->sliceDensity();
    auto body_functor = KOKKOS_LAMBDA( const int pid, const double )
    {
        f( pid, 2 ) -= 9.8 * rho( pid ); // * vol( pid );
    };
    auto gravity = CabanaPD::createBodyTerm( body_functor, true );

    // ====================================================
    //                   Simulation run
    // ====================================================
    cabana_pd->update();
    cabana_pd->run( gravity );
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
